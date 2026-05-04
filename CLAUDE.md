# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Patho-Bench is a Python library for benchmarking pathology foundation models. It consumes patch-level features extracted by [Trident](https://github.com/mahmoodlab/trident) (a sibling library; Patho-Bench does NOT do WSI → patch feature extraction) and evaluates them via several frameworks.

## Install / environment

```bash
conda create -n pathobench python=3.10 && conda activate pathobench
pip install -r requirements.txt   # installs trident from git + other deps
pip install -e .
```

Python >=3.10. `datasets==3.6.0` is pinned intentionally (see #19). Some encoders need extra packages — follow the error messages when they appear.

## High-level architecture

Two usage modes share the same core:

1. **Library mode** (`tutorial/tutorial.ipynb`, README examples): call `SplitFactory` + `ExperimentFactory` directly.
2. **Sweep mode** (`advanced_usage/`): `run.sh` → `run.py` → `patho_bench/Runner.py` launches many experiments in parallel `tmux` panes, one per task line in `tasks.yaml`. Each pane invokes `patho_bench/scripts/sweep_single_task.py`, which loops over hyperparameter combinations for a single task.

### Core flow (per experiment)

```
SplitFactory.from_hf(...)          # downloads split + task config from HuggingFace
        │
        ▼
ExperimentFactory.{linprobe|coxnet|retrieval|finetune}(...)
        │  builds → DatasetFactory → {Patch,Slide}EmbeddingsDataset + LabelDataset
        │  (pools patch embeddings via Pooler if pooled_embeddings_dir is empty)
        ▼
experiments/{LinearProbe,CoxNet,Retrieval,Finetuning}Experiment
    .train() / .test() / .report_results(metric=...)
```

Key abstractions:
- `ExperimentFactory` — top-level entry; also exposes `generate_arg_combinations`, `generate_exp_id`, `parse_task_code` (used by the sweep Runner). Task codes in sweep mode use `{train}==={test}--{task}` for cross-cohort evaluation.
- `SplitFactory` — fetches canonical train/test splits from the HuggingFace Patho-Bench dataset; HF splits have no validation set, so reserve it manually from train if needed.
- `DatasetFactory` + `patho_bench/datasets/*` — wraps patch or pooled feature directories plus labels into `torch.utils.data.Dataset`s. `BaseDataset`, `CombinedDataset`, `DataSplit`, `LabelDataset`, `PatchEmbeddingsDataset`, `SlideEmbeddingsDataset`.
- `Pooler.py` — generalized multi-slide fusion for patient-level tasks. Patho-Bench's pooling (NOT Trident's) is required because Trident does not handle patient-level tasks with multiple slides per patient.
- `TrainableSlideEncoder.py` — end-to-end finetuning wrapper around a Trident slide encoder.
- `experiments/utils/` — mixins shared across experiments: `ClassificationMixin`, `SurvivalMixin`, `RetrievalMixin`, `LoggingMixin`, plus `Retriever` and `FancyLayers`.
- `experiments/GeneralizabilityExperimentWrapper.py` — wraps an internal experiment so the same trained model is evaluated on an external cohort (`external_split`, `external_pooled_embeddings_dir`).
- `helpers/GPUManager.py` — auto-selects a GPU when `gpu=-1`; used for load balancing in parallel sweeps.
- `config/ConfigMixin.py`, `config/JSONSaver.py` — experiment config/result serialization.
- `optim/NLLSurvLoss.py`, `optim/GigaPathOptim.py` — survival loss and model-specific optimizer setup.

### `combine_slides_per_patient` semantics

Important parameter on every factory method. If True: concatenate all patches from a patient into one bag before pooling (early fusion). If False: pool each slide independently, then mean over slide features. Titan requires **False** (uses patch coordinates). Consult the model's docs before changing.

### Sweep mode specifics (`advanced_usage/`)

- `configs/tasks.yaml`: space-separated task codes run serially in one pane; newline-separated lines run in parallel panes.
- `configs/{linprobe,coxnet,retrieval,finetune}/*.yaml`: hyperparameter grids. Newline-separated values in a list → full combinatorial sweep. For `COST` specifically, `COST: auto` sweeps `np.logspace(log10(10e-6), log10(10e5), num=45)` (defined in `ExperimentFactory.py`).
- `configs/{patch,pooled}_embeddings_paths.yaml`: nested `{datasource: {model_name: path}}` dictionaries that tell the runner where features live. Pooled features must be produced by Patho-Bench, not Trident.
- `run.sh` invokes `run.py`; `--delay_interval N` staggers pane startup by N seconds to avoid thrashing a small machine.

## Common commands

Run the tutorial notebook: `tutorial/tutorial.ipynb`.

Single experiment (library mode): see the snippet in the README.

Large sweep:
```bash
cd advanced_usage
chmod +x run.sh
./run.sh              # launches tmux session(s)
```
Edit `run.sh` to change `--experiment_type`, `--model_name`, paths, and `--hyperparams_yaml` before running. Monitor via `patho_bench/scripts/monitor_progress.py` or by attaching to the tmux session (name defaults to `basename(saveto)`).

No test suite, lint config, or CI is defined in-repo (dev extras declare `pytest`/`black`/`mypy` but there are no tests or configs committed). Don't fabricate commands.

## Conventions / gotchas

- Patch-level features only — Patho-Bench expects Trident patch features, not Trident slide features. Slide-level features are produced internally via `Pooler`.
- When writing custom splits, match the format of HuggingFace splits returned by `SplitFactory.from_hf`.
- `COMBINE_TRAIN_VAL` and `TEST_EXTERNAL_ONLY` are module-level constants in `ExperimentFactory.py` that change cross-cohort evaluation semantics — grep before editing.
- `Runner.py` adds `'../'` to `sys.path`; when running sweeps the CWD is expected to be `advanced_usage/`.
