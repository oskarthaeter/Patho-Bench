# Patho-Bench

[arxiv](https://arxiv.org/pdf/2502.06750) | [HuggingFace](https://huggingface.co/datasets/MahmoodLab/Patho-Bench) | [Cite](https://github.com/mahmoodlab/patho-bench?tab=readme-ov-file#reference) | [License](https://github.com/mahmoodlab/patho-bench/blob/main/LICENSE)

**Patho-Bench is a Python library designed to benchmark foundation models for pathology.** 

This project was developed by the [Mahmood Lab](https://faisal.ai/) at Harvard Medical School and Brigham and Women's Hospital. This work was funded by NIH NIGMS R35GM138216.

> [!NOTE]
> Please report any issues on GitHub and contribute by opening a pull request.

## &#9889; Key features:
- **Reproducibility**: Canonical train-test splits for 42 tasks across 20 public datasets.
- **Evaluation frameworks**: Supports linear probing, prototyping (coming soon), retrieval, Cox survival prediction, and supervised fine-tuning.
- **Scalability**: Scales to thousands of experiments with automatic GPU load-balancing.

## 🚨 Updates
- **February 2025**: Patho-Bench is public!

## **Installation**:

- Create a virtual environment, e.g., `conda create -n "pathobench" python=3.10`, and activate it `conda activate pathobench`.
- **From local clone**:
    - `git clone https://github.com/mahmoodlab/Patho-Bench.git && cd Patho-Bench`.
    - Local install with running `pip install -e .`.
- **Using pip**:
    - `pip install git+https://github.com/mahmoodlab/Patho-Bench.git`
Additional packages may be required if you are loading specific pretrained models. Follow error messages for additional instructions.

> [!NOTE]  
> Patho-Bench works with encoders implemented in [Trident](https://github.com/mahmoodlab/trident); use Trident to extract patch embeddings for your WSIs prior to running Patho-Bench.

> [!NOTE]
> Patho-Bench relies on our [HuggingFace repo](https://huggingface.co/datasets/MahmoodLab/patho-bench) to directly read splits for tasks. If you want to use custom splits, format them similarly to our HuggingFace splits.

## 🏃 **Running Patho-Bench**

Patho-Bench supports various evaluation frameworks:
- `linprobe`  ➡️  Linear probing (using pre-pooled features)
- `coxnet`  ➡️  Cox proportional hazards model for survival prediction (using pre-pooled features)
- `protonet`  ➡️  Prototyping (using pre-pooled features) (Coming soon!)
- `retrieval`  ➡️  Retrieval (using pre-pooled features)
- `finetune`  ➡️  Supervised finetuning or training from scratch (using patch features)

Patho-Bench can be used in two ways: 
1. **Basic:** Importable classes and functions for easy integration into custom codebases
2. **Advanced:** Large-scale benchmarking using automated scripts

## 🔨 Basic Usage: Importing and using Patho-Bench in your custom workflows
Running any of the evaluation frameworks is straightforward (see example below). Define general-purpose arguments for setting up the experiment and framework-specific arguments. For a detailed introduction, follow our end-to-end [tutorial](https://github.com/mahmoodlab/patho-bench/blob/main/tutorial/tutorial.ipynb).

```python
from patho_bench.ExperimentFactory import ExperimentFactory # Make sure you have installed Patho-Bench and this imports correctly

model_name = 'titan'
train_source = 'cptac_ccrcc' 
task_name = 'BAP1_mutation'

# Initialize the experiment
experiment = ExperimentFactory.linprobe( # This is linear probing, but similar APIs are available for coxnet, protonet, retrieval, and finetune
                    model_name = model_name,
                    train_source = train_source,
                    test_source = None, # Leave as default (None) to automatically use the test split of the training source
                    task_name = task_name,
                    patch_embeddings_dirs = '/path/to/job_dir/20x_512px_0px_overlap/features_conch_v15', # Can be list of paths if patch features are split across multiple directories. See NOTE below.
                    pooled_embeddings_root = './_test_pooled_features',
                    splits_root = './_test_splits', # Splits are downloaded here from HuggingFace. You can also provide your own splits using the path_to_split and path_to_task_config arguments
                    combine_slides_per_patient = False, # Only relevant for patient-level tasks with multiple slides per patient. See NOTE below.
                    cost = 1,
                    balanced = False,
                    saveto = './_test_linprobe'
                )
experiment.train()
experiment.test()
result = experiment.report_results(metric = 'macro-ovr-auc')
```
> [!NOTE]  
> Regarding the `combine_slides_per_patient` argument: If True, will perform early fusion by combining patches from all slides in to a single bag prior to pooling. If False, will pool each slide individually and take the mean of the slide-level features. The ideal value of this parameter depends on what pooling model you are using. For example, Titan requires this to be False because it uses spatial information (patch coordinates) during pooling. If a model doesn't use spatial information, you can usually set this to True, but it's best to consult with model documentation.

> [!NOTE]  
> Provide `patch_embeddings_dirs` so Patho-Bench knows where to find the patch embeddings for pooling. While `Trident` also supports pooling, it doesn't handle patient-level tasks with multiple slides per patient. Patho-Bench uses a generalized pooling function for multi-slide fusion. Patho-Bench requires Trident patch-level features, NOT slide-level features.

## 🛋️ Advanced Usage: Large-scale benchmarking

Patho-Bench offers a `Runner` class for large parallel runs with automatic GPU load balancing and experiment monitoring. Edit the following files:
1. `./advanced_usage/configs/tasks.yaml`: Define tasks. Tasks separated by spaces run in series, while newline-separated tasks run in parallel.
2. `./advanced_usage/configs/patch_embeddings_paths.yaml`: Dictionary of patch embedding locations, indexed by datasource and model name. Extract using Trident and provide paths here.
3. `./advanced_usage/run.sh`: Define evaluation type and parameters.
4. `./advanced_usage/configs/`: Hyperparameter YAMLs for each evaluation framework. Provide multiple newline-separated values to run all hyperparameter combinations in series.

> [!NOTE]
> Be mindful of the hardware constraints of your machine when determining how many parallel experiments to run. Some evaluations, e.g. finetuning, are more compute-intensive than others.

1. Go to `./advanced_usage/configs/tasks.yaml` and input which tasks you want to run. Here's a potential set of tasks that you can run
    - **Note**: Space-separated task codes are run sequentially while newline-separated args are run in parallel.
    - **Note**: To run experiments in which you train on cohort A and test on cohort B, you need to construct the task code as follows in the `tasks.yaml` file: {train_dataset}=={test_dataset}--{task}. For example, `cptac_ccrcc==mut-het-rcc--BAP1_mutation` will run train BAP1 mutation prediction on CPTAC CCRCC and test on MUT-HET-RCC.
2. Depending on the evaluation framework, navigate to the correct folder at `./advanced_usage/configs/` and define the set of hyperparameters you want to run. For linear probe, you could define:
    ```yaml
    COST: # Regularization cost
    - 0.1
    - 0.5
    - 1.0
    - 10
    - adaptive

    BALANCED: # Balanced class weights
    - True

    NUM_BOOTSTRAPS: # Number of bootstrap iterations for single-fold tasks
    - 100
    ```
> [!NOTE]
> Instead of providing a list of `COST` values, you can set `COST: auto` to automatically sweep over `np.logspace(np.log10(10e-6), np.log10(10e5), num=45)`. This behavior can be modified in `ExperimentFactory.py`.

3. Navigate to `./advanced_usage/run.sh` and edit the command to run the desired evaluation:
    ```bash
    python run.py \
        --experiment_type linprobe \
        --model_name titan \
        --tasks_yaml configs/tasks.yaml \
        --combine_slides_per_patient False \  # This parameter is different for different models. Titan requires this to be False.
        --saveto ../artifacts/example_runs/titan_linprobe \
        --hyperparams_yaml "configs/linprobe/linprobe.yaml" \
        --pooled_dirs_root "../artifacts/pooled_features" \
        --patch_dirs_yaml "configs/patch_embeddings_paths.yaml" \
        --splits_root "../artifacts/splits" \
        --conda_venv pathobench \
        --delay_factor 2 # Controls how much each pane is delayed by. Pane i will start after (i**delay_factor) seconds
    ```
4. Run `./run.sh`: This command will launch `tmux` windows for each parallel process and will close tmux windows automatically as the tasks are done.
    - You may need to make the script executable first: `chmod +x run.sh`

## Funding
This work was funded by NIH NIGMS [R35GM138216](https://reporter.nih.gov/search/sWDcU5IfAUCabqoThQ26GQ/project-details/10029418).

## Reference

If you find our work useful in your research or if you use parts of this code, please consider citing the following papers:

```
@article{zhang2025standardizing,
  title={Accelerating Data Processing and Benchmarking of AI Models for Pathology},
  author={Zhang, Andrew and Jaume, Guillaume and Vaidya, Anurag and Ding, Tong and Mahmood, Faisal},
  journal={arXiv preprint arXiv:2502.06750},
  year={2025}
}

@article{vaidya2025molecular,
  title={Molecular-driven Foundation Model for Oncologic Pathology},
  author={Vaidya, Anurag and Zhang, Andrew and Jaume, Guillaume and Song, Andrew H and Ding, Tong and Wagner, Sophia J and Lu, Ming Y and Doucet, Paul and Robertson, Harry and Almagro-Perez, Cristina and others},
  journal={arXiv preprint arXiv:2501.16652},
  year={2025}
}
```

<img src=".github/logo.png">
