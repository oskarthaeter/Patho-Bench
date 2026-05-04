#!/usr/bin/env bash

python run.py \
  --experiment_type finetune \
  --model_name abmil \
  --patch_model_name virchow2 \
  --model_kwargs_yaml "configs/abmil_kwargs_virchow2.yaml" \
  --tasks_yaml configs/tasks_cptac_virchow2.yaml \
  --combine_slides_per_patient True \
  --saveto ../artifacts/virchow2_finetune \
  --hyperparams_yaml "configs/finetune/virchow2.yaml" \
  --patch_dirs_yaml "configs/patch_embeddings_paths.yaml" \
  --splits_root "../artifacts/splits" \
  --conda_venv pathobench \
  --delay_interval 5 \
