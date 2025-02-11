#!/usr/bin/env bash

python run.py \
  --experiment_type coxnet \
  --model_name threads \
  --tasks_yaml configs/tasks.yaml \
  --combine_slides_per_patient False \
  --saveto ../artifacts/2025-02-04/threads_coxnet \
  --hyperparams_yaml "configs/coxnet/coxnet.yaml" \
  --pooled_dirs_root "../artifacts/pooled_features" \
  --patch_dirs_yaml "configs/patch_embeddings_paths.yaml" \
  --splits_root "../artifacts/splits_" \
  --conda_venv pathobench \
  --global_delay 0 \
  --delay_factor 2.5 \