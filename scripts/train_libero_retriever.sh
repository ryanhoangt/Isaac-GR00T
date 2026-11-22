#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/.."
policy_python="${POLICY_PYTHON:-${HOME}/miniconda3/envs/gr00t/bin/python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export USE_TF=0
data_root="${LIBERO_DATA_ROOT:-$PWD/ra-vla-official-assets/ravla_code/ravla_code_no_model/datasets}"
exec "$policy_python" -u -m scripts.rag.1_train_retriever \
  --dataset-paths "$data_root/libero_goal" "$data_root/libero_object" "$data_root/libero_spatial" \
  --retriever-path outputs/retriever/libero_gos_seed42 --seed 42 "$@"
