#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/.."
policy_python="${POLICY_PYTHON:-${HOME}/miniconda3/envs/gr00t/bin/python}"
export NO_ALBUMENTATIONS_UPDATE=1
export HF_HUB_OFFLINE=1
export USE_TF=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
exec "$policy_python" -u -m scripts.rag.precompute_retrieval "$@"
