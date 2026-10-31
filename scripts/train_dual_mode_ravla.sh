#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo 'First request a GPU: salloc --ntasks=1 --cpus-per-task=8 --mem=64G --gres=gpu:1' >&2
    exit 1
fi
if [[ -z "${SLURM_STEP_ID:-}" ]]; then
    exec srun --ntasks=1 --cpus-per-task=8 bash scripts/train_dual_mode_ravla.sh "$@"
fi
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1
export NO_ALBUMENTATIONS_UPDATE=1 USE_TF=0 TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT=icl-vla-uncertainty
exec "${POLICY_PYTHON:-/home/vhoangth2/miniconda3/envs/gr00t/bin/python}" -m scripts.rag.train_dual_mode "$@"
