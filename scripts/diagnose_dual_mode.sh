#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo 'Request a GPU allocation, then run this script (or use srun --jobid=YOUR_JOB).' >&2
    exit 1
fi
if [[ -z "${SLURM_STEP_ID:-}" ]]; then
    exec srun --ntasks=1 --cpus-per-task=8 bash scripts/diagnose_dual_mode.sh "$@"
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 USE_TF=0
export NO_ALBUMENTATIONS_UPDATE=1 TOKENIZERS_PARALLELISM=false
exec "${POLICY_PYTHON:-/home/vhoangth2/miniconda3/envs/gr00t/bin/python}" -u -m scripts.rag.diagnose_dual_mode "$@"
