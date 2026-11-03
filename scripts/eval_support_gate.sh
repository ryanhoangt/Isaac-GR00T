#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo 'Request a Slurm GPU allocation and run this script with srun.' >&2
    exit 1
fi
if [[ -z "${SLURM_STEP_ID:-}" ]]; then
    exec srun --ntasks=1 --cpus-per-task=8 bash scripts/eval_support_gate.sh "$@"
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 USE_TF=0 NO_ALBUMENTATIONS_UPDATE=1 TOKENIZERS_PARALLELISM=false
stage="${1:-}"
if [[ "$stage" != calibrate && "$stage" != eval ]]; then
    echo 'Usage: bash scripts/eval_support_gate.sh {calibrate|eval} [arguments]' >&2
    exit 2
fi
shift
if [[ "$stage" == calibrate ]]; then
    exec "${POLICY_PYTHON:-/home/vhoangth2/miniconda3/envs/gr00t/bin/python}" -u -m scripts.rag.calibrate_support "$@"
fi
exec "${POLICY_PYTHON:-/home/vhoangth2/miniconda3/envs/gr00t/bin/python}" -u -m scripts.rag.eval_support_gate "$@"
