#!/usr/bin/env bash
# First-pass evaluation of the matched Pilot 2 baseline and dual checkpoints.
set -euo pipefail

if (( $# < 1 )); then
    echo "Usage: bash scripts/eval_pilot2.sh NEW_OUTPUT_ROOT [Pilot 2 screen options]" >&2
    exit 2
fi

run_root="$1"
shift
python_bin="${GR00T_PYTHON:-/home/vhoangth2/miniconda3/envs/gr00t/bin/python}"
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$repo"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export NO_ALBUMENTATIONS_UPDATE=1
export USE_TF=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export WANDB_PROJECT=icl-vla-uncertainty

exec "$python_bin" -u -m examples.Libero.pilot2 screen --output "$run_root" "$@"
