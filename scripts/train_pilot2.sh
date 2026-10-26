#!/usr/bin/env bash
# Run both matched variants sequentially in the current GPU allocation.
set -euo pipefail

if (( $# < 1 )); then
    echo "Usage: bash scripts/train_pilot2.sh NEW_OUTPUT_ROOT [Pilot 2 train options]" >&2
    echo "Example: bash scripts/train_pilot2.sh outputs/pilot2/run_2k_20261026 --max-steps 2000" >&2
    exit 2
fi

run_root="$1"
shift
python_bin="${GR00T_PYTHON:-/home/vhoangth2/miniconda3/envs/gr00t/bin/python}"
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$repo"
if [[ -e "$run_root" ]]; then
    echo "Output root already exists; choose a new directory: $run_root" >&2
    exit 2
fi
mkdir -p -- "$run_root"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export NO_ALBUMENTATIONS_UPDATE=1
export USE_TF=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export WANDB_PROJECT=icl-vla-uncertainty

for variant in baseline dual; do
    echo "Starting $variant; log: $run_root/$variant.log"
    "$python_bin" -u -m examples.Libero.pilot2 train "$@" \
        --variant "$variant" --output "$run_root/$variant" \
        2>&1 | tee "$run_root/$variant.log"
done
echo "Completed both variants: $run_root"
