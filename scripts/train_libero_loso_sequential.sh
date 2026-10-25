#!/bin/bash
# Run LOSO folds sequentially in the current GPU allocation, without Slurm launches.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
seed="${GR00T_TRAIN_SEED:-42}"
run_root="${LOSO_RUN_ROOT:-}"
dry_run=0
held_out_suites=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed)
            [[ $# -ge 2 ]] || { echo "--seed requires a value" >&2; exit 2; }
            seed="$2"; shift 2 ;;
        --seed=*) seed="${1#*=}"; shift ;;
        --output-root)
            [[ $# -ge 2 ]] || { echo "--output-root requires a value" >&2; exit 2; }
            run_root="$2"; shift 2 ;;
        --output-root=*) run_root="${1#*=}"; shift ;;
        --dry-run) dry_run=1; shift ;;
        -h|--help)
            echo "Usage: bash scripts/train_libero_loso_sequential.sh [--seed N] [--output-root DIR] [--dry-run] [spatial object goal long]"
            exit 0 ;;
        *) held_out_suites+=("$1"); shift ;;
    esac
done
if [[ ! "$seed" =~ ^[0-9]+$ ]] || (( ${#seed} > 10 )) || (( 10#$seed > 4294967295 )); then
    echo "seed must be an integer between 0 and 4294967295" >&2
    exit 2
fi
seed=$((10#$seed))
if [[ ${#held_out_suites[@]} -eq 0 ]]; then
    held_out_suites=(spatial object goal long)
fi
declare -A seen=()
for suite in "${held_out_suites[@]}"; do
    case "$suite" in spatial|object|goal|long) ;; *) echo "Unknown suite: $suite" >&2; exit 2 ;; esac
    [[ -z "${seen[$suite]:-}" ]] || { echo "Duplicate suite: $suite" >&2; exit 2; }
    seen[$suite]=1
done
run_root="${run_root:-$repo/outputs/libero_loso/$(date -u +%Y%m%dT%H%M%SZ)_seed${seed}_sequential}"
[[ "$run_root" = /* ]] || run_root="$repo/$run_root"
[[ ! -e "$run_root" ]] || { echo "Refusing to overwrite existing output root: $run_root" >&2; exit 1; }
python_bin="${GR00T_PYTHON:-/home/vhoangth2/miniconda3/envs/gr00t/bin/python}"
[[ -x "$python_bin" ]] || { echo "Python executable not found: $python_bin" >&2; exit 1; }
data_root="${LIBERO_DATA_ROOT:-$repo/demo_data/libero_loso_clean}"
for suite in spatial object goal long; do
    [[ -f "$data_root/$suite/meta/modality.json" ]] || { echo "Dataset metadata missing: $data_root/$suite/meta/modality.json" >&2; exit 1; }
done
printf 'Sequential LOSO: seed=%s; folds=%s\nOutput root: %s\n' "$seed" "${held_out_suites[*]}" "$run_root"
if [[ "$dry_run" = 1 ]]; then
    for held_out in "${held_out_suites[@]}"; do
        printf 'GR00T_LAUNCH_MODE=direct '
        printf '%q ' bash "$repo/scripts/libero_loso.slurm" "$held_out" "$run_root" "$seed"
        printf '\n'
    done
    exit 0
fi
"$python_bin" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable: run inside your existing GPU allocation"; print("Using GPU:", torch.cuda.get_device_name(0))'
mkdir -p "$(dirname "$run_root")"
mkdir "$run_root"
printf 'held_out_suite\tseed\tstatus\texit_code\toutput_dir\n' > "$run_root/sequence.tsv"
for held_out in "${held_out_suites[@]}"; do
    printf 'Starting held-out %s, seed %s\n' "$held_out" "$seed"
    printf '%s\t%s\trunning\t\t%s\n' "$held_out" "$seed" "$run_root/heldout_$held_out" >> "$run_root/sequence.tsv"
    if GR00T_LAUNCH_MODE=direct bash "$repo/scripts/libero_loso.slurm" "$held_out" "$run_root" "$seed" \
        2>&1 | tee "$run_root/heldout_${held_out}.log"; then
        printf '%s\t%s\tcompleted\t0\t%s\n' "$held_out" "$seed" "$run_root/heldout_$held_out" >> "$run_root/sequence.tsv"
    else
        fold_exit=$?
        printf '%s\t%s\tfailed\t%s\t%s\n' "$held_out" "$seed" "$fold_exit" "$run_root/heldout_$held_out" >> "$run_root/sequence.tsv"
        echo "Stopped after $held_out failed; inspect $run_root/heldout_${held_out}.log" >&2
        exit "$fold_exit"
    fi
done
printf 'All selected folds completed. Status: %s/sequence.tsv\n' "$run_root"
