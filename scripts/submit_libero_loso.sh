#!/bin/bash
# Submit LOSO fine-tuning: [--seed N] [--dry-run] [spatial object goal long].
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
seed="${GR00T_TRAIN_SEED:-42}"
dry_run=0
held_out_suites=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed)
            [[ $# -ge 2 ]] || { echo "--seed requires a value" >&2; exit 2; }
            seed="$2"; shift 2 ;;
        --seed=*) seed="${1#*=}"; shift ;;
        --dry-run) dry_run=1; shift ;;
        -h|--help)
            echo "Usage: bash scripts/submit_libero_loso.sh [--seed N] [--dry-run] [spatial object goal long]"
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
for suite in "${held_out_suites[@]}"; do
    case "$suite" in spatial|object|goal|long) ;; *) echo "Unknown suite: $suite" >&2; exit 2 ;; esac
done
run_root="${LOSO_RUN_ROOT:-$repo/outputs/libero_loso/$(date -u +%Y%m%dT%H%M%SZ)_seed${seed}}"
[[ "$run_root" = /* ]] || run_root="$repo/$run_root"
if [[ -e "$run_root/submissions.tsv" ]]; then
    echo "Already submitted: $run_root/submissions.tsv" >&2
    exit 1
fi
for suite in spatial object goal long; do
    test -f "${LIBERO_DATA_ROOT:-$repo/demo_data/libero_loso_clean}/$suite/meta/modality.json"
done
if [[ "$dry_run" = 0 ]]; then
    mkdir -p "$repo/outputs/slurm" "$run_root"
    printf 'held_out_suite\tjob_id\toutput_dir\tseed\n' > "$run_root/submissions.tsv"
else
    printf 'Dry run: seed=%s; output root=%s\n' "$seed" "$run_root"
fi
for held_out in "${held_out_suites[@]}"; do
    # CLI resources override any SBATCH_* settings inherited from an interactive shell.
    command=(sbatch --parsable --partition=main --nodes=1 --ntasks=1 \
        --cpus-per-task=8 --mem=128G --gres=gpu:nvidia_h100_80gb_hbm3:1 --time=24:00:00 \
        --chdir="$repo" --job-name="gr00t-loso-${held_out}-s${seed}" \
        "$repo/scripts/libero_loso.slurm" "$held_out" "$run_root" "$seed")
    if [[ "$dry_run" = 1 ]]; then
        printf '%q ' "${command[@]}"
        printf '\n'
        continue
    fi
    job_id="$("${command[@]}")"
    printf '%s\t%s\t%s\t%s\n' "$held_out" "$job_id" "$run_root/heldout_$held_out" "$seed" \
        >> "$run_root/submissions.tsv"
    printf 'Submitted held-out %s, seed %s: %s\n' "$held_out" "$seed" "$job_id"
done
if [[ "$dry_run" = 0 ]]; then
    printf 'Submission manifest: %s/submissions.tsv\n' "$run_root"
fi
