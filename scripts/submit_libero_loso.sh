#!/bin/bash
# Submit independent leave-one-suite-out runs; optionally specify suites as arguments.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
held_out_suites=("$@")
if [[ ${#held_out_suites[@]} -eq 0 ]]; then
    held_out_suites=(spatial object goal long)
fi
for suite in "${held_out_suites[@]}"; do
    case "$suite" in spatial|object|goal|long) ;; *) echo "Unknown suite: $suite" >&2; exit 2 ;; esac
done
run_root="${LOSO_RUN_ROOT:-$repo/outputs/libero_loso/$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ -e "$run_root/submissions.tsv" ]]; then
    echo "Already submitted: $run_root/submissions.tsv" >&2
    exit 1
fi
mkdir -p "$repo/outputs/slurm" "$run_root"
for suite in spatial object goal long; do
    test -f "${LIBERO_DATA_ROOT:-$repo/demo_data/libero_loso_clean}/$suite/meta/modality.json"
done
printf 'held_out_suite\tjob_id\toutput_dir\n' > "$run_root/submissions.tsv"
for held_out in "${held_out_suites[@]}"; do
    # CLI resources override any SBATCH_* settings inherited from an interactive shell.
    job_id="$(sbatch --parsable --partition=main --nodes=1 --ntasks=1 \
        --cpus-per-task=8 --mem=128G --gres=gpu:nvidia_h100_80gb_hbm3:1 --time=24:00:00 \
        --chdir="$repo" --job-name="gr00t-loso-$held_out" \
        "$repo/scripts/libero_loso.slurm" "$held_out" "$run_root")"
    printf '%s\t%s\t%s\n' "$held_out" "$job_id" "$run_root/heldout_$held_out" \
        >> "$run_root/submissions.tsv"
    printf 'Submitted held-out %s: %s\n' "$held_out" "$job_id"
done
printf 'Submission manifest: %s/submissions.tsv\n' "$run_root"
