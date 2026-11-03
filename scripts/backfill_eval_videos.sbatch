#!/usr/bin/env bash
#SBATCH --job-name=ravla_video_backfill
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --gres=none
#SBATCH --time=24:00:00

set -uo pipefail
cd /mnt/data/vhoangth2/repos/Isaac-GR00T-RA-VLA
batch_dir="outputs/ravla_eval/video_backfill_20261102/job_${SLURM_JOB_ID}"
mkdir -p "$batch_dir"
cp examples/Libero/eval/replay_support_videos.py "$batch_dir/replay_saved_videos.py"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export LP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 HF_HUB_OFFLINE=1 USE_TF=0 NO_ALBUMENTATIONS_UPDATE=1
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
libero_python=/home/vhoangth2/miniconda3/envs/libero/bin/python
workers="${SLURM_CPUS_PER_TASK:-2}"
runs=(
  ravla_local_newbuffer_100_20261031
  support_gate_10trials_15k_20261031
  support_gate_untilq80_10trials_15k_20261031
  dual_action_ckpt5k_5trials_20261101
  dual_action_ckpt10k_5trials_20261101
)
printf 'run\tstate\texit_code\n' > "$batch_dir/status.tsv"
printf 'JOB=%s HOST=%s CPU_WORKERS=%s RENDERER=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$workers" "$MUJOCO_GL"

# Validate both saved-artifact schemas before starting the long render.
for run in "${runs[@]}"; do
  srun --ntasks=1 --cpus-per-task=1 --cpu-bind=none "$libero_python" -u \
    "$batch_dir/replay_saved_videos.py" \
    --run "outputs/ravla_eval/$run" --audit-only || exit 1
done

failures=0
for run in "${runs[@]}"; do
  printf 'START %s\n' "$run"
  printf '%s\trunning\t-\n' "$run" >> "$batch_dir/status.tsv"
  srun --ntasks=1 --cpus-per-task="$workers" --cpu-bind=none "$libero_python" -u \
    "$batch_dir/replay_saved_videos.py" \
    --run "outputs/ravla_eval/$run" \
    --output "outputs/ravla_eval/$run/video_replay_ordered" \
    --workers "$workers"
  code=$?
  if [[ "$code" -eq 0 ]]; then
    printf '%s\tcompleted\t0\n' "$run" >> "$batch_dir/status.tsv"
    printf 'FINISHED %s\n' "$run"
  else
    printf '%s\tfailed\t%s\n' "$run" "$code" >> "$batch_dir/status.tsv"
    printf 'FAILED %s exit=%s; continuing other folders\n' "$run" "$code"
    failures=$((failures + 1))
  fi
done
printf 'BACKFILL_FINISHED failed_folders=%s\n' "$failures"
exit "$failures"
