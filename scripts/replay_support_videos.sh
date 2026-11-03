#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo 'Run inside a Slurm allocation. Software rendering needs CPUs only.' >&2
    exit 1
fi
if [[ -z "${SLURM_STEP_ID:-}" ]]; then
    exec srun --ntasks=1 --cpus-per-task="${REPLAY_CPUS:-4}" --cpu-bind=none bash scripts/replay_support_videos.sh "$@"
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export LP_NUM_THREADS=1 HF_HUB_OFFLINE=1 USE_TF=0 NO_ALBUMENTATIONS_UPDATE=1
export MUJOCO_GL="${REPLAY_GL:-osmesa}"
export PYOPENGL_PLATFORM="$MUJOCO_GL"
exec "${LIBERO_PYTHON:-/home/vhoangth2/miniconda3/envs/libero/bin/python}" -u \
    -m examples.Libero.eval.replay_support_videos "$@"
