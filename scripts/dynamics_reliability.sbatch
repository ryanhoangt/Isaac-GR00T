#!/usr/bin/env bash
#SBATCH --job-name=ravla_dynamics_filter
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm-logs/dynamics-filter-%j.out
#SBATCH --error=slurm-logs/dynamics-filter-%j.err

set -euo pipefail
cd /mnt/data/vhoangth2/repos/Isaac-GR00T-RA-VLA
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 USE_TF=0 NO_ALBUMENTATIONS_UPDATE=1 TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR="${TMPDIR:-/tmp}/ravla-dynamics-filter-${SLURM_JOB_ID}"
python_bin=/home/vhoangth2/miniconda3/envs/gr00t/bin/python
pilot_output="outputs/dual_mode/dynamics_reliability_20261103/job_${SLURM_JOB_ID}"
printf 'JOB=%s OUTPUT=%s\n' "$SLURM_JOB_ID" "$pilot_output"
srun --ntasks=1 "$python_bin" -m pytest -q tests/test_dynamics_reliability.py tests/test_local_dynamics.py tests/test_goal_control.py tests/test_dual_mode_ravla.py
srun --ntasks=1 "$python_bin" -u -m scripts.rag.diagnose_dynamics_reliability \
  --output "$pilot_output" "$@"
