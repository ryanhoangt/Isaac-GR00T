#!/usr/bin/env bash
#SBATCH --job-name=ravla_endpoint_pilot
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm-logs/endpoint-%j.out
#SBATCH --error=slurm-logs/endpoint-%j.err

set -euo pipefail
cd /mnt/data/vhoangth2/repos/Isaac-GR00T-RA-VLA
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 USE_TF=0 NO_ALBUMENTATIONS_UPDATE=1 TOKENIZERS_PARALLELISM=false
python_bin=/home/vhoangth2/miniconda3/envs/gr00t/bin/python
pilot_output="outputs/dual_mode/endpoint_pilot_20261103/job_${SLURM_JOB_ID}"
printf 'JOB=%s HOST=%s OUTPUT=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$pilot_output"
srun --ntasks=1 "$python_bin" -m pytest -q tests/test_local_dynamics.py tests/test_goal_control.py tests/test_dual_mode_ravla.py
srun --ntasks=1 "$python_bin" -u -m scripts.rag.train_endpoint_pilot \
  --output "$pilot_output" \
  --updates "${ENDPOINT_UPDATES:-1000}" \
  --dynamics-epochs "${DYNAMICS_EPOCHS:-100}" \
  --batch-size "${ENDPOINT_BATCH_SIZE:-4}" \
  --gradient-accumulation "${ENDPOINT_ACCUMULATION:-4}" \
  --save-steps 1000 \
  "$@"
