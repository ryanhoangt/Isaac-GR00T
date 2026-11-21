# Command reference: copy sections into separate terminals, not bash this file.
# Both terminals must be on the same compute node inside a GPU allocation.
# All16-trained checkpoint, evaluated on the SAME 4 tasks selected by the primary
# manifest: ButtonUnmaskSwap, PatternLock, PickHighlight, PickXtimes.
# These tasks were seen during all16 training: this is in-distribution evaluation.
# This command does not evaluate all 16 tasks; the current runner is manifest-based.

# TERMINAL 1: start the all16 checkpoint server and leave it running.
# Port 8001 keeps this server distinct from the primary checkpoint on port 8000.
cd /mnt/data/vhoangth2/repos/Isaac-GR00T

env PYTHONPATH=/mnt/data/vhoangth2/repos/Isaac-GR00T \
    USE_TF=0 USE_FLAX=0 \
    HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    NO_ALBUMENTATIONS_UPDATE=1 TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/vhoangth2/miniconda3/envs/gr00t/bin/python -u scripts/inference_service.py \
    --server --http-server --host 127.0.0.1 --port 8001 \
    --model-path /mnt/data/vhoangth2/repos/Isaac-GR00T/outputs/robomme_all16_seed42/checkpoint-5000 \
    --data-config examples.RoboMME.custom_data_config:RoboMMEDataConfig \
    --embodiment-tag new_embodiment \
    --denoising-steps 4 --seed 42

# TERMINAL 2: wait for "Uvicorn running", then check health before starting eval.
cd /mnt/data/vhoangth2/repos/Isaac-GR00T

curl -fsS http://127.0.0.1:8001/health

# Quick run: 4 tasks x 5 episodes. Set --num-episodes 50 for 50 episodes/task.
# Choose a fresh output directory for a new run. Add --resume only to continue
# this exact run; already recorded error episodes are also skipped on resume.
env PYTHONPATH=/mnt/data/vhoangth2/repos/Isaac-GR00T:/mnt/data/vhoangth2/repos/Isaac-GR00T/.cache/robomme_eval_deps \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/vhoangth2/.local/share/mamba/envs/robomme/bin/python -u -m examples.RoboMME.eval \
    --manifest /mnt/data/vhoangth2/datasets/robomme_primary_train_v21/manifest.json \
    --checkpoint /mnt/data/vhoangth2/repos/Isaac-GR00T/outputs/robomme_all16_seed42/checkpoint-5000 \
    --benchmark-root /mnt/data/vhoangth2/repos/robomme_benchmark \
    --output-dir outputs/robomme_eval/all16_ckpt5000_primary_tasks_seed42_ep5 \
    --server-url http://127.0.0.1:8001 \
    --num-episodes 5 \
    --exec-horizon 16 --max-steps 1300 --seed 42
