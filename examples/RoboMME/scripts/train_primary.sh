env PYTHONPATH=/mnt/data/vhoangth2/repos/Isaac-GR00T \
    USE_TF=0 USE_FLAX=0 \
    HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    NO_ALBUMENTATIONS_UPDATE=1 TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    WANDB_MODE=online WANDB_PROJECT=gr00t-robomme WANDB_LOG_MODEL=false \
  /home/vhoangth2/miniconda3/envs/gr00t/bin/python -u scripts/gr00t_finetune.py \
    --dataset-path /mnt/data/vhoangth2/datasets/robomme_primary_train_v21 \
    --base-model-path /mnt/data/vhoangth2/hf_cache/hub/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e \
    --data-config examples.RoboMME.custom_data_config:RoboMMEDataConfig \
    --embodiment-tag new_embodiment \
    --num-gpus 1 --seed 42 \
    --batch-size 16 --gradient-accumulation-steps 2 \
    --learning-rate 3e-5 --warmup-ratio 0.05 \
    --max-steps 20000 \
    --save-steps 5000 --save-total-limit 4 --no-save-final-model \
    --dataloader-num-workers 2 --dataloader-prefetch-factor 2 \
    --video-backend torchvision_av --report-to wandb \
    --output-dir outputs/robomme_primary_seed42
