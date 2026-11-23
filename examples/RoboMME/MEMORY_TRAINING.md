# FrameSamp+Modul fine-tuning (step 5)

The existing `scripts/gr00t_finetune.py` now supports RoboMME memory training.
Specify both `--memory-config` and `--memory-cache`. The former is a YAML file;
the latter is the completed visual-cache directory from [step 2](FEATURE_CACHE.md).
The dataset uses the execution-only GR00T conversion and its own normalization
statistics, as described in [step 3](MEMORY_DATASET.md). The model path is the
[step-4 memory-as-modulator architecture](MEMORY_MODEL.md).

Run from this checkout inside the **existing GPU allocation**, preserving its
`CUDA_VISIBLE_DEVICES`. These commands do not request another allocation:

```bash
export PYTHONPATH=/mnt/data/vhoangth2/repos/Isaac-GR00T
export USE_TF=0 USE_FLAX=0 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export NO_ALBUMENTATIONS_UPDATE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export WANDB_MODE=offline WANDB_PROJECT=gr00t-robomme
ROBOMME_PYTHON=/home/vhoangth2/miniconda3/envs/gr00t/bin/python

"$ROBOMME_PYTHON" -u scripts/gr00t_finetune.py \
    --dataset-path /mnt/data/vhoangth2/datasets/robomme_primary_train_v21 \
    --base-model-path /mnt/data/vhoangth2/hf_cache/hub/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e \
    --memory-config examples/RoboMME/config/perceptual-framesamp-modul.yaml \
    --memory-cache outputs/robomme_memory_features \
    --data-config examples.RoboMME.memory_data_config:RoboMMEMemoryDataConfig \
    --embodiment-tag new_embodiment --num-gpus 1 --seed 42 \
    --no-tune-visual --no-tune-llm \
    --batch-size 2 --gradient-accumulation-steps 16 \
    --learning-rate 3e-5 --warmup-ratio 0.05 --max-steps 20000 \
    --save-steps 1000 --no-save-final-model \
    --dataloader-num-workers 2 --dataloader-prefetch-factor 2 \
    --video-backend torchvision_av \
    --output-dir outputs/robomme_memory_primary_seed42
```

The starting recipe has effective batch size 32 on one GPU. It is a full
action-head fine-tuning recipe; its long-run behavior and batch-size memory
requirements have not been measured. Both vision and language are frozen by
default; the current GR00T action-head projectors, diffusion model and memory
modules train. Initialize primary from the original GR00T base rather than an
all16-trained checkpoint.

For all16, change the dataset to
`/mnt/data/vhoangth2/datasets/robomme_gr00t_lerobot_v21`, use a separate output
such as `outputs/robomme_memory_all16_seed42`, and optionally use the existing
all16 budget of 30,000 steps. Both protocols can read the same frozen full-task
visual cache; only the selected execution dataset supplies samples and stats.
The small two-episode cache used for integration tests cannot train the full
all16/primary datasets. Full-cache extraction remains a separate operation.

The memory data config is also selected automatically when memory flags are
paired with the default data-config name or the vanilla RoboMME data config.
The explicit memory name above makes the recipe easy to inspect. Other custom
configs must preserve the memory fields. Frozen vision/connector weights are
required; a trainable vision encoder would invalidate cached features. LoRA is
currently rejected for this path because the existing PEFT wrapper would freeze
the newly added memory parameters. `tune_diffusion_model` must remain enabled.

The trainer defaults to `spawn` workers for memory. The general optional flag
`--dataloader-multiprocessing-context` can select a process context explicitly;
vanilla training retains its previous default. With zero workers, context and
prefetching are omitted. `--memory-max-open-episodes` controls the per-worker
memmap limit (default 8).

Before constructing the optimizer, the script:

1. Loads and validates the execution dataset/cache alignment.
2. Loads pretrained GR00T and adapts action dimensions/horizon as before.
3. Checks frozen vision/connector weights and extraction/preprocessing identity
   against the cache. This inspects live weights without moving or casting them.
4. Calls `model.enable_memory` and persists the resolved model architecture.
5. Writes memory configuration and training provenance, then constructs the
   normal Trainer/AdamW optimizer and DataLoader. New parameters are included.

Each checkpoint includes:

```text
config.json                          # action_head_cfg.memory_config
model*.safetensors
optimizer.pt / scheduler.pt / trainer_state.json / rng_state*.pth
experiment_cfg/
  metadata.json                      # normalization/modality metadata
  memory_config.yaml                 # resolved memory YAML
  memory_training.json               # identity and embedded extraction specs
```

Resume with the same command/output and `--resume`. The script pins the latest
checkpoint, loads its actual model weights, checks those vision weights against
the cache, and validates the saved training identity before rewriting sidecars.
The same pinned checkpoint is then passed to Trainer to restore optimizer,
scheduler, RNG and global step. It rejects missing memory provenance and changed
data/normalization, memory architecture, cache, seed, tuning flags, effective
batch configuration or optimization recipe. Fresh memory runs refuse existing
training artifacts; use another output directory for another run.

Operational controls such as checkpoint frequency/retention, worker count and
training budget may change. Keep the original max-steps/epochs for an ordinary
interrupted-run resume; increasing the budget also changes the future cosine LR
schedule. Changing workers or resuming stochastic image augmentation is not
claimed to reproduce an uninterrupted run bit-for-bit. Keep dataset/cache paths
stable for this initial implementation. Checkpoint retention remains unlimited
unless `--save-total-limit` is explicitly supplied.

Validation uses a tiny real SigLIP/Eagle backbone and the actual CLI `main`,
TrainRunner, optimizer, collator and spawned workers. It stops after step 1,
saves, resumes to step 2 and checks optimizer counters. It also covers frozen
weight/preprocessing mismatch, incompatible resume, zero-worker memory training
and the vanilla training path.

On 2026-11-23, a full-backbone GR00T GPU smoke ran inside existing allocation
66728 with two spawned workers, real current images and real cached RoboMME
history/state/action labels (332 execution samples available). It trained one
optimizer step, saved, restored the same memory weights and optimizer step 1,
then continued to step 3. All 100 memory parameter tensors had finite, nonzero
gradients. Checkpoints include the resolved configs, optimizer and scheduler.
Peak allocated VRAM was 11,187,591,680 bytes (10.42 GiB).

To fit the existing allocation's approximately 19 GiB free VRAM, this GPU smoke
used a **test-only trainable-parameter mask**, updating only the new memory
modules. The normal CLI recipe above tunes the full action head. Full-head
optimizer/resume behavior is covered by the tiny-model trainer tests; full-size
full-head GPU training has not been run. The smoke checkpoint records this
override and is for integration debugging. No full benchmark training run was
launched. A separately submitted smoke job 66888 remained pending and was
canceled; the successful smoke used allocation 66728.

Artifacts: `outputs/robomme_memory_training_smoke_step5/smoke_summary.json` and
`outputs/robomme_memory_training_smoke_step5/train/checkpoint-3/`.
Stateful memory serving/client support is described in [step 6](MEMORY_INFERENCE.md).
Benchmark protocol selection and evaluation recipes remain step 7.
