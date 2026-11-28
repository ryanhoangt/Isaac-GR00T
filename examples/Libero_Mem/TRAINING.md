# Two-fold vanilla GR00T baseline

Unit 6 reuses `scripts/gr00t_finetune.py`, `LiberoDataConfig` and the existing
LeRobot mixture/trainer/checkpoint machinery. The
[`training.py`](training.py) launcher validates inputs, records a manifest and
runs selected folds sequentially inside the current GPU allocation. It does
not submit Slurm jobs. Both folds start independently from the same pretrained
GR00T-N1.5 checkpoint; fold B does not resume fold A.

## Training data and normalization

The default `--mode cotrain` implements the proposal's auxiliary LIBERO training:

| Group | Data | Fraction of every microbatch |
| --- | --- | --- |
| Primary | The selected copied LIBERO-Mem training fold | Exactly 50% |
| Auxiliary | Cleaned Spatial, Object, Goal and Long datasets | Exactly 50% combined |

The primary LIBERO-Mem fold is shuffled and traversed **without replacement**:
every frame anchor occurs exactly once in each complete epoch, as in standalone
BC training. Within the auxiliary half, the four suites have equal sampling
probability; auxiliary trajectories/frames are sampled with replacement and
trajectory selection is weighted by length. Batch size must be even. The mixture weights are
`1, 0.25, 0.25, 0.25, 0.25`, with dataset-length reweighting disabled; they
produce dataset probabilities `0.5, 0.125, 0.125, 0.125, 0.125`.

The launcher defaults to `--primary-sampling epoch`. The paired sampler shuffles
primary frame indices and adds one auxiliary draw per primary frame. The final
batch remains partial when needed; Accelerate padding is disabled so it cannot
repeat primary frames to fill a batch. Thus both groups still contribute exactly
half of that final batch. This mode currently supports one training process and
requires `drop_last=False`. Epoch information travels with sampler indices so
persistent workers also sample auxiliary data using the current epoch.

`--primary-sampling replacement` reproduces the original co-training sampler,
where primary frames are also sampled with replacement. The generic fine-tuner
retains that legacy default; only the LIBERO-Mem launcher's default changes.
Its optional flags are `--dataset-weights`, `--balanced-primary-batches` and
`--primary-sampling`. Existing LIBERO and RoboMME launches keep their prior defaults.

Fold A contains T1/3/5/7/9; fold B contains T2/4/6/8/10. The launcher validates
fold manifests, instruction vocabularies, episode mappings and all parquet/video
paths before starting either fold. Only that fold's LIBERO-Mem data is used for
training. Held-out task demonstrations and evaluation states are not added.
Auxiliary inputs must be standard LIBERO, not LIBERO-Mem datasets.

Co-training uses the existing mixture normalization: min/max over the selected
training datasets, probability-weighted means/variances, and weighted-average
percentiles. Thus each checkpoint's normalization combines its own training
fold with the shared auxiliary data. The held-out LIBERO-Mem fold contributes
no statistics. `--mode mem-only` disables auxiliary inputs and uses only the
fold's normalization; identify this as a separate diagnostic protocol.

## Recipe and execution

Default recipe: 10,000 optimizer updates, batch size 32 on one GPU, accumulation
1, AdamW learning rate `1e-4`, weight decay `1e-5`, cosine scheduling and warmup
ratio `0.05`. The LLM and vision backbone are frozen; the action projector and
diffusion model are trained. This follows the existing fine-tuner defaults.
Data remains float32; the unchanged trainer uses BF16 compute/autocast and TF32.

Both folds use the same seed, base checkpoint, optimizer budget and effective
batch size. Their frame counts differ, so an equal step budget is not an equal
epoch budget. At batch 32, accumulation 1, one LIBERO-Mem epoch takes about
9,247 optimizer updates for fold A and 10,842 for fold B (including the partial
last batch). Thus 10,000 updates cover about 1.08 / 0.92 primary epochs; a run
stopped before an epoch ends has not yet visited all its primary frames.
The 10,000-step default is configurable, not a budget fixed by the
proposal; keep the chosen budget consistent across the compared methods.

From the Isaac-GR00T root, inspect the full plan first:

```bash
PYTHONPATH="$PWD" /home/vhoangth2/miniconda3/envs/gr00t/bin/python \
    scripts/train_libero_mem.py \
    --output-root outputs/libero_mem_baseline_s42 \
    --dataloader-num-workers 2 --dry-run
```

Dry runs validate both folds and print the exact fine-tuner commands without
importing Torch, checking CUDA, creating output or starting a process.
The default data paths are the copied datasets under
`/mnt/data/vhoangth2/datasets/libero_mem_fold_{a,b}_train`, and
`demo_data/libero_loso_clean/{spatial,object,goal,long}`. Override with
`--datasets-root` or `--auxiliary-root`. The default base model is the local
cached GR00T-N1.5 snapshot; `--base-model` accepts another complete local checkpoint.

Run full training yourself in an active GPU allocation:

```bash
srun --jobid=67873 --overlap --ntasks=1 --cpus-per-task=4 --mem=30G \
    env PYTHONPATH="$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 USE_TF=0 \
    /home/vhoangth2/miniconda3/envs/gr00t/bin/python -u \
    scripts/train_libero_mem.py \
    --output-root outputs/libero_mem_baseline_s42 \
    --folds A B --mode cotrain --primary-sampling epoch --seed 42 \
    --max-steps 10000 --save-steps 1000 --save-total-limit 2 \
    --batch-size 32 --gradient-accumulation-steps 1 \
    --dataloader-num-workers 2 --report-to tensorboard
```

Replace the allocation ID when necessary. `--folds A` or `--folds B` runs one
direction. The launcher preserves Slurm's GPU assignment. TensorBoard is the
default local logger; `--report-to wandb` enables the existing W&B integration
with your environment's project/entity settings. Training uses local cached
model assets with the Hugging Face offline flags enabled.

For a small smoke test, choose a fresh output root and use:

```bash
# Append these settings to the launcher command above.
--max-steps 2 --save-steps 2 --save-total-limit 1 \
--batch-size 2 --dataloader-num-workers 0
```

`max_steps` must be divisible by `save_steps`, ensuring the final update is
saved. Existing output roots are refused. A failed fold stops the sequence
before the next starts; completed earlier results remain recorded.

## Outputs and evaluation

Each root contains `summary.json`, and each `fold_a/` or `fold_b/` contains:

- `run_manifest.json`: fold/task identities, dataset metadata hashes, mixture
  weights/probabilities, normalization scope, seed, base model/config hash,
  complete recipe, exact command and Slurm job ID.
- `train.log`, local TensorBoard `runs/`, and final `trainer_state.json`.
- `checkpoint-N/`: model weights/config, optimizer/scheduler/RNG state and
  `experiment_cfg/metadata.json`, using the existing checkpoint callback.
- `checkpoints.json`: retained complete checkpoints with optimizer step and epoch.

The launcher checks that the final optimizer update exists and each checkpoint's
normalization metadata matches the training run. Reuse the
[unit-5 inference and evaluation commands](EVALUATION.md) with, for example,
`outputs/libero_mem_baseline_s42/fold_a/checkpoint-10000` and
`--training-fold A --task-group held-out`. Serve using the same
`LiberoDataConfig` and `new_embodiment`; inference loads that checkpoint's
normalization automatically. The proposal's evaluation uses
`--selection unseen --num-episodes 20` on each held-out task.

## Local validation

On allocation `67873` (H100 80GB, four CPUs, 32 GiB RAM), both folds completed
two optimizer updates with batch 2 and zero workers. Final training losses were
approximately **0.7131 / 0.6716**. Both standalone `checkpoint-2` models were
loaded through the existing HTTP service and run in the unit-5 evaluator on an
unseen held-out initial state (T2 for A, T1 for B), with eight action steps and
a valid 20 FPS two-view video. These truncated smoke rollouts are not task
performance measurements.

The independent validator recomputed normalization from each manifest's exact
training datasets and compared the entire checkpoint metadata. Both matched.
It also verified that a pretrained action-head weight changed after training
(maximum absolute difference approximately `1e-4`). The local report is
`outputs/libero_mem_training_smoke_unit6/validation.json`.

A separate fold-A check also completed two optimizer updates with the full
recipe's **batch 32 and two workers**, saving a complete checkpoint. Its report
is `outputs/libero_mem_training_recipe_smoke_unit6/summary.json`. This checks
the batch/worker configuration used in the full command above without launching
the full 10,000-step run.

After the primary-sampling revision, another two-update GPU check used batch 32,
two workers and `--primary-sampling epoch` in allocation `67919`. It saved a
complete checkpoint under `outputs/libero_mem_primary_epoch_smoke/`. Earlier
unit-6 checkpoint/reload reports above used the original replacement sampler.
`tests/test_primary_epoch_sampling.py` verifies exact frame coverage, partial
batches, epoch shuffling, persistent-worker epoch updates and deterministic
sampler resume/skip behavior, including the actual Trainer/Accelerate loader.

A five-update VRAM profile for each fold on allocation `67919`, with batch 32,
accumulation 1, two workers, BF16/TF32 and the frozen LLM/vision recipe, measured
**27.70 GiB peak PyTorch allocated**, **29.36 GiB peak reserved**, and
**30.10 GiB sampled total GPU usage**. PyTorch peaks use its allocator counters;
total usage is polled with `nvidia-smi`. These are short-run observations, not
the maximum over a full 10,000-update run. Reports are under
`outputs/libero_mem_vram_profile/summary.json`.

All **108 focused/regression tests passed**, covering exact microbatch composition with real DataLoaders (0/2
workers), deterministic sampling, legacy unbalanced behavior, fold separation,
malformed inputs, dry runs, seed/command forwarding, overwrite protection,
failure stopping and checkpoint normalization. Runtime validation also covers
reinitializing the same LIBERO namespace checkout in a process while rejecting
mixed or foreign checkouts.

Full training has not been launched as part of implementation.
