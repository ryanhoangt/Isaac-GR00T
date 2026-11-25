# Train RA-VLA on RoboMME

Use `scripts.rag.3_train_ravla` with the trained retriever and its full primary
memory. The script infers data config/base GR00T from the memory and retriever
training records. Explicit conflicting overrides are rejected.

```bash
cd /mnt/data/vhoangth2/repos/Isaac-GR00T-RA-VLA

srun --jobid=67148 --overlap --nodes=1 --ntasks=1 --cpus-per-task=4 \
  --gres=gpu:1 \
  env PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 \
    USE_TF=0 USE_FLAX=0 HF_HUB_OFFLINE=1 \
    NO_ALBUMENTATIONS_UPDATE=1 TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/vhoangth2/miniconda3/envs/gr00t/bin/python -u \
    -m scripts.rag.3_train_ravla \
    --dataset-paths /mnt/data/vhoangth2/datasets/robomme_primary_train_v21 \
    --retriever-path outputs/retriever/robomme_primary_1000_seed42 \
    --memory-name memory_primary_seed42 \
    --vla-path outputs/ravla/robomme_primary_3epochs_seed42 \
    --batch-size 32 --num-epochs 3 --num-workers 2 \
    --video-cache-size 2 --seed 42 \
    --wandb-mode online --wandb-name robomme_primary_3epochs_seed42
```

Defaults retain the original policy recipe: K=1, margin=0.01, adherence weight
0.1, AdamW learning rate 2e-4 and cosine decay. At 338,246 primary frames and
batch size 32 (`drop_last=True`), three epochs are 31,710 optimizer steps.
This is an initial reproduction budget, not a tuned RoboMME result.

For a bounded GPU smoke test, choose a fresh output path, add `--max-steps 3`
and `--no-wandb-enable`, and remove the W&B options. `max_steps` overrides the
epoch budget. A run stopping mid-epoch saves `checkpoint_step_3`; complete
epochs retain names `checkpoint_1`, `checkpoint_2`, etc. The final checkpoint
is always saved. Checkpoints contain weights/config/training normalization;
optimizer/RNG resume is not implemented. Existing output runs are rejected.

## Canonical task routing

With memory task routing enabled, `FramePathsMixtureDataset` validates the
episode-task index and attaches `memory_key` to each transformed sample.
`RAVLADataCollator` removes that metadata before GR00T image/text collation and
passes the canonical task as the lookup key. The original instruction stays in
the language input. Two tasks sharing an instruction retain separate banks;
new instructions can retrieve within their canonical task's three-demo bank.

Unindexed LIBERO datasets retain the original instruction-key fallback.
Canonical training rejects incomplete/capped memory, missing/extra task banks,
insufficient candidates for K, mismatched data config, and changed
retriever/normalization/dataset metadata. File fingerprints match content and
filenames across VM moves rather than requiring old absolute paths.

Normalization is loaded from the retriever's training `metadata.pt`, assigned
to every dataset transform, and copied into policy checkpoints. The backbone
and retriever remain frozen. Only trainable policy parameters enter AdamW;
action MSE and adherence differences are computed in FP32 with the query action
mask (RoboMME: eight physical coordinates, padded to 32).

The DataLoader uses bounded video caching by default and prefetches one batch
per worker. Each dataset has its own transform object. Memory tensors stay on
the training GPU, so training VRAM includes the full memory file as well as
policy/optimizer/activations.

## Padding-aware RoboMME checkpoints

RoboMME memory instructions have different token lengths. For this mode,
the VL projection/self-attention processes groups of unpadded equal-length
sequences, then restores masks/layout. The DiT cross-attention masks both query
and retrieved padding, including the matching mask when context is shuffled
for the adherence branch. Retrieved action tokens remain valid context.

This mode is persisted as `ravla_mask_padding=true` in policy config and is
restored on checkpoint load. Existing checkpoints without that flag keep their
original attention behavior. Canonical key type and RoboMME data-config path
are also saved in policy config. Tests exercise padding invariance and gradient
flow, including K=1/K=2 and the random-context branch.

## Outputs and subsequent evaluation work

- `training_config.json`: resolved training settings and source paths.
- `memory_validation.json`: validated memory source, key type, bank/candidate
  counts, manifest and task-map fingerprints.
- `first_batch.json`: actual lookup keys, state/action shapes and physical
  action dimensions for the first training batch.
- `metrics.jsonl`: per-step total/MSE/margin losses, gradient norm and LR.
- `completed.json`: step/epoch counts, checkpoints, parameter-change/frozen
  component checks, peak VRAM and elapsed time.
- `checkpoint_*`: policy weights/config and training `metadata.pt`.

W&B uses the existing policy logging options; online/offline/disabled modes
remain supported. This step prepares policy training and checkpoint loading.
Server-side canonical routing and the RoboMME evaluator backend are documented
in [EVAL_ROBOMME.md](EVAL_ROBOMME.md). Separate held-out support memory remains
to be prepared before the held-out experiment.

Run CPU checks inside an allocation:

```bash
USE_TF=0 USE_FLAX=0 HF_HUB_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1 \
  /home/vhoangth2/miniconda3/envs/gr00t/bin/python -m pytest -q \
  tests/test_ravla_robomme.py tests/test_ravla_training_inputs.py \
  tests/test_ravla_training_logging.py tests/test_retriever_sampling.py \
  tests/test_memory_builder.py
```

Validation in allocation 67148: 46 CPU tests passed. A real GPU run used the
full 12-task/2,830-candidate primary memory, batch 32 and two workers, stopping
after exactly three optimizer steps. Losses/gradients were finite, the retrieval
encoder changed, and the frozen retriever/backbone samples stayed unchanged
with no gradients. Peak allocated/reserved VRAM was 39.45/42.03 GiB.

The saved `checkpoint_step_3` was reloaded in a fresh process. Padding mode and
training normalization were preserved. Inference without target actions on
VideoUnmask and VideoUnmaskSwap sharing one instruction returned finite
`[2,16,32]` action chunks. Artifacts, including `completed.json`,
`first_batch.json` and `reload_validation.json`, are under
`outputs/ravla/robomme_primary_smoke3_b32_seed42_20261124/`.
No full training or environment rollout was run; these checks establish pipeline
compatibility rather than task success rate.
