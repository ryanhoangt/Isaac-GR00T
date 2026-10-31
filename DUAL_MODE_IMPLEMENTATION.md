# Dual-mode RA-VLA pilot

Implemented separately from the original RA-VLA pipeline. Train on Goal/Object/Spatial; hold out Long. No uncertainty gate, conformal calibration, geometric loss, or recovery guarantee is implemented yet.

## Recipe

- Each sample independently selects action mode (0) or goal mode (1), probability 50/50. Language stays active in both.
- Action mode keeps learned top-1 retrieval and retrieved-action flow initialization. Its explicit negative is a different same-task memory row.
- Goal mode uses another demo of the same task, raw-action DTW alignment, then a **16-control-step** lookahead, clamped at the demo end. It uses both cached-dataset camera views at that milestone. Frozen Eagle encodes these training goals on demand.
- Expert action tokens are replaced with a learned goal token plus horizon-position embeddings. A learned goal-view embedding marks milestone visual tokens. Gaussian source is used in **both training and inference**.
- Goal adherence compares the positive milestone with a frame at least 32 steps away in the **same reference demo**. Both passes mask actions and share flow noise/time/target and dropout randomness. This is a temporal-negative proxy, not a verified semantic negative. Unavailable negatives are excluded from the margin.
- Frozen Eagle vision/LLM; trainable action-head adapters, DiT, and goal embeddings. No retrieved proprioception or proprioceptive noise in this first version.
- Defaults: batch 32, accumulation 1, AdamW LR 2e-4, weight decay 1e-5, cosine schedule, 3 epochs, seed 42, four inference Euler steps. Fixed W&B project: `icl-vla-uncertainty`.
- Released GOS normalization is used for current actions, reference actions, and states. The local GOS cache has **171,996 training frames**, not the nominal complete paper dataset. Default budget: **5,374 steps/epoch; 16,122 steps total**.

The legacy training memory lacks demo provenance, so action-mode retrieval has no new cross-demo exclusion claim. Goal-mode DTW pairing explicitly guarantees different demos. The new action-only control uses the same explicit-negative implementation; it is not an exact reproduction of the original batch-roll margin loss.

## Train

Run from this repo. The launcher requires a Slurm allocation and enters an `srun` worker step automatically when needed; it refuses a plain login-node invocation.

```bash
salloc --ntasks=1 --cpus-per-task=8 --mem=64G --gres=gpu:1
cd /mnt/data/vhoangth2/repos/Isaac-GR00T-RA-VLA

bash scripts/train_dual_mode_ravla.sh \
  --output outputs/dual_mode/gos_dual_seed42 \
  --num-epochs 3 --batch-size 32 --gradient-accumulation 1 \
  --wandb-entity ryan-nlp --wandb-name gos_dual_seed42
```

For a shorter pilot, add `--max-steps 2000`. For a matched action-only control, use a fresh output/name and add `--variant action`. The launcher does not start the control automatically.

Checkpoints are saved every **1,000 optimizer steps**, plus the final step, including optimizer/scheduler/RNG state. Existing output directories are refused. Resume with the **same original recipe**, adding:

```bash
--resume outputs/dual_mode/gos_dual_seed42/checkpoint-1000
```

Normalization metadata, training recipe, data inventory, example goal pairings, per-mode losses/margins, mode counts, gradients, and W&B logs are saved. Resume restores training state; bitwise augmentation replay across changed worker settings is not promised.

## Reference memory and inference after training

Training does not require this additional cache. Before goal-mode evaluation, build a buffer with demo/frame provenance. This command uses only three held-out Long demos per task and performs no weight updates:

```bash
srun --ntasks=1 --cpus-per-task=8 env \
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 \
  NO_ALBUMENTATIONS_UPDATE=1 USE_TF=0 \
  /home/vhoangth2/miniconda3/envs/gr00t/bin/python \
  -m scripts.rag.build_dual_mode_memory \
  --output outputs/dual_mode/long_reference
```

Default stride is 4, with terminal frames included. Add `--resume` with identical settings after interruption. Selection is the first three available episode IDs/task, which may differ from the released paper buffer. Compare methods using the same newly built buffer. Cached backbone/retriever hashes and normalization are checked at inference.

Serve a fixed mode on the worker (`--mode 0` for action, `--mode 1` for goal):

```bash
srun --ntasks=1 --cpus-per-task=8 env \
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 \
  NO_ALBUMENTATIONS_UPDATE=1 USE_TF=0 \
  /home/vhoangth2/miniconda3/envs/gr00t/bin/python \
  -m scripts.rag.serve_dual_mode \
  --checkpoint outputs/dual_mode/gos_dual_seed42/checkpoint-16122 \
  --memory outputs/dual_mode/long_reference --mode 1 --port 8085
```

The server exposes the usual `get_action` RPC plus `get_action_with_context` for provenance. It also accepts an original one-mode RA-VLA checkpoint in mode 0, allowing comparison on the same buffer. A simulator client still needs to be run separately; adaptive switching and a new rollout/recovery launcher are not part of this training implementation.

## Validation

Worker allocation 62764, H100 80GB:

- Real small-head tests: original/dual action-mode output parity; no goal-mode expert-action leakage; goal gradients; shared-dropout adherence; DTW pairing; milestone boundaries.
- **40 tests passed**, including existing training logging and context-VFD regressions; Ruff checks passed.
- Pretrained model + real GOS data: two optimizer updates at batch 8/accumulation 4 and batch 32/accumulation 1. Finite losses/gradients; observed peak allocated VRAM **18.3 / 35.3 GiB** respectively.
- Saved pretrained checkpoint reloads, predicts both modes, verifies reference-action normalization, and has exactly zero expert-action leakage under a fixed goal/source.
- Resume was exercised from optimizer step 1 to 2. The logged step-2 losses match the uninterrupted run; final BF16 head tensors are not bitwise identical (maximum observed absolute difference 2.44e-4). Do not claim bitwise training replay.
- Smoke reference cache covers one dataset task at stride 64. It is a software artifact, not the full stride-4 evaluation buffer.

Artifacts: `outputs/dual_mode/smoke_batch32_20261030/` and its `checkpoint_check.json`. These checks establish execution, not improved goal grounding or recovery success. Full training/evaluation has not been run by the agent.
