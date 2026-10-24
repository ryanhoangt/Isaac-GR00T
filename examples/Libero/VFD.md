# GR00T LIBERO velocity-disagreement monitoring

`scripts/eval_libero_vfd.sh` records a VFD score at each action-generation step,
fits thresholds from a separate calibration set, and plots individual rollouts
with a camera filmstrip. It does not change the existing rollout evaluator.
Alarms are logged but do not stop the actor, so final task outcomes remain observable.

The actor integrates its original flow; the scorer evaluates velocities at the
same intermediate states. The score sums `t/(1-t) * delta_t * squared_L2`, masks
the 25 padded action dimensions, and averages over noise samples. Models must
use the same normalization metadata. The actor executes only action 0 of its
16-action chunk, preserving the current evaluator. Additional uncertainty samples
use a private RNG and do not batch or consume the actual actor's random stream.

## Existing Spatial checkpoint pair

Run from the repository root in a GPU allocation:

```bash
ACTOR=outputs/libero_loso/20261023T175731Z/heldout_spatial/checkpoint-13774
SCORER=outputs/libero_loso/20261023T175731Z/heldout_spatial/checkpoint-6887

# Short end-to-end check; these truncated rollouts cannot be used for calibration.
bash scripts/eval_libero_vfd.sh record \
  --actor "$ACTOR" --scorer "$SCORER" --suite spatial \
  --task-ids 0 --trials 1 --max-steps 5 \
  --gpu-memory-limit-gb 18 --verify-actor --output-dir outputs/libero_vfd/spatial_smoke
```

The two epochs are correlated snapshots from one training run. This is an
exploratory checkpoint-disagreement experiment, not the independently trained
ensemble used in the paper. A stronger experiment uses two different-seed models
trained on the same three source suites, at comparable training budgets. Never
pair different LOSO folds: one member would have trained on the other's held-out suite.

## Calibration and held-out test

The following example uses a global threshold calibrated on successful rollouts
of the held-out Spatial suite. This uses target-suite calibration data and must
be reported as such; it is not a completely zero-shot LOSO detector. A pooled
threshold is useful when there are too few successes per task for taskwise
calibration. At least 20 successful calibration rollouts are required by default.

```bash
# Full default task horizons; no --max-steps here. Check success counts afterward.
bash scripts/eval_libero_vfd.sh record \
  --actor "$ACTOR" --scorer "$SCORER" --suite spatial \
  --trials 20 --initial-state-start 0 \
  --output-dir outputs/libero_vfd/spatial_calibration

bash scripts/eval_libero_vfd.sh calibrate \
  --inputs outputs/libero_vfd/spatial_calibration \
  --scope global --quantile 0.9 \
  --output outputs/libero_vfd/spatial_threshold.json

# Different official initial states: calibration 0..19, test 20..39.
bash scripts/eval_libero_vfd.sh record \
  --actor "$ACTOR" --scorer "$SCORER" --suite spatial \
  --trials 20 --initial-state-start 20 \
  --calibration outputs/libero_vfd/spatial_threshold.json \
  --save-video --output-dir outputs/libero_vfd/spatial_test
```

For source-only calibration, record the same actor/scorer on its three training
suites (Object, Goal and Long for this fold), pass those run directories to
`calibrate --inputs ... --scope global`, then test on Spatial. Such transfer has
no guaranteed target-domain false-positive coverage. `--scope task` instead
fits one threshold per suite/task and requires sufficient successful rollouts
for every included task. Test initial states must not overlap calibration
candidates; the script enforces this even if simulator seeds differ.

The threshold is the empirical 0.9 quantile of successful rollout maxima,
following the paper's convention, not an exact finite-sample conformal guarantee.
The script never chooses a threshold using the test outcomes. Existing output
directories are refused; use a fresh directory for each new recording.

## Outputs and plotting

PNG/PDF plots and `summary.json` are updated immediately after each completed
episode, before the next episode starts. Existing episode plots are not redrawn
automatically. If interrupted, completed episodes remain plotted and the partial
episode is retained as raw data but excluded from plots and metrics.

Each recording contains `manifest.json`, `summary.json`, per-episode `.npz` and
`.json`, incremental `.jsonl` step logs, PNG/PDF plots, and optional MP4 videos.
The NPZ includes uncertainty, per-sample scores, executed actions, and camera
snapshots. Snapshot step indices align with observations before action execution.
The manifest records both checkpoints, score settings, and measured GPU peak
memory. Shortened smoke horizons are labelled and excluded from deployment metrics.
No uncertainty is retroactively synthesized for existing MP4-only evaluations.

To regenerate plots or apply a separately fitted threshold (no GPU required):

```bash
bash scripts/eval_libero_vfd.sh plot \
  --run-dir outputs/libero_vfd/spatial_test \
  --calibration outputs/libero_vfd/spatial_threshold.json --normalize
```

Plots show a single rollout's score, threshold, first crossing, and camera frames.
`summary.json` reports per-task TPR, TNR, balanced accuracy, AUROC and TWA when
both outcome classes exist. Single-class metrics are null, not misleading zeros.

Useful flags: `--renderer osmesa`, `--num-action-samples 8`,
`--flow-times 0.5,0.75,0.875` with 8 denoising steps, `--snapshot-stride 20`,
`--verify-actor`, and `--dry-run`. Keep score settings fixed between calibration and test.
`--verify-actor` checks the first monitored action chunk against original actor
inference with the same random state and records the maximum absolute difference.
`--gpu-memory-limit-gb` caps the PyTorch allocator only; it does not reserve VRAM
or cover all CUDA/OpenGL allocations. Prefer a separate GPU allocation for real experiments.
The wrapper respects `CUDA_VISIBLE_DEVICES`, `GR00T_PYTHON`, and `LIBERO_REPO`.

```bash
PYTHONPATH=. /home/vhoangth2/miniconda3/envs/gr00t/bin/python -m pytest -q tests/test_libero_vfd.py
```
