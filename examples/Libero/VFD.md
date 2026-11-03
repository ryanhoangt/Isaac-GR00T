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

To prepare a second independent Spatial member, use
`bash scripts/submit_libero_loso.sh --seed 43 --dry-run spatial` to review the
submission, then remove `--dry-run` to submit. The default seed remains 42.
Pair equal-epoch checkpoints from the two runs, and recalibrate thresholds for
the new pair; thresholds from the snapshot pair are not interchangeable.

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

## Offline calibration from existing demonstrations

`score-dataset` reads existing LeRobot parquet/video files. It never creates an
environment, executes robot actions, or imports LIBERO/MuJoCo/OpenGL. GPU inference
is still needed for both checkpoints. Images are already in training orientation;
state and images go through the actor's checkpoint-owned evaluation transforms,
without recomputing normalization from the calibration dataset. Demo actions are
not inputs to VFD: the actor samples its own flow at each recorded observation.

The examples below use 30 assumed-successful demos for each calibration method,
selected per task with a fixed seed independently of their scores. Run from the
repo root with the same `ACTOR` and `SCORER` paths defined above:

```bash
# Source-demo: one demo per task from the three source suites (30 episodes).
bash scripts/eval_libero_vfd.sh score-dataset \
  --actor "$ACTOR" --scorer "$SCORER" \
  --suites object goal long --episodes-per-task 1 \
  --assume-successful-demos --gpu-memory-limit-gb 18 \
  --output-dir outputs/libero_vfd/source_demo_calibration

# Target-demo: three demos per Spatial task (30 episodes).
bash scripts/eval_libero_vfd.sh score-dataset \
  --actor "$ACTOR" --scorer "$SCORER" \
  --suites spatial --episodes-per-task 3 \
  --assume-successful-demos --gpu-memory-limit-gb 18 \
  --output-dir outputs/libero_vfd/target_demo_calibration

# CPU only: apply either fitted threshold to an existing policy recording.
bash scripts/eval_libero_vfd.sh plot \
  --run-dir outputs/libero_vfd/spatial_5trials_liveplots_20261024T142428Z \
  --calibration outputs/libero_vfd/source_demo_calibration/threshold.json
```

Each output contains `manifest.json`, incremental `summary.json`, per-demo
JSON/NPZ/JSONL scores, and (on full completion) `threshold.json`. The default fit
uses a global threshold, quantile 0.9, at least 20 successful demos, and every
frame's score. `--scope task` requires enough demos per task. `--score-only` saves
scores for a later `calibrate` call. `--dry-run` prints selected episode IDs and
frame counts without loading checkpoints or decoding videos. `--max-frames 3`
is a truncated GPU smoke test and cannot produce a threshold. `--episode-start`
is an offset in each task's shuffled list; it is not an official initial-state ID.
`--episodes-per-task 0` selects all remaining demos. The dataset root defaults to
`demo_data/libero_loso_clean`; no training data is modified.
Add `--verify-actor` to verify that the first monitored action chunk matches
ordinary actor inference under identical noise, without executing either action.

`--assume-successful-demos` explicitly declares that these input episodes are
successful expert demonstrations; their success is not measured by this script.
LIBERO task IDs are matched by instruction because dataset-local task IDs differ.
The literal task map is read from `LIBERO_REPO` (or `--task-map`) without importing
LIBERO. The current evaluator uses LIBERO's default task order.

Report this as **demo-calibrated VFD**, distinct from calibration on successful
policy rollouts. Expert states and episode lengths can differ from policy states
and horizons, so false-positive rates need not transfer. Source demos may also
have been used to train these checkpoints; they are not an independent policy
calibration set. Target demos use held-out-suite data. Dataset episode IDs do not
identify official LIBERO initial states: overlap with the 50 policy test rollouts
is **unverified**, and both manifest and threshold retain that limitation. The
code rejects repeated demo identities and mixed demo/policy calibration, but it
does not claim to prove demo/test initial-state disjointness. Existing MP4-only
policy evaluations cannot replace these data because they lack robot state.

Plotting twice with different thresholds overwrites the same plot files and
`summary.json`; the raw recorded uncertainty is unchanged. Copy the output plots
and summary elsewhere if you want to retain both comparisons.

## Interactive uncertainty viewer

The Streamlit viewer compares recorded epoch/seed pairs on the same suite, task,
and official initial state. It reads saved scores and videos on CPU; no model,
GPU, simulator, or new rollout is needed. Setup uses a separate environment:

```bash
uv venv --python python3.10 .venv-vfd-viewer
uv pip install --python .venv-vfd-viewer/bin/python -r requirements/vfd_viewer.txt
bash scripts/view_libero_vfd.sh
```

Open `http://localhost:8501`. For a remote server, forward that port from your
laptop, e.g. `ssh -N -L 8501:127.0.0.1:8501 USER@SERVER`. Use a second port with
`bash scripts/view_libero_vfd.sh --server.port 8502` if needed. Choose a different
recording directory with `bash scripts/view_libero_vfd.sh -- --root /path/to/recordings`.
`VFD_VIEWER_PYTHON` can point to another environment with the viewer requirements.

The sidebar discovers recordings and source/target calibrations by exact score
signature; a different pair's threshold cannot silently be used. Select one or
two pairs, then filter suite/task/outcome and select a rollout. Videos and chart
cursors share one step slider. Click/drag a chart, use frame buttons or arrow
keys, play slowly, zoom around the selected frame, or jump to peaks and first
threshold crossings. Both thresholds and first-crossing markers are available
for each pair. The raw score remains visible; the optional trailing five-step
mean is only a display aid and never changes alarms.

Scores at step t are aligned to MP4 presentation timestamp t, which shows the
observation **before** action t executes. The consequence of that action usually
appears in frame t+1. Video time is playback time, not measured inference wall
time. The viewer checks video frame counts against scores; if exact video is
missing/misaligned, it explicitly shows the nearest saved snapshot instead.
For two recordings, exact equality of saved executed actions is checked rather
than assuming their trajectories match merely because their initial IDs match.

Mark critical steps with notes (grasp/contact/slip/release, etc.) and download the
notes before switching episodes: notes live only in the current open view. The
export section also downloads a standalone interactive HTML with embedded videos
and a CSV of raw scores, thresholds, and alarm flags. The HTML works offline and
has no CDN dependencies. Existing rollout files and calibration outputs are never
modified. Observed outcome labels and first alarms are not annotated failure-onset
ground truth.

Viewer data/AppTest checks: `.venv-vfd-viewer/bin/python -m pytest -q tests/test_libero_vfd_viewer.py`
(install `pytest` in that environment). The opt-in browser test additionally needs
`playwright` and its Chromium installation; set `VFD_VIEWER_BROWSER_TESTS=1` and
`PLAYWRIGHT_BROWSERS_PATH` to your browser cache. It checks both the standalone
export and a temporary live Streamlit instance, including task switching.

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
