# Offline DTW positive pairs for LIBERO and RoboMME

For online retriever training with this sampler, see [TRAIN_RETRIEVER.md](TRAIN_RETRIEVER.md).

The exporter calls the same
`gr00t.rag.utils.get_aligned_frame_ids()` function as online training. It selects two different
demonstrations of the same task, obtains action-DTW frame matches and exports
the sampled frame IDs. The original training sampler does this during batch
construction; this wrapper makes a finite sample available for inspection.

The default protocol is **Goal/Object/Spatial as ID; Long (`libero_10`) held
out**. Long is not read. No retriever/policy training or rollout is performed.
Both export and visualization run on CPU. RoboMME uses an explicitly selected
dataset instead of the LIBERO suite protocol; see below.

## RoboMME v2.1

Reuse the same exporter and viewer. No new dataset is created, no HDF5 is read,
and no model checkpoint is needed. The exported pairs are for inspection;
retriever training continues to compute DTW online.

```bash
# Run CPU work inside an allocation. Replace the job ID if it has ended.
# Two eligible (canonical task, instruction) groups, one demo pair per group.
srun --jobid=66728 --overlap --nodes=1 --ntasks=1 --cpus-per-task=2 \
  env CUDA_VISIBLE_DEVICES= USE_TF=0 USE_FLAX=0 HF_HUB_OFFLINE=1 \
  bash scripts/mine_dtw_pairs.sh \
    --dataset-kind robomme \
    --dataset-path /mnt/data/vhoangth2/datasets/robomme_primary_train_v21 \
    --tasks-per-suite 2 --pairs-per-task 1 --num-frames 20 --seed 42 \
    --output-dir outputs/dtw_pairs/robomme_primary_smoke

# Preview all eligible groups from primary without reading videos or mining.
srun --jobid=66728 --overlap --nodes=1 --ntasks=1 --cpus-per-task=2 \
  env CUDA_VISIBLE_DEVICES= USE_TF=0 USE_FLAX=0 HF_HUB_OFFLINE=1 \
  bash scripts/mine_dtw_pairs.sh \
    --dataset-kind robomme \
    --dataset-path /mnt/data/vhoangth2/datasets/robomme_primary_train_v21 \
    --pairs-per-task 10 --num-frames 20 --dry-run

# Launch the viewer on a compute node, then forward its port.
srun --jobid=66728 --overlap --nodes=1 --ntasks=1 --cpus-per-task=2 \
  bash scripts/visualize_dtw_pairs.sh --server.port 8502 -- \
    --pairs-dir outputs/dtw_pairs/robomme_primary_smoke
```

`--dataset-path` points to one dataset; it can be primary or any LOSO subset.
`--tasks VideoUnmask VideoUnmaskSwap` optionally restricts canonical task names.
`--task-ids` optionally restricts LeRobot instruction IDs, which are different
from canonical task IDs. For RoboMME, the legacy option `--tasks-per-suite`
limits **eligible compound groups**, and `--pairs-per-task` means demo pairs
per compound group. Omit the limit or use zero to include all eligible groups.

The exporter reuses the online sampler's **(canonical task, exact instruction)**
groups, including short-demo and singleton exclusions. With primary and 20
frames, 81 of 99 groups are eligible; 18 singleton groups are skipped. All 1,200
source demos remain intact. A sampled pair never crosses canonical task families,
even when they share an instruction. Metadata retains the complete eligibility
report plus skipped groups matching any task/instruction filters. This report
describes dataset eligibility; the plan describes the smaller selected export.

The subset's `meta/episode_tasks.json` is loaded automatically and validated
against its metadata. For all16, add
`--task-map examples/RoboMME/metadata/robomme_task_index_v21.json`.
The export fingerprints the map, modality metadata and grouping implementation;
changed inputs require a fresh output directory. Resume uses the same command.
No held-out split is inferred from a dataset name; select the desired training
subset explicitly.

RoboMME DTW uses **raw 8D absolute joint/gripper actions** from `actions`.
Videos use `image`/`wrist_image`. The viewer separates canonical task and
instruction, plots all eight action dimensions, and compares seven-joint state
RMSE in radians. It excludes the gripper from that state metric. Raw action RMSE
includes the gripper and thus mixes coordinates. RoboMME joint state is not an
end-effector XYZ position and is not displayed as a distance in cm.

## Run from the repo

The existing `gr00t` environment already has the export dependencies
(`dtaidistance`, NumPy, pandas, PyArrow and torch). The default dataset root is
`ra-vla-official-assets/ravla_code/ravla_code_no_model/datasets`.

```bash
# Small test: one task and one demonstration pair in each G/O/S suite.
bash scripts/mine_dtw_pairs.sh \
  --tasks-per-suite 1 --pairs-per-task 1 --num-frames 20 \
  --output-dir outputs/dtw_pairs/smoke_20261122

# Inspect the planned sample without mining or writing output.
bash scripts/mine_dtw_pairs.sh --dry-run --pairs-per-task 10 --num-frames 20

# All 30 ID tasks, 10 different demo pairs/task, 20 frame pairs/demo pair.
# This gives 300 demonstration-pair records / 6,000 positive candidates.
bash scripts/mine_dtw_pairs.sh \
  --pairs-per-task 10 --num-frames 20 --seed 42 \
  --output-dir outputs/dtw_pairs/gos_offline
```

“All tasks” does not mean all possible demonstration combinations: the
`--pairs-per-task` budget controls that sample. It is capped at the available
unique unordered demo pairs. Task selection and pair selection are reproducible.
`--tasks-per-suite 0` (the default) selects all tasks. Tasks with fewer than two
demos of at least `--num-frames` frames are skipped and recorded in metadata.

To select tasks/suites or a different data location:

```bash
bash scripts/mine_dtw_pairs.sh \
  --suites libero_goal --task-ids 0 1 --pairs-per-task 5 --num-frames 20 \
  --dataset-root ra-vla-official-assets/ravla_code/ravla_code_no_model/datasets \
  --output-dir outputs/dtw_pairs/goal_tasks_0_1
```

Task IDs are zero-based and apply to each selected suite. Supported suites for
this protocol are `libero_goal`, `libero_object`, `libero_spatial`.
Override the Python interpreter with `POLICY_PYTHON=/path/to/python`.

## Output and resume

- `metadata.json`: settings, seed, selected demo pairs, source fingerprints,
  sampler source hash, completion status and counts.
- `pairs.jsonl`: one record per demonstration pair, containing the sampled
  positive frame pairs, a relative-progress baseline and source file paths.
- `.precompute.lock`: the shared run-lock helper prevents concurrent writers.

Example record (shortened):

```json
{
  "suite": "libero_goal",
  "task_id": 0,
  "task": "put the bowl on the plate",
  "episode_a": 0,
  "episode_b": 43,
  "frame_pairs": [[3, 2], [4, 3]],
  "progress_frame_pairs": [[3, 4], [4, 5]]
}
```

Actual records also include a deterministic per-record seed, dataset path,
Parquet/video fingerprints, camera paths and lengths/FPS. Images and model
embeddings are not stored in JSON. `frame_pairs` come directly from the original
function, which uses **raw 7D LIBERO actions** (8D for RoboMME), unconstrained DTW, the first target
match for each source frame, then random sampling while preserving order.
The relative-progress matches use the same source frames and rounded normalized
trajectory progress; they are only a comparison baseline.

Rerun **the same command** to resume. Completed records are validated and
skipped; an unfinished final JSONL line is discarded and that record recomputed.
Changing inputs/settings requires a new output directory. Per-record seeds
make results independent of interruption/resume. They do not reproduce the
exact batch RNG sequence of a historical training run.

## Visualize

The existing `.venv-visualizer` environment is ready in this checkout. For setup
on another checkout, follow [RETRIEVAL_VIEWER.md](RETRIEVAL_VIEWER.md).

```bash
# Inspect the small test that has already been run.
bash scripts/visualize_dtw_pairs.sh --server.port 8502 -- \
  --pairs-dir outputs/dtw_pairs/smoke_20261122

# Inspect your full export.
bash scripts/visualize_dtw_pairs.sh --server.port 8502 -- \
  --pairs-dir outputs/dtw_pairs/gos_offline
```

Open <http://localhost:8502>. If running remotely, forward **8502 from the host
running Streamlit** in VS Code's Ports panel or via SSH:

```bash
ssh -N -L 8502:127.0.0.1:8502 vhoangth2@HOST_RUNNING_THE_APP
```

The sidebar selects suite, task, demo pair and external/wrist/both cameras.
Move the positive-pair slider or use Previous/Next. Three columns show the
anchor observation, its DTW positive, and the relative-progress match.
Additional panels show action RMSE, end-effector position distance, sampled
frame correspondences and raw action chunks. The chunk length only changes
what is plotted; it does not rerun mining. Tail chunks are shortened to the
common number of available steps without padding. You can download the selected
demo-pair record as JSON. The app also reads completed records during export.
End-effector distance applies to LIBERO; RoboMME displays joint-state RMSE.

These are **positive candidates**, not ground-truth semantic matches. DTW
optimizes action similarity, so low action error alone does not establish pair
quality. End-effector position ignores object state. Check the images and
action traces before treating candidates as good supervision. The plots show
sampled correspondences, not the complete warping path. Inspecting these pairs
does not measure retrieval accuracy or Long task success rate.

## Verification

```bash
.venv-visualizer/bin/python -m pytest -q tests/test_dtw_pairs.py
```

Tests compare exported frame pairs against a direct call to the original
function, check RNG preservation/resume/provenance, and exercise Streamlit
frame, camera, task and chunk-length navigation for both schemas. RoboMME tests
also cover shared-instruction separation, singleton/short-demo exclusions,
portable-map provenance and the eighth action dimension. Run tests inside an
allocation. Separate smoke checks use real datasets without a GPU.

RoboMME CPU validation in allocation 66728 used primary and all16 metadata
(81 and 107 eligible groups respectively). A shared instruction produced one
demo-pair record per VideoUnmask/VideoUnmaskSwap family, 40 frame pairs total;
resume preserved the JSONL bytes. Streamlit checks exercised real videos,
navigation, wrist view, the eighth action dimension and joint-state metrics.
The local artifacts are in `outputs/dtw_pairs/robomme_step3_validation_20261122/`
(`validation.json`, `metadata.json`, `pairs.jsonl`). These checks establish
export/viewer compatibility, not the semantic quality of DTW matches.
