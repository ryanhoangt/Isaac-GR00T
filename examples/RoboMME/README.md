# RoboMME data preparation for GR00T N1.5

The primary converter now accepts a **local RoboMME LeRobot v2.x directory**,
not an HDF5 file or a Hub repo ID. It does not download anything. The earlier
single-episode HDF5 converter is also retained below as an optional alternative.
A RoboMME training data config, training launcher and evaluation adapter remain
separate next steps and have not been implemented yet.

## Convert a local LeRobot v2.1 dataset (all episodes)

Use a local download of
[`Yinpei/robomme_data_lerobot`](https://huggingface.co/datasets/Yinpei/robomme_data_lerobot),
with this structure:

```text
/path/to/local/robomme_data_lerobot/
  meta/info.json
  meta/episodes.jsonl
  meta/tasks.jsonl
  data/chunk-000/episode_000000.parquet
  ...
```

The converter reads the source episode list, task vocabulary, image features,
chunk layout and FPS. The published v2.1 metadata reports 1,600 episodes,
116 instructions, 768,897 frames and 10 FPS. The `image` / `wrist_image`
columns contain images embedded in parquet. The other published dataset,
[`lerobot/robomme`](https://huggingface.co/datasets/lerobot/robomme), uses v3.0
shared video files and different episode metadata; this script rejects v3 input.
Both datasets' metadata was inspected when choosing the conversion layout.

From this repository, **on the compute node inside your allocation**, use the
existing GR00T environment. For 8 CPUs and 64 GB RAM:

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  /home/vhoangth2/miniconda3/envs/gr00t/bin/python -u scripts/convert_robomme_lerobot.py \
    --source /path/to/local/robomme_data_lerobot \
    --output demo_data/robomme_gr00t_v21 \
    --workers 4 \
    --resume
```

Replace `--source` with your local directory. All episodes are selected by
default; no GPU is required. Pillow decodes local image bytes (or local image
paths) and PyAV encodes the two videos. No simulator or upstream LeRobot
installation is needed. Each worker reads one episode parquet at a time and
uses a single encoder thread per camera; images are decoded one frame at a
time. The final statistics pass retains only state/action vectors, not images.

To check a subset first, use a different output directory:

```bash
python scripts/convert_robomme_lerobot.py \
    --source /path/to/local/robomme_data_lerobot \
    --output /tmp/robomme_gr00t_ep0 \
    --episode-indices 0 \
    --workers 1
```

The output is one video-based v2.1 dataset: per-episode parquet, per-camera
MP4s, `info.json`, `episodes.jsonl`, `tasks.jsonl`, `episodes_stats.jsonl`,
`stats.json`, GR00T `modality.json`, `episode_mapping.json` and a completion
manifest. Input data and the current GR00T loader are unchanged.

Local v2.1 conversion conventions:

- Raw names `state`, `actions`, `image` and `wrist_image` are retained. The
  modality mapping exposes `state.joint_position` / `state.gripper`,
  `action.joint_position` / `action.gripper` and `video.image` /
  `video.wrist_image` to GR00T. Language maps through `task_index`.
- Only execution rows (`is_demo=False`) are exported. Demo flags must form a
  prefix and agree with `exec_start_idx`. The output therefore has fewer than
  the published 768,897 frames. The existing local MME-VLA preprocessing
  reports 476,857 execution samples, but that count is not hard-coded.
- State/action values remain paired at the same source row, with absolute
  joint targets and -1/+1 gripper commands. Subgoal annotation columns and
  task text are preserved; the baseline language mapping uses only task text.
- Episode IDs, task IDs and global frame indices are compact and deterministic.
  Local frame indices and timestamps restart at zero; `exec_start_idx` becomes
  zero. `step_idx` retains its original values. FPS is read from metadata,
  without an override or temporal resampling.
- Original `source_episode_index`, `source_frame_index`, `source_index`,
  `source_timestamp`, `source_exec_start_idx` and `source_task_index` preserve
  row provenance. Demo images remain in the source dataset.
- Per-episode and global state/action statistics, including quantiles, are
  recomputed from selected execution rows. No validation split is created.
- H.264 CRF 18, `yuv420p` video encoding is lossy; images are not resized.

Existing final output is never overwritten. Completed episode encodes are
cached in `.<output-name>.work/` next to the output. Repeat the same command
with `--resume` after interruption. Source metadata and selected parquet
size/mtime must match the saved configuration; worker count can change. A lock
prevents simultaneous writers to one cache. The final dataset appears only
after all episodes and metadata are complete. The cache remains after success;
output videos are hard links to cached videos, so video bytes are not duplicated.

Run the local dataset converter tests with:

```bash
python -m pytest -q tests/test_robomme_lerobot_conversion.py
```

Tests use local fixtures with actual PNG bytes inside parquet. They cover
parallel conversion, chunk boundaries, pairing, annotation retention, remapping,
statistics, decoded video order, GR00T loading/padding, resume without re-encoding,
changed-source rejection, invalid demo flags, corrupt images and v3 rejection.
The downloaded public dataset has not yet been converted in full by the assistant.

## Alternative: convert one HDF5 episode

Use the GR00T Python environment, which provides `h5py`, `numpy`, `pandas`,
`pyarrow` and PyAV (`av`). The converter does not need the simulator, the
policy-learning package, a model checkpoint, or the upstream LeRobot package.

From the repository root:

```bash
python scripts/convert_robomme_hdf5.py \
    --source /mnt/data/vhoangth2/repos/robomme_policy_learning/data/robomme_data_h5/record_dataset_InsertPeg.h5 \
    --episode-index 0 \
    --fps 10 \
    --output /tmp/robomme_insertpeg_ep0
```

The output directory must not already exist. The raw HDF5 file is opened
read-only. Images are read one frame at a time, with only the episode's small
state/action arrays retained for parquet and exact normalization statistics.
Temporary output is published only when conversion completes; a failure cleans
up that invocation's temporary files.

The export contains:

```text
data/chunk-000/episode_000000.parquet
videos/chunk-000/observation.images.image/episode_000000.mp4
videos/chunk-000/observation.images.wrist_image/episode_000000.mp4
meta/info.json
meta/episodes.jsonl
meta/tasks.jsonl
meta/modality.json
meta/stats.json
meta/episode_mapping.json
manifest.json
```

- Output episode ID is always 0; `--episode-index` selects the source episode.
- Only execution frames are exported. Demo flags must form a contiguous prefix.
- Observation and action stay at the **same source timestep**, following the
  policy-learning builder. Joint targets remain absolute; no delta transform
  or action chunks are baked into parquet.
- State uses 7 joint positions and the first gripper finger position. The
  action's last value remains -1 (close) or +1 (open).
- The first source task instruction, lowercased like the reference builder,
  becomes task 0. All original instructions are retained in the mapping.
- `source_timestep` in parquet and `meta/episode_mapping.json` preserve source
  provenance and the original demo boundary. Demo video itself is not copied.
- Statistics are computed only from exported execution states/actions. Every
  exported frame, including a recorded terminal frame, is retained.
- Two H.264 videos use CRF 18 and `yuv420p`; RGB compression is lossy. No spatial
  resizing is performed.

`--fps` is required: timestamps restart at zero and advance by `1 / fps`; one
source timestep becomes one video frame. **There is no temporal resampling.**
The example uses 10 FPS, the convention advertised by the published RoboMME
LeRobot datasets. This does not prove the physical control frequency of the
raw HDF5 recording or reproduce any upstream resampling. Record the selected
FPS when comparing experiments.

The output can be inspected with `LeRobotSingleDataset` using video keys
`video.image` / `video.wrist_image`, state/action keys `joint_position` /
`gripper`, language key `annotation.human.action.task_description` and embodiment
`new_embodiment`. A train-ready `RoboMMEDataConfig` is a separate next step.

## Local data inspected

Assessment date: 2026-11-20.

Raw HDF5 directory on this machine:

```text
/mnt/data/vhoangth2/repos/robomme_policy_learning/data/robomme_data_h5
```

The directory contains 16 `record_dataset_<task>.h5` files, each with 100 episode
groups: 1,600 episodes in total. Their combined file size is 512,595,968,744 bytes
(512.60 GB / 477.39 GiB). The existing `robomme_preprocessed_data` directory
contains the policy-learning repo's custom pickle data and cached features;
it is not a LeRobot dataset that GR00T can load directly.

The dataset README advertises `Yinpei/robomme_data_lerobot`, but no local copy was
identified in the policy-learning data directory, the inspected Hugging Face
cache locations, or the top-level shared datasets directory. Subsequent remote
metadata inspection found two published variants:

- [`Yinpei/robomme_data_lerobot`](https://huggingface.co/datasets/Yinpei/robomme_data_lerobot/blob/main/meta/info.json):
  v2.1 with images stored in parquet, rather than separate video files.
- [`lerobot/robomme`](https://huggingface.co/datasets/lerobot/robomme): v3.0 with
  camera videos and shared-file episode indexing.

Both advertise 1,600 episodes, 768,897 frames and 10 FPS. Neither is directly
loadable by this repo's existing video-based v2 loader. Their metadata was
inspected, but their training data was not downloaded or compared frame by
frame with the local HDF5 files. The converter uses the available local HDF5.

The following inventory comes from `episode_0` of each HDF5 file. These lengths
and demo boundaries are examples, not assumptions for every episode in a task.
Timestep indices start at zero.

| Task | Episodes in file | Steps in episode 0 | First execution timestep |
| --- | ---: | ---: | ---: |
| BinFill | 100 | 550 | 0 |
| ButtonUnmask | 100 | 291 | 0 |
| ButtonUnmaskSwap | 100 | 514 | 0 |
| InsertPeg | 100 | 475 | 238 |
| MoveCube | 100 | 425 | 234 |
| PatternLock | 100 | 96 | 48 |
| PickHighlight | 100 | 308 | 0 |
| PickXtimes | 100 | 529 | 0 |
| RouteStick | 100 | 200 | 100 |
| StopCube | 100 | 459 | 0 |
| SwingXtimes | 100 | 482 | 0 |
| VideoPlaceButton | 100 | 1048 | 765 |
| VideoPlaceOrder | 100 | 1200 | 940 |
| VideoRepick | 100 | 756 | 321 |
| VideoUnmask | 100 | 239 | 66 |
| VideoUnmaskSwap | 100 | 326 | 168 |

## Observed schema and proposed baseline mapping

The baseline would consume the current front/wrist images and robot state,
condition on the task instruction, and predict absolute joint-space actions.
It would not consume history or the conditioning video.

| Source HDF5 field | Observed shape / meaning | Proposed GR00T use |
| --- | --- | --- |
| `setup/task_goal` | List of language instructions | Task text, indexed by `task_index` |
| `timestep_<t>/obs/front_rgb` | `uint8 (256, 256, 3)` | `video.image` |
| `timestep_<t>/obs/wrist_rgb` | `uint8 (256, 256, 3)` | `video.wrist_image` |
| `timestep_<t>/obs/joint_state` | 7 absolute joint positions | `state.joint_position`, dimensions 0:7 |
| `timestep_<t>/obs/gripper_state` | 2 finger positions | First finger as `state.gripper`, dimension 7:8 |
| `timestep_<t>/action/joint_action` | 7 absolute joint targets + gripper | `action.joint_position` (0:7) and `action.gripper` (7:8) |
| `timestep_<t>/info/is_video_demo` | Conditioning-video flag | Exclude demo frames as supervised action examples |

The proposed 8-dimensional state follows `pack_state()` in the existing RoboMME
evaluator. Gripper state is a physical opening (documented range 0 to 0.04),
whereas the gripper action is a command: -1 closes and +1 opens. They must not
share normalization statistics. The stick tasks have a fixed gripper slot.

Demo prefixes also occur in tasks without `Video` in their name. Conversion
must inspect `is_video_demo` for each episode rather than infer the boundary
from the task name. Preserve the source task, source episode, original timestep,
and demo boundary so a later memory implementation can recover the demo.

## Compatibility with the current GR00T loader

`gr00t/data/dataset.py` reads episode parquet files and camera videos, with:

- `meta/info.json`: features, FPS, chunk size and parquet/video path patterns.
- `meta/episodes.jsonl`: episode IDs and lengths.
- `meta/tasks.jsonl`: task indices and language instructions.
- `meta/modality.json`: GR00T mappings for state, action, cameras and language.
- `meta/stats.json`: normalization statistics (the loader can also compute
  missing statistics from parquet).

The proposed baseline fits the existing loader/model interface. A dedicated
RoboMME data config and compatible dataset export are still required. The
existing LIBERO config describes a different state/action representation and
cannot be reused unchanged.

The single-episode converter uses the available raw files without changing the
core GR00T loader. It does not implement bulk conversion, automatic train/val
splits, dataset merging or model training.

## Training conventions and limitations

- **Observation/action alignment:** the current benchmark `RecordWrapper.step()`
  calls `super().step(action)` before recording images and joint state alongside
  that action. The current MME-VLA builder instead uses observation and action
  from the same HDF5 timestep. This is a possible post-action observation issue;
  the checked-out recorder alone does not establish how these released HDF5
  files were generated. The converter deliberately follows the reference
  builder's same-row pairing to preserve comparability, without claiming to
  resolve that recording-provenance question. The reference trainer additionally
  converts joint targets to deltas relative to the current state; that is an
  action representation transform, not a temporal shift, and is not applied by
  this converter.
- **Timing:** the inspected files have no root attributes specifying FPS. Do
  not infer controller frequency from a visualization video's playback FPS.
  This converter requires an explicit export FPS as described above.
- **Training subset:** the GR00T loader enumerates all episodes listed in
  `episodes.jsonl`; it does not filter training samples using `info.json`'s
  `splits` field. Any validation subset must be excluded from the training
  episode list and training normalization statistics.
- **Tail padding:** parquet stores one absolute action per frame. The current
  GR00T loader repeats the final action for out-of-range future indices when
  the modality has `absolute=true`, as emitted here. This matches the reference
  builder's final-action padding and does not cross the episode boundary.

## Scope of verification

Opened all 16 HDF5 files read-only and counted their episode groups. Inspected
the first, first-execution and final timestep of `episode_0` in each file for
camera shape/dtype, state/action shape, demo/completion flags, and finite joint
state/action values. Read actual image arrays from selected timesteps of
BinFill, InsertPeg and VideoPlaceOrder successfully.

This initial assessment was a schema/sample inspection, not a full integrity
scan of 1,600 episodes. It did not modify the source data or train a model.
Converter tests use synthetic HDF5 to verify same-row pairing, demo exclusion,
execution-only statistics, decoded video frame order, GR00T loading and tail
padding, rejection of invalid input, and protection of existing output.

Run the converter tests with:

```bash
python -m pytest -q tests/test_robomme_conversion.py
```

The four tests passed in the existing GR00T environment. A real-data check also
converted InsertPeg `episode_0`: 238 demo frames were excluded and 237 execution
frames (source timesteps 238 through 474) were exported. All exported states and
actions matched their source HDF5 rows exactly; both videos decoded all 237
frames. GR00T loaded the first, middle and final samples, including 16-step
action chunks with final-action padding. The real-data output was written only
under `/tmp`; it is not part of the repository.

Source code reviewed in the sibling checkouts:

- `robomme_benchmark/doc/h5_data_format.md` and `doc/env_format.md`.
- `robomme_benchmark/src/robomme/env_record_wrapper/RecordWrapper.py`.
- `robomme_policy_learning/src/mme_vla_suite/dataset_builder/build_robomme_dataset.py`
  and `robomme_h5_utils.py`.
- `robomme_policy_learning/examples/robomme/env_runner.py`.

The standalone benchmark and policy-learning benchmark submodule both reported
revision `985caf6` during inspection.
