# RoboMME data preparation for GR00T N1.5

The converter accepts a **local RoboMME LeRobot v2.x directory**, not a Hub
repo ID, and does not download anything. `RoboMMEDataConfig` now connects the
converted dataset to GR00T transforms and the existing collator. Fine-tuning
uses `scripts/gr00t_finetune.py` directly. The 10-step GPU smoke result is
recorded below; full training and an evaluation adapter remain separate steps.

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

## RoboMME data configuration (step 2)

Load the config through the existing external-config interface:

```python
from gr00t.experiment.data_config import load_data_config

config = load_data_config("examples.RoboMME.custom_data_config:RoboMMEDataConfig")
modalities = config.modality_config()
transforms = config.transform()
```

The corresponding fine-tuning argument is
`--data-config examples.RoboMME.custom_data_config:RoboMMEDataConfig`, with
`--embodiment-tag new_embodiment`. This step does not add a training launcher.

The configuration defines:

- Current front and wrist RGB images, in that order; no history or demo frames.
- Seven joint positions plus the first finger's physical opening as state.
- Sixteen consecutive absolute actions: seven joint targets plus a -1/+1
  gripper command. No temporal shift or conversion to relative joint targets.
- The task instruction as the only language input. Stored subgoal annotations
  are not read by the config.
- Per-field min-max normalization from the converted dataset's statistics.
  State/action and joints/gripper use separate statistics; the physical state
  gripper range and action command range are never interchanged. A joint's
  normalized target is `2 * (target - min) / (max - min) - 1`.
- The same image recipe as the LIBERO example: a 0.95 crop and color jitter
  during training, resize to 224 by 224, with deterministic preprocessing in
  eval mode.
- State padding from 8 to 64 dimensions and action padding from 8 to 32, with
  masks identifying only the eight real dimensions. Action inverse transforms
  recover physical absolute joint targets and gripper commands.

After dataset metadata is set, training mode produces a padded state of shape
`(1, 64)` and action chunk `(16, 32)`. `DefaultDataCollator` adds the batch
axis and processes the two images plus instruction with the bundled Eagle
processor. Inference uses `transforms.eval()` and accepts observations without
any action labels.

The exported dataset remains in absolute joint space. Unlike the reference
MME-VLA trainer's delta-joint transform, this initial GR00T recipe learns
normalized absolute targets. This is an explicit baseline choice; changing it
later requires consistent inverse transforms and statistics.

Use `LeRobotSingleDataset` with the config's modalities/transforms and
`embodiment_tag="new_embodiment"`. The CPU checks use `video_backend="decord"`
and `video_backend_kwargs={"num_threads": 1}` to bound decoder CPU use. The
validated local dataset is:

```text
/mnt/data/vhoangth2/datasets/robomme_gr00t_lerobot_v21
```

It has 1,600 episodes and 476,857 execution frames. A separate data audit matched
all exported state/action rows to the local source, recomputed the global
statistics, checked all 3,200 video headers and decoded 54 loader samples.

Run the config tests **inside a CPU allocation**, not on the login node:

```bash
env USE_TF=0 USE_FLAX=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    NO_ALBUMENTATIONS_UPDATE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    python -m pytest -q tests/test_robomme_data_config.py
```

These checks use the bundled Eagle processor and real collator on CPU; no model
weights, model forward/backward pass or GPU are needed. They verify absolute
normalization targets, independent gripper scales, dimension masks, action
round-trip, inference without labels, camera/language order and finite collated
outputs. The GPU smoke result is recorded below.

## Extract task subsets for the five held-out checkpoints

`scripts/extract_robomme_tasks.py` copies selected task families from the converted
video dataset into a standalone v2.1 dataset. It supports the proposal's primary
split, four leave-one-suite-out (LOSO) splits, or explicit `--tasks` /
`--exclude-tasks` selections. Each preset below is a **training set**; the named
held-out tasks are excluded from its files and normalization statistics.

| Preset | Held-out tasks | Train episodes | Execution frames |
| --- | --- | ---: | ---: |
| `primary` | PickXtimes, ButtonUnmaskSwap, PickHighlight, PatternLock | 1,200 | 338,246 |
| `loso_counting` | BinFill, PickXtimes, SwingXtimes, StopCube | 1,200 | 287,822 |
| `loso_permanence` | VideoUnmask, VideoUnmaskSwap, ButtonUnmask, ButtonUnmaskSwap | 1,200 | 375,791 |
| `loso_reference` | PickHighlight, VideoRepick, VideoPlaceButton, VideoPlaceOrder | 1,200 | 360,139 |
| `loso_imitation` | MoveCube, InsertPeg, PatternLock, RouteStick | 1,200 | 406,819 |

These counts were checked by indexing all 1,600 local episodes and dry-running
all five selections in CPU allocation 66463. Each contains 12 task families and
2,400 videos. The actual copy path was tested with small local datasets and the
GR00T loader; the five full datasets have not been copied by the assistant.

Language `task_index` identifies an instruction, not one of the 16 environments.
The `index` command matches instructions to original HDF5 task metadata. For
shared instructions (VideoUnmask and VideoUnmaskSwap), it also matches episode
lengths, demo boundaries and execution state/action samples. Ambiguous matches
are rejected. The raw HDF5 files are needed only for indexing; no images are
decoded during this step. The generated map is bound to source metadata and
records a complete state/action hash for each episode, verified during copying.

Run from this repository **inside a CPU allocation**, using the GR00T environment.
For example, start a shell in the current allocation with
`srun --jobid=66463 --overlap --ntasks=1 --cpus-per-task=8 --mem=64G --pty bash`.
Then set these paths (the dataset and index destinations are customizable):

```bash
cd /mnt/data/vhoangth2/repos/Isaac-GR00T
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ROBOMME_PYTHON=/home/vhoangth2/miniconda3/envs/gr00t/bin/python
ROBOMME_SOURCE=/mnt/data/vhoangth2/datasets/robomme_gr00t_lerobot_v21
ROBOMME_INDEX=/mnt/data/vhoangth2/datasets/robomme_episode_tasks_v21.json

"$ROBOMME_PYTHON" -u scripts/extract_robomme_tasks.py index \
    --source "$ROBOMME_SOURCE" \
    --hdf5-root /mnt/data/vhoangth2/repos/robomme_policy_learning/data/robomme_data_h5 \
    --output "$ROBOMME_INDEX"
```

Build the index once and reuse it for every extraction. An already validated
local index is also available at `.cache/robomme_task_index_v21.json`; to reuse
that file, set `ROBOMME_INDEX` to its absolute path and skip the `index` command.
Both indexing and extraction refuse existing output paths.

Preview the primary selection without creating a dataset:

```bash
"$ROBOMME_PYTHON" -u scripts/extract_robomme_tasks.py extract \
    --source "$ROBOMME_SOURCE" --task-map "$ROBOMME_INDEX" \
    --split primary \
    --output /mnt/data/vhoangth2/datasets/robomme_primary_train_v21 \
    --dry-run
```

To create all five independent training datasets, copy them sequentially:

```bash
for split in primary loso_counting loso_permanence loso_reference loso_imitation; do
    "$ROBOMME_PYTHON" -u scripts/extract_robomme_tasks.py extract \
        --source "$ROBOMME_SOURCE" --task-map "$ROBOMME_INDEX" \
        --split "$split" --workers 4 \
        --output "/mnt/data/vhoangth2/datasets/robomme_${split}_train_v21" || break
done
```

For a custom subset, replace `--split primary` with e.g.
`--tasks BinFill StopCube`, or use `--exclude-tasks PickXtimes PatternLock`.
Named presets require a source containing all 16 task families. Explicit task
selection also works on an extracted dataset, using its own
`meta/episode_tasks.json` as `--task-map`.

Parquet is rewritten to compact episode IDs, instruction IDs and global row
indices. Video bytes are copied directly with `shutil.copy2`, creating separate
files without symlinks, hard links or re-encoding. All `source_*` columns and
subgoal annotations are retained. Episode and global state/action statistics
(min/max/mean/std/q01/q99) are recalculated using only the selected execution
frames. The output also includes the modality config, provenance mapping,
canonical task index and completion manifest. The final directory appears only
after all copying and metadata generation succeed; interrupted copies must be
rerun from the start.

The five copies contain about 9.1 GiB of video data in total, plus parquet and
metadata. No model weights or GPU are needed. Run the extraction tests inside
a CPU allocation with the existing GR00T environment and this checkout on
`PYTHONPATH`:

```bash
env PYTHONPATH=/mnt/data/vhoangth2/repos/Isaac-GR00T \
    USE_TF=0 USE_FLAX=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    NO_ALBUMENTATIONS_UPDATE=1 TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$ROBOMME_PYTHON" -m pytest -q tests/test_extract_robomme_tasks.py
```

## Fine-tune with the existing trainer

Use `scripts/gr00t_finetune.py` directly; no RoboMME-specific training launcher
or trainer changes are needed. Run from the repository root **inside a GPU
allocation**, using the existing GR00T environment. Preserve Slurm's
`CUDA_VISIBLE_DEVICES`; the commands use one allocated GPU.

Set `PYTHONPATH` to this checkout explicitly: the shared `gr00t` environment's
editable installation currently points to `HAMLET-VLA`. Without the override,
running a script file can import that repo's model/trainer instead. The override
below applies only to the command and does not reinstall or change the environment.

Start with this 10-step smoke run. The existing trainer logs every 10 steps,
so this checks its normal logging interval and saves one checkpoint without a
duplicate final model export:

```bash
env PYTHONPATH=/mnt/data/vhoangth2/repos/Isaac-GR00T \
    USE_TF=0 USE_FLAX=0 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    NO_ALBUMENTATIONS_UPDATE=1 TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 WANDB_MODE=disabled \
  /home/vhoangth2/miniconda3/envs/gr00t/bin/python -u scripts/gr00t_finetune.py \
    --dataset-path /mnt/data/vhoangth2/datasets/robomme_gr00t_lerobot_v21 \
    --base-model-path /mnt/data/vhoangth2/hf_cache/hub/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e \
    --data-config examples.RoboMME.custom_data_config:RoboMMEDataConfig \
    --embodiment-tag new_embodiment --num-gpus 1 --seed 42 \
    --batch-size 2 --gradient-accumulation-steps 1 \
    --learning-rate 3e-5 --warmup-ratio 0 --max-steps 10 \
    --save-steps 10 --save-total-limit 1 --no-save-final-model \
    --dataloader-num-workers 2 --dataloader-prefetch-factor 2 \
    --video-backend torchvision_av --report-to wandb \
    --output-dir outputs/robomme_smoke_seed42
```

After creating `robomme_primary_train_v21`, a starting full-training recipe for
the primary held-out protocol is below. Initialize each fold from the original
GR00T base model: the smoke checkpoint saw all 16 tasks and is only for debugging.

```bash
env PYTHONPATH=/mnt/data/vhoangth2/repos/Isaac-GR00T \
    USE_TF=0 USE_FLAX=0 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    NO_ALBUMENTATIONS_UPDATE=1 TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    WANDB_MODE=offline WANDB_PROJECT=gr00t-robomme WANDB_LOG_MODEL=false \
  /home/vhoangth2/miniconda3/envs/gr00t/bin/python -u scripts/gr00t_finetune.py \
    --dataset-path /mnt/data/vhoangth2/datasets/robomme_primary_train_v21 \
    --base-model-path /mnt/data/vhoangth2/hf_cache/hub/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e \
    --data-config examples.RoboMME.custom_data_config:RoboMMEDataConfig \
    --embodiment-tag new_embodiment --num-gpus 1 --seed 42 \
    --batch-size 16 --gradient-accumulation-steps 2 \
    --learning-rate 3e-5 --warmup-ratio 0.05 --max-steps 20000 \
    --save-steps 1000 --save-total-limit 2 --no-save-final-model \
    --dataloader-num-workers 2 --dataloader-prefetch-factor 2 \
    --video-backend torchvision_av --report-to wandb \
    --output-dir outputs/robomme_primary_seed42
```

The full recipe has effective batch size 32 on one GPU. These hyperparameters
are a starting point; the full batch size and long-run behavior have not yet
been validated. The GPU smoke test below used batch size 2. Both commands retain the trainer defaults: frozen
vision/language backbone, trainable projector and diffusion action head,
bfloat16 training and TF32. Choose a GPU compatible with those defaults.

Use a new output directory for each fresh run. To continue an interrupted run,
keep its recipe and output directory and add `--resume`. No command above
submits a Slurm job or requests resources automatically.

The CPU validation in allocation 66430 checks the existing CLI argument parser
and the real converted dataset through `torchvision_av`, two DataLoader workers
and `DefaultDataCollator`. It uses the first and final samples, verifying finite
state/action tensors and masks. This validates the input path only; no model
weights are loaded and no model forward/backward or training is run on CPU.

## GPU smoke result

On 2026-11-20, the existing trainer completed 10 optimizer steps in allocation
66426 on one H100 80 GB, using batch size 2, accumulation 1 and seed 42. Logged
loss was 0.5206, gradient norm was 3.4809, and Trainer reported 28.34 seconds
including checkpoint saving. The full dataset was available to the sampler;
only 20 training samples were consumed in this short run.

The checkpoint contains readable safetensors shards/index (899 tensors),
optimizer, scheduler, RNG state and `experiment_cfg/metadata.json` with the
RoboMME normalization and modality metadata. The saved action horizon is 16 and
the padded action dimension is 32. The extra final-model export was disabled.

Local artifacts:

```text
outputs/robomme_smoke/20261120T101021Z_job66426_seed42/train.log
outputs/robomme_smoke/20261120T101021Z_job66426_seed42/smoke_summary.json
outputs/robomme_smoke/20261120T101021Z_job66426_seed42/checkpoint-10/
```

The initial attempt stopped before step 1 because the environment imported
`TrainRunner` from its editable `HAMLET-VLA` installation. Setting the per-command
`PYTHONPATH` as shown above resolved the mismatch without changing trainer code
or reinstalling the environment. Resume execution and simulator rollouts have
not been tested with this checkpoint.

## Observation/action mapping (adapter step 1)

`examples/RoboMME/adapter.py` provides two NumPy-only mapping functions. They
follow the benchmark's buffered observation interface and the existing
`RoboMMEDataConfig`. The HTTP client and evaluation entry point described below
use this mapping.

```python
from examples.RoboMME.adapter import to_gr00t_observation, to_robomme_actions

# benchmark_inputs contains the four observation buffers and task_goal.
observation = to_gr00t_observation(benchmark_inputs)
# With an initialized Gr00tPolicy, later in the inference integration:
prediction = policy.get_action(observation)
actions = to_robomme_actions(prediction)  # float32 (H, 8), normally (16, 8)
```

The four buffers (`front_rgb_list`, `wrist_rgb_list`, `joint_state_list`,
`gripper_state_list`) must have equal nonzero lengths. The mapping takes their
last entries, including when the initial request contains a conditioning video.
Front/wrist RGB arrays retain their uint8 HWC layout and gain a single time
dimension. Seven joint positions become `state.joint_position` `(1, 7)`; the
first of two physical finger positions becomes `state.gripper` `(1, 1)`.
`task_goal[0]` is passed verbatim as the instruction (a single string is also
accepted). History, oracle subgoals and other optional observations are not
consumed by this current-observation baseline. Returned arrays are independent
copies, so model preprocessing cannot modify simulator buffers.

`to_robomme_actions` expects the unbatched **physical** output fields from
`Gr00tPolicy.get_action`: `action.joint_position` `(H, 7)` and `action.gripper`
`(H, 1)`. It preserves absolute joint targets and the predicted gripper command
values in their learned -1/+1 scale, without another inverse normalization,
threshold or delta conversion. The existing benchmark wrapper drops the gripper
dimension for PatternLock/RouteStick. Image preprocessing and normalization use
the existing GR00T transforms and checkpoint metadata. Invalid shapes, empty or
unaligned buffers, and non-finite state/action values are rejected.

Run mapping and transform integration checks **inside a CPU allocation**:

```bash
env PYTHONPATH=/mnt/data/vhoangth2/repos/Isaac-GR00T \
    USE_TF=0 USE_FLAX=0 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    NO_ALBUMENTATIONS_UPDATE=1 TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    /home/vhoangth2/miniconda3/envs/gr00t/bin/python -m pytest -q \
    tests/test_robomme_adapter.py tests/test_robomme_data_config.py
```

The mapping and transform checks need no simulator, checkpoint weights or GPU.
The HTTP checks below additionally start a local test server on the CPU node.

## HTTP inference client (adapter step 2)

`RoboMMEGr00tClient`, in the same adapter module, exposes `infer(inputs)` and
`reset()` for the benchmark's observation format. It reuses the existing
`scripts/inference_service.py --server --http-server`; no benchmark or policy
repo modifications are needed. The simulator environment needs NumPy, Requests
and `json-numpy` for the client. Model loading stays in the GR00T environment.

To serve a trained checkpoint, run from this repo **inside a GPU allocation**:

```bash
ROBOMME_CHECKPOINT=/path/to/checkpoint

env PYTHONPATH=/mnt/data/vhoangth2/repos/Isaac-GR00T \
    USE_TF=0 USE_FLAX=0 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    NO_ALBUMENTATIONS_UPDATE=1 TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    /home/vhoangth2/miniconda3/envs/gr00t/bin/python -u scripts/inference_service.py \
    --server --http-server --host 0.0.0.0 --port 8000 \
    --model-path "$ROBOMME_CHECKPOINT" \
    --data-config examples.RoboMME.custom_data_config:RoboMMEDataConfig \
    --embodiment-tag new_embodiment --seed 42
```

Use the checkpoint directory containing `experiment_cfg/metadata.json`. The
existing policy reads that checkpoint's normalization statistics and applies
eval transforms and inverse action transforms. The client does not load a
dataset or apply normalization itself.

The client can be called in the simulator process, with this checkout on its
`PYTHONPATH`:

```python
from examples.RoboMME.adapter import RoboMMEGr00tClient

# For a server on another node, replace 127.0.0.1 with its reachable hostname/IP.
with RoboMMEGr00tClient("http://127.0.0.1:8000", timeout=60.0) as client:
    assert client.reset()["reset_finished"]
    actions = client.infer(benchmark_inputs)["actions"]  # float32 (H, 8)
```

`infer()` maps the current observation, posts `{"observation": ...}` to `/act`
using explicit NumPy-aware JSON encoding, decodes the physical action fields
and packs them with `to_robomme_actions`. The client uses a persistent Requests
session and does not patch Python's global JSON functions. HTTP errors and
timeouts propagate, and malformed responses or actions are rejected. Inference
is never retried automatically: an interrupted request may already have
advanced the server's sampling RNG.

`reset()` returns `{"reset_finished": True}` locally. This baseline retains no
history or action queue, and the GR00T HTTP server has no reset endpoint. Reset
does not reseed the model server; the rollout runner owns its action queue.
The context manager closes the HTTP session when evaluation finishes.

The test command above also exercises a local instance of the **actual GR00T
HTTP server with a fake policy**, in a separate process. Checks cover NumPy
dtype/shape and Unicode instruction round-trips, consecutive episodes, reset,
server errors, invalid action fields, malformed responses and timeouts without
retry. These tests load no model weights and use no simulator or GPU. A real
checkpoint inference and simulator rollout remain for the later GPU smoke step.

## Held-out evaluation entry point (adapter step 3)

`python -m examples.RoboMME.eval` runs a remote checkpoint on the `excluded_tasks`
in the **training dataset's** extraction manifest. All integration code lives in
this checkout. The runner imports `BenchmarkEnvBuilder` from the explicitly
selected RoboMME clone and uses its environment `reset`/`step` API, test episode
metadata and joint-angle controller. The small local rollout loop adds an
execution horizon, a strict step budget and cleanup on errors.

First preview the protocol **inside a CPU allocation**. This reads only JSON
metadata, without importing the simulator, contacting the server or creating an
output directory:

```bash
cd /mnt/data/vhoangth2/repos/Isaac-GR00T
ROBOMME_CHECKPOINT=/path/to/checkpoint

env PYTHONPATH=/mnt/data/vhoangth2/repos/Isaac-GR00T \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    /home/vhoangth2/miniconda3/envs/gr00t/bin/python -m examples.RoboMME.eval \
    --manifest /mnt/data/vhoangth2/datasets/robomme_primary_train_v21/manifest.json \
    --checkpoint "$ROBOMME_CHECKPOINT" \
    --benchmark-root /mnt/data/vhoangth2/repos/robomme_benchmark \
    --output-dir outputs/robomme_eval/primary_seed42 \
    --server-url http://127.0.0.1:8000 \
    --num-episodes 50 --exec-horizon 16 --max-steps 1300 --seed 42 \
    --dry-run
```

For the real run, start the checkpoint's HTTP server as in step 2 and use the
**RoboMME simulator environment** for this command, inside an allocation with
the required simulation/rendering resources. Set the same checkpoint and model
seed in the server command. If the server runs on another node, replace the URL
with that node's reachable hostname/IP:

```bash
env PYTHONPATH=/mnt/data/vhoangth2/repos/Isaac-GR00T PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    /home/vhoangth2/.local/share/mamba/envs/robomme/bin/python -m examples.RoboMME.eval \
    --manifest /mnt/data/vhoangth2/datasets/robomme_primary_train_v21/manifest.json \
    --checkpoint "$ROBOMME_CHECKPOINT" \
    --benchmark-root /mnt/data/vhoangth2/repos/robomme_benchmark \
    --output-dir outputs/robomme_eval/primary_seed42 \
    --server-url http://127.0.0.1:8000 \
    --num-episodes 50 --exec-horizon 16 --max-steps 1300 --seed 42
```

The primary protocol selects ButtonUnmaskSwap, PatternLock, PickHighlight and
PickXtimes, with 50 test episodes each. For a LOSO checkpoint, supply its training
dataset's manifest and a separate output directory. All five local manifests
were checked against the benchmark's test metadata: each selects four held-out
tasks with episode IDs 0..49 available.

For a single-episode smoke run, replace `--num-episodes 50` with
`--episode-indices 0` and add e.g. `--tasks PickXtimes`, using a fresh output
directory. `--tasks` accepts only a subset of the manifest's excluded tasks;
training tasks are rejected. `--episode-indices` also supports non-contiguous
selections. `--exec-horizon` can be 1..16 and determines how many predicted
actions execute before replanning; episode termination stops a chunk immediately.
The execution budget excludes the initial demo prefix. Use `--no-video` to
disable recording; the default video FPS is 10.

The checkpoint string is a **declared identity** for the remote server; the
current GR00T HTTP API does not expose its loaded checkpoint or verify its
relationship to a dataset manifest. Use the matching manifest/checkpoint pair.
`--seed` seeds local Python, NumPy and Torch; the server's sampling seed is set
separately by its `--seed`. Environment episode seeds come from test metadata.

Each run saves:

```text
config.json        # checkpoint identity, manifest/metadata hashes, protocol and endpoint
progress.json      # task/episode, outcome, step count, error and video path
metrics.json       # overall and per-task counts and success rates
videos/            # streamed front+wrist MP4s, demo-prefix frames marked in red
```

Results distinguish success, fail, timeout and error. Success rates use **all
attempted episodes**, including errors, as the denominator; pending episodes are
listed through expected/completed counts. Environment and video resources are
closed even when inference or stepping fails. Cleanup errors are logged
separately. The command exits nonzero if any episode or resource cleanup failed.

Fresh runs refuse an existing output directory. Add `--resume` with the same
arguments/output directory to skip already recorded episodes, including recorded
errors. Resume rejects changed configuration or manifest/test metadata hashes.
It preserves results, but does not restore the remote model's sampling RNG;
resumed runs are not guaranteed to match an uninterrupted stochastic rollout.

Run CPU tests with the existing GR00T environment inside an allocation:

```bash
env PYTHONPATH=/mnt/data/vhoangth2/repos/Isaac-GR00T \
    USE_TF=0 USE_FLAX=0 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    NO_ALBUMENTATIONS_UPDATE=1 TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    /home/vhoangth2/miniconda3/envs/gr00t/bin/python -m pytest -q \
    tests/test_robomme_eval.py tests/test_robomme_adapter.py tests/test_robomme_data_config.py
```

The evaluation tests use fake environments/clients and include actual MP4
encoding/decoding, early termination, replanning, queue reset, timeouts, structured
simulator errors and interrupted-run resume. Real checkpoint inference and a
real RoboMME episode remain for the GPU smoke step.
