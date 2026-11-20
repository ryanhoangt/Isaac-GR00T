# RoboMME data preparation for GR00T N1.5

The converter accepts a **local RoboMME LeRobot v2.x directory**, not a Hub
repo ID, and does not download anything. `RoboMMEDataConfig` now connects the
converted dataset to GR00T transforms and the existing collator. Fine-tuning
uses `scripts/gr00t_finetune.py` directly. GPU training checks and an evaluation
adapter are separate next steps.

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
outputs. GPU training validation remains the next stage.

## Fine-tune with the existing trainer

Use `scripts/gr00t_finetune.py` directly; no RoboMME-specific training launcher
or trainer changes are needed. Run from the repository root **inside a GPU
allocation**, using the existing GR00T environment. Preserve Slurm's
`CUDA_VISIBLE_DEVICES`; the commands use one allocated GPU.

Start with this 10-step smoke run. The existing trainer logs every 10 steps,
so this checks its normal logging interval and saves one checkpoint without a
duplicate final model export:

```bash
env USE_TF=0 USE_FLAX=0 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
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

After the smoke run succeeds, a starting full-training recipe is:

```bash
env USE_TF=0 USE_FLAX=0 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    NO_ALBUMENTATIONS_UPDATE=1 TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    WANDB_MODE=offline WANDB_PROJECT=gr00t-robomme WANDB_LOG_MODEL=false \
  /home/vhoangth2/miniconda3/envs/gr00t/bin/python -u scripts/gr00t_finetune.py \
    --dataset-path /mnt/data/vhoangth2/datasets/robomme_gr00t_lerobot_v21 \
    --base-model-path /mnt/data/vhoangth2/hf_cache/hub/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e \
    --data-config examples.RoboMME.custom_data_config:RoboMMEDataConfig \
    --embodiment-tag new_embodiment --num-gpus 1 --seed 42 \
    --batch-size 16 --gradient-accumulation-steps 2 \
    --learning-rate 3e-5 --warmup-ratio 0.05 --max-steps 20000 \
    --save-steps 1000 --save-total-limit 2 --no-save-final-model \
    --dataloader-num-workers 2 --dataloader-prefetch-factor 2 \
    --video-backend torchvision_av --report-to wandb \
    --output-dir outputs/robomme_train_seed42
```

The full recipe has effective batch size 32 on one GPU. These hyperparameters
are a starting point; GPU memory use, loss and checkpoint saving have not yet
been validated on RoboMME. Both commands retain the trainer defaults: frozen
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
