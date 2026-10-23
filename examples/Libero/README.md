# GR00T Libero Benchmarks

This directory contains fine-tuning and evaluation scripts for **GR00T N1.5** on the Libero benchmark suite.



## 🎯 Model Evaluation

Evaluation is performed using [`run_libero_eval.py`](https://github.com/NVIDIA/Isaac-GR00T/examples/Libero/eval/run_libero_eval.py).

<!-- Spatial: /mnt/amlfs-02/shared/checkpoints/xiaoweij/0827/libero-checkpoints-20K/checkpoint-20000/ -->
<!-- Goal: /mnt/amlfs-02/shared/checkpoints/xiaoweij/0911/libero-goal-checkpoints-20K/ https://wandb.ai/nv-gear/huggingface/runs/wibov9ph?nw=nwuserxiaoweij -->
<!-- Object: /mnt/amlfs-02/shared/checkpoints/xiaoweij/0904/libero-object-checkpoints-20K/ https://wandb.ai/nv-gear/huggingface/runs/38tmzwcw?nw=nwuserxiaoweij -->
<!-- Libero-90: /mnt/amlfs-02/shared/checkpoints/xiaoweij/0905/libero-90-checkpoints-60K/  https://wandb.ai/nv-gear/huggingface/runs/3wpxrsri?nw=nwuserxiaoweij -->
<!-- Libero-Long: /mnt/amlfs-02/shared/checkpoints/xiaoweij/0908/libero-10-checkpoints-60K/ https://wandb.ai/nv-gear/huggingface/runs/cyh7mdtx?nw=nwuserxiaoweij  -->

### Eval Result and Training Config Table

| Task      | Success rate       | max_steps | grad_accum_steps | batch_size | Data config                                                     |                   Checkpoint                  |
|-----------|--------------------|-----------|------------------|------------|-----------------------------------------------------------------|-----------------------------------------------|
| Spatial   | 46/50 (92%)        | 20K       | 1                | 128        | examples.Libero.custom_data_config:LiberoDataConfig             |youliangtan/gr00t-n1.5-libero-spatial-posttrain|
| Goal      | 43/50 (86%)        | 20K       | 4                | 72         | examples.Libero.custom_data_config:LiberoDataConfigMeanStd      |youliangtan/gr00t-n1.5-libero-goal-posttrain|
| Object    | 46/50 (92%)        | 20K       | 1                | 128        | examples.Libero.custom_data_config:LiberoDataConfig             |youliangtan/gr00t-n1.5-libero-object-posttrain|
| Libero-90 | 402/450 (89.3%)    | 60K       | 1                | 128        | examples.Libero.custom_data_config:LiberoDataConfig             |youliangtan/gr00t-n1.5-libero-90-posttrain|
| Long      | 38/50 (76%)        | 60K       | 1                | 128        | examples.Libero.custom_data_config:LiberoDataConfig             |youliangtan/gr00t-n1.5-libero-long-posttrain|





> Note: The results reported above were obtained with minimal hyperparameter tuning and are intended primarily for demonstration purposes. More comprehensive studies have fine-tuned GR00T on LIBERO and achieved substantially higher performance. For example, see Table 3 in this [paper](https://arxiv.org/pdf/2508.21112).
----

To evaluate, first start the inference server with our provided checkpoint:

```bash
python scripts/inference_service.py \
    --model_path youliangtan/gr00t-n1.5-libero-spatial-posttrain \
    --server \
    --data_config examples.Libero.custom_data_config:LiberoDataConfig \
    --denoising-steps 8 \
    --port 5555 \
    --embodiment-tag new_embodiment
```

> Note, for **Libero-Goal**, the checkpoints are trained using data config: `examples.Libero.custom_data_config:LiberoDataConfigMeanStd`. So the corresponding checkpoints should be served using commands:
```bash
python scripts/inference_service.py \
    --model_path /mnt/amlfs-02/shared/checkpoints/xiaoweij/0913/libero-goal-checkpoints-20K/checkpoint-20000 \
    --server \
    --data_config examples.Libero.custom_data_config:LiberoDataConfigMeanStd \
    --denoising-steps 8 \
    --port 5555 \
    --embodiment-tag new_embodiment
```
----

### Installation

Follow the [official Libero installation guide](https://lifelong-robot-learning.github.io/LIBERO/html/getting_started/installation.html).

### Troubleshooting

If you see:
```
ModuleNotFoundError: No module named 'robosuite.environments.manipulation.single_arm_env'
```

Make sure you install:
```bash
pip install robosuite==1.4.0
```

Then run the evaluation:
```bash
cd examples/Libero/eval
python run_libero_eval.py --task_suite_name libero_spatial
```

----

## Reproduce Training Results

To reproduce the training results, you can use the following steps:
1. Download the datasets
2. Add the modality configuration files
3. Fine-tune the model
4. Evaluate the model (same as above)

## 📦 1. Dataset Preparation

### Dataset Downloads
Download LeRobot-compatible datasets directly from Hugging Face.

```bash
huggingface-cli download \
    --repo-type dataset IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot \
    --local-dir /tmp/libero_spatial/
```

> 🔄 Replace with the appropriate dataset name:
> - `IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot` (for **goal**)
> - `IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot` (for **object**)
> - `IPEC-COMMUNITY/libero_90_no_noops_lerobot` (for **libero-90**)
> - `IPEC-COMMUNITY/libero_10_no_noops_lerobot` (for **libero-10**)

### Modality Configuration

After downloading the datasets, you need to add the appropriate modality configuration files to make them compatible with GR00T N1.5. These configuration files define the observation and action space mappings.

```bash
cp examples/Libero/modality.json /tmp/libero_spatial/meta/modality.json
```

## 🚀 Model Fine-tuning

### Training Commands

The fine-tuning script supports multiple configurations.

```bash
python scripts/gr00t_finetune.py \
    --dataset-path /tmp/libero_spatial/ \
    --data_config examples.Libero.custom_data_config:LiberoDataConfig \
    --num-gpus 8 \
    --batch-size 128 \
    --output-dir /tmp/my_libero_spatial_checkpoint/ \
    --max-steps 60000
```
> Note, replace with the corresponding data config class and training configs according to the [table](#training-config-table).

## Leave-one-suite-out training on Slurm

The cluster scripts train four independent models from the original GR00T-N1.5
checkpoint. Each model holds out one of Spatial, Object, Goal, and Long, and
trains on the other three. All runs use `LiberoDataConfig` (min-max action
normalization), batch size 32, one H100 80GB, 8 CPUs, and 128GB RAM.

Stage the already downloaded, filtered LeRobot datasets without changing their
shared cache (Long maps to `libero_10_no_noops_1.0.0_lerobot`):

```bash
/home/vhoangth2/miniconda3/envs/gr00t/bin/python scripts/prepare_libero_loso.py \
    --source-root /mnt/data/vhoangth2/hf_cache/lerobot/IPEC-COMMUNITY \
    --destination-root demo_data/libero_loso
/home/vhoangth2/miniconda3/envs/gr00t/bin/python scripts/validate_libero_videos.py
PYTHONPATH=. /home/vhoangth2/miniconda3/envs/gr00t/bin/python scripts/prepare_libero_clean.py \
    --validation-report outputs/video_debug/full_validation.json
bash scripts/submit_libero_loso.sh
```

These local datasets contain 432 / 454 / 428 / 379 demonstrations for
Spatial / Object / Goal / Long, respectively, rather than 500 per suite.
The preparation script verifies all parquet/video paths and records the counts
in `demo_data/libero_loso/manifest.json`.
The video validation decodes all frames using one decoder thread per process.
The cleaning step excludes each demonstration with a damaged video, renumbers
retained episodes consistently across parquet and video paths, and recomputes
state/action normalization from the retained frames. It writes a separate
`demo_data/libero_loso_clean` dataset; the original cache is unchanged.
The excluded episode IDs and counts are recorded in that directory's manifest.
The Slurm scripts default to this cleaned dataset.
The current cache has one damaged stream among 3,386 videos: original Goal
episode 82, wrist camera, fails at frame 105 (5.25 seconds). Excluding that
entire demonstration leaves Goal with 427 demonstrations and 51,913 frames.
Use `goal/meta/episode_mapping.json` to map the cleaned IDs back to the originals.

To restart only selected configurations in a new output directory, pass their
held-out suites, for example `bash scripts/submit_libero_loso.sh spatial object long`.

The submission script records four job IDs in
`outputs/libero_loso/<timestamp>/submissions.tsv`. Logs go to
`outputs/slurm/gr00t-loso-<held-out-suite>_<job-id>.{out,err}`.
Metrics are logged to W&B online in project `gr00t-libero-loso`, grouped by
the experiment timestamp, with one run per held-out suite and Slurm job ID.
Existing W&B login credentials must be available on compute nodes. Override
`WANDB_PROJECT` and `WANDB_ENTITY` before submission to choose a destination.
Checkpoint uploads to W&B are disabled; weights stay on the cluster.
Override `LOSO_RUN_ROOT`, `GR00T_PYTHON`, `LIBERO_DATA_ROOT`, or
`GR00T_BASE_MODEL` in the environment if needed.

Each job trains for **2 epochs**, with `max_steps=-1` and `save_strategy=epoch`,
and retains **two standalone checkpoints** with optimizer/scheduler/RNG state.
The extra final model export is disabled to avoid a third copy of the weights.
Checkpoints live in `heldout_<suite>/checkpoint-<optimizer-step>`; the job
verifies epochs 1 and 2 and writes their paths to `checkpoints.json`.
Mixture epochs use the existing random sampling with replacement, with subset
probabilities proportional to frame counts. Two epochs do not guarantee that
each individual frame was sampled twice.

The scripts submit training only. Evaluate each checkpoint on its held-out
suite separately, using the same `LiberoDataConfig` and `new_embodiment` tag.
