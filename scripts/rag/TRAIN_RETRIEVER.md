# Train the action-aware LIBERO retriever

After training, see [the fixed-policy A/B sanity check](RETRIEVER_SANITY_EVAL.md).
For RoboMME training/support memory, see [BUILD_MEMORY.md](BUILD_MEMORY.md).

The launcher trains on Goal/Object/Spatial and holds out Long. It uses the
original online action-DTW sampler; it does not read the offline `pairs.jsonl`.
GR00T is frozen and only the retriever is optimized.

From the repo on a GPU node, try 100 optimizer steps with a small batch:

```bash
bash scripts/train_libero_retriever.sh \
  --retriever-path outputs/retriever/libero_gos_trial_seed42 \
  --num-iterations 100 --num-trajs 3 --num-frames 4 \
  --num-workers 2 --print-interval 10
```

From the login node, use the supplied allocation while it is still active:

```bash
srun --jobid=66728 --overlap --nodes=1 --ntasks=1 --cpus-per-task=4 \
  --gres=gpu:1 --time=00:30:00 \
  bash scripts/train_libero_retriever.sh \
  --retriever-path outputs/retriever/libero_gos_trial_seed42 \
  --num-iterations 100 --num-trajs 3 --num-frames 4 \
  --num-workers 2 --print-interval 10
```

For the authors' original step/sampling budget, use 750 steps and a larger batch:

```bash
bash scripts/train_libero_retriever.sh \
  --retriever-path outputs/retriever/libero_gos_750_seed42 \
  --num-iterations 750 --num-trajs 10 --num-frames 20 \
  --num-workers 2 --print-interval 25
```

The launcher sets seed 42; override with `--seed 0` for the original script's
default seed. `num_trajs` counts task/demo-pair groups per iteration, distributed
randomly across suites. Each group uses two demos and `num_frames` positive
matches, so **batch observations = 2 × num_trajs × num_frames**. The small run
uses 24 observations; the original budget uses 400. Changing this also changes
the contrastive negative set. The small run is a pipeline trial, not an
equivalent reproduction. Retriever training is step-based, independent of the
three-epoch RA-VLA policy-training budget.

Because the sampler can assign every group to one suite, `num_trajs` must not
exceed the eligible alignment-group count in any selected dataset (10 for each
G/O/S suite). Groups need two demonstrations with at least `num_frames` frames.
Singletons and short demonstrations are excluded only from retriever pairing;
dataset episodes, policy-training frames and normalization metadata are retained.

## Configuration and memory

The existing `gr00t` environment and cached `nvidia/GR00T-N1.5-3B` are used.
The corrected data config is `examples.Libero.custom_data_config:LiberoDataConfig`.
Overrides include:

- `--data-config`: data transform/config class.
- `--pretrained-model-path`: a local base GR00T checkpoint or model ID.
- `--task-map`: optional episode-to-task index for a single-dataset run; by
  default use the dataset's `meta/episode_tasks.json` when available.
- `--video-cache-size 2`: camera videos retained per dataset per worker.
- `--num-workers 0`: load in the main process for debugging.
- `LIBERO_DATA_ROOT`: root containing `libero_goal`, `libero_object`, `libero_spatial`.
- `POLICY_PYTHON`: Python executable instead of the existing `gr00t` environment.

Video loading defaults to the existing bounded exact-frame-index cache. It does
not load/create a dataset-wide `frames.pt`. Prefetch is limited to one batch per
worker. `--cache-video` opts into the original full RAM cache. Hugging Face
access defaults to offline, using the model already cached here. On a fresh
machine, download the model first or set `HF_HUB_OFFLINE=0` when launching.

## Outputs

The chosen `--retriever-path` contains:

- `training_config.json`: settings and dataset paths.
- `sampling_report.json`: grouping strategy, task-map fingerprint, eligible
  group/demo counts and skipped groups/short demonstrations for each dataset.
- `metrics.jsonl`: per-step loss, gradient norm, learning rate and batch size.
- `metadata.pt`, `config.json`, `model.safetensors`: trained model/data metadata.
- `completed.json`: completed steps, final loss, parameter-update check,
  frozen-base checks and peak GPU memory.
- `messages.log`: mean loss for each logging interval, including the final one.

Existing model checkpoints are rejected to prevent overwriting them. Choose a
fresh output directory for another run. Optimizer/RNG state is not saved, so
training resume is not supported (unlike the offline pair exporter).

Before using a new retriever with RA-VLA, rebuild memory keys with that
retriever. Released memory keys belong to the released retriever's embedding
space and should not be mixed with the new model's queries.

## RoboMME pairing compatibility (step 2)

The retriever can read the already converted RoboMME v2.1 datasets with
`--data-config examples.RoboMME.custom_data_config:RoboMMEDataConfig` and
`--dataset-paths /path/to/robomme_primary_train_v21`. Its embedded task index
is validated against the dataset metadata, and pairing groups use
**(task family, exact instruction)**. This prevents VideoUnmask and
VideoUnmaskSwap from being paired merely because they share an instruction.
LIBERO retains instruction-based grouping and the original batch ordering/RNG
behavior when its demonstrations satisfy the pairing requirements.

For the local primary dataset and `--num-frames 20`, there are 99 compound
groups: 81 eligible groups and 18 singleton groups. Only the 1,182 eligible
demonstrations supply retriever positives; all 1,200 demonstrations remain in
the dataset. The sampler writes the exact exclusions to `sampling_report.json`
and never edits the source dataset or `dataset.task_groups`.

For the all16 dataset, which has no embedded task index, pass the portable map:

```bash
--task-map examples/RoboMME/metadata/robomme_task_index_v21.json
```

An explicit map requires a single dataset path. Converted RoboMME datasets
without a map are rejected rather than falling back to instruction-only grouping.
The map's HDF5 provenance paths are not opened.

DTW reads the raw action column named by modality metadata (`actions`, 8D, for
RoboMME; `action`, 7D, for LIBERO). Its distance, path filtering and frame sampling
are unchanged; no action normalization, dimension dropping or new positive-pair
algorithm is introduced. The offline exporter/viewer also support RoboMME;
see [DTW_PAIRS.md](DTW_PAIRS.md). Their exports are optional inspection artifacts;
retriever training still computes DTW online. Changing the
sampler utility source changes export fingerprints, so existing DTW exports stay
viewable but re-mining/resuming requires a fresh output directory.

CPU checks (run inside an allocation with this checkout on `PYTHONPATH`):

```bash
USE_TF=0 USE_FLAX=0 NO_ALBUMENTATIONS_UPDATE=1 HF_HUB_OFFLINE=1 \
  /home/vhoangth2/miniconda3/envs/gr00t/bin/python -m pytest -q \
  tests/test_retriever_sampling.py tests/test_dtw_pairs.py \
  tests/test_retriever_loss.py tests/test_ravla_video_dataset.py
```

## Validation

Allocation 66728 was used for small GPU smoke tests: 3 steps with 12 observations
and no DataLoader workers, then 2 steps with 24 observations and 2 workers.
These exercise actual forward/backward/optimizer/save operations. An additional
5-step benchmark with the original 400-observation batch and 2 workers succeeded:
the four intervals after the first optimizer step averaged 7.23 seconds/step,
with 13.31 GiB peak GPU allocations. This extrapolates to about 90 minutes for
750 steps; allow roughly 1.5–2 hours for I/O and shared-node variation. The
benchmark used a process-local 20% VRAM cap to leave memory for the existing
workload. It did not run all 750 steps. Details are saved in
`outputs/retriever/gos_batch400_timing_20261122/timing.json`.
The smoke tests do not establish retrieval quality or downstream success rate.

```bash
USE_TF=0 NO_ALBUMENTATIONS_UPDATE=1 HF_HUB_OFFLINE=1 \
  /home/vhoangth2/miniconda3/envs/gr00t/bin/python -m pytest -q \
  tests/test_retriever_loss.py tests/test_ravla_video_dataset.py
```
