# Standalone training folds (unit 3)

`scripts/extract_libero_mem_fold.py` calls [`splits.py`](splits.py) to select
the proposal's training tasks from the canonical ten-task LeRobot dataset.
Each fold contains its own parquet, video, metadata and statistics files.
**All files are independent copies: no hard links or symlinks.** Provenance paths
in JSON are records only; the GR00T loader does not need the canonical dataset,
HDF5 sources or conversion work cache to read a fold.

| Training fold | Train tasks | Held-out tasks | Episodes | Frames | Videos |
| --- | --- | --- | ---: | ---: | ---: |
| A | T1, T3, T5, T7, T9 | T2, T4, T6, T8, T10 | 483 | 147,948 | 966 |
| B | T2, T4, T6, T8, T10 | T1, T3, T5, T7, T9 | 478 | 173,457 | 956 |

These are counts observed on the local source, not required episode counts.
Every fold requires all five specified training task families. The extractor
requires a complete canonical dataset containing all ten task families.

## Extract a fold

Run from the repository root in an existing CPU allocation:

```bash
srun --jobid=67857 --overlap --ntasks=1 --cpus-per-task=2 --mem=12G \
    env PYTHONPATH="$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    /home/vhoangth2/miniconda3/envs/gr00t/bin/python -u \
    scripts/extract_libero_mem_fold.py \
    --source /mnt/data/vhoangth2/datasets/libero_mem_lerobot_v21 \
    --fold A \
    --output /mnt/data/vhoangth2/datasets/libero_mem_fold_a_train
```

For B, use `--fold B` and `libero_mem_fold_b_train`. Both local folders above
were created during unit-3 validation, so rerunning these exact commands will
refuse the existing outputs. Choose a new output path for another extraction.
Append `--dry-run` with a new path to inspect task selection, counts and video
bytes without copying. `--chunks-size` defaults to 1000.

The extractor validates canonical metadata, task names, instruction vocabulary,
episode provenance and the existence of every canonical parquet/video. It checks
each selected episode's vectors, indices, timestamps, done flag and gripper
convention. Video headers must match episode length, 256x256 RGB dimensions and
20 FPS. Videos are copied byte-for-byte and checksummed, with no re-encoding or
pixel transformation. Image payloads are not decoded during extraction.

Temporary output lives beside the destination and is renamed into place after
the complete fold is ready. Input changes during extraction cause an error;
existing output is never overwritten. This copy operation has no resume cache:
retry the command after an interruption if final output was not published.

## Indices, provenance and normalization

Episode indices, global frame indices and instruction `task_index` values are
compact within each fold. Frame indices, timestamps and all source identity
columns retain their values. Three added columns preserve parent indices:

- `canonical_episode_index`
- `canonical_task_index`
- `canonical_index`

`meta/episode_mapping.json` retains original task/demo identity and initial-state
fingerprints, and adds canonical episode/instruction IDs. `meta/tasks.jsonl`
contains only the five training instructions. Task selection uses canonical task
names from the registry, independently of source instruction-ID order.

`meta/stats.json` and `meta/episodes_stats.jsonl` are recomputed from selected
training rows: min, max, mean, population std and 1st/99th percentiles for the 8D
state and 7D action. Source global statistics are never used for these values.
The manifest records train/held-out task lists, counts, source metadata hashes
and `storage: independent_copies`. `meta/asset_checksums.json` records output
parquet and video SHA256 values.

Use the fold path with the unchanged configuration
`examples.Libero.custom_data_config:LiberoDataConfig` and
`--embodiment-tag new_embodiment` when wiring training. Evaluation initial states
and retrieval support buffers are separate protocol inputs, not additional
training data created by this extractor.

## Validation

`tests/test_libero_mem_splits.py` checks complementary task/episode partitions,
remapping with reordered instruction IDs, chunk boundaries, byte-identical video
copies with independent inodes, normalization recomputation against deliberately
incorrect source global statistics, malformed provenance and video headers,
source changes, dry runs and overwrite protection. A loader test changes and
moves the canonical source after extraction and still reads the copied fold.

Run in the CPU allocation with the existing GR00T environment:

```bash
srun --jobid=67857 --overlap --ntasks=1 --cpus-per-task=2 --mem=12G \
    env PYTHONPATH="$PWD" USE_TF=0 USE_FLAX=0 HF_HUB_OFFLINE=1 \
        TRANSFORMERS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1 \
        OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    /home/vhoangth2/miniconda3/envs/gr00t/bin/python -m pytest -q \
    tests/test_libero_mem_splits.py
```

On 2026-11-27 in CPU allocation `67857`, all 15 focused tests passed. Both full
local folds were copied and independently verified: all 961 episodes' state and
action rows match the canonical source, all 1,922 videos match source SHA256,
and every output file has link count one with no symlinks. The episode partitions
are disjoint and cover all canonical episodes. All six normalization statistics
were recomputed independently for comparison, and 20 samples across the ten
tasks were read through the existing GR00T loader. The local report is
`outputs/libero_mem_folds_validation_unit3.json`. This validates dataset copying
and loading; it does not run a simulator or measure policy performance.
