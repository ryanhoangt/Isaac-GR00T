# Cached-history training data (step 3)

`RoboMMEMemoryDataset` pairs the existing converted execution samples with the
full-episode visual cache from [step 2](FEATURE_CACHE.md). It subclasses
`LeRobotSingleDataset`, so dataset metadata and the existing sampler/collator
interfaces are retained. The vanilla dataset and `RoboMMEDataConfig` are unchanged.

Example for a completed full cache and the primary training dataset:

```python
from torch.utils.data import DataLoader

from examples.RoboMME.memory_dataset import RoboMMEMemoryDataset
from gr00t.model.transforms import DefaultDataCollator


def main():
    dataset = RoboMMEMemoryDataset(
        dataset_path="/mnt/data/vhoangth2/datasets/robomme_primary_train_v21",
        memory_cache="outputs/robomme_memory_features",
        memory_config="examples/RoboMME/config/perceptual-framesamp-modul.yaml",
        video_backend="torchvision_av",
        max_open_episodes=8,
    )
    loader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=2,
        multiprocessing_context="spawn",
        collate_fn=DefaultDataCollator(),
    )
    batch = next(iter(loader))
    print(batch["static_image_emb"].shape)  # (2,512,2048)
    print(batch["static_pos_emb"].shape)    # (2,512,768)
    print(batch["static_mask"].dtype)       # torch.bool, True means valid
    dataset.close()


if __name__ == "__main__":
    main()
```

Use the all-task converted dataset path for all16. A cache containing all source
episodes can serve both protocols: the execution dataset determines which
episodes/steps are training samples. Unselected cache episode files are not
opened, and cached observations never contribute to normalization statistics.
The two-episode smoke cache is insufficient for a full all16/primary dataset;
the loader rejects missing selected episodes.

The optional data-config name is
`examples.RoboMME.memory_data_config:RoboMMEMemoryDataConfig`. The dataset uses
it by default. It preserves the current two camera views, task text, state,
16-action chunk, normalization and masks from the existing config. Its final
`RoboMMEMemoryTransform` keeps the three memory arrays that `GR00TTransform`
would otherwise drop when constructing its output dictionary. The stock
`DefaultDataCollator` already stacks fixed-size NumPy arrays, so no collator
change is needed. Both single-sample and batched transforms preserve memory,
including evaluation transforms without action labels.

For converted episode `e` and execution step `i`, the dataset reads the stored
`source_episode_index` and `source_frame_index` mapping. For example, converted
step zero might correspond to source frame 48 after a 48-frame conditioning
video. FrameSamp then samples only `[0,48]` from the original episode's cache,
retaining access to the conditioning video. Source indices are also used for
3D positional features. The current frame is included, matching upstream.

Before sample access, every selected converted episode is checked against its
cache record: integer provenance columns, complete execution suffix, demo
boundary, original episode/frame order, and execution state/action hashes.
Cache completion/version, encoder fingerprint, feature dimensions, pooling and
FPS must match. Truncated or relabeled execution trajectories are rejected;
the supported converter and task-extraction presets retain complete episodes.
This step checks dataset/cache alignment. Matching the cached encoder to the
trainable model's actual frozen vision weights belongs to training wiring in
step 5; the dataset exposes `memory_encoder_signature` and
`memory_encoder_sha256` for that check.

Feature arrays are opened lazily as read-only memmaps. A checksum is verified
once per process for each unchanged selected file. The loader retains at most
`max_open_episodes` mappings per process, reopens evicted arrays without hashing
the entire file again, and rechecks a file if its size/mtime changes. Spawned
workers serialize no open memmaps; forked processes detect a changed PID and
reopen inherited mappings. Call `dataset.close()` to release mappings explicitly.

Memory is attached in `get_step_data` rather than only `__getitem__`, because
`LeRobotMixtureDataset` calls `get_step_data` and then applies the dataset's
transforms. This prevents the mixture path from silently dropping history.

Use `spawn` for multiworker loading in this environment. During validation,
default `fork` workers crashed after earlier parent-process sample processing;
the same focused test passed in isolation, so the root native-library cause is
not established. The two-worker test and real-data smoke use `spawn`. A separate
forked-process test checks inherited memmap reopening without video decoding or
Torch DataLoader worker initialization. Step 5 should retain this process-context
choice when connecting the dataset to the trainer.

Run checks inside a CPU allocation with this checkout on `PYTHONPATH`:

```bash
python -m pytest -q \
    tests/test_robomme_memory_config.py \
    tests/test_robomme_feature_cache.py \
    tests/test_robomme_data_config.py \
    tests/test_robomme_memory_dataset.py
```

On 2026-11-23 in allocation 66728, source episodes 0 and 1001 were converted
into a small execution dataset and paired with the verified step-2 cache.
Their execution starts at source frames 48 and 284, respectively. All 332
execution samples passed initialization/alignment checks. A two-worker spawned
DataLoader produced memory `(2,512,2048)`, positions `(2,512,768)` and boolean
masks. Physical action round-trip checks passed. A second converted dataset
containing only episode 1001 yielded 284 samples from the same cache.
Artifacts: `outputs/robomme_memory_dataset_smoke_step3/smoke_summary.json`.

The optional model path added in step 4 is documented in
[MEMORY_MODEL.md](MEMORY_MODEL.md). Connecting this dataset/model pair to
`gr00t_finetune.py` remains step 5.
