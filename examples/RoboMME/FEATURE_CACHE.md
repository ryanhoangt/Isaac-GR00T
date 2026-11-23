# Frozen visual memory cache (step 2)

The extractor reads the **original** local RoboMME LeRobot v2 image-parquet
dataset. It keeps every frame, including the conditioning video before robot
execution. Do not point it at the execution-only GR00T conversion.

Edit [`config/cache_visual_features.yaml`](config/cache_visual_features.yaml)
for paths, batch size, compute/storage dtype and attention implementation.
[`config/perceptual-framesamp-modul.yaml`](config/perceptual-framesamp-modul.yaml)
controls the memory representation, including mean/max pooling. CLI options
override YAML values; relative paths are relative to the working directory.

Run from this checkout inside a compute allocation using the GR00T environment.
Set `PYTHONPATH` explicitly because this environment has another editable GR00T
checkout installed. These commands neither submit jobs nor create allocations:

```bash
export PYTHONPATH=/mnt/data/vhoangth2/repos/Isaac-GR00T
export USE_TF=0 USE_FLAX=0 HF_HUB_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ROBOMME_PYTHON=/home/vhoangth2/miniconda3/envs/gr00t/bin/python

# Metadata only: no model loading, decoding or output writes.
"$ROBOMME_PYTHON" scripts/cache_robomme_visual_features.py --dry-run

# Small GPU check. Use a new output directory.
"$ROBOMME_PYTHON" -u scripts/cache_robomme_visual_features.py \
    --episode-indices 0 1001 --batch-size 4 \
    --output outputs/robomme_memory_features_check

# Full cache; omit episode-indices to select all 1,600 source episodes.
"$ROBOMME_PYTHON" -u scripts/cache_robomme_visual_features.py \
    --config examples/RoboMME/config/cache_visual_features.yaml

# Continue exactly the same episode selection/output after an interruption.
"$ROBOMME_PYTHON" -u scripts/cache_robomme_visual_features.py \
    --config examples/RoboMME/config/cache_visual_features.yaml --resume
```

The full source has 768,897 frames. At 16 tokens/frame, 2,048 channels and
float16 storage, feature payloads occupy 50,390,433,792 bytes (46.93 GiB), plus
small NumPy headers, configs and manifests. Only a batch of decoded images and
one episode's parquet are held in RAM. Output `.npy` files support memory-mapped
reads. NumPy has no native bfloat16 storage, so disk dtype is float16 by default;
the vision computation still uses bfloat16. Float32 disk storage is also supported.

Each cache contains:

```text
manifest.json
memory_config.yaml             # resolved representation config at creation
cache_config.yaml              # resolved extraction config at creation
episode_000000/                # original source episode ID, never remapped
  features.npy                 # [all_source_frames,16,2048], source frame order
  record.json
episode_001001/
  features.npy
  record.json
```

The loader constructs the repository's Eagle model with its existing SigLIP
vision encoder/connector and a parameter-free language placeholder. It reads
only `backbone.eagle_model.vision_model.*` and `backbone.eagle_model.mlp1.*`
tensors from a **local safetensors checkpoint**, checks all required tensors,
and freezes both modules. No language or action model weights are loaded.
Pooling follows Eagle's existing `extract_feature`: vision tokens pass through
the connector, then their spatial grid is reduced to 4x4 in float32. Mean
pooling follows the released RoboMME code; max pooling is configurable.

The shared `preprocess_memory_images` function resizes RGB with bilinear
antialiasing, preserves aspect ratio with centered black padding to 224x224,
and normalizes to [-1,1]. Memory features use deterministic preprocessing;
there is no crop/jitter or multi-tile image expansion on this branch. This
function is intended for the online memory policy too. Current-observation VLM
augmentation remains in the existing data config. Positional embeddings and the
trainable memory projection are **not** cached; step 1 constructs positions
from selected source frame indices.

The manifest records source metadata hashes, selected episode IDs, extraction
settings, the vision/connector weight digest, model/implementation identity and
preprocessing. Each episode record adds the source parquet and local-image
hashes, demo boundary, execution state/action hashes, feature shape/dtype and
feature-file checksum. The execution hashes will allow step 3 to check alignment
against the converted training dataset. No oracle annotations are read.

Each episode is published only after complete extraction and verification.
`verify: true` re-encodes its first, middle and last frames in their original
batch context and compares them to disk features, allowing disk-quantization
error. A writer lock prevents concurrent writes to the same cache. Resume
checks source/config/encoder identity and existing episode checksums before
reusing them. A failed partial episode is re-extracted, while completed episodes
are retained. `manifest.json` has `status: complete` only after all selected
episodes finish. Changing the selected episode set requires a different output.

Sampling budget, positional settings and DiT width do not alter frozen visual
tokens, so they can be changed later without re-extraction. Changing the
vision/connector weights, preprocessing, pooling or compute/storage precision
requires a separate cache. Keep `tune_visual=False` when training with this
cache. Both all16 and primary can read one frozen all-task cache, provided step 3
selects samples exclusively from the chosen training dataset; cache extraction
performs no weight updates or dataset-wide feature normalization.

Run focused tests inside a CPU allocation:

```bash
"$ROBOMME_PYTHON" -m pytest -q \
    tests/test_robomme_memory_config.py tests/test_robomme_feature_cache.py
```

Tests cover source/demo alignment, causal sampling from cache, interruption and
resume, stale encoder/source rejection, corruption checks, deterministic image
preprocessing and spatial pooling. A tiny real Eagle/SigLIP model checks the
selective safetensors loader against the original forward path on CPU.

On 2026-11-23, H100 smoke extraction in allocation 66728 processed source
episodes 0 and 1001: 96 and 568 frames, including 48 and 284 demo frames.
The cache includes all 664 frames. Per-episode cached/direct verification passed.
Artifacts: `outputs/robomme_memory_cache_smoke_step2_verified/`. The step-3
dataset adapter is described in [MEMORY_DATASET.md](MEMORY_DATASET.md).
Full-dataset extraction and trainer/inference integration remain subsequent work.
