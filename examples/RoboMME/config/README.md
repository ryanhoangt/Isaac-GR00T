# FrameSamp+Modul configuration

`perceptual-framesamp-modul.yaml` configures the memory-only RoboMME baseline.
It follows the local policy-learning implementation, with the action width
adapted from pi0.5's 1024 to GR00T's DiT inner width of 1536 (six query heads,
one KV head, 256 channels/head).

The default memory budget is 512 tokens: 32 front-view frames, each pooled to
4x4 tokens. `pool_type: mean` follows the released code/config; the paper
describes max pooling, available here as an explicit configuration choice.
Historical proprioception and additional camera views are not part of this
baseline. Current observations and action labels keep their existing GR00T
data configuration.

The sampling/configuration utilities can be used independently of Torch:

```python
import numpy as np

from examples.RoboMME.memory_config import load_memory_config, prepare_frame_sampling

config = load_memory_config()  # or pass a custom YAML path
features = np.load("episode_features.npy", mmap_mode="r")
# features: [all_episode_frames, 16, 2048], including conditioning-video frames
memory = prepare_frame_sampling(features, step_idx=100, config=config)
```

`step_idx` is the current frame's index in the complete source episode. The
sampler retains all observed frames while they fit, then samples evenly over
`[0, step_idx]` using the same integer rounding as RoboMME's
`shared/data_utils.py:even_sampling_indices`. Sampling does not reserve a recent
window. Right-padding produces these fixed-size fields:

| Field | Default shape | Type |
| --- | --- | --- |
| `static_image_emb` | `(512, 2048)` | float32 |
| `static_pos_emb` | `(512, 768)` | float32 |
| `static_mask` | `(512,)` | bool; True means valid |

Position features match `shared/posemb_3d.py:PosEmb3D`: concatenated sin/cos
features for source time and 4x4 spatial grid centers at 2, 6, 10, and 14.
These are separate from the 1D RoPE inside the planned memory-attention module.
The sampler reads only selected frames up to `step_idx`; it does not read future
features, mutate the cache, or duplicate frames to fill padding slots.

This YAML and its NumPy utilities do not activate memory in the existing trainer.
For the frozen visual extractor added in step 2, see
[`FEATURE_CACHE.md`](../FEATURE_CACHE.md). Model modulation and training/inference
wiring are subsequent changes.

Run the focused checks inside a CPU allocation with the GR00T environment:

```bash
PYTHONPATH=. python -m pytest -q tests/test_robomme_memory_config.py
```
