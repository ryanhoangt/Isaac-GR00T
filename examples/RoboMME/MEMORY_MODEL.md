# Optional memory modulation in GR00T (step 4)

The GR00T action head now supports the FrameSamp+Modul memory architecture.
Its configuration is the resolved contents of
[`config/perceptual-framesamp-modul.yaml`](config/perceptual-framesamp-modul.yaml).
With no memory configuration, there are no additional model parameters and the
existing vanilla train/inference path is retained.

Enable memory **after** loading the vanilla pretrained model and **before**
creating its optimizer/DDP wrapper:

```python
from examples.RoboMME.memory_config import load_memory_config
from gr00t.model.gr00t_n1 import GR00T_N1_5

model = GR00T_N1_5.from_pretrained(
    "/path/to/local/GR00T-N1.5-3B",
    tune_visual=False,
    tune_llm=False,
)
model.enable_memory(load_memory_config())
```

`enable_memory` adds modules in place, preserving every original parameter
object and value. Calling it again with the same config is a no-op; a different
config raises an error to avoid reinitializing learned adapters. Use the top-level
model method so `model.config.action_head_cfg["memory_config"]` is updated too.
The YAML content is stored directly in the model config rather than as a path
dependency. A saved memory checkpoint reconstructs its modules on load.

Both `model(batch)` and `model.get_action(batch)` consume the same three fields
provided by the [step-3 dataset](MEMORY_DATASET.md), in addition to the existing
GR00T inputs:

| Field | Default batched shape | Type |
| --- | --- | --- |
| `static_image_emb` | `(B,512,2048)` | floating tensor |
| `static_pos_emb` | `(B,512,768)` | floating tensor |
| `static_mask` | `(B,512)` | bool; True means valid |

Every sample needs at least one valid token. Wrong dimensions, missing fields,
nonboolean/empty masks and nonfinite valid features fail explicitly. Masked raw
features are cleared before projection so padded values cannot affect the
result. Supplying memory fields to a vanilla head also raises an error instead
of silently ignoring them.

`MemoryFeatureEncoder` follows the policy-learning repo: positional features
pass through Linear+SiLU, are concatenated with visual features, then projected
to the DiT inner width. This projection is computed once per training call or
inference request, then reused across all denoising steps.

At every DiT block, after its existing attention/residual and before the existing
FFN normalization, memory is applied as follows:

```text
r = MemoryAttention(query=x, key=memory, value=memory, mask=mask)
scale, shift = Linear(r)
x_mem = RMSNorm(x) * (1 + scale) + shift
x_out = x + existing_FFN(existing_norm3(x_mem))
```

The residual uses the original `x`. Both self-attention and cross-attention DiT
blocks receive memory. GR00T's existing timestep AdaLN before attention, current
VLM attention and flow-matching objective are retained; memory is not concatenated
with VLM features. State, future and noised-action tokens all participate as
queries in this action-expert sequence.

The added attention follows `mme_vla_suite/models/integration/history_gemma.py`:
shared learned RMSNorm scale for query/context, one KV head, split-half 1D RoPE,
and separate output/modulation projections. Key RoPE positions span the complete
padded memory sequence, with query positions immediately afterward, matching
the upstream implementation. Normalization arithmetic uses float32, then casts
back to the activation dtype. Attention masking uses True for valid tokens.

The default modulation projection uses normal initialization with standard
deviation 0.002 and zero bias, as in the released code. This is **not exact
identity initialization**: enabling memory introduces an additional RMSNorm even
if its scale/shift projection were zero-initialized. Exact vanilla parity applies
to the disabled-memory configuration.

GR00T's inner DiT width is 1536; the YAML therefore uses six query heads of
dimension 256, compared with four heads at width 1024 in pi0.5. Across 16 DiT
layers and the feature encoder, this adds 168,569,088 parameters. The feature
encoder and modulation modules follow the existing `tune_diffusion_model` flag;
the state/action projection flag retains its existing role.

Validation includes explicit attention/RMSNorm/RoPE math, padding invariance and
zero padded gradients, nonzero gradients in all memory parameter tensors,
memory sensitivity under fixed sampling noise, training batch expansion and
single encoding per multi-step inference call. Tiny model tests check strict
weights/config restoration, Diffusers `DiT.from_config`, and complete GR00T
`save_pretrained`/`from_pretrained` with a lightweight test backbone.

An independent comparison against the pre-change source confirmed exact vanilla
state-dict keys/values, training loss and inference outputs with fixed seeds.
An H100 bfloat16 smoke loaded the real 1,068,806,144-parameter pretrained action
head and added all 16 memory modules. Real cached RoboMME episode-0 memory,
state and action labels produced finite flow loss (0.52756), finite/nonzero
gradients for all 100 new parameter tensors, and inference shape `(1,16,32)`.
Only new memory parameters required gradients in this smoke. Current VLM
features were synthetic to isolate the action-head path; this was not a full
backbone/trainer run or a policy-performance evaluation. Peak allocated GPU
memory was 2,928,743,424 bytes.

Artifacts: `outputs/robomme_memory_model_smoke_step4/smoke_summary.json`.
CLI training integration, encoder/cache weight matching and full-model training
smoke remain step 5. The stateful online memory policy remains step 6; this step
accepts already-prepared memory tensors. TensorRT deployment is not extended.

Run focused model checks inside a CPU allocation with this checkout on
`PYTHONPATH`:

```bash
python -m pytest -q tests/test_robomme_memory_model.py
```
