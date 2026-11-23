# Stateful FrameSamp+Modul inference (step 6)

`MemoryActionPolicy` loads a memory checkpoint and the normal GR00T
normalization/inverse-action transforms. `MemoryEpisodeController` retains pooled
visual features for the active episode. `RoboMMEMemoryClient` exposes the
benchmark's `reset()` / `infer(inputs)` interface over a dedicated HTTP server.

This follows `robomme_policy_learning/challenge_interface/policy.py`: append
every newly observed front-camera frame, including demo frames on the first
request, and use the latest front/wrist RGB, robot state and task instruction
for the current VLM input. Subsequent requests carry only the frames observed
since the previous request. Historical state is not used by this memory config.
Instructions retain the existing GR00T adapter/training convention.

The server uses the checkpoint's frozen Eagle/SigLIP vision tower and connector,
the exact cache preprocessing/pooling functions, and the same storage precision
before FrameSamp. Source time indices start at zero for the first demo frame;
the current frame is included. Sampling, 3D positions, zero padding and valid
masks use the training utilities without a separate online implementation.
Equal images are retained as separate timesteps.

All pooled frame features stay in CPU memory because uniform sampling can select
previously unselected frames later. Only the configured token budget enters the
model: normally 32 frames × 16 tokens = 512 tokens. The default episode cap of
4096 frames bounds the float16 history buffer to 256 MiB (plus request and
encoding temporaries); exceeding the cap raises an error instead of truncating
history. The cap includes conditioning-video frames and can be edited in YAML.

## Start the server

Run inside the **existing GPU allocation**, preserving `CUDA_VISIBLE_DEVICES`:

```bash
export PYTHONPATH=/mnt/data/vhoangth2/repos/Isaac-GR00T
export USE_TF=0 USE_FLAX=0 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export NO_ALBUMENTATIONS_UPDATE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

/home/vhoangth2/miniconda3/envs/gr00t/bin/python -m examples.RoboMME.serve_memory \
    --model-path outputs/robomme_memory_primary_seed42/checkpoint-20000 \
    --config examples/RoboMME/config/memory_inference.yaml
```

The model path is an example for a future trained checkpoint. Runtime settings
are in [memory_inference.yaml](config/memory_inference.yaml): host, port, device,
seed, denoising steps, encoding batch size and episode cap. Corresponding CLI
flags override YAML. The default URL is `http://127.0.0.1:8000`.

Architecture, visual weights, extraction precision and preprocessing come from
the checkpoint. The server checks these before casting the model. Device type,
library versions and attention implementation must match the training cache.
The extraction encoder is shared with the current VLM, so it loads no second
vision model. It encodes only newly received frames.

New training checkpoints embed the complete extraction specification in
`experiment_cfg/memory_training.json`; serving does not require cache feature
files. Earlier step-5 checkpoints use their original cache manifest for the same
validation. If that path moved, pass `--cache-manifest /path/to/manifest.json`;
its checksum must match the training record. This sidecar extension leaves the
existing resume identity unchanged. The full cache being extracted does not
need to be regenerated.

## Client and episode lifecycle

In the simulator environment (NumPy, requests and json_numpy suffice):

```python
from examples.RoboMME.memory_client import RoboMMEMemoryClient

with RoboMMEMemoryClient("http://127.0.0.1:8000", timeout=120) as policy:
    policy.reset()  # once per episode, before inference
    result = policy.infer(inputs)  # benchmark buffers; is_first_step=True
    actions = result["actions"]  # physical float32 [16, 8]
    # Execute the chosen horizon, collect all new observations, then call infer
    # with only these new buffers and is_first_step=False.
```

The rollout runner owns the action queue/execution horizon. Demo observations
only condition the policy; they are not executed as predicted actions.

`POST /reset` with `{}` clears history and returns a fresh episode token.
`POST /act` carries that token, a request counter and the start index of the new
frames. The server serializes reset/inference and rejects stale tokens,
repeated/out-of-order requests, inconsistent first-step flags, a changed task
instruction and a mismatched current front frame. Successful replies acknowledge
the request and total observed frame count. `GET /health` identifies this protocol.

There is **one active episode per server**. Parallel rollouts need separate
servers. A second reset invalidates the first client's episode token. A model
or encoding failure requires reset; the client also invalidates its token after
an ambiguous HTTP failure. Calls are never automatically retried. Reset does
not reseed diffusion sampling; the configured seed is applied once at startup,
matching the existing reactive server's episode-level RNG behavior.

The existing reactive adapter, inference service and HTTP API are unchanged.
Selecting this client in the benchmark runner and defining all16/held-out eval
recipes is step 7; this step supplies the complete serving/client interface.

## Validation

`tests/test_robomme_memory_inference.py` checks incremental history against the
offline representation, float16 quantization, demo prefill, duplicate frames,
reset isolation, input/order errors, transactional encoding, failed requests,
real HTTP JSON/NumPy round trips, and actual tiny SigLIP/GR00T checkpoint
load/encode/action inference. It also checks embedded provenance and compatibility
with old manifests. Run alongside the earlier RoboMME memory tests.

The bundled Eagle module imports CUDA-specific RADIO code even for a tiny CPU
model; use the existing GPU allocation for these checkpoint tests. The tiny
model's inference itself runs on CPU.

On 2026-11-23, the real GR00T step-5 smoke checkpoint served three HTTP inference
calls inside existing allocation 66728: 48 demo frames + current frame, then 16
new execution frames, followed by reset and a one-frame episode. All returned
finite physical actions `[16, 8]`; reset cleared the history. Peak allocated
VRAM was 6,092,758,528 bytes (5.67 GiB).

Positions and masks matched the offline cache exactly. With online encoding
batch 16 versus the smoke cache's batch 4, the 49-frame request differed only
at its last frame, encoded alone online: maximum feature error 0.03076 and
relative RMS error 0.1755% across the selected memory. Re-encoding that frame
in its original offline batch context matched the cache exactly. At 65 frames,
all selected features matched exactly. BF16 results can vary with batch context;
bitwise equality across arbitrary online batch sizes is not assumed. This
diagnostic used future frames only for the separate offline comparison, never
as inputs to online policy inference.

Artifacts: `outputs/robomme_memory_inference_smoke_step6/summary.json`.
The checkpoint is an integration smoke, not a trained benchmark baseline.
