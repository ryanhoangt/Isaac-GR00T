# Serve and evaluate RA-VLA on RoboMME

Reuse `scripts.rag.4_serve_ravla` and `examples.RoboMME.eval`. RA-VLA uses the
existing ZeroMQ transport; select `--policy-backend ravla` and a `tcp://` URL.
The HTTP GR00T backend remains the default, including its existing run/resume
configuration format. No benchmark-repository changes are required.

## Start the current primary checkpoint

From this checkout, in one terminal:

```bash
cd /mnt/data/vhoangth2/repos/Isaac-GR00T-RA-VLA

srun --jobid=67148 --overlap --nodes=1 --ntasks=1 --cpus-per-task=2 \
  --gres=gpu:1 \
  env PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 \
    USE_TF=0 USE_FLAX=0 HF_HUB_OFFLINE=1 \
    NO_ALBUMENTATIONS_UPDATE=1 TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/vhoangth2/miniconda3/envs/gr00t/bin/python -u \
    -m scripts.rag.4_serve_ravla \
    --checkpoint-path outputs/ravla/robomme_primary_3epochs_seed42/checkpoint_1 \
    --retriever-path outputs/retriever/robomme_primary_1000_seed42 \
    --memory-name memory_primary_seed42 \
    --host 127.0.0.1 --port 8071 --seed 42 \
    --gpu-memory-fraction 0.25
```

Data config and canonical/instruction key type come from checkpoint config;
legacy checkpoints fall back to LIBERO instruction keys. Normalization comes
from the policy checkpoint. Canonical serving validates it against the frozen
retriever's training metadata, and verifies the memory manifest, retriever
fingerprints, candidate counts, key type and data config. Separate held-out
support banks are allowed; they need not contain the training task keys.

`--gpu-memory-fraction` is optional and limits this server's PyTorch allocator.
The example cap was tested while training shared the H100. K=1 and four
denoising steps are defaults. `server_info` exposes checkpoint/retriever/memory
identity, task/demo counts, a source fingerprint and peak VRAM.

The primary memory above contains **12 training-task banks only**. It can be
used for seen-task plumbing checks, but held-out evaluation will reject it
before loading the simulator because the four required banks are absent.

## Client environment

The simulator remains in the existing Python 3.11 `robomme` environment; the
model server uses the Python 3.10 `gr00t` environment. The new client imports
NumPy, PyZMQ, msgpack and Pydantic, with no Torch/model dependency.

This session installed the missing PyZMQ wheel in an isolated, ignored repo
cache. On a fresh checkout, install it once inside an allocation:

```bash
srun --jobid=67148 --overlap --nodes=1 --ntasks=1 --cpus-per-task=1 \
  /home/vhoangth2/.local/share/mamba/envs/robomme/bin/python -m pip install \
    --no-deps --only-binary=:all: --no-compile \
    --target "$PWD/.cache/robomme_ravla_eval_deps" pyzmq==27.1.0
```

The existing simulator environment already provides the other dependencies.
Use the cache path on `PYTHONPATH` for client/evaluator commands below.

## Held-out evaluation after building support memory

Prepare a separate full support memory with three demos per held-out canonical
task (PickXtimes, ButtonUnmaskSwap, PickHighlight, PatternLock for primary).
Use the same trained retriever and primary training normalization; select
support demonstrations separately from evaluation-query episodes. The existing
[memory builder](BUILD_MEMORY.md) supports a task filter and the portable all16
task map. Building those held-out banks is the next experiment-preparation step;
this serving integration did not encode them.

Restart the server with `--memory-name memory_primary_heldout_seed42` after that
file and its manifest exist. Then, in another terminal:

```bash
cd /mnt/data/vhoangth2/repos/Isaac-GR00T-RA-VLA

srun --jobid=67148 --overlap --nodes=1 --ntasks=1 --cpus-per-task=2 \
  --gres=gpu:1 \
  env PYTHONPATH="$PWD:$PWD/.cache/robomme_ravla_eval_deps" \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/vhoangth2/.local/share/mamba/envs/robomme/bin/python -u \
    -m examples.RoboMME.eval \
    --policy-backend ravla \
    --manifest /mnt/data/vhoangth2/datasets/robomme_primary_train_v21/manifest.json \
    --checkpoint "$PWD/outputs/ravla/robomme_primary_3epochs_seed42/checkpoint_1" \
    --benchmark-root /mnt/data/vhoangth2/repos/robomme_benchmark \
    --server-url tcp://127.0.0.1:8071 \
    --output-dir outputs/robomme_ravla_eval/primary_epoch1_heldout_seed42 \
    --num-episodes 1 --support-demos-per-task 3 \
    --exec-horizon 16 --max-steps 1300 --seed 42 --timeout 120
```

This selects all four excluded tasks. Increase `--num-episodes` to 50 for the
proposal budget; use `--tasks` to select a held-out subset. `--dry-run` checks
the manifest/test-episode metadata and prints the plan without connecting to a
server or simulator. `--resume` requires an identical plan and remote source
fingerprint; changing checkpoint, memory, K or denoising settings starts a new
run. Server PID and transient runtime counters are excluded from that identity.

Every request carries canonical task identity independently of `task_goal`.
The server removes the lookup metadata before visual/language transforms and
retains the actual instruction in the model input. The adapter uses only the
latest two RGB views, seven joint positions and first physical finger position,
then packs the single inverse-transformed prediction into float32 `(16,8)`
absolute joint/gripper actions. It adds no clipping, delta conversion or gripper
threshold. Episode reset clears no server RNG/state; the evaluator owns the
action queue. Inference is not retried after failures/timeouts.

## Validation

84 CPU tests passed in allocation 67148, including real ZeroMQ round trips,
canonical policy routing, unchanged-language/physical-action mapping, source
validation, preflight failure, evaluator task propagation and resume identity.
The existing HTTP adapter/evaluator tests also passed.

Checkpoint `robomme_primary_3epochs_seed42/checkpoint_1` was served on the GPU
with the existing primary memory. Real VideoUnmask/VideoUnmaskSwap observations
sharing one instruction returned finite `(16,8)` chunks through the network.
A MoveCube test episode then executed four simulator steps through the adapter,
with two-step replanning and no errors. Its `timeout` result is the deliberate
four-step smoke limit, not a task score. Peak server VRAM was 14.67 GiB allocated
and 14.71 GiB reserved. The test server stopped afterward; training was not stopped.

Artifacts: `outputs/robomme_ravla_smoke/checkpoint1_20261125T073551Z/`, including
`summary.json`, `server_info.json`, `rpc_predictions.json`, logs and
`MoveCube_smoke.mp4`. No full or held-out rollout evaluation was performed.
