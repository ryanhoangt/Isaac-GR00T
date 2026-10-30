# RA-VLA training and LIBERO evaluation

## Training logs with Weights & Biases

Training decodes videos on demand by default (`--no-cache-video`), keeping an LRU
cache of at most `--video-cache-size 2` camera videos per dataset per worker.
It selects frames by the same integer indices as the original RAM cache,
avoiding rounding differences from timestamp-based seeking. It does not load or
create the datasets' `.cache/frames.pt` files. The recipe, dataset metadata,
augmentations and model are unchanged. `--cache-video`
opts into the original full-RAM cache; building it can use substantially more
memory than the final arrays because concatenation and serialization create
temporary copies. A 128 GiB allocation can run out of memory during this step.

`python -m scripts.rag.3_train_ravla` enables W&B logging by default. Add these
options to the training command to label a run explicitly:

```bash
--wandb-project icl-vla-uncertainty \
--wandb-name gos_seed42 \
--wandb-group libero_long_loso \
--wandb-mode online \
--log-freq 20
```

`--wandb-entity` selects a team or account. Authenticate with `wandb login` before
online training if the environment is not already logged in. Without an explicit
mode, W&B respects `WANDB_MODE`; `--wandb-mode offline` records locally and
`--no-wandb-enable` turns logging off. The project defaults to
`icl-vla-uncertainty` (override with `--wandb-project`), and the name defaults to
`WANDB_NAME` or the output directory name.

Step logs include total, MSE and margin losses, the learning rate used for that
update, and the gradient norm before clipping. They use optimizer `global_step`
as the x-axis and are emitted at the first step, every `log_freq` steps and the
end of each epoch. Epoch logs include mean losses, duration and peak allocated
GPU memory. Run config records the recipe and seed; summary records completed
epochs, steps and the last saved checkpoint path. Checkpoint weights are not
uploaded. `training_config.json` and the local W&B files are stored in `vla_path`.
The run is finalized on completion, errors and keyboard interruption. Training
itself still has no checkpoint-resume support.

## Evaluate the released RA-VLA checkpoints on LIBERO

For concurrent task evaluation, use one policy server and a bounded simulator
pool (defaults: five workers, 20 trials/task):

```bash
bash scripts/eval_ravla_parallel.sh --subset long \
  --checkpoint-path outputs/ravla_train/YOUR_RUN/checkpoint_3 \
  --workers 5 --trials 20 --seed 0 --k 1 --denoising-steps 4 \
  --port 8085 --run-name gos_seed42_long20_parallel
```

`--task-ids 0 3 --trials 1 --workers 2` is a small smoke test. `--dry-run`
validates paths and prints commands without loading models or creating outputs.
One process is started per task, up to the worker limit; slots take the next task
when one finishes. Every simulator owns its action queue, output directory, and
Numba/Hugging Face/Matplotlib caches. The policy server serializes requests using
the existing ZeroMQ REP protocol; it does not batch or concurrently invoke the
action head. The shared-server RPC timeout defaults to 120 seconds. Simulator
BLAS/OpenMP threads are limited to one, while environment seeds, initial states,
action horizon and task horizons retain the original evaluator's behavior.

The launcher verifies a unique server ID before starting workers, locks its port,
refuses existing output directories, and terminates its child process groups on
failure or interruption. It never adopts or kills another run's server. A failed
process aborts evaluation rather than being counted as a policy failure.
`tasks/taskNN/` holds each task's logs/videos; `progress.json` is atomically updated
with live episode counts. `rollouts/summary.json` aggregates validated completed
tasks; check `status: completed` before interpreting it as a full-suite result.
The aggregate rollout directory links to the original per-task videos with
unique episode numbers. Summary/progress files have a single parent writer.

Use `scripts/eval_ravla_libero.sh` from any working directory. Relative paths in
its arguments are resolved against the repository root. The policy and simulator run in separate
Python environments and communicate over a local ZeroMQ connection. The runner
starts the policy server, waits for a successful ping, runs the simulator, and
stops the server on completion or failure.

The policy environment needs the GR00T dependencies and `dtaidistance`. The LIBERO
environment needs LIBERO/robosuite, `tyro`, `pyzmq`, `msgpack`, `pydantic`, and video
encoding support. On Python 3.8, `eval-type-backport` is also needed for modern
type annotations. LIBERO's configured assets, BDDL files, and initial states must
exist. Demonstration datasets and training caches are not needed for evaluation
with the released memory buffers.

The Bash script defaults to `$HOME/miniconda3/envs/gr00t/bin/python` and
`$HOME/miniconda3/envs/libero/bin/python`. Override these with `--policy-python`
and `--libero-python`, or the environment variables below. The environments used
for the initial smoke run were:

```bash
export POLICY_PYTHON=/home/vhoangth2/miniconda3/envs/gr00t/bin/python
export LIBERO_PYTHON=/home/vhoangth2/miniconda3/envs/libero/bin/python
```

Evaluate one rollout on each of Goal tasks 0 and 1:

```bash
./scripts/eval_ravla_libero.sh \
  --subset goal \
  --task-ids 0 1 \
  --trials 1
```

Evaluate all ten Goal tasks with 50 initial states each:

```bash
./scripts/eval_ravla_libero.sh --subset goal --trials 50
```

The subset can be `goal`, `spatial`, `object`, or `long`. Full suite names such
as `libero_goal` and `libero_10` are also accepted. To use custom model files:

```bash
./scripts/eval_ravla_libero.sh \
  --subset goal \
  --checkpoint-path ra-vla-official-assets/ravla/ravla/libero_los/checkpoint_3 \
  --retriever-path ra-vla-official-assets/libero_los \
  --memory-name memory_g \
  --trials 50 \
  --run-name goal_custom \
  --output-dir outputs/ravla_eval/goal_custom
```

`--memory-name` selects a file inside the retriever directory. Alternatively,
pass `--memory-path /path/to/memory.pt` to use a memory stored elsewhere.
Checkpoint, retriever, and memory overrides are independent; choose compatible
files for your experiment. Omitting each override retains the default for the
selected subset.

Inspect the resolved paths and commands without loading models, rendering,
creating output directories, or using GPU:

```bash
./scripts/eval_ravla_libero.sh --subset object --trials 2 --dry-run
./scripts/eval_ravla_libero.sh --help
```

The runner defaults to `ra-vla-official-assets/` and automatically selects the
matching policy, retriever, and held-out memory:

| Evaluation suite | Checkpoint/retriever split | Evaluation memory |
| --- | --- | --- |
| `libero_spatial` | `libero_lgo` | `memory_s.pt` |
| `libero_object` | `libero_lgs` | `memory_o.pt` |
| `libero_goal` | `libero_los` | `memory_g.pt` |
| `libero_10` | `libero_gos` | `memory_l.pt` |

Policies are loaded from `ravla/ravla/libero_*/checkpoint_3/` under the assets
directory. Retrievers and memories are loaded from `libero_*/`. The runner uses
the provided buffers without rebuilding them. The defaults are policy seed 0,
K=1, and four denoising steps; override them with `--seed`, `--k`, and
`--denoising-steps`. The simulator retains the author's fixed seed 0. `--cuda`
selects the policy GPU index within `CUDA_VISIBLE_DEVICES`; use different `--port`
values when running multiple servers concurrently. It configures EGL rendering
and offline Hugging Face access. CUDA and EGL
must be available to the process; use a GPU allocation when running on a cluster.

For each run, `outputs/ravla_eval/<run-name>/` contains `server.log`, `eval.log`,
and `commands.json`. Its `rollouts/` directory contains `results.log`, rollout
videos, and `summary.json`. `--output-dir` replaces this run directory; an existing
directory is refused so previous results are preserved. Simulator or inference errors
abort the run instead of counting as policy failures. A small smoke run checks
the execution pipeline and does not estimate the paper's success rate.

The Python entry point `python -m scripts.rag.eval_libero` exposes the same options.
When running it directly, supply `--libero-python` explicitly. The standalone
evaluator `python -m examples.Libero.eval.run_libero_eval` still defaults to
`runs/gr00t/<suite>/<run-name>/` when no `--output-dir` is supplied.

## Pilot 1B: context-conditioned velocity disagreement

`gr00t/rag/context_vfd.py` implements a two-checkpoint `ContextVFDPolicy` scorer.
For the current LIBERO-Long pilot, use the released `libero_gos/checkpoint_3` as
the acting policy and the separately trained seed-42 checkpoint as the scoring
member, with `libero_gos/memory_l.pt`. Both were trained on Goal/Object/Spatial.
The scorer does not train, perturb weights, calibrate thresholds, or change the
acting policy to an ensemble-mean action.

For each raw observation it retrieves **one shared context**, computes the
current observation's VLM features once, and evaluates each head with its own
learned state, visual, retrieved-action and diffusion projections. Before sharing
features, it checks identical normalization metadata across actor, scorer and
retriever; matching configurations; identical backbone tensor hashes; and
different head hashes. Only the second action head is loaded, not another VLM.
The memory identity, model hashes, preprocessing source hashes, K, dimensions,
flow grid and score definition are recorded in `get_score_signature()`.

Unlike vanilla GR00T, RA-VLA's initial action chunk is the mean of the selected
normalized retrieved actions, not Gaussian noise. The actor alone integrates
this chunk with the original Euler updates. At the same intermediate action
and time, both heads predict velocity under the same retrieved context. The
score is the sum over the 16-step horizon and the **seven real action dimensions**
(padding dimensions 7–31 are excluded):

```text
U_ctx = sum_t [ t/(1-t) * delta_t * ||v_actor(x_t, o, H) - v_scorer(x_t, o, H)||² ]
```

Differences and accumulation use FP32. With four denoising steps the positive
score times are `0.25, 0.5, 0.75` and weights are `1/12, 1/4, 3/4`; t=0 has zero
weight and t=1 is never evaluated. `--flow-times` can select a subset of the
inference grid, with spacing to the next selected time or to 1, matching Part A
and [VFD Eq. 42](https://arxiv.org/html/2606.18043v2#A2.E42).
There is one deterministic trajectory per fixed observation/context; repeating
it does not create additional independent action samples. The Gaussian-source
KL derivation does not directly apply to RA-VLA's deterministic retrieved-action
source, so this adaptation is a **VFD diagnostic**, not a calibrated KL quantity.
Do not directly subtract its raw scale from vanilla `U_prior`.

### Score lossless observations without a rollout

From the repository root, using the `gr00t` environment on a GPU node:

```bash
python -m scripts.rag.score_context \
  --actor ra-vla-official-assets/ravla/ravla/libero_gos/checkpoint_3 \
  --scorer outputs/ravla_train/gos_seed42_20261026T211436Z/checkpoint_3 \
  --retriever ra-vla-official-assets/libero_gos \
  --memory ra-vla-official-assets/libero_gos/memory_l.pt \
  --observations /path/to/raw_observation.npz \
  --denoising-steps 4 --k 1 --check-parity \
  --output outputs/uctx/score_debug.json
```

Each NPZ uses raw `VLAPolicy` modality keys: `video.image` and `video.wrist_image`
as uint8 RGB `[1,H,W,3]` arrays in policy orientation; `state.x/y/z/roll/pitch/yaw`
as `[1,1]`; `state.gripper` as `[1,2]`; and
`annotation.human.action.task_description` as a Unicode `[1]` array. Save with
`np.savez(path, **raw_policy_observation)`. Object arrays are refused. MP4s alone
cannot reproduce the score because they do not contain robot proprioception
and introduce image compression. This command accepts multiple NPZ paths,
publishes complete JSON atomically, and refuses existing output files.

Results contain the actor's action chunk, `u_ctx`, per-flow-time squared and
weighted differences, selected memory-row indices, cosine similarities, task,
and score ID. `--check-parity` additionally checks exact original-actor action
parity, deterministic repeat scores, and zero disagreement for identical heads.
It is a debugging option that performs extra inference, not a rollout setting.
`--context-indices 17` fixes the context for a K=1 comparison; the Python API
also accepts a boolean `candidate_mask` for excluding verified memory rows.

### Atomic inference RPC

Add `--scorer-path /path/to/second/checkpoint_3` to
`python -m scripts.rag.4_serve_ravla`. It registers two optional endpoints:

```python
signature = client.call_endpoint("get_score_signature", requires_input=False)
result = client.call_endpoint("get_action_with_uncertainty", raw_observation)
action_chunk, u_ctx = result["action"], result["u_ctx"]
```

The latter returns the action and score from the **same retrieval and actor
trajectory**. The regular `get_action` endpoint keeps its original behavior.
The existing evaluators do not yet request or record this new endpoint; wiring
rollout traces and LODO calibration is the next step. In particular, the released
memory has no demo IDs or original frame IDs: logged memory-row indices identify
cached context entries, **not demonstrations**. Building reliable provenance is
required before claiming leave-one-demo-out retrieval or calibrated thresholds.

### Record a small on-policy pilot with context diagnostics

The evaluator can request `get_action_with_uncertainty` at each action-chunk
boundary and record the score with the returned actor actions. The optional
diagnostics endpoint additionally scores random and lowest-similarity contexts
from the **same task**, excluding the acting top-K rows. Alternative actions are
never executed. It computes retrieval/current-observation backbone features once
per query and reuses them for all contexts and both heads. Each context has its
own RA-VLA retrieved-action source and actor flow, so the comparison measures the
full context-conditioned policy, not attention alone. Random choices use a
private generator seeded from the query hash; worker scheduling does not affect
the selection or the actor RNG.

For the released Long checkpoint, task 3 init 0/1 previously failed/succeeded and
task 5 init 0/1 succeeded/failed. This four-episode pilot deliberately selects
outcomes for debugging, not for estimating SR or failure detection accuracy:

```bash
bash scripts/eval_ravla_parallel.sh --subset long \
  --scorer-path outputs/ravla_train/gos_seed42_20261026T211436Z/checkpoint_3 \
  --task-ids 3 5 --trials 2 --workers 2 --seed 0 --k 1 --denoising-steps 4 \
  --record-uctx --context-diagnostics --save-observations \
  --port 8087 --run-name uctx_long_pilot4

python -m scripts.rag.analyze_context_rollouts \
  outputs/ravla_eval/uctx_long_pilot4 \
  --baseline-run outputs/ravla_eval/long_20trials --filmstrips
```

Each task's rollout directory contains `uncertainty/taskNN_initMMM/` with
incremental `queries.jsonl`, lossless raw observation NPZs (when requested), and
`episode.json`. Metadata remains `complete: false` after interruption; analysis
excludes such episodes. Raw observations are saved only at action-generation
steps, every 16 simulator control steps, excluding the 10 warm-up steps. JSONL
includes the executed action chunk, all three context selections, per-flow and
per-action-dimension score contributions, and alternative normalized action
distances. Each episode has one writer and existing episode directories are
refused. Standalone evaluation exposes the same three recording flags.

Analysis saves PNG/PDF traces, context distributions, dimension contributions,
context image contact sheets, raw `queries.csv`, `metrics.json`, and `report.md`.
The optional baseline comparison checks outcomes, lengths, and sampled decoded
video frames against the old run. Context aggregates weight episodes equally;
time-correlated queries are not treated as independent experimental repeats.
Low similarity/random are **distractor proxies**, not verified phase-mismatch
labels; the retrieved context is not guaranteed to be aligned either. Therefore
this pilot reports paired changes, not phase-alignment AUROC or calibrated
failure thresholds. A timeout is an episode outcome, not a failure-onset label.
