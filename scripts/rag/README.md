# Evaluate the released RA-VLA checkpoints on LIBERO

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
