# GR00T policy rollouts

Unit 5 runs a reactive GR00T checkpoint through the existing HTTP inference
service and the LIBERO-Mem simulator. Episode initialization and sequence scoring
are the same as [expert replay](EXPERT_REPLAY.md), including the compiled XML
round trip, ordered subgoals and final overshoot checks.

## Shared components

- [`examples/Libero/adapter.py`](../Libero/adapter.py) contains the existing
  LIBERO image/state/action mapping. The original LIBERO evaluator also uses it.
  Camera images rotate 180 degrees; the state is EEF position, axis-angle
  orientation and both finger positions. The roll/pitch/yaw field names retain
  the existing LIBERO convention; their values are axis-angle components.
- [`gr00t/eval/http_client.py`](../../gr00t/eval/http_client.py) is the NumPy HTTP
  transport extracted from RoboMME, which also continues to use it. It calls the
  existing `/act` endpoint without retrying failed inference requests.
- [`selection.py`](selection.py) shares audited initial-state selection with
  expert replay. [`environment.py`](environment.py) shares runtime setup,
  environment creation and deterministic episode initialization.
- [`policy.py`](policy.py) converts physical prediction fields into simulator
  actions and owns the action queue. [`evaluation.py`](evaluation.py) runs and
  scores episodes, writes per-step logs and aggregates metrics.

The policy receives only current RGB images, proprioception and the canonical
task instruction. State/action normalization and inverse normalization remain
on the model server, using the checkpoint's training config and metadata.
Actions are packed as float32, matching the existing LIBERO evaluator.
The client maps GR00T gripper values with LIBERO's existing `1 - 2*g` formula
and binarization; it does not inverse-normalize the arm a second time.

## Start the existing model server

From the Isaac-GR00T root, use an active GPU allocation. This example checkpoint
is an existing LIBERO LOSO model suitable for checking the pipeline; benchmark
experiments require a model trained on the intended LIBERO-Mem fold.

```bash
srun --jobid=67873 --overlap --ntasks=1 --cpus-per-task=2 --mem=24G \
    env PYTHONPATH="$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 USE_TF=0 \
    /home/vhoangth2/miniconda3/envs/gr00t/bin/python \
    scripts/inference_service.py --server --http-server \
    --model-path outputs/libero_loso/20261023T163134Z/heldout_goal/checkpoint-13840 \
    --data-config examples.Libero.custom_data_config:LiberoDataConfig \
    --embodiment-tag new_embodiment --denoising-steps 4 --seed 0 \
    --host 127.0.0.1 --port 56873
```

Use the checkpoint's actual data config, embodiment tag and normalization
metadata. Future fold checkpoints should carry the fold's training statistics.
The evaluator connects to an externally started server; `--policy-label` records
the checkpoint identifier you supply. It does not inspect the remote checkpoint
or verify its training fold. The HTTP dependencies are the existing service's
`fastapi`, `uvicorn`, `requests` and `json-numpy`.

## Evaluate and select folds

In another terminal, on the same allocation/node:

```bash
srun --jobid=67873 --overlap --ntasks=1 --cpus-per-task=2 --mem=8G \
    env PYTHONPATH="$PWD/outputs/libero_mem_mujoco322:$PWD" \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    /home/vhoangth2/miniconda3/envs/gr00t/bin/python \
    scripts/eval_libero_mem.py \
    --raw-root /mnt/data/vhoangth2/datasets/LIBERO-Mem-Raw \
    --audit outputs/libero_mem_source_audit_unit1_verified.json \
    --output-dir outputs/libero_mem_policy_eval \
    --server-url http://127.0.0.1:56873 \
    --training-fold A --task-group held-out \
    --selection unseen --num-episodes 1 --n-action-steps 1 \
    --policy-label /path/to/fold_a/checkpoint --save-videos
```

The simulator uses OSMesa by default; RGB observations remain enabled even
without `--save-videos`, because the policy needs them. Use the isolated MuJoCo
3.2.2 installation described in [expert replay setup](EXPERT_REPLAY.md) to match
the upstream version. The model server can use its own Python environment.
Replace allocation IDs when needed. Both commands above execute on the same
node, so the localhost URL reaches the server.

Task-fold and initial-state selection are independent:

| Option | Meaning |
| --- | --- |
| `--training-fold A --task-group held-out` | T2, T4, T6, T8, T10 |
| `--training-fold B --task-group held-out` | T1, T3, T5, T7, T9 |
| `--task-group train` | Tasks in the specified training fold |
| `--task-group all` | All ten tasks; fold labels are recorded if supplied |
| `--task-ids 1 7` | Explicit subset within the chosen task group |
| `--selection unseen` | Audited final twenty Raw initial states absent from same-task Full |
| `--selection seen` | Raw initial states that appear in same-task Full |

Without `--training-fold`, use `--task-group all` (the default). Explicit tasks
outside the selected group are rejected. `--demo-ids` selects audited source
IDs in the chosen initial-state pool. `--dry-run` prints selections without
importing the simulator, connecting to the server or creating output.

For a short smoke test, use `--task-ids 1 7 --max-steps 16 --n-action-steps 8`
with a fresh output directory. Each episode is marked `truncated: true`.

## Actions, budgets and metrics

`--n-action-steps K` executes the first K actions of each predicted chunk before
querying again. The default K=1 matches the existing LIBERO evaluator. K must
fit the server's prediction horizon. At every episode boundary, the queue and
request counters are cleared; no action from the previous episode is reused.
The reactive HTTP service has no remote reset endpoint, and resetting the local
queue does not reseed the server's sampling RNG.

The default `--budget-mode benchmark` matches upstream `mem_6`'s horizon formula:
the maximum of that task's `TASK_LENGTHS` and its trimmed Raw expert program
length plus twenty tail steps. Raw actions are read and audited to determine
that horizon, then discarded; they are never policy inputs or executed in place
of policy predictions. No recorded future simulator state is restored.

`--budget-mode fixed` uses only `TASK_LENGTHS`, without reading expert actions.
The ten limits are `180, 285, 320, 365, 445, 650, 475, 670, 365, 390` action
steps. This is a different budget protocol and must be identified in comparisons.
`--max-steps N` caps either mode for smoke checks. Settling steps are separate
from the action budget, and the simulator horizon includes them.

Rollouts run their full configured budget even after an intermediate `done`,
temporary full success or overshoot, matching the benchmark's fixed rollout
loop. Final success requires the ordered sequence, final predicate and no
overshoot. `first_success_step` retains transient successes for diagnosis.

Each fresh output directory contains `summary.json`, `Tn_demo_m.jsonl` action
and score traces, a private `.libero/config.yaml`, and optional two-view MP4s at
20 FPS. The summary records source fingerprints, BDDL hashes, runtime versions,
budget settings, per-episode request counts and prediction horizons. It reports
success rate and mean BDDL subgoal completion by task, proposal category, memory
type and train/held-out role. Memory-type groups overlap because a task may have
several memory types. These are the fork's BDDL subgoals, not a separate
paper-level action decomposition.

Existing output directories are refused. HTTP, malformed prediction and simulator
errors mark the run `failed`, preserve completed results and the active episode
identity, close resources and propagate. Infrastructure errors are not counted
as completed task failures. Truncated results are pipeline checks, not benchmark
performance.

## Local validation

| Check | Result | Local report |
| --- | --- | --- |
| New policy/adapter tests and existing LIBERO/RoboMME regression tests | 98 passed | `tests/test_libero_mem_evaluation.py`, `tests/test_libero_mem_episode.py`, `tests/test_libero_rollout.py`, `tests/test_robomme_adapter.py` |
| Real GR00T HTTP inference on H100, allocation 67873, T1/T7 unseen `demo_120` | Both 16-step smoke episodes completed; two 16-action predictions per episode, first eight executed per query | `outputs/libero_mem_policy_gpu_smoke_unit5/validation.json` |
| Scripted expert through the real HTTP server and simulator, T7/T8 seen `demo_1` | All 475/670 executed actions exactly equal the expected float32 commands; videos and 60/84 request counts verified | `outputs/libero_mem_policy_cpu_smoke_unit5/action_validation.json`, `transport_validation.json` |

The GPU check used the existing LIBERO LOSO `heldout_goal/checkpoint-13840`,
`LiberoDataConfig`, `new_embodiment`, four denoising steps and seed 0. Both
episodes are explicitly truncated; their final success is false and is not a
LIBERO-Mem performance result. Each MP4 decodes to sixteen 512×256 frames at
20 FPS. The smoke server was stopped afterward.

The scripted CPU diagnostic preserves a failed success assertion: T7 completes
3/3 subgoals, but T8 reaches 3/4 rather than the expected 4/4. Every executed
action matches the float32-cast expert exactly, including chunk boundaries and
idle padding. The largest arm-command rounding difference from the original
float64 source is approximately `1.4e-8`. This diagnostic checks transport and
mapping, but cannot be treated as exact float64 expert replay equivalence.

A controlled direct-simulator comparison with the same T8 initial state and
670-step program reproduces the difference without HTTP: float64 completes
4/4, float32 reaches 3/4. The report is
`outputs/libero_mem_action_precision_unit5/results.json`. Expert replay keeps
the Raw source's float64 commands; policy rollouts retain the existing LIBERO
evaluator's float32 contract. This contact-sensitive example demonstrates why
expert replay success should not be assumed after rounding the commands.

Focused tests cover mapping parity with the original LIBERO adapter, camera
orientation, gripper conversion, finite/common-horizon action validation,
HTTP NumPy payloads and failures, chunk replanning and queue reset, independent
fold/state selection, complete-budget sequence accounting, overshoot, aggregation
and error cleanup. Existing LIBERO and RoboMME tests also exercise the extracted
shared code.

Training LIBERO-Mem fold models and adding memory/retrieval remain separate units.
