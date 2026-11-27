# Sequence-aware expert replay

Unit 4 checks environment setup and episode accounting before connecting a GR00T
policy. It replays audited Raw controller commands, using the existing
`examples/Libero/eval/utils.py` environment, image and video helpers. The reusable
[`LiberoMemEpisodeProtocol`](episode.py) owns sequence accounting;
[`environment.py`](environment.py) owns benchmark-compatible episode setup;
[`replay.py`](replay.py) owns initial-state selection and the expert lifecycle.
This command does not load model weights or require a GPU with OSMesa.

## Episode semantics

The LIBERO-Mem fork can return `done=True` for an intermediate subgoal. Replay
therefore runs the entire expert program rather than terminating on `done`.
Before each action, the protocol calls the domain's `_check_success(inc=True)`
once, matching the author's `mem_5`/`mem_6` loops. The domain retains its original
timer thresholds, goal predicates, alternative sequences and overshoot checks.
Metric snapshots do not call `_check_success` or advance those timers.

Final `success` requires the complete ordered BDDL sequence, its final predicate
still holding, and no overshoot. `first_success_step` separately records the first
successful step: an episode that succeeds briefly and later overshoots ends as a
failure. Reset clears progress, timers and `_overshot`; the fork's own progress
reset omits the last flag. Settling steps are excluded from episode counters.

`subgoal_completion_rate` uses the number of sequence steps in the resolved BDDL.
These are not necessarily the paper's verbal action steps: T1 has one BDDL `on`
goal, while T3 has three repeated `on` goals. The fork's non-live/live timers
account for leaving and returning to the plate. Reports retain the resolved goal
alternatives and BDDL SHA256. Do not compare these denominators directly with a
different paper-level subgoal decomposition.

## Initial states and expert commands

- `--selection seen` selects Raw states whose exact initial-state fingerprint
  occurs in the same task's audited Full demonstrations. The default is the
  first such demo per task.
- `--selection unseen` checks that the final twenty Raw demos have no initial
  state in that task's Full set, then takes the last N, matching the author's
  last-N convention. Locally, N=1 selects `demo_120`. Other Raw-only demos are
  not automatically evaluation episodes.
- `--demo-ids` selects explicit source IDs within the chosen pool. `--task-ids`
  selects canonical T1–T10 IDs. Initial-state selection is independent of the
  proposal's train/held-out **task** folds.

The runner verifies source size/mtime, action and initial-state fingerprints, and
the recorded model XML hash against the unit-1 audit. After resetting the BDDL
environment, it saves the simulator's compiled XML and a copy of its flattened
state, reloads that XML, resets the simulator, and restores the saved state.
This matches `DeterministicInitWrapper._start_new_episode` in the benchmark's
`scripts/experiments/robot/libero/libero_utils.py`. It then checks state
dimensions, installs Raw `init_state`, settles for twenty open-gripper steps,
and resets sequence accounting. The recorded Raw XML is checked for identity;
the XML reloaded here comes from the freshly reset local simulator.

Following the source refinement code, replay removes leading actions with arm
norm below `1e-4`, maps gripper command 0 to -1, and appends twenty open-gripper
steps. Raw commands otherwise remain in simulator units; they do not pass
through the LeRobot gripper conversion. Replay never restores recorded future
states to force success. Physics differences can consequently change outcomes.

## Run on the CPU allocation

Run from the Isaac-GR00T root. Replace the job ID when the allocation expires.
The GR00T environment needs MuJoCo, PyOpenGL and an OSMesa system library; these
were available in the local environment used for validation.

For benchmark runs, match the upstream `scripts/requirements.txt` pin of
**MuJoCo 3.2.2**. The GR00T environment currently has 3.6.0. Local comparisons
installed 3.2.2 separately, leaving the environment's installed version intact:

```bash
/home/vhoangth2/miniconda3/envs/gr00t/bin/python -m pip install \
    --no-deps --target outputs/libero_mem_mujoco322 mujoco==3.2.2
```

The command below selects that copy through `PYTHONPATH`. The runtime
paths/version in `summary.json` confirm which copy was loaded. The separate
directory is an ignored local dependency, not part of the integration's source code.

```bash
srun --jobid=67857 --overlap --ntasks=1 --cpus-per-task=2 --mem=12G \
    env PYTHONPATH="$PWD/outputs/libero_mem_mujoco322:$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    /home/vhoangth2/miniconda3/envs/gr00t/bin/python \
    scripts/replay_libero_mem_expert.py \
    --libero-repo /mnt/data/vhoangth2/repos/libero-mem \
    --raw-root /mnt/data/vhoangth2/datasets/LIBERO-Mem-Raw \
    --audit outputs/libero_mem_source_audit_unit1_verified.json \
    --output-dir outputs/libero_mem_expert_unseen \
    --selection unseen --num-episodes 1
```

Add `--dry-run` to inspect selected task/demo IDs without importing the simulator
or creating output. Add `--task-ids 1 3` for a small check. Add `--save-videos` to
enable both 256×256 cameras and save combined 512×256 MP4s at 20 FPS, using the
existing LIBERO image orientation. Without rendering flags, per-step camera
observations are disabled. `--max-steps 8 --save-videos` is a truncated video
smoke check; `truncated: true` marks it in the report.

The process prepends the LIBERO-Mem checkout and its customized third-party
robosuite/robomimic to its import path. It refuses already imported modules from
other checkouts and records the resolved simulator paths/versions. A private
`OUTPUT/.libero/config.yaml` points at the fork's assets/BDDL and Raw data; the
global LIBERO configuration is not edited. Upstream has two different
`experiments/robot/libero/libero_utils.py` copies: the root copy contains a
deliberate exception, while the copy under `scripts/` used by the replay scripts
adds the deterministic reset wrapper. This runner uses the GR00T factory with
fork-specific options and reproduces that wrapper's reset sequence in
`initialize_episode`, which policy rollouts should also call.

Each fresh output directory contains:

- `summary.json`: configuration, `environment_reset` protocol, audit SHA256, runtime, source demo identities,
  BDDL SHA256, sequence progress, final success, overshoot and truncation.
- `Tn_demo_m.jsonl`: progress, reward and the original `environment_done` after
  every action. Intermediate `environment_done` is retained for diagnosis.
- `videos/`: optional combined camera videos.

Existing directories are refused. Summaries update after each completed episode;
simulator/setup errors mark the run `failed` and propagate instead of being
counted as task failures. `expert_replay_success_rate` describes this diagnostic,
not learned-policy performance. Truncated runs are smoke checks, not benchmarks.

## Local validation

Validation used CPU allocation `67857`, OSMesa, MuJoCo **3.2.2 / 3.6.0**, and the fork's
customized robosuite **1.4.1** at checkout
`0eee0defa4024e51511b6e15a79f30e9620209de`.

| Check | Result | Local report |
| --- | --- | --- |
| Original runner without XML round trip, seen `demo_1`, all ten tasks | 8/10 successful | `outputs/libero_mem_expert_seen_verified_unit4/summary.json` |
| Fixed reset, MuJoCo 3.6.0, seen `demo_1`, all ten tasks | 10/10 successful | `outputs/libero_mem_expert_seen_fixed_mj360/summary.json` |
| Fixed reset, MuJoCo 3.2.2, seen `demo_1`, all ten tasks | 10/10 successful | `outputs/libero_mem_expert_seen_fixed_mj322/summary.json` |
| Fixed reset, unseen T1/T8, eight rendered steps each | Both videos decode to eight frames, 20 FPS, 512×256; marked truncated | `outputs/libero_mem_expert_video_fixed/summary.json` |
| Episode/replay and existing rollout tests | 31 passed | `tests/test_libero_mem_episode.py`, `tests/test_libero_rollout.py` |

The original runner omitted the benchmark's compiled XML round trip. It failed
T7/demo_1 at 1/3 subgoals and T8/demo_1 after reaching 4/4 then overshooting.
Controlled comparisons kept the same Raw states, actions, waits, goal thresholds
and scoring logic:

| MuJoCo | Without compiled XML round trip | With compiled XML round trip |
| --- | --- | --- |
| 3.6.0 | T7 1/3; T8 overshot | T7 3/3 and T8 4/4, both successful |
| 3.2.2 | T7 1/3; T8 2/4 | T7 3/3 and T8 4/4, both successful |

The complete action programs match Full exactly for both demos (425 and 613
actions respectively). Changing the MuJoCo version alone did not resolve these
failures; matching the benchmark's reset sequence did. The comparison reports
are `outputs/libero_mem_reset_roundtrip/results.json`,
`outputs/libero_mem_reset_plain_mj322/results.json` and
`outputs/libero_mem_reset_roundtrip_mj322/results.json`. The old failed reports
remain intact. Dynamics still differ between MuJoCo versions, so use the upstream
pin when comparing benchmark results. These are expert diagnostics, not model
performance measurements or a guarantee that every Raw demonstration replays.

Tests cover intermediate `done`, ordered/alternative sequences, final predicates,
overshoot/reset, read-only metrics, audited pools and overlap rejection, expert
integrity, full-program execution, truncation, cleanup and failure reporting.
They also verify that the shared LIBERO helpers retain their existing defaults
while accepting fork options and a 20 FPS video override, and that setup reloads
compiled XML and preserves a copy of the saved state across simulator reset.

GR00T inference and HTTP policy rollout are the next unit; they should use
`initialize_episode` and the same episode protocol. Actual model inference will
require a GPU allocation.
