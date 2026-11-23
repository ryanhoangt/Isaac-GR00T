# Memory benchmark evaluation (step 7)

`python -m examples.RoboMME.eval` now selects the reactive or stateful memory
client. The memory client calls the server's reset before every episode, sends
demo + current frames on the first inference and only newly observed frames
thereafter. The runner owns the action queue and executes `exec_horizon` actions
before replanning; it stops immediately on episode termination. Current
front/wrist RGB and physical state/action mapping are unchanged.

## Protocol files

| YAML in `config/` | Training data | Default test evaluation | Reporting |
| --- | --- | --- | --- |
| `eval_memory_all16.yaml` | All 16 tasks | 16 tasks × 50 episodes | 16 seen; no held-out tasks |
| `eval_memory_primary.yaml` | 12 tasks | 16 tasks × 50 episodes | 12 seen and 4 held-out separately |

Both presets also report `primary_comparison`: ButtonUnmaskSwap, PatternLock,
PickHighlight and PickXtimes. These four tasks are **seen for all16** and
**held-out for primary**. Compare this group between the two checkpoints using
the same test episodes, seeds and execution horizon. The comparison group
overlaps the seen/held-out groups; do not sum their counts.

The all16 training set uses the existing indexed, execution-only dataset. The
task index must match its source metadata and cover every training episode once.
Primary uses `meta/episode_tasks.json` generated during extraction and verifies
that exactly the expected 12 tasks are present, with the expected 4 excluded.
Instruction IDs are not treated as environment task IDs. Neither protocol
recomputes normalization statistics or opens the memory cache for evaluation.

YAML keeps paths, client/protocol selection, task group, test episode count,
execution horizon, step budget, local/server seed, denoising steps, HTTP timeout
and video settings together. CLI flags override YAML. Relative paths resolve
from the working directory; run from this checkout. `checkpoint: null` requires
an explicit `--checkpoint`, avoiding an accidental default checkpoint.

## Run

Start the [memory server](MEMORY_INFERENCE.md) in the existing GPU allocation,
using the trained checkpoint's own normalization metadata:

```bash
cd /mnt/data/vhoangth2/repos/Isaac-GR00T
export PYTHONPATH=/mnt/data/vhoangth2/repos/Isaac-GR00T
export USE_TF=0 USE_FLAX=0 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export NO_ALBUMENTATIONS_UPDATE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

ROBOMME_MEMORY_CHECKPOINT=/path/to/trained/primary/checkpoint
/home/vhoangth2/miniconda3/envs/gr00t/bin/python -m examples.RoboMME.serve_memory \
    --model-path "$ROBOMME_MEMORY_CHECKPOINT" \
    --config examples/RoboMME/config/memory_inference.yaml \
    --port 8000 --seed 42 --denoising-steps 4
```

In another terminal on the same node, preview the selection with the GR00T or
simulator Python environment. Dry-run reads JSON/YAML only, without contacting
the server, loading model weights, importing the simulator or creating output:

```bash
ROBOMME_MEMORY_CHECKPOINT=/path/to/trained/primary/checkpoint
/home/vhoangth2/miniconda3/envs/gr00t/bin/python -m examples.RoboMME.eval \
    --config examples/RoboMME/config/eval_memory_primary.yaml \
    --checkpoint "$ROBOMME_MEMORY_CHECKPOINT" --dry-run
```

Then run evaluation with the simulator environment:

```bash
env PYTHONPATH=/mnt/data/vhoangth2/repos/Isaac-GR00T:/mnt/data/vhoangth2/repos/Isaac-GR00T/.cache/robomme_eval_deps \
    PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    /home/vhoangth2/.local/share/mamba/envs/robomme/bin/python -m examples.RoboMME.eval \
    --config examples/RoboMME/config/eval_memory_primary.yaml \
    --checkpoint "$ROBOMME_MEMORY_CHECKPOINT"
```

The local `.cache/robomme_eval_deps` contains the previously prepared json_numpy
dependency for this simulator environment. A simulator environment with
json_numpy installed directly can omit that path. Model and simulator processes
use separate Python environments, as in reactive evaluation.

For a quick primary smoke, append:

```text
--tasks PickXtimes --episode-indices 0 --max-steps 32 --no-video
--output-dir outputs/robomme_eval/memory_primary_smoke
```

For only the 4 primary held-out tasks, use `--task-group held-out` and a separate
output directory. For all16, serve its checkpoint on port 8001, select
`eval_memory_all16.yaml` and pass that checkpoint. To evaluate only the common
four tasks with all16, use:

```text
--tasks ButtonUnmaskSwap PatternLock PickHighlight PickXtimes
--output-dir outputs/robomme_eval/memory_all16_primary_comparison
```

`--tasks` must stay within `--task-group`. All16 has no held-out group.
`--episode-indices` overrides the YAML episode count; `--num-episodes` overrides
YAML explicit episode IDs. `--video` / `--no-video` override video settings.

## Provenance and results

Before importing the simulator, memory evaluation verifies:

1. The checkpoint has memory enabled and a consistent training identity.
2. For all16/primary, its training metadata and normalization hashes match the
   indexed training dataset selected by the protocol; saved normalization values
   also match the checkpoint's training record. Mixtures are not supported by
   these two canonical protocol checks.
3. `/health` reports the memory protocol, the requested checkpoint path and
   the matching model-config, training-sidecar and normalization metadata hashes
   from server startup.
4. Server startup seed and denoising steps match the evaluation configuration.

Checkpoint files must be readable from the evaluator, normally through the
shared filesystem. Restart servers launched before this step to expose the new
health metadata. A two-episode smoke checkpoint cannot pass the full all16 or
primary dataset check. Frozen full-task cache contents do not count as training
samples: the identity is bound to the execution dataset actually used.

Each output contains:

```text
config.json       # resolved protocol, YAML hash, checkpoint/server identity,
                  # training/test metadata hashes, tasks, episode IDs and seeds
progress.json     # per-episode task group, outcome, steps, input_frame_counts
metrics.json      # overall, per_task, by_task_group (seen/held_out/primary_comparison)
videos/           # optional streamed videos, demo prefix marked in red
```

Success rates include errors in the denominator. Empty groups have zero expected
episodes and a null success rate. `input_frame_counts` records the buffer size
at every inference attempt; for memory these frames are all sent to the server.
Demo frames do not consume the environment execution-step budget.

Use a fresh output directory for another run. `--resume` skips recorded episodes
(including errors), checks the effective configuration and server provenance,
and resets memory before each remaining episode. It does not restore remote
diffusion RNG state; a resumed stochastic rollout need not match an uninterrupted
run. Local `seed` and server startup `server_seed` are saved separately. Actual
environment seeds remain the ones in the hashed test episode metadata.

The default `--protocol held-out --policy-type reactive` keeps the previous CLI
and declared-manifest behavior, including existing reactive command references.
Legacy held-out mode with memory verifies the served checkpoint and seed but
records `training_manifest_verified: false`: its task manifest is only a declared
selection. Use the canonical all16/primary presets for research results.

## Validation

`tests/test_robomme_eval_memory.py` covers both indexed protocols, YAML/CLI
overrides, task subsets, grouped metrics, training/checkpoint mismatches, incorrect
server provenance and real HTTP memory-client rollouts over fake environments.
The runner tests verify demo prefill, new-frame-only requests, episode resets,
resume and cleanup without importing a model or simulator.

Both presets were dry-run against the real training and test metadata: all16 has
1600 indexed training episodes, primary 1200, and all 16 test tasks provide IDs
0..49. Full benchmark training/evaluation has not been launched by this step.

On 2026-11-23, the full GR00T step-5 smoke checkpoint was served through the
production memory entry point and evaluated against real RoboMME in existing
allocation 66728. PatternLock episode 0 sent `[87, 1]` frames across two inference
calls (86 demo frames + current, then one new execution frame). PickXtimes
episode 0 sent `[1, 1]`, exercising an episode without a demo prefix and a reset
between tasks. Both ran two physical action steps and returned timeout at the
configured two-step budget, with zero inference/environment/cleanup errors.
The temporary server was stopped after the smoke.

This used legacy held-out task selection with `training_manifest_verified: false`
because the checkpoint was trained on just two episodes. It is an integration
diagnostic, not a held-out score; the canonical protocols reject this training
dataset mismatch. The final regression suite passed 232 tests.

Artifacts:
`outputs/robomme_memory_eval_smoke_step7/20261123T194857Z/summary.json`,
`evaluation/config.json`, `evaluation/progress.json`, `evaluation/metrics.json`,
`server.log` and `evaluation.log` under that run directory.
