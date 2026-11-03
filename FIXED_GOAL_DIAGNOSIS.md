# Fixed-goal, observation-only controller diagnostic

Goal: separate short-horizon goal-controller error from online milestone-selection error. No training, new training data, object ground truth or simulator snapshots.

## Protocol

- Reuse the six successful Mode-1 references in `outputs/ravla_eval/goal_oracle_gos_3trials_20261103/action_reference/` (drawer and alphabet soup, three initial states each).
- Reference loader reads only `agent`, `wrist`, `ee`, `quat`, `gripper`, `actions`. Older archive columns such as object-phase features and simulator snapshots are never loaded.
- Standard LIBERO benchmark reset initializes each trial. Official benchmark initial-state vectors are used only as an opaque reset argument; they are never inspected or supplied to the policy, goal selector or metrics.
- Replay the recorded commands through control step 64. Check **both cameras and robot proprioception at all 65 observed frames**. Default camera tolerance is exact pixel equality; proprio tolerance is 1e-6. Mismatched replay fails the run instead of silently evaluating a different scene.
- Fix the goal to reference frame **80**. Compare Mode 1 and the isolated goal head from the same observed start for **16 control steps**. No nearest-frame search, object-state matching, phase model, uncertainty gate or moving goal.
- Model checkpoint: `outputs/dual_mode/debug_20261102/goal_same_demo_seed42_spawn/checkpoint-2000`. Frozen Mode-1 owner is verified against local RA-VLA checkpoint 3. Horizon 16, four Euler steps. Both goal views and language match the goal branch's training recipe.
- Mode 1 uses its original GOS retrieval memory. With execute-steps 16, its new action chunk must match the saved reference continuation within 1e-5, or the run fails.
- Six references × two modes = **12 local trials**. Only the 16-step local window plus the final observation is recorded in each 17-frame video.

This is a diagnostic on trained GOS tasks with future frames from successful Mode-1 trajectories. It is neither a held-out Long evaluation nor perturbation recovery. These reference actions are a comparator, not new expert corrective labels.

## Metrics

EE distance/orientation/gripper error relative to the fixed reference goal uses only robot proprioception. Workspace/wrist RGB MAE uses observed images. Action agreement is measured against the recorded Mode-1 continuation. None is a semantic sub-goal success detector: a nearby hand can still miss a grasp. Inspect object/contact state visually in the videos.

## Run

```bash
cd /mnt/data/vhoangth2/repos/Isaac-GR00T-RA-VLA
srun --jobid=63760 --overlap --ntasks=1 --cpus-per-task=6 --mem=48G \
  bash -lc 'cd /mnt/data/vhoangth2/repos/Isaac-GR00T-RA-VLA && export PYTHONPATH=$PWD OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 && exec /home/vhoangth2/miniconda3/envs/gr00t/bin/python -u -m scripts.rag.eval_fixed_goal --output outputs/ravla_eval/fixed_goal_prefix64_exec16_20261103'
```

Use `--resume` to continue the identical run after interruption. `--limit-references 1 --wandb-mode disabled` with a separate output runs a one-pair smoke test. `--execute-steps 4` with a new output performs an execution-frequency ablation: predict 16, execute four, re-observe, keep the same frame-80 goal. It does not change training or denoising steps.

Three simulator workers share one GPU policy server. W&B project: `ryan-nlp/icl-vla-uncertainty`, group `isolated_goal_fixed_window`.

Outputs include `manifest.json`, `progress.json`, `summary.json`, `episodes.csv`, `paired.csv`, `audit.json`, `RESULTS.md`, and `fixed_goal_traces.png`. Each `action/` or `goal/` case has video, observed trajectory, query log, replay checks and episode metadata. Videos show current workspace, current wrist and the fixed goal workspace. Check local progress without W&B:

```bash
cat outputs/ravla_eval/fixed_goal_prefix64_exec16_20261103/progress.json
```

Implementation: `gr00t/rag/fixed_goal_protocol.py`, `examples/Libero/eval/run_fixed_goal.py`, `scripts/rag/eval_fixed_goal.py`. Reuses the existing isolated-head server; existing training and earlier evaluators are unchanged.

## Completed results — 2026-11-03

38 tests passed and lint passed. One-pair smoke, 12 primary local trials and 12 execution-frequency ablation trials completed on allocation 63760. Observation audits passed: all replay frames matched exactly, fixed-goal image hashes matched the frames delivered to the head, paired starts matched, and all 24 full-cohort videos contain 17 frames.

Mean initial EE distance to the fixed goal: **12.31 cm**.

| Controller | Execute before replanning | Final EE error (cm) | Workspace RGB MAE |
|---|---:|---:|---:|
| Mode 1 | 16 | 0.00 | 0.00 |
| Mode 1 | 4 | 1.23 | 2.00 |
| Goal head | 16 | 6.70 | 7.13 |
| Goal head | 4 | 5.08 | 6.11 |

Mode 1 execute16 exactly reproduces its own recorded continuation; zero error is a reproducibility check, not a general-performance claim. Goal-head error improves by about 24% with execute4, but residual local error remains substantial. This comparison changes both feedback frequency and later Gaussian seeds. It does not uniquely identify a feedback mechanism or prove semantic goal attainment. These are six selected successful-reference starts at one phase in two trained GOS tasks, not held-out Long or recovery results.

The fixed-goal test removes online milestone selection, so selector tuning alone cannot explain/fix the residual errors in these cases. Focus next on goal-controller precision before cross-demo scaling or uncertainty switching. No additional training or perturbation experiment was launched.

- [Combined report](reports/dual_mode_ravla/fixed_goal_20261103/RESULTS.md)
- [Comparison plot](reports/dual_mode_ravla/fixed_goal_20261103/execution_comparison.png)
- [Primary run](outputs/ravla_eval/fixed_goal_prefix64_exec16_20261103/RESULTS.md), W&B: https://wandb.ai/ryan-nlp/icl-vla-uncertainty/runs/sa84dgq7
- [Execute4 run](outputs/ravla_eval/fixed_goal_prefix64_exec4_20261103/RESULTS.md), W&B: https://wandb.ai/ryan-nlp/icl-vla-uncertainty/runs/age8eyad

Audit command (on a worker): `python -m scripts.rag.audit_fixed_goal --output RUN`. Comparison command: `python -m scripts.rag.compare_fixed_goal --runs RUN_EXEC16 RUN_EXEC4 --output REPORT_DIR`.
