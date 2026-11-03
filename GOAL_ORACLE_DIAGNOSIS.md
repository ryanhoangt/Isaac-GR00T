# Isolated goal-controller rollout pilot

Purpose: test the new isolated goal head in an easy same-scene closed-loop setting, before testing cross-demo goals, perturbations or retrieval-score gating. No training or new training data.

## Protocol

- Checkpoint: `outputs/dual_mode/debug_20261102/goal_same_demo_seed42_spawn/checkpoint-2000`.
- The original Mode-1 head is verified tensor-for-tensor against `outputs/ravla_train/gos_seed42_20261026T211436Z/checkpoint_3` (317 tensors). Both modes share frozen Eagle; horizon 16, NFE 4, execute 16 steps/chunk.
- Three tasks used to train the isolated goal head: GOS task 0 in each suite (middle drawer, alphabet soup, black bowl between plate and ramekin). Official initial states 0, 1, 2; seed 0. Suite budgets: Goal 600, Object 280, Spatial 220 control steps, plus 10 warm-up steps.
- First run unchanged Mode 1 with its original `memory_gos.pt`; save its reference trajectory. A successful reference supplies future milestone images for the goal-head rollout from the exact same initial state. Failed references are explicitly ineligible; their goal cases are skipped, not counted as goal failures.
- This is a privileged diagnostic: local nearest reference state using simulator EE/object positions and gripper opening, monotone reference pointer, search at most 32 frames forward, then goal +16 (clamped). It is a physical phase proxy, not guaranteed semantic phase annotation. No learned retriever or uncertainty threshold selects goals.
- Goal tokens contain both camera views and language, matching the goal branch's training recipe. No reference action or proprioception is provided to the goal head. The original action head still uses standard retrieved action context.
- Each branch uses a fresh seeded simulator; post-warm-up simulator-state hashes must match. Flow seeds are deterministic per suite, initial state and query step.
- At most 18 rollouts: nine Mode-1 references plus one goal rollout per successful reference. Goal SR is conditional on successful references; do not compare it directly to an unconditional benchmark SR. These are trained tasks, not held-out Long generalization results.

## Run inside an existing GPU allocation

```bash
cd /mnt/data/vhoangth2/repos/Isaac-GR00T-RA-VLA
srun --jobid=63753 --overlap --ntasks=1 --cpus-per-task=6 --mem=48G \
  bash -lc 'cd /mnt/data/vhoangth2/repos/Isaac-GR00T-RA-VLA && export PYTHONPATH=$PWD OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 && exec /home/vhoangth2/miniconda3/envs/gr00t/bin/python -u -m scripts.rag.eval_goal_oracle --output outputs/ravla_eval/goal_oracle_gos_3trials_20261103'
```

W&B: `ryan-nlp/icl-vla-uncertainty`, group `isolated_goal_oracle`. Three simulator workers share one policy server. Use `--resume` with the identical output/protocol to continue after interruption. Completed results and videos are retained; incomplete episodes rerun. An exclusive output lock and server-identity checks prevent accidental mixing.

## Outputs and monitoring

`outputs/ravla_eval/goal_oracle_gos_3trials_20261103/` contains:

- `manifest.json`, `progress.json`, `summary.json`, `episodes.csv`, `server_info.json`.
- `action_reference/<suite>_task00_init000/` and `goal_oracle/<suite>_task00_init000/`: `episode.json`, `queries.jsonl`, `trace.npz`, `rollout.mp4` for executed episodes. Every executed trial has video.
- Video panels: current workspace, current wrist, active goal workspace (blank for Mode 1). Overlay includes mode, control step, anchor and goal indices.
- Successful reference traces include rendered images and simulator states, strictly for this evaluation. They are never added to a training dataset.
- Suite logs and `server.log` capture failures; manifests record failed/interrupted/completed status.

```bash
cat outputs/ravla_eval/goal_oracle_gos_3trials_20261103/progress.json
tail -f outputs/ravla_eval/goal_oracle_gos_3trials_20261103/libero_goal.log
```

Smoke mode (`--smoke --trials 1`, separate output) runs 32 control steps in each mode even if the short reference is unsuccessful; such goal results remain ineligible. It verifies inference/restore/video plumbing and must not be reported as task SR.

Interpretation: first inspect goal progress and phase alignment in video. If clean goal control works, consider a paired perturbation test next. No automatic perturbation job is launched by this command.

## Completed pilot — 2026-11-03

Slurm step **63753.7** completed successfully in **8m 13s**. **28 tests passed**, lint passed; two-mode smoke generated 32 video frames per branch with identical simulator-start hashes. Final audit passed for all six executed goal rollouts, including exact goal-image identities.

| Task | Mode-1 reference successes | Goal successes / eligible references |
|---|---:|---:|
| Middle drawer | 3/3 | 0/3 |
| Alphabet soup | 3/3 | 1/3 |
| Black bowl | 0/3 | No eligible reference (3 skipped) |
| Total | 6/9 | 1/6 |

The 18 protocol cases comprise 15 executed rollouts and 3 explicit skips. This is an in-domain positive control; the conditional 1/6 goal result is not a LIBERO-Long benchmark SR.

Goal control can complete one case but is not reliable in clean rollouts. Do not infer that uncertainty gating or extra denoising will fix this. The phase selector is still a proxy: in the alphabet-soup init-1 failure video, the can remains on the table while later goals show grasping/placement. This leaves goal feasibility/phase selection as a confound alongside controller errors. Next isolate short milestone execution from a known reference snapshot before perturbation or cross-demo scaling. No perturbation experiment was launched.

Artifacts: [results](reports/dual_mode_ravla/goal_oracle_20261103/RESULTS.md), [trace plot](reports/dual_mode_ravla/goal_oracle_20261103/goal_traces.png), [audit](reports/dual_mode_ravla/goal_oracle_20261103/audit.json). Videos are under each case's episode directory. W&B: https://wandb.ai/ryan-nlp/icl-vla-uncertainty/runs/bvds6t0m.
