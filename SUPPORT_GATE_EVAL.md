# Frozen retrieval-support switching on LIBERO-Long

This extends the dual-mode RA-VLA pilot with **offline empirical calibration**, followed by fixed-threshold inference. It makes no conformal coverage, drift-detection, or recovery guarantee. The existing diagnosis/screening entrypoints are unchanged.

## Calibration

Already computed at `outputs/dual_mode/support_calibration/calibration.json`: **10 tasks, 567 expert queries**. Reuse this artifact with the existing Long reference cache; no extra Long demos or rollout outcomes are used.

- Query each of the three reference demos against the other two, excluding the query's entire demo.
- Query stride 16 control steps, plus terminal frames. Every demo receives equal total weight; quantiles use the inverse weighted empirical CDF.
- Preserve legacy BF16 top-1 selection, then compute `1 - cosine` by renormalizing the selected vectors in FP32, exactly as runtime does.
- Save task-specific q90/q95 entry thresholds and q80 exit thresholds. Observed q95 range: approximately **0.0134–0.0237**.
- Runtime uses all three reference demos; LODO uses two. Correlated expert frames are not independent calibration samples. These thresholds describe expert reference support, not a 90/95% safety or success probability. High similarity can still hide a wrong manipulation phase.

To build a fresh artifact on a worker:

```bash
cd /mnt/data/vhoangth2/repos/Isaac-GR00T-RA-VLA
srun --jobid=62771 --overlap --ntasks=1 --cpus-per-task=2 \
  bash scripts/eval_support_gate.sh calibrate \
  --memory outputs/dual_mode/long_reference_diagnosis \
  --output outputs/dual_mode/support_calibration_NEW
```

Existing completed calibration outputs are refused. `scores.csv` records query/retrieved demo IDs, frames, BF16/FP32 scores and weights; `summary.md` lists thresholds. Memory manifest and retrieval-key/provenance fingerprints bind calibration to its reference pool. Changing reference data requires recalibration.

## Inference rule

Evaluate only at action-chunk boundaries (16 steps):

1. Retrieve nearest context and compute support distance against it.
2. In action mode, enter goal mode when distance **exceeds** the task's q90 or q95 threshold.
3. Pin the retrieved anchor and its +16-step milestone for that recovery burst. Use goal mode with Gaussian initialization and masked expert actions. Language stays present.
4. After at least one goal chunk, return to action mode if distance is **at or below q80**, or after a maximum of **two goal chunks**. The cap is a fallback, not evidence of recovery.
5. Execute at least **one action chunk** before another goal burst can start. Thresholds never update from rollout data.

The nearest reference is recomputed even during a pinned-goal burst, so support measures current buffer support rather than distance to the pinned goal. Mode selection happens before action-head inference: one head prediction per chunk, with no discarded Mode-1 prediction. Episode-local state is returned through RPC and bound to episode/task/variant/calibration/configuration; parallel clients do not share gate state.

Flags: `--max-goal-chunks 2`, `--cooldown-chunks 1`, `--enter-consecutive 1`. Cooldown includes the action chunk that ends a goal burst. Raising `--enter-consecutive` delays activation by chunk intervals. The server's normal `get_action` endpoint remains fixed action mode; adaptive clients must call **`get_action_support`**, passing `episode_id` and the previously returned `gate_state` (reset to `{}` each episode).

## Evaluate

Default conditions are **dual action-only**, **adaptive q90**, and **adaptive q95**, all using the same dual-mode checkpoint and reference pool. This control isolates the effect of switching; original RA-VLA is evaluated separately by the existing screening job. Standard unperturbed Long rollouts use official initial states, 10 settling steps, 1,000 control steps maximum, and four Euler steps. Matched conditions share initial-state indices and per-step flow seeds.

Quick run: **10 tasks × 1 trial × 3 conditions = 30 rollouts**. Prefer running after the existing screening finishes to avoid CPU/GPU contention.

```bash
srun --jobid=62771 --overlap --ntasks=1 --cpus-per-task=8 \
  bash scripts/eval_support_gate.sh eval \
  --checkpoint outputs/dual_mode/gos_dual_seed42/checkpoint-15000 \
  --calibration outputs/dual_mode/support_calibration/calibration.json \
  --output outputs/ravla_eval/support_gate_quick_15k \
  --trials 1 --workers 2 --wandb-name support_gate_quick_15k
```

For 150 rollouts, use `--trials 5` and a fresh output/name. Use `--variants q95` to evaluate only q95. Add `--task-ids 0 4` for selected tasks. `--smoke` forces tasks 0/4, one trial and 160 steps (six software-check rollouts, not benchmark estimates). Port defaults to **8097**, separate from the existing screening server on 8095.

Append `--resume` to the exact original command after interruption. Completed episodes are skipped; incomplete episodes restart with fresh episode state. Model/calibration/protocol/inference-source fingerprints must match. Use fresh outputs after changing implementation or settings. From a new allocation, omit `--jobid=62771 --overlap`; the launcher requires an `srun` worker step. W&B uses **`ryan-nlp/icl-vla-uncertainty`**, group `retrieval_support_gate`.

## Outputs

- `summary.md/csv`, `per_task.csv`: full-task success, final predicate completion, executed goal-step/chunk fractions, entries, support exits, forced cap exits, inference/RPC latency.
- `gate_events.csv`: every query's score, thresholds, mode, transition reason, nearest reference, and fixed milestone.
- `gate_audit.json`: replays gate decisions against frozen calibration and verifies the recorded per-step modes, including partially executed final chunks.
- `success_modes.png`, `support_switches.png`: success/mode usage and score/threshold curves with goal-mode intervals shaded.
- Per episode: video with mode/threshold/anchor overlay, `queries.jsonl`, `trace.npz` (EE positions, reference distance, actions, executed modes), and atomic `episode.json`.
- `manifest.json`, `progress.json`, server/task logs, and a calibration snapshot preserve provenance.

Videos are saved for the first **10 trials/task/condition** by default (`--video-trials 10`), covering all trials when `--trials` is at most 10. Increase `--video-trials` for larger runs, or set it to 0 to disable videos. To regenerate a partial or completed report, run the eval command with `--stage report`. Trigger frequency is not a measured false-positive rate; task completion and a paired recovery experiment are still needed to test usefulness.

### Backfill videos from saved actions

Changing `--video-trials` does not backfill completed episodes: resume skips them, and changing the setting can conflict with the original manifest. Instead, replay `trace.npz` actions without running a policy:

```bash
# CPU allocation is sufficient with software rendering; run on a worker.
salloc --ntasks=1 --cpus-per-task=4 --mem=16G --time=06:00:00
cd /mnt/data/vhoangth2/repos/Isaac-GR00T-RA-VLA
bash scripts/replay_support_videos.sh \
  --run outputs/ravla_eval/support_gate_untilq80_10trials_15k_20261031 \
  --workers 4
```

Default selection skips episodes that have original videos. Workers process separate case/task groups, preserving the original trial/reset order within a group. Earlier episodes are simulated as needed even if their videos already exist; this avoids resetting a fresh simulator for a later trial, which can change the trajectory. Filter with `--cases adaptive_q90`, `--task-ids 0 3`, or `--initial-indices 1 2`; `--limit 1` is useful for a first test. Use `--include-existing` to compare a reconstruction with an original video. Fully cached groups are skipped on rerun; partially cached groups reproduce the necessary prefix without encoding its videos again.

Validated videos are published at **`RUN/CASE/taskXX_initYYY/rollout.mp4`**, matching the original trial-0 structure. A sibling **`replay_validation.json`** identifies the video as a saved-action reconstruction and records checks/source hashes. Original videos are never replaced. Episode JSONs, traces, and summaries stay unchanged; a historical `episode.json` may still have `video: null`, so parsers should check the canonical sibling `rollout.mp4` path. Render artifacts remain in `RUN/video_replay/CASE/taskXX_initYYY/` (or `--output`); publication uses a hardlink on the same filesystem, without duplicating video storage.

Each replay verifies the initial-state hash, the complete EE path (maximum error ≤0.1 mm), success step and final task predicates. Failed checks produce `rollout_replay.UNVERIFIED.mp4` only in the artifact directory, with a nonzero exit code; they are never published as canonical videos. No full simulator-state trajectory was recorded, so these checks do not prove identical object states at every frame. Overlays explicitly identify action replay and show scores from the original log, without recomputing retrieval.

Software rendering defaults to `MUJOCO_GL=osmesa` and needs the existing LIBERO Python environment plus OSMesa. Rendering an episode takes compute time even though no model is loaded.

Validated on CPU workers (2026-11-02): until-q80/task 0 initials 0 and 1 (1,000 steps each), plus capped-q90/task 0 initial 0 (success at step 329). Ordered replay matched every saved EE position exactly (maximum error 0), final predicates, and success timing. A fresh-simulator replay of initial 1 diverged; task-local trial order is therefore preserved by default. The grouped implementation and cached resume were exercised. This is a small software check; the remaining missing videos have not been rendered yet.

## Validation (2026-10-31, GPU allocation 62771)

- **59 tests passed**, including existing dual-mode, diagnosis, logging and VFD regressions. Additional unequal-demo-length weighting checks passed; Ruff, shell syntax and Python 3.8 compilation passed.
- Calibration: 567 expert queries across all 10 tasks; 3-demo LODO with equal demo mass.
- `outputs/ravla_eval/support_gate_smoke_v2_15k_20261031`: **6/6 truncated 160-step rollouts**, checkpoint 15000, tasks 0/4, action/q90/q95. Gate replay and executed-mode audits passed. Task 4/q90 entered goal mode at step 112, used two pinned-goal chunks, then exited by cap at 144. q95 did not trigger in these short rollouts. This is software validation, not a success-rate comparison. [W&B](https://wandb.ai/ryan-nlp/icl-vla-uncertainty/runs/h95xl1jm).
- New action-only control matched the existing screening exactly for the first 160 executed actions and EE states on task 0/init 0.
- A real pretrained RPC test with **synthetic observations** exercised goal → goal → action → goal and deterministic episode-state reset. Unit tests exercise support-based early exit and cross-episode/variant rejection.
- Completed smoke resumed successfully, skipping all six episodes. The original 30-rollout screening job continued throughout; concurrent timing is not a standalone latency benchmark.
