# Dual-mode RA-VLA diagnosis

Train on Goal/Object/Spatial; evaluate on held-out **LIBERO-Long (`libero_10`)**. Language remains present in both modes. This pilot uses fixed modes or scheduled recovery, with **no adaptive gate or conformal calibration**.

## What is compared

- `standard`: original RA-VLA, dual action mode, dual goal mode. Default: 10 tasks × 5 initial states × 3 policies = **150 rollouts**, maximum 1,000 control steps each.
- `probes`: 200 cached Long reference states, balanced across tasks. Exclude the query's own demo from retrieval. Compare the retrieved forward milestone against a same-demo wrong-time goal (gap ≥32) and a random same-task goal, using identical Gaussian sources. Sensitivity measures action changes, not whether changes are correct.
- `recovery`: all strategies use dual action mode before disturbance. At steps 60–63, add a Cartesian command offset; at step 64, A continues action mode, B copies one expert chunk (16 steps), C uses a **fixed forward milestone for two chunks (32 steps)** then returns to action mode. Default levels are 0× sham and 1×, giving **300 rollouts**. Recovery initial states default to 5–9, separate from standard screening's 0–4.

All policies share the same buffer: first three available Long demos/task, stride 4, including terminal frames. Future goals use +16 **control steps**, clamped within the same demo. Task identity is checked by normalized instruction, rather than assuming cache and simulator task orders match. The diagnostic memory is already built at `outputs/dual_mode/long_reference_diagnosis`.

To rebuild this cache on a worker, run `python -m scripts.rag.build_dual_mode_memory --output YOUR_NEW_CACHE --diagnosis` using the `gr00t` Python environment, then pass `--memory YOUR_NEW_CACHE` to diagnosis.

The disturbance requests a total 5–8 cm vector at 1×, divided across four commands using the OSC controller's scaling. It modifies generated XYZ commands, not simulator state. Command clipping and delivered command offsets are logged; actual EE displacement also depends on normal motion, dynamics, and contact. ABC share initial state, flow seeds, and disturbance. Full simulator/action/EE hashes audit identical prefixes through the disturbance; B/C must use the same anchor.

## Run on a GPU worker

From this repo, use allocation 62771 while it exists. A new allocation can be requested with `salloc --ntasks=1 --cpus-per-task=8 --mem=64G --gres=gpu:1`; then omit `--jobid=62771 --overlap` below. The launcher refuses to run compute on a login node.

Start with standard screening:

```bash
cd /mnt/data/vhoangth2/repos/Isaac-GR00T-RA-VLA
srun --jobid=62771 --overlap --ntasks=1 --cpus-per-task=8 \
  bash scripts/diagnose_dual_mode.sh \
  --checkpoint outputs/dual_mode/gos_dual_seed42/checkpoint-15000 \
  --output outputs/ravla_eval/dual_diag_standard_15k \
  --stage standard --trials 5 --workers 2 \
  --wandb-name dual_diag_standard_15k
```

Use the completed final checkpoint instead of `checkpoint-15000` when available. This default baseline is the existing local GOS RA-VLA `outputs/ravla_train/gos_seed42_20261026T211436Z/checkpoint_3`; override `--baseline` explicitly for a different control. This is an existing baseline, not a newly trained compute-matched action-only control.

Goal sensitivity probes (no simulator rollouts):

```bash
srun --jobid=62771 --overlap --ntasks=1 --cpus-per-task=8 \
  bash scripts/diagnose_dual_mode.sh \
  --checkpoint outputs/dual_mode/gos_dual_seed42/checkpoint-15000 \
  --output outputs/ravla_eval/dual_diag_probes_15k \
  --stage probes --probe-states 200 \
  --wandb-name dual_diag_probes_15k
```

Then scheduled recovery, including the necessary sham:

```bash
srun --jobid=62771 --overlap --ntasks=1 --cpus-per-task=8 \
  bash scripts/diagnose_dual_mode.sh \
  --checkpoint outputs/dual_mode/gos_dual_seed42/checkpoint-15000 \
  --output outputs/ravla_eval/dual_diag_recovery_15k \
  --stage recovery --trials 5 --triggers 60 --scales 0 1 --workers 2 \
  --wandb-name dual_diag_recovery_15k
```

Run these sequentially on one GPU. For a smaller first pass add `--task-ids 0 4 --trials 1` (preserves the full 1,000-step budget). `--smoke` instead forces two tasks, one initial state, 110 steps, eight probes: **software checks only**. `--stage all` combines all stages (450 default rollouts).

Append `--resume` to the exact original command after interruption. Completed episodes are skipped; an interrupted episode restarts. Checkpoint, protocol, and inference-source hashes must match. Use a fresh output after changing the model or implementation. `--port` defaults to 8095; use separate ports and outputs if deliberately running separate jobs. `--workers 2` runs two simulator clients sharing one policy server; latency includes contention with other GPU jobs.

W&B uses project **`icl-vla-uncertainty`**, entity `ryan-nlp`. `--wandb-mode offline` or `disabled` is available. Scalars and summary plots are logged; videos are saved locally by default for the first **10 trials/task/condition** (`--video-trials 10`). Increase this value for larger runs, or use `--video-trials 0` to disable videos.

## Outputs and interpretation

Each output contains `manifest.json`, `progress.json`, logs, `summary.md`, `summary.csv`, `per_task.csv`, `latency_by_phase.csv`, support/success plots, and (for recovery) `pairing_audit.json`, `paired_comparisons.csv`, `contraction.csv/png`. Each episode directory contains `rollout.mp4`, `queries.jsonl`, `trace.npz`, and an atomic completion record `episode.json`. Probes save their CSV, three predicted chunks/state, and aggregate sensitivity statistics under `probes/`.

Support distance is **1 − cosine**, recomputed in FP32 after renormalizing the trained RA-VLA retriever's stored BF16 vectors, for the unchanged legacy-selected top-1. It is logged every policy query and never triggers switching. The legacy BF16 score is also saved for comparison. During fixed-goal recovery, nearest-support and the fixed goal anchor are recorded separately.

Recovery success counts only episodes that received all four disturbance commands. EE contraction uses minimum absolute position distance to the task's cached reference manifold at 0/8/16/32 steps after perturbation ends. Curves use a common ABC cohort that reached all four timestamps, with no terminal padding. This global geometric distance and final BDDL predicate completion are diagnostic proxies, not semantic sub-goal success or a stability guarantee. Early termination can reduce the contraction cohort; check its count.

To regenerate reports from completed or partial output, run the launcher with the same checkpoint/output and `--stage report`. Small samples, especially truncated smoke rollouts, do not establish benchmark success rates or validate a switching threshold.

## Validation (2026-10-31, allocation 62771)

- 48 tests passed, including existing dual-mode/training-logging/VFD regressions. Ruff, shell syntax, and Python 3.8 simulator compilation passed.
- Full reference cache: 10 tasks, three demos/task, 2,132 cached frames. No weight updates.
- `outputs/ravla_eval/dual_diag_smoke_20261031`: all 18 truncated rollouts and eight probes completed; four complete ABC groups passed physics/action/EE prefix and B/C-anchor audits. Inspected video frames and confirmed fixed goal at steps 64/80, then action mode at 96. No command clipping in these checks. Smoke task-success numbers are not benchmark estimates.
- `outputs/ravla_eval/dual_diag_probes_15k_20261031`: 200 states, 20/task, checkpoint 15000. Median relative action change (all seven channels) is **0.548** for wrong-time goals and **0.419** for random same-task goals. Both predictions share the same Gaussian source. This establishes context sensitivity, not correct goal following or recovery benefit. [W&B](https://wandb.ai/ryan-nlp/icl-vla-uncertainty/runs/mpyscknk).
- `outputs/ravla_eval/dual_diag_task0_15k_20261031`: three full-budget 1,000-step rollouts (task 0, initial 0). All three policies timed out. Dual action satisfied one of two final task predicates; original RA-VLA and dual goal satisfied neither. One trial/condition cannot estimate comparative success rates. [W&B](https://wandb.ai/ryan-nlp/icl-vla-uncertainty/runs/ymx8nl8g).
- Real episode resume preserved JSON/NPZ/video hashes and skipped inference; the completed 200-state probe run also resumed without recomputation. Concurrent action/goal RPCs matched their sequential repeats exactly.
