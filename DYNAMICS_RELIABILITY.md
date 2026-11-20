# Dynamics reliability diagnosis

Continue the local-dynamics pilot without relaxing its physical error criteria.
No simulator object states or additional recovery data are used.

## Protocol

- For each of the three trained GOS tasks, shuffle the original training demos
  with seed `20261103`: reserve **6 calibration + 6 test episodes**, fit dynamics
  on the rest. Keep the original 3 validation episodes/task as a separate audit.
  No overlapping windows from one episode cross these groups. The starting
  policy has seen the original training pool; this is a dynamics split, not a
  claim of unseen-task policy evaluation.
- Fit the same 3-member 128×128 normalized residual MLP ensemble for 100 epochs.
- Reliability scores: ensemble prediction spread (meters), and nearest training
  input RMS distance after train-only normalization (8 proprio + 16×7 actions).
- Choose a joint threshold from a fixed 6×6 calibration-quantile grid, maximizing
  retained coverage subject to **mean ≤5 mm, p95 ≤1 cm per task**. Each task must
  retain ≥10% of windows, ≥32 windows and ≥2 episodes.
- Freeze thresholds. Test and original-validation audit must independently pass
  the same criteria before endpoint fine-tuning is considered. No candidate is
  selected from test/audit results. The original audit aggregate was inspected
  in the preceding pilot, so the new test split is also reported separately.
- Window counts are correlated within episodes. These are empirical selection
  criteria, not conformal coverage bounds or guarantees on dynamics gradients.

## Run

```bash
cd /mnt/data/vhoangth2/repos/Isaac-GR00T-RA-VLA
sbatch scripts/dynamics_reliability.sbatch
```

The job requests 1 GPU, 8 CPU and 64 GB RAM. Tests and all model computation run
on the worker through `srun`. W&B project: `ryan-nlp/icl-vla-uncertainty`.

Outputs: `outputs/dual_mode/dynamics_reliability_20261103/job_JOBID/` contains the
episode manifest, frozen threshold selection, per-window CSVs, per-task report,
`risk_coverage.png`/`.svg`, and a dynamics checkpoint. `status.json` explicitly
states whether the quality gate passed. This script only diagnoses reliability;
it never silently starts policy training or overrides a rejected dynamics gate.

```bash
tail -f slurm-logs/dynamics-filter-JOBID.out
cat outputs/dual_mode/dynamics_reliability_20261103/job_JOBID/status.json
```

## Result: batch job 63892

Allocation 63891 was cancelled at the user's request. Batch job 63892 completed
in 39 seconds on worker-0; **36 tests passed**. No allocation remains active.

- Dynamics fit: 10,049 windows / 87 episodes; calibration: 2,052 / 18; test:
  2,130 / 18; original validation audit: 1,030 / 9.
- Unfiltered test errors: mean **0.988 cm**, p95 **3.392 cm**.
- Test Spearman error correlation: ensemble spread **0.567**, input distance
  **0.555**. These scores rank risk somewhat, but that is insufficient to
  certify the required retained-set accuracy.
- **None of the 36 calibration candidates passes all three tasks.** No threshold
  was selected and no endpoint fine-tuning was started. CSV `accepted=false`
  means no deployable filter was selected; it does not mean every individual
  window is inaccurate.
- An illustrative *calibration-only* candidate (spread q40, distance q60) retains
  33.9% overall. Soup and bowl pass; drawer retains 36.0% but mean **0.630 cm**,
  p95 **2.396 cm**, failing the criteria. This candidate is not a validated
  deployment threshold and was not selected using test errors.

The next model change to test is short robot proprio/action history, keeping
these splits and criteria fixed. Further tuning should treat this test set as
development data or reserve a new final holdout. No claim of valid action
gradients or physical recovery follows from prediction-error ranking alone.

[Result snapshot and plots](reports/dual_mode_ravla/dynamics_reliability_20261103/).
[W&B run](https://wandb.ai/ryan-nlp/icl-vla-uncertainty/runs/rp4mttrs).
