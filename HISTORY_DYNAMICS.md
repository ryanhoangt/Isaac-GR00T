# Causal robot-history dynamics ablation

Compare history **0, 4 and 8 past control steps**. The extra features are previous
robot proprioception relative to the current normalized proprioception, plus
previous executed commands. Current proprio + the candidate future action chunk
remain the base inputs. Horizon is 16; output is EE displacement in meters.
No object coordinates, simulator snapshots or new recovery data are used.

- Same episode partition as reliability job 63892, seed `20261103`.
- All variants use identical **t≥8** windows, with an observed t+16 endpoint;
  no history padding or crossing episode boundaries.
- Three MLP members, hidden 128×128, 100 epochs each; same optimizer, data order
  and training-only normalization. Inputs have 120/180/240 dimensions, so the
  larger histories also add first-layer parameters; this is not a parameter-
  matched architecture ablation.
- Same fixed calibration grid and gates: per-task mean ≤5 mm and p95 ≤1 cm,
  retaining ≥10%, ≥32 windows and ≥2 episodes/task. Test and original-validation
  audit use frozen calibration thresholds. All variants are reported; none is
  chosen by searching test thresholds.
- The previous test aggregate has already been inspected. This comparison is
  development evidence and does not create a fresh final holdout.
- Also predict the measured displacement of the six existing goal-policy and
  six action-policy windows. Past observations/commands come from their already
  audited replay prefixes; start EE/quaternion/gripper must match the reference.
  Only an explicit robot-proprio/action key allowlist is loaded from traces.
- This batch diagnoses dynamics only. Prediction accuracy does not establish
  valid counterfactual action gradients or physical policy recovery.

```bash
cd /mnt/data/vhoangth2/repos/Isaac-GR00T-RA-VLA
sbatch scripts/history_dynamics.sbatch
```

Resources: 1 GPU, 8 CPU, 64 GB RAM. Tests and fitting run through worker `srun`.
W&B: `ryan-nlp/icl-vla-uncertainty`, group `history_dynamics`.

```bash
tail -f slurm-logs/history-dynamics-JOBID.out
cat outputs/dual_mode/history_dynamics_20261103/job_JOBID/status.json
```

Outputs include per-variant dynamics weights, calibration grids, per-window CSVs,
per-task reports, a comparison CSV and `history_comparison.png`/`.svg`.

## Result: job 63893

Completed in 46 seconds on worker-0. **41 tests passed**. All three variants use
9,353 fitting, 1,908 calibration, 1,986 test and 958 audit windows. The episode
counts are 87/18/18/9, respectively.

| Past steps | Test mean error (cm) | Test p95 (cm) | Mean prediction error on 6 goal-policy windows (cm) | Gate passed |
| --- | ---: | ---: | ---: | --- |
| 0 | 1.030 | 3.273 | 0.575 | No |
| 4 | 0.970 | 3.250 | 0.732 | No |
| 8 | 0.954 | 3.185 | 0.485 | No |

Eight past steps reduce test mean error by about **7.4%**, but p95 by only
**2.7%**. The improvement is modest and does not eliminate the large-error tail.
None of the fixed calibration grids produces a threshold meeting all per-task
error and coverage criteria. `accepted=false` in CSVs means no deployable filter
was selected, not that all individual windows have poor predictions.

No endpoint fine-tuning was started. These results do not establish that an
endpoint loss cannot help; they fail the empirical quality gate chosen for
using this dynamics model across the training distribution. The fixed-goal
policy itself was not changed or re-evaluated in new simulator rollouts.

Next useful diagnosis: inspect the high-error windows by trajectory phase and
command saturation before adding more policy optimization. A further model
comparison needs a new final holdout, since this test set has now informed
development twice.

[Result snapshot and plot](reports/dual_mode_ravla/history_dynamics_20261103/).
[W&B run](https://wandb.ai/ryan-nlp/icl-vla-uncertainty/runs/kg8ogguo).
