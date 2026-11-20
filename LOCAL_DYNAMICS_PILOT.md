# Offline local dynamics + goal-endpoint pilot

## Sources and differences from CCIL

Read the [original ICLR paper](https://arxiv.org/abs/2310.12972), [IROS real-robot study](https://arxiv.org/abs/2405.19307), [project](https://personalrobotics.cs.washington.edu/CCIL/), and official [`nn_dynamics.py`](https://github.com/personalrobotics/CCIL/blob/main/correct_il/models/nn_dynamics.py).

CCIL fits normalized residual dynamics using a ReLU MLP (code default 64×64), offers spectral/soft continuity regularization, synthesizes corrective state/action labels, and filters them using continuity/error heuristics. The real-robot paper emphasizes local validity and harmful contact-region labels; it also permits starting with unconstrained models.

This pilot adopts a normalized residual MLP but predicts **16-step EE displacement**, not one-step full-state dynamics. It does **not** synthesize CCIL labels or inherit its error guarantees. Inputs are robot proprioception 8 + action chunk 16×7 (120 coordinates); output is xyz displacement in meters. Three 128×128 ReLU models, optional spectral normalization; report a train-only affine ridge baseline. Normalization uses training rows only, with mean-absolute-deviation scaling. Proprio alone can omit contact/controller state, so prediction/gradient validity must be checked rather than assumed.

## Data and gates

- Use exactly the three GOS tasks/episode split of the existing isolated-goal checkpoint. First three demos/task remain excluded from updates; Long stays held out.
- Read `observation.state` and `action` from existing Parquet files. A training window requires an observed t+16 state; terminal padding/clamping never supplies dynamics targets.
- Match policy min/max normalization, command saturation and binary gripper convention. Endpoint gradients do not pass through the discrete gripper conversion; FM still supervises gripper actions.
- Validate on all held-out-demo windows, reporting per-task xyz errors in meters, a zero-motion baseline and affine ridge baseline.
- Independently audit the existing six fixed-goal action windows using RGB/proprio/action traces only. These are evaluation inputs, never dynamics training samples. No simulator snapshots or object states are used.
- Initial empirical gates: **per-task mean ≤5 mm and p95 ≤1 cm**, plus **max error ≤2 cm** on the six actual goal-policy action windows. Failures produce `status: dynamics_rejected`; no policy fine-tuning starts. Gates are configurable experiment criteria, not statistical coverage guarantees.

## Matched continuation experiment (only if dynamics passes)

Freeze the dynamics ensemble and original RA-VLA Mode-1 owner. Continue the trained 2k isolated goal checkpoint in two branches with the same data order, initialization and flow seeds:

1. FM-only continuation, 1,000 updates.
2. FM + 0.1 × endpoint MSE, 1,000 updates. Endpoint error is scaled by 5 cm before MSE.

The endpoint term differentiates through the **actual four-step goal-mode sampler** with dropout disabled, then through frozen dynamics. All gradients to dynamics parameters are disabled; gradients to actions remain. During FM training the usual dropout remains active. New Gaussian seeds use an independent generator, preserving the matched FM RNG schedule. Keep the empirical ensemble-spread/support filter and report the fraction of samples actually contributing endpoint gradients. Ensemble agreement is a heuristic, not proof of a trustworthy model gradient.

Frozen Eagle features are cached once with image augmentation disabled, per-episode shards and a bounded mmap reader, to avoid repeatedly decoding videos. Cache/proprio supervision remains same-demo; the policy sees goal images, not target proprioception. Checkpoints save at update 1,000 and the final update. Mode-1 weight hashes and deterministic actions must remain unchanged.

After both branches finish, automatically run the existing 12-case fixed-goal tests with execute16 and execute4 for each checkpoint. Dynamics-predicted validation errors are labeled separately from measured rollout errors. This is not an OOD recovery claim.

## Submit and monitor

```bash
cd /mnt/data/vhoangth2/repos/Isaac-GR00T-RA-VLA
mkdir -p slurm-logs
sbatch scripts/endpoint_pilot.sbatch
```

Resources: **1 GPU, 8 CPU, 64 GB RAM**, 12h. Every compute/test stage runs with `srun` on the worker. W&B: `ryan-nlp/icl-vla-uncertainty`, group `goal_endpoint_pilot`.

```bash
tail -f slurm-logs/endpoint-JOBID.out
cat outputs/dual_mode/endpoint_pilot_20261103/job_JOBID/dynamics/report.json
cat outputs/dual_mode/endpoint_pilot_20261103/job_JOBID/status.json
```

Overrides: `ENDPOINT_UPDATES`, `DYNAMICS_EPOCHS`, `ENDPOINT_BATCH_SIZE`, `ENDPOINT_ACCUMULATION`. Extra CLI arguments can be passed after the sbatch script, e.g. `--no-eval`. Gates should not be loosened merely to make a run proceed. A rejected dynamics model needs investigation before it supplies a training loss.

## First completed pilot: job 63889

The worker run completed in 35 seconds; **30 tests passed**, including exact four-step sampler parity, gradient flow through frozen dynamics, fixed owner preservation and Pydantic checkpoint metadata handling. No pretrained endpoint fine-tuning was performed because the empirical quality gates rejected the dynamics model.

- Training: **14,231 transitions**, held-out validation: **1,030 transitions**, whole-episode split.
- Ensemble validation: **mean 0.985 cm, p95 3.095 cm**. Affine ridge baseline: mean **2.221 cm**, p95 **5.343 cm**; zero-motion baseline mean **10.869 cm**.
- Per-task p95: drawer **4.090 cm**, alphabet soup **1.970 cm**, black bowl **2.030 cm**. All exceed the 1 cm gate.
- Existing goal-policy windows: mean prediction error **0.560 cm**, maximum **0.971 cm** (six windows). These local checks pass the 2 cm gate, but do not justify a global endpoint loss across all expert states.
- Status: `dynamics_rejected`. Gates were not relaxed after seeing results.

This supports investigating a locally filtered dynamics model or additional observable robot history/contact information before applying endpoint training. It does not establish that MLP dynamics is universally unsuitable. The numerical gates are experiment criteria, not values prescribed by CCIL or theoretical certificates.

[Full result snapshot](reports/dual_mode_ravla/local_dynamics_20261103/report.json). [W&B run](https://wandb.ai/ryan-nlp/icl-vla-uncertainty/runs/zcpoo71w). The first launch (63888) exposed a metadata-adapter bug, fixed before 63889 and covered by a regression test.
