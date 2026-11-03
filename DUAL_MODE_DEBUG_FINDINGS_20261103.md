# Dual-mode RA-VLA: diagnostic findings — 2026-11-03

All three GPU diagnostic branches completed. Action-only training reached 16,122 updates; rollout evaluation used checkpoints 5k and 15k. Goal positive control reached 2,000 updates.

## Matched LIBERO-Long rollout results

10 tasks × 5 initial states (indices 0–4), 50 rollouts/model; horizon 16, four Euler steps, seed 0, same reference buffer and 1,000-step limit.

| Model | Successes | SR |
|---|---:|---:|
| Original/local RA-VLA | 7/50 | 14% |
| New action-only training control, 5k | 9/50 | 18% |
| Old dual 5k, evaluated in action mode | 3/50 | 6% |
| New action-only training control, 15k | 8/50 | 16% |
| Old dual 15k, evaluated in action mode | 1/50 | 2% |

The training control uses 100% action-mode samples; it is distinct from evaluating a dual-trained model in action mode. Control and dual share the GOS data, initialization, seed, optimizer settings and effective batch size. Their action/goal sampling proportions differ.

Audit passed: all 50 official initial-state hashes, initial retrieved demo/frame/index and initial EE positions match for every model. All episodes are complete and non-smoke. See [audit](reports/dual_mode_ravla/debug_controls_20261103/audit.json), [summary CSV](reports/dual_mode_ravla/debug_controls_20261103/summary.csv) and [paired outcomes](reports/dual_mode_ravla/debug_controls_20261103/paired_outcomes.csv).

**Interpretation:** the action-only training pipeline retains baseline-level performance. The regression is associated with the joint dual-mode training recipe. This does not uniquely identify gradient interference: dual training also halves expected action-mode exposure per optimizer budget. Goal adherence/negative quality, cross-demo pairing and mode-dependent flow targets remain possible contributors. One seed and 50 benchmark initial states do not establish superiority over original RA-VLA; 9 versus 8 successes also does not establish overfitting.

## Isolated goal branch: positive offline control

Warm-start RA-VLA; freeze the original action head and train a separate copied goal head. Same-demo +16 future goals, language active, no adherence, no image augmentation. One task per GOS suite (three tasks), 16,199 training frames; 180 validation expert states from the first three demos/task excluded from goal updates. The original warm-start model may already have seen those demos.

| Validation normalized action MSE, NFE 4 | Update 0 | Update 2,000 |
|---|---:|---:|
| Correct future goal | 1.2778 | 0.0681 |
| Null goal | 1.2778 | 0.1427 |
| Different-time same-demo goal | 1.2778 | 0.3216 |

Null means zero goal features in the same goal branch, not Mode 1. Different-time goals are temporal proxies, not verified semantic negatives. All 317 original head tensors and the deterministic Mode-1 action probe remain exactly unchanged after training.

This supports useful goal dependence in the easier same-scene offline setting. It does not yet demonstrate rollout success, cross-demo/object-relative control, held-out-suite generalization or recovery. Several recipe factors changed together, so this is a positive control rather than a single-factor ablation.

## More flow steps did not help offline

Old dual 15k, 1,000 Long expert states, shared Gaussian across conditions:

| Goal condition / action MSE | NFE 4 | NFE 8 | NFE 16 |
|---|---:|---:|---:|
| Same-demo future | 0.1490 | 0.1559 | 0.1606 |
| Cross-demo retrieved | 0.1650 | 0.1708 | 0.1758 |
| Random same-task | 0.2347 | 0.2409 | 0.2456 |

This does not establish a rollout-SR ordering; retain NFE 4 for the next diagnostic.

## Recommended next experiment

1. Preserve the validated RA-VLA Mode-1 head and test the isolated goal branch in closed loop with privileged, phase-correct same-scene goals.
2. If it controls successfully, test paired perturb-and-recover, then introduce cross-demo goals before scaling to all training tasks.
3. Keep retrieval-score switching/calibration deferred until goal control works in isolation. To isolate the old regression, compare equal action-sample exposure and probe per-mode gradients.

These are recommendations; no new experiment was launched during this result review.
