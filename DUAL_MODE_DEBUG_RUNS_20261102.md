# Dual-mode debugging runs — 2026-11-02

Allocation **63520**, one H100 80 GB. Training branches run concurrently; W&B project: **icl-vla-uncertainty**.

| Branch | Setup | Artifacts |
|---|---|---|
| Matched action-only training | Original GR00T initialization; identical GOS cache, batch 32, LR 2e-4, 3 epochs / 16,122 updates; explicit same-task negatives; 100% action mode | `outputs/dual_mode/debug_20261102/action_matched_seed42` |
| Goal positive control | Warm-start local RA-VLA; frozen Mode-1 head; separate copied/trainable goal head; shared frozen Eagle; same-demo +16 goals; language active; no adherence; no image augmentation | `outputs/dual_mode/debug_20261102/goal_same_demo_seed42_spawn` |
| Euler-step ablation | Existing dual checkpoint 15k; 1,000 held-out Long expert states; identical Gaussian across goals and NFE 4/8/16 | `outputs/dual_mode/debug_20261102/nfe_4_8_16_15k` |

Positive control: one task per GOS suite, **16,199 training frames**; first three demos/task excluded from goal updates; **180 validation states**. The warm-start RA-VLA may have seen these demos during its original training. Batch 8 × accumulation 4, LR 2e-4, 2,000 updates. Checkpoints every 1,000 updates. Validation at updates 0/1,000/2,000 compares correct, wrong-time and null goals, plus NFE 4/8/16. These are offline action metrics, not milestone/recovery success.

**Goal control completed at 14:33 UTC.** Frozen-owner weights and the deterministic Mode-1 probe were preserved through update 2,000.

| Validation MSE (4 Euler steps) | Update 1,000 | Update 2,000 |
|---|---:|---:|
| Correct same-demo future goal | 0.0879 | 0.0681 |
| Null goal | 0.1574 | 0.1427 |
| Different-time goal | 0.3198 | 0.3216 |

This supports useful goal dependence in the easier same-scene offline test. Rollout success, cross-demo transfer and recovery remain unmeasured for this new goal branch.

Mode-1 preservation checks: all **317 original head tensors** match the RA-VLA checkpoint; frozen-owner hashes and deterministic cached actions are checked after goal updates. **20 tests passed**, and real two-update smoke/checkpoint reload passed for both inference modes. New missing goal embeddings are explicitly initialized; the added branch's Beta sampler stays FP32 under BF16 loading. Data workers use **spawn** after validation; the earlier fork-based positive-control attempt stalled after one update and was cancelled.

## Completed NFE probe

Mean normalized action MSE across seven channels; not rollout SR:

| Goal | 4 steps | 8 steps | 16 steps |
|---|---:|---:|---:|
| Same-demo future | 0.1490 | 0.1559 | 0.1606 |
| Cross-demo retrieved | 0.1650 | 0.1708 | 0.1758 |
| Random same-task | 0.2347 | 0.2409 | 0.2456 |

Increasing NFE did not improve this offline metric. This does not establish a rollout-success ordering. Latencies were measured on a shared GPU and are not isolated performance benchmarks.

## Automatic matched-control rollout evaluation

The watcher starts at **checkpoint-15000**, matching the old dual checkpoint. LIBERO-Long, **10 tasks × 5 trials = 50 rollouts**, initial states 0–4/task, seed 0, original reference pool, horizon 16, four Euler steps. All five videos/task are requested. Port **8105**.

An additional early **checkpoint-5000** screening uses the same 50 initial states and reference pool, port **8106**. Output: `outputs/ravla_eval/debug_action_matched_5k_50_20261102`; log: `outputs/dual_mode/debug_20261102/logs/action_eval_5k.log`.

- Watcher state: `outputs/dual_mode/debug_20261102/action_matched_seed42/eval_watch.json`
- Results: `outputs/ravla_eval/debug_action_matched_15k_50_20261102`
- W&B: [action control](https://wandb.ai/ryan-nlp/icl-vla-uncertainty/runs/0yn4yqx6), [active goal positive control](https://wandb.ai/ryan-nlp/icl-vla-uncertainty/runs/863yoifm), [NFE probe](https://wandb.ai/ryan-nlp/icl-vla-uncertainty/runs/0au1qzzn).

## Final status — 2026-11-03

All branches completed: action-only control **16,122 updates**, isolated goal branch **2,000 updates**, NFE probe completed. Matched 50-rollout Long evaluation: action-only control **9/50 (18%) at 5k**, **8/50 (16%) at 15k**; original RA-VLA **7/50 (14%)**, old dual action-mode evaluation **3/50 (6%) at 5k**, **1/50 (2%) at 15k**. Initial states and initial retrieval match across all models. See [findings and limitations](DUAL_MODE_DEBUG_FINDINGS_20261103.md) and `outputs/dual_mode/debug_20261102/analysis_final/`.

## Training logs

```bash
cd /mnt/data/vhoangth2/repos/Isaac-GR00T-RA-VLA
tail -f outputs/dual_mode/debug_20261102/logs/action_matched.log \
        outputs/dual_mode/debug_20261102/logs/goal_positive_spawn.log
```

Code: `scripts/rag/train_goal_control.py`, `gr00t/rag/goal_control.py`, `scripts/rag/ablate_goal_nfe.py`, `scripts/rag/eval_when_checkpoint_ready.py`. The existing dual-mode training implementation remains the matched-control entrypoint.
