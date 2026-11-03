# CF-VLA: Goal-Grounding Loss Ideas

Research notes for extending RA-VLA; not implemented.

- **Goal-aware adherence:** In Mode 2, enforce `E(correct_goal) + margin < E(wrong_goal)`, where `E` is Flow Matching MSE. Use same-task wrong-phase/relationship goals as hard negatives; keep observation, noise, flow time, and target action fixed. Mask expert actions in both passes.
- **Relational contrastive loss:** Pull together goal representations with the same object-relative configuration across different layouts; separate different configurations. Requires reliable correspondence—DTW action alignment alone is a proxy. Do not force raw actions across layouts to match.
- **Action-to-goal consistency:** Penalize relational distance between the predicted post-action state and the goal. Requires differentiable kinematics/dynamics or a learned dynamics model, plus goals compatible with the action horizon.

**Try first:** Goal-aware adherence, then paired correct/wrong-phase/null-goal probes and perturb-and-recover evaluation. Add relational supervision if needed; leave uncertainty gating/calibration for later.

**Limits:** Goal sensitivity is not semantic correctness. Margin can improve by worsening negative predictions. Proprioceptive noise with unchanged images/actions is augmentation, not recovery supervision; recovery needs coherent states and valid corrective-action targets.

Reference for object-action consistency: [Imagine2Act](https://arxiv.org/abs/2509.17125).
