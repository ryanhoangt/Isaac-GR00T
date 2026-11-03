# Pilot 2 — Goal-Conditioned Funnel Pullback on GR00T-N1.5

Suggested content for seven slides. “Figure placeholder” notes indicate where to insert visuals and can be omitted from the slide text.

---

## Slide 1 — Objective and Dataset

**Research question:** Can a goal image help a VLA policy guide a robot back toward a valid task trajectory after a disturbance?

- Base model: **NVIDIA GR00T-N1.5-3B**; benchmark: **LIBERO-Long**, 10 tasks (`libero_10`).
- Training and evaluation cover the same task suite: **in-domain fine-tuning**.
- Available cache: **379 demonstrations → 349 training + 30 reference**; hold out three demonstrations per task for evaluation-time goal retrieval.
- Reference demonstrations are excluded from weight updates and normalization statistics.
- Two objectives: train one policy supporting both language and goal conditioning; test **perturb-and-recover** behavior.

> **Figure placeholder:** A LIBERO task showing the current observation and a milestone image from a reference demonstration.

*Scope: a diagnostic pilot using fewer than the originally planned 500 demonstrations. The uncertainty gate from Pilot 1 is not yet integrated.*

---

## Slide 2 — Dual-Mode Training

**Inputs:** third-person camera + wrist camera + proprioception; an additional goal image when goal conditioning is active.

- Freeze the Eagle vision/language backbone; train the projector, DiT action head, and new conditioning embeddings.
- Encode the goal separately using the same Eagle weights; append goal tokens to DiT conditioning with a **goal-view embedding + mode token**.
- Training goals: **50% same-demonstration**, **50% different-demonstration within the same task**, aligned by relative progress; sample a future image **16–32 steps** ahead.
- For cross-demonstration samples, add state noise with **σ = 0.02** in raw state units while retaining the original action labels.

| Share of training samples | Language | Goal image | Mode ID |
|---|---|---|---:|
| 40% | Present | Null | 0 |
| 40% | Present | Present | 1 |
| 20% | Empty prompt | Present | 1 |

**Matched baseline:** 100% language-only, using the same training split and budget.

**Both models:** 5,000 optimizer steps ≈ **6.87 epochs**; effective batch size 128; AdamW, LR 10⁻⁴, cosine schedule, 5% warmup; seed 42. Evaluate the step-5,000 checkpoints.

> **Figure placeholder:** `Current views + language → Eagle → current tokens` and `Goal image → shared Eagle → goal tokens`; concatenate with the mode token → DiT → action chunk.
>
> Optional training curves: [training_curves.png](../../../outputs/pilot2/run_2k_20261026/training_review/training_curves.png).

*The 40/40/20 mixture is this pilot’s recipe, inspired by Octo’s modality masking.*

---

## Slide 3 — Eval 1: Standard Rollouts and Multi-Mode Sanity Check

- **10 tasks × 5 initial states = 50 rollouts per condition**, with matched initial states across conditions.
- Predict and execute **16 actions per query**; use **4 Flow Matching Euler steps**; allow up to 1,000 control steps per rollout.
- Goal retrieval: find the nearest frame by cosine similarity of frozen Eagle features, **restricted to the three reference demonstrations of the same task**; select the milestone **16 steps ahead** and update each chunk.

| Policy / inference mode | Task success |
|---|---:|
| Vanilla baseline - language-only | **47/50 = 94%** |
| Dual - language-only | **45/50 = 90%** |
| Dual - goal-only | **39/50 = 78%** |
| Dual - language + goal | **46/50 = 92%** |

**Observation:** The dual policy operates in all three modes. Adding a goal improves success by only one episode relative to dual language-only; consistent gains and semantic goal grounding remain unproven.

- Shared-noise probes on 1,000 states: goal/null sensitivity has **mean 5.37%, median 1.83%**. Actions change, but this does not establish that they change in a useful direction.
- Model latency for language-only / language+goal: **72 / 97 ms per chunk**; goal retrieval adds approximately **110 ms per query** in the current implementation.

> **Figure placeholder:** [screening_overview.png](../../../outputs/pilot2/screen_5k_taskfix/analysis/screening_overview.png).

*These results use the corrected task mapping. Goal-only retrieval still receives the task ID from the benchmark. Goal/null probes change both the mode token and the goal input.*

---

## Slide 4 — Eval 2: Perturb-and-Recover Protocol

**Perturbation:** Add a Cartesian command offset over **four control steps**, starting at **t = 25 or t = 60**.

- **1×:** requested total offset vector has norm **5–8 cm**; **4×:** multiply the same base vector by four → **20–32 cm**.
- The disturbance modifies **control commands**; actual displacement depends on the controller, clipping, and contact dynamics.
- Before and during the disturbance, all A/B/C conditions use **dual language-only** control.

| Post-perturbation strategy | Control behavior |
|---|---|
| **A - Language-only** | Continue dual language-only until the episode ends |
| **B - Naive action copy** | Retrieve the nearest reference state, execute one expert chunk of **16 steps**, then return to language-only |
| **C - Goal pullback** | Retrieve the +16 milestone; use the same goal for **two chunks = 32 steps**, then return to language-only |

- **Five trials per task**, initial-state indices 5–9; match initial states, flow seeds, and disturbances across A/B/C. Simulator states at recovery start were verified to match.
- Total: **10 tasks × 5 trials × 2 triggers × 3 strategies × 2 scales = 600 rollouts**.

> **Figure placeholder:** Timeline: `Language-only → Perturb for 4 steps → A / B / C → Language-only`; alternatively, insert a rollout clip covering the disturbance and recovery window.

---

## Slide 5 — Main Result: No Consistent Advantage over Language-Only

**Recovery rate = full-task completion rate following the disturbance. Each cell contains 50 rollouts.**

| Perturbation scale | Trigger | A: Language-only | B: Action copy | C: Goal pullback | C − A |
|---|---:|---:|---:|---:|---:|
| 1× | 25 | 84% | 80% | **88%** | +4 pp |
| 1× | 60 | **88%** | 74% | **88%** | 0 pp |
| 4× | 25 | **84%** | 52% | 78% | −6 pp |
| 4× | 60 | **86%** | 80% | 84% | −2 pp |

- **C outperforms B in all four conditions**, but **does not consistently improve on A**.
- Action copying can be harmful: at 4×/t25, B achieves **52%**, compared with **84%** for A.
- Language-only is already fairly robust under this protocol. Beating action copying alone does not establish the added value of goal conditioning.

> **Main figure placeholder:** [recovery_success.png](../../../outputs/pilot2/recovery_scale_comparison_20261026/recovery_success.png).

*pp = percentage points. These pilot results use five initial states per task and one training seed; statistical superiority has not been established.*

---

## Slide 6 — Diagnostics: Weak Goal Influence and Clipping at 4×

**1. No distinct contraction benefit from goal pullback**

- Measure end-effector (EE) distance to the same-task reference set at **0 / 8 / 16 / 32 steps after the disturbance ends**.
- A and C have nearly overlapping curves; at +32, their mean distances differ by only **0.01–0.12 cm** across conditions.
- Starting from identical recovery states and flow seeds: the median relative difference in first-chunk Cartesian commands between A/C is approximately **2.3–3.4%**; their median EE separation after 16 steps is approximately **2 mm**.

**2. Requested perturbation differs from actual displacement**

| Trigger | Mean requested total offset, 1× → 4× | EE motion during the four steps, 1× → 4× | Trials clipped at 4× |
|---|---:|---:|---:|
| 25 | 6.74 → 26.98 cm | 3.02 → 5.06 cm | 76% |
| 60 | 6.74 → 26.98 cm | 2.56 → 4.36 cm | 70% |

**Interpretation:** The results are consistent with weak practical goal influence. They do not yet distinguish limited policy use of the goal from poorly aligned retrieved milestones.

> **Figure placeholder:** [paired_contraction.png](../../../outputs/pilot2/recovery_scale_comparison_20261026/paired_contraction.png). Consider showing only the two 4× panels to keep the slide compact.

*EE distance is a geometric proxy, not semantic sub-goal success or Lyapunov evidence. Observed EE motion includes nominal policy motion and contact response.*

---

## Slide 7 — Conclusions and Next Steps

**Completed**

- Trained a dual-mode policy and a matched baseline using the same data and budget.
- Completed **200 standard rollouts + 600 perturb-and-recover rollouts**, along with sensitivity probes and latency measurements.
- The policy operates with goal-only conditioning; open-loop expert action copying is less robust than closed-loop policy control in the tested conditions.

**Current conclusion**

> **We have not established a consistent goal-conditioned funnel pullback benefit over language-only control.**

- A/C differences in actions and trajectories are small; increasing the disturbance to 4× does not reveal a clear advantage for C.
- Limitations: 379 cached demonstrations; five trials per task; one training seed; no **0× control** with matched initial states and query schedules to isolate the disturbance’s effect.

**Priority next steps**

1. From the same post-perturbation state and shared noise, compare **null goal / retrieved goal / same-task wrong-phase goal / distractor goal**; inspect whether the retrieved milestone matches the manipulation phase.
2. Add a 0× control with the same query schedule. For stronger stress tests, prioritize disturbances during **grasp/carry events** and measure actual displacement.
3. Use these diagnostics to decide whether to revise retrieval/conditioning or increase the training budget.

---

## Sources and Visual Assets — Not a Presentation Slide

- [Training review](../../../outputs/pilot2/run_2k_20261026/training_review/review.md): 5,000 steps, 6.869 epochs, matched recipes.
- [Corrected screening review](../../../outputs/pilot2/screen_5k_taskfix/analysis/review.md): four inference conditions, sensitivity, and latency.
- [Recovery comparison review](../../../outputs/pilot2/recovery_scale_comparison_20261026/review.md): 1×/4× results, pairing audit, and diagnostics.
- [Recovery summary CSV](../../../outputs/pilot2/recovery_scale_comparison_20261026/summary.csv).
- [Disturbance delivery CSV](../../../outputs/pilot2/recovery_scale_comparison_20261026/disturbance_delivery.csv).
- Example 1× video: [A/t25/task00, slow motion](../../../outputs/pilot2/recovery_5k_5trials/preview/strategy_A_t25_task00_init005_slow.mp4).
- Scale illustration from a smoke test: [1×/2×/4× side-by-side](../../../outputs/pilot2/perturb_scale_smoke_20261026/scale_comparison_slow.mp4). For illustration only; **a full 2× recovery evaluation has not been run**.

**Reporting notes:** Exclude goal-retrieval results from before the task-mapping fix. Published RA-VLA results use different training/evaluation protocols and do not provide a matched improvement comparison. Initial states come from the benchmark; their exclusion from demonstrations has not been verified. Semantic labels for individual goal frames are unavailable, so EE proximity should not be reported as semantic sub-goal success.
