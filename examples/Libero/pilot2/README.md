# Pilot 2: goal-conditioned funnel pullback

This pilot adds optional goal-image conditioning to GR00T-N1.5, trains an
in-domain vanilla baseline on exactly the same demonstrations, and evaluates
standard control and controlled perturb/recovery on LIBERO-Long (`libero_10`).
CPU debugging and GPU smoke checks are available. Pretrained 3B training,
frozen-Eagle reference indexing, policy sensitivity and real policy rollouts
require the GPU environment. Smoke runs check execution, not scientific success.

## Agreed protocol

- Use the existing 379 demonstrations, report their actual per-task counts.
  The deterministic split holds out the first **three available episodes per
  task** (not global episode IDs 0/1/2 for every task). It yields **349 train +
  30 reference demos**, and recomputes normalization on the train partition.
- Execute **16 actions per model query**, matching the local RA-VLA evaluator
  at `../Isaac-GR00T-RA-VLA/examples/Libero/eval/run_libero_eval.py`, lines 106–114,
  and its bundled released code. All pilot conditions use the same cadence.
- Report final BDDL task-predicate completion as simulator subtask progress.
  Report retrieved-goal EE proximity separately as a proxy. The dataset has no
  predicate labels for individual retrieved frames; this implementation does
  not pretend that these task predicates are exact milestone success labels.
- This is an in-domain experiment. Previously trained LOSO checkpoints and the
  published RA-VLA Long result use different training protocols and are not
  substitutes for the matched vanilla baseline.

## Implementation

`data.py` stages disjoint, reindexed LeRobot datasets with source episode
provenance. Only parquet/metadata files are copied; videos are symlinked.
Source datasets are not modified. `split.json` identifies the split and records
the original episode mapping. `--strict-50` is available for future 500-demo data.

`dataset.py` samples a same-trajectory or same-task cross-demo goal with equal
probability, relative-progress alignment, and inclusive lookahead 16..32.
It adds sigma=0.02 noise in **raw state units** on cross-demo samples, including
the two gripper coordinates, and retains the original action labels. Dropout
is categorical and independent per sample: 40% language, 40% both, 20% goal.
The goal-only prompt contains no task text. Reference demos never enter the
training goal pool or train normalization.

`transforms.py` uses the normal two current cameras and a **separate** Eagle
conversation for the goal image. Both conversations share frozen Eagle weights.
The current conversation cannot see the goal. The DiT gets a learned mode token
and a goal-view embedding; inactive goal sequences are replaced with a learned
null goal and masked before cross-attention. An entirely language-only inference
batch skips the goal Eagle pass. The 40/40/20 mixture is this pilot's recipe,
inspired by [Octo](https://arxiv.org/abs/2405.12213), not a claim to reproduce
Octo's exact training configuration.

The attention-mask implementation is opt-in through `goal_conditioning`; the
vanilla head retains its previous unmasked attention behavior. Both baseline
and dual transforms explicitly convert a one-element language array to a string.
The optional parameters and flag persist in standard GR00T checkpoints. Training
recipes and split provenance are copied into each checkpoint's `experiment_cfg/`.

`runtime.py` builds a cosine index using mean-pooled, frozen Eagle visual tokens
before the trainable backbone projection. Retrieval searches only the three
reference episodes of the requested task, then takes frame +16. Task identity is
provided by the benchmark even in goal-only evaluation. The index defaults to
every frame; `--stride` trades index construction time against retrieval density.

## CPU checks

Run from the repository root with the existing GR00T Python:

```bash
export PYTHON=/home/vhoangth2/miniconda3/envs/gr00t/bin/python
export HF_HUB_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1 USE_TF=0

# Already prepared in this workspace; choose a fresh --output to rerun.
"$PYTHON" -m examples.Libero.pilot2 prepare

# Real repository flow head and small DiT, with synthetic Eagle features.
"$PYTHON" -m examples.Libero.pilot2 debug

# Real AV1 videos -> all three modes -> Eagle processor -> mixed batch.
"$PYTHON" -m examples.Libero.pilot2 debug-data

"$PYTHON" -m pytest tests/test_pilot2.py tests/test_libero_vfd.py -q

# Print the full recipe, without loading the model or creating the run.
"$PYTHON" -m examples.Libero.pilot2 train \
  --variant dual --output outputs/pilot2/dual --dry-run
```

Outputs are `outputs/pilot2/cpu_debug.json`, `cpu_debug.pt` and
`data_debug.json`. Existing debug/training/evaluation outputs are not overwritten.
The tiny head predicts padded `(B,16,32)` tensors; the real action mask selects
the seven LIBERO channels. Its loss values and goal sensitivity are software
checks, not evidence for either scientific hypothesis.

## GPU training

Switch to a GPU allocation before these commands. Both variants use the same
train split and the same optimizer/schedule/budget. Defaults are batch 32 per
GPU with four gradient accumulation steps (effective batch 128 on one GPU),
2K optimizer steps (about 2.75 train epochs), AdamW lr=1e-4,
weight decay=1e-5, cosine schedule,
warmup=0.05, frozen vision/language, trainable projector/DiT and seed 42. The
default is one GPU; `--num-gpus N` launches torchrun. The published repository
example uses eight GPUs; per-GPU batch and total batch are recorded explicitly.
If memory requires another batch/accumulation setting, use it for both variants.
On the H100 80GB smoke check, dual batch 32/accumulation 4 completed two optimizer
steps with finite loss and about 36.4 GiB peak allocated VRAM. Batch 128 with no
accumulation ran out of memory on its second step. The safe defaults apply to
both variants. Longer runs can have different peaks depending on prompt lengths.

```bash
"$PYTHON" -m examples.Libero.pilot2 train \
  --variant baseline --output outputs/pilot2/baseline
"$PYTHON" -m examples.Libero.pilot2 train \
  --variant dual --output outputs/pilot2/dual

# Or run both sequentially, with a separate log for each:
bash scripts/train_pilot2.sh outputs/pilot2/run_2k_20261026 \
  --batch-size 32 --gradient-accumulation-steps 4 --max-steps 2000
```

W&B online logging is enabled by default in the fixed project
`icl-vla-uncertainty`. The authenticated default workspace is used unless
`--wandb-entity` is specified. Baseline and dual have distinct run IDs/names and
share a group derived from the output root. `--wandb-group` and `--run-name`
customize those labels; an inherited project setting cannot change the fixed
project. Every ten optimizer steps the Trainer logs loss, learning rate, epoch
and gradient norm. Each checkpoint retains the split/training recipe, and
`wandb_run.json` records its dashboard URL. Model checkpoints stay local;
W&B receives metrics/config and final runtime/VRAM summaries. Use
`--report-to tensorboard` for local-only logs or explicitly select
`--wandb-mode offline` when disconnected. Neither switch silently falls back
from a requested online run.

Checkpoints are saved every 1000 optimizer steps (1000/2000 by default).
The dataset uses the repository's PyAV backend. Checkpoints retain their normal
optimizer/scheduler state; the redundant final model export is disabled. The
existing generic fine-tuning script additionally
accepts `--pilot2-goals` with `GoalDataConfig`, but the pilot wrapper is preferred
because it records the split and matched training recipe.

## GPU evaluation

Simulator task indices and converted dataset task IDs have different orders.
Evaluation matches their instruction strings before retrieval and records both
IDs in every rollout. Never use the simulator list position as a reference
dataset task ID. Reports reject goal/recovery runs without this provenance.

The evaluation Python must contain GR00T plus LIBERO's simulator dependencies.
`--libero-repo` defaults to `../LIBERO`. A config pointing at that checkout is
created inside each evaluation run; the user's home LIBERO config is untouched.
The default renderer is EGL on CUDA. An explicitly set `MUJOCO_GL` is respected.

The recommended first pass uses the matched **checkpoint-5000** pair from the
completed training run. It builds a frozen Eagle feature index from the 30
held-out reference demonstrations, then retrieves the nearest frame within the
current task and uses its frame `+16` as the goal. The index is a cache for
nearest-neighbor lookup during rollouts; it does not train or update the model.

```bash
# Inspect the plan without loading a model or writing output.
bash scripts/eval_pilot2.sh outputs/pilot2/screen_5k --dry-run

# Screening: 10 tasks × 5 official initial states × 4 standard conditions.
bash scripts/eval_pilot2.sh outputs/pilot2/screen_5k \
  --wandb-entity ryan-nlp

# Continue the same plan after interruption; completed episodes are skipped.
bash scripts/eval_pilot2.sh outputs/pilot2/screen_5k \
  --wandb-entity ryan-nlp --resume
```

The first `screen_5k` run was audited and its goal-only/both conditions used
incorrect reference task IDs. Its baseline, dual language-only, index and
sensitivity diagnostic remain reusable. Run only the two corrected goal
conditions, retaining the completed valid stages, with:

```bash
bash scripts/eval_pilot2.sh outputs/pilot2/screen_5k_taskfix \
  --reuse-screen outputs/pilot2/screen_5k --wandb-entity ryan-nlp
```

This creates a new output directory, links the four reusable stages and runs
goal-only/both again with instruction-based task matching. Use the same command
with `--resume` if interrupted. The original rollout files remain available for
auditing. Training and the frozen feature index do not need to be rerun.

This runs index (stride 1), 1,000 shared-noise goal/null/distractor probes,
baseline vanilla, dual language-only, dual goal-only, dual language+goal, and a
CSV/Markdown/PNG report. Each rollout has the full 1,000-action budget; the
first trial per task/condition is saved as video. Results and stage logs go
under the new screen output directory. Evaluation metrics also go to the fixed
W&B project `icl-vla-uncertainty`. `--wandb-mode disabled` keeps evaluation
local. The command does not launch the perturb-and-recover study; run that
after inspecting the screening results. For a short pipeline check, use
`--smoke`: task 0, one initial state, 80 steps, sparse stride-64 index and four
sensitivity states. Its report is labelled diagnostic, not a task success
estimate.

For a fresh run that also includes all six paired perturbation conditions and
logs their recovery/contraction metrics to the same W&B project, use a **new**
output directory:

```bash
bash scripts/eval_pilot2.sh outputs/pilot2/full_eval_5k \
  --recovery --wandb-entity ryan-nlp
```

This runs 10 conditions × 10 tasks × 5 initials = 500 rollouts. The
`--recovery --smoke` form uses a 110-step budget so the trigger-60 recovery
window can finish. The shorter screening run above is useful before spending
GPU time on the full protocol.

For a standalone perturb-and-recover run using the full stride-1 index:

```bash
DUAL=outputs/pilot2/run_2k_20261026/dual/checkpoint-5000
INDEX=outputs/pilot2/screen_5k/reference.npz
for trigger in 25 60; do
  for strategy in A B C; do
    "$PYTHON" -m examples.Libero.pilot2 evaluate \
      --checkpoint "$DUAL" --index "$INDEX" --strategy "$strategy" \
      --trigger "$trigger" --task-ids 0 1 2 3 4 5 6 7 8 9 \
      --trials 5 --initial-start 0 --max-steps 1000 \
      --output "outputs/pilot2/recovery_5k_${strategy}_t${trigger}"
  done
done
```

Each recovery run has the same seeds, official initial states and dual-model
language-only prefix. Use `--resume` on an interrupted individual evaluation.
Once six recovery runs complete, call `report` with the four screen condition
directories and six recovery directories as `--inputs`, the screen
`sensitivity.csv`, and a new report output directory. The original screen
report remains the four-mode comparison.

The lower-level commands remain available for custom experiments:

```bash
BASELINE=outputs/pilot2/run_2k_20261026/baseline/checkpoint-5000
DUAL=outputs/pilot2/run_2k_20261026/dual/checkpoint-5000
INDEX=outputs/pilot2/reference.npz

"$PYTHON" -m examples.Libero.pilot2 index \
  --checkpoint "$DUAL" --output "$INDEX"

"$PYTHON" -m examples.Libero.pilot2 evaluate \
  --checkpoint "$BASELINE" --index "$INDEX" --mode vanilla \
  --output outputs/pilot2/eval_vanilla

for mode in language goal both; do
  "$PYTHON" -m examples.Libero.pilot2 evaluate \
    --checkpoint "$DUAL" --index "$INDEX" --mode "$mode" \
    --output "outputs/pilot2/eval_$mode"
done

for trigger in 25 60; do
  for strategy in A B C; do
    "$PYTHON" -m examples.Libero.pilot2 evaluate \
      --checkpoint "$DUAL" --index "$INDEX" --strategy "$strategy" \
      --trigger "$trigger" --output "outputs/pilot2/recovery_${strategy}_t${trigger}"
  done
done

"$PYTHON" -m examples.Libero.pilot2 sensitivity \
  --checkpoint "$DUAL" --index "$INDEX" --states 1000 \
  --output outputs/pilot2/sensitivity.csv

"$PYTHON" -m examples.Libero.pilot2 report \
  --inputs outputs/pilot2/eval_* outputs/pilot2/recovery_* \
  --sensitivity outputs/pilot2/sensitivity.csv \
  --output outputs/pilot2/report
```

Each evaluation defaults to all ten tasks, 20 official initial states per task,
seed 0, four Euler steps, and a 1000-action budget plus ten stabilization steps.
Use `--trials 50` for 50 episodes/task. The default four standard conditions plus
six perturbation conditions total 2000 episodes at 20 trials/task. Initial
states are matched across conditions. The available cache does not establish
whether those official states occurred in the training demonstrations, so they
are **not labelled provably unseen**. `--initial-start` selects a disjoint
official-state range when needed. `--task-ids 0 --trials 1 --max-steps 80` is a
smoke run and is excluded from scientific reports unless `--allow-smoke` is used.

## Recovery-only evaluation at scale

Use the dedicated runner after screening. It uses the existing dual checkpoint
and stride-1 reference index and executes only A/B/C at triggers 25 and 60.
Instruction-based task matching is applied to retrieval and EE reference
distances. All three strategies use the same dual checkpoint and language-only
policy before the disturbance.

```bash
# CPU preflight and exact command plan; no output/model/W&B initialization.
bash scripts/eval_pilot2_recovery.sh outputs/pilot2/recovery_5k_20trials \
  --wandb-entity ryan-nlp --dry-run

# Recommended large run: 10 tasks × 20 initials × 2 triggers × 3 strategies.
bash scripts/eval_pilot2_recovery.sh outputs/pilot2/recovery_5k_20trials \
  --wandb-entity ryan-nlp

# Continue the same run after Ctrl+C or allocation expiration.
bash scripts/eval_pilot2_recovery.sh outputs/pilot2/recovery_5k_20trials \
  --wandb-entity ryan-nlp --resume
```

Defaults: checkpoint-5000, 1,000 executed actions/rollout, four Euler steps,
full 16-action chunks, seed 0, and official initial-state indices **5..24**.
Those indices were reserved from screening (0..4); exclusion from training
demonstrations is still unverified. There are **1,200 rollouts**, running
sequentially on the current GPU. A 50-trial/task study uses
`--trials 50 --initial-start 0` and a new output directory (3,000 rollouts),
including the previously screened official states.

For debugging, `--smoke` selects tasks 0 and 4, initial index 5 and a 110-action
budget, giving 12 short rollouts across both triggers and all strategies. The
report marks these as diagnostics. `--triggers 25` runs one trigger only.

The launcher logs episode progress, per-condition recovery rates, paired
success differences and plots to the fixed W&B project
`icl-vla-uncertainty`. `--wandb-mode disabled` runs locally. Weights and videos
stay local. Each episode saves JSON and NPZ arrays containing EE positions,
reference distances, nominal action commands and issued action commands.
By default the first initial per task/condition gets an MP4 (**60 videos** for
the large run); use `--video-trials 0` before starting a new run to disable
videos. Runtime input hashes prevent resuming with a changed checkpoint recipe,
index, seed, task set or rollout budget. A file lock prevents two launchers
from writing the same output directory.

Outputs under `<output>/report/` include:

- `summary.csv` / `summary.md`: task and predicate completion, conditional
  recovery rates, model latency, available-case contraction with counts.
- `paired_trials.csv` / `paired_comparisons.csv`: matched A/B/C outcomes,
  improvements and regressions on identical eligible trials.
- `paired_contraction.csv` / `paired_contraction.png`: a common cohort that
  has all four distance observations across all three strategies. This avoids
  changing trial composition along the curve, but excludes episodes ending
  before +32, including early successes. Raw distances remain in the summary.
- `latency_by_phase.csv`: separates model inference, retrieval and their
  serial sum; expert-copy chunks have no model inference.
- `pairing_audit.json`: checks identical initial states, disturbance targets
  and applied commands, pre-recovery EE trajectory, simulator state (robot and
  objects), and issued commands. B and C must retrieve the same milestone
  from the same recovery state. A mismatch stops the comparison.
- `recovery_success.png`: A/B/C completion rates after full perturbation.

An episode finishing before all four disturbance actions is excluded from the
recovery denominator and counted separately. The requested 5..8 cm Cartesian
offset is converted to OSC input units and clipped against actual controller
limits. Per-action clipping and the added command-equivalent displacement are
logged. Observed EE displacement includes the nominal policy movement and
contacts; it is not a measurement of disturbance-only causal displacement.
Global nearest-reference EE contraction is descriptive and does not establish
functional object alignment or a Lyapunov guarantee. The sensitivity diagnostic
from screening remains a separate model-grounding measurement.

## Exact recovery semantics

### Perturbation severity: matched 1x / 2x / 4x runs

`recover` and `evaluate` accept `--perturb-scale` (default 1). The original
seeded 5..8 cm command-offset vector is sampled first and then multiplied by
the scale. Keep `--seed`, task set, official initial range, triggers and four
perturbation actions fixed. This changes magnitude while preserving direction
and the pre-disturbance policy trajectory. Base vectors and pre-disturbance
EE/action/simulator hashes are stored in new episodes for matching across levels.
Existing 1x outputs/settings remain compatible with the updated runner.

```bash
# Separate pilot runs; run sequentially on one GPU.
bash scripts/eval_pilot2_recovery.sh outputs/pilot2/recovery_5k_5trials_2x \
  --trials 5 --perturb-scale 2 --wandb-entity ryan-nlp
bash scripts/eval_pilot2_recovery.sh outputs/pilot2/recovery_5k_5trials_4x \
  --trials 5 --perturb-scale 4 --wandb-entity ryan-nlp
```

Each command runs 300 rollouts, with the same initials 5..9 as the original
1x pilot. A requested 2x offset has norm 10..16 cm; 4x has norm 20..32 cm.
These are summed command offsets over four actions, **not guaranteed physical
EE displacements**. Controller clipping can reduce the delivered offset and
alter its direction, especially at 4x. The duration and recovery windows remain
unchanged. Do not increase duration as part of this amplitude comparison.

W&B run names/config identify the scale. `disturbance_delivery.csv` records
requested norm, delivered command-equivalent norm, projection onto the requested
direction, clipping counts and observed EE motion. Compare recovery success
against both requested severity and delivered severity; observed EE motion also
contains nominal policy motion and contact response. Reports refuse to mix
severity levels within an A/B/C comparison. To resume a 2x/4x run, repeat its
command with the same `--perturb-scale` and append `--resume`.

- At default 1x, perturb actions at t=25..28 or t=60..63. The sampled target Cartesian offset
  has components in +/-7cm and norm 5..8cm. Divide it over four actions and
  convert meters using the actual OSC controller input/output scales. Controller
  clipping and contacts mean the measured displacement can differ; both the
  requested offset and actual EE displacement are saved.
- Immediately after those four actions, flush the pending chunk in **all**
  strategies. A/B/C share the dual checkpoint and a language-only prefix, the
  same initial state, disturbance RNG and per-query flow seed.
- A: language-only continuation.
- B: retrieve once and execute one open-loop reference chunk (16 actions), then
  continue language-only. There is no repeated expert copying after this chunk.
- C: retrieve once after perturbation, keep that milestone fixed for two newly
  generated chunks (32 actions), then flush/switch back to language-only.
- In standard goal/both rollouts, retrieve a fresh milestone at each query.
- d_EE(delta=0) is measured **after** the disturbance, with later samples at
  +8/+16/+32 control steps. Missing post-terminal observations are not fabricated;
  the CSV includes the denominator at each offset. Recovery rate includes only
  episodes that actually received all four perturbation actions. Episodes that
  complete before the trigger are reported as ineligible.

## Outputs and interpretation

Each run saves `manifest.json`, per-episode JSON, EE/distance NPZ traces, predicate
histories and optional `--save-video` MP4s. The report creates `summary.csv`,
`summary.md`, `latency_by_mode.csv`, `success.png`, `contraction.png`, and, with
the sensitivity input, `sensitivity_summary.json` and `sensitivity_latency.png`.
Tables include per-task and aggregate task success, macro task success,
simulator subtask predicate completion, EE goal proxy, recovery eligibility/rate,
contraction with sample counts, and synchronized latency.

Sensitivity uses 1000 distinct reference validation states (never used for
weights), a same-demo +16 goal, a different-task random distractor, and shared
initial flow noise for goal/null/random. It measures denormalized 16x7 action
chunks, clamps the denominator at 1e-8, records goal/null and goal/random effects,
and times both modes after warmup. These probes reuse reference episodes for
diagnostics; do not use them to repeatedly tune hyperparameters and then claim
an untouched validation set. Latency includes preprocessing/inference but
excludes retrieval, which is logged separately. A separate Eagle pass costs
compute; low overhead is an empirical question, not an implementation guarantee.

Cross-demo relative progress is a coarse alignment. Keeping an action label
while changing the goal can create inconsistent supervision, and state-input
noise does not teach physically correct off-manifold recovery by itself.
The pilot tests whether this recipe helps; it does not assume that it forces
object-relative grounding. Goal/null/random differences establish dependence,
not correctness. Global nearest-EE distance ignores objects and task phase, so
decreasing distance is only descriptive evidence, not a Lyapunov proof. Use the
matched task/recovery success and recorded videos to assess the hypotheses.
