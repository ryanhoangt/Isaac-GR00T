# HDF5 to the existing LIBERO LeRobot v2.1 layout (unit 2)

`scripts/convert_libero_mem_lerobot.py` calls the modular implementation in
[`conversion.py`](conversion.py). It consumes the **Full** HDF5 download and a
completed [unit-1 audit](README.md), and exports a single canonical dataset.
It does not require the Raw files at conversion time, load simulator/model
code, or download anything. The audit's metadata checksum and source file
size/mtime must match; every selected state/action trajectory is checked against
its audited fingerprint before encoding.

The four existing local LIBERO datasets all have `codebase_version: v2.1` and
20 FPS. This converter writes the same schema and copies
[`examples/Libero/modality.json`](../Libero/modality.json) byte-for-byte. The
data config remains `examples.Libero.custom_data_config:LiberoDataConfig`.

## Mapping and alignment

| Full HDF5 | Converted field |
| --- | --- |
| `obs/ee_states` (6D axis-angle pose) + `obs/gripper_states` (2D) | `observation.state`, float32 `[T,8]` |
| First six `actions` components | First six `action` components, float32, unchanged controller deltas |
| Simulator gripper `g`, -1/+1 | Dataset gripper `(1-g)/2`, 0/1 |
| `obs/agentview_rgb` | `observation.images.image` video |
| `obs/eye_in_hand_rgb` | `observation.images.wrist_image` video |
| Registry instruction | `task_index` + `meta/tasks.jsonl` |

The existing LIBERO adapter applies `normalize_gripper_action` to turn dataset
predictions back into simulator commands. This is distinct from the two physical
finger positions in the state. The existing fields named `roll/pitch/yaw` hold
axis-angle components; the converter does not reinterpret them as Euler angles.

Refinement stores `rot180(raw[:, ::-1])`, while GR00T's existing
`get_libero_image` uses `rot180(raw)`. Conversion applies a horizontal flip to
the stored image to match that evaluator convention. No resize is performed.
H.264 CRF 18 / yuv420p encoding is lossy. Two videos are streamed frame by frame
per episode; depth, masks, boxes and oracle subgoal text stay in the source.

All source rows, including stationary/terminal rows, are preserved at 20 FPS.
There is no temporal shift, no-op filtering or resampling. Output episode/task
indices are compact and deterministic. `source_task_id`, `source_demo_id`,
`source_episode_key` (`task_name/demo_id`) and `source_frame_index` preserve stable
identity across later subsets; `source_action_gripper` retains the original
simulator gripper command. `meta/episode_mapping.json` also records the matched
Raw demo ID and simulator initial-state fingerprint.

Global and per-episode state/action statistics are computed from exported float32
rows. The canonical all-task dataset's statistics include all ten tasks.
**Recompute normalization on each training fold before held-out experiments**;
the [unit-3 fold extractor](FOLDS.md) handles this and copies standalone datasets.

## Run the full conversion yourself

From the repository root, using the existing four-CPU allocation:

```bash
cd /mnt/data/vhoangth2/repos/Isaac-GR00T

srun --jobid=67857 --overlap --ntasks=1 --cpus-per-task=4 --mem=24G \
    env PYTHONPATH="$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    /home/vhoangth2/miniconda3/envs/gr00t/bin/python -u \
    scripts/convert_libero_mem_lerobot.py \
    --source /mnt/data/vhoangth2/datasets/LIBERO-Mem \
    --audit "$PWD/outputs/libero_mem_source_audit_unit1_verified.json" \
    --output /mnt/data/vhoangth2/datasets/libero_mem_lerobot_v21 \
    --workers 4
```

Append `--dry-run` to validate the plan first, without creating directories or
reading RGB payloads. The current full plan is 961 episodes / 321,405 frames.
No full conversion was launched during implementation.

For a small conversion, append `--task-ids 1 7 --demo-ids 1` and choose a separate
output. `--demo-ids` means original numeric demo IDs, not ordinal positions;
every requested demo must exist in each selected task. Missing refined demos
are not silently substituted. `--chunks-size` defaults to 1000.

## Interruption and resume

Re-run the same command with `--resume` after an interruption. Worker count may
change; source identities, selection, chunk size and conversion settings must
remain the same. Cached videos/parquets are checksummed before reuse. A work
directory lock prevents concurrent writers to the same conversion.

Completed episodes live in `.libero_mem_lerobot_v21.work/` beside the intended
output. The final directory appears only after all episodes and metadata are
complete. Published video/parquet files are hard links to the cache, so their
payloads are not duplicated. Existing final output is always refused, including
with `--resume`. Keep the work directory available while resuming.

## Validation

Focused tests run inside a CPU allocation:

```bash
srun --jobid=67857 --overlap --ntasks=1 --cpus-per-task=2 --mem=12G \
    env PYTHONPATH="$PWD" USE_TF=0 USE_FLAX=0 HF_HUB_OFFLINE=1 \
        TRANSFORMERS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1 \
        OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    /home/vhoangth2/miniconda3/envs/gr00t/bin/python -m pytest -q \
    tests/test_libero_mem_source.py tests/test_libero_mem_conversion.py \
    tests/test_robomme_lerobot_conversion.py
```

Tests exercise spawned parallel workers, sparse source IDs, chunk boundaries,
decoded frame order/orientation, simulator gripper round-trip through the actual
LIBERO helper, unchanged source files, normalization, the existing GR00T loader
and transforms, interruption/resume, corrupt cache rejection and stale audits.
Shared paths, JSON writers and statistics were extracted from the RoboMME
converter to `gr00t/utils/lerobot_conversion.py`; its public imports and behavior
are retained and its conversion tests are included above.

The real-data smoke uses T1/demo_1 and T7/demo_1: 2 episodes / 587 frames / 4
videos at `outputs/libero_mem_conversion_smoke_unit2`. The separate local
`validation_summary.json` records all-frame source comparison and CPU
loader/transform/collator checks. Simulation replay and model performance are
outside this conversion unit.
