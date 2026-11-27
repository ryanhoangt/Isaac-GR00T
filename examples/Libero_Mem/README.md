# LIBERO-Mem integration

This directory extends the existing [`examples/Libero`](../Libero) support.
Unit 1 defines task identity and the proposal's two task folds, and audits the
downloaded Full/Raw sources. Unit 2 adds [audited HDF5-to-LeRobot conversion](CONVERSION.md)
using the existing LIBERO data configuration. Unit 3 adds
[standalone copied training folds](FOLDS.md) with their own normalization.
Unit 4 adds [sequence-aware episode accounting and audited expert replay](EXPERT_REPLAY.md)
on the existing LIBERO environment/image/video helpers. Unit 5 adds
[GR00T HTTP policy rollouts](EVALUATION.md), shared LIBERO mapping, action chunking,
fold/task selection and grouped metrics. Training integration follows.
No simulator, model weights or GPU are needed
for source audit, conversion or fold extraction; expert replay runs on CPU with OSMesa.

Sources:

- [Paper](https://arxiv.org/abs/2511.11478).
- [Benchmark code](https://github.com/libero-mem/libero-mem), inspected at
  `0eee0defa4024e51511b6e15a79f30e9620209de`.
- [Full demonstrations](https://huggingface.co/datasets/libero-mem/LIBERO-Mem).
- [Raw trajectories and simulator initial states](https://huggingface.co/datasets/libero-mem/LIBERO-Mem-Raw).

## Task identity and folds

[`tasks.py`](tasks.py) contains the ten canonical task names and instructions.
Names match the upstream `libero_suite_task_map.py`. Instructions are the task
text, without the numeric filename prefix. Registry imports use only the Python
standard library, independently of the installed LIBERO checkout.

| Task | Description | Proposal group | Memory types | Fold |
| --- | --- | --- | --- | --- |
| T1 | Bowl lift/place | Single | OM | A |
| T2 | Bottle lift/place | Single | OM | B |
| T3 | Bowl ×3 | Counting | OM, OS | A |
| T4 | Bottle ×3 | Counting | OM, OS | B |
| T5 | Bowl ×5 | Counting | OM, OS | A |
| T6 | Bowl ×7 | Counting | OM, OS | B |
| T7 | Swap two bowls | Sequencing/spatial | OM, OR | A |
| T8 | Rotate three bowls | Sequencing/spatial | OM, OR | B |
| T9 | Cheese in basket, move that basket | Sequencing/spatial | OM, OO | A |
| T10 | Cheese in basket, move empty basket | Sequencing/spatial | OM, OO | B |

OM/OS/OR/OO are the benchmark's motion, sequence, relations and occlusion types.
The three reporting groups and symmetric folds are our proposal's protocol.
Each fold has one single, two counting and two sequencing/spatial tasks, covering
all four memory types. `get_fold("A")` means **train A / held-out B**;
`get_fold("B")` reverses the direction. These functions return task names, not
LeRobot instruction IDs or indices from a potentially reordered simulator suite.

```python
from examples.Libero_Mem.tasks import TASKS, get_fold

training_names = get_fold("A")["train_tasks"]
held_out_names = get_fold("A")["held_out_tasks"]
```

## Audit downloaded sources

Keep the downloads outside both repositories. From the GR00T repository root,
run in an existing CPU allocation using the GR00T Python environment:

```bash
srun --jobid=67857 --overlap --ntasks=1 --cpus-per-task=2 --mem=24G \
    env PYTHONPATH="$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    /home/vhoangth2/miniconda3/envs/gr00t/bin/python \
    -m examples.Libero_Mem.source \
    --full-root /mnt/data/vhoangth2/datasets/LIBERO-Mem \
    --raw-root /mnt/data/vhoangth2/datasets/LIBERO-Mem-Raw \
    --output outputs/libero_mem_source_audit.json
```

Replace the job ID with your active CPU allocation. `metainfo.json` is large
(about 681 MiB locally) and parsed in RAM; the 24 GiB step allowance includes
Python's expanded JSON objects. No image payload is read. `--task-ids 1 7` scopes
the HDF5 checks to those tasks, while the JSON metadata is still parsed in full.
The report explicitly marks partial audits with `all_tasks_audited: false`.
Omitting `--output` only prints the summary. Existing output files are refused.

The audit checks:

- Known task filenames, canonical sparse demo IDs and natural numeric ordering.
- Full: aligned RGB headers, finite EEF pose/gripper/joint states and 7D actions,
  final-only `dones`, and temporal alignment of optional HDF5 fields.
- Metadata: matching task/demo identities and instructions, successful refined
  demonstrations, annotation lengths and finite simulator initial states.
- Raw: finite actions/states, compatible simulator initial-state dimensions,
  and the per-demo model XML needed for replay.
- Full/Raw pairing by exact float64 initial-state fingerprint, preferring the
  same demo ID; a remapped ID must have a unique state match and is reported.

The JSON inventory retains original demo IDs, lengths, schema examples,
state/action fingerprints, model XML hashes, source file size/mtime and a SHA256
of `metainfo.json`. It lists Raw-only demos by **initial-state exclusion**:
a different demo ID sharing a Full initial state is excluded too. Counts that
differ from the nominal 100 Full / 120 Raw are warnings, not silently repaired.

Raw-only entries include both unrefined training candidates and the intended
evaluation tail. They are **not an automatic evaluation split**. Unit 4 checks
that each task's final twenty Raw initial states are absent from that task's Full
demonstrations before selecting unseen episodes. This initial-state selection is
separate from the proposal's task folds; support/test episode separation for
retrieval remains part of subsequent integration.
Full and Raw actions are not required to match: refinement trims/appends actions
and replays trajectories. This audit checks metadata identity and numeric
integrity, not replay equivalence, image orientation or physical task success.
It does not hash the full HDF5 payloads or validate every pixel/annotation value.

## Local validation

The complete source audit on 2026-11-27 in CPU allocation `67857` found:

- 961 refined Full demos, 321,405 frames; 1,200 Raw demos, 404,392 frames.
- Full counts T1–T10: `100, 98, 98, 99, 97, 100, 88, 82, 100, 99`.
- All 961 Full demos matched the same Raw demo ID and exact initial state.
- EEF state is 6D axis-angle pose plus 2D gripper position (8D total); actions
  are 7D controller commands, consistent with the existing LIBERO modalities.

The local inventory is `outputs/libero_mem_source_audit_unit1_verified.json` (ignored by
Git). It records the actual files; these counts are observations, not loader
requirements. Source data and the large JSON metadata remain outside Git.

Run the focused tests inside a CPU allocation:

```bash
srun --jobid=67857 --overlap --ntasks=1 --cpus-per-task=2 --mem=8G \
    env PYTHONPATH="$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    /home/vhoangth2/miniconda3/envs/gr00t/bin/python \
    -m pytest -q tests/test_libero_mem_source.py
```

Tests cover complementary folds, sparse IDs, state-based leakage prevention,
unique/ambiguous remapping, malformed schema/metadata, nonfinite values, duplicate
JSON keys, concurrent source modification, partial audits and overwrite protection. An image-read guard verifies
that the audit does not read RGB/depth/segmentation payloads.
