# Offline RA-VLA retrieval diagnostics

Recompute top-K retrieved segments from saved LIBERO rollout MP4s without running
the simulator again. The script reads each run's `commands.json` to select the
same RA-VLA checkpoint, retriever and task memory used during evaluation.

This is **recomputed retrieval from compressed video**, not a recording of the
original policy's retrieval decisions. Video compression and batched inference
can change close rankings. The original runs used policy K=1; diagnostic K=20
does not alter those runs.

## Run

From the repo on a GPU node with the `gr00t` environment available:

```bash
# Validate inputs and count queries, without loading a model or writing output.
bash scripts/precompute_ravla_retrieval.sh --dry-run

# All four existing runs: 800 videos, 21,276 query frames.
bash scripts/precompute_ravla_retrieval.sh --top-k 20 --batch-size 16

# Just one subset.
bash scripts/precompute_ravla_retrieval.sh \
  --runs goal_20trials --top-k 20 --batch-size 16

# Small test: one episode PER RUN, saved separately.
bash scripts/precompute_ravla_retrieval.sh \
  --max-episodes 1 --output-dir outputs/retrieval_cache/my_smoke_test
```

From the login node, use the existing allocation while it remains active:

```bash
srun --jobid=62121 --overlap --nodes=1 --ntasks=1 --cpus-per-task=4 \
  --gres=gpu:1 --time=01:00:00 \
  bash scripts/precompute_ravla_retrieval.sh --top-k 20 --batch-size 16
```

The four default run names are `spatial_20trials`, `object_20trials`,
`goal_20trials` and `long_20trials`. `--runs` also accepts run directories.
Use `--task-ids 0 1` and `--episode-indices 0 1` to select tasks and trials
(both zero-based). `--max-episodes N` applies after these filters, per run.

`--cuda 0` selects the visible GPU index. Lower `--batch-size` if GPU memory is
tight. The full RA-VLA checkpoint is loaded, although inference only executes
the components needed by the retriever. Memory tensors stay CPU-mapped; only
the current task's retrieval keys move to GPU. No additional dependencies are
needed in the existing `gr00t` environment. Override its Python path with
`POLICY_PYTHON=/path/to/python` if necessary.

## Sampling and output

The saved MP4 has external camera on the left and wrist camera on the right.
The script splits the views and applies the checkpoint's evaluation transforms;
it does not rotate the already rotated video again. Default sampling follows
the checkpoint's action horizon: frames `0, 16, 32, ...` in the existing runs.
These videos exclude simulator warmup, so simulator steps are frame indices
plus the warmup count (10 for these runs). `--frame-stride` can override the
sampling interval for diagnostics.

Outputs go to `outputs/retrieval_cache/<run_name>/`:

- `manifest.json`: selected episodes, source/model fingerprints, completion
  status and counts/timings for the latest invocation.
- `episode_000001.npz`, etc.: one file per episode. File numbering follows
  position in the original run's summary, starting at 1, even after filtering.
- `.precompute.lock`: prevents concurrent writers to the same run directory;
  its presence alone does not mean a process is running.

Each NPZ contains:

| Array | Contents |
| --- | --- |
| `metadata_json` | JSON string with source paths, task, episode, settings, timing and provenance |
| `frame_indices` | Zero-based MP4 query frames, shape `[Q]` |
| `simulator_steps` | Query frame indices plus simulator warmup, shape `[Q]` |
| `queries` | Query embeddings exported as float32, shape `[Q, 512]` |
| `topk_scores` | Similarity scores, descending, shape `[Q, K]` |
| `topk_indices` | Task-local memory indices, shape `[Q, K]` |

K is clipped to the task's candidate count. Scores use the original retriever's
matrix multiplication dtype before export to float32. To resolve a segment,
use `memory[task_description]["pixel_values"][index]` (two camera views) and
`memory[task_description]["action"][index]`. The official memory does not store
the source demonstration ID or starting frame, so those cannot be recovered
from the index alone.

```python
import json
import numpy as np

with np.load("outputs/retrieval_cache/goal_20trials/episode_000001.npz",
             allow_pickle=False) as cache:
    metadata = json.loads(cache["metadata_json"].item())
    print(metadata["episode"]["task_description"])
    print(cache["frame_indices"][0], cache["topk_indices"][0])
```

Completed episode files are written atomically. Rerun the same command to
resume; existing caches are validated and skipped. If inputs/settings have
changed, the script stops and asks for `--overwrite` or a different output
directory. Batch size may change across resumes. A fully cached run needs no
GPU. If interrupted mid-video, only that unfinished video needs recomputing.
Separate output directories allow different diagnostic K/stride settings.

## Validation

GPU smoke test on allocation 62121 (H100): one video from each of the four runs,
123 queries total, all succeeded. Cache resume skipped all four without model
loading. Small CPU tests cover frame alignment, warmup offsets, clipping K,
resume validation and refusing to publish a cache when video length disagrees
with the summary:

```bash
/home/vhoangth2/miniconda3/envs/gr00t/bin/python -m pytest -q \
  tests/test_precompute_retrieval.py
```
