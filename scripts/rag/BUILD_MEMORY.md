# Build RA-VLA demonstration memory

Use the existing `scripts.rag.2_build_memory` entry point. Indexed RoboMME
datasets create **one bank per canonical task**, with exactly three distinct
demos per bank by default. Each demo is encoded with its own instruction.
Lookup is over that task's candidates without an exact-instruction prefilter.
This is separate from DTW positive-pair grouping: singleton instructions and
short demos are eligible for memory. LIBERO without an episode-task index keeps
instruction keys.

## Primary training memory

From the RA-VLA checkout, run inside a GPU allocation:

```bash
srun --jobid=67148 --overlap --nodes=1 --ntasks=1 --cpus-per-task=4 \
  --gres=gpu:1 \
  env PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 \
    USE_TF=0 USE_FLAX=0 HF_HUB_OFFLINE=1 \
    NO_ALBUMENTATIONS_UPDATE=1 TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/vhoangth2/miniconda3/envs/gr00t/bin/python -u \
    -m scripts.rag.2_build_memory \
    --dataset-paths /mnt/data/vhoangth2/datasets/robomme_primary_train_v21 \
    --retriever-path outputs/retriever/robomme_primary_1000_seed42 \
    --memory-name memory_primary_seed42 \
    --num-trajs-per-task 3 --frame-stride 4 --batch-size 4 \
    --video-cache-size 2 --seed 42
```

The data config and base GR00T path default to the retriever's
`training_config.json`. An explicit conflicting override is rejected. Released
retrievers without that file retain the original LIBERO/base-GR00T defaults.
Normalization always comes from the retriever's `metadata.pt`; encoding support
demos does not fit or save new normalization statistics.

The primary command selects 36 demos across 12 tasks. `--tasks BinFill` limits
the task banks. Add `--dry-run` to print the full selection/frame plan without
loading models, decoding videos or writing output. Run dry-run on an allocated
CPU as dataset initialization still reads metadata/Parquet files.

For a small GPU test, add `--tasks BinFill VideoUnmask VideoUnmaskSwap
--max-frames-per-demo 2 --batch-size 4`, and use a separate `--memory-name`.
The cap retains up to two strided frames from **each** chosen demo, not just
the first demo. Omit it for a full build. The manifest marks capped samples.

## Data, cache and outputs

The subset's `meta/episode_tasks.json` is validated against the source metadata.
For all16, supply `--task-map examples/RoboMME/metadata/robomme_task_index_v21.json`.
An explicit map requires one dataset path. Unknown task names, insufficient
demos, duplicate memory keys across datasets and existing outputs are rejected.
Selection uses a stable per-task seed: changing the task filter does not change
the demos chosen for a retained task.

Video caching is bounded to two camera videos per dataset by default;
`--cache-video` opts into the original full RAM cache. Transforms stream one
inference batch at a time. Images are processed in eval mode; the final action
transform retains its training output mode to produce normalized/padded action
chunks and masks. Models remain frozen in eval mode throughout encoding.

Cached tensors move to CPU after each batch. Different instruction lengths are
unpadded and then padded to the bank's maximum sequence length, with attention
masks preserved. All completed banks stay in CPU RAM until the final save, so
this bounds GPU use rather than making the entire output disk-streamed.

Under `--retriever-path`, the command writes:

- `<memory-name>.pt`: task-keyed tensor dictionaries containing pixels, actions,
  action masks, keys, backbone features/masks and candidate episode/frame IDs.
  Action chunks use the existing horizon 16/padded dimension 32 for RoboMME.
- `<memory-name>.json`: selected demos/instructions/frames, dataset/task-map
  hashes, retriever/normalization fingerprints, config, progress, peak VRAM and
  completion status. The tensor file is saved atomically and verified by reload.

Choose a new memory name after a failed/interrupted run; encoding resume is not
implemented. Full output files are not overwritten. The manifest's completion
status refers to the requested task/frame plan, including any smoke cap.

## Runtime contract and next integration step

For RoboMME, pass canonical task IDs such as `VideoUnmask` to
`retrieve_from_memory`. Passing the same instruction to a different task cannot
merge their banks. Candidate text is the actual demo instruction, not the
canonical task name. The existing retrieval function already accepts these
string keys and preserves top-K retrieval within each bank.

The current RA-VLA policy-training collator and server still supply instruction
keys and use LIBERO configuration. Wiring canonical task IDs/data configuration
through those callers is the next integration step; building this file alone
does not complete policy training/evaluation support. Held-out support memory
is built separately from selected support demonstrations, with no weight updates
or inclusion of evaluation-query trajectories.

## CPU tests

Inside an allocation, with this checkout on `PYTHONPATH`:

```bash
USE_TF=0 USE_FLAX=0 HF_HUB_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1 \
  /home/vhoangth2/miniconda3/envs/gr00t/bin/python -m pytest -q \
  tests/test_memory_builder.py tests/test_retriever_sampling.py
```

Allocation 67148 was used for a GPU smoke build with the trained primary
retriever: BinFill, VideoUnmask and VideoUnmaskSwap, three demos/task, two
frames/demo (18 candidates total), batch size four. Save/reload, CPU tensor
storage, 8D action masks and canonical-task top-2 lookup passed. Real cached
sequences included lengths 559 and 569. Peak allocated/reserved VRAM was
5.89/6.66 GiB. The sample is
`outputs/retriever/robomme_primary_1000_seed42/memory_build_smoke_step4_seed42_v2.pt`,
with the manifest beside it and `memory_build_smoke_step4_validation.json`.
Full encoding was not run. A metadata-only dry-run selected 36 demos and 2,830
candidates across all 12 primary tasks with seed 42 and stride four.
