# Sanity-check a new retriever with a fixed RA-VLA policy

Use the released GOS RA-VLA `checkpoint_3` for both arms, eval on held-out Long,
and preserve the authors' memory candidates. Only the retriever weights and
their associated memory keys differ.

## Re-encode the existing Long buffer

Run in the `gr00t` environment on a GPU node. The following command uses the
existing allocation while it remains active:

```bash
srun --jobid=66728 --overlap --nodes=1 --ntasks=1 --cpus-per-task=4 \
  --gres=gpu:1 --time=00:20:00 \
  env USE_TF=0 HF_HUB_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1 \
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  /home/vhoangth2/miniconda3/envs/gr00t/bin/python -u -m scripts.rag.rekey_memory \
  --source-memory ra-vla-official-assets/libero_gos/memory_l.pt \
  --output-memory outputs/retriever/libero_gos_750_seed0/memory_l_rekeyed.pt \
  --retriever-path outputs/retriever/libero_gos_750_seed0 \
  --checkpoint-path ra-vla-official-assets/ravla/ravla/libero_gos/checkpoint_3 \
  --reference-retriever-path ra-vla-official-assets/libero_gos \
  --batch-size 4
```

The output must not already exist. The buffer produced in this session is
already available at the output path above; its companion `.json` contains
the validation report.

The script uses the standard image/text prompt and inserts the exact cached
normalized pixels. It avoids reapplying crop/resize to those pixels. It checks
the first batch of each task with the original retriever against the stored
keys, requiring cosine similarity >=0.995. This session obtained >=0.99997.
Then it encodes all 2,045 candidates with the new retriever. After save/reload,
all non-key tensors are checked for bitwise equality, including images, actions,
backbone features and masks. The source file is never overwritten.

The RA-VLA checkpoint provides its frozen visual/text encoder and original
normalization metadata. There is no need to regenerate demonstrations or memory
payloads with the new retriever's action normalization.

## Matched A/B evaluation

Run these arms sequentially to avoid loading two policy servers into VRAM.
Each uses 5 isolated task processes sharing one serial policy server, with 3
trials on each of the 10 tasks. The LIBERO evaluator uses initial-state indices
0, 1 and 2 for each task. Source policy, seed, K, denoising steps and task states
are fixed across arms. Use new run names if these directories already exist.

```bash
# Authors' retriever and original memory keys.
srun --jobid=66728 --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 \
  --gres=gpu:1 --time=02:00:00 \
  bash scripts/eval_ravla_parallel.sh \
  --subset long --trials 3 --workers 5 --seed 0 --k 1 --denoising-steps 4 \
  --port 18071 --run-name retriever_ab_original_long3_parallel_20261123

# Trained retriever, same memory payloads with re-encoded keys.
srun --jobid=66728 --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 \
  --gres=gpu:1 --time=02:00:00 \
  bash scripts/eval_ravla_parallel.sh \
  --subset long --trials 3 --workers 5 --seed 0 --k 1 --denoising-steps 4 \
  --retriever-path outputs/retriever/libero_gos_750_seed0 \
  --memory-path outputs/retriever/libero_gos_750_seed0/memory_l_rekeyed.pt \
  --port 18072 --run-name retriever_ab_trained_long3_parallel_20261123
```

Both default to the same released `libero_gos/checkpoint_3`. Results live in
`outputs/ravla_eval/<run-name>/`: `progress.json`, aggregate
`rollouts/summary.json`, per-task logs and MP4 videos. The parallel runner
terminates its own simulators and policy server on exit, leaving the allocation
and unrelated workloads alone.

This checks replacement of one component in a fixed policy, rather than an
isolated retrieval metric or a full retraining reproduction. Thirty episodes
per arm are a small sanity sample. The earlier historical baseline used 20
trials/task (200 episodes): Long 28/200=14%, Spatial 57/200=28.5%, Object
110/200=55%, Goal 91/200=45.5%. Use the fresh matched 30-episode baseline for the
direct A/B comparison, keeping the historical result as context.

## Result from this session

Both arms completed 30 episodes (3 trials/task) in allocation 66728:

| Retriever | Successes | Success rate |
|---|---:|---:|
| Authors' GOS retriever | 5/30 | 16.7% |
| Trained GOS retriever, 750 steps, seed 0 | 2/30 | 6.7% |

The trained retriever is 10 percentage points lower in this small check. Paired
outcomes show 1 improvement, 4 regressions, 1 shared success and 24 shared
failures. The fresh baseline exactly reproduces all 30 historical outcomes and
control-step counts for the same initial states. All 60 rollout videos exist.

The fixed-policy pipeline runs successfully with the new retriever, but this
sample does not show a performance improvement and does not isolate intrinsic
retrieval quality. Full results, per-task/episode CSVs and links to changed
outcome videos are in
`outputs/ravla_eval/retriever_ab_long3_comparison_20261123/REPORT.md`.

## Tests

Tests for cached-pixel handling and save/reload payload preservation:

```bash
USE_TF=0 /home/vhoangth2/miniconda3/envs/gr00t/bin/python -m pytest -q \
  tests/test_rekey_memory.py
```
