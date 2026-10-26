# RA-VLA retrieval viewer

Streamlit app for the offline caches described in [PRECOMPUTE_RETRIEVAL.md](PRECOMPUTE_RETRIEVAL.md).
It runs on CPU and uses the existing MP4s and task memory. No policy server,
model checkpoint loading or simulator is needed.

## Setup and launch

From the repo, create a separate environment that inherits torch/PyAV from the
existing `gr00t` environment, then install the viewer dependencies:

```bash
/home/vhoangth2/miniconda3/envs/gr00t/bin/python -m venv \
  --system-site-packages .venv-visualizer
uv pip install --python .venv-visualizer/bin/python \
  -r scripts/rag/requirements-visualizer.txt

bash scripts/visualize_ravla_retrieval.sh --server.port 8501
```

The environment is already installed in this checkout. The separate venv keeps
the training/evaluation environments unchanged. Without `uv`, use
`.venv-visualizer/bin/python -m pip install -r scripts/rag/requirements-visualizer.txt`.
To use a different environment, set `VIEWER_PYTHON=/path/to/python` when launching.

Open <http://localhost:8501> on the same machine. When working remotely, forward
8501 in VS Code's Ports panel, or use a tunnel from your laptop to **the host
running the app**:

```bash
ssh -N -L 8501:127.0.0.1:8501 vhoangth2@HOST_RUNNING_THE_APP
```

For a worker behind a login host, use your configured SSH alias/ProxyJump to
reach that worker. The wrapper binds to `127.0.0.1`; forwarding is required
when the browser is on another machine. Choose another `--server.port` if 8501
is occupied. This port is independent of the evaluation policy server ports.

For another cache root (or one run directory):

```bash
bash scripts/visualize_ravla_retrieval.sh --server.port 8502 -- \
  --cache-root outputs/retrieval_cache/smoke_20261026
```

The cache directory can also be changed inside the sidebar. Source video and
memory paths must remain accessible on the machine running the app.

## Explore a rollout

1. Choose the run/subset, task, outcome filter and episode in the sidebar.
   Trials are displayed starting at 1; underlying task IDs remain zero-based.
2. Move the query slider or use Previous/Next. The app shows the exact sampled
   MP4 frame, simulator step, similarity and top-1/top-2 margin.
3. Compare the query against the ranked memory cards. Choose both cameras,
   external only, or wrist only. Adjust how many cached candidates to show.
4. Inspect the score timeline and the top-1 memory ID table to identify changes
   in retrieval. Under Debug, jump to queries with the smallest score margins.
5. Expand the action inspector to select a candidate, plot its normalized
   16-step action chunk, inspect the ranking table or download it as CSV.

The video player starts at the selected query, using the MP4's recorded frame
rate. Scrubbing/playing the video does **not** move the retrieval slider; use
the slider for exact frame alignment. Streamlit's [video player API](https://docs.streamlit.io/develop/api-reference/media/st.video)
provides the playback start time, while exact query selection is handled by the
app's slider.

## Interpretation and performance

- The caches are **recomputed retrieval from compressed MP4s**, not the
  historical retrieval trace used during evaluation. Close rankings can differ.
  The app displays both the cached K and original policy K. The current full
  caches contain top-10 candidates; the app does not compute additional ones.
- Memory IDs are local to the selected task. Cards display the cached segment
  observation, with Eagle normalization inverted using its image configuration.
  The official buffer contains two observation images and an action chunk per
  segment. It has no segment video or original demo ID/frame provenance.
- Action plots show the first seven stored normalized action dimensions
  (`x, y, z, roll, pitch, yaw, gripper`); the remaining dimensions are padding.
  These are demonstration actions, rather than actions executed in the rollout.
- The app CPU-maps memory files and reads only the selected candidates' images
  and actions. It never copies the entire memory to GPU or runs the retriever.
  Query frames are decoded once per episode and cached for slider navigation.
  Caches retain two decoded episodes and two mapped memory files at a time.
- Use Refresh caches if precompute is still running. Only completed episode
  files listed in the manifest are selectable. Each cache's source summary is
  used to populate task/trial/outcome labels. A changed memory file is rejected
  to avoid showing unrelated segment IDs.

## Checks

```bash
.venv-visualizer/bin/python -m pytest -q \
  tests/test_retrieval_viewer.py tests/test_precompute_retrieval.py
```

Tests cover exact decoded frame selection, camera splitting, inverse image
normalization, index validation and Streamlit query/episode/task/filter changes.
