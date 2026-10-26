"""Streamlit explorer for precomputed LIBERO retrieval; start with the bash wrapper."""

import argparse
import json
from pathlib import Path
import sys

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import torch

# Streamlit executes this file as a script, outside the scripts.rag package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.rag import retrieval_viewer_data as data  # noqa: E402


@st.cache_data(max_entries=16, show_spinner=False)
def load_run(path, manifest_token, summary_token):
    return data.read_run(path)


@st.cache_data(max_entries=16, show_spinner=False)
def load_episode(path, token):
    return data.read_episode(path)


@st.cache_data(max_entries=2, show_spinner="Decoding rollout query frames…")
def load_frames(path, token, frame_indices, expected_frames):
    return data.decode_frames(path, frame_indices, expected_frames)


@st.cache_resource(max_entries=2, show_spinner="Opening task memory on CPU…")
def load_memory(path, token):
    # Read-only use of CPU-mapped tensors. Never copy backbone features or run a model.
    return torch.load(path, map_location="cpu", mmap=True, weights_only=True)


def move_query(delta, count):
    st.session_state.query = min(max(st.session_state.query + delta, 0), count - 1)


def pick_query(index):
    st.session_state.query = int(index)


def timeline(episode, query):
    trace = data.score_trace(episode)
    score_columns = [name for name in ("top1_score", "top2_score") if name in trace]
    long = trace.melt(
        id_vars=["frame", "top1_memory"], value_vars=score_columns, var_name="rank", value_name="score"
    )
    lines = (
        alt.Chart(long)
        .mark_line(point=True)
        .encode(
            x=alt.X("frame:Q", title="MP4 frame"),
            y=alt.Y("score:Q", scale=alt.Scale(zero=False), title="Similarity"),
            color=alt.Color("rank:N", title=None),
            tooltip=["frame:Q", "rank:N", alt.Tooltip("score:Q", format=".5f"), "top1_memory:Q"],
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"frame": [int(episode["frame_indices"][query])]}))
        .mark_rule(color="#e06030")
        .encode(x="frame:Q")
    )
    st.altair_chart((lines + rule).properties(height=190), use_container_width=True)
    st.caption("Hover for scores and top-1 memory ID. The orange line marks the selected query.")
    with st.expander("Top-1 segment and score margin over time"):
        st.dataframe(trace, hide_index=True, use_container_width=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", default="outputs/retrieval_cache")
    args = parser.parse_args()
    st.set_page_config(page_title="RA-VLA · Retrieval explorer", page_icon="🔎", layout="wide")
    st.title("RA-VLA retrieval explorer")
    st.caption("Inspect rollout observations alongside retrieved memory segments.")

    with st.sidebar:
        st.header("Rollout selection")
        root = st.text_input("Cache directory", args.cache_root)
        if st.button("Refresh caches"):
            load_run.clear()
            load_episode.clear()
        directories = data.discover_runs(root)
        if not directories:
            st.info("No retrieval manifests found. Select a cache directory or run precompute first.")
            st.stop()
        run_path = st.selectbox("Run / subset", directories, format_func=lambda value: Path(value).name)
        manifest_path = Path(run_path) / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        summary_path = data.resolve_path(manifest["run"]) / "rollouts/summary.json"
        run = load_run(run_path, data.file_token(manifest_path), data.file_token(summary_path))
        episodes = run["episodes"]
        if not episodes:
            st.info("No completed episode caches in this run yet.")
            st.stop()
        tasks = {episode["task_id"]: episode["task_description"] for episode in episodes}
        task_id = st.selectbox("Task", sorted(tasks), format_func=lambda task: f"{task}: {tasks[task]}")
        outcome = st.radio("Outcome", ["All", "Success", "Failure"], horizontal=True)
        selected = [
            ep
            for ep in episodes
            if ep["task_id"] == task_id and (outcome == "All" or ep["success"] == (outcome == "Success"))
        ]
        if not selected:
            st.info("No cached episodes match this task and outcome.")
            st.stop()
        by_ordinal = {episode["ordinal"]: episode for episode in selected}

        def episode_label(ordinal):
            ep = by_ordinal[ordinal]
            return f"Trial {ep['episode_index'] + 1:02d} · {'success' if ep['success'] else 'failure'} · {ep['control_steps']} steps"

        ordinal = st.selectbox("Episode", list(by_ordinal), format_func=episode_label)
        entry = by_ordinal[ordinal]
        st.caption(f"{len(episodes)} cached episodes · {sum(ep['queries'] for ep in episodes):,} queries")
        if manifest["status"] != "complete":
            st.info(f"Precompute status: {manifest['status']}. Showing completed caches.")
        camera = st.radio("Camera", ["Both", "External", "Wrist"], horizontal=True)

    episode = load_episode(entry["cache"], data.file_token(entry["cache"]))
    metadata = episode["metadata"]
    count, available_k = episode["topk_scores"].shape
    context = (entry["cache"], data.file_token(entry["cache"]))
    if st.session_state.get("episode_context") != context:
        st.session_state.episode_context = context
        st.session_state.query = 0
    st.session_state.query = min(st.session_state.query, count - 1)
    with st.sidebar:
        display_k = st.number_input(
            "Retrieved candidates shown", min_value=1, max_value=available_k, value=min(6, available_k)
        )
        show_video = st.checkbox("Show rollout video", value=True)

    st.subheader(tasks[task_id])
    st.caption(
        f"{metadata['suite']} · trial {entry['episode_index'] + 1} · {'SUCCESS' if entry['success'] else 'FAILURE'} · {entry['control_steps']} control steps"
    )
    st.info(
        f"Retrieval recomputed from saved MP4. Cached top-{available_k}; original policy K={metadata['policy_k']}. Compression and batched inference can change close rankings."
    )
    navigation = st.columns([1, 7, 1])
    navigation[0].button(
        "← Previous",
        on_click=move_query,
        args=(-1, count),
        disabled=st.session_state.query == 0,
        use_container_width=True,
    )
    with navigation[1]:
        if count > 1:
            st.slider("Retrieval query (zero-based)", 0, count - 1, key="query")
        else:
            st.caption("One retrieval query: frame 0")
    navigation[2].button(
        "Next →",
        on_click=move_query,
        args=(1, count),
        disabled=st.session_state.query == count - 1,
        use_container_width=True,
    )
    query = st.session_state.query
    frame_index = int(episode["frame_indices"][query])
    scores, indices = episode["topk_scores"][query], episode["topk_indices"][query]
    metrics = st.columns(4)
    metrics[0].metric("Query", f"{query + 1} / {count}")
    metrics[1].metric("MP4 frame / simulator step", f"{frame_index} / {int(episode['simulator_steps'][query])}")
    metrics[2].metric("Top-1 similarity", f"{scores[0]:.5f}")
    metrics[3].metric("Top-1 − top-2", f"{scores[0] - scores[1]:.5f}" if available_k > 1 else "N/A")

    video = data.resolve_path(metadata["episode"]["video"]["path"])
    frames = load_frames(
        str(video),
        data.file_token(video),
        tuple(int(value) for value in episode["frame_indices"]),
        metadata["decoded_frames"],
    )
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Query observation")
        st.image(
            data.rollout_view(frames[query], camera),
            caption=f"Frame {frame_index} · {'external | wrist' if camera == 'Both' else camera.lower()}",
            use_container_width=True,
        )
    with right:
        st.subheader("Retrieval across the rollout")
        timeline(episode, query)
    if show_video:
        with st.expander("Play rollout video", expanded=True):
            fps = metadata.get("video_fps") or 30
            st.video(str(video), start_time=float(frame_index / fps))
            st.caption(
                "Video starts at the selected query. Player scrubbing does not update the query slider; use the slider for exact retrieval alignment."
            )

    memory_path = data.resolve_path(metadata["memory"])
    saved_memory = run["manifest"]["signature"]["memory"]
    if data.file_token(memory_path) != (saved_memory["bytes"], saved_memory["mtime_ns"]):
        st.error(
            "Memory file changed since precompute. Regenerate the cache before inspecting retrieved segments."
        )
        st.stop()
    memory = load_memory(str(memory_path), data.file_token(memory_path))
    task = metadata["episode"]["task_description"]
    if task not in memory:
        raise ValueError(f"Task missing from memory: {task}")
    task_memory = memory[task]
    if len(task_memory["pixel_values"]) != metadata["memory_candidates"]:
        raise ValueError("Memory candidate count differs from the precomputed cache")
    config = json.loads(data.IMAGE_CONFIG.read_text())

    st.subheader("Retrieved memory segments")
    st.caption(
        "Each card shows the segment's cached observation, external | wrist. The memory stores an observation and an action chunk, rather than a segment video or source demonstration/frame IDs."
    )
    for start in range(0, display_k, 3):
        columns = st.columns(3)
        for column, rank in zip(columns, range(start, min(start + 3, display_k))):
            index = int(indices[rank])
            views = data.memory_images(
                task_memory["pixel_values"][index], config["image_mean"], config["image_std"]
            )
            with column, st.container(border=True):
                st.markdown(f"**#{rank + 1} · memory {index}**")
                st.caption(f"Similarity {scores[rank]:.5f}")
                st.image(data.paired_views(views, camera), use_container_width=True)

    with st.expander("Inspect retrieved actions and ranking table"):
        rank = st.selectbox(
            "Candidate rank",
            list(range(available_k)),
            format_func=lambda value: f"#{value + 1} · memory {indices[value]}",
        )
        action = task_memory["action"][int(indices[rank])].detach().float().cpu().numpy()
        action_frame = pd.DataFrame(action[:, :7], columns=data.ACTION_NAMES).rename_axis("chunk step")
        st.caption(
            "Stored normalized demonstration actions for this segment (first 7 dimensions; remaining dimensions are padding). These are not the rollout's executed actions."
        )
        st.line_chart(action_frame, use_container_width=True)
        st.dataframe(action_frame, use_container_width=True)
        ranking = pd.DataFrame(
            {"rank": np.arange(1, available_k + 1), "memory_index": indices, "similarity": scores}
        )
        st.dataframe(ranking, hide_index=True, use_container_width=True)
        st.download_button(
            "Download this query's ranking (CSV)",
            ranking.to_csv(index=False),
            file_name=f"episode_{ordinal:06d}_frame_{frame_index}_retrieval.csv",
            mime="text/csv",
        )

    with st.expander("Debug: low margins and provenance"):
        trace = data.score_trace(episode)
        if "margin" in trace:
            candidates = trace.nsmallest(min(10, count), "margin")
            labels = [f"Frame {int(row.frame)} · margin {row.margin:.5f}" for row in candidates.itertuples()]
            target = st.selectbox(
                "Queries with smallest top-1/top-2 margin",
                list(candidates.index),
                format_func=lambda index: labels[list(candidates.index).index(index)],
            )
            st.button("Jump to query", on_click=pick_query, args=(target,))
        st.json(metadata, expanded=False)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, IndexError) as error:
        st.error(str(error))
        st.caption("Check the cache directory and the source video/memory paths in the episode metadata.")
