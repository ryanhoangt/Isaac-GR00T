"""Streamlit viewer for pairs exported by the original RA-VLA DTW sampler."""

import argparse
import json
from pathlib import Path
import sys

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.rag import mine_dtw_pairs as mining  # noqa: E402
from scripts.rag import retrieval_viewer_data as media  # noqa: E402


@st.cache_data(max_entries=4, show_spinner=False)
def load_export(directory, metadata_token, pairs_token):
    root = Path(directory)
    return json.loads((root / "metadata.json").read_text()), mining.read_records(root / "pairs.jsonl")


def check_source(source):
    if media.file_token(source["path"]) != (source["bytes"], source["mtime_ns"]):
        raise ValueError(f"Source changed since mining: {source['path']}. Export to a new directory.")


@st.cache_data(max_entries=6, show_spinner="Decoding demonstration frames…")
def load_frames(path, token, indices, length):
    arrays = media.decode_frames(path, indices, length)
    return {index: image for index, image in zip(indices, arrays)}


@st.cache_data(max_entries=8, show_spinner=False)
def load_actions(path, token, action_column="action", state_column="observation.state"):
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=[action_column, state_column])
    return np.array(table[action_column].to_pylist()), np.array(table[state_column].to_pylist())


def move_pair(delta, count):
    st.session_state.pair_index = min(max(st.session_state.pair_index + delta, 0), count - 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-dir", default="outputs/dtw_pairs/gos_offline")
    args = parser.parse_args()
    st.set_page_config(page_title="RA-VLA · DTW positive pairs", page_icon="↔", layout="wide")
    st.title("DTW positive-pair explorer")
    st.caption("Inspect cross-demonstration frame matches produced by the original RA-VLA sampler.")
    with st.sidebar:
        directory = media.resolve_path(st.text_input("Pairs directory", args.pairs_dir))
        metadata_path, pairs_path = directory / "metadata.json", directory / "pairs.jsonl"
        if not metadata_path.is_file() or not pairs_path.is_file():
            st.info("Select an exported pair directory, or run scripts/mine_dtw_pairs.sh first.")
            st.stop()
        metadata, records = load_export(
            str(directory), media.file_token(metadata_path), media.file_token(pairs_path)
        )
        if st.button("Refresh pairs"):
            load_export.clear()
            st.rerun()
        if not records:
            st.info("No completed pairs yet.")
            st.stop()
        suite = st.selectbox("Suite", sorted({record["suite"] for record in records}))
        robomme = any("instruction" in row for row in records if row["suite"] == suite)
        tasks = {
            record["task"] if robomme else record["task_id"]: record["task"]
            for record in records
            if record["suite"] == suite
        }
        task = st.selectbox(
            "Task", sorted(tasks), format_func=lambda value: value if robomme else f"{value}: {tasks[value]}"
        )
        task_records = [
            record
            for record in records
            if record["suite"] == suite and (record["task"] if robomme else record["task_id"]) == task
        ]
        if robomme:
            instruction = st.selectbox("Instruction", sorted({row["instruction"] for row in task_records}))
            task_records = [row for row in task_records if row["instruction"] == instruction]
        selected = {record["id"]: record for record in task_records}
        identity = st.selectbox(
            "Demonstration pair",
            list(selected),
            format_func=lambda key: f"Demo {selected[key]['episode_a']} ↔ demo {selected[key]['episode_b']}",
        )
        record = selected[identity]
        cameras = [
            name
            for name in ("External", "Wrist")
            if name in record["source_a"]["videos"] and name in record["source_b"]["videos"]
        ]
        if not cameras:
            st.warning("These demonstrations have no accessible camera videos in the export.")
            st.stop()
        camera = st.radio("Camera", (["Both"] if len(cameras) == 2 else []) + cameras, horizontal=True)
        horizon = st.number_input("Action chunk length", min_value=1, max_value=64, value=16)
        st.caption(
            f"{len(records)} demonstration pairs · {sum(len(row['frame_pairs']) for row in records):,} positive candidates"
        )
        st.caption(
            f"Precompute: {metadata['status']} · held out: {metadata.get('held_out_suite') or 'not specified by export'}"
        )
    pairs = np.array(record["frame_pairs"], dtype=int)
    baseline = np.array(record["progress_frame_pairs"], dtype=int)
    if (
        pairs.ndim != 2
        or pairs.shape[1] != 2
        or baseline.shape != pairs.shape
        or not np.array_equal(pairs[:, 0], baseline[:, 0])
    ):
        raise ValueError("Invalid exported pair arrays")
    context = (str(directory), metadata["fingerprint"], identity)
    if st.session_state.get("pair_context") != context:
        st.session_state.pair_context = context
        st.session_state.pair_index = 0
    count = len(pairs)
    st.subheader(record["task"])
    if "instruction" in record:
        st.caption(record["instruction"])
    st.info(
        "Action-DTW positive candidates, not human-verified labels. Relative progress uses the same anchor at the same percentage of the other demonstration. Neither is a retriever prediction."
    )
    navigation = st.columns([1, 7, 1])
    navigation[0].button(
        "← Previous",
        on_click=move_pair,
        args=(-1, count),
        disabled=st.session_state.pair_index == 0,
        use_container_width=True,
    )
    with navigation[1]:
        if count > 1:
            st.slider("Sampled positive pair (zero-based)", 0, count - 1, key="pair_index")
        else:
            st.caption("One sampled positive pair")
    navigation[2].button(
        "Next →",
        on_click=move_pair,
        args=(1, count),
        disabled=st.session_state.pair_index == count - 1,
        use_container_width=True,
    )
    index = st.session_state.pair_index
    frame_a, frame_b = pairs[index]
    progress_b = baseline[index, 1]
    views = {}
    for side, wanted in (("source_a", pairs[:, 0]), ("source_b", np.r_[pairs[:, 1], baseline[:, 1]])):
        source = record[side]
        for name in cameras if camera == "Both" else [camera]:
            video = source["videos"][name]
            check_source(video)
            views[side, name] = load_frames(
                video["path"],
                media.file_token(video["path"]),
                tuple(sorted(set(map(int, wanted)))),
                source["length"],
            )

    columns = st.columns(3)
    cards = [
        ("Anchor", "source_a", frame_a),
        ("DTW positive", "source_b", frame_b),
        ("Relative progress", "source_b", progress_b),
    ]
    for column, (label, side, frame) in zip(columns, cards):
        images = [views[side, name][int(frame)] for name in cameras if camera == "Both" or name == camera]
        with column:
            st.subheader(label)
            st.image(
                np.concatenate(images, axis=1),
                caption=f"Demo {record[side]['episode']} · frame {frame} · {'external | wrist' if camera == 'Both' else camera.lower()}",
                use_container_width=True,
            )
    trajectories = {}
    schema = record.get("observation_schema", {})
    for side in ("source_a", "source_b"):
        parquet = record[side]["parquet"]
        check_source(parquet)
        trajectories[side] = load_actions(
            parquet["path"],
            media.file_token(parquet["path"]),
            schema.get("action_column", "action"),
            schema.get("state_column", "observation.state"),
        )
    actions_a, state_a = trajectories["source_a"]
    actions_b, state_b = trajectories["source_b"]

    def distance(target):
        if schema.get("state_metric") == "joint_rmse_rad":
            return float(np.sqrt(np.mean((state_a[frame_a, :7] - state_b[target, :7]) ** 2)))
        return float(np.linalg.norm(state_a[frame_a, :3] - state_b[target, :3]) * 100)

    def rmse(target):
        return float(np.sqrt(np.mean((actions_a[frame_a] - actions_b[target]) ** 2)))

    metrics = st.columns(4)
    metrics[0].metric("DTW action RMSE", f"{rmse(frame_b):.4f}")
    metrics[1].metric("Progress action RMSE", f"{rmse(progress_b):.4f}")
    if schema.get("state_metric") == "joint_rmse_rad":
        metrics[2].metric("DTW joint-state RMSE", f"{distance(frame_b):.4f} rad")
        metrics[3].metric("Progress joint-state RMSE", f"{distance(progress_b):.4f} rad")
        st.caption(
            "RoboMME state contains seven joint angles and a gripper coordinate; joint-state RMSE uses only the seven angles. Raw action RMSE mixes joint and gripper coordinates."
        )
    else:
        metrics[2].metric("DTW end-effector distance", f"{distance(frame_b):.1f} cm")
        metrics[3].metric("Progress end-effector distance", f"{distance(progress_b):.1f} cm")
    st.caption(
        "Action similarity is what this DTW optimizes; lower action error alone does not prove correct semantic matching. Robot-state similarity also does not describe object state."
    )
    chart, actions = st.columns(2)
    with chart:
        st.subheader("Sampled frame correspondences")
        table = pd.DataFrame(
            {"anchor_frame": pairs[:, 0], "DTW": pairs[:, 1], "Relative progress": baseline[:, 1]}
        )
        long = table.melt("anchor_frame", var_name="method", value_name="target_frame")
        lines = (
            alt.Chart(long)
            .mark_line(point=True)
            .encode(
                x=alt.X("anchor_frame:Q", title="Demo A frame"),
                y=alt.Y("target_frame:Q", title="Demo B frame"),
                color="method:N",
                tooltip=["anchor_frame:Q", "target_frame:Q", "method:N"],
            )
        )
        rule = (
            alt.Chart(pd.DataFrame({"anchor_frame": [int(frame_a)]}))
            .mark_rule(color="orange")
            .encode(x="anchor_frame:Q")
        )
        st.altair_chart((lines + rule).properties(height=230), use_container_width=True)
        st.caption("Sampled correspondences only; the export does not recompute or store the full DTW path.")
    with actions:
        st.subheader("Raw demonstration actions")
        action_names = schema.get("action_names", media.ACTION_NAMES)
        if len(action_names) != actions_a.shape[1] or actions_a.shape[1] != actions_b.shape[1]:
            raise ValueError("Action names/dimensions disagree with exported schema")
        dimension = st.selectbox(
            "Action dimension", list(range(len(action_names))), format_func=lambda dim: action_names[dim]
        )
        available = min(horizon, len(actions_a) - frame_a, len(actions_b) - frame_b, len(actions_b) - progress_b)
        traces = pd.DataFrame(
            {
                "Anchor": actions_a[frame_a : frame_a + available, dimension],
                "DTW": actions_b[frame_b : frame_b + available, dimension],
                "Relative progress": actions_b[progress_b : progress_b + available, dimension],
            }
        ).rename_axis("chunk step")
        st.line_chart(traces, use_container_width=True)
        st.caption(
            f"{available} common available steps (requested {horizon}); raw dataset actions, not model outputs."
        )
    with st.expander("Frame IDs and provenance"):
        st.dataframe(table, hide_index=True, use_container_width=True)
        st.json(record, expanded=False)
        st.download_button(
            "Download this demonstration pair (JSON)",
            json.dumps(record, indent=2),
            file_name=f"{suite}_ep{record['episode_a']}_ep{record['episode_b']}.json",
            mime="application/json",
        )


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, IndexError) as error:
        st.error(str(error))
        st.caption("Check the export directory and the source Parquet/video paths in pairs.jsonl.")
