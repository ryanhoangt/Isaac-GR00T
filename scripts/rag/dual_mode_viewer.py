"""Streamlit view of historical dual-mode context; no inference or simulation."""

from pathlib import Path

import altair as alt
import av
import numpy as np
import pandas as pd
import streamlit as st

from scripts.rag import dual_mode_viewer_data as data
from scripts.rag.retrieval_viewer_data import ACTION_NAMES, paired_views

AV_ERROR = getattr(av, "FFmpegError", getattr(av, "AVError", RuntimeError))


@st.cache_data(max_entries=12, show_spinner="Reading completed rollout logs…")
def index(path, signature):
    return data.read_index(path)


@st.cache_data(max_entries=12, show_spinner=False)
def episode(path, signature):
    return data.read_episode(path)


@st.cache_resource(max_entries=2, show_spinner="Opening reference provenance on CPU…")
def reference(path, signature):
    return data.load_reference(path)


@st.cache_data(max_entries=96, show_spinner=False)
def frame(path, signature, position, fps):
    return data.decode_frame(path, position, fps)


def pick(position):
    st.session_state.dual_query = int(position)


def timeline(value, query_index):
    rows = pd.DataFrame(data.trace_rows(value))
    base = alt.Chart(rows)
    shade = (
        base.transform_filter(alt.datum.mode == "goal")
        .mark_rect(color="#e8b557", opacity=0.2)
        .encode(x="step:Q", x2="end:Q")
    )
    lines = base.mark_line(point=True, color="#285d85").encode(
        x=alt.X("step:Q", title="Control step"),
        y=alt.Y("support:Q", title="1 − cosine", scale=alt.Scale(zero=False)),
        tooltip=[
            "step:Q",
            "mode:N",
            "event:N",
            alt.Tooltip("support:Q", format=".5f"),
            "nearest_demo:Q",
            "nearest_frame:Q",
            "anchor_demo:Q",
            "anchor_frame:Q",
            "goal_frame:Q",
        ],
    )
    layers = [shade, lines]
    if rows["enter"].notna().any():
        for key, color, dash in (("enter", "#b24a45", [6, 3]), ("exit", "#378b80", [2, 3])):
            layers.append(base.mark_line(color=color, strokeDash=dash).encode(x="step:Q", y=alt.Y(key + ":Q")))
    marker = (
        alt.Chart(pd.DataFrame({"step": [value["queries"][query_index]["control_step"]]}))
        .mark_rule(color="#d66d20")
        .encode(x="step:Q")
    )
    st.altair_chart(alt.layer(*layers, marker).properties(height=220), use_container_width=True)
    st.caption(
        "Blue: recorded support distance · dashed red: entry threshold · dotted green: exit threshold · shaded: goal-mode chunks."
    )
    return rows


def render(root):
    st.title("Dual-mode RA-VLA · Rollout debugger")
    st.caption("Historical context from rollout logs. No retrieval recomputation, model loading or simulator.")
    with st.sidebar:
        root = st.text_input("Evaluation directory", root)
        if st.button("Refresh rollouts / videos"):
            index.clear()
            episode.clear()
            frame.clear()
            reference.clear()
        runs = data.discover_runs(root)
        if not runs:
            st.info("No dual-mode evaluation manifests found.")
            return
        preferred = next(
            (i for i, p in enumerate(runs) if Path(p).name == "support_gate_10trials_15k_20261031"), 0
        )
        selected_run = st.selectbox("Run", runs, index=preferred, format_func=lambda p: Path(p).name)
    manifest_path = Path(selected_run) / "manifest.json"
    run = index(selected_run, data.token(manifest_path))
    entries = run["episodes"]
    if not entries:
        st.info("No completed episodes with query logs yet.")
        return
    with st.sidebar:
        conditions = list(dict.fromkeys(e["condition"] for e in entries))
        condition = st.selectbox(
            "Condition", conditions, index=conditions.index("adaptive_q90") if "adaptive_q90" in conditions else 0
        )
        entries = [e for e in entries if e["condition"] == condition]
        tasks = {e["task_id"]: e["task"] for e in entries}
        task = st.selectbox("Task", sorted(tasks), format_func=lambda t: f"{t}: {tasks[t]}")
        outcome = st.radio("Outcome", ["All", "Success", "Failure"], horizontal=True)
        entries = [
            e
            for e in entries
            if e["task_id"] == task and (outcome == "All" or e["success"] == (outcome == "Success"))
        ]
        if not entries:
            st.info("No completed episodes match this task and outcome.")
            return
        by_path = {e["path"]: e for e in entries}
        path = st.selectbox(
            "Episode",
            list(by_path),
            key=f"dual_episode:{selected_run}:{condition}:{task}:{outcome}",
            format_func=lambda p: (
                f"Init {by_path[p]['initial_index']} · {'success' if by_path[p]['success'] else 'failure'} · {by_path[p]['control_steps']} steps"
            ),
        )
        camera = st.radio("Camera", ["Both", "External", "Wrist"], horizontal=True)
        show_video = st.checkbox("Play rollout video", value=False)
        st.caption("Task and initial-state indices are zero-based.")
    value = episode(path, data.token(path))
    context_key = (path, data.token(path))
    if st.session_state.get("dual_episode_context") != context_key:
        st.session_state.dual_episode_context = context_key
        st.session_state.dual_query = 0
    queries = value["queries"]
    st.session_state.dual_query = min(st.session_state.dual_query, len(queries) - 1)
    nav = st.columns([1, 6, 1])
    nav[0].button(
        "← Previous",
        on_click=pick,
        args=(max(0, st.session_state.dual_query - 1),),
        disabled=st.session_state.dual_query == 0,
    )
    with nav[1]:
        if len(queries) > 1:
            st.slider("Recorded query", 0, len(queries) - 1, key="dual_query")
        else:
            st.caption("One recorded query")
    nav[2].button(
        "Next →",
        on_click=pick,
        args=(min(len(queries) - 1, st.session_state.dual_query + 1),),
        disabled=st.session_state.dual_query == len(queries) - 1,
    )
    position = st.session_state.dual_query
    query = queries[position]
    switches = [i for i in range(1, len(queries)) if queries[i]["mode"] != queries[i - 1]["mode"]]
    later = next((i for i in switches if i > position), None)
    st.button(
        "Next mode switch", on_click=pick, args=(later if later is not None else position,), disabled=later is None
    )
    st.subheader(value["task"])
    metrics = st.columns(4)
    metrics[0].metric("Control step", query["control_step"])
    metrics[1].metric("Active mode", "Mode 2 · goal" if query["mode"] == "goal" else "Mode 1 · action")
    metrics[2].metric("Support distance", f"{query['support_fp32']:.5f}")
    metrics[3].metric("Gate event", query.get("gate_event", query.get("phase", query["mode"])))
    st.caption("Recorded top-1 context (K=1); no historical top-k rankings were saved for these runs.")
    video = None
    try:
        video = data.resolve_video(path, value)
    except (OSError, ValueError, KeyError) as error:
        st.error(f"Video provenance check: {error}")
    memory = data.reference_manifest(run)
    record = data.task_record(memory, value["task"])
    memory_path = Path(memory["directory"]) / record["file"]
    entry = reference(str(memory_path), data.token(memory_path))
    contexts = data.context_rows(value, query, entry, record)
    panels = st.columns(4)
    with panels[0]:
        st.markdown("**Current observation**")
        if video:
            try:
                image = frame(video["path"], data.token(video["path"]), query["control_step"], video["fps"])
                st.image(paired_views(data.rollout_views(image), camera), use_container_width=True)
                st.caption(f"{video['origin']} · frame {query['control_step']}")
            except (OSError, ValueError, AV_ERROR) as error:
                st.error(f"Cannot read current frame: {error}")
        else:
            st.info("Video not available yet. Reference context and logs remain inspectable.")
    titles = [
        "Nearest reference now",
        "Active anchor",
        "Active goal" if contexts["goal_active"] else "Candidate goal · inactive",
    ]
    for panel, key, title in zip(panels[1:], ("nearest", "anchor", "goal"), titles):
        row = contexts[key]
        with panel:
            st.markdown(f"**{title}**")
            try:
                images = []
                for cam in data.CAMERAS:
                    source, fps = data.reference_video(memory, row, cam)
                    images.append(frame(source, data.token(source), row["frame"], fps))
                st.image(paired_views(np.stack(images), camera), use_container_width=True)
            except (OSError, ValueError, AV_ERROR) as error:
                st.error(f"Reference frame unavailable: {error}")
            st.caption(f"Demo {row['episode']} · frame {row['frame']} · memory row {row['index']}")
    if contexts["goal_active"]:
        st.info(
            f"Head uses the active goal image; expert action tokens are masked. Milestone: +{contexts['goal']['frame'] - contexts['anchor']['frame']} frames from the active anchor. The anchor/goal may stay pinned while nearest reference changes."
        )
    else:
        st.caption(
            "Head uses action context from the active anchor. The candidate goal is shown for inspection and is not active."
        )
    if video and "replay" in video["origin"].lower():
        st.caption(
            "Current frames are reconstructed from saved actions. EE/success/predicate checks passed; complete object-state equality was not recorded."
        )
    rows = timeline(value, position)
    if show_video and video:
        st.video(video["path"], start_time=float(query["control_step"] / video["fps"]))
        st.caption("The player starts at the selected query. Player scrubbing does not move the query slider.")
    with st.expander("Actions and query table"):
        predicted, expert = data.action_comparison(query, entry)
        action_rows = []
        for label, chunk in (("Predicted", predicted), ("Anchor expert", expert)):
            for step, values in enumerate(chunk):
                action_rows.extend(
                    dict(step=step, channel=name, value=float(v), source=label)
                    for name, v in zip(ACTION_NAMES, values)
                )
        chart = (
            alt.Chart(pd.DataFrame(action_rows))
            .mark_line()
            .encode(x="step:Q", y=alt.Y("value:Q", title="Normalized action"), color="source:N")
            .facet(facet=alt.Facet("channel:N", sort=ACTION_NAMES), columns=4)
        )
        st.altair_chart(chart, use_container_width=True)
        st.caption(
            "Both chunks use stored policy normalization. In goal mode the anchor expert actions are not supplied to the head; they are diagnostic only. Executed commands also include action conversion/gripper processing."
        )
        st.dataframe(rows, hide_index=True, use_container_width=True)
        st.download_button(
            "Download query trace CSV",
            rows.to_csv(index=False),
            file_name=f"task{task}_init{value['initial_index']}_queries.csv",
            mime="text/csv",
        )
    with st.expander("Context provenance"):
        st.json(
            dict(
                episode=path,
                memory=str(memory_path),
                dataset=memory["dataset_path"],
                context=contexts,
                flow_seed=query.get("flow_seed"),
            ),
            expanded=False,
        )
