"""Streamlit shell for synchronized, CPU-only rollout inspection."""

import argparse
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import streamlit as st  # noqa: E402

from scripts.libero_vfd_viewer import (  # noqa: E402
    ASSETS,
    build_payload,
    discover,
    export_csv,
    matching_calibrations,
    read_scores,
    standalone_html,
)


@st.cache_data(max_entries=16, show_spinner=False)
def cached_payload(runs, key, selections, file_versions):
    return build_payload(runs, key, selections)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=REPO / "outputs/libero_vfd")
    args, _ = parser.parse_known_args()
    st.set_page_config(page_title="LIBERO uncertainty inspector", layout="wide")
    st.title("LIBERO uncertainty inspector")
    st.caption(
        "Inspect recorded observations alongside velocity disagreement. CPU only; existing recordings are read-only."
    )
    root = Path(st.sidebar.text_input("Recording root", str(args.root))).expanduser()
    if st.sidebar.button("Refresh recordings"):
        cached_payload.clear()
    runs, calibrations, errors = discover(root)
    if errors:
        with st.expander("Files that could not be indexed"):
            st.write(errors)
    if not runs:
        st.info("No complete policy episodes found. Choose a directory containing recording manifests.")
        return
    by_path = {run["path"]: run for run in runs}
    complete = [run for run in runs if run["manifest"]["status"] == "completed"]
    defaults = sorted(complete or runs, key=lambda run: run["manifest"].get("created_utc", ""))[-2:]
    options = list(by_path)
    left = st.sidebar.selectbox(
        "Left pair",
        options,
        index=options.index(defaults[0]["path"]),
        format_func=lambda path: by_path[path]["label"] + " · " + Path(path).name,
    )
    compare = st.sidebar.checkbox("Compare a second pair", value=len(defaults) > 1)
    selected = [by_path[left]]
    if compare:
        remaining = [path for path in options if path != left]
        if remaining:
            default_right = defaults[-1]["path"] if defaults[-1]["path"] in remaining else remaining[-1]
            right = st.sidebar.selectbox(
                "Right pair",
                remaining,
                index=remaining.index(default_right),
                format_func=lambda path: by_path[path]["label"] + " · " + Path(path).name,
            )
            selected.append(by_path[right])
    choices = []
    for index, run in enumerate(selected):
        st.sidebar.markdown(f"**{'Left' if index == 0 else 'Right'} thresholds**")
        chosen = {}
        for kind in ("source", "target"):
            matches = matching_calibrations(run, calibrations, kind)
            items = {item["path"]: item for item in matches}
            paths = [None, *items]
            path = st.sidebar.selectbox(
                f"{kind.capitalize()} calibration",
                paths,
                index=len(paths) - 1,
                key=f"calibration_{index}_{kind}_{run['path']}",
                format_func=lambda value: "Unavailable / hidden" if value is None else Path(value).parent.name,
            )
            chosen[kind] = items.get(path)
        choices.append(chosen)
    common = set(selected[0]["episodes"])
    for run in selected[1:]:
        common &= set(run["episodes"])
    if not common:
        st.info("These runs have no complete episodes with matching suite/task/initial-state IDs.")
        return
    suites = sorted({key[0] for key in common})
    suite = st.sidebar.selectbox("Suite", suites)
    tasks = sorted({key[1] for key in common if key[0] == suite})
    task = st.sidebar.selectbox("Task", ["All", *tasks])
    outcome = st.sidebar.selectbox("Outcome in left recording", ["All", "Failure", "Success"])
    keys = sorted(
        key
        for key in common
        if key[0] == suite
        and (task == "All" or key[1] == task)
        and (outcome == "All" or selected[0]["episodes"][key]["metadata"]["success"] == (outcome == "Success"))
    )
    if not keys:
        st.info("No episodes match these filters.")
        return
    key = st.selectbox(
        "Rollout",
        keys,
        format_func=lambda value: f"Task {value[1]:02d} · initial state {value[2]:02d} · "
        + ("success" if selected[0]["episodes"][value]["metadata"]["success"] else "failure"),
    )
    versions = []
    for run, chosen in zip(selected, choices):
        path = Path(run["episodes"][key]["path"])
        for file in [
            Path(run["path"]) / "manifest.json",
            path,
            path.with_suffix(".npz"),
            path.with_suffix(".mp4"),
            *(Path(item["path"]) for item in chosen.values() if item),
        ]:
            if file.is_file():
                stat = file.stat()
                versions.append((str(file), stat.st_size, stat.st_mtime_ns))
    try:
        with st.spinner("Loading this episode's video and uncertainty…"):
            payload = cached_payload(selected, key, choices, versions)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        st.error(f"Cannot display this episode: {exc}")
        return
    component = st.components.v2.component(
        "libero_vfd_timeline", css=(ASSETS / "viewer.css").read_text(), js=(ASSETS / "viewer.js").read_text()
    )
    component(data=payload, key=f"episode_{key}_{','.join(run['path'] for run in selected)}")
    st.caption(
        "Ground-truth success/failure is the observed final outcome, not a threshold prediction. "
        "First alarm does not identify the true failure-onset time. Demo/test initial-state overlap is unverified."
    )
    with st.expander("Export and provenance"):
        a, b = st.columns(2)
        a.download_button(
            "Download interactive HTML (videos included)",
            standalone_html(payload),
            file_name=f"vfd_task{key[1]:02d}_init{key[2]:02d}.html",
            mime="text/html",
        )
        b.download_button(
            "Download uncertainty CSV",
            export_csv(payload),
            file_name=f"vfd_task{key[1]:02d}_init{key[2]:02d}.csv",
            mime="text/csv",
        )
        st.json(
            {
                "episode": payload["episode"],
                "actions_match": payload["same_actions"],
                "pairs": [
                    {
                        "run": panel["run"],
                        "thresholds": panel["thresholds"],
                        "calibration_paths": panel["calibration_paths"],
                        "video_fps": panel["video_fps"],
                    }
                    for panel in payload["panels"]
                ],
            }
        )
    with st.expander("Episode index"):
        rows = []
        for item_key in keys:
            row = {"task": item_key[1], "initial_state": item_key[2]}
            for i, run in enumerate(selected):
                entry = run["episodes"][item_key]
                row[f"{'Left' if i == 0 else 'Right'} outcome"] = (
                    "success" if entry["metadata"]["success"] else "failure"
                )
                row[f"{'Left' if i == 0 else 'Right'} peak VFD"] = float(max(read_scores(entry["path"])))
            rows.append(row)
        st.dataframe(rows, hide_index=True)


if __name__ == "__main__":
    main()
