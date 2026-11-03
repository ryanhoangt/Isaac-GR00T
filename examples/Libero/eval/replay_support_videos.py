"""Render saved LIBERO diagnosis/support-eval actions without a model or policy server.

Run on a Slurm worker with the LIBERO Python environment. Original evaluation
episode JSONs/traces and original videos stay unchanged. Validated reconstructions
are published as EPISODE/rollout.mp4 with replay_validation.json; render artifacts
live under RUN/video_replay. Action replay is checked against saved EE paths and predicates;
full simulator states were not recorded, so exact object-state replay is not proven.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import errno
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import time
import uuid

import numpy as np


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_episode(directory):
    """Check saved inputs before importing the simulator."""
    episode = json.loads((directory / "episode.json").read_text())
    if not episode.get("complete"):
        raise ValueError("Incomplete source episode: " + str(directory))
    count = episode["control_steps"]
    queries = episode["queries"]
    with np.load(directory / "trace.npz", allow_pickle=False) as trace:
        actions = trace["executed_actions"].copy()
        ee = trace["ee"].copy()
        modes = trace["goal_mode"].copy() if "goal_mode" in trace else None
    if modes is None:
        # Diagnosis traces predate the per-control-step mode array.
        modes = np.zeros(count, dtype=np.int8)
        for index, query in enumerate(queries):
            end = queries[index + 1]["control_step"] if index + 1 < len(queries) else count
            modes[query["control_step"] : end] = query.get("mode", query.get("phase")) == "goal"
    if actions.shape != (count, 7) or ee.shape != (count + 1, 3) or modes.shape != (count,):
        raise ValueError("Trace shape does not match episode: " + str(directory))
    if not np.isfinite(actions).all() or not np.isfinite(ee).all():
        raise ValueError("Nonfinite trace: " + str(directory))
    steps = [q["control_step"] for q in queries]
    if not steps or steps[0] != 0 or steps != sorted(set(steps)) or steps[-1] >= count:
        raise ValueError("Invalid query steps")
    return episode, actions, ee, modes


def frame_labels(query, step, mode):
    event = query.get("gate_event", query.get("phase", mode))
    score = f"d={query['support_fp32']:.4f}"
    if "threshold_enter" in query:
        score += f" enter={query['threshold_enter']:.4f} exit={query['threshold_exit']:.4f}"
    nearest_episode = query.get("nearest_episode", query["retrieved_episode"])
    nearest_frame = query.get("nearest_frame", query.get("retrieved_frame", "?"))
    return [
        "ACTION REPLAY | overlay from original evaluation log",
        f"t={step} mode={mode} event={event}",
        score,
        f"nearest={nearest_episode}:{nearest_frame} goal={query['retrieved_episode']}:{query['goal_frame']}",
    ]


def publish_replay(directory, output, report):
    """Expose a verified reconstruction at the same path as an original video.

    Hardlinks avoid a second video copy on the shared filesystem. Source metadata
    stays immutable so original result/provenance hashes remain valid.
    """
    if not report.get("validated"):
        raise ValueError("Unverified replay cannot be published as rollout.mp4")
    directory, output = Path(directory), Path(output)
    source = output / report["video"]
    destination = directory / "rollout.mp4"
    sidecar = directory / "replay_validation.json"
    expected = {name: file_hash(directory / name) for name in ("episode.json", "trace.npz")}
    if expected != report["source_sha256"]:
        raise ValueError("Replay source changed before publication")
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists():
        if not sidecar.exists():
            # An original video always takes precedence, including --include-existing.
            return "original_preserved"
        saved = json.loads(sidecar.read_text())
        if not saved.get("validated") or saved.get("source_sha256") != expected:
            raise ValueError("Existing published replay has different provenance")
        if saved.get("published_video_sha256") != file_hash(destination):
            raise ValueError("Existing published replay video changed")
        return "already_published"
    video_hash = file_hash(source)
    try:
        os.link(source, destination)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        # Copy to the destination filesystem, then publish without replacing a file.
        temporary = directory / (".replay-publish-" + uuid.uuid4().hex + ".mp4")
        try:
            shutil.copyfile(source, temporary)
            os.link(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    publication = dict(report)
    publication.update(
        video="rollout.mp4",
        origin="saved_action_replay",
        artifact_video=str(source.resolve()),
        artifact_validation=str((output / "validation.json").resolve()),
        published_video_sha256=video_hash,
    )
    atomic_json(sidecar, publication)
    print(f"PUBLISHED {destination}", flush=True)
    return "published"


def cached_replay(job):
    directory, output, tolerance, fps, warmup_prior = job
    directory, output = Path(directory), Path(output)
    path = output / "validation.json"
    if not path.exists():
        return None
    old = json.loads(path.read_text())
    if not old.get("validated"):
        return None
    source = {name: file_hash(directory / name) for name in ("episode.json", "trace.npz")}
    if (
        old.get("source_sha256") != source
        or old.get("ee_tolerance_m") != tolerance
        or old.get("fps") != fps
        or old.get("warmup_prior", False) != warmup_prior
    ):
        raise ValueError("Replay settings/source changed; choose a fresh --output directory")
    if (output / old["video"]).is_file():
        publish_replay(directory, output, old)
        return dict(episode=str(directory), status="resumed", max_ee_error_m=old["max_ee_error_m"])
    return None


def advance_episode(env, suite, directory):
    """Reproduce preceding trials without encoding a video."""
    from examples.Libero.eval.utils import get_libero_dummy_action
    from gr00t.rag.diagnosis_protocol import array_digest

    episode, actions, reference, _ = load_episode(directory)
    initial = suite.get_task_init_states(episode["task_id"])[episode["initial_index"]]
    if array_digest(initial) != episode["initial_sha256"]:
        raise ValueError("Warmup initial state differs")
    env.reset()
    obs = env.set_init_state(initial)
    for _ in range(10):
        obs, _, _, _ = env.step(get_libero_dummy_action())
    maximum = float(np.linalg.norm(np.asarray(obs["robot0_eef_pos"]) - reference[0]))
    for step, action in enumerate(actions):
        obs, _, _, _ = env.step(action.tolist())
        maximum = max(maximum, float(np.linalg.norm(np.asarray(obs["robot0_eef_pos"]) - reference[step + 1])))
    print(f"WARMED {directory} max_ee_error_m={maximum}", flush=True)
    return maximum


def replay_task(jobs):
    """One simulator per case/task, preserving original episode/reset order."""
    cached = [cached_replay(job) for job in jobs]
    if all(cached):
        return cached
    from libero.libero import benchmark

    from examples.Libero.eval.utils import get_libero_env

    first = Path(jobs[0][0])
    episode = json.loads((first / "episode.json").read_text())
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    task_id = episode["task_id"]
    run = first.parent.parent
    settings = json.loads((run / "manifest.json").read_text())["settings"]
    selected = {json.loads((Path(job[0]) / "episode.json").read_text())["initial_index"]: job for job in jobs}
    env, _ = get_libero_env(suite.get_task(task_id), resolution=256)
    results = []
    try:
        for initial in range(settings["initial_start"], max(selected) + 1):
            directory = first.parent / f"task{task_id:02d}_init{initial:03d}"
            if initial in selected:
                result = replay(selected[initial], shared=(suite, env))
                results.append(result)
                if result["status"] != "resumed":
                    print(json.dumps(result), flush=True)
                    continue
            # Even a cached video must advance the simulator before later trials.
            error = advance_episode(env, suite, directory)
            if error > jobs[0][2]:
                raise ValueError("Preceding trial diverges; later replays cannot be verified: " + str(directory))
    finally:
        env.close()
    return results


def replay(job, shared=None):
    # Lazy imports keep simulator/OpenGL initialization within worker processes.
    import cv2
    import imageio
    from libero.libero import benchmark

    from examples.Libero.eval.utils import get_libero_dummy_action, get_libero_env, get_libero_image
    from gr00t.rag.diagnosis_protocol import array_digest, instruction_key

    directory, output, tolerance, fps, warmup_prior = job
    directory, output = Path(directory), Path(output)
    episode, actions, reference, modes = load_episode(directory)
    output.mkdir(parents=True, exist_ok=True)
    source = {name: file_hash(directory / name) for name in ("episode.json", "trace.npz")}
    report_path = output / "validation.json"
    cached = cached_replay(job)
    if cached is not None:
        return cached

    suite = shared[0] if shared is not None else benchmark.get_benchmark_dict()["libero_10"]()
    task = suite.get_task(episode["task_id"])
    initial = suite.get_task_init_states(episode["task_id"])[episode["initial_index"]]
    if instruction_key(task.language) != instruction_key(episode["task"]):
        raise ValueError("Task order/language differs from original run")
    if array_digest(initial) != episode["initial_sha256"]:
        raise ValueError("Benchmark initial state differs from original run")
    env, writer = (shared[1] if shared is not None else None), None
    temporary = output / "rollout_replay.tmp.mp4"
    positions, first_success = [], None
    started = time.perf_counter()
    try:
        if shared is None:
            env, _ = get_libero_env(task, resolution=256)
        if warmup_prior and shared is None:
            run = directory.parent.parent
            settings = json.loads((run / "manifest.json").read_text())["settings"]
            for previous in range(settings["initial_start"], episode["initial_index"]):
                previous_dir = directory.parent / f"task{episode['task_id']:02d}_init{previous:03d}"
                previous_episode, previous_actions, _, _ = load_episode(previous_dir)
                previous_initial = suite.get_task_init_states(episode["task_id"])[previous]
                if array_digest(previous_initial) != previous_episode["initial_sha256"]:
                    raise ValueError("Warmup initial state differs")
                env.reset()
                env.set_init_state(previous_initial)
                for _ in range(10):
                    env.step(get_libero_dummy_action())
                for action in previous_actions:
                    env.step(action.tolist())
                print(f"WARMED {previous_dir}", flush=True)
        env.reset()
        obs = env.set_init_state(initial)
        for _ in range(10):
            obs, _, _, _ = env.step(get_libero_dummy_action())
        writer = imageio.get_writer(str(temporary), fps=fps, macro_block_size=1)
        query_index = 0
        queries = episode["queries"]
        for step, action in enumerate(actions):
            positions.append(np.asarray(obs["robot0_eef_pos"]).copy())
            while query_index + 1 < len(queries) and queries[query_index + 1]["control_step"] <= step:
                query_index += 1
            query = queries[query_index]
            mode = "goal" if modes[step] else "action"
            image, wrist = get_libero_image(obs)
            frame = np.pad(np.concatenate((image, wrist), axis=1), ((64, 0), (0, 0), (0, 0)))
            labels = frame_labels(query, step, mode)
            for line, text in enumerate(labels):
                cv2.putText(frame, text, (5, 13 + 15 * line), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (230, 230, 230), 1)
            writer.append_data(frame)
            obs, _, done, _ = env.step(action.tolist())
            if done and first_success is None:
                first_success = step + 1
            if (step + 1) % 250 == 0:
                print(f"REPLAY {directory.name} {step + 1}/{len(actions)}", flush=True)
        positions.append(np.asarray(obs["robot0_eef_pos"]).copy())
        goals = env.env.parsed_problem.get("goal_state", [])
        status = {json.dumps(g): bool(env.env._eval_predicate(g)) for g in goals} if goals else None
    finally:
        if writer is not None:
            writer.close()
        if env is not None and shared is None:
            env.close()
    error = np.linalg.norm(np.asarray(positions) - reference, axis=1)
    np.savez_compressed(output / "replay_trace.npz", ee=np.asarray(positions), ee_error_m=error)
    expected_success_step = len(actions) if episode["success"] else None
    validated = bool(
        error.max() <= tolerance
        and first_success == expected_success_step
        and status == episode["final_predicates"]
    )
    video = "rollout_replay.mp4" if validated else "rollout_replay.UNVERIFIED.mp4"
    temporary.replace(output / video)
    report = dict(
        validated=validated,
        source_episode=str(directory),
        source_sha256=source,
        video=video,
        control_steps=len(actions),
        fps=fps,
        warmup_prior=warmup_prior,
        ee_tolerance_m=tolerance,
        initial_ee_error_m=float(error[0]),
        first_ee_mismatch_step=int(np.flatnonzero(error > tolerance)[0]) if (error > tolerance).any() else None,
        prefix128_max_ee_error_m=float(error[:129].max()),
        max_ee_error_m=float(error.max()),
        mean_ee_error_m=float(error.mean()),
        first_success_step=first_success,
        expected_success_step=expected_success_step,
        final_predicates=status,
        expected_final_predicates=episode["final_predicates"],
        wall_seconds=time.perf_counter() - started,
        renderer=os.environ.get("MUJOCO_GL"),
        note="Saved-action reconstruction, not original frames. EE/predicate checks do not prove identical full object states.",
    )
    atomic_json(report_path, report)
    if validated:
        publish_replay(directory, output, report)
    return dict(
        episode=str(directory),
        status="validated" if validated else "UNVERIFIED",
        max_ee_error_m=report["max_ee_error_m"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--task-ids", type=int, nargs="+")
    parser.add_argument("--initial-indices", type=int, nargs="+")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--warmup-prior",
        action="store_true",
        default=True,
        help="Preserve task-local trial/reset order (default); needed for faithful replay",
    )
    parser.add_argument(
        "--independent",
        action="store_false",
        dest="warmup_prior",
        help="Debug only: reset a fresh simulator per episode; may diverge",
    )
    parser.add_argument(
        "--include-existing", action="store_true", help="Also replay episodes that already have an original video"
    )
    parser.add_argument(
        "--ee-tolerance", type=float, default=0.0001, help="Maximum EE deviation in meters (default 0.1 mm)"
    )
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--audit-only", action="store_true", help="Validate saved traces/overlays without simulator execution"
    )
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_STEP_ID"):
        parser.error("Run on a Slurm worker (srun); a policy/GPU is not required with MUJOCO_GL=osmesa")
    if (
        args.workers < 1
        or args.fps < 1
        or not np.isfinite(args.ee_tolerance)
        or args.ee_tolerance <= 0
        or (args.limit is not None and args.limit < 1)
    ):
        parser.error("Workers, FPS, limit and tolerance must be positive")
    run = args.run.resolve()
    manifest = json.loads((run / "manifest.json").read_text())
    output = (args.output or run / "video_replay").resolve()
    known_cases = {case["name"] for case in manifest["cases"]}
    if args.cases and not set(args.cases) <= known_cases:
        parser.error("Unknown case")
    paths = []
    for case in manifest["cases"]:
        if args.cases and case["name"] not in args.cases:
            continue
        for path in sorted((run / case["name"]).glob("task*_init*/episode.json")):
            episode = json.loads(path.read_text())
            if args.task_ids is not None and episode["task_id"] not in args.task_ids:
                continue
            if args.initial_indices is not None and episode["initial_index"] not in args.initial_indices:
                continue
            if not args.include_existing and (
                (path.parent / "rollout.mp4").is_file()
                or (episode.get("video") and (path.parent / episode["video"]).is_file())
            ):
                continue
            paths.append(path.parent)
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        print("No selected episodes need replay; existing videos are preserved", flush=True)
        return
    if args.audit_only:
        for directory in paths:
            episode, _, _, _ = load_episode(directory)
            for query in episode["queries"]:
                frame_labels(query, query["control_step"], query.get("mode", "action"))
            # All earlier trials are needed to reproduce reset/controller history.
            for initial in range(manifest["settings"]["initial_start"], episode["initial_index"]):
                prior = directory.parent / f"task{episode['task_id']:02d}_init{initial:03d}"
                if not (prior / "episode.json").is_file() or not (prior / "trace.npz").is_file():
                    raise ValueError("Missing replay prefix: " + str(prior))
        print(json.dumps(dict(audit="passed", run=str(run), selected_missing_videos=len(paths))), flush=True)
        return
    # Refuse destinations that could replace any source episode/video.
    for path in paths:
        destination = output / path.relative_to(run)
        if destination == path or (destination / "episode.json").exists():
            parser.error("Output would overlap original evaluation artifacts")
    jobs = [
        (str(path), str(output / path.relative_to(run)), args.ee_tolerance, args.fps, args.warmup_prior)
        for path in paths
    ]
    print(f"Replay plan: {len(jobs)} episodes, {args.workers} worker(s), output={output}", flush=True)
    failed = 0
    if args.warmup_prior:
        groups = defaultdict(list)
        for job in jobs:
            path = Path(job[0])
            groups[(path.parent, path.name.split("_")[0])].append(job)
        function, work = replay_task, list(groups.values())
    else:
        function, work = replay, jobs
    # Release native renderer/JIT allocations between task groups. Long-lived
    # OSMesa workers can retain substantial memory after env.close().
    for offset in range(0, len(work), args.workers):
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            for group in pool.map(function, work[offset : offset + args.workers]):
                for result in group if isinstance(group, list) else [group]:
                    print(json.dumps(result), flush=True)
                    failed += result["status"] == "UNVERIFIED"
    if failed:
        raise SystemExit(f"{failed} replays failed validation; inspect validation.json and UNVERIFIED videos")
    print("All selected replays passed validation", flush=True)


if __name__ == "__main__":
    main()
