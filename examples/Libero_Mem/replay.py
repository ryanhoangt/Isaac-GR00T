"""Audited Raw expert replay for checking LIBERO-Mem environment/evaluation.

This is an expert diagnostic, not a trained-policy benchmark. No model or GPU
is required with OSMesa; simulation imports are delayed until after dry-run.
"""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import h5py
import numpy as np

from examples.Libero.eval.utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    save_rollout_video,
)
from examples.Libero_Mem.episode import LiberoMemEpisodeProtocol
from examples.Libero_Mem.source import AUDIT_VERSION, vector_fingerprint
from examples.Libero_Mem.tasks import TASK_BY_NUMBER


REPO = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ReplayConfig:
    raw_root: Path
    audit: Path
    output_dir: Path
    libero_repo: Path = REPO.parent / "libero-mem"
    task_ids: tuple[int, ...] = tuple(range(1, 11))
    demo_ids: tuple[int, ...] | None = None
    selection: str = "unseen"
    num_episodes: int = 1
    num_steps_wait: int = 20
    tail_steps: int = 20
    seed: int = 0
    max_steps: int | None = None
    renderer: str = "osmesa"
    render: bool = False
    save_videos: bool = False
    dry_run: bool = False


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_identity(path, expected):
    stat = path.stat()
    if stat.st_size != expected["size_bytes"] or stat.st_mtime_ns != expected["mtime_ns"]:
        raise ValueError(f"Raw source differs from audit: {path}")


def replay_plan(config):
    if config.selection not in ("seen", "unseen") or config.renderer not in ("osmesa", "egl"):
        raise ValueError("Unsupported initial-state selection or renderer")
    for name in ("num_episodes", "num_steps_wait", "tail_steps", "seed"):
        value = getattr(config, name)
        if type(value) is not int or value < (1 if name == "num_episodes" else 0):
            raise ValueError(f"Invalid {name}")
    if config.max_steps is not None and (type(config.max_steps) is not int or config.max_steps < 1):
        raise ValueError("max_steps must be positive")
    if (
        not config.task_ids
        or len(set(config.task_ids)) != len(config.task_ids)
        or any(type(i) is not int or i not in TASK_BY_NUMBER for i in config.task_ids)
    ):
        raise ValueError("task_ids must be distinct integers in 1..10")
    if config.demo_ids is not None and (
        not config.demo_ids
        or len(set(config.demo_ids)) != len(config.demo_ids)
        or any(type(i) is not int or i < 0 for i in config.demo_ids)
    ):
        raise ValueError("demo_ids must be distinct nonnegative integers")
    report = json.loads(Path(config.audit).read_text())
    if report.get("audit_version") != AUDIT_VERSION or report.get("status") != "complete":
        raise ValueError("Expected a completed unit-1 source audit")
    records = {entry["number"]: entry for entry in report["tasks"]}
    if len(records) != len(report["tasks"]) or set(config.task_ids) - records.keys():
        raise ValueError("Audit does not uniquely cover the selected tasks")
    raw_root = Path(config.raw_root).expanduser().resolve(strict=True)
    libero_repo = Path(config.libero_repo).expanduser().resolve(strict=True)
    jobs = []
    for number in sorted(config.task_ids):
        task, record = TASK_BY_NUMBER[number], records[number]
        if record["name"] != task.name or record["instruction"] != task.instruction:
            raise ValueError("Audited task identity differs from registry")
        raw = sorted(record["raw_demos"], key=lambda item: int(item["demo_id"].removeprefix("demo_")))
        if not raw or len({entry["demo_id"] for entry in raw}) != len(raw):
            raise ValueError(f"Empty or duplicate Raw demo inventory: {task.name}")
        full_states = {entry["initial_state_sha256"] for entry in record["full_demos"]}
        if config.selection == "unseen":
            if len(raw) < 20:
                raise ValueError("Unseen initial-state protocol requires the final twenty Raw demos")
            pool = raw[-20:]
            if any(entry["initial_state_sha256"] in full_states for entry in pool):
                raise ValueError(f"Unseen Raw tail overlaps Full initial states: {task.name}")
        else:
            pool = [entry for entry in raw if entry["initial_state_sha256"] in full_states]
        if config.demo_ids is None:
            if config.num_episodes > len(pool):
                raise ValueError(f"Not enough {config.selection} initial states: {task.name}")
            # Match the author's last-N convention for unseen, first-N for seen.
            selected = (
                pool[-config.num_episodes :] if config.selection == "unseen" else pool[: config.num_episodes]
            )
        else:
            names = {f"demo_{i}" for i in config.demo_ids}
            if names - {entry["demo_id"] for entry in pool}:
                raise ValueError(
                    f"Requested demo is outside the {config.selection} initial-state pool: {task.name}"
                )
            selected = [entry for entry in pool if entry["demo_id"] in names]
        path = raw_root / task.filename
        _source_identity(path, record["raw_file"])
        bddl = libero_repo / "libero/libero/bddl_files/libero_mem" / f"{task.name}.bddl"
        bddl_hash = _file_hash(bddl)
        for entry in selected:
            jobs.append(
                {
                    "task_number": number,
                    "task": task.name,
                    "instruction": task.instruction,
                    "raw_file": str(path),
                    "raw_identity": record["raw_file"],
                    "demo": entry,
                    "bddl_file": str(bddl),
                    "bddl_sha256": bddl_hash,
                }
            )
    return {
        "protocol": "raw_expert_replay",
        "selection": config.selection,
        "audit_sha256": _file_hash(config.audit),
        "jobs": jobs,
    }


def load_expert(job, tail_steps):
    path = Path(job["raw_file"])
    _source_identity(path, job["raw_identity"])
    with h5py.File(path, "r") as handle:
        group = handle[f"data/{job['demo']['demo_id']}"]
        actions, initial = group["actions"][:], group["init_state"][:]
        model_xml = group.attrs["model_file"]
    if (
        actions.shape != (job["demo"]["frames"], 7)
        or initial.ndim != 1
        or vector_fingerprint(actions) != job["demo"]["actions_sha256"]
        or vector_fingerprint(initial) != job["demo"]["initial_state_sha256"]
    ):
        raise ValueError("Raw expert vectors differ from audited identity")
    xml_bytes = model_xml.encode() if isinstance(model_xml, str) else model_xml
    if hashlib.sha256(xml_bytes).hexdigest() != job["demo"]["model_xml_sha256"]:
        raise ValueError("Raw model XML differs from audit")
    _source_identity(path, job["raw_identity"])
    moving = np.flatnonzero(np.linalg.norm(actions[:, :6], axis=1) >= 1e-4)
    if not len(moving):
        raise ValueError("Raw expert trajectory contains no arm motion")
    start = int(moving[0])
    program = actions[start:].astype(np.float64, copy=True)
    program[program[:, -1] == 0, -1] = -1
    tail = np.tile(np.asarray(get_libero_dummy_action(), np.float64), (tail_steps, 1))
    return initial, np.concatenate([program, tail]), start


def configure_runtime(config, output):
    """Use the fork and its customized robosuite in this process only."""
    repo = Path(config.libero_repo).expanduser().resolve(strict=True)
    for name, relative in (
        ("libero", "libero"),
        ("robosuite", "thirdparty/robosuite"),
        ("robomimic", "thirdparty/robomimic"),
    ):
        loaded = sys.modules.get(name)
        if loaded is not None and (
            not getattr(loaded, "__file__", None)
            or not Path(loaded.__file__).resolve().is_relative_to(repo / relative)
        ):
            raise ValueError(f"{name} already imported from another checkout; start a fresh replay process")
    sys.path[:0] = [str(repo), str(repo / "thirdparty/robosuite"), str(repo / "thirdparty/robomimic")]
    os.environ["MUJOCO_GL"] = config.renderer
    os.environ["PYOPENGL_PLATFORM"] = config.renderer
    os.environ.setdefault("LP_NUM_THREADS", "2")
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
    directory = output / ".libero"
    directory.mkdir()
    base = repo / "libero/libero"
    paths = {
        "benchmark_root": str(base),
        "assets": str(base / "assets"),
        "bddl_files": str(base / "bddl_files"),
        "init_states": str(repo / "scripts/__init_data"),
        "datasets": str(Path(config.raw_root).resolve()),
    }
    (directory / "config.yaml").write_text(json.dumps(paths, indent=2) + "\n")
    os.environ["LIBERO_CONFIG_PATH"] = str(directory)
    modules = {name: importlib.import_module(name) for name in ("mujoco", "robosuite", "libero.libero.envs")}
    expected = {"robosuite": repo / "thirdparty/robosuite", "libero.libero.envs": repo}
    for name, root in expected.items():
        if not Path(modules[name].__file__).resolve().is_relative_to(root):
            raise ValueError(f"Imported {name} from unexpected source")
    return {
        name: {"version": getattr(module, "__version__", None), "path": module.__file__}
        for name, module in modules.items()
    }


def make_environment(task, *, horizon, render, seed):
    descriptor = SimpleNamespace(
        language=task.instruction, problem_folder="libero_mem", bddl_file=f"{task.name}.bddl"
    )
    env, _ = get_libero_env(
        descriptor, horizon=horizon, use_camera_obs=render, camera_depths=False, camera_segmentations=None
    )
    env.seed(seed)
    actual = env.language_instruction.strip()
    if actual != task.instruction:
        env.close()
        raise ValueError("Simulator BDDL instruction differs from registry")
    return env


def _write_summary(path, summary):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run_replay(config, *, env_factory=None):
    output = Path(config.output_dir).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if output.resolve().is_relative_to(Path(config.raw_root).resolve()):
        raise ValueError("Replay output must be outside the Raw source")
    plan = replay_plan(config)
    if config.dry_run:
        return {
            "status": "dry_run",
            "protocol": plan["protocol"],
            "selection": config.selection,
            "episodes": len(plan["jobs"]),
            "jobs": [{"task": job["task"], "demo_id": job["demo"]["demo_id"]} for job in plan["jobs"]],
        }
    output.mkdir(parents=True)
    summary = {
        "status": "running",
        "protocol": plan["protocol"],
        "selection": config.selection,
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "audit_sha256": plan["audit_sha256"],
        "episodes": [],
        "total_episodes": 0,
        "expert_successes": 0,
        "expert_replay_success_rate": 0.0,
    }
    _write_summary(output / "summary.json", summary)
    try:
        if env_factory is None:
            summary["runtime"] = configure_runtime(config, output)
            env_factory = make_environment
        else:
            summary["runtime"] = {"environment_factory": f"{env_factory.__module__}.{env_factory.__qualname__}"}
        for job in plan["jobs"]:
            if _file_hash(job["bddl_file"]) != job["bddl_sha256"]:
                raise ValueError("BDDL changed after replay planning")
            initial, program, start = load_expert(job, config.tail_steps)
            full_budget = len(program)
            budget = min(full_budget, config.max_steps) if config.max_steps is not None else full_budget
            env = env_factory(
                TASK_BY_NUMBER[job["task_number"]],
                horizon=budget + config.num_steps_wait + 1,
                render=config.render or config.save_videos,
                seed=config.seed,
            )
            try:
                protocol = LiberoMemEpisodeProtocol(env)
                protocol.reset()
                env.reset()
                if len(env.sim.get_state().flatten()) != len(initial):
                    raise ValueError("Raw initial-state dimension differs from simulator model")
                obs = env.set_init_state(initial)
                for _ in range(config.num_steps_wait):
                    obs, _, _, _ = env.step(get_libero_dummy_action())
                protocol.reset()
                top, wrist = [], []
                with (output / f"T{job['task_number']}_{job['demo']['demo_id']}.jsonl").open("x") as log:
                    for action in program[:budget]:
                        if config.save_videos:
                            a, b = get_libero_image(obs)
                            top.append(a.copy())
                            wrist.append(b.copy())
                        obs, score = protocol.step(action.tolist())
                        log.write(json.dumps(score, allow_nan=False) + "\n")
                result = {
                    "task": job["task"],
                    "task_number": job["task_number"],
                    "raw_demo_id": job["demo"]["demo_id"],
                    "bddl_sha256": job["bddl_sha256"],
                    "initial_state_sha256": job["demo"]["initial_state_sha256"],
                    "trimmed_initial_noops": start,
                    "tail_steps": config.tail_steps,
                    "full_replay_budget": full_budget,
                    "truncated": budget < full_budget,
                    "goal_sequences": protocol.sequences,
                    **protocol.snapshot(),
                }
                if config.save_videos:
                    result["video"] = save_rollout_video(
                        top,
                        wrist,
                        len(summary["episodes"]),
                        result["success"],
                        job["instruction"],
                        output_dir=output / "videos",
                        fps=20,
                    )
            finally:
                env.close()
            summary["episodes"].append(result)
            summary["total_episodes"] += 1
            summary["expert_successes"] += int(result["success"])
            summary["expert_replay_success_rate"] = summary["expert_successes"] / summary["total_episodes"]
            _write_summary(output / "summary.json", summary)
            print(
                f"T{job['task_number']}/{job['demo']['demo_id']}: success={result['success']} "
                f"subgoals={result['subgoals_completed']}/{result['subgoals_total']} overshot={result['overshot']}",
                flush=True,
            )
        summary["status"] = "completed"
        _write_summary(output / "summary.json", summary)
    except BaseException as exc:
        summary["status"] = "failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        _write_summary(output / "summary.json", summary)
        raise
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--libero-repo", type=Path, default=REPO.parent / "libero-mem")
    parser.add_argument("--task-ids", type=int, nargs="+", choices=range(1, 11), default=list(range(1, 11)))
    parser.add_argument("--demo-ids", type=int, nargs="+")
    parser.add_argument("--selection", choices=("seen", "unseen"), default="unseen")
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--num-steps-wait", type=int, default=20)
    parser.add_argument("--tail-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--renderer", choices=("osmesa", "egl"), default="osmesa")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = vars(parser.parse_args(argv))
    args["task_ids"] = tuple(args["task_ids"])
    if args["demo_ids"] is not None:
        args["demo_ids"] = tuple(args["demo_ids"])
    result = run_replay(ReplayConfig(**args))
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
