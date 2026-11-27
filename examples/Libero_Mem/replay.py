"""Audited Raw expert replay for checking LIBERO-Mem environment/evaluation.

This is an expert diagnostic, not a trained-policy benchmark. No model or GPU
is required with OSMesa; simulation imports are delayed until after dry-run.
"""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from examples.Libero.eval.utils import (
    get_libero_dummy_action,
    get_libero_image,
    save_rollout_video,
)
from examples.Libero_Mem.environment import RESET_PROTOCOL, configure_runtime, initialize_episode, make_environment
from examples.Libero_Mem.selection import (
    REPO,
    SourceConfig,
    episode_plan,
    file_hash as _file_hash,
    source_identity as _source_identity,
    write_summary as _write_summary,
)
from examples.Libero_Mem.source import vector_fingerprint
from examples.Libero_Mem.tasks import TASK_BY_NUMBER


@dataclass(frozen=True)
class ReplayConfig(SourceConfig):
    tail_steps: int = 20
    max_steps: int | None = None
    render: bool = False
    save_videos: bool = False


def replay_plan(config):
    if type(config.tail_steps) is not int or config.tail_steps < 0:
        raise ValueError("Invalid tail_steps")
    if config.max_steps is not None and (type(config.max_steps) is not int or config.max_steps < 1):
        raise ValueError("max_steps must be positive")
    plan = episode_plan(config)
    plan["protocol"] = "raw_expert_replay"
    return plan


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
        "environment_reset": RESET_PROTOCOL,
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
                obs, protocol = initialize_episode(env, initial, num_steps_wait=config.num_steps_wait)
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
