"""Audited LIBERO-Mem policy rollouts through GR00T's existing HTTP service."""

import argparse
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import random

import numpy as np

from examples.Libero.eval.utils import get_libero_image, save_rollout_video
from examples.Libero_Mem.environment import RESET_PROTOCOL, configure_runtime, initialize_episode, make_environment
from examples.Libero_Mem.policy import LiberoMemPolicy
from examples.Libero_Mem.replay import load_expert
from examples.Libero_Mem.selection import (
    REPO,
    SourceConfig,
    episode_plan,
    file_hash,
    load_initial_state,
    write_summary,
)
from examples.Libero_Mem.tasks import TASK_BY_NUMBER


# mem_6_run_evaluation_env_pred.py TASK_LENGTHS, keyed by canonical task number.
TASK_ACTION_STEPS = dict(enumerate((180, 285, 320, 365, 445, 650, 475, 670, 365, 390), 1))


@dataclass(frozen=True)
class EvalConfig(SourceConfig):
    task_ids: tuple[int, ...] | None = None
    training_fold: str | None = None
    task_group: str = "all"
    server_url: str = "http://127.0.0.1:5555"
    timeout: float = 60.0
    n_action_steps: int = 1
    budget_mode: str = "benchmark"
    max_steps: int | None = None
    save_videos: bool = False
    policy_label: str = "unspecified"


def evaluation_plan(config):
    if config.training_fold not in (None, "A", "B") or config.task_group not in ("all", "train", "held-out"):
        raise ValueError("Invalid training fold or task group")
    if config.task_group != "all" and config.training_fold is None:
        raise ValueError("train/held-out task selection requires training_fold")
    if config.budget_mode not in ("benchmark", "fixed"):
        raise ValueError("budget_mode must be benchmark or fixed")
    for name in ("n_action_steps", "max_steps"):
        value = getattr(config, name)
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(f"{name} must be a positive integer")
    if config.n_action_steps is None or not math.isfinite(config.timeout) or config.timeout <= 0:
        raise ValueError("n_action_steps and timeout must be positive")
    allowed = {
        number
        for number, task in TASK_BY_NUMBER.items()
        if config.task_group == "all" or (task.fold == config.training_fold) == (config.task_group == "train")
    }
    selected = tuple(sorted(allowed)) if config.task_ids is None else config.task_ids
    if set(selected) - allowed:
        raise ValueError("Explicit task_ids fall outside the selected task group")
    plan = episode_plan(replace(config, task_ids=selected))
    plan["protocol"] = "libero_mem_reactive_policy_v1"
    return plan


def _stats(episodes):
    count = len(episodes)
    successes = sum(int(episode["success"]) for episode in episodes)
    return {
        "episodes": count,
        "successes": successes,
        "success_rate": successes / count if count else 0.0,
        "mean_subgoal_completion_rate": (
            sum(episode["subgoal_completion_rate"] for episode in episodes) / count if count else 0.0
        ),
    }


def _aggregate(summary):
    episodes = summary["episodes"]
    totals = _stats(episodes)
    summary.update(
        total_episodes=totals["episodes"],
        total_successes=totals["successes"],
        success_rate=totals["success_rate"],
        mean_subgoal_completion_rate=totals["mean_subgoal_completion_rate"],
    )
    summary["tasks"] = {
        str(number): _stats([episode for episode in episodes if episode["task_number"] == number])
        for number in sorted({episode["task_number"] for episode in episodes})
    }
    summary["groups"] = {}
    for field in ("category", "memory_types", "task_split"):
        values = {
            value
            for episode in episodes
            for value in (episode[field] if field == "memory_types" else [episode[field]])
            if value is not None
        }
        summary["groups"][field] = {
            value: _stats(
                [
                    episode
                    for episode in episodes
                    if (value in episode[field] if field == "memory_types" else episode[field] == value)
                ]
            )
            for value in sorted(values)
        }


def run_evaluation(config, *, env_factory=None, policy_factory=None):
    output = Path(config.output_dir).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if output.resolve().is_relative_to(Path(config.raw_root).resolve()):
        raise ValueError("Evaluation output must be outside the Raw source")
    plan = evaluation_plan(config)
    if config.dry_run:
        return {
            "status": "dry_run",
            "protocol": plan["protocol"],
            "selection": config.selection,
            "training_fold": config.training_fold,
            "task_group": config.task_group,
            "episodes": len(plan["jobs"]),
            "jobs": [
                {"task_number": job["task_number"], "demo_id": job["demo"]["demo_id"]} for job in plan["jobs"]
            ],
        }
    output.mkdir(parents=True)
    summary = {
        "status": "running",
        "protocol": plan["protocol"],
        "environment_reset": RESET_PROTOCOL,
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "audit_sha256": plan["audit_sha256"],
        "episodes": [],
        "active_episode": None,
    }
    _aggregate(summary)
    write_summary(output / "summary.json", summary)
    policy = None
    try:
        random.seed(config.seed)
        np.random.seed(config.seed)
        if env_factory is None:
            summary["runtime"] = configure_runtime(config, output)
            env_factory = make_environment
        else:
            summary["runtime"] = {"environment_factory": f"{env_factory.__module__}.{env_factory.__qualname__}"}
        factory = policy_factory or LiberoMemPolicy
        policy = factory(config.server_url, timeout=config.timeout, n_action_steps=config.n_action_steps)
        for job in plan["jobs"]:
            if file_hash(job["bddl_file"]) != job["bddl_sha256"]:
                raise ValueError("BDDL changed after evaluation planning")
            if config.budget_mode == "benchmark":
                # Match mem_6's max(TASK_LENGTHS, trimmed expert length + 20).
                # Only the horizon is used; expert commands never reach the policy.
                initial, program, _ = load_expert(job, 20)
                full_budget = max(TASK_ACTION_STEPS[job["task_number"]], len(program))
                del program
            else:
                initial = load_initial_state(job)
                full_budget = TASK_ACTION_STEPS[job["task_number"]]
            budget = min(full_budget, config.max_steps) if config.max_steps is not None else full_budget
            task = TASK_BY_NUMBER[job["task_number"]]
            identity = {"task_number": task.number, "raw_demo_id": job["demo"]["demo_id"]}
            summary["active_episode"] = identity
            write_summary(output / "summary.json", summary)
            env = env_factory(task, horizon=budget + config.num_steps_wait + 1, render=True, seed=config.seed)
            try:
                obs, protocol = initialize_episode(env, initial, num_steps_wait=config.num_steps_wait)
                policy.reset()
                top, wrist = [], []
                with (output / f"T{task.number}_{job['demo']['demo_id']}.jsonl").open("x") as log:
                    for _ in range(budget):
                        if config.save_videos:
                            image, hand = get_libero_image(obs)
                            top.append(image.copy())
                            wrist.append(hand.copy())
                        action = np.asarray(policy.get_action(obs, task.instruction), dtype=np.float32)
                        if action.shape != (7,) or not np.isfinite(action).all():
                            raise ValueError("Policy action must be a finite 7D vector")
                        obs, score = protocol.step(action.tolist())
                        log.write(json.dumps({**score, "action": action.tolist()}, allow_nan=False) + "\n")
                result = {
                    **identity,
                    "task": task.name,
                    "instruction": task.instruction,
                    "category": task.category,
                    "memory_types": task.memory_types,
                    "task_split": ("train" if task.fold == config.training_fold else "held-out")
                    if config.training_fold
                    else None,
                    "bddl_sha256": job["bddl_sha256"],
                    "initial_state_sha256": job["demo"]["initial_state_sha256"],
                    "full_action_budget": full_budget,
                    "truncated": budget < full_budget,
                    "goal_sequences": protocol.sequences,
                    "policy_requests": policy.requests,
                    "prediction_horizon": policy.prediction_horizon,
                    **protocol.snapshot(),
                }
                if config.save_videos:
                    result["video"] = save_rollout_video(
                        top,
                        wrist,
                        len(summary["episodes"]),
                        result["success"],
                        task.instruction,
                        output_dir=output / "videos",
                        fps=20,
                    )
            finally:
                env.close()
            summary["episodes"].append(result)
            summary["active_episode"] = None
            _aggregate(summary)
            write_summary(output / "summary.json", summary)
            print(
                f"T{task.number}/{job['demo']['demo_id']}: success={result['success']} "
                f"subgoals={result['subgoals_completed']}/{result['subgoals_total']} overshot={result['overshot']}",
                flush=True,
            )
        summary["status"] = "completed"
        write_summary(output / "summary.json", summary)
    except BaseException as exc:
        summary["status"] = "failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        write_summary(output / "summary.json", summary)
        raise
    finally:
        if policy is not None:
            policy.close()
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--libero-repo", type=Path, default=REPO.parent / "libero-mem")
    parser.add_argument("--training-fold", choices=("A", "B"))
    parser.add_argument("--task-group", choices=("all", "train", "held-out"), default="all")
    parser.add_argument("--task-ids", type=int, nargs="+", choices=range(1, 11))
    parser.add_argument("--demo-ids", type=int, nargs="+")
    parser.add_argument("--selection", choices=("seen", "unseen"), default="unseen")
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--num-steps-wait", type=int, default=20)
    parser.add_argument("--server-url", default="http://127.0.0.1:5555")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--n-action-steps", type=int, default=1)
    parser.add_argument("--budget-mode", choices=("benchmark", "fixed"), default="benchmark")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--policy-label", default="unspecified")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--renderer", choices=("osmesa", "egl"), default="osmesa")
    parser.add_argument("--dry-run", action="store_true")
    args = vars(parser.parse_args(argv))
    for key in ("task_ids", "demo_ids"):
        if args[key] is not None:
            args[key] = tuple(args[key])
    result = run_evaluation(EvalConfig(**args))
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
