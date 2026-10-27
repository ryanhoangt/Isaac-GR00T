"""Evaluate tasks concurrently against one serial RA-VLA policy server."""

import argparse
from collections import deque
from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import time
import uuid

from scripts.rag.eval_libero import SPLITS, parse_suite


@dataclass
class Plan:
    repo: Path
    output: Path
    run_name: str
    suite: str
    task_ids: list[int]
    trials: int
    workers: int
    port: int
    server_id: str
    server_command: list[str]
    task_commands: dict[int, list[str]]
    env: dict[str, str]
    startup_timeout: float = 300
    poll_interval: float = 0.5


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def stop_processes(processes):
    """Signal only child process groups created by this launcher, then reap them."""
    processes = list(processes)
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 5
    for process in processes:
        try:
            process.wait(timeout=max(0.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def task_directory(plan, task):
    return plan.output / "tasks" / f"task{task:02d}"


def task_environment(plan, task):
    env = plan.env.copy()
    directory = task_directory(plan, task)
    env["NUMBA_CACHE_DIR"] = str(directory / "numba_cache")
    env["HF_MODULES_CACHE"] = str(directory / "hf_modules")
    env["MPLCONFIGDIR"] = str(directory / "mpl_cache")
    return env


def read_task_summary(plan, task):
    directory = task_directory(plan, task) / "rollouts"
    summary = json.loads((directory / "summary.json").read_text())
    episodes = summary["episodes"]
    identities = {(e["task_id"], e["episode_index"]) for e in episodes}
    expected = {(task, i) for i in range(plan.trials)}
    if (
        summary["task_suite_name"] != plan.suite
        or summary["task_ids"] != [task]
        or len(episodes) != plan.trials
        or identities != expected
        or summary["num_episodes"] != plan.trials
        or summary["num_trials_per_task"] != plan.trials
        or summary["num_successes"] != sum(e["success"] for e in episodes)
    ):
        raise ValueError(f"Incomplete or inconsistent task summary: {directory}")
    result = []
    for episode in sorted(episodes, key=lambda e: e["episode_index"]):
        if not isinstance(episode["success"], bool):
            raise ValueError(f"Invalid outcome in {directory}")
        label = (
            episode["task_description"]
            .lower()
            .replace(" ", "_")
            .replace("\n", "_")
            .replace(".", "_")[:128]
        )
        source = (
            directory
            / f"episode={episode['episode_index'] + 1}--success={episode['success']}--task={label}.mp4"
        )
        if not source.is_file():
            raise FileNotFoundError(source)
        ordinal = plan.task_ids.index(task) * plan.trials + episode["episode_index"] + 1
        video = (
            plan.output
            / "rollouts"
            / f"episode={ordinal}--success={episode['success']}--task={label}.mp4"
        )
        if video.is_symlink():
            if video.resolve() != source.resolve():
                raise ValueError(f"Conflicting video link: {video}")
        elif video.exists():
            raise FileExistsError(video)
        else:
            video.symlink_to(os.path.relpath(source, video.parent))
        result.append({**episode, "video_path": str(video.relative_to(plan.output))})
    return result


def running_counts(plan, task):
    path = task_directory(plan, task) / "rollouts" / "results.log"
    if not path.exists():
        return 0, 0
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - 20000))
        tail = handle.read().decode(errors="replace")
    counts = re.findall(
        r"# episodes completed so far:\s*(\d+)\s*# successes:\s*(\d+)", tail
    )
    return tuple(map(int, counts[-1])) if counts else (0, 0)


def publish(plan, state, completed):
    episodes = [e for task in plan.task_ids for e in completed.get(task, [])]
    successes = sum(e["success"] for e in episodes)
    atomic_json(
        plan.output / "rollouts" / "summary.json",
        {
            "task_suite_name": plan.suite,
            "run_name": plan.run_name,
            "task_ids": plan.task_ids,
            "completed_task_ids": sorted(completed),
            "num_trials_per_task": plan.trials,
            "num_episodes": len(episodes),
            "num_successes": successes,
            "success_rate": successes / len(episodes) if episodes else None,
            "status": state["status"],
            "episodes": episodes,
        },
    )
    state["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    state["episodes_completed"] = sum(t["episodes"] for t in state["tasks"].values())
    state["successes"] = sum(t["successes"] for t in state["tasks"].values())
    atomic_json(plan.output / "progress.json", state)


def run_plan(plan, client_factory=None):
    """Only this parent writes aggregate files; each task owns its directory."""
    lock_dir = plan.repo / "outputs/ravla_eval/.ports"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock = (lock_dir / f"{socket.gethostname()}_{plan.port}.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", plan.port))
        _run_locked(plan, client_factory)
    finally:
        lock.close()


def _run_locked(plan, client_factory):
    plan.output.mkdir(parents=True, exist_ok=False)
    (plan.output / "rollouts").mkdir()
    atomic_json(
        plan.output / "commands.json",
        {
            "suite": plan.suite,
            "run_name": plan.run_name,
            "cwd": str(plan.repo),
            "server": plan.server_command,
            "server_id": plan.server_id,
            "workers": plan.workers,
            "task_ids": plan.task_ids,
            "tasks": {str(k): v for k, v in plan.task_commands.items()},
            "environment": {
                k: plan.env[k]
                for k in ["MUJOCO_GL", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"]
            },
        },
    )
    state = {
        "status": "starting",
        "pid": os.getpid(),
        "server_id": plan.server_id,
        "tasks": {
            str(t): {"status": "pending", "episodes": 0, "successes": 0}
            for t in plan.task_ids
        },
    }
    completed, active = {}, {}
    server = None
    logs = []
    try:
        server_log = (plan.output / "server.log").open("w")
        logs.append(server_log)
        env = plan.env.copy()
        env["HF_MODULES_CACHE"] = str(plan.output / "server_hf_modules")
        server = subprocess.Popen(
            plan.server_command,
            cwd=plan.repo,
            env=env,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        state["server_pid"] = server.pid
        publish(plan, state, completed)
        if client_factory is None:
            from gr00t.eval.service import BaseInferenceClient

            client_factory = BaseInferenceClient
        client = client_factory(host="127.0.0.1", port=plan.port, timeout_ms=1000)
        try:
            deadline = time.monotonic() + plan.startup_timeout
            while True:
                if server.poll() is not None:
                    raise RuntimeError(
                        f"Policy server exited: {server.returncode}; see server.log"
                    )
                try:
                    info = client.call_endpoint("server_info", requires_input=False)
                except RuntimeError:
                    raise  # An occupied port serving a different API must not receive rollouts.
                except Exception:
                    client._init_socket()
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Policy server startup timed out")
                    time.sleep(plan.poll_interval)
                    continue
                if info.get("server_id") != plan.server_id or server.poll() is not None:
                    raise RuntimeError("Policy server identity mismatch")
                break
        finally:
            client.socket.close(linger=0)
            client.context.term()
        state["status"] = "running"
        pending = deque(plan.task_ids)
        last_publish = 0
        while pending or active:
            if server.poll() is not None:
                raise RuntimeError(
                    f"Policy server died: {server.returncode}; see server.log"
                )
            while pending and len(active) < plan.workers:
                task = pending.popleft()
                directory = task_directory(plan, task)
                directory.mkdir(parents=True, exist_ok=False)
                log = (directory / "eval.log").open("w")
                logs.append(log)
                process = subprocess.Popen(
                    plan.task_commands[task],
                    cwd=plan.repo,
                    env=task_environment(plan, task),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                active[task] = process
                state["tasks"][str(task)].update(status="running", pid=process.pid)
                print(f"Started task {task:02d}, pid={process.pid}", flush=True)
            for task, process in list(active.items()):
                count, successes = running_counts(plan, task)
                state["tasks"][str(task)].update(episodes=count, successes=successes)
                code = process.poll()
                if code is None:
                    continue
                if code != 0:
                    state["tasks"][str(task)].update(status="failed", exit_code=code)
                    raise RuntimeError(
                        f"Task {task:02d} exited {code}; see tasks/task{task:02d}/eval.log"
                    )
                completed[task] = read_task_summary(plan, task)
                state["tasks"][str(task)].update(
                    status="completed",
                    exit_code=0,
                    episodes=plan.trials,
                    successes=sum(e["success"] for e in completed[task]),
                )
                del active[task]
                print(
                    f"Completed task {task:02d}: {state['tasks'][str(task)]['successes']}/{plan.trials}",
                    flush=True,
                )
                publish(plan, state, completed)
            if time.monotonic() - last_publish >= 5:
                publish(plan, state, completed)
                last_publish = time.monotonic()
            time.sleep(plan.poll_interval)
        state["status"] = "completed"
        publish(plan, state, completed)
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        for task in active:
            if state["tasks"][str(task)]["status"] == "running":
                state["tasks"][str(task)]["status"] = "cancelled"
        publish(plan, state, completed)
        raise
    finally:
        stop_processes([*active.values(), *([server] if server else [])])
        for log in logs:
            log.close()


def build_plan(args):
    repo = Path(__file__).resolve().parents[2]
    task_ids = args.task_ids if args.task_ids is not None else list(range(10))
    if (
        not task_ids
        or len(set(task_ids)) != len(task_ids)
        or any(t not in range(10) for t in task_ids)
    ):
        raise ValueError("task IDs must be distinct values in 0..9")
    if args.workers < 1 or args.trials < 1 or not 1 <= args.port <= 65535:
        raise ValueError("workers/trials must be positive; port must be in 1..65535")
    if (
        args.cuda < 0
        or not 1 <= args.k <= 8
        or args.denoising_steps < 1
        or not 0 <= args.seed < 2**32
    ):
        raise ValueError("invalid CUDA index, K, denoising steps or seed")
    if args.startup_timeout <= 0 or args.inference_timeout_ms < 1:
        raise ValueError("timeouts must be positive")
    policy_python = shutil.which(args.policy_python)
    libero_python = shutil.which(args.libero_python)
    if not policy_python or not libero_python:
        raise ValueError("policy/libero Python executable not found")
    assets = (repo / args.assets_dir).resolve()
    split, suffix = SPLITS[args.subset]
    checkpoint = (
        (repo / args.checkpoint_path).resolve()
        if args.checkpoint_path
        else assets / f"ravla/ravla/libero_{split}/checkpoint_3"
    )
    retriever = (
        (repo / args.retriever_path).resolve()
        if args.retriever_path
        else assets / f"libero_{split}"
    )
    memory = (
        (repo / args.memory_path).resolve()
        if args.memory_path
        else retriever
        / f"{(args.memory_name or f'memory_{suffix}').removesuffix('.pt')}.pt"
    )
    for path in [
        checkpoint / "config.json",
        checkpoint / "metadata.pt",
        retriever / "config.json",
        memory,
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)
    name = (
        args.run_name
        or f"ravla_parallel_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{uuid.uuid4().hex[:6]}"
    )
    if Path(name).name != name or name in ("", ".", ".."):
        raise ValueError("run-name must be a single directory name")
    output = (
        (repo / args.output_dir).resolve()
        if args.output_dir
        else repo / "outputs/ravla_eval" / name
    )
    if output.exists():
        raise FileExistsError(output)
    identity = uuid.uuid4().hex
    server = [
        policy_python,
        "-u",
        "-m",
        "scripts.rag.4_serve_ravla",
        "--checkpoint-path",
        str(checkpoint),
        "--retriever-path",
        str(retriever),
        "--memory-path",
        str(memory),
        "--k",
        str(args.k),
        "--denoising-steps",
        str(args.denoising_steps),
        "--seed",
        str(args.seed),
        "--cuda",
        str(args.cuda),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--server-id",
        identity,
    ]
    tasks = {
        task: [
            libero_python,
            "-u",
            "-m",
            "examples.Libero.eval.run_libero_eval",
            "--task-suite-name",
            args.subset,
            "--task-ids",
            str(task),
            "--num-trials-per-task",
            str(args.trials),
            "--run-name",
            f"{name}_task{task:02d}",
            "--output-dir",
            str(output / "tasks" / f"task{task:02d}" / "rollouts"),
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--headless",
            "--inference-timeout-ms",
            str(args.inference_timeout_ms),
        ]
        for task in task_ids
    }
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(repo) + os.pathsep + env.get("PYTHONPATH", ""),
        PYTHONUNBUFFERED="1",
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        NO_ALBUMENTATIONS_UPDATE="1",
        MUJOCO_GL="egl",
        PYOPENGL_PLATFORM="egl",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        USE_TF="0",
        PYTHONDONTWRITEBYTECODE="1",
    )
    return Plan(
        repo,
        output,
        name,
        args.subset,
        task_ids,
        args.trials,
        min(args.workers, len(task_ids)),
        args.port,
        identity,
        server,
        tasks,
        env,
        args.startup_timeout,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subset", type=parse_suite, choices=SPLITS, default="libero_10"
    )
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--retriever-path", type=Path)
    memory = parser.add_mutually_exclusive_group()
    memory.add_argument("--memory-path", type=Path)
    memory.add_argument("--memory-name")
    parser.add_argument(
        "--assets-dir", type=Path, default=Path("ra-vla-official-assets")
    )
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--task-ids", type=int, nargs="+")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--port", type=int, default=8085)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--denoising-steps", type=int, default=4)
    parser.add_argument("--startup-timeout", type=float, default=300)
    parser.add_argument("--inference-timeout-ms", type=int, default=120000)
    parser.add_argument("--run-name")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--policy-python",
        default=os.environ.get(
            "POLICY_PYTHON", str(Path.home() / "miniconda3/envs/gr00t/bin/python")
        ),
    )
    parser.add_argument(
        "--libero-python",
        default=os.environ.get(
            "LIBERO_PYTHON", str(Path.home() / "miniconda3/envs/libero/bin/python")
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    plan = build_plan(args)
    print(
        f"Shared server: {plan.port}; workers: {plan.workers}; tasks: {plan.task_ids}; trials/task: {plan.trials}",
        flush=True,
    )
    print(f"Output: {plan.output}", flush=True)
    if args.dry_run:
        print(
            json.dumps(
                {"server": plan.server_command, "tasks": plan.task_commands}, indent=2
            )
        )
        return

    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    run_plan(plan)


if __name__ == "__main__":
    main()
