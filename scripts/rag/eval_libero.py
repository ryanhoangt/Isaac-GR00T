"""Run a RA-VLA server and LIBERO evaluation in separate Python environments."""

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time

SPLITS = {
    "libero_spatial": ("lgo", "s"),
    "libero_object": ("lgs", "o"),
    "libero_goal": ("los", "g"),
    "libero_10": ("gos", "l"),
}


def parse_suite(value):
    aliases = {
        "spatial": "libero_spatial",
        "object": "libero_object",
        "goal": "libero_goal",
        "long": "libero_10",
        "libero_long": "libero_10",
    }
    return aliases.get(value.lower(), value.lower())


def exit_on_signal(signum, _frame):
    # Let subprocess.run and the server's finally block clean up on cancellation.
    raise SystemExit(128 + signum)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--libero-python", required=True)
    parser.add_argument("--policy-python", default=sys.executable)
    parser.add_argument("--suite", "--subset", type=parse_suite, choices=SPLITS, default="libero_goal")
    parser.add_argument("--checkpoint-path", "--ckpt-path", type=Path)
    parser.add_argument("--retriever-path", type=Path)
    memory_options = parser.add_mutually_exclusive_group()
    memory_options.add_argument("--memory-name", help="Name in the retriever directory, with or without .pt")
    memory_options.add_argument("--memory-path", type=Path, help="Use a memory file at an arbitrary path")
    parser.add_argument("--task-ids", type=int, nargs="+")
    parser.add_argument("--trials-per-task", "--trials", "--num-trials-per-task", type=int, default=50)
    parser.add_argument("--cuda", type=int, default=0, help="GPU index within CUDA_VISIBLE_DEVICES")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--denoising-steps", type=int, default=4)
    parser.add_argument("--port", type=int, default=8071)
    parser.add_argument("--startup-timeout", type=float, default=300)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-dir", type=Path, help="Run directory for logs, commands, and rollouts")
    parser.add_argument("--assets-dir", type=Path, default=Path("ra-vla-official-assets"))
    parser.add_argument("--dry-run", action="store_true", help="Validate paths and print commands without running")
    args = parser.parse_args()

    if args.trials_per_task < 1 or args.denoising_steps < 1 or args.startup_timeout <= 0:
        parser.error("trials-per-task, denoising-steps, and startup-timeout must be positive")
    if not 1 <= args.k <= 8:
        parser.error("k must be between 1 and 8 for the released RA-VLA architecture")
    if not 1 <= args.port <= 65535 or args.cuda < 0 or not 0 <= args.seed < 2**32:
        parser.error("port must be 1..65535, cuda >= 0, and seed in 0..2**32-1")
    if args.task_ids is not None and (
        len(set(args.task_ids)) != len(args.task_ids) or any(t < 0 or t >= 10 for t in args.task_ids)
    ):
        parser.error("task-ids must be unique IDs between 0 and 9")
    for name in ("policy_python", "libero_python"):
        executable = shutil.which(getattr(args, name))
        if executable is None:
            parser.error(f"Python executable not found: {getattr(args, name)}")
        setattr(args, name, executable)

    repo = Path(__file__).resolve().parents[2]
    assets = (repo / args.assets_dir).resolve()
    split, memory = SPLITS[args.suite]
    retriever = (repo / args.retriever_path).resolve() if args.retriever_path else assets / f"libero_{split}"
    checkpoint = (
        (repo / args.checkpoint_path).resolve()
        if args.checkpoint_path
        else assets / f"ravla/ravla/libero_{split}/checkpoint_3"
    )
    memory_name = (args.memory_name or f"memory_{memory}").removesuffix(".pt")
    memory_path = (repo / args.memory_path).resolve() if args.memory_path else retriever / f"{memory_name}.pt"
    for path in (memory_path, retriever / "config.json", checkpoint / "config.json", checkpoint / "metadata.pt"):
        if not path.is_file():
            parser.error(f"Required file not found: {path}")

    if args.run_name is None:
        args.run_name = (
            f"ravla_{args.suite}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{time.time_ns() % 10**9:09d}"
        )
    if Path(args.run_name).name != args.run_name or args.run_name in ("", ".", ".."):
        parser.error("run-name must be a single directory name; use output-dir to choose its location")
    output = (repo / args.output_dir).resolve() if args.output_dir else repo / "outputs/ravla_eval" / args.run_name
    rollouts = output / "rollouts"
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(repo) + os.pathsep + env.get("PYTHONPATH", ""),
            "PYTHONUNBUFFERED": "1",
            "NO_ALBUMENTATIONS_UPDATE": "1",
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "NUMBA_CACHE_DIR": str(output / "numba_cache"),
        }
    )
    server_command = [
        args.policy_python,
        "-m",
        "scripts.rag.4_serve_ravla",
        "--retriever-path",
        str(retriever),
        "--memory-path",
        str(memory_path),
        "--checkpoint-path",
        str(checkpoint),
        "--denoising-steps",
        str(args.denoising_steps),
        "--k",
        str(args.k),
        "--seed",
        str(args.seed),
        "--cuda",
        str(args.cuda),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
    ]
    eval_command = [
        args.libero_python,
        "-m",
        "examples.Libero.eval.run_libero_eval",
        "--task-suite-name",
        args.suite,
        "--num-trials-per-task",
        str(args.trials_per_task),
        "--run-name",
        args.run_name,
        "--host",
        "127.0.0.1",
        "--output-dir",
        str(rollouts),
        "--port",
        str(args.port),
        "--headless",
    ]
    if args.task_ids is not None:
        eval_command.extend(["--task-ids", *map(str, args.task_ids)])
    manifest = {
        "suite": args.suite,
        "run_name": args.run_name,
        "cwd": str(repo),
        "server": server_command,
        "eval": eval_command,
        "environment": {
            key: env[key]
            for key in (
                "MUJOCO_GL",
                "PYOPENGL_PLATFORM",
                "HF_HUB_OFFLINE",
                "PYTHONPATH",
            )
        },
    }
    print(f"Suite: {args.suite}; tasks: {args.task_ids or 'all'}; trials/task: {args.trials_per_task}")
    print(f"Checkpoint: {checkpoint}\nRetriever: {retriever}\nMemory: {memory_path}")
    print(f"Server command: {shlex.join(server_command)}")
    print(f"Eval command: {shlex.join(eval_command)}")
    print(f"Logs: {output}\nSummary: {rollouts / 'summary.json'}", flush=True)
    if args.dry_run:
        return

    from gr00t.eval.service import BaseInferenceClient

    # Refuse an occupied port so a different policy cannot receive this evaluation.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    output.mkdir(parents=True, exist_ok=False)
    (output / "commands.json").write_text(json.dumps(manifest, indent=2) + "\n")
    signal.signal(signal.SIGTERM, exit_on_signal)

    with (output / "server.log").open("w") as server_log:
        server = subprocess.Popen(server_command, cwd=repo, env=env, stdout=server_log, stderr=subprocess.STDOUT)
        try:
            client = BaseInferenceClient(host="127.0.0.1", port=args.port, timeout_ms=1000)
            try:
                deadline = time.monotonic() + args.startup_timeout
                while not client.ping():
                    if server.poll() is not None:
                        raise RuntimeError(f"Server exited with {server.returncode}; see {output / 'server.log'}")
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"Server was not ready within {args.startup_timeout} seconds")
                    time.sleep(1)
            finally:
                client.socket.close()
                client.context.term()
            print("Server ready; starting LIBERO rollouts", flush=True)
            with (output / "eval.log").open("w") as eval_log:
                subprocess.run(
                    eval_command, cwd=repo, env=env, stdout=eval_log, stderr=subprocess.STDOUT, check=True
                )
            summary = rollouts / "summary.json"
            print(summary.read_text(), flush=True)
            print(f"Summary: {summary}", flush=True)
        finally:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == "__main__":
    main()
