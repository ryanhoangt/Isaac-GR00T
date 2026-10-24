"""Evaluate local LIBERO checkpoints sequentially with a managed inference server."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[1]
SUITES = {"spatial": "libero_spatial", "object": "libero_object",
          "goal": "libero_goal", "long": "libero_10", "90": "libero_90"}


def complete_checkpoint(path):
    if not (path / "config.json").is_file():
        return False
    if path.name.startswith("checkpoint-") and not (path / "trainer_state.json").is_file():
        return False  # An active training job may still be writing this checkpoint.
    index = path / "model.safetensors.index.json"
    if index.is_file():
        names = set(json.loads(index.read_text())["weight_map"].values())
        return bool(names) and all((path / name).is_file() and (path / name).stat().st_size for name in names)
    return (path / "model.safetensors").is_file()


def discover(inputs, latest=False):
    found = set()
    for value in inputs:
        root = Path(value).expanduser().resolve(strict=True)
        if complete_checkpoint(root):
            found.add(root)
            continue
        candidates = list(root.glob("checkpoint-*")) + list(root.glob("heldout_*/checkpoint-*"))
        complete = [p.resolve() for p in candidates if complete_checkpoint(p)]
        if not complete:
            raise ValueError(f"No complete checkpoints found in {root}")
        if latest:
            groups = defaultdict(list)
            for path in complete:
                groups[path.parent].append(path)
            complete = [max(paths, key=lambda p: int(p.name.split("-")[-1])) for paths in groups.values()]
        found.update(complete)
    return sorted(found, key=lambda p: (str(p.parent), int(p.name.split("-")[-1])
                                       if p.name.startswith("checkpoint-") else 0))


def checkpoint_settings(path, suite="auto", data_config=None):
    manifest_path = path.parent / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    if suite == "auto":
        suite = manifest.get("held_out_suite")
        if suite is None and path.parent.name.startswith("heldout_"):
            suite = path.parent.name.removeprefix("heldout_")
        if suite not in SUITES:
            raise ValueError(f"Cannot infer held-out suite for {path}; pass --suite explicitly")
    suite = SUITES.get(suite, suite)
    return suite, data_config or manifest.get(
        "data_config", "examples.Libero.custom_data_config:LiberoDataConfig")


def stop_process(process):
    if process is not None and process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def wait_for_server(process, port, timeout):
    import msgpack
    import zmq

    deadline = time.monotonic() + timeout
    with zmq.Context() as context:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("Inference server exited; inspect server.log")
            with context.socket(zmq.REQ) as client:
                client.setsockopt(zmq.LINGER, 0)
                client.setsockopt(zmq.RCVTIMEO, 500)
                client.setsockopt(zmq.SNDTIMEO, 500)
                client.connect(f"tcp://127.0.0.1:{port}")
                try:
                    client.send(msgpack.packb({"endpoint": "ping"}))
                    if msgpack.unpackb(client.recv(), raw=False).get("status") == "ok":
                        return
                except zmq.ZMQError:
                    pass
            time.sleep(0.5)
    raise TimeoutError(f"Server was not ready after {timeout}s; inspect server.log")


def evaluate(checkpoint, args, suite, data_config):
    tag = hashlib.sha256(str(checkpoint.parent).encode()).hexdigest()[:8]
    parent = args.output_root / f"{checkpoint.parent.name}-{tag}" / checkpoint.name / suite
    settings = {"checkpoint": str(checkpoint), "suite": suite, "data_config": data_config,
                "trials_per_task": args.trials, "seed": args.seed,
                "denoising_steps": args.denoising_steps, "embodiment_tag": args.embodiment_tag,
                "save_videos": not args.no_videos}
    if not args.rerun:
        for manifest in sorted(parent.glob("*/eval_manifest.json")):
            summary_path = manifest.parent / "summary.json"
            if json.loads(manifest.read_text()) == settings and summary_path.exists():
                summary = json.loads(summary_path.read_text())
                if summary.get("status") == "completed":
                    print(f"SKIP completed: {summary_path}", flush=True)
                    return
    print(f"EVAL {checkpoint} -> {suite}, {args.trials} trials/task", flush=True)
    if args.dry_run:
        print(f"  config={data_config}; output parent={parent}", flush=True)
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = parent / stamp
    output.mkdir(parents=True)
    (output / "eval_manifest.json").write_text(json.dumps(settings, indent=2) + "\n")
    env = os.environ.copy()
    libero_repo = Path(env.get("LIBERO_REPO", REPO.parent / "LIBERO")).expanduser().resolve()
    env.update(PYTHONUNBUFFERED="1", MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", USE_TF="0",
               NO_ALBUMENTATIONS_UPDATE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               TOKENIZERS_PARALLELISM="false")
    env["PYTHONPATH"] = os.pathsep.join([str(REPO), str(libero_repo), env.get("PYTHONPATH", "")])
    env.setdefault("HF_MODULES_CACHE", str(REPO / ".cache/huggingface/modules"))
    if (libero_repo / "libero/libero/assets").is_dir() and "LIBERO_CONFIG_PATH" not in env:
        base = libero_repo / "libero/libero"
        config_dir = output / ".libero"
        config_dir.mkdir()
        config = {"benchmark_root": str(base), "assets": str(base / "assets"),
                  "bddl_files": str(base / "bddl_files"), "init_states": str(base / "init_files"),
                  "datasets": str(base.parent / "datasets")}
        (config_dir / "config.yaml").write_text(json.dumps(config, indent=2))  # JSON is valid YAML.
        env["LIBERO_CONFIG_PATH"] = str(config_dir)
    subprocess.run([args.eval_python, "-c",
                    "from libero.libero import benchmark; "
                    "from libero.libero.envs import OffScreenRenderEnv; "
                    "from examples.Libero.eval.utils import get_libero_env; "
                    "import zmq, msgpack, tyro; print('LIBERO imports OK')"],
                   cwd=REPO, env=env, check=True)
    subprocess.run([sys.executable, "-c", "import torch; "
                    "assert torch.cuda.is_available(), 'Run inside a GPU allocation'"],
                   cwd=REPO, env=env, check=True)
    # Reject occupied explicit ports; use a fresh free port by default.
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", args.port))
        port = reserved.getsockname()[1]
    server = evaluator = None
    try:
        with (output / "server.log").open("w") as server_log, (output / "eval.log").open("w") as eval_log:
            server = subprocess.Popen(
                [sys.executable, str(REPO / "scripts/inference_service.py"), "--server",
                 "--model-path", str(checkpoint), "--data-config", data_config,
                 "--embodiment-tag", args.embodiment_tag, "--denoising-steps", str(args.denoising_steps),
                 "--seed", str(args.seed), "--port", str(port)],
                cwd=REPO, env=env, stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)
            print(f"  Logs: {output}", flush=True)
            wait_for_server(server, port, args.server_timeout)
            command = [args.eval_python, str(REPO / "examples/Libero/eval/run_libero_eval.py"),
                       "--task-suite-name", suite, "--num-trials-per-task", str(args.trials),
                       "--headless", "--port", str(port), "--seed", str(args.seed),
                       "--request-timeout-ms", str(args.request_timeout * 1000),
                       "--output-dir", str(output)]
            if args.no_videos:
                command.append("--no-save-videos")
            evaluator = subprocess.Popen(command, cwd=REPO, env=env, stdout=eval_log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
            while evaluator.poll() is None:
                if server.poll() is not None:
                    raise RuntimeError(f"Inference server exited; inspect {output / 'server.log'}")
                time.sleep(1)
            if evaluator.returncode:
                raise RuntimeError(f"Evaluator exited with {evaluator.returncode}; inspect {output / 'eval.log'}")
            summary = json.loads((output / "summary.json").read_text())
            if summary["status"] != "completed":
                raise RuntimeError(f"Incomplete evaluation: {output}")
            print(f"RESULT {suite}: {summary['total_successes']}/{summary['total_episodes']} "
                  f"= {summary['success_rate']:.2%}; {output / 'summary.json'}", flush=True)
    finally:
        stop_process(evaluator)
        stop_process(server)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Checkpoint, training run, or experiment directory")
    parser.add_argument("--suite", default="auto", choices=["auto", *SUITES, *SUITES.values()])
    parser.add_argument("--latest", action="store_true", help="Only latest complete checkpoint per run")
    parser.add_argument("--trials", type=int, default=5, help="Rollouts/task; use 50 for 500 rollouts/suite")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--denoising-steps", type=int, default=8)
    parser.add_argument("--data-config", default=None, help="Defaults to training manifest, then min-max LIBERO config")
    parser.add_argument("--embodiment-tag", default="new_embodiment")
    parser.add_argument("--output-root", type=Path, default=REPO / "outputs/libero_eval")
    parser.add_argument("--eval-python", default=os.environ.get("LIBERO_EVAL_PYTHON", sys.executable))
    parser.add_argument("--port", type=int, default=0, help="0 chooses a free port automatically")
    parser.add_argument("--server-timeout", type=int, default=600)
    parser.add_argument("--request-timeout", type=int, default=60)
    parser.add_argument("--no-videos", action="store_true")
    parser.add_argument("--rerun", action="store_true", help="Reevaluate already completed identical settings")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without loading a model or simulator")
    args = parser.parse_args()
    if min(args.trials, args.denoising_steps, args.server_timeout, args.request_timeout) < 1:
        parser.error("Trials, denoising steps, and timeouts must be positive")
    if not 0 <= args.port <= 65535:
        parser.error("Port must be in 0..65535")
    args.output_root = args.output_root.expanduser().resolve()
    # Validate the whole plan before starting the first GPU server.
    plan = [(path, *checkpoint_settings(path, args.suite, args.data_config))
            for path in discover(args.paths, args.latest)]
    for checkpoint, _, _ in plan:
        if not (checkpoint / "experiment_cfg/metadata.json").is_file():
            raise FileNotFoundError(f"Missing normalization metadata in {checkpoint}")

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    for checkpoint, suite, data_config in plan:
        evaluate(checkpoint, args, suite, data_config)


if __name__ == "__main__":
    main()
