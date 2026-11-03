"""Launch/resume privileged same-scene rollout diagnosis on a Slurm GPU worker."""

import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import uuid

from gr00t.eval.service import BaseInferenceClient
from gr00t.rag.diagnosis_protocol import digest
from gr00t.rag.oracle_goal_protocol import summarize_episodes
from scripts.rag.diagnose_dual_mode import checkpoint_signature, environment, file_sha
from scripts.rag.eval_libero_parallel import atomic_json, stop_processes


REPO = Path(__file__).resolve().parents[2]
SUITES = ["libero_goal", "libero_object", "libero_spatial"]
LIMITS = dict(libero_goal=600, libero_object=280, libero_spatial=220)


def report(output, expected):
    rows = [json.loads(p.read_text()) for p in sorted(output.glob("*/*/episode.json"))]
    summary = summarize_episodes(rows)
    summary.update(completed_cases=len(rows), expected_cases=expected)
    atomic_json(output / "summary.json", summary)
    fields = [
        "case",
        "suite",
        "task_id",
        "initial_index",
        "success",
        "eligible",
        "skipped",
        "control_steps",
        "predicate_completion",
    ]
    with (output / "episodes.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return summary


def execute(args, manifest, recipes):
    output = args.output
    env = environment(output)
    identity = uuid.uuid4().hex
    server, workers, handles = None, [], []
    run = None

    def stop(*_):
        raise KeyboardInterrupt

    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        if args.wandb_mode != "disabled":
            import wandb

            run = wandb.init(
                project="icl-vla-uncertainty",
                entity="ryan-nlp",
                name=output.name,
                group="isolated_goal_oracle",
                mode=args.wandb_mode,
                dir=str(output),
                config=manifest["settings"],
            )
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", args.port))
        log = (output / "server.log").open("a")
        handles.append(log)
        command = [
            sys.executable,
            "-u",
            "-m",
            "scripts.rag.serve_goal_oracle",
            "--checkpoint",
            str(args.checkpoint),
            "--baseline",
            str(args.baseline),
            "--retriever",
            str(args.retriever),
            "--port",
            str(args.port),
            "--server-id",
            identity,
        ]
        server = subprocess.Popen(
            command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        client = BaseInferenceClient(host="127.0.0.1", port=args.port, timeout_ms=1000)
        try:
            deadline = time.monotonic() + 600
            while not client.ping():
                if server.poll() is not None:
                    raise RuntimeError("Oracle policy server exited; see server.log")
                if time.monotonic() > deadline:
                    raise TimeoutError("Policy startup timed out")
                time.sleep(0.5)
            info = client.call_endpoint("server_info", requires_input=False)
            if info["server_id"] != identity:
                raise ValueError("Server identity mismatch")
            atomic_json(output / "server_info.json", info)
        finally:
            client.socket.close(linger=0)
            client.context.term()
        manifest["status"] = "running"
        atomic_json(output / "manifest.json", manifest)
        for recipe in recipes:
            config_path = output / "configs" / (recipe["suite"] + ".json")
            config_path.parent.mkdir(exist_ok=True)
            atomic_json(config_path, recipe)
            task_env = env.copy()
            task_env["NUMBA_CACHE_DIR"] = str(output / "cache" / recipe["suite"])
            log = (output / (recipe["suite"] + ".log")).open("a")
            handles.append(log)
            workers.append(
                subprocess.Popen(
                    [
                        args.libero_python,
                        "-u",
                        "-m",
                        "examples.Libero.eval.run_goal_oracle",
                        "--config",
                        str(config_path),
                    ],
                    cwd=REPO,
                    env=task_env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )
            print("START", recipe["suite"], flush=True)
        previous_count = -1
        while any(p.poll() is None for p in workers):
            if server.poll() is not None:
                raise RuntimeError("Policy server died during rollout")
            if any(p.poll() not in (0, None) for p in workers):
                raise RuntimeError("Simulator worker failed; see suite logs")
            summary = report(output, manifest["expected_cases"])
            atomic_json(output / "progress.json", dict(status="running", **summary))
            if summary["completed_cases"] != previous_count:
                print("PROGRESS", json.dumps(summary), flush=True)
                if run is not None:
                    run.log({k: v for k, v in summary.items() if isinstance(v, (int, float))})
                previous_count = summary["completed_cases"]
            time.sleep(2)
        if any(p.returncode for p in workers):
            raise RuntimeError("Simulator worker failed; see suite logs")
        summary = report(output, manifest["expected_cases"])
        if summary["completed_cases"] != manifest["expected_cases"]:
            raise RuntimeError("Incomplete cohort")
        manifest["status"] = "completed"
        atomic_json(output / "progress.json", dict(status="completed", **summary))
        if run is not None:
            run.summary.update({k: v for k, v in summary.items() if isinstance(v, (int, float))})
        print("COMPLETED", json.dumps(summary), flush=True)
    except BaseException as error:
        manifest.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error)
        )
        raise
    finally:
        stop_processes([*workers, *([server] if server is not None else [])])
        atomic_json(output / "manifest.json", manifest)
        for handle in handles:
            handle.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if run is not None:
            run.finish()


def main(args):
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_STEP_ID"):
        raise RuntimeError("Use srun on a Slurm GPU worker")
    for key in ("output", "checkpoint", "baseline", "retriever"):
        setattr(args, key, (REPO / getattr(args, key)).resolve())
    if args.trials < 1 or not 0 < args.port < 65536:
        raise ValueError("Invalid trials/port")
    training = json.loads((args.checkpoint / "training_config.json").read_text())
    selected = list(range(3)) if not args.smoke else [0]
    settings = dict(
        checkpoint=str(args.checkpoint),
        baseline=str(args.baseline),
        retriever=str(args.retriever),
        trials=args.trials,
        seed=args.seed,
        smoke=args.smoke,
        horizon=16,
        nfe=4,
        suites=[SUITES[i] for i in selected],
        goal_reference="successful same-initial-state Mode-1 rollout",
        phase_selector="privileged monotone local EE/object-position/gripper matching; window32; lookahead16",
        reference_conditioning="both camera views and language, matching same-demo goal training",
        evaluation_only=True,
    )
    memory_stat = (args.retriever / "memory_gos.pt").stat()
    inputs = dict(
        checkpoint=checkpoint_signature(args.checkpoint),
        baseline=checkpoint_signature(args.baseline),
        retriever=file_sha(args.retriever / "model.safetensors"),
        memory_identity=dict(
            path=str(args.retriever / "memory_gos.pt"), size=memory_stat.st_size, mtime_ns=memory_stat.st_mtime_ns
        ),
    )
    sources = [
        "gr00t/rag/oracle_goal_protocol.py",
        "gr00t/rag/oracle_goal_policy.py",
        "scripts/rag/serve_goal_oracle.py",
        "scripts/rag/eval_goal_oracle.py",
        "examples/Libero/eval/run_goal_oracle.py",
        "gr00t/rag/goal_control.py",
        "gr00t/rag/dual_mode.py",
    ]
    inputs["source"] = {name: file_sha(REPO / name) for name in sources}
    signature = digest(dict(settings=settings, inputs=inputs))
    recipes = [
        dict(
            signature=signature,
            suite=SUITES[i],
            task_id=0,
            task=training["split"][i]["task"],
            trials=args.trials,
            seed=args.seed,
            max_steps=32 if args.smoke else LIMITS[SUITES[i]],
            smoke=args.smoke,
            port=args.port,
            output=str(args.output),
        )
        for i in selected
    ]
    manifest = dict(
        signature=signature,
        settings=settings,
        inputs=inputs,
        expected_cases=2 * len(recipes) * args.trials,
        status="starting",
    )
    if args.output.exists():
        if not args.resume:
            raise FileExistsError("Output exists; use --resume with the same protocol")
        old = json.loads((args.output / "manifest.json").read_text())
        if old["signature"] != signature:
            raise ValueError("Resume source/checkpoint/protocol differs")
    else:
        if args.resume:
            raise FileNotFoundError(args.output)
        args.output.mkdir(parents=True)
    with (args.output / ".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_json(args.output / "manifest.json", manifest)
        execute(args, manifest, recipes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/dual_mode/debug_20261102/goal_same_demo_seed42_spawn/checkpoint-2000"),
    )
    parser.add_argument(
        "--baseline", type=Path, default=Path("outputs/ravla_train/gos_seed42_20261026T211436Z/checkpoint_3")
    )
    parser.add_argument("--retriever", type=Path, default=Path("ra-vla-official-assets/libero_gos"))
    parser.add_argument("--libero-python", default="/home/vhoangth2/miniconda3/envs/libero/bin/python")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", type=int, default=8115)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    main(parser.parse_args())
