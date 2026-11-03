"""Resumeable LIBERO-Long diagnosis: standard, probes, and scheduled recovery."""

import argparse
from collections import deque
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import uuid

from gr00t.rag.diagnosis_protocol import digest, instruction_key, make_cases
from scripts.rag.eval_libero_parallel import atomic_json, stop_processes


REPO = Path(__file__).resolve().parents[2]


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def checkpoint_signature(path):
    names = [path / "config.json", path / "metadata.pt", *sorted(path.glob("*.safetensors"))]
    if len(names) < 3 or any(not item.is_file() for item in names):
        raise FileNotFoundError(f"Incomplete checkpoint: {path}")
    return {item.name: file_sha(item) for item in names}


def environment(output):
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(REPO) + os.pathsep + env.get("PYTHONPATH", ""),
        PYTHONUNBUFFERED="1",
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        USE_TF="0",
        NO_ALBUMENTATIONS_UPDATE="1",
        TOKENIZERS_PARALLELISM="false",
        MUJOCO_GL="egl",
        PYOPENGL_PLATFORM="egl",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        MPLCONFIGDIR=str(output / "mpl_cache"),
    )
    return env


def run(args):
    args.output = (REPO / args.output).resolve()
    if args.stage == "report":
        from scripts.rag.diagnosis_report import summarize

        summarize(args.output)
        return
    args.checkpoint = (REPO / args.checkpoint).resolve()
    args.baseline = (REPO / args.baseline).resolve()
    args.memory = (REPO / args.memory).resolve()
    args.retriever = (REPO / args.retriever).resolve()
    if args.smoke:
        args.task_ids, args.trials, args.max_steps, args.probe_states = [0, 4], 1, 110, 8
    if (
        len(set(args.task_ids)) != len(args.task_ids)
        or not args.task_ids
        or any(t not in range(10) for t in args.task_ids)
    ):
        raise ValueError("task_ids must be distinct values in 0..9")
    if min(args.trials, args.workers, args.max_steps, args.probe_states) < 1 or args.video_trials < 0:
        raise ValueError("Invalid trial/worker/step/probe/video counts")
    if any(t < 0 or t + 36 > args.max_steps for t in args.triggers):
        raise ValueError("Episode must cover all four disturbance steps and the 32-step recovery window")
    if any(s < 0 or not __import__("math").isfinite(s) for s in args.scales):
        raise ValueError("Disturbance scales must be finite and nonnegative")
    if args.initial_start is not None and args.initial_start < 0:
        raise ValueError("initial_start must be nonnegative")
    cases = make_cases(args.stage, args.triggers, args.scales)
    conditions = getattr(args, "conditions", None)
    if conditions is not None:
        if args.stage != "standard" or len(set(conditions)) != len(conditions):
            raise ValueError("--conditions requires standard stage and distinct condition names")
        cases = [case for case in cases if case["name"] in conditions]
        if not cases:
            raise ValueError("No matching standard conditions")
    memory_manifest = json.loads((args.memory / "manifest.json").read_text())
    if memory_manifest.get("status") != "completed" or not memory_manifest.get("diagnosis_state"):
        raise ValueError("First build a completed reference memory with --diagnosis")
    references = {}
    for task, record in memory_manifest["tasks"].items():
        key = instruction_key(task)
        path = args.memory / record["ee_file"]
        if key in references or not path.is_file():
            raise ValueError("Ambiguous task identity or missing reference EE array")
        references[key] = str(path)
    settings = {
        "protocol_version": 1,
        "stage": args.stage,
        "checkpoint": str(args.checkpoint),
        "baseline": str(args.baseline),
        "memory": str(args.memory),
        "retriever": str(args.retriever),
        "task_ids": args.task_ids,
        "trials": args.trials,
        "max_steps": args.max_steps,
        "initial_start": args.initial_start,
        "seed": args.seed,
        "video_trials": args.video_trials,
        "triggers": args.triggers,
        "scales": args.scales,
        "probe_states": args.probe_states,
        "smoke": args.smoke,
        "action_horizon": 16,
        "denoising_steps": 4,
        "support": "FP32 cosine to unchanged BF16-selected top-1; log-only",
    }
    expected = len(cases) * len(args.task_ids) * args.trials
    if conditions is not None:
        settings["conditions"] = conditions
    print(json.dumps({"settings": settings, "expected_rollouts": expected, "cases": cases}, indent=2), flush=True)
    if args.dry_run:
        return
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_STEP_ID"):
        raise RuntimeError("Run diagnosis with srun on a Slurm GPU worker")
    inputs = {"dual": checkpoint_signature(args.checkpoint), "memory": file_sha(args.memory / "manifest.json")}
    if any(case["model"] == "baseline" for case in cases):
        inputs["baseline"] = checkpoint_signature(args.baseline)
    sources = [
        "gr00t/rag/dual_mode_diagnosis.py",
        "gr00t/rag/diagnosis_protocol.py",
        "examples/Libero/eval/run_dual_diagnosis.py",
        "scripts/rag/diagnose_dual_mode.py",
        "scripts/rag/probe_dual_mode.py",
    ]
    inputs["source"] = {name: file_sha(REPO / name) for name in sources}
    signature = digest({"settings": settings, "inputs": inputs})
    manifest = {
        "settings": settings,
        "inputs": inputs,
        "signature": signature,
        "cases": cases,
        "expected_episodes": expected,
        "status": "starting",
    }
    if args.output.exists():
        if not args.resume:
            raise FileExistsError("Output exists; use --resume with the same protocol")
        old = json.loads((args.output / "manifest.json").read_text())
        if old["signature"] != signature:
            raise ValueError("Resume protocol/checkpoint/source signature differs")
    else:
        if args.resume:
            raise FileNotFoundError(args.output)
        args.output.mkdir(parents=True)
    with (args.output / ".diagnosis.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(args, manifest, references)


def execute(args, manifest, references):
    from gr00t.eval.service import BaseInferenceClient
    from scripts.rag.diagnosis_report import summarize

    env = environment(args.output)
    run = None
    server = None
    active = {}
    logs = []
    seen = set()
    atomic_json(args.output / "manifest.json", manifest)

    def progress():
        for path in sorted(args.output.glob("*/task*_init*/episode.json")):
            if path in seen:
                continue
            row = json.loads(path.read_text())
            if row.get("complete"):
                seen.add(path)
                if run is not None:
                    run.log(
                        {
                            "episodes": len(seen),
                            "progress/success": int(row["success"]),
                            f"{row['case']['name']}/success": int(row["success"]),
                        }
                    )
        atomic_json(
            args.output / "progress.json",
            {
                "status": manifest["status"],
                "completed_episodes": len(seen),
                "expected_episodes": manifest["expected_episodes"],
                "active": list(active),
                "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )

    try:
        if args.wandb_mode != "disabled":
            import wandb

            run = wandb.init(
                project="icl-vla-uncertainty",
                entity=args.wandb_entity,
                name=args.wandb_name or args.output.name,
                group="dual_mode_diagnosis",
                mode=args.wandb_mode,
                dir=str(args.output),
                config=manifest["settings"],
            )
            run.define_metric("episodes")
            run.define_metric("progress/*", step_metric="episodes")
        for model_name in ("baseline", "dual"):
            cases = [case for case in manifest["cases"] if case["model"] == model_name]
            if not cases:
                continue
            identity = uuid.uuid4().hex
            checkpoint = args.baseline if model_name == "baseline" else args.checkpoint
            command = [
                args.policy_python,
                "-u",
                "-m",
                "scripts.rag.serve_dual_mode",
                "--checkpoint",
                str(checkpoint),
                "--memory",
                str(args.memory),
                "--retriever",
                str(args.retriever),
                "--port",
                str(args.port),
                "--server-id",
                identity,
            ]
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", args.port))
            log = (args.output / f"server_{model_name}.log").open("a")
            logs.append(log)
            server = subprocess.Popen(
                command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            client = BaseInferenceClient(host="127.0.0.1", port=args.port, timeout_ms=1000)
            try:
                deadline = time.monotonic() + 300
                while not client.ping():
                    if server.poll() is not None:
                        raise RuntimeError(f"Policy server exited; see server_{model_name}.log")
                    if time.monotonic() > deadline:
                        raise TimeoutError("Policy server startup timed out")
                    time.sleep(0.5)
                if client.call_endpoint("server_info", requires_input=False)["server_id"] != identity:
                    raise RuntimeError("Policy server identity mismatch")
            finally:
                client.socket.close(linger=0)
                client.context.term()
            manifest["status"] = "running"
            pending = deque((case, task) for case in cases for task in args.task_ids)
            while pending or active:
                if server.poll() is not None:
                    raise RuntimeError("Policy server died during rollouts")
                while pending and len(active) < args.workers:
                    case, task = pending.popleft()
                    name = f"{case['name']}_task{task:02d}"
                    start = (
                        args.initial_start
                        if args.initial_start is not None
                        else (5 if case["kind"] == "recovery" else 0)
                    )
                    config = {
                        "signature": manifest["signature"],
                        "case": case,
                        "task_id": task,
                        "trials": args.trials,
                        "initial_start": start,
                        "seed": args.seed,
                        "max_steps": args.max_steps,
                        "video_trials": args.video_trials,
                        "references": references,
                        "port": args.port,
                        "output": str(args.output),
                    }
                    config_path = args.output / "configs" / f"{name}.json"
                    config_path.parent.mkdir(exist_ok=True)
                    atomic_json(config_path, config)
                    log = (args.output / f"{name}.log").open("a")
                    logs.append(log)
                    task_env = env.copy()
                    task_env.update(
                        NUMBA_CACHE_DIR=str(args.output / "cache" / name / "numba"),
                        HF_MODULES_CACHE=str(args.output / "cache" / name / "hf"),
                    )
                    command = [
                        args.libero_python,
                        "-u",
                        "-m",
                        "examples.Libero.eval.run_dual_diagnosis",
                        "--config",
                        str(config_path),
                    ]
                    active[name] = subprocess.Popen(
                        command,
                        cwd=REPO,
                        env=task_env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    print(f"START {name}", flush=True)
                for name, process in list(active.items()):
                    code = process.poll()
                    if code is not None:
                        if code:
                            raise RuntimeError(f"{name} exited {code}; see {name}.log")
                        del active[name]
                        print(f"FINISHED {name}", flush=True)
                progress()
                time.sleep(0.5)
            stop_processes([server])
            server = None
        if args.stage in ("probes", "all"):
            probe_output = args.output / "probes"
            if not (probe_output / "summary.json").exists():
                command = [
                    args.policy_python,
                    "-u",
                    "-m",
                    "scripts.rag.probe_dual_mode",
                    "--checkpoint",
                    str(args.checkpoint),
                    "--memory",
                    str(args.memory),
                    "--retriever",
                    str(args.retriever),
                    "--output",
                    str(probe_output),
                    "--states",
                    str(args.probe_states),
                    "--seed",
                    str(args.seed),
                ]
                with (args.output / "probes.log").open("a") as log:
                    process = subprocess.Popen(
                        command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
                    )
                    active["probes"] = process
                    if process.wait():
                        raise RuntimeError("Goal probes failed; see probes.log")
                    del active["probes"]
        progress()
        if len(seen) != manifest["expected_episodes"]:
            raise ValueError("Rollout count differs from the plan")
        rows, audit = summarize(args.output)
        if audit["partial_groups"]:
            raise ValueError("Incomplete recovery A/B/C groups")
        manifest["status"] = "completed"
        if run is not None:
            import wandb

            for row in rows:
                run.summary[f"{row['condition']}/success_pct"] = row["success_pct"]
            for filename in ("success.png", "contraction.png", "support_traces.png"):
                path = args.output / filename
                if path.exists():
                    run.log({filename.removesuffix(".png"): wandb.Image(str(path))})
            probe_summary = args.output / "probes" / "summary.json"
            if probe_summary.exists():
                summary = json.loads(probe_summary.read_text())
                for key, value in summary.items():
                    if isinstance(value, dict):
                        run.summary[f"probe/{key}/median"] = value["median"]
        print(f"COMPLETED: {args.output / 'summary.md'}", flush=True)
    except BaseException as error:
        manifest.update(
            status="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
            error=f"{type(error).__name__}: {error}",
        )
        raise
    finally:
        stop_processes([*active.values(), *([server] if server is not None else [])])
        for log in logs:
            log.close()
        atomic_json(args.output / "manifest.json", manifest)
        progress()
        if run is not None:
            run.finish(exit_code=0 if manifest["status"] == "completed" else 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--baseline", type=Path, default=Path("outputs/ravla_train/gos_seed42_20261026T211436Z/checkpoint_3")
    )
    parser.add_argument("--retriever", type=Path, default=Path("ra-vla-official-assets/libero_gos"))
    parser.add_argument("--memory", type=Path, default=Path("outputs/dual_mode/long_reference_diagnosis"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("standard", "probes", "recovery", "all", "report"), default="standard")
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=("baseline", "dual_action", "dual_goal"),
        help="Run selected standard conditions, e.g. --conditions baseline",
    )
    parser.add_argument("--task-ids", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--initial-start", type=int)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--triggers", type=int, nargs="+", default=[60])
    parser.add_argument("--scales", type=float, nargs="+", default=[0, 1])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--video-trials", type=int, default=10)
    parser.add_argument("--probe-states", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", type=int, default=8095)
    parser.add_argument("--policy-python", default=sys.executable)
    parser.add_argument("--libero-python", default=str(Path.home() / "miniconda3/envs/libero/bin/python"))
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--wandb-entity", default="ryan-nlp")
    parser.add_argument("--wandb-name")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    run(args)


if __name__ == "__main__":
    main()
