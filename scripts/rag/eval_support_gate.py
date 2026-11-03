"""Frozen-threshold LIBERO-Long eval: dual-action control versus adaptive q90/q95."""

import argparse
from collections import deque
from dataclasses import asdict
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

from gr00t.rag.diagnosis_protocol import digest, instruction_key
from gr00t.rag.support_gate import GateConfig
from scripts.rag.diagnose_dual_mode import checkpoint_signature, environment, file_sha
from scripts.rag.eval_libero_parallel import atomic_json, stop_processes


REPO = Path(__file__).resolve().parents[2]


def run(args):
    args.output = (REPO / args.output).resolve()
    if args.stage == "report":
        from scripts.rag.support_gate_report import summarize

        summarize(args.output)
        return
    for name in ("checkpoint", "calibration", "memory", "retriever"):
        setattr(args, name, (REPO / getattr(args, name)).resolve())
    if args.smoke:
        args.task_ids, args.trials, args.max_steps = [0, 4], 1, 160
    if (
        not args.task_ids
        or len(set(args.task_ids)) != len(args.task_ids)
        or any(t not in range(10) for t in args.task_ids)
    ):
        raise ValueError("task_ids must be distinct values in 0..9")
    if min(args.trials, args.max_steps, args.workers) < 1 or args.video_trials < 0 or args.initial_start < 0:
        raise ValueError("Invalid evaluation counts")
    if not args.variants or len(set(args.variants)) != len(args.variants):
        raise ValueError("Variants must be distinct")
    config = GateConfig(args.max_goal_chunks, args.cooldown_chunks, args.enter_consecutive)
    calibration = json.loads(args.calibration.read_text())
    if calibration.get("signature") != digest({k: v for k, v in calibration.items() if k != "signature"}):
        raise ValueError("Calibration content signature differs")
    if calibration.get("status") != "completed" or calibration["settings"]["device_type"] != "cuda":
        raise ValueError("A completed CUDA calibration is required")
    memory = json.loads((args.memory / "manifest.json").read_text())
    if memory.get("status") != "completed" or not memory.get("diagnosis_state"):
        raise ValueError("A completed --diagnosis reference cache is required")
    if file_sha(args.memory / "manifest.json") != calibration["inputs"]["manifest_sha256"]:
        raise ValueError("Calibration memory manifest differs")
    references = {
        instruction_key(task): str(args.memory / record["ee_file"]) for task, record in memory["tasks"].items()
    }
    if len(references) != len(memory["tasks"]) or any(not Path(p).is_file() for p in references.values()):
        raise ValueError("Ambiguous task mapping or missing reference EE array")
    cases = [
        dict(
            name="dual_action" if v == "action" else f"adaptive_{v}",
            kind="support",
            strategy="action" if v == "action" else "adaptive",
            variant="q95" if v == "action" else v,
        )
        for v in args.variants
    ]
    settings = dict(
        checkpoint=str(args.checkpoint),
        calibration=str(args.calibration),
        memory=str(args.memory),
        retriever=str(args.retriever),
        task_ids=args.task_ids,
        trials=args.trials,
        initial_start=args.initial_start,
        max_steps=args.max_steps,
        seed=args.seed,
        video_trials=args.video_trials,
        variants=args.variants,
        gate=asdict(config),
        action_horizon=16,
        denoising_steps=4,
        smoke=args.smoke,
        protocol_version=1,
    )
    expected = len(cases) * len(args.task_ids) * args.trials
    print(json.dumps(dict(settings=settings, expected_rollouts=expected), indent=2), flush=True)
    if args.dry_run:
        return
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_STEP_ID"):
        raise RuntimeError("Run with srun on a GPU worker")
    sources = [
        "gr00t/rag/support_gate.py",
        "gr00t/rag/support_calibration.py",
        "gr00t/rag/support_policy.py",
        "gr00t/rag/dual_mode.py",
        "gr00t/rag/dual_mode_data.py",
        "gr00t/rag/dual_mode_memory.py",
        "gr00t/rag/dual_mode_diagnosis.py",
        "gr00t/rag/retriever.py",
        "gr00t/rag/diagnosis_protocol.py",
        "scripts/rag/serve_dual_mode.py",
        "scripts/rag/serve_support_gate.py",
        "scripts/rag/eval_support_gate.py",
        "examples/Libero/eval/run_support_gate.py",
        "examples/Libero/eval/run_dual_diagnosis.py",
        "examples/Libero/eval/run_libero_eval.py",
        "examples/Libero/eval/utils.py",
    ]
    inputs = dict(
        checkpoint=checkpoint_signature(args.checkpoint),
        calibration=file_sha(args.calibration),
        source={name: file_sha(REPO / name) for name in sources},
    )
    signature = digest(dict(settings=settings, inputs=inputs))
    manifest = dict(
        settings=settings,
        inputs=inputs,
        signature=signature,
        cases=cases,
        calibration=calibration,
        expected_episodes=expected,
        status="starting",
    )
    if args.output.exists():
        if not args.resume:
            raise FileExistsError("Output exists; add --resume with identical settings")
        old = json.loads((args.output / "manifest.json").read_text())
        if old["signature"] != signature:
            raise ValueError("Resume protocol/checkpoint/calibration/source signature differs")
    else:
        if args.resume:
            raise FileNotFoundError(args.output)
        args.output.mkdir(parents=True)
    with (args.output / ".support_eval.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(args, manifest, references)


def execute(args, manifest, references):
    from gr00t.eval.service import BaseInferenceClient
    from scripts.rag.support_gate_report import summarize

    env = environment(args.output)
    server = run = None
    active, logs, seen = {}, [], set()
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
                        dict(
                            episodes=len(seen),
                            **{
                                f"{row['case']['name']}/success": int(row["success"]),
                                f"{row['case']['name']}/goal_step_pct": 100
                                * row["goal_control_steps"]
                                / max(row["control_steps"], 1),
                            },
                        )
                    )
        atomic_json(
            args.output / "progress.json",
            dict(
                status=manifest["status"],
                completed_episodes=len(seen),
                expected_episodes=manifest["expected_episodes"],
                active=list(active),
                updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            ),
        )

    try:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", args.port))
        if args.wandb_mode != "disabled":
            import wandb

            run = wandb.init(
                project="icl-vla-uncertainty",
                entity=args.wandb_entity,
                name=args.wandb_name or args.output.name,
                group="retrieval_support_gate",
                mode=args.wandb_mode,
                dir=str(args.output),
                config=manifest["settings"],
            )
        atomic_json(args.output / "calibration.json", manifest["calibration"])
        identity = uuid.uuid4().hex
        command = [
            args.policy_python,
            "-u",
            "-m",
            "scripts.rag.serve_support_gate",
            "--checkpoint",
            str(args.checkpoint),
            "--calibration",
            str(args.calibration),
            "--memory",
            str(args.memory),
            "--retriever",
            str(args.retriever),
            "--port",
            str(args.port),
            "--server-id",
            identity,
            "--max-goal-chunks",
            str(args.max_goal_chunks),
            "--cooldown-chunks",
            str(args.cooldown_chunks),
            "--enter-consecutive",
            str(args.enter_consecutive),
        ]
        log = (args.output / "server.log").open("a")
        logs.append(log)
        server = subprocess.Popen(
            command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        client = BaseInferenceClient(host="127.0.0.1", port=args.port, timeout_ms=1000)
        try:
            deadline = time.monotonic() + 300
            while not client.ping():
                if server.poll() is not None:
                    raise RuntimeError("Support server exited; see server.log")
                if time.monotonic() > deadline:
                    raise TimeoutError("Support server startup timed out")
                time.sleep(0.5)
            info = client.call_endpoint("server_info", requires_input=False)
            if (
                info["server_id"] != identity
                or info["calibration_signature"] != manifest["calibration"]["signature"]
            ):
                raise RuntimeError("Server identity/calibration mismatch")
        finally:
            client.socket.close(linger=0)
            client.context.term()
        manifest["status"] = "running"
        pending = deque((case, task) for case in manifest["cases"] for task in args.task_ids)
        while pending or active:
            if server.poll() is not None:
                raise RuntimeError("Support server died during rollouts")
            while pending and len(active) < args.workers:
                case, task = pending.popleft()
                name = f"{case['name']}_task{task:02d}"
                config = dict(
                    signature=manifest["signature"],
                    case=case,
                    task_id=task,
                    trials=args.trials,
                    initial_start=args.initial_start,
                    seed=args.seed,
                    max_steps=args.max_steps,
                    video_trials=args.video_trials,
                    references=references,
                    port=args.port,
                    output=str(args.output),
                )
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
                active[name] = subprocess.Popen(
                    [
                        args.libero_python,
                        "-u",
                        "-m",
                        "examples.Libero.eval.run_support_gate",
                        "--config",
                        str(config_path),
                    ],
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
        rows = summarize(args.output)
        if len(seen) != manifest["expected_episodes"]:
            raise ValueError("Completed rollout count differs from plan")
        manifest["status"] = "completed"
        if run is not None:
            import wandb

            for row in rows:
                for key, value in row.items():
                    if isinstance(value, (int, float)):
                        run.summary[f"{row['condition']}/{key}"] = value
            for filename in ("success_modes.png", "support_switches.png"):
                if (args.output / filename).is_file():
                    run.log({filename.removesuffix(".png"): wandb.Image(str(args.output / filename))})
        print(f"COMPLETED {args.output / 'summary.md'}", flush=True)
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
        "--calibration", type=Path, default=Path("outputs/dual_mode/support_calibration/calibration.json")
    )
    parser.add_argument("--memory", type=Path, default=Path("outputs/dual_mode/long_reference_diagnosis"))
    parser.add_argument("--retriever", type=Path, default=Path("ra-vla-official-assets/libero_gos"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("eval", "report"), default="eval")
    parser.add_argument("--task-ids", type=int, nargs="+", default=list(range(10)))
    parser.add_argument(
        "--variants", nargs="+", choices=("action", "q90", "q95"), default=["action", "q90", "q95"]
    )
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--initial-start", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--video-trials", type=int, default=10)
    parser.add_argument("--max-goal-chunks", type=int, default=2)
    parser.add_argument("--cooldown-chunks", type=int, default=1)
    parser.add_argument("--enter-consecutive", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", type=int, default=8097)
    parser.add_argument("--policy-python", default=sys.executable)
    parser.add_argument("--libero-python", default=str(Path.home() / "miniconda3/envs/libero/bin/python"))
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--wandb-entity", default="ryan-nlp")
    parser.add_argument("--wandb-name")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    run(args)


if __name__ == "__main__":
    main()
