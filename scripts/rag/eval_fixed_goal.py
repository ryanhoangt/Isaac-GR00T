"""Launch RGB/proprio-only fixed-goal tests inside a Slurm GPU allocation."""

import argparse
from collections import deque
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

import numpy as np

from gr00t.eval.service import BaseInferenceClient
from gr00t.rag.diagnosis_protocol import digest
from scripts.rag.diagnose_dual_mode import checkpoint_signature, environment, file_sha
from scripts.rag.eval_libero_parallel import atomic_json, stop_processes


REPO = Path(__file__).resolve().parents[2]


def report(output, expected, final=False):
    rows = [json.loads(path.read_text()) for path in sorted(output.glob("*/*/episode.json"))]
    summary = dict(
        completed_cases=len(rows),
        expected_cases=expected,
        replay_matches=sum(row["replay_matched"] for row in rows),
        note="Short fixed-goal RGB/proprio diagnostics; no semantic success label or recovery claim.",
    )
    metrics = (
        "end_ee_distance_m",
        "reduction_ee_distance_m",
        "end_orientation_error_deg",
        "end_gripper_distance_m",
        "end_workspace_mae",
        "end_wrist_mae",
    )
    flat = []
    for row in rows:
        flat.append(
            {k: row[k] for k in ("suite", "initial_index", "mode", "prefix", "fixed_goal_frame", "execute_steps")}
            | row["metrics"]
            | row["action_agreement"]
        )
    for mode in ("action", "goal"):
        selected = [row for row in rows if row["mode"] == mode]
        summary[mode + "_cases"] = len(selected)
        for metric in metrics:
            summary[mode + "_mean_" + metric] = (
                float(np.mean([r["metrics"][metric] for r in selected])) if selected else None
            )
    atomic_json(output / "summary.json", summary)
    if flat:
        with (output / "episodes.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)
    if not final:
        return summary
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = {}
    for row in rows:
        groups.setdefault((row["suite"], row["initial_index"]), {})[row["mode"]] = row
    figure, axes = plt.subplots(len(groups), 2, figsize=(12, 2.5 * len(groups)), squeeze=False)
    lines = [
        "# Fixed-goal short-window results",
        "",
        "RGB and robot proprioception only. Benchmark initial states are used only through standard reset; no simulator snapshots or object-state queries.",
        "",
        "| Reference | Initial EE-to-goal (cm) | Mode 1 end (cm) | Goal head end (cm) | Goal-head RGB MAE |",
        "|---|---:|---:|---:|---:|",
    ]
    pairs = []
    for index, ((suite, initial), pair) in enumerate(sorted(groups.items())):
        if set(pair) != {"action", "goal"}:
            raise ValueError("Incomplete mode pair")
        action, goal = pair["action"], pair["goal"]
        if action["start_observation_identity"] != goal["start_observation_identity"]:
            raise ValueError("Mode pair started from different observed states")
        if action["goal_observation_identity"] != goal["goal_observation_identity"]:
            raise ValueError("Mode pair targets differ")
        if any(q["fixed_goal_frame"] != goal["fixed_goal_frame"] for q in goal["queries"]):
            raise ValueError("Goal changed during the short window")
        am, gm = action["metrics"], goal["metrics"]
        lines.append(
            f"| {suite} / init {initial} | {gm['start_ee_distance_m'] * 100:.3f} | {am['end_ee_distance_m'] * 100:.3f} | {gm['end_ee_distance_m'] * 100:.3f} | {gm['end_workspace_mae']:.3f} |"
        )
        pairs.append(
            dict(
                suite=suite,
                initial_index=initial,
                action_end_ee_cm=100 * am["end_ee_distance_m"],
                goal_end_ee_cm=100 * gm["end_ee_distance_m"],
                goal_ee_progress_cm=100 * gm["reduction_ee_distance_m"],
                action_reference_max_abs=action["action_agreement"]["reference_action_max_abs"],
            )
        )
        for mode, color in (("action", "tab:blue"), ("goal", "tab:orange")):
            row = pair[mode]
            curve = row["target_error_curve"]
            axes[index, 0].plot([100 * r["ee_distance_m"] for r in curve], label=mode, color=color)
            axes[index, 1].plot([r["workspace_mae"] for r in curve], label=mode, color=color)
        for col, label in ((0, "EE distance to fixed goal (cm)"), (1, "Workspace RGB MAE (0–255)")):
            axes[index, col].set(title=f"{suite} / init {initial}", xlabel="Local control step", ylabel=label)
            axes[index, col].grid(alpha=0.2)
        axes[index, 0].legend()
    figure.suptitle("Fixed-goal local control — identical replayed prefixes, no online phase selection")
    figure.tight_layout()
    figure.savefig(output / "fixed_goal_traces.png", dpi=150)
    plt.close(figure)
    with (output / "paired.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    atomic_json(
        output / "audit.json",
        dict(
            status="passed",
            paired_cases=len(groups),
            replay_frames_checked=sum(r["replay_frames_checked"] for r in rows),
            max_replay_rgb_error=max(r["max_replay_rgb_error"] for r in rows),
            max_replay_proprio_error=max(r["max_replay_proprio_error"] for r in rows),
            max_mode1_reference_action_error=max(
                r["action_agreement"]["reference_action_max_abs"] for r in rows if r["mode"] == "action"
            ),
            fixed_goal=True,
            observed_starts_match=True,
            robot_observations_only=True,
        ),
    )
    lines += [
        "",
        "EE proximity and RGB similarity are proxies, not semantic goal success; inspect grasp/contact in the videos.",
        "References are successful Mode-1 rollout trajectories, not newly collected expert training labels.",
        "This tests trained GOS tasks and short windows; it is not held-out Long SR or perturbation recovery.",
        "",
        "![Fixed-goal traces](fixed_goal_traces.png)",
        "",
    ]
    (output / "RESULTS.md").write_text("\n".join(lines))
    return summary


def execute(args, manifest, recipes):
    output = args.output
    env = environment(output)
    server, active, handles = None, {}, []
    run = None

    def interrupted(*_):
        raise KeyboardInterrupt

    handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        if args.wandb_mode != "disabled":
            import wandb

            run = wandb.init(
                project="icl-vla-uncertainty",
                entity="ryan-nlp",
                name=output.name,
                group="isolated_goal_fixed_window",
                mode=args.wandb_mode,
                dir=str(output),
                config=manifest["settings"],
            )
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", args.port))
        identity = uuid.uuid4().hex
        handle = (output / "server.log").open("a")
        handles.append(handle)
        server = subprocess.Popen(
            [
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
            ],
            cwd=REPO,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        client = BaseInferenceClient(host="127.0.0.1", port=args.port, timeout_ms=1000)
        try:
            deadline = time.monotonic() + 600
            while not client.ping():
                if server.poll() is not None:
                    raise RuntimeError("Policy server exited; see server.log")
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
        pending = deque(recipes)
        previous = -1
        while pending or active:
            if server.poll() is not None:
                raise RuntimeError("Policy server died")
            while pending and len(active) < args.workers:
                recipe = pending.popleft()
                name = Path(recipe["reference"]).name
                config_path = output / "configs" / (name + ".json")
                config_path.parent.mkdir(exist_ok=True)
                atomic_json(config_path, recipe)
                handle = (output / (name + ".log")).open("a")
                handles.append(handle)
                task_env = env.copy()
                task_env["NUMBA_CACHE_DIR"] = str(output / "cache" / name)
                active[name] = subprocess.Popen(
                    [
                        args.libero_python,
                        "-u",
                        "-m",
                        "examples.Libero.eval.run_fixed_goal",
                        "--config",
                        str(config_path),
                    ],
                    cwd=REPO,
                    env=task_env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                print("START", name, flush=True)
            for name, process in list(active.items()):
                if process.poll() is not None:
                    if process.returncode:
                        raise RuntimeError(f"{name} failed; see {name}.log")
                    del active[name]
            summary = report(output, manifest["expected_cases"])
            atomic_json(output / "progress.json", dict(status="running", **summary))
            if previous != summary["completed_cases"]:
                print("PROGRESS", json.dumps(summary), flush=True)
                if run is not None:
                    run.log({k: v for k, v in summary.items() if isinstance(v, (int, float))})
                previous = summary["completed_cases"]
            time.sleep(2)
        summary = report(output, manifest["expected_cases"], final=True)
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
        stop_processes([*active.values(), *([server] if server is not None else [])])
        atomic_json(output / "manifest.json", manifest)
        for handle in handles:
            handle.close()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        if run is not None:
            run.finish()


def main(args):
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_STEP_ID"):
        raise RuntimeError("Run with srun on a GPU worker")
    for key in ("output", "reference_root", "checkpoint", "baseline", "retriever"):
        setattr(args, key, (REPO / getattr(args, key)).resolve())
    if args.prefix < 0 or args.prefix % 16 or args.horizon != 16 or args.execute_steps not in (1, 4, 8, 16):
        raise ValueError("Use a chunk-aligned prefix, horizon 16, execute steps 1/4/8/16")
    if args.workers < 1 or args.limit_references < 0 or min(args.rgb_tolerance, args.proprio_tolerance) < 0:
        raise ValueError("Invalid worker/count/tolerances")
    source = json.loads((args.reference_root / "manifest.json").read_text())
    if source["status"] != "completed":
        raise ValueError("Reference run is incomplete")
    seed = source["settings"]["seed"]
    references = []
    for path in sorted((args.reference_root / "action_reference").glob("*/episode.json")):
        row = json.loads(path.read_text())
        if row.get("complete") and row.get("success") and not row.get("smoke_only"):
            if row["control_steps"] < args.prefix + args.horizon:
                raise ValueError("Successful reference is too short")
            references.append((path, row))
    if args.limit_references:
        references = references[: args.limit_references]
    if not references:
        raise ValueError("No eligible successful references")
    settings = dict(
        reference_root=str(args.reference_root),
        checkpoint=str(args.checkpoint),
        baseline=str(args.baseline),
        retriever=str(args.retriever),
        prefix=args.prefix,
        horizon=args.horizon,
        execute_steps=args.execute_steps,
        seed=seed,
        rgb_tolerance=args.rgb_tolerance,
        proprio_tolerance=args.proprio_tolerance,
        references=[str(p.parent) for p, _ in references],
        modes=["action", "goal"],
        selection="fixed reference t+16; no online selector",
        inputs="RGB, robot proprioception, recorded actions",
        reset="standard benchmark initial-state API only; no snapshot/object-state access",
        metrics="RGB/proprio proxies and video; no automatic semantic goal-success label",
    )
    sources = [
        "gr00t/rag/fixed_goal_protocol.py",
        "examples/Libero/eval/run_fixed_goal.py",
        "scripts/rag/eval_fixed_goal.py",
        "gr00t/rag/oracle_goal_policy.py",
        "scripts/rag/serve_goal_oracle.py",
        "gr00t/rag/goal_control.py",
        "gr00t/rag/dual_mode.py",
    ]
    memory_stat = (args.retriever / "memory_gos.pt").stat()
    inputs = dict(
        checkpoint=checkpoint_signature(args.checkpoint),
        baseline=checkpoint_signature(args.baseline),
        retriever=file_sha(args.retriever / "model.safetensors"),
        memory_stat=dict(size=memory_stat.st_size, mtime_ns=memory_stat.st_mtime_ns),
        source={name: file_sha(REPO / name) for name in sources},
        references={
            str(p.parent): dict(episode=file_sha(p), trace=file_sha(p.parent / "trace.npz")) for p, _ in references
        },
    )
    signature = digest(dict(settings=settings, inputs=inputs))
    manifest = dict(
        signature=signature,
        settings=settings,
        inputs=inputs,
        expected_cases=len(references) * 2,
        status="starting",
    )
    recipes = [
        dict(
            signature=signature,
            reference=str(p.parent),
            suite=r["suite"],
            task_id=r["task_id"],
            task=r["task"],
            initial_index=r["initial_index"],
            prefix=args.prefix,
            horizon=args.horizon,
            execute_steps=args.execute_steps,
            rgb_tolerance=args.rgb_tolerance,
            proprio_tolerance=args.proprio_tolerance,
            seed=seed,
            modes=["action", "goal"],
            port=args.port,
            output=str(args.output),
        )
        for p, r in references
    ]
    if args.output.exists():
        if not args.resume:
            raise FileExistsError("Output exists; use --resume with the same protocol")
        old = json.loads((args.output / "manifest.json").read_text())
        if old["signature"] != signature:
            raise ValueError("Resume protocol/source/checkpoint differs")
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
        "--reference-root", type=Path, default=Path("outputs/ravla_eval/goal_oracle_gos_3trials_20261103")
    )
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
    parser.add_argument("--prefix", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--execute-steps", type=int, default=16)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit-references", type=int, default=0)
    parser.add_argument("--rgb-tolerance", type=float, default=0.0)
    parser.add_argument("--proprio-tolerance", type=float, default=1e-6)
    parser.add_argument("--port", type=int, default=8116)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--resume", action="store_true")
    main(parser.parse_args())
