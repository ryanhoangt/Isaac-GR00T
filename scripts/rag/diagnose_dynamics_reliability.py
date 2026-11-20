"""Offline reliability diagnosis; never select thresholds on final test rows."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from gr00t.rag.dynamics_reliability import (
    ReliabilityFilter,
    acceptance_summary,
    episode_partition,
    select_threshold,
)
from gr00t.rag.endpoint_data import collect_transitions
from gr00t.rag.local_dynamics import DeltaMLP, DynamicsEnsemble, PolicyCodec
from scripts.rag.train_dual_mode import check_worker
from scripts.rag import train_local_dynamics


REPO = Path(__file__).resolve().parents[2]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def csv_rows(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rank_correlation(x, y):
    # pandas ranks assign average ranks to ties, unlike double argsort.
    import pandas as pd

    result = pd.Series(x).rank().corr(pd.Series(y).rank())
    return float(result) if np.isfinite(result) else None


def plot(results, selected, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    names = ("calibration", "test", "audit")
    for name, ax in zip(names, axes):
        rows = results[name]
        # Diagnostic curves at fixed score quantiles; no test threshold selection.
        for score, label in (("spread_m", "Ensemble disagreement"), ("distance", "Input distance")):
            values = np.asarray([r[score] for r in rows])
            error = np.asarray([r["error_m"] for r in rows])
            order = np.argsort(values, kind="stable")
            counts = np.maximum(1, (np.linspace(0.05, 1, 20) * len(rows)).astype(int))
            ax.plot(counts / len(rows), [np.quantile(error[order[:n]], 0.95) * 100 for n in counts], label=label)
        if selected is not None:
            accepted = np.asarray([r["accepted"] for r in rows])
            if accepted.any():
                ax.scatter(
                    accepted.mean(),
                    np.quantile(np.asarray([r["error_m"] for r in rows])[accepted], 0.95) * 100,
                    marker="*",
                    s=100,
                    color="black",
                    label="Frozen joint filter",
                )
        ax.axhline(1, color="gray", linestyle="--", linewidth=1)
        ax.set(xlabel="Retained fraction of windows", ylabel="Prediction error p95 (cm)", title=name)
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Local dynamics reliability — episode-separated evaluation")
    fig.tight_layout()
    fig.savefig(output / "risk_coverage.png", dpi=180)
    fig.savefig(output / "risk_coverage.svg")
    plt.close(fig)


def run(args, run):
    recipe = json.loads((args.checkpoint / "training_config.json").read_text())
    partition = episode_partition(recipe, args.split_seed)
    write_json(args.output / "episode_split.json", partition)
    criteria = dict(
        max_mean=args.max_mean_m,
        max_p95=args.max_p95_m,
        min_coverage=args.min_coverage,
        min_windows=32,
        min_episodes=2,
    )
    # The fixed recipe/epochs are not tuned using test or audit errors.
    train_local_dynamics.fit(args, partition, args.output / "dynamics", run)
    path = args.output / "dynamics/dynamics.pt"
    saved = torch.load(path, map_location="cpu", weights_only=False)
    dynamics = DynamicsEnsemble(saved["count"], saved["hidden"], saved["spectral"]).cuda().freeze()
    dynamics.load_state_dict(saved["ensemble"])
    codec = PolicyCodec(**saved["codec"]).cuda()
    training, provenance = collect_transitions(partition, "train")
    with torch.no_grad():
        states, actions, _ = (v.cuda() for v in training)
        member = dynamics.members[0]
        reference = (
            DeltaMLP.inputs(codec.state(states), codec.actions(actions)) - member.input_mean
        ) / member.input_scale
    support = ReliabilityFilter(
        reference, member.input_mean, member.input_scale, float("inf"), float("inf")
    ).cuda()
    all_rows, evaluated = {}, {}
    selected, candidates = None, []
    for split in ("calibration", "test", "audit"):
        values, ids = collect_transitions(partition, split)
        with torch.no_grad():
            s, a, y = (v.cuda() for v in values)
            state, action = codec.state(s), codec.actions(a)
            prediction, spread = dynamics(state, action)
            error = (prediction - y).norm(dim=-1).cpu().numpy()
            distance = support.distances(DeltaMLP.inputs(state, action)).cpu().numpy()
            spread = spread.cpu().numpy()
        if split == "calibration":
            selected, candidates = select_threshold(error, spread, distance, ids, **criteria)
            write_json(args.output / "calibration_selection.json", dict(selected=selected, candidates=candidates))
        accepted = (
            np.zeros(len(error), dtype=bool)
            if selected is None
            else ((spread <= selected["std_limit_m"]) & (distance <= selected["distance_limit"]))
        )
        evaluated[split] = dict(
            windows=len(error),
            episodes=len({(d, e) for d, e, _ in ids}),
            unfiltered=train_local_dynamics.statistics(error),
            filtered=acceptance_summary(error, accepted, ids, **criteria),
            spread_error_spearman=rank_correlation(spread, error),
            distance_error_spearman=rank_correlation(distance, error),
        )
        all_rows[split] = [
            dict(
                split=split,
                dataset=d,
                episode=e,
                timestep=t,
                error_m=float(err),
                spread_m=float(std),
                distance=float(dist),
                accepted=bool(keep),
            )
            for (d, e, t), err, std, dist, keep in zip(ids, error, spread, distance, accepted)
        ]
        csv_rows(args.output / f"{split}_windows.csv", all_rows[split])
        print("RELIABILITY_SPLIT", split, json.dumps(evaluated[split]), flush=True)
    # The old validation set has already been reported once; use it only as an
    # additional audit, never to choose a threshold. Both final sets must pass.
    ready = bool(
        selected is not None
        and evaluated["test"]["filtered"]["passed"]
        and evaluated["audit"]["filtered"]["passed"]
    )
    policy = saved["report"]["observed_policy_errors"]
    if max(r["error_m"] for r in policy if r["mode"] == "goal") > args.max_policy_error_m:
        ready = False
    report = dict(
        status="completed",
        ready_for_endpoint=ready,
        training_windows=len(provenance),
        training_episodes=len({(d, e) for d, e, _ in provenance}),
        split_seed=args.split_seed,
        criteria=criteria,
        selected=selected,
        evaluation=evaluated,
        observed_policy_errors=policy,
        caveats=[
            "Empirical filter, not a conformal bound or guarantee on action gradients.",
            "Overlapping windows are correlated; episode counts and per-task results are reported.",
            "Thresholds use calibration only. Test and earlier validation audit never tune thresholds.",
            "The starting policy may have seen these expert demos; this split validates dynamics only.",
        ],
    )
    if selected is not None:
        saved["reliability_filter"] = dict(
            reference=reference.cpu(),
            mean=member.input_mean.cpu(),
            scale=member.input_scale.cpu(),
            std_limit=selected["std_limit_m"],
            distance_limit=selected["distance_limit"],
        )
        saved["std_limit"] = selected["std_limit_m"]
    saved["global_validation_report"] = saved["report"]
    saved["report"] = report
    torch.save(saved, path)
    write_json(args.output / "report.json", report)
    plot(all_rows, selected, args.output)
    status = "reliability_passed" if ready else "reliability_rejected"
    write_json(
        args.output / "status.json", dict(status=status, ready_for_endpoint=ready, policy_training_started=False)
    )
    if run is not None:
        run.log(
            {
                "reliability/ready": int(ready),
                **{
                    f"reliability/{split}_coverage": row["filtered"]["coverage"]
                    for split, row in evaluated.items()
                },
            }
        )
        run.save(str(args.output / "risk_coverage.png"), base_path=str(args.output))
    print("RELIABILITY_RESULT", json.dumps(dict(status=status, selected=selected)), flush=True)


def main(args):
    check_worker()
    for name in ("output", "checkpoint", "policy_reference"):
        setattr(args, name, (REPO / getattr(args, name)).resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    write_json(args.output / "config.json", config)
    run_object = None
    try:
        if args.wandb_mode != "disabled":
            import wandb

            run_object = wandb.init(
                project="icl-vla-uncertainty",
                entity="ryan-nlp",
                group="dynamics_reliability",
                name=args.output.name,
                mode=args.wandb_mode,
                dir=str(args.output),
                config=config,
            )
        run(args, run_object)
    except BaseException as error:
        write_json(args.output / "status.json", dict(status="failed", error=str(error)))
        raise
    finally:
        if run_object is not None:
            run_object.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/dual_mode/debug_20261102/goal_same_demo_seed42_spawn/checkpoint-2000"),
    )
    parser.add_argument(
        "--policy-reference", type=Path, default=Path("outputs/ravla_eval/fixed_goal_prefix64_exec16_20261103")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=20261103)
    parser.add_argument("--dynamics-epochs", type=int, default=100)
    parser.add_argument("--ensemble", type=int, default=3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--spectral", action="store_true")
    parser.add_argument("--max-mean-m", type=float, default=0.005)
    parser.add_argument("--max-p95-m", type=float, default=0.01)
    parser.add_argument("--max-policy-error-m", type=float, default=0.02)
    parser.add_argument("--min-coverage", type=float, default=0.1)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    main(parser.parse_args())
