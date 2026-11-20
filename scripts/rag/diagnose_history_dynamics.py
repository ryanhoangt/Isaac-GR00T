"""Matched 0/4/8-step history ablation on offline observable robot data."""

import argparse
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
from gr00t.rag.history_dynamics import HistoryEnsemble, collect_history, history_inputs
from gr00t.rag.local_dynamics import PolicyCodec
from scripts.rag.diagnose_dynamics_reliability import csv_rows, rank_correlation, write_json
from scripts.rag.train_dual_mode import check_worker
from scripts.rag.train_local_dynamics import quaternion_axis_angle, statistics


REPO = Path(__file__).resolve().parents[2]


def fit(inputs, target, steps, args, run):
    torch.manual_seed(args.seed)
    model = HistoryEnsemble(steps, count=3, hidden=128).cuda()
    for number, member in enumerate(model.members):
        member.fit_normalization(inputs, target)
        optimizer = torch.optim.Adam(member.parameters(), lr=1e-3, weight_decay=1e-5)
        generator = torch.Generator(device=inputs.device).manual_seed(args.seed + number)
        for epoch in range(100):
            order = torch.randperm(len(inputs), generator=generator, device=inputs.device)
            total = 0.0
            for indices in order.split(512):
                prediction = member(inputs[indices])
                loss = ((prediction - target[indices]) / member.output_scale).square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite history dynamics loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(indices)
            if epoch == 0 or (epoch + 1) % 20 == 0:
                row = dict(history=steps, member=number, epoch=epoch + 1, loss=total / len(inputs))
                print("HISTORY_FIT", json.dumps(row), flush=True)
                if run is not None:
                    run.log({f"history{steps}/member{number}_loss": row["loss"], "fit_epoch": epoch + 1})
    return model.freeze()


def trace_proprio(trace, indices):
    # Explicit allowlist: never load object poses, simulator snapshots or RGB.
    return torch.cat(
        (
            torch.as_tensor(trace["ee"][indices]),
            quaternion_axis_angle(trace["quat"][indices]),
            torch.as_tensor(trace["gripper"][indices]),
        ),
        -1,
    ).float()


def demo_action_from_deployed(values):
    action = torch.as_tensor(values, dtype=torch.float32).clone()
    action[..., 6] = (1 - action[..., 6]) / 2
    return action


@torch.no_grad()
def policy_errors(model, codec, support, selected, steps, root):
    rows = []
    for path in sorted(root.glob("*/*/trace.npz")):
        episode = json.loads((path.parent / "episode.json").read_text())
        if episode["execute_steps"] != 16:
            continue
        prefix = episode["prefix"]
        if prefix < 8 or not episode["replay_matched"]:
            raise ValueError("Policy history requires an audited prefix of >=8 steps")
        with (
            np.load(path, allow_pickle=False) as trace,
            np.load(Path(episode["reference"]) / "trace.npz", allow_pickle=False) as reference,
        ):
            # The recorded prefix was already replay-checked at every frame.
            for key in ("ee", "quat", "gripper"):
                np.testing.assert_allclose(trace[key][0], reference[key][prefix], rtol=0, atol=1e-6)
            current = trace_proprio(trace, 0)[None].cuda()
            past = trace_proprio(reference, slice(prefix - 8, prefix))[None].cuda()
            previous = demo_action_from_deployed(reference["actions"][prefix - 8 : prefix])[None].cuda()
            action = demo_action_from_deployed(trace["actions"])[None].cuda()
            target = torch.as_tensor(trace["ee"][-1] - trace["ee"][0], dtype=torch.float32, device="cuda")[None]
        inputs = history_inputs(codec, current, action, past, previous, steps)
        prediction, spread = model(inputs)
        distance = support.distances(inputs)
        accepted = bool(
            selected and spread.item() <= selected["std_limit_m"] and distance.item() <= selected["distance_limit"]
        )
        rows.append(
            dict(
                case=path.parent.name,
                mode=episode["mode"],
                error_m=float((prediction - target).norm()),
                spread_m=float(spread),
                distance=float(distance),
                accepted=accepted,
            )
        )
    if sum(r["mode"] == "goal" for r in rows) != 6 or len(rows) != 12:
        raise ValueError("Expected six action and six goal policy windows")
    return rows


def compare_plot(reports, rows_by_variant, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    names = [f"{h} past steps" for h in reports]
    x = np.arange(len(names))
    for offset, metric, label in ((-0.18, "mean_m", "Mean"), (0.18, "p95_m", "p95")):
        axes[0].bar(
            x + offset,
            [100 * r["evaluation"]["test"]["unfiltered"][metric] for r in reports.values()],
            width=0.35,
            label=label,
        )
    axes[0].set_xticks(x, names)
    axes[0].set(title="Unfiltered test error", ylabel="EE prediction error (cm)")
    axes[0].legend()
    for split, ax in zip(("calibration", "test"), axes[1:]):
        for steps in reports:
            rows = rows_by_variant[steps][split]
            error = np.asarray([r["error_m"] for r in rows])
            order = np.argsort([r["spread_m"] for r in rows], kind="stable")
            counts = np.maximum(1, (np.linspace(0.05, 1, 20) * len(rows)).astype(int))
            ax.plot(
                counts / len(rows),
                [100 * np.quantile(error[order[:n]], 0.95) for n in counts],
                label=f"history {steps}",
            )
        ax.axhline(1, color="gray", linestyle="--")
        ax.set(title=f"{split}: ensemble-spread ranking", xlabel="Retained fraction", ylabel="Error p95 (cm)")
        ax.legend()
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Causal robot-history dynamics — matched windows, 100 epochs each")
    fig.tight_layout()
    fig.savefig(output / "history_comparison.png", dpi=180)
    fig.savefig(output / "history_comparison.svg")
    plt.close(fig)


def execute(args, run):
    recipe = json.loads((args.checkpoint / "training_config.json").read_text())
    partition = episode_partition(recipe)
    write_json(args.output / "episode_split.json", partition)
    metadata = torch.load(args.checkpoint / "metadata.pt", map_location="cpu", weights_only=False)
    codec = PolicyCodec.from_metadata(metadata).cuda()
    data, ids = {}, {}
    for split in ("train", "calibration", "test", "audit"):
        values, ids[split] = collect_history(partition, split, context=8)
        data[split] = tuple(v.cuda() for v in values)
    group_ids = [{(d, e) for d, e, _ in ids[split]} for split in ids]
    if any(group_ids[i] & group_ids[j] for i in range(4) for j in range(i)):
        raise ValueError("Episode leakage")
    criteria = dict(max_mean=0.005, max_p95=0.01, min_coverage=0.1, min_windows=32, min_episodes=2)
    reports, all_rows, table = {}, {}, []
    for steps in (0, 4, 8):
        output = args.output / f"history{steps}"
        output.mkdir()
        inputs = {
            split: history_inputs(codec, values[0], values[1], values[3], values[4], steps)
            for split, values in data.items()
        }
        model = fit(inputs["train"], data["train"][2], steps, args, run)
        member = model.members[0]
        support = ReliabilityFilter(
            (inputs["train"] - member.input_mean) / member.input_scale,
            member.input_mean,
            member.input_scale,
            float("inf"),
            float("inf"),
        ).cuda()
        evaluated, rows, selected = {}, {}, None
        for split in ("calibration", "test", "audit"):
            with torch.no_grad():
                prediction, spread = model(inputs[split])
                error = (prediction - data[split][2]).norm(dim=-1).cpu().numpy()
                distance = support.distances(inputs[split]).cpu().numpy()
                spread = spread.cpu().numpy()
            if split == "calibration":
                selected, candidates = select_threshold(error, spread, distance, ids[split], **criteria)
                write_json(output / "calibration_selection.json", dict(selected=selected, candidates=candidates))
            accepted = (
                np.zeros(len(error), dtype=bool)
                if selected is None
                else ((spread <= selected["std_limit_m"]) & (distance <= selected["distance_limit"]))
            )
            per_task = [
                dict(dataset=d, **statistics(error[np.asarray([p[0] == d for p in ids[split]])]))
                for d in sorted({p[0] for p in ids[split]})
            ]
            evaluated[split] = dict(
                windows=len(error),
                unfiltered=statistics(error),
                per_task_unfiltered=per_task,
                filtered=acceptance_summary(error, accepted, ids[split], **criteria),
                spread_error_spearman=rank_correlation(spread, error),
                distance_error_spearman=rank_correlation(distance, error),
            )
            rows[split] = [
                dict(
                    dataset=d,
                    episode=e,
                    timestep=t,
                    error_m=float(err),
                    spread_m=float(std),
                    distance=float(dist),
                    accepted=bool(keep),
                )
                for (d, e, t), err, std, dist, keep in zip(ids[split], error, spread, distance, accepted)
            ]
            csv_rows(output / f"{split}_windows.csv", rows[split])
        observed = policy_errors(model, codec, support, selected, steps, args.policy_reference)
        ready = bool(
            selected is not None
            and evaluated["test"]["filtered"]["passed"]
            and evaluated["audit"]["filtered"]["passed"]
            and max(r["error_m"] for r in observed if r["mode"] == "goal") <= 0.02
        )
        report = dict(
            history_steps=steps,
            input_dim=120 + 15 * steps,
            ready_for_endpoint=ready,
            selected=selected,
            evaluation=evaluated,
            observed_policy_errors=observed,
            goal_policy_prediction_error=statistics([r["error_m"] for r in observed if r["mode"] == "goal"]),
            fit_epochs=100,
            hidden=[128, 128],
            ensemble=3,
            criteria=criteria,
        )
        torch.save(
            dict(
                architecture=dict(history_steps=steps, count=3, hidden=128),
                model=model.cpu().state_dict(),
                codec=codec.cpu().state_dict(),
                reference=support.reference.cpu(),
                report=report,
                partition=partition,
            ),
            output / "dynamics.pt",
        )
        codec.cuda()
        write_json(output / "report.json", report)
        reports[steps], all_rows[steps] = report, rows
        for split, metrics in evaluated.items():
            table.append(
                dict(
                    history_steps=steps,
                    split=split,
                    **metrics["unfiltered"],
                    retained_coverage=metrics["filtered"]["coverage"],
                    filtered_pass=metrics["filtered"]["passed"],
                )
            )
        print("HISTORY_RESULT", json.dumps(report), flush=True)
        if run is not None:
            run.log(
                {
                    f"history{steps}/test_mean_m": evaluated["test"]["unfiltered"]["mean_m"],
                    f"history{steps}/test_p95_m": evaluated["test"]["unfiltered"]["p95_m"],
                    f"history{steps}/ready": int(ready),
                }
            )
        del model, inputs, support
        torch.cuda.empty_cache()
    csv_rows(args.output / "comparison.csv", table)
    compare_plot(reports, all_rows, args.output)
    result = dict(
        status="completed",
        policy_training_started=False,
        variants=reports,
        split_counts={
            split: dict(windows=len(rows), episodes=len({(d, e) for d, e, _ in rows}))
            for split, rows in ids.items()
        },
        caveats=[
            "This is a dynamics diagnosis, not a policy recovery experiment.",
            "Identical t>=8 windows for all variants; no history padding or episode crossing.",
            "This test split was inspected in the preceding study; results are development evidence.",
            "Variant/grid definitions fixed before this run; filter thresholds use calibration only.",
            "Passing prediction-error gates does not validate counterfactual action gradients.",
        ],
    )
    write_json(args.output / "report.json", result)
    write_json(
        args.output / "status.json",
        dict(
            status="completed",
            policy_training_started=False,
            passing_history_steps=[h for h, r in reports.items() if r["ready_for_endpoint"]],
        ),
    )
    if run is not None:
        run.save(str(args.output / "history_comparison.png"), base_path=str(args.output))


def main(args):
    check_worker()
    for name in ("output", "checkpoint", "policy_reference"):
        setattr(args, name, (REPO / getattr(args, name)).resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "config.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    )
    run = None
    try:
        if args.wandb_mode != "disabled":
            import wandb

            run = wandb.init(
                project="icl-vla-uncertainty",
                entity="ryan-nlp",
                group="history_dynamics",
                name=args.output.name,
                mode=args.wandb_mode,
                dir=str(args.output),
            )
        execute(args, run)
    except BaseException as error:
        write_json(args.output / "status.json", dict(status="failed", error=str(error)))
        raise
    finally:
        if run is not None:
            run.finish()


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
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    main(parser.parse_args())
