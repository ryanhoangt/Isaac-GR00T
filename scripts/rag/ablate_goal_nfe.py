"""Paired offline 4/8/16-step goal-mode probe. This does not measure rollout SR."""

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature

from gr00t.rag.dual_mode_data import pack_context
from gr00t.rag.dual_mode_memory import milestone_index
from scripts.rag.serve_dual_mode import Args, load_policy


@torch.inference_mode()
def predict(policy, task, current, goal, noise, nfe):
    head = policy.vla.action_head
    entry = policy.memory[task]
    context, actions = pack_context(
        [(entry["backbone_features"][goal], entry["backbone_attention_mask"][goal], entry["action"][goal])],
        head.device,
        head.dtype,
    )
    visual = BatchFeature(
        data={
            "backbone_features": entry["backbone_features"][current : current + 1].to(
                head.device, dtype=head.dtype
            ),
            "backbone_attention_mask": entry["backbone_attention_mask"][current : current + 1].to(head.device),
        }
    )
    inputs = BatchFeature(
        data={
            "state": entry["state"][current : current + 1].to(head.device, dtype=head.dtype),
            "embodiment_id": entry["embodiment_id"][current : current + 1].to(head.device),
            "ret_actions": actions,
            "ret_backbone_output": context,
            "context_mode": torch.ones(1, device=head.device),
            "flow_source": noise,
        }
    )
    head.num_inference_timesteps = nfe
    torch.cuda.synchronize()
    start = time.perf_counter()
    output = head.get_action(visual, inputs).action_pred[0, :, :7].float().cpu().numpy()
    torch.cuda.synchronize()
    return output, (time.perf_counter() - start) * 1000


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    policy = load_policy(Args(checkpoint=args.checkpoint, memory=args.memory, retriever=args.retriever))
    rng = np.random.default_rng(args.seed)
    run = None
    if not args.no_wandb:
        import wandb

        run = wandb.init(
            project="icl-vla-uncertainty",
            entity="ryan-nlp",
            name=args.output.name,
            group="dual_mode_debug_nfe",
            dir=str(args.output.resolve()),
            config=vars(args),
        )
    rows, predictions = [], []
    try:
        tasks = list(policy.memory.manifest["tasks"])
        for ti, task in enumerate(tasks):
            entry = policy.memory[task]
            eligible = torch.where(entry["frame_index"] + 16 <= entry["trajectory_length"])[0].numpy()
            count = args.states // len(tasks) + int(ti < args.states % len(tasks))
            selected = rng.permutation(eligible)[:count]
            for current in selected:
                current = int(current)
                own = int(entry["episode_id"][current])
                scores = entry["keys"][current] @ entry["keys"].T
                other = entry["episode_id"] != own
                anchor = int(scores.masked_fill(~other, -torch.inf).argmax())
                goals = dict(
                    same_demo_future=milestone_index(entry, current, 16),
                    cross_demo_retrieved=milestone_index(entry, anchor, 16),
                )
                candidates = torch.where(other & (torch.arange(len(scores)) != goals["cross_demo_retrieved"]))[
                    0
                ].numpy()
                goals["random_same_task"] = int(rng.choice(candidates))
                state_seed = args.seed + len(predictions)
                generator = torch.Generator(device=policy.vla.device).manual_seed(state_seed)
                noise = torch.randn(
                    (1, 16, policy.vla.action_head.action_dim),
                    generator=generator,
                    device=policy.vla.device,
                    dtype=policy.vla.action_head.dtype,
                )
                target = entry["action"][current, :, :7].float().numpy()
                sample = []
                for condition, goal in goals.items():
                    outputs = {}
                    for nfe in (4, 8, 16):
                        output, latency = predict(policy, task, current, goal, noise, nfe)
                        outputs[nfe] = output
                        delta = output - target
                        row = dict(
                            task=task,
                            query_episode=own,
                            query_frame=int(entry["frame_index"][current]),
                            goal_episode=int(entry["episode_id"][goal]),
                            goal_frame=int(entry["frame_index"][goal]),
                            condition=condition,
                            nfe=nfe,
                            flow_seed=state_seed,
                            normalized_mse7=float(np.mean(delta**2)),
                            xyz_mse=float(np.mean(delta[:, :3] ** 2)),
                            rotation_mse=float(np.mean(delta[:, 3:6] ** 2)),
                            gripper_mse=float(np.mean(delta[:, 6:] ** 2)),
                            shared_gpu_latency_ms=latency,
                        )
                        sample.append((row, output))
                    denominator = max(float(np.linalg.norm(outputs[16])), 1e-6)
                    for row, output in sample[-3:]:
                        row["relative_to_nfe16"] = float(np.linalg.norm(output - outputs[16]) / denominator)
                for row, _ in sample:
                    rows.append(row)
                predictions.append([output for _, output in sample])
                if len(predictions) % 10 == 0:
                    print(f"NFE_STATES {len(predictions)}/{args.states}", flush=True)
            print("NFE_TASK_COMPLETE", task, len(selected), flush=True)
        with (args.output / "probes.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        np.savez_compressed(args.output / "predictions.npz", actions=np.asarray(predictions))
        summary = []
        for condition in ("same_demo_future", "cross_demo_retrieved", "random_same_task"):
            for nfe in (4, 8, 16):
                selected = [r for r in rows if r["condition"] == condition and r["nfe"] == nfe]
                item = dict(condition=condition, nfe=nfe, states=len(selected))
                for key in (
                    "normalized_mse7",
                    "xyz_mse",
                    "rotation_mse",
                    "gripper_mse",
                    "relative_to_nfe16",
                    "shared_gpu_latency_ms",
                ):
                    item[key] = float(np.mean([r[key] for r in selected]))
                summary.append(item)
                if run is not None:
                    run.log(
                        {f"{condition}/{k}": v for k, v in item.items() if k not in ("condition", "nfe")}
                        | {"nfe": nfe}
                    )
        report = dict(
            status="completed",
            states=len(predictions),
            checkpoint=str(args.checkpoint),
            scope="offline expert states; not rollout success",
            same_noise_across_nfe_and_goals=True,
            terminal_chunks_excluded=True,
            latency_is_contended=True,
            results=summary,
        )
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--memory", type=Path, default=Path("outputs/dual_mode/long_reference_diagnosis"))
    parser.add_argument("--retriever", type=Path, default=Path("ra-vla-official-assets/libero_gos"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--states", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    if args.states < 1:
        parser.error("states must be positive")
    main(args)
