"""Quality-gated offline dynamics + matched goal-head endpoint fine-tuning."""

import argparse
import gc
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers.feature_extraction_utils import BatchFeature

from gr00t.rag.endpoint_data import EndpointCache, build_cache, collate_cached
from gr00t.rag.goal_control import GoalControlRAVLA, assert_warm_owner, owner_digest
from gr00t.rag.local_dynamics import differentiable_goal_action, endpoint_loss
from gr00t.rag.utils import seed_everything
from scripts.rag.train_dual_mode import load_data, check_worker
from scripts.rag import train_local_dynamics


REPO = Path(__file__).resolve().parents[2]


def move_cache(batch, device, dtype):
    def move(value):
        return value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)

    visual = BatchFeature(data={key: move(value) for key, value in batch["current"].items()})
    context = BatchFeature(data={key: move(value) for key, value in batch["goal"].items()})
    data = BatchFeature(
        data={key: move(batch[key]) for key in ("state", "embodiment_id", "action", "action_mask")}
    )
    data.update(
        ret_actions=torch.zeros_like(data.action),
        ret_backbone_output=context,
        context_mode=torch.ones(len(data.state), device=device, dtype=torch.long),
    )
    return visual, context, data


def owner_probe(model, dataset):
    batch = collate_cached([dataset[0]])
    visual, context, data = move_cache(batch, model.device, model.action_head.dtype)
    data.context_mode.zero_()
    data.ret_actions = data.action.clone()
    data.ret_backbone_output = visual
    with torch.no_grad():
        return model.action_head.get_action(visual, data).action_pred.cpu()


def validate(model, dataset, dynamics, codec, std_limit, limit=180):
    model.eval()
    indices = np.unique(np.linspace(0, len(dataset) - 1, min(limit, len(dataset))).astype(int))
    results = []
    for start in range(0, len(indices), 4):
        batch = collate_cached([dataset[i] for i in indices[start : start + 4]])
        visual, context, data = move_cache(batch, model.device, model.action_head.dtype)
        gen = torch.Generator(device=model.device).manual_seed(9000 + start)
        noise = torch.randn(data.action.shape, generator=gen, device=model.device, dtype=data.action.dtype)
        with torch.no_grad():
            prediction = differentiable_goal_action(
                model.action_head.goal_branch, visual, data.state, data.embodiment_id, context, noise
            )
            physical, spread = dynamics(
                codec.state(batch["raw_state"].to(model.device)), codec.actions(prediction, normalized=True)
            )
            mse = ((prediction - data.action).float().square() * data.action_mask).sum(
                (1, 2)
            ) / data.action_mask.sum((1, 2)).clamp_min(1)
            error = (physical - batch["delta"].to(model.device)).norm(dim=-1)
            for m, e, s in zip(mse, error, spread):
                results.append(
                    dict(action_mse=float(m), predicted_endpoint_error_m=float(e), dynamics_std_m=float(s))
                )
    model.train()
    return dict(
        states=len(results),
        action_mse=float(np.mean([r["action_mse"] for r in results])),
        predicted_endpoint_error_m=float(np.mean([r["predicted_endpoint_error_m"] for r in results])),
        note="Dynamics-predicted endpoint error is not a measured physical rollout metric.",
        rows=results,
    )


def train_variant(args, cache, weight, output, run):
    seed_everything(args.seed)
    model = GoalControlRAVLA.from_pretrained(
        str(args.checkpoint), torch_dtype=torch.bfloat16, device_map="cuda:0", tune_visual=False, tune_llm=False
    ).train()
    # Already trained isolated goal checkpoint: do not reinitialize its branch.
    assert_warm_owner(model.action_head, args.baseline)
    model.action_head.num_inference_timesteps = 4
    before = owner_digest(model.action_head)
    train = EndpointCache(cache, "train")
    validation = EndpointCache(cache, "validation")
    dynamics, codec, std_limit = train_local_dynamics.load(args.output / "dynamics/dynamics.pt")
    probe = owner_probe(model, validation)
    params = [p for p in model.parameters() if p.requires_grad]
    if any(
        p.requires_grad for name, p in model.action_head.named_parameters() if not name.startswith("goal_branch.")
    ):
        raise ValueError("Mode-1 owner is not frozen")
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.updates)
    output.mkdir(parents=True, exist_ok=False)
    recipe = json.loads((args.checkpoint / "training_config.json").read_text())
    recipe.update(
        endpoint_lambda=weight,
        starting_checkpoint=str(args.checkpoint),
        dynamics=str(args.output / "dynamics/dynamics.pt"),
        updates=args.updates,
        effective_batch=args.batch_size * args.gradient_accumulation,
        same_demo_only=True,
        endpoint_horizon=16,
        frozen_visual_cache=str(cache),
    )
    (output / "training_config.json").write_text(json.dumps(recipe, indent=2) + "\n")
    report = validate(model, validation, dynamics, codec, std_limit)
    (output / "validation_00000.json").write_text(json.dumps(report, indent=2) + "\n")
    step, epoch = 0, 0
    accumulated = []
    name = "control" if weight == 0 else "endpoint"
    with (output / "training.jsonl").open("w") as log:
        while step < args.updates:
            generator = torch.Generator().manual_seed(args.seed + epoch)
            loader = DataLoader(
                train,
                batch_size=args.batch_size,
                shuffle=True,
                generator=generator,
                num_workers=0,
                collate_fn=collate_cached,
                drop_last=True,
            )
            optimizer.zero_grad(set_to_none=True)
            for batch_index, batch in enumerate(loader):
                visual, context, data = move_cache(batch, model.device, model.action_head.dtype)
                fm = model.action_head.goal_branch(visual, data).loss
                gen = torch.Generator(device=model.device).manual_seed(args.seed + epoch * 100000 + batch_index)
                noise = torch.randn(data.action.shape, generator=gen, device=model.device, dtype=data.action.dtype)
                # Both variants use common data/order/flow samples; the endpoint
                # control is measured without retaining an extra graph.
                with torch.set_grad_enabled(weight > 0):
                    actions = differentiable_goal_action(
                        model.action_head.goal_branch, visual, data.state, data.embodiment_id, context, noise
                    )
                    end, metrics = endpoint_loss(
                        dynamics,
                        codec,
                        batch["raw_state"].to(model.device),
                        actions,
                        batch["delta"].to(model.device),
                        std_limit,
                    )
                loss = fm + weight * end
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite endpoint fine-tuning loss")
                start = batch_index // args.gradient_accumulation * args.gradient_accumulation
                window = min(args.gradient_accumulation, len(loader) - start)
                (loss / window).backward()
                accumulated.append(dict(fm=float(fm.detach()), endpoint=float(end.detach()), **metrics))
                if (batch_index + 1) % args.gradient_accumulation and batch_index + 1 != len(loader):
                    continue
                norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
                if not torch.isfinite(norm):
                    raise FloatingPointError("Nonfinite goal-head gradients")
                if any(p.grad is not None for p in dynamics.parameters()):
                    raise ValueError("Frozen dynamics received parameter gradients")
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                row = dict(step=step, **{k: float(np.mean([v[k] for v in accumulated])) for k in accumulated[0]})
                accumulated.clear()
                log.write(json.dumps(row) + "\n")
                log.flush()
                if step == 1 or step % 10 == 0:
                    print("ENDPOINT_STEP", name, json.dumps(row), flush=True)
                    if run is not None:
                        run.log({f"{name}/{k}": v for k, v in row.items()})
                if step % args.save_steps == 0 or step == args.updates:
                    if owner_digest(model.action_head) != before:
                        raise ValueError("Frozen owner weights changed")
                    after_probe = owner_probe(model, validation)
                    if not torch.equal(probe, after_probe):
                        raise ValueError("Mode-1 action probe changed")
                    model.train()
                    target = output / f"checkpoint-{step}"
                    model.save_pretrained(target)
                    shutil.copyfile(args.checkpoint / "metadata.pt", target / "metadata.pt")
                    shutil.copyfile(output / "training_config.json", target / "training_config.json")
                    torch.save(
                        dict(step=step, optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict()),
                        target / "trainer_state.pt",
                    )
                    print("ENDPOINT_CHECKPOINT", name, target, flush=True)
                if step >= args.updates:
                    break
            epoch += 1
    report = validate(model, validation, dynamics, codec, std_limit)
    report.update(owner_preserved=owner_digest(model.action_head) == before, updates=step, endpoint_lambda=weight)
    (output / "completed.json").write_text(json.dumps(report, indent=2) + "\n")
    del model, optimizer, scheduler, params, dynamics, codec
    gc.collect()
    torch.cuda.empty_cache()
    return output / f"checkpoint-{step}"


def main(args):
    check_worker()
    for name in ("output", "checkpoint", "baseline", "retriever", "policy_reference"):
        setattr(args, name, (REPO / getattr(args, name)).resolve())
    if min(args.updates, args.batch_size, args.gradient_accumulation, args.dynamics_epochs, args.save_steps) < 1:
        raise ValueError("Invalid training budget")
    args.output.mkdir(parents=True, exist_ok=False)
    recipe = json.loads((args.checkpoint / "training_config.json").read_text())
    args.dataset_paths = recipe["args"]["dataset_paths"]
    args.video_cache_size = 4
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (args.output / "pilot_config.json").write_text(json.dumps(config, indent=2) + "\n")
    run = None
    status = dict(status="starting")
    try:
        if args.wandb_mode != "disabled":
            import wandb

            run = wandb.init(
                project="icl-vla-uncertainty",
                entity="ryan-nlp",
                name=args.output.name,
                group="goal_endpoint_pilot",
                mode=args.wandb_mode,
                dir=str(args.output),
                config=config,
            )
        dynamics = train_local_dynamics.fit(args, recipe, args.output / "dynamics", run)
        if not dynamics["ready_for_endpoint"]:
            status = dict(
                status="dynamics_rejected",
                reason="Empirical validation gates failed; endpoint fine-tuning was not started.",
            )
            print("ENDPOINT_SKIPPED", json.dumps(status), flush=True)
            return
        seed_everything(args.seed)
        model = GoalControlRAVLA.from_pretrained(
            str(args.checkpoint),
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            tune_visual=False,
            tune_llm=False,
        ).eval()
        mixture = load_data(args)
        cache = args.output / "visual_cache"
        build_cache(model, mixture, recipe["split"], cache)
        del model, mixture
        gc.collect()
        torch.cuda.empty_cache()
        checkpoints = []
        for name, weight in [("control", 0.0), ("endpoint", args.endpoint_lambda)]:
            checkpoints.append((name, train_variant(args, cache, weight, args.output / name, run)))
        if not args.no_eval:
            for name, checkpoint in checkpoints:
                for execute in (16, 4):
                    output = args.output / "eval" / f"{name}_execute{execute}"
                    command = [
                        sys.executable,
                        "-u",
                        "-m",
                        "scripts.rag.eval_fixed_goal",
                        "--checkpoint",
                        str(checkpoint),
                        "--baseline",
                        str(args.baseline),
                        "--output",
                        str(output),
                        "--execute-steps",
                        str(execute),
                        "--port",
                        "8120",
                    ]
                    subprocess.run(command, cwd=REPO, check=True)
        status = dict(status="completed", checkpoints={name: str(path) for name, path in checkpoints})
    except BaseException as error:
        status = dict(status="failed", error=str(error))
        raise
    finally:
        (args.output / "status.json").write_text(json.dumps(status, indent=2) + "\n")
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
        "--baseline", type=Path, default=Path("outputs/ravla_train/gos_seed42_20261026T211436Z/checkpoint_3")
    )
    parser.add_argument("--retriever", type=Path, default=Path("ra-vla-official-assets/libero_gos"))
    parser.add_argument(
        "--policy-reference", type=Path, default=Path("outputs/ravla_eval/fixed_goal_prefix64_exec16_20261103")
    )
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--endpoint-lambda", type=float, default=0.1)
    parser.add_argument("--dynamics-epochs", type=int, default=100)
    parser.add_argument("--ensemble", type=int, default=3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--spectral", action="store_true")
    parser.add_argument("--max-mean-m", type=float, default=0.005)
    parser.add_argument("--max-p95-m", type=float, default=0.01)
    parser.add_argument("--max-policy-error-m", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    main(parser.parse_args())
