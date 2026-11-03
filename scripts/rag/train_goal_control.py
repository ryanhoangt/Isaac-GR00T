"""Warm-started same-demo goal positive control, with a frozen Mode-1 owner."""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers.feature_extraction_utils import BatchFeature
import tyro

from gr00t.rag.dual_mode_data import encode_context
from gr00t.rag.dual_mode_memory import metadata_digest
from gr00t.rag.goal_control import GoalControlRAVLA, assert_warm_owner, owner_digest
from gr00t.rag.goal_control_data import GoalCollator, SameDemoGoals, split_tasks, validation_paths
from gr00t.rag.utils import seed_everything
from scripts.rag.train_dual_mode import DATA_ROOT, check_worker, load_data


@dataclass
class Args:
    output: Path
    warm_start: Path = Path("outputs/ravla_train/gos_seed42_20261026T211436Z/checkpoint_3")
    dataset_paths: list[str] = field(
        default_factory=lambda: [f"{DATA_ROOT}/libero_{s}" for s in ("goal", "object", "spatial")]
    )
    retriever: Path = Path("ra-vla-official-assets/libero_gos")
    video_cache_size: int = 4
    seed: int = 42
    tasks_per_suite: int = 1
    heldout_demos: int = 3
    states_per_demo: int = 20
    lookahead: int = 16
    updates: int = 2000
    batch_size: int = 8
    gradient_accumulation: int = 4
    num_workers: int = 2
    learning_rate: float = 2e-4
    save_steps: int = 1000
    validation_steps: int = 1000
    log_steps: int = 10
    no_wandb: bool = False


def prepare(model, batch):
    current = dict(batch["current"])
    current.pop("tasks")
    context = encode_context(model, batch["goal"])
    current.update(
        context_mode=torch.ones(len(current["state"]), dtype=torch.long),
        ret_actions=torch.zeros_like(current["action"]),
        ret_backbone_output=context,
    )
    return current


@torch.inference_mode()
def build_validation(model, mixture, paths, lookahead=16):
    samples = SameDemoGoals(mixture, paths, lookahead)
    loader = DataLoader(samples, batch_size=4, collate_fn=GoalCollator(), num_workers=0)
    result = []
    for batch in loader:
        current = encode_context(model, batch["current"])
        goal = encode_context(model, batch["goal"])
        for i, provenance in enumerate(batch["provenance"]):
            result.append(
                dict(
                    provenance=provenance,
                    current={
                        k: current[k][i : i + 1].cpu() for k in ("backbone_features", "backbone_attention_mask")
                    },
                    goal={k: goal[k][i : i + 1].cpu() for k in ("backbone_features", "backbone_attention_mask")},
                    state=batch["current"]["state"][i : i + 1].cpu(),
                    embodiment_id=batch["current"]["embodiment_id"][i : i + 1].cpu(),
                    action=batch["current"]["action"][i : i + 1].cpu(),
                    mask=batch["current"]["action_mask"][i : i + 1].cpu(),
                )
            )
    print("VALIDATION_CACHE", len(result), flush=True)
    return result


@torch.inference_mode()
def cached_action(model, row, goal, seed, nfe=4, mode=1):
    head = model.action_head

    def move(value):
        return value.to(model.device, dtype=head.dtype) if value.is_floating_point() else value.to(model.device)

    visual = BatchFeature(data={k: move(v) for k, v in row["current"].items()})
    context = BatchFeature(data={k: move(v) for k, v in goal.items()})
    target = row["action"].to(model.device, dtype=head.dtype)
    generator = torch.Generator(device=model.device).manual_seed(seed)
    noise = torch.randn(target.shape, generator=generator, device=model.device, dtype=head.dtype)
    inputs = BatchFeature(
        data=dict(
            state=row["state"].to(model.device, dtype=head.dtype),
            embodiment_id=row["embodiment_id"].to(model.device),
            ret_actions=target if mode == 0 else torch.zeros_like(target),
            ret_backbone_output=context,
            context_mode=torch.full((1,), mode, device=model.device),
            flow_source=noise,
        )
    )
    head.num_inference_timesteps = nfe
    result = head.get_action(visual, inputs).action_pred.cpu().float()
    if not torch.isfinite(result).all():
        raise FloatingPointError(f"Nonfinite cached action prediction (mode={mode}, NFE={nfe})")
    return result


@torch.inference_mode()
def evaluate(model, cache, step, output, baseline_action=None, run=None, expected_owner=None):
    model.eval()
    rows = []
    by_demo = {}
    for i, row in enumerate(cache):
        p = row["provenance"]
        by_demo.setdefault((p["dataset"], p["episode"]), []).append(i)
    for i, row in enumerate(cache):
        p = row["provenance"]
        group = by_demo[(p["dataset"], p["episode"])]
        wrong = cache[group[(group.index(i) + max(1, len(group) // 2)) % len(group)]]["goal"]
        null = {k: torch.zeros_like(v) if k == "backbone_features" else v for k, v in row["goal"].items()}
        variants = [
            ("correct", row["goal"], 4),
            ("wrong_time", wrong, 4),
            ("null", null, 4),
            ("correct", row["goal"], 8),
            ("correct", row["goal"], 16),
        ]
        correct = None
        for name, goal, nfe in variants:
            prediction = cached_action(model, row, goal, 10000 + i, nfe)
            mask = row["mask"].float().clone()
            mask[..., 7:] = 0
            delta = (prediction - row["action"].float()).square() * mask
            item = dict(
                state=i, condition=name, nfe=nfe, normalized_mse7=float(delta.sum() / mask.sum().clamp_min(1))
            )
            if name == "correct" and nfe == 4:
                correct = prediction[..., :7]
            item["relative_to_correct4"] = float(
                torch.linalg.vector_norm(prediction[..., :7] - correct)
                / torch.linalg.vector_norm(correct).clamp_min(1e-6)
            )
            rows.append(item)
        if (i + 1) % 30 == 0:
            print("GOAL_VALIDATION", step, i + 1, len(cache), flush=True)
    probe = cached_action(model, cache[0], cache[0]["current"], 777, 4, mode=0)
    current_owner = owner_digest(model.action_head)
    parity = dict(
        step=step,
        owner_sha256=current_owner,
        expected_owner_sha256=expected_owner,
        weights_equal=expected_owner is None or current_owner == expected_owner,
        output_equal=baseline_action is None or torch.equal(probe, baseline_action),
        max_abs_difference=0.0 if baseline_action is None else float((probe - baseline_action).abs().max()),
        relative_l2=0.0
        if baseline_action is None
        else float(
            torch.linalg.vector_norm(probe - baseline_action)
            / torch.linalg.vector_norm(baseline_action).clamp_min(1e-6)
        ),
    )
    (output / f"mode1_parity_step{step:05d}.json").write_text(json.dumps(parity, indent=2) + "\n")
    print("MODE1_PARITY", json.dumps(parity), flush=True)
    if not parity["weights_equal"]:
        raise ValueError("Frozen Mode-1 weights changed during goal training")
    if baseline_action is not None and not torch.equal(probe, baseline_action):
        raise ValueError("Mode-1 output changed during isolated goal training")
    summary = []
    for condition, nfe in [("correct", 4), ("wrong_time", 4), ("null", 4), ("correct", 8), ("correct", 16)]:
        selected = [r for r in rows if r["condition"] == condition and r["nfe"] == nfe]
        item = dict(
            condition=condition,
            nfe=nfe,
            mse=float(np.mean([r["normalized_mse7"] for r in selected])),
            relative_to_correct4=float(np.mean([r["relative_to_correct4"] for r in selected])),
        )
        summary.append(item)
        if run is not None:
            run.log(
                {
                    "step": step,
                    f"validation/{condition}_{nfe}/mse": item["mse"],
                    f"validation/{condition}_{nfe}/relative": item["relative_to_correct4"],
                }
            )
    report = dict(
        step=step,
        states=len(cache),
        scope="offline same-scene expert-state positive control; not rollout/recovery SR",
        validation_demos_excluded_from_goal_updates=True,
        warm_start_may_have_seen_validation_demos=True,
        mode1_exactly_preserved=baseline_action is None or torch.equal(probe, baseline_action),
        results=summary,
        rows=rows,
    )
    (output / f"validation_step{step:05d}.json").write_text(json.dumps(report, indent=2) + "\n")
    print("GOAL_VALIDATION_COMPLETE", json.dumps({k: v for k, v in report.items() if k != "rows"}), flush=True)
    head = model.action_head
    head.num_inference_timesteps = 4
    model.train()
    return probe


def main(args):
    check_worker()
    if (
        min(
            args.updates,
            args.batch_size,
            args.gradient_accumulation,
            args.tasks_per_suite,
            args.heldout_demos,
            args.states_per_demo,
            args.lookahead,
            args.save_steps,
            args.validation_steps,
            args.log_steps,
        )
        < 1
        or args.num_workers < 0
    ):
        raise ValueError("Invalid positive-control budget")
    seed_everything(args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    mixture = load_data(args)
    # Remove image augmentation for the easiest same-scene positive control.
    for dataset in mixture.datasets:
        dataset.transforms.eval()
        dataset.transforms.transforms[-1].train()  # Include supervised action tensors.
    mixture.save_metadata(str(args.output))
    old_metadata = torch.load(args.warm_start / "metadata.pt", map_location="cpu", weights_only=False)
    new_metadata = torch.load(args.output / "metadata.pt", map_location="cpu", weights_only=False)
    if metadata_digest(old_metadata) != metadata_digest(new_metadata):
        raise ValueError("Warm-start normalization differs from goal-control data")
    train_paths, val_episodes, inventory = split_tasks(mixture, args.tasks_per_suite, args.heldout_demos)
    val_paths = validation_paths(mixture, val_episodes, args.states_per_demo)
    train_set = SameDemoGoals(mixture, train_paths, args.lookahead)
    recipe = dict(
        args=asdict(args),
        goal_probability=1.0,
        goal_pairing=f"same-demo fixed +{args.lookahead}",
        adherence_weight=0.0,
        owner_frozen=True,
        goal_branch="full copied action head; shared frozen Eagle",
        image_augmentation=False,
        training_frames=len(train_set),
        split=inventory,
        validation_paths=val_paths,
    )
    (args.output / "training_config.json").write_text(json.dumps(recipe, indent=2, default=str) + "\n")
    model = GoalControlRAVLA.from_pretrained(
        str(args.warm_start), torch_dtype=torch.bfloat16, device_map="cuda:0", tune_visual=False, tune_llm=False
    ).train()
    model.action_head.initialize_goal_from_owner()
    verified_tensors = assert_warm_owner(model.action_head, args.warm_start)
    print("WARM_OWNER_VERIFIED", verified_tensors, flush=True)
    model.action_head.num_inference_timesteps = 4
    model.config.action_head_cfg = model.action_head.config.to_dict()
    frozen_digest = owner_digest(model.action_head)
    params = [p for p in model.parameters() if p.requires_grad]
    if any(
        p.requires_grad for name, p in model.action_head.named_parameters() if not name.startswith("goal_branch.")
    ):
        raise ValueError("Mode-1 owner is trainable")
    print(
        "GOAL_CONTROL_READY",
        json.dumps(
            dict(
                training_frames=len(train_set),
                validation_states=len(val_paths),
                trainable_parameters=sum(p.numel() for p in params),
                owner_sha256=frozen_digest,
            )
        ),
        flush=True,
    )
    run = None
    if not args.no_wandb:
        import wandb

        run = wandb.init(
            project="icl-vla-uncertainty",
            entity="ryan-nlp",
            name=args.output.name,
            group="dual_mode_goal_positive_control",
            dir=str(args.output.resolve()),
            config=json.loads(json.dumps(recipe, default=str)),
        )
        run.define_metric("step")
        run.define_metric("train/*", step_metric="step")
    try:
        cache = build_validation(model, mixture, val_paths, args.lookahead)
        baseline = evaluate(model, cache, 0, args.output, run=run, expected_owner=frozen_digest)
        # Validation has already used native video/tokenizer/torch thread pools.
        # Do not fork that state into data workers, or pickle decoded video caches.
        for dataset in mixture.datasets:
            dataset._video_cache.clear()
        optimizer = AdamW(params, lr=args.learning_rate, weight_decay=1e-5)
        scheduler = CosineAnnealingLR(optimizer, args.updates)
        step, epoch, batch_index = 0, 0, 0
        optimizer.zero_grad(set_to_none=True)
        model.train()
        while step < args.updates:
            generator = torch.Generator().manual_seed(args.seed + epoch)
            loader = DataLoader(
                train_set,
                batch_size=args.batch_size,
                shuffle=True,
                generator=generator,
                num_workers=args.num_workers,
                collate_fn=GoalCollator(),
                drop_last=True,
                **({"prefetch_factor": 1, "multiprocessing_context": "spawn"} if args.num_workers else {}),
            )
            if not len(loader):
                raise ValueError("Empty positive-control loader")
            window_loss = []
            for batch_index, batch in enumerate(loader):
                if step < 2:
                    print("GOAL_BATCH_READY", epoch, batch_index, flush=True)
                batch_started = time.perf_counter()
                start = batch_index // args.gradient_accumulation * args.gradient_accumulation
                window = min(args.gradient_accumulation, len(loader) - start)
                loss = model(prepare(model, batch)).loss
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite goal loss")
                (loss / window).backward()
                if step < 2:
                    print(
                        "GOAL_BACKWARD_DONE", epoch, batch_index, time.perf_counter() - batch_started, flush=True
                    )
                window_loss.append(float(loss.detach()))
                if (batch_index + 1) % args.gradient_accumulation and batch_index + 1 != len(loader):
                    continue
                norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
                if not torch.isfinite(norm):
                    raise FloatingPointError("Nonfinite goal gradient")
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                metrics = dict(
                    step=step,
                    mse=float(np.mean(window_loss)),
                    grad_norm=float(norm),
                    learning_rate=optimizer.param_groups[0]["lr"],
                )
                window_loss.clear()
                if step == 1 or step % args.log_steps == 0:
                    print("GOAL_STEP", json.dumps(metrics), flush=True)
                    if run is not None:
                        run.log({"step": step, **{f"train/{k}": v for k, v in metrics.items() if k != "step"}})
                if step % args.validation_steps == 0 or step == args.updates:
                    evaluate(model, cache, step, args.output, baseline, run, expected_owner=frozen_digest)
                if step % args.save_steps == 0 or step == args.updates:
                    if owner_digest(model.action_head) != frozen_digest:
                        raise ValueError("Frozen Mode-1 weights changed")
                    temporary = args.output / f".checkpoint-{step}.tmp"
                    final = args.output / f"checkpoint-{step}"
                    mixture.save_metadata(str(temporary))
                    model.save_pretrained(temporary)
                    torch.save(
                        dict(
                            step=step,
                            epoch=epoch,
                            next_batch=batch_index + 1,
                            optimizer=optimizer.state_dict(),
                            scheduler=scheduler.state_dict(),
                            torch_rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state(),
                        ),
                        temporary / "trainer_state.pt",
                    )
                    (temporary / "training_config.json").write_text(
                        (args.output / "training_config.json").read_text()
                    )
                    temporary.rename(final)
                    print("GOAL_CHECKPOINT", final, flush=True)
                if step >= args.updates:
                    break
            epoch += 1
        report = dict(
            status="completed",
            steps=step,
            owner_sha256=frozen_digest,
            mode1_weights_preserved=owner_digest(model.action_head) == frozen_digest,
            gpu_peak_gib=torch.cuda.max_memory_allocated() / 1024**3,
        )
        (args.output / "completed.json").write_text(json.dumps(report, indent=2) + "\n")
        print("GOAL_CONTROL_COMPLETED", json.dumps(report), flush=True)
    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main(tyro.cli(Args))
