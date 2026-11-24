"""Plan and encode bounded batches for canonical-task or LIBERO memory banks."""

import hashlib
import json
from pathlib import Path
import random
import tempfile

import torch
from torch.nn.utils.rnn import pad_sequence

from gr00t.rag.data import _alignment_task_names


def file_fingerprint(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def plan_memory(
    datasets, *, num_trajs_per_task=3, frame_stride=4, seed=0, tasks=None, task_map=None, max_frames_per_demo=None
):
    """Select exactly N demos per bank, including singleton-instruction demos.

    An indexed dataset uses canonical task banks, with no exact-instruction
    prefilter at lookup time. Demos retain their own instruction for encoding.
    Unindexed LIBERO datasets keep their original instruction keys.
    """
    if not datasets or min(num_trajs_per_task, frame_stride) < 1:
        raise ValueError("Datasets, demo count and frame stride must be positive")
    if max_frames_per_demo is not None and max_frames_per_demo < 1:
        raise ValueError("max_frames_per_demo must be positive")
    if task_map is not None and len(datasets) != 1:
        raise ValueError("An explicit --task-map requires one dataset")
    groups, reports = {}, []
    for dataset_id, dataset in enumerate(datasets):
        owners, index_path = _alignment_task_names(dataset, task_map)
        grouping = "canonical_task" if owners is not None else "instruction"
        lengths = dict(zip(map(int, dataset.trajectory_ids), map(int, dataset.trajectory_lengths)))
        instructions, local_groups = {}, {}
        for instruction, episodes in dataset.task_groups.items():
            for episode in episodes:
                episode = int(episode)
                if episode in instructions:
                    raise ValueError(f"Episode {episode} occurs in multiple instruction groups")
                instructions[episode] = instruction
                key = owners[episode] if owners is not None else instruction
                local_groups.setdefault(key, []).append(episode)
        if set(instructions) != set(lengths) or any(length < 1 for length in lengths.values()):
            raise ValueError("Memory groups must cover every nonempty episode exactly once")
        for key, episodes in local_groups.items():
            if key in groups:
                raise ValueError(f"Duplicate memory key across datasets: {key}")
            groups[key] = (dataset_id, episodes, lengths, instructions)
        root = Path(dataset.dataset_path)
        reports.append(
            {
                "dataset": str(root.resolve()),
                "grouping": grouping,
                "task_map": file_fingerprint(index_path) if index_path is not None else None,
                "metadata": [
                    file_fingerprint(root / "meta" / name)
                    for name in ("info.json", "episodes.jsonl", "tasks.jsonl", "modality.json")
                    if (root / "meta" / name).is_file()
                ],
                "num_tasks": len(local_groups),
                "num_demonstrations": len(lengths),
            }
        )
    if len({report["grouping"] for report in reports}) != 1:
        raise ValueError("Cannot mix canonical-task and instruction-keyed datasets")
    if tasks is not None and (not tasks or set(tasks) - set(groups)):
        raise ValueError(f"Unknown or empty --tasks: {tasks}")
    banks = []
    for key in sorted(groups if tasks is None else set(tasks)):
        dataset_id, episodes, lengths, instructions = groups[key]
        if len(episodes) < num_trajs_per_task:
            raise ValueError(f"{key}: need {num_trajs_per_task} distinct demos, found {len(episodes)}")
        task_seed = int(hashlib.sha256(json.dumps([seed, key]).encode()).hexdigest()[:16], 16)
        selected = random.Random(task_seed).sample(sorted(episodes), num_trajs_per_task)
        demos = [
            {
                "episode_index": episode,
                "length": lengths[episode],
                "instruction": instructions[episode],
                "frame_indices": list(range(0, lengths[episode], frame_stride))[:max_frames_per_demo],
            }
            for episode in selected
        ]
        banks.append(
            {
                "key": key,
                "dataset_index": dataset_id,
                "demos": demos,
                "num_candidates": sum(len(demo["frame_indices"]) for demo in demos),
            }
        )
    return {
        "version": 1,
        "key_type": reports[0]["grouping"],
        "seed": seed,
        "num_trajs_per_task": num_trajs_per_task,
        "frame_stride": frame_stride,
        "max_frames_per_demo": max_frames_per_demo,
        "is_smoke_sample": max_frames_per_demo is not None,
        "datasets": reports,
        "banks": banks,
        "num_candidates": sum(bank["num_candidates"] for bank in banks),
    }


@torch.inference_mode()
def encode_memory_bank(dataset, plan, collator, vla, retriever, batch_size=8):
    """Stream transforms/inference and retain only CPU tensors between batches."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    frames = [(demo["episode_index"], frame) for demo in plan["demos"] for frame in demo["frame_indices"]]
    if not frames:
        raise ValueError("Cannot encode an empty memory bank")
    data = {key: [] for key in ("pixel_values", "action", "action_mask", "keys")}
    features, masks = [], []
    for start in range(0, len(frames), batch_size):
        batch_frames = frames[start : start + batch_size]
        inputs = collator(
            [dataset.transforms(dataset.get_step_data(episode, frame)) for episode, frame in batch_frames]
        )
        pixels = inputs["eagle_pixel_values"]
        if pixels.shape[0] != 2 * len(batch_frames):
            raise ValueError("Memory encoder requires exactly two camera images per candidate")
        data["pixel_values"].append(pixels.detach().cpu().unflatten(0, (-1, 2)))
        for name in ("action", "action_mask"):
            data[name].append(inputs[name].detach().cpu())
        data["keys"].append(retriever(vla, inputs).detach().cpu())
        backbone_inputs, _ = vla.prepare_input(inputs)
        output = vla.backbone(backbone_inputs)
        batch_features = output["backbone_features"].detach().cpu()
        batch_masks = output["backbone_attention_mask"].detach().cpu()
        for value, mask in zip(batch_features, batch_masks):
            valid = mask.bool()
            if not valid.any():
                raise ValueError("Cached backbone sequence has no valid tokens")
            # Different instructions/batches have different padding lengths.
            features.append(value[valid])
            masks.append(mask[valid])
        del output, backbone_inputs, inputs
    result = {key: torch.cat(parts) for key, parts in data.items()}
    result["backbone_features"] = pad_sequence(features, batch_first=True, padding_side="left")
    result["backbone_attention_mask"] = pad_sequence(masks, batch_first=True, padding_side="left")
    result["episode_ids"] = torch.tensor([episode for episode, _ in frames])
    result["frame_ids"] = torch.tensor([frame for _, frame in frames])
    if any(value.shape[0] != len(frames) for value in result.values()):
        raise ValueError("Cached candidate counts disagree")
    if any(value.is_floating_point() and not torch.isfinite(value).all() for value in result.values()):
        raise ValueError("Non-finite cached memory tensors")
    return result


def save_memory(memory, path):
    """Save CPU tensors atomically, verifying a reload before publishing."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Memory already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix="pending_memory_", suffix=".pt", delete=False
    ) as file:
        temporary = Path(file.name)
    try:
        torch.save(memory, temporary)
        restored = torch.load(temporary, map_location="cpu", weights_only=True, mmap=True)
        for key, bank in memory.items():
            for name, tensor in bank.items():
                if (
                    tensor.device.type != "cpu"
                    or tensor.requires_grad
                    or not torch.equal(tensor, restored[key][name])
                ):
                    raise ValueError(f"Memory reload/device validation failed: {key}/{name}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
