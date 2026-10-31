"""Cache frozen Eagle context with explicit demo/frame identity on a GPU worker."""

from dataclasses import dataclass
import json
from pathlib import Path

import torch
import tyro
from tqdm import tqdm

from examples.Libero.custom_data_config import LiberoDataConfig
from gr00t.data.schema import EmbodimentTag
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.rag.dual_mode_data import encode_context
from gr00t.rag.dual_mode_memory import metadata_digest, module_digest
from gr00t.rag.ravla import RAVLADataCollator
from gr00t.rag.retriever import RetrieverWrapperWithHead
from gr00t.rag.utils import seed_everything
from gr00t.rag.video_dataset import BoundedLeRobotVideoDataset
from scripts.rag.train_dual_mode import check_worker


@dataclass
class Args:
    output: Path
    dataset_path: str = "ra-vla-official-assets/ravla_code/ravla_code_no_model/datasets/libero_10"
    retriever: Path = Path("ra-vla-official-assets/libero_gos")
    base_model: str = "nvidia/GR00T-N1.5-3B"
    demos_per_task: int = 3
    frame_stride: int = 4
    batch_size: int = 8
    seed: int = 42
    limit_tasks: int = 0
    resume: bool = False


def save_manifest(directory, manifest):
    temporary = directory / ".manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(directory / "manifest.json")


def main(args: Args):
    check_worker()
    if min(args.demos_per_task, args.frame_stride, args.batch_size) < 1:
        raise ValueError("Demo count, stride, and batch size must be positive")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    seed_everything(args.seed)
    config = LiberoDataConfig()
    transforms = config.transform()
    metadata = torch.load(args.retriever / "metadata.pt", map_location="cpu", weights_only=False)
    transforms.set_metadata(metadata["new_embodiment"])
    transforms.eval()
    dataset = BoundedLeRobotVideoDataset(
        dataset_path=args.dataset_path,
        modality_configs=config.modality_config(),
        transforms=transforms,
        embodiment_tag=EmbodimentTag("new_embodiment"),
        video_backend="torchvision_av",
        cache_size=6,
    )
    # Dataset construction installs its own statistics; restore GOS statistics
    # afterwards so held-out Long actions never define normalization.
    transforms.set_metadata(metadata["new_embodiment"])
    transforms.eval()
    transforms.transforms[-1].train()  # Include actions; keep image augmentation disabled.
    model = GR00T_N1_5.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="cuda:0", tune_llm=False, tune_visual=False
    ).eval()
    retriever = RetrieverWrapperWithHead.from_pretrained(
        args.retriever, torch_dtype=torch.bfloat16, device_map="cuda:0"
    ).eval()
    collator = RAVLADataCollator()
    manifest = {
        "schema_version": 1,
        "status": "building",
        "dataset_path": str(Path(args.dataset_path).resolve()),
        "base_model": args.base_model,
        "retriever": str(args.retriever.resolve()),
        "demos_per_task": args.demos_per_task,
        "limit_tasks": args.limit_tasks,
        "seed": args.seed,
        "normalization_sha256": metadata_digest(metadata),
        "frame_stride": args.frame_stride,
        "backbone_sha256": module_digest(model.backbone),
        "retriever_sha256": module_digest(retriever),
        "selection": "first available episode IDs per task",
        "tasks": {},
    }
    if args.resume:
        saved = json.loads((args.output / "manifest.json").read_text())

        def signature(value):
            return {k: v for k, v in value.items() if k not in ("status", "tasks")}

        if signature(saved) != signature(manifest):
            raise ValueError("Memory resume settings or model fingerprints differ")
        manifest["tasks"] = saved["tasks"]
    save_manifest(args.output, manifest)
    tasks = list(dataset.task_groups)
    if args.limit_tasks:
        tasks = tasks[: args.limit_tasks]
    with torch.no_grad():
        for task_index, task in enumerate(tqdm(tasks, desc="cache tasks")):
            ids = sorted(dataset.task_groups[task])[: args.demos_per_task]
            if len(ids) != args.demos_per_task:
                raise ValueError(f"Not enough reference demos for {task}")
            coordinates = []
            for episode in ids:
                length = int(dataset.trajectory_lengths[dataset.get_trajectory_index(episode)])
                frames = sorted(set(range(0, length, args.frame_stride)) | {length - 1})
                coordinates.extend((episode, frame, length) for frame in frames)
            if task in manifest["tasks"]:
                record = manifest["tasks"][task]
                entry = torch.load(args.output / record["file"], map_location="cpu", mmap=True, weights_only=True)
                if (
                    record["episode_ids"] != ids
                    or len(entry["keys"]) != len(coordinates)
                    or not torch.equal(entry["episode_id"], torch.tensor([v[0] for v in coordinates]))
                    or not torch.equal(entry["frame_index"], torch.tensor([v[1] for v in coordinates]))
                    or not torch.equal(entry["trajectory_length"], torch.tensor([v[2] for v in coordinates]))
                ):
                    raise ValueError("Cached memory provenance differs")
                continue
            chunks = {key: [] for key in ("keys", "action", "backbone_features", "backbone_attention_mask")}
            for start in range(0, len(coordinates), args.batch_size):
                samples = [
                    transforms(dataset.get_step_data(ep, frame))
                    for ep, frame, _ in coordinates[start : start + args.batch_size]
                ]
                batch = collator(samples)
                if any(value != task for value in batch.pop("tasks")):
                    raise ValueError("Task identity differs between dataset and collator")
                chunks["keys"].append(retriever(model, batch).cpu())
                visual = encode_context(model, batch)
                chunks["action"].append(batch["action"].cpu())
                chunks["backbone_features"].append(visual.backbone_features.cpu())
                chunks["backbone_attention_mask"].append(visual.backbone_attention_mask.cpu())
            entry = {key: torch.cat(values) for key, values in chunks.items()}
            entry.update(
                episode_id=torch.tensor([v[0] for v in coordinates]),
                frame_index=torch.tensor([v[1] for v in coordinates]),
                trajectory_length=torch.tensor([v[2] for v in coordinates]),
            )
            filename = f"task-{task_index:02d}.pt"
            torch.save(entry, args.output / f".{filename}.tmp")
            (args.output / f".{filename}.tmp").rename(args.output / filename)
            manifest["tasks"][task] = {"file": filename, "episode_ids": ids, "frames": len(coordinates)}
            save_manifest(args.output, manifest)
    manifest["status"] = "completed"
    save_manifest(args.output, manifest)


if __name__ == "__main__":
    main(tyro.cli(Args))
