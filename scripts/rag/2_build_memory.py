"""Build auditable RA-VLA demo memory with bounded video/GPU batches."""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from time import perf_counter
from typing import List

import torch
from tqdm import tqdm
import tyro

from gr00t.data.dataset import CachedLeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import load_data_config
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.model.transforms import DefaultDataCollator
from gr00t.rag.memory import encode_memory_bank, file_fingerprint, plan_memory, save_memory
from gr00t.rag.retriever import RetrieverWrapperWithHead
from gr00t.rag.utils import seed_everything
from gr00t.rag.video_dataset import BoundedLeRobotVideoDataset
from scripts.rag.precompute_retrieval import atomic_json, run_lock, stamp

DATA_CONFIG = "examples.Libero.custom_data_config:LiberoDataConfig"
EMBODIMENT_TAG = "new_embodiment"
VIDEO_BACKEND = "torchvision_av"
PRETRAINED_VLA_PATH = "nvidia/GR00T-N1.5-3B"


@dataclass
class ArgsConfig:
    seed: int = 0
    cuda: int = 0
    retriever_path: str = "models/retriever/default"
    dataset_paths: List[str] = field(default_factory=list)
    memory_name: str = "memory"
    data_config: str | None = None
    """Defaults to retriever training config, or LIBERO for released models."""
    pretrained_model_path: str | None = None
    task_map: str | None = None
    """For one dataset; otherwise auto-load meta/episode_tasks.json."""
    tasks: List[str] | None = None
    """Canonical task names for RoboMME; instruction keys for LIBERO."""
    num_trajs_per_task: int = 3
    frame_stride: int = 4
    batch_size: int = 4
    cache_video: bool = False
    video_cache_size: int = 2
    max_frames_per_demo: int | None = None
    """Smoke-test cap; omit to encode every strided frame of each demo."""
    dry_run: bool = False
    """Print the plan without loading models or writing files."""


def resolve_training_config(config):
    path = Path(config.retriever_path) / "training_config.json"
    trained = json.loads(path.read_text()) if path.is_file() else {}
    for name, default in (("data_config", DATA_CONFIG), ("pretrained_model_path", PRETRAINED_VLA_PATH)):
        supplied, expected = getattr(config, name), trained.get(name)
        if supplied is not None and expected is not None and supplied != expected:
            raise ValueError(f"{name} disagrees with retriever training_config.json")
        setattr(config, name, supplied or expected or default)


def main(config: ArgsConfig):
    if not config.dataset_paths or min(config.batch_size, config.video_cache_size) < 1 or config.cuda < 0:
        raise ValueError("Supply datasets, positive batch/cache sizes and a nonnegative GPU index")
    if config.task_map is not None and len(config.dataset_paths) != 1:
        raise ValueError("An explicit --task-map requires one dataset")
    if Path(config.memory_name).name != config.memory_name or config.memory_name in ("", ".", ".."):
        raise ValueError("memory_name must be a filename stem, not a path")
    resolve_training_config(config)
    seed_everything(config.seed)
    root = Path(config.retriever_path)
    output, manifest_path = root / f"{config.memory_name}.pt", root / f"{config.memory_name}.json"
    if not config.dry_run and (output.exists() or manifest_path.exists()):
        raise FileExistsError("Memory or manifest already exists; choose another --memory-name")
    data_config = load_data_config(config.data_config)
    full_cache = config.cache_video and not config.dry_run
    dataset_class = CachedLeRobotSingleDataset if full_cache else BoundedLeRobotVideoDataset
    datasets = []
    for path in config.dataset_paths:
        transforms = data_config.transform()
        transforms.eval()  # Deterministic images; keep action output in the final transform.
        transforms.transforms[-1].train()
        dataset = dataset_class(
            dataset_path=path,
            modality_configs=data_config.modality_config(),
            transforms=transforms,
            embodiment_tag=EmbodimentTag(EMBODIMENT_TAG),
            video_backend=VIDEO_BACKEND,
            **({} if full_cache else {"cache_size": config.video_cache_size}),
        )
        datasets.append(dataset)
    plan = plan_memory(
        datasets,
        num_trajs_per_task=config.num_trajs_per_task,
        frame_stride=config.frame_stride,
        seed=config.seed,
        tasks=config.tasks,
        task_map=config.task_map,
        max_frames_per_demo=config.max_frames_per_demo,
    )
    print(
        f"Memory: {len(plan['banks'])} {plan['key_type']} banks, "
        f"{config.num_trajs_per_task} demos/bank, {plan['num_candidates']} candidates; "
        f"smoke_cap={plan['is_smoke_sample']}",
        flush=True,
    )
    if config.dry_run:
        print(json.dumps(plan, indent=2))
        return plan
    if not torch.cuda.is_available():
        raise RuntimeError("Run encoding in a GPU allocation; --dry-run supports CPU")
    # Use TRAIN normalization even when encoding held-out support demos.
    metadata_path = root / "metadata.pt"
    metadata = torch.load(metadata_path, map_location="cpu", weights_only=False)
    for dataset in datasets:
        dataset.set_transforms_metadata(metadata[dataset.tag])
    weight_paths = sorted({*root.glob("*.safetensors"), *root.glob("pytorch_model*.bin")})
    if not weight_paths:
        raise FileNotFoundError(f"No trained retriever weights in {root}")
    provenance_paths = [metadata_path, root / "config.json", *weight_paths]
    if (root / "training_config.json").is_file():
        provenance_paths.append(root / "training_config.json")
    sources = []
    for bank in plan["banks"]:
        dataset = datasets[bank["dataset_index"]]
        for demo in bank["demos"]:
            episode = demo["episode_index"]
            parquet = dataset.dataset_path / dataset.data_path_pattern.format(
                episode_chunk=dataset.get_episode_chunk(episode),
                episode_index=episode,
            )
            sources.append(
                {
                    "dataset_index": bank["dataset_index"],
                    "episode_index": episode,
                    "parquet": file_fingerprint(parquet),
                    "videos": {
                        key: stamp(dataset.get_video_path(episode, key.removeprefix("video.")))
                        for key in dataset.modality_keys["video"]
                    },
                }
            )
    manifest = {
        "version": 1,
        "status": "planned",
        "config": asdict(config),
        "plan": plan,
        "retriever": [file_fingerprint(path) for path in provenance_paths],
        "sources": sources,
        "implementation": [
            file_fingerprint(Path(__file__)),
            file_fingerprint(Path(__file__).resolve().parents[2] / "gr00t/rag/memory.py"),
        ],
        "completed_banks": [],
        "normalization_source": str(metadata_path.resolve()),
        "lookup": "Canonical task IDs for indexed datasets; candidate instructions remain in embeddings.",
    }
    started = perf_counter()
    with run_lock(root):
        if output.exists() or manifest_path.exists():
            raise FileExistsError("Memory or manifest already exists; choose another --memory-name")
        atomic_json(manifest_path, manifest)
        try:
            device = f"cuda:{config.cuda}"
            torch.cuda.set_device(config.cuda)
            torch.cuda.reset_peak_memory_stats(device)
            vla = (
                GR00T_N1_5.from_pretrained(
                    config.pretrained_model_path,
                    torch_dtype=torch.bfloat16,
                    device_map=device,
                )
                .eval()
                .requires_grad_(False)
            )
            vla.compute_dtype = vla.config.compute_dtype = "bfloat16"
            retriever = (
                RetrieverWrapperWithHead.from_pretrained(
                    config.retriever_path,
                    torch_dtype=torch.bfloat16,
                    device_map=device,
                )
                .eval()
                .requires_grad_(False)
            )
            collator = DefaultDataCollator()
            memory = {}
            manifest["status"] = "encoding"
            for bank in tqdm(plan["banks"], desc="Memory banks"):
                memory[bank["key"]] = encode_memory_bank(
                    datasets[bank["dataset_index"]],
                    bank,
                    collator,
                    vla,
                    retriever,
                    config.batch_size,
                )
                manifest["completed_banks"].append(bank["key"])
                atomic_json(manifest_path, manifest)
            save_memory(memory, output)
            manifest.update(
                {
                    "status": "complete",
                    "output": {"path": str(output.resolve()), "bytes": output.stat().st_size},
                    "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
                    "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
                    "candidate_counts": {key: len(bank["keys"]) for key, bank in memory.items()},
                }
            )
        except BaseException as error:
            manifest.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
            raise
        finally:
            manifest["seconds"] = perf_counter() - started
            atomic_json(manifest_path, manifest)
    print(f"Saved {output} and {manifest_path}", flush=True)
    return manifest


if __name__ == "__main__":
    main(tyro.cli(ArgsConfig))
