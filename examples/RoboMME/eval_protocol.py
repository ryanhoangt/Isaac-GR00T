"""Task selection and checkpoint/server provenance without model or simulator imports."""

from collections import Counter
import hashlib
import json
from pathlib import Path

from examples.RoboMME.memory_client import PROTOCOL


ALL_TASKS = frozenset(
    (
        "BinFill",
        "PickXtimes",
        "SwingXtimes",
        "StopCube",
        "VideoUnmask",
        "VideoUnmaskSwap",
        "ButtonUnmask",
        "ButtonUnmaskSwap",
        "PickHighlight",
        "VideoRepick",
        "VideoPlaceButton",
        "VideoPlaceOrder",
        "MoveCube",
        "InsertPeg",
        "PatternLock",
        "RouteStick",
    )
)
PRIMARY_HELD_OUT = frozenset(("PickXtimes", "ButtonUnmaskSwap", "PickHighlight", "PatternLock"))
DEFAULT_TASK_INDEX = Path(__file__).parent / "metadata/robomme_task_index_v21.json"
INDEX_META = ("info.json", "episodes.jsonl", "tasks.jsonl", "modality.json")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def training_protocol(args, manifest_path, manifest):
    """Verify an indexed training dataset; instruction IDs are not task IDs."""
    root = manifest_path.parent
    index_path = args.task_index or (
        root / "meta/episode_tasks.json" if args.protocol == "primary" else DEFAULT_TASK_INDEX
    )
    index_path = Path(index_path).expanduser().resolve(strict=True)
    index = json.loads(index_path.read_text())
    actual = {name: digest(root / "meta" / name) for name in INDEX_META}
    if index.get("version") != 1 or index.get("source_metadata_sha256") != actual:
        raise ValueError("Task index does not match the training dataset metadata")
    episodes = [
        json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines() if line.strip()
    ]
    rows = index["episodes"]
    ids = {row["episode_index"] for row in episodes}
    indexed_ids = {row["episode_index"] for row in rows}
    if len(ids) != len(episodes) or len(indexed_ids) != len(rows) or ids != indexed_ids:
        raise ValueError("Task index must cover each training episode exactly once")
    counts = Counter(row["task"] for row in rows)
    train = set(counts)
    held_out = set(ALL_TASKS) - train
    expected = set(ALL_TASKS) - set(PRIMARY_HELD_OUT) if args.protocol == "primary" else set(ALL_TASKS)
    if train != expected:
        raise ValueError(f"{args.protocol} requires exactly its expected training tasks")
    if args.protocol == "primary" or "selected_tasks" in manifest:
        selected, excluded = manifest.get("selected_tasks", []), manifest.get("excluded_tasks", [])
        if (
            len(selected) != len(set(selected))
            or len(excluded) != len(set(excluded))
            or set(selected) != train
            or set(excluded) != held_out
        ):
            raise ValueError("Manifest task selection differs from indexed training episodes")
    hashes = {
        path.name: digest(path)
        for path in sorted((root / "meta").glob("*"))
        if path.is_file() and path.suffix in (".json", ".jsonl")
    }
    hashes["../manifest.json"] = digest(manifest_path)
    return (
        sorted(train),
        sorted(held_out),
        {
            "training_dataset": str(root),
            "training_metadata_sha256": hashes,
            "task_index": str(index_path),
            "task_index_sha256": digest(index_path),
            "training_episodes_per_task": dict(sorted(counts.items())),
        },
    )


def verify_memory_server(config):
    """Check the local checkpoint sidecar and the server's startup attestation.

    Canonical all16/primary protocols bind the actual training metadata. Legacy
    held-out mode keeps its declared-manifest semantics, useful for integration
    diagnostics, and explicitly records that the training split was not verified.
    """
    import requests

    root = Path(config["checkpoint"]).expanduser().resolve(strict=True)
    model_config = json.loads((root / "config.json").read_text())
    memory = model_config.get("action_head_cfg", {}).get("memory_config")
    if not memory:
        raise ValueError("Memory evaluation requires a memory-enabled checkpoint")
    path = root / "experiment_cfg/memory_training.json"
    record = json.loads(path.read_text())
    identity = record["identity"]
    if record.get("identity_sha256") != fingerprint(identity) or identity.get("memory_config") != memory:
        raise ValueError("Checkpoint memory training provenance is inconsistent")
    verified = config["protocol"] in ("all16", "primary")
    if verified:
        components = identity["datasets"]
        if len(components) != 1 or components[0]["metadata_sha256"] != config["training_metadata_sha256"]:
            raise ValueError("Checkpoint was trained on a different dataset than the evaluation protocol")
        normalization = json.loads((root / "experiment_cfg/metadata.json").read_text())
        if normalization.get("new_embodiment") != components[0]["normalization"]:
            raise ValueError("Checkpoint normalization differs from its training provenance")
    response = requests.get(config["server_url"].rstrip("/") + "/health", timeout=config["timeout"])
    response.raise_for_status()
    health = response.json()
    if (
        health.get("protocol") != PROTOCOL
        or health.get("stateful") is not True
        or health.get("checkpoint") != str(root)
    ):
        raise ValueError("Memory server protocol/checkpoint differs from the requested evaluation")
    expected = {
        "memory_training_sha256": digest(path),
        "checkpoint_config_sha256": digest(root / "config.json"),
        "normalization_metadata_sha256": digest(root / "experiment_cfg/metadata.json"),
    }
    if any(health.get(key) != value for key, value in expected.items()):
        raise ValueError("Memory server checkpoint provenance differs; restart the matching server")
    runtime = health.get("runtime", {})
    if runtime.get("seed") != config["server_seed"] or runtime.get("denoising_steps") != config["denoising_steps"]:
        raise ValueError("Memory server seed/denoising steps differ from the evaluation configuration")
    return {
        **expected,
        "checkpoint": str(root),
        "protocol": PROTOCOL,
        "training_identity_sha256": record["identity_sha256"],
        "memory_config_sha256": fingerprint(memory),
        "training_manifest_verified": verified,
        "runtime": runtime,
    }
