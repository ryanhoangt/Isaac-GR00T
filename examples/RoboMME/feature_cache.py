"""Episode-wise, resumable visual memory cache for original RoboMME LeRobot v2.

Source rows keep their original episode/frame indices. Conditioning-video rows
are inputs, never training labels. The cache has no trainable transformations,
normalization statistics, subgoal/oracle annotations or cross-episode context.
"""

from collections.abc import Mapping
from contextlib import contextmanager
import copy
import fcntl
import hashlib
import json
from numbers import Integral
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import yaml

from examples.RoboMME.memory_config import load_memory_config, validate_memory_config
from scripts.convert_robomme_lerobot import decode_image, load_source


CACHE_VERSION = 1
DEFAULT_CACHE_CONFIG = Path(__file__).parent / "config/cache_visual_features.yaml"


def json_fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_cache_config(path=DEFAULT_CACHE_CONFIG):
    config = yaml.safe_load(Path(path).read_text())
    return validate_cache_config(config)


def validate_cache_config(config):
    fields = {
        "source",
        "model_path",
        "output",
        "memory_config",
        "batch_size",
        "device",
        "compute_dtype",
        "storage_dtype",
        "attention_implementation",
        "verify",
    }
    if not isinstance(config, Mapping) or set(config) != fields:
        raise ValueError(f"Cache YAML must contain exactly {sorted(fields)}")
    for name in ("source", "model_path", "output", "memory_config", "device"):
        if not isinstance(config[name], str) or not config[name].strip():
            raise ValueError(f"{name} must be a nonempty string")
    if (
        isinstance(config["batch_size"], bool)
        or not isinstance(config["batch_size"], Integral)
        or config["batch_size"] < 1
    ):
        raise ValueError("batch_size must be a positive integer")
    if config["compute_dtype"] not in ("float32", "float16", "bfloat16"):
        raise ValueError("Unsupported compute_dtype")
    if config["storage_dtype"] not in ("float16", "float32"):
        raise ValueError("storage_dtype must be float16 or float32 (NumPy has no native bfloat16)")
    if config["attention_implementation"] not in ("eager", "sdpa", "flash_attention_2"):
        raise ValueError("Unsupported attention_implementation")
    if not isinstance(config["verify"], bool):
        raise ValueError("verify must be boolean")
    return config


def cache_plan(source, memory_config, *, episode_indices=None, storage_dtype="float16"):
    """Metadata-only dry run: no model import, image decoding or writes."""
    if storage_dtype not in ("float16", "float32"):
        raise ValueError("storage_dtype must be float16 or float32")
    source, info, _, jobs = load_source(source, episode_indices)
    config = (
        copy.deepcopy(validate_memory_config(memory_config))
        if isinstance(memory_config, Mapping)
        else load_memory_config(memory_config)
    )
    feature_spec = {
        "camera": "image",
        "num_views": 1,
        "token_per_image": config["token_per_image"],
        "image_dim": config["memory_feature"]["img"]["input_dim"],
        "pool_type": config["pool_type"],
        "storage_dtype": storage_dtype,
    }
    total_frames = sum(job["source_length"] for job in jobs)
    return {
        "cache_version": CACHE_VERSION,
        "source": str(source),
        "source_metadata_sha256": {
            name: file_sha256(source / "meta" / name) for name in ("info.json", "episodes.jsonl", "tasks.jsonl")
        },
        "source_fps": info["fps"],
        "memory_config": config,
        "feature_spec": feature_spec,
        "source_episodes": jobs,
        "total_frames": total_frames,
        "feature_bytes": total_frames * 16 * feature_spec["image_dim"] * np.dtype(storage_dtype).itemsize,
    }


def _read_episode(job):
    path = Path(job["parquet"])
    data = pd.read_parquet(
        path,
        columns=["image", "episode_index", "frame_index", "is_demo", "exec_start_idx", "state", "actions"],
        engine="pyarrow",
        use_threads=False,
    )
    length = job["source_length"]
    if len(data) != length or not (data["episode_index"] == job["source_episode_index"]).all():
        raise ValueError(f"Wrong source episode identity/length: {path}")
    if not np.array_equal(data["frame_index"], np.arange(length)):
        raise ValueError(f"Non-contiguous source frame indices: {path}")
    if not pd.api.types.is_bool_dtype(data["is_demo"]):
        raise ValueError(f"is_demo must be boolean: {path}")
    flags = data["is_demo"].to_numpy()
    execution = np.flatnonzero(~flags)
    if not len(execution):
        raise ValueError(f"No execution frames: {path}")
    start = int(execution[0])
    if flags[start:].any() or not (data["exec_start_idx"] == start).all():
        raise ValueError(f"Invalid demo prefix or exec_start_idx: {path}")
    vector_hashes = {}
    for key in ("state", "actions"):
        vectors = np.stack(data[key]).astype("<f4")
        if vectors.shape != (length, 8) or not np.isfinite(vectors).all():
            raise ValueError(f"Expected finite [frames,8] {key}: {path}")
        vector_hashes[key] = hashlib.sha256(vectors[start:].tobytes()).hexdigest()
    # Embedded image bytes are covered by the parquet hash. Also bind any local
    # image assets referenced through paths, so changing a PNG invalidates resume.
    assets = {}
    for value in data["image"]:
        if isinstance(value, dict):
            value = None if value.get("bytes") is not None else value.get("path")
        if isinstance(value, str):
            if "://" in value:
                raise ValueError("Only local image assets are supported")
            candidate = Path(value)
            candidates = (
                [candidate]
                if candidate.is_absolute()
                else [Path(job["source"]) / candidate, path.parent / candidate]
            )
            asset = next((p.resolve() for p in candidates if p.is_file()), None)
            if asset is None:
                raise FileNotFoundError(value)
            if str(asset) not in assets:
                assets[str(asset)] = file_sha256(asset)
    provenance = {
        "source_episode_index": job["source_episode_index"],
        "frames": length,
        "exec_start_idx": start,
        "demo_frames": start,
        "execution_frames": length - start,
        "source_parquet_sha256": file_sha256(path),
        "local_image_sha256": assets,
        "execution_vectors_sha256": vector_hashes,
    }
    return data, provenance


def _decode_batch(data, job, start, end):
    frames = []
    for index in range(start, end):
        image = decode_image(data["image"].iloc[index], Path(job["source"]), Path(job["parquet"]))
        if list(image.shape) != job["shapes"]["image"]:
            raise ValueError(f"Wrong image shape at source frame {index}: {job['parquet']}")
        frames.append(image)
    return np.stack(frames)


def load_cached_episode(root, episode_index, *, verify_checksum=True):
    """Open features as a read-only memmap and validate the per-episode record."""
    directory = Path(root) / f"episode_{episode_index:06d}"
    record = json.loads((directory / "record.json").read_text())
    path = directory / "features.npy"
    if record.get("cache_version") != CACHE_VERSION or record.get("source_episode_index") != episode_index:
        raise ValueError(f"Invalid episode cache identity: {directory}")
    if verify_checksum and file_sha256(path) != record["features_sha256"]:
        raise ValueError(f"Feature checksum mismatch: {path}")
    features = np.load(path, mmap_mode="r", allow_pickle=False)
    if list(features.shape) != record["shape"] or str(features.dtype) != record["storage_dtype"]:
        raise ValueError(f"Cached shape/dtype mismatch: {path}")
    return features, record


@contextmanager
def _cache_lock(output):
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / f".{output.name}.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another writer is using {output}") from error
        yield


def build_feature_cache(plan, output, encoder, *, batch_size=16, resume=False, verify=True, run_config=None):
    """Publish one episode atomically; a failed run can reuse completed episodes.

    encoder must expose a JSON signature and encode(uint8 NHWC)->float32 [N,16,D].
    This small interface allows cache integrity tests without model weights/GPU.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    output = Path(output).expanduser().resolve()
    source = Path(plan["source"])
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Cache output must not contain or be inside the source dataset")
    # Check that source metadata/stat identity still matches the preview.
    fresh = cache_plan(
        source,
        plan["memory_config"],
        episode_indices=[j["source_episode_index"] for j in plan["source_episodes"]],
        storage_dtype=plan["feature_spec"]["storage_dtype"],
    )
    if fresh != plan:
        raise ValueError("Source or plan changed before cache construction")
    encoder_identity = copy.deepcopy(encoder.signature)
    for name in ("pool_type", "token_per_image", "image_dim"):
        if encoder_identity[name] != plan["feature_spec"][name]:
            raise ValueError(f"Encoder/config mismatch: {name}")
    encoder_hash = json_fingerprint(encoder_identity)
    # Sampling budget, positions and DiT width do not affect cached visual tokens.
    identity = {
        key: plan[key]
        for key in ("cache_version", "source", "source_metadata_sha256", "source_episodes", "feature_spec")
    }
    identity["encoder_sha256"] = encoder_hash
    identity_hash = json_fingerprint(identity)
    with _cache_lock(output):
        if output.exists() and not resume:
            raise FileExistsError(f"Output exists; use --resume for this cache: {output}")
        if output.exists():
            manifest = json.loads((output / "manifest.json").read_text())
            if manifest.get("identity_sha256") != identity_hash:
                raise ValueError("Cache source, encoder weights, preprocessing or feature settings differ")
        else:
            output.mkdir()
            manifest = {
                **plan,
                "identity_sha256": identity_hash,
                "encoder_sha256": encoder_hash,
                "encoder": encoder_identity,
                "status": "building",
                "episodes": [],
            }
            write_json_atomic(output / "manifest.json", manifest)
            (output / "memory_config.yaml").write_text(yaml.safe_dump(plan["memory_config"], sort_keys=False))
            if run_config is not None:
                (output / "cache_config.yaml").write_text(yaml.safe_dump(run_config, sort_keys=False))
        records = []
        for number, job in enumerate(plan["source_episodes"], 1):
            data, provenance = _read_episode(job)
            episode_id = job["source_episode_index"]
            directory = output / f"episode_{episode_id:06d}"
            if directory.exists():
                _, record = load_cached_episode(output, episode_id)
                if record.get("source") != provenance or record.get("encoder_sha256") != encoder_hash:
                    raise ValueError(f"Completed episode provenance differs: {episode_id}")
                if record.get("feature_spec") != plan["feature_spec"]:
                    raise ValueError(f"Completed episode feature settings differ: {episode_id}")
                if verify and not record.get("verification", {}).get("passed"):
                    raise ValueError("Existing episode was not verified; use a new output for verification")
            else:
                record = _encode_episode(
                    data, job, provenance, output, encoder, encoder_hash, plan["feature_spec"], batch_size, verify
                )
            records.append(record)
            manifest["episodes"] = records
            manifest["status"] = "building"
            write_json_atomic(output / "manifest.json", manifest)
            print(
                f"[{number}/{len(plan['source_episodes'])}] source episode {episode_id}: {provenance['frames']} frames",
                flush=True,
            )
        manifest["status"] = "complete"
        manifest["completed_frames"] = sum(record["shape"][0] for record in records)
        write_json_atomic(output / "manifest.json", manifest)
        return manifest


def _encode_episode(data, job, provenance, output, encoder, encoder_hash, feature_spec, batch_size, verify):
    episode_id, frames = job["source_episode_index"], provenance["frames"]
    shape = (frames, 16, feature_spec["image_dim"])
    with tempfile.TemporaryDirectory(prefix=f".episode_{episode_id:06d}_", dir=output) as temporary:
        stage = Path(temporary) / "episode"
        stage.mkdir()
        path = stage / "features.npy"
        features = np.lib.format.open_memmap(path, mode="w+", dtype=feature_spec["storage_dtype"], shape=shape)
        for start in range(0, frames, batch_size):
            end = min(start + batch_size, frames)
            encoded = np.asarray(encoder.encode(_decode_batch(data, job, start, end)))
            if encoded.shape != (end - start, 16, shape[-1]) or not np.isfinite(encoded).all():
                raise ValueError(f"Invalid encoder output for episode {episode_id}, frame {start}")
            features[start:end] = encoded
            if not np.isfinite(features[start:end]).all():
                raise ValueError("Storage dtype overflow; choose storage_dtype=float32")
        features.flush()
        del features
        features = np.load(path, mmap_mode="r", allow_pickle=False)
        verification = {"passed": False, "source_frame_indices": [], "max_abs_error": None}
        if verify:
            indices = sorted({0, frames // 2, frames - 1})
            max_error = 0.0
            for batch_start in sorted({index // batch_size * batch_size for index in indices}):
                end = min(batch_start + batch_size, frames)
                direct = np.asarray(encoder.encode(_decode_batch(data, job, batch_start, end)))
                for index in (i for i in indices if batch_start <= i < end):
                    reference = direct[index - batch_start].astype(np.float32)
                    actual = features[index].astype(np.float32)
                    max_error = max(max_error, float(np.max(np.abs(actual - reference))))
                    tolerance = 5e-4 if feature_spec["storage_dtype"] == "float16" else 1e-6
                    if not np.allclose(actual, reference, rtol=tolerance, atol=tolerance):
                        raise ValueError(f"Cached/direct feature mismatch at source frame {index}")
            verification = {"passed": True, "source_frame_indices": indices, "max_abs_error": max_error}
        del features
        record = {
            "cache_version": CACHE_VERSION,
            "source_episode_index": episode_id,
            "source": provenance,
            "encoder_sha256": encoder_hash,
            "feature_spec": feature_spec,
            "shape": list(shape),
            "storage_dtype": feature_spec["storage_dtype"],
            "features_sha256": file_sha256(path),
            "verification": verification,
            "batch_size": batch_size,
        }
        write_json_atomic(stage / "record.json", record)
        stage.rename(output / f"episode_{episode_id:06d}")
    return record
