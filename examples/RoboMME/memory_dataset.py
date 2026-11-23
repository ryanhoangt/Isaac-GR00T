"""Execution-only GR00T samples paired with causal, full-episode visual memory.

This opt-in LeRobotSingleDataset subclass also works in LeRobotMixtureDataset:
memory is attached in get_step_data, which the mixture loader calls directly.
No new training samples or normalization statistics are created from the cache.
"""

from collections import OrderedDict
from collections.abc import Mapping
import copy
import hashlib
import json
from numbers import Integral
import os
from pathlib import Path

import numpy as np
import pandas as pd

from examples.RoboMME.feature_cache import CACHE_VERSION, json_fingerprint, load_cached_episode
from examples.RoboMME.memory_config import (
    DEFAULT_MEMORY_CONFIG,
    load_memory_config,
    prepare_frame_sampling,
    validate_memory_config,
)
from examples.RoboMME.memory_data_config import RoboMMEMemoryDataConfig, RoboMMEMemoryTransform
from gr00t.data.dataset import LeRobotSingleDataset


class RoboMMEMemoryDataset(LeRobotSingleDataset):
    def __init__(
        self,
        dataset_path,
        modality_configs=None,
        embodiment_tag="new_embodiment",
        video_backend="torchcodec",
        video_backend_kwargs=None,
        transforms=None,
        *,
        memory_cache,
        memory_config=DEFAULT_MEMORY_CONFIG,
        max_open_episodes=8,
    ):
        if (
            isinstance(max_open_episodes, bool)
            or not isinstance(max_open_episodes, Integral)
            or max_open_episodes < 1
        ):
            raise ValueError("max_open_episodes must be a positive integer")
        self.memory_config = (
            copy.deepcopy(validate_memory_config(memory_config))
            if isinstance(memory_config, Mapping)
            else load_memory_config(memory_config)
        )
        self.memory_cache = Path(memory_cache).expanduser().resolve(strict=True)
        self.max_open_episodes = max_open_episodes
        self._feature_maps = OrderedDict()
        self._verified_files = {}
        self._owner_pid = os.getpid()
        config = RoboMMEMemoryDataConfig()
        if transforms is None:
            transforms = config.transform()
        if not transforms.transforms or not isinstance(transforms.transforms[-1], RoboMMEMemoryTransform):
            raise ValueError("Use RoboMMEMemoryDataConfig transforms to preserve memory fields")
        super().__init__(
            dataset_path=dataset_path,
            modality_configs=modality_configs if modality_configs is not None else config.modality_config(),
            embodiment_tag=embodiment_tag,
            video_backend=video_backend,
            video_backend_kwargs=video_backend_kwargs,
            transforms=transforms,
        )
        self._cache_manifest = json.loads((self.memory_cache / "manifest.json").read_text())
        manifest = self._cache_manifest
        if manifest.get("status") != "complete" or manifest.get("cache_version") != CACHE_VERSION:
            raise ValueError("Expected a complete visual-feature cache with the supported version")
        if manifest.get("encoder_sha256") != json_fingerprint(manifest["encoder"]):
            raise ValueError("Cache encoder fingerprint is inconsistent")
        self.memory_encoder_sha256 = manifest["encoder_sha256"]
        self.memory_encoder_signature = copy.deepcopy(manifest["encoder"])
        spec = manifest["feature_spec"]
        expected = {
            "camera": "image",
            "num_views": 1,
            "token_per_image": 16,
            "image_dim": self.memory_config["memory_feature"]["img"]["input_dim"],
            "pool_type": self.memory_config["pool_type"],
        }
        if any(spec.get(key) != value for key, value in expected.items()):
            raise ValueError("Memory config and cached visual feature settings differ")
        if any(manifest["encoder"].get(key) != spec[key] for key in ("pool_type", "image_dim", "token_per_image")):
            raise ValueError("Cache encoder and visual feature settings are inconsistent")
        if spec.get("storage_dtype") not in ("float16", "float32"):
            raise ValueError("Unsupported cached feature dtype")
        if float(self.lerobot_info_meta["fps"]) != float(manifest["source_fps"]):
            raise ValueError("Training dataset and cache source FPS differ")
        manifest_records = manifest["episodes"]
        records = {record["source_episode_index"]: record for record in manifest_records}
        if len(records) != len(manifest_records):
            raise ValueError("Duplicate source episode IDs in cache manifest")
        self._episode_bindings = {}
        self._cache_records = {}
        seen_source_ids = set()
        for trajectory_id, length in zip(self.trajectory_ids, self.trajectory_lengths):
            source_id, source_frames = self._validate_execution_episode(int(trajectory_id), int(length), records)
            if source_id in seen_source_ids:
                raise ValueError("A source episode appears more than once in the training dataset")
            seen_source_ids.add(source_id)
            self._episode_bindings[int(trajectory_id)] = (source_id, source_frames)
            self._cache_records[source_id] = records[source_id]

    def _validate_execution_episode(self, trajectory_id, length, records):
        path = self.dataset_path / self.data_path_pattern.format(
            episode_chunk=self.get_episode_chunk(trajectory_id),
            episode_index=trajectory_id,
        )
        columns = [
            "episode_index",
            "frame_index",
            "source_episode_index",
            "source_frame_index",
            "source_exec_start_idx",
            "is_demo",
            "state",
            "actions",
        ]
        data = pd.read_parquet(path, columns=columns, engine="pyarrow", use_threads=False)
        if len(data) != length or not (data["episode_index"] == trajectory_id).all():
            raise ValueError(f"Invalid converted episode identity/length: {path}")
        if not np.array_equal(data["frame_index"], np.arange(length)):
            raise ValueError(f"Converted frame indices must be contiguous: {path}")
        for name in ("source_episode_index", "source_frame_index", "source_exec_start_idx"):
            if data[name].dtype.kind not in "iu":
                raise ValueError(f"{name} must contain integer indices: {path}")
        source_ids = data["source_episode_index"].unique()
        if len(source_ids) != 1:
            raise ValueError(f"Converted episode contains multiple source episode IDs: {path}")
        source_id = int(source_ids[0])
        if source_id not in records:
            raise ValueError(f"Missing cached source episode {source_id} for {path}")
        record = records[source_id]
        episode_record_path = self.memory_cache / f"episode_{source_id:06d}/record.json"
        if json.loads(episode_record_path.read_text()) != record:
            raise ValueError(f"Cache episode record differs from manifest: {source_id}")
        provenance = record["source"]
        shape = [provenance["frames"], 16, self.memory_config["memory_feature"]["img"]["input_dim"]]
        if (
            record["shape"] != shape
            or record["encoder_sha256"] != self.memory_encoder_sha256
            or record["feature_spec"] != self._cache_manifest["feature_spec"]
            or record["storage_dtype"] != self._cache_manifest["feature_spec"]["storage_dtype"]
            or provenance["source_episode_index"] != source_id
        ):
            raise ValueError(f"Inconsistent cached source episode metadata: {source_id}")
        start = provenance["exec_start_idx"]
        source_frames = data["source_frame_index"].to_numpy(dtype=np.int64, copy=True)
        if (
            start < 0
            or not np.array_equal(source_frames, np.arange(start, provenance["frames"]))
            or length != provenance["execution_frames"]
            or not (data["source_exec_start_idx"] == start).all()
        ):
            raise ValueError(f"Converted execution rows do not match the full cached execution suffix: {path}")
        if not pd.api.types.is_bool_dtype(data["is_demo"]) or data["is_demo"].any():
            raise ValueError(f"Training samples must contain only execution rows: {path}")
        for name in ("state", "actions"):
            vectors = np.stack(data[name]).astype("<f4")
            if vectors.shape != (length, 8) or not np.isfinite(vectors).all():
                raise ValueError(f"Expected finite [execution_frames,8] {name}: {path}")
            if hashlib.sha256(vectors.tobytes()).hexdigest() != provenance["execution_vectors_sha256"][name]:
                raise ValueError(f"Execution {name} differ from the cached source: {path}")
        return source_id, source_frames

    def _close_maps(self):
        for features in self._feature_maps.values():
            features._mmap.close()
        self._feature_maps.clear()

    def close(self):
        """Release this process's open memmaps; other DataLoader workers own theirs."""
        self._close_maps()
        self._verified_files.clear()

    def __getstate__(self):
        # A spawned worker must reopen files instead of serializing array payloads.
        state = self.__dict__.copy()
        state["_feature_maps"] = OrderedDict()
        state["_verified_files"] = {}
        state["_owner_pid"] = None
        return state

    def _get_memory_features(self, source_id):
        if os.getpid() != self._owner_pid:
            self.close()
            self._owner_pid = os.getpid()
        path = self.memory_cache / f"episode_{source_id:06d}/features.npy"
        stat = path.stat()
        identity = (stat.st_size, stat.st_mtime_ns)
        if self._verified_files.get(source_id) != identity:
            previous = self._feature_maps.pop(source_id, None)
            if previous is not None:
                previous._mmap.close()
            features, record = load_cached_episode(self.memory_cache, source_id)
            if record != self._cache_records[source_id]:
                features._mmap.close()
                raise ValueError(f"Cache episode metadata changed after initialization: {source_id}")
            self._verified_files[source_id] = identity
            self._feature_maps[source_id] = features
        elif source_id not in self._feature_maps:
            features, record = load_cached_episode(self.memory_cache, source_id, verify_checksum=False)
            if record != self._cache_records[source_id]:
                features._mmap.close()
                raise ValueError(f"Cache episode metadata changed after initialization: {source_id}")
            self._feature_maps[source_id] = features
        self._feature_maps.move_to_end(source_id)
        while len(self._feature_maps) > self.max_open_episodes:
            _, evicted = self._feature_maps.popitem(last=False)
            evicted._mmap.close()
        return self._feature_maps[source_id]

    def get_step_data(self, trajectory_id, base_index):
        source_id, frames = self._episode_bindings[int(trajectory_id)]
        if (
            isinstance(base_index, bool)
            or not isinstance(base_index, Integral)
            or not 0 <= base_index < len(frames)
        ):
            raise IndexError("Execution step is outside this trajectory")
        data = super().get_step_data(trajectory_id, base_index)
        data.update(
            prepare_frame_sampling(
                self._get_memory_features(source_id), int(frames[base_index]), self.memory_config
            )
        )
        return data
