"""Bounded video caching for RA-VLA training with exact frame-index selection."""

from collections import OrderedDict

import numpy as np

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.utils.video import get_all_frames


class BoundedLeRobotVideoDataset(LeRobotSingleDataset):
    """Decode videos on demand, keeping at most cache_size camera videos per worker.

    Unlike the timestamp-based reader, select frames by the same integer indices
    as CachedLeRobotSingleDataset. No dataset-wide frames.pt is read or written.
    """

    def __init__(self, *args, cache_size=2, **kwargs):
        if cache_size < 1:
            raise ValueError("video cache_size must be positive")
        self._video_cache_size = cache_size
        self._video_cache = OrderedDict()
        super().__init__(*args, **kwargs)

    def get_video(self, trajectory_id, key, base_index):
        trajectory_index = self.get_trajectory_index(trajectory_id)
        length = int(self.trajectory_lengths[trajectory_index])
        indices = np.clip(self.delta_indices[key] + base_index, 0, length - 1)
        cache_key = (trajectory_id, key)
        frames = self._video_cache.pop(cache_key, None)
        if frames is None:
            # Evict before decoding, avoiding a transient cache_size + 1 videos.
            if len(self._video_cache) >= self._video_cache_size:
                self._video_cache.popitem(last=False)
            path = self.get_video_path(trajectory_id, key.removeprefix("video."))
            frames = get_all_frames(
                str(path), video_backend=self.video_backend,
                video_backend_kwargs=self.video_backend_kwargs,
            )
            if isinstance(frames, tuple):  # torchcodec also returns timestamps.
                frames = frames[0]
            if len(frames) != length:
                raise ValueError(
                    f"Video frame count mismatch for {path}: decoded {len(frames)}, metadata {length}"
                )
        self._video_cache[cache_key] = frames
        return frames[indices]
