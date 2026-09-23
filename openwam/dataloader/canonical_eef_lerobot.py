"""Strict LeRobot-v3 reader for already-materialized absolute EEF20 data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, ClassVar, Iterable, Optional

import numpy as np

from openwam.dataloader.bases.lerobot_v3_reader import LeRobotV3Reader
from openwam.dataloader.utils import get_cfg
from openwam.dataloader.utils.normalization import apply_normalization, load_stats_file, load_stats_metadata


_MISSING = object()
_ROT6D_SLICES = (slice(3, 9), slice(13, 19))


def canonical_schema_fingerprint(
    features: dict,
    *,
    state_key: str,
    action_key: str,
    camera_keys: Iterable[str],
) -> str:
    """Return a stable digest for the exact fields that define this reader."""

    keys = (state_key, action_key, *tuple(camera_keys))
    selected = {key: features[key] for key in keys}
    encoded = json.dumps(selected, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _inverse_normalization(arr: np.ndarray, stats: dict, mode: Optional[str]) -> np.ndarray:
    values = np.asarray(arr, dtype=np.float32)
    if mode in (None, "none", "null"):
        return values.copy()
    if mode == "z-score":
        return values * stats["std"] + stats["mean"]
    if mode == "min-max":
        return (values + 1.0) * 0.5 * (stats["max"] - stats["min"]) + stats["min"]
    if mode == "quantile":
        return (values + 1.0) * 0.5 * (stats["q99"] - stats["q01"]) + stats["q01"]
    raise ValueError(f"unsupported normalize_mode={mode!r}")


class CanonicalEEFLerobotDataset(LeRobotV3Reader):
    """Read absolute bimanual EEF20 fields and expose OpenWAM samples."""

    DATASET_NAME: ClassVar[str] = "CanonicalEEFLeRobot"
    ACTION_DIM: ClassVar[int] = 20
    DEFAULT_NORMALIZE_MODE: ClassVar[Optional[str]] = "quantile"
    NEEDED_COLS: ClassVar[tuple[str, ...]] = (
        "observation.states.eef",
        "actions.eef",
        "task_index",
    )

    def __init__(
        self,
        dataset_dir: str,
        *,
        state_key: str = "observation.states.eef",
        action_key: str = "actions.eef",
        head_camera: str = "observation.images.cam_high",
        left_wrist_camera: str = "observation.images.cam_left_wrist",
        right_wrist_camera: str = "observation.images.cam_right_wrist",
        video_root: Optional[str] = None,
        normalization_stats_path: str,
        episode_indices: Optional[Iterable[int]] = None,
        max_episodes: Optional[int] = 2,
        max_windows: Optional[int] = 8,
        return_trace: bool = False,
        **kwargs: Any,
    ):
        self._state_key = str(state_key)
        self._action_key = str(action_key)
        self._head_camera_key = str(head_camera)
        self._left_wrist_camera_key = str(left_wrist_camera)
        self._right_wrist_camera_key = str(right_wrist_camera)
        self._video_root = Path(video_root).expanduser().resolve() if video_root else Path(dataset_dir).resolve()
        self._stats_path = Path(normalization_stats_path).expanduser().resolve()
        self._requested_episode_indices = None if episode_indices is None else tuple(int(x) for x in episode_indices)
        self._max_episodes = None if max_episodes is None else int(max_episodes)
        self._max_windows = None if max_windows is None else int(max_windows)
        self._return_trace = bool(return_trace)

        if self._max_episodes is not None and self._max_episodes <= 0:
            raise ValueError("max_episodes must be positive or null")
        if self._max_windows is not None and self._max_windows <= 0:
            raise ValueError("max_windows must be positive or null")

        info_path = Path(dataset_dir) / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(info_path)
        info = json.loads(info_path.read_text(encoding="utf-8"))
        features = info.get("features", {})
        required = (
            self._state_key,
            self._action_key,
            self._head_camera_key,
            self._left_wrist_camera_key,
            self._right_wrist_camera_key,
        )
        missing = [key for key in required if key not in features]
        if missing:
            companion = " Use robotwin_traj_fullfield for RobotWin EEF/calibration metadata."
            raise ValueError(f"Canonical EEF fields/cameras missing from {info_path}: {missing}.{companion}")
        wrong_width = {
            key: features[key].get("shape")
            for key in (self._state_key, self._action_key)
            if list(features[key].get("shape", [])) != [20]
        }
        if wrong_width:
            raise ValueError(f"Canonical EEF state/action must be width 20, got {wrong_width}")

        self.NEEDED_COLS = (self._state_key, self._action_key, "task_index")
        self._schema_fingerprint = canonical_schema_fingerprint(
            features,
            state_key=self._state_key,
            action_key=self._action_key,
            camera_keys=(self._head_camera_key, self._left_wrist_camera_key, self._right_wrist_camera_key),
        )
        super().__init__(dataset_dir, **kwargs)
        if self._max_windows is not None:
            self._n_total = min(self._n_total, self._max_windows)

    def _resolve_cameras(self, info: dict):
        return self._head_camera_key, self._left_wrist_camera_key, self._right_wrist_camera_key

    def _resolve_video_file(self, camera: str, row) -> Path:
        return self._video_root / self._video_path_template.format(
            video_key=camera,
            chunk_index=int(row[f"videos/{camera}/chunk_index"]),
            file_index=int(row[f"videos/{camera}/file_index"]),
        )

    def _filter_episodes(self, eps_df):
        available = set(int(x) for x in eps_df["episode_index"].tolist())
        if self._requested_episode_indices is not None:
            missing = sorted(set(self._requested_episode_indices) - available)
            if missing:
                raise ValueError(f"requested episode_indices are absent from split: {missing}")
            requested = set(self._requested_episode_indices)
            eps_df = eps_df[eps_df["episode_index"].isin(requested)]
        if self._max_episodes is not None:
            eps_df = eps_df.iloc[: self._max_episodes]
        if eps_df.empty:
            raise ValueError("episode selection is empty")
        return eps_df.reset_index(drop=True)

    def _load_stats(self, info: dict) -> dict:
        if not self._stats_path.is_file():
            raise FileNotFoundError(f"canonical EEF normalization stats missing: {self._stats_path}")
        metadata = dict(load_stats_metadata(self._stats_path, action_mode="metadata"))
        selected = [int(x) for x in self._eps_df["episode_index"].tolist()]
        expected = {
            "dataset_dir": str(self._dataset_dir.resolve()),
            "state_key": self._state_key,
            "action_key": self._action_key,
            "episode_indices": selected,
            "state_width": 20,
            "action_width": 20,
            "schema_fingerprint": self._schema_fingerprint,
        }
        mismatches = {key: (metadata.get(key), value) for key, value in expected.items() if metadata.get(key) != value}
        if mismatches:
            raise ValueError(f"normalization stats provenance mismatch for {self._stats_path}: {mismatches}")
        self._stats_metadata = metadata
        self._state_stats = load_stats_file(
            self._stats_path,
            action_mode="eef_state",
            normalize_mode=self._normalize_mode,
            dim=20,
        )
        return load_stats_file(
            self._stats_path,
            action_mode="eef",
            normalize_mode=self._normalize_mode,
            dim=20,
        )

    @staticmethod
    def _validated_eef(values: Any, *, field: str) -> np.ndarray:
        arr = np.stack([np.asarray(value, dtype=np.float32) for value in values])
        if arr.ndim != 2 or arr.shape[1] != 20:
            raise ValueError(f"{field} must have shape (N, 20), got {arr.shape}")
        if not np.isfinite(arr).all():
            raise ValueError(f"{field} contains non-finite EEF values")
        for block in _ROT6D_SLICES:
            a = arr[:, block.start : block.start + 3]
            b = arr[:, block.start + 3 : block.stop]
            if np.any(np.linalg.norm(a, axis=1) < 1e-6) or np.any(np.linalg.norm(b, axis=1) < 1e-6):
                raise ValueError(f"{field} contains degenerate rot6d columns")
        return arr

    def _action_20d(self, win):
        raw = self._validated_eef(win[self._action_key].tolist(), field=self._action_key)
        normalized = apply_normalization(raw, self._normalization_stats, self._normalize_mode).astype(np.float32)
        if self._return_trace:
            self._trace_action_raw = raw
            self._trace_action_normalized = normalized
        return normalized

    def _proprio_20d(self, win):
        raw = self._validated_eef(win[self._state_key].iloc[:1].tolist(), field=self._state_key)
        normalized = apply_normalization(raw, self._state_stats, self._normalize_mode).astype(np.float32)
        if self._return_trace:
            self._trace_state_raw = raw
            self._trace_state_normalized = normalized
        return normalized

    def __getitem__(self, idx: int) -> dict:
        # This reader is used for auditable smoke/eval data. Contract violations
        # must identify the requested window rather than silently retrying another.
        return self._getitem_impl(idx)

    def _getitem_impl(self, idx: int) -> dict:
        ep_local = int(np.searchsorted(self._cum_n_starts, idx, side="right") - 1)
        offset = int(self._ep_valid_start[ep_local]) + (
            idx - int(self._cum_n_starts[ep_local])
        ) * self._window_stride
        row = self._eps_df.iloc[ep_local]
        sample = super()._getitem_impl(idx)
        if not self._return_trace:
            return sample

        raw_action = np.zeros((self._num_frames - 1, 20), dtype=np.float32)
        n = min(len(self._trace_action_raw), self._num_frames - 1)
        raw_action[:n] = self._trace_action_raw[:n]
        sample["trace"] = {
            "dataset_dir": str(self._dataset_dir.resolve()),
            "dataset_id": self._dataset_id,
            "episode_index": int(row["episode_index"]),
            "window_offset": int(offset),
            "raw_frame_indices": (offset + np.arange(self._num_frames)).tolist(),
            "video_frame_indices": (offset + self._video_sample_indices).tolist(),
            "state_key": self._state_key,
            "action_key": self._action_key,
            "raw_action_eef20": raw_action,
            "normalized_action_eef20": self.gather_unified_eef(sample["action"].numpy()),
            "unified_action80": sample["action"].numpy().copy(),
            "raw_proprio_eef20": self._trace_state_raw.copy(),
            "normalized_proprio_eef20": self._trace_state_normalized.copy(),
            "unified_proprio80": sample["proprio"].numpy().copy(),
            "stats_path": str(self._stats_path),
            "stats_metadata": dict(self._stats_metadata),
        }
        return sample

    def gather_unified_eef(self, array: np.ndarray) -> np.ndarray:
        values = np.asarray(array)
        if values.shape[-1] != 80:
            raise ValueError(f"unified array must have width 80, got {values.shape}")
        return np.concatenate([values[..., 0:10], values[..., 34:44]], axis=-1)

    def denormalize_eef(self, array: np.ndarray, stream: str = "action") -> np.ndarray:
        if stream not in {"action", "state"}:
            raise ValueError("stream must be 'action' or 'state'")
        stats = self._normalization_stats if stream == "action" else self._state_stats
        return _inverse_normalization(array, stats, self._normalize_mode).astype(np.float32)

    @classmethod
    def from_config(cls, config, split: str = "train") -> "CanonicalEEFLerobotDataset":
        dataset_dir = get_cfg(config, "dataset_dir")
        stats_path = get_cfg(config, "normalization_stats_path")
        if dataset_dir is None:
            raise ValueError("CanonicalEEFLerobotDataset: missing dataset_dir")
        if stats_path is None:
            raise ValueError("CanonicalEEFLerobotDataset: missing normalization_stats_path")
        keys = (
            *LeRobotV3Reader.CONFIG_KEYS,
            "state_key",
            "action_key",
            "head_camera",
            "left_wrist_camera",
            "right_wrist_camera",
            "video_root",
            "episode_indices",
            "max_episodes",
            "max_windows",
            "return_trace",
            "dataset_id",
        )
        kwargs: dict[str, Any] = {"split": split, "normalization_stats_path": stats_path}
        for key in keys:
            value = get_cfg(config, key, _MISSING)
            if value is not _MISSING:
                kwargs[key] = value
        return cls(str(dataset_dir), **kwargs)


__all__ = ["CanonicalEEFLerobotDataset", "canonical_schema_fingerprint"]
