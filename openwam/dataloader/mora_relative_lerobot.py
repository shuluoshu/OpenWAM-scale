"""Mora next-state reader: TCP-local EEF poses plus independent arm joints."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from openwam.dataloader.bases.lerobot_v3_reader import LeRobotV3Reader
from openwam.dataloader.canonical_eef_lerobot import CanonicalEEFLerobotDataset, _inverse_normalization
from openwam.dataloader.utils import get_cfg
from openwam.dataloader.utils.mora_relative import MoraHeadTCPRepresentation, MoraRelativeRepresentation
from openwam.dataloader.utils.normalization import apply_normalization, materialize_eef_stats

SOURCE_KEYS = ("state_key", "joint_state_key", "tcp_pose_keys", "tcp_from_eef", "joint_indices")


class MoraRelativeJointDataset(CanonicalEEFLerobotDataset):
    """Append joints at 80:94 and, for dual-frame mode, head poses at 94:112.

    Pose and joint targets refer to the window's current state. Gripper targets
    remain absolute. Missing TCP sources disable pose supervision per side.
    """
    DATASET_NAME = "MoraRelativeJointLeRobot"
    ACTION_DIM = 34
    UNIFIED_ACTION_DIM = 94

    def __init__(self, dataset_dir, *, normalization_stats_path, episode_indices=None,
                 max_episodes=2, max_windows=8, return_trace=False, video_root=None,
                 head_camera="observation.images.cam_high",
                 left_wrist_camera="observation.images.cam_left_wrist",
                 right_wrist_camera="observation.images.cam_right_wrist",
                 action_representation="tcp_relative_joint94",
                 head_world_to_camera_key="observation.states.camera.world_to_camera",
                 primary_camera=None, tcp_reference_to_world_key=None, fk_supervision=None, **kwargs):
        source = {key: kwargs.pop(key) for key in SOURCE_KEYS if key in kwargs}
        kwargs.pop("action_key", None)  # precomputed action columns are never read
        info = json.loads((Path(dataset_dir) / "meta/info.json").read_text())
        if action_representation == "head_tcp_relative_joint112":
            head_available = head_camera is not None and head_camera in info["features"]
            self.representation = MoraHeadTCPRepresentation(
                info, head_world_to_camera_key=head_world_to_camera_key,
                head_camera_available=head_available,
                tcp_reference_to_world_key=tcp_reference_to_world_key, **source)
            if len(kwargs.get("head_camera_choices") or []) > 1:
                raise ValueError("head-relative targets require a fixed head image and calibration")
            if not head_available:
                head_camera = primary_camera
                if head_camera is None:
                    raise ValueError("missing head camera: set primary_camera to an available conditioning video")
        elif action_representation == "tcp_relative_joint94":
            self.representation = MoraRelativeRepresentation(info, **source)
        else:
            raise ValueError(f"unknown relative representation: {action_representation}")
        self.ACTION_DIM = self.representation.raw_dim
        self.UNIFIED_ACTION_DIM = self.representation.unified_dim
        self._scatter = self.representation.scatter
        self._state_key = self.representation.state_key
        self._action_key = "next_state"
        self._fk_builder = None
        if fk_supervision is not None:
            from openwam.dataloader.utils.fk_supervision import FKSupervisionBuilder

            self._fk_builder = FKSupervisionBuilder(fk_supervision, self.representation, info)
        self.NEEDED_COLS = tuple(dict.fromkeys((*self.representation.columns, "task_index",
            *(self._fk_builder.columns if self._fk_builder is not None else []))))
        self._head_camera_key = head_camera
        self._left_wrist_camera_key = left_wrist_camera
        self._right_wrist_camera_key = right_wrist_camera
        for camera in (head_camera, left_wrist_camera, right_wrist_camera):
            if camera is not None and camera not in info["features"]:
                raise ValueError(f"camera {camera!r} absent; explicitly set unavailable wrist cameras to null")
        self._video_root = Path(video_root or dataset_dir).expanduser().resolve()
        self._stats_path = Path(normalization_stats_path).expanduser().resolve()
        self._requested_episode_indices = None if episode_indices is None else tuple(map(int, episode_indices))
        self._max_episodes = max_episodes
        self._max_windows = max_windows
        self._return_trace = return_trace
        if any(x is not None and x <= 0 for x in (max_episodes, max_windows)):
            raise ValueError("max_episodes/max_windows must be positive or null")
        if not kwargs.get("unify_action", True):
            raise ValueError("relative representations require unify_action=true")
        kwargs.setdefault("unify_action", True)
        kwargs.setdefault("unify_action_map", ["0-9", "34-43", "80-93"] +
                          (["94-111"] if self.UNIFIED_ACTION_DIM == 112 else []))
        LeRobotV3Reader.__init__(self, dataset_dir, **kwargs)
        if not np.array_equal(self._unify_dst_index, self._scatter) or not np.array_equal(self._unify_state_dst_index, self._scatter):
            raise ValueError("relative representation requires its fixed scatter layout")
        if max_windows is not None:
            self._n_total = min(self._n_total, max_windows)

    def _load_stats(self, info):
        payload = np.load(self._stats_path, allow_pickle=True).item()
        metadata = payload["metadata"]
        expected = {**self.representation.contract, "dataset_dir": str(self._dataset_dir.resolve()),
                    "episode_indices": self._eps_df.episode_index.astype(int).tolist(),
                    "num_frames": self._num_frames, "window_stride": self._window_stride, "split": self._split}
        differences = {k: (metadata.get(k), v) for k, v in expected.items() if metadata.get(k) != v}
        if differences:
            raise ValueError(f"relative stats provenance mismatch: {differences}; regenerate with prepare_openwam_relative_stats.py")
        self._stats_metadata = metadata
        self._state_stats = materialize_eef_stats(payload["state"], self._normalize_mode, dim=self.representation.raw_dim, strict_minmax=True)
        return materialize_eef_stats(payload["action"], self._normalize_mode, dim=self.representation.raw_dim, strict_minmax=True)

    def _action_20d(self, win):
        columns = {key: win[key].tolist() for key in self.representation.columns}
        columns["_rows"] = np.arange(len(win))  # permits video-only windows
        self._raw_action, self._raw_action_mask, self._raw_state, self._raw_state_mask = self.representation.encode_window(columns)
        if self._fk_builder is not None:
            gt_columns = {key: win[key].tolist() for key in self._fk_builder.columns}
            self._fk_payload = self._fk_builder.build(
                gt_columns, self._raw_action, self._raw_action_mask, self._raw_state,
                self._raw_state_mask, self._normalization_stats, self._normalize_mode, self._num_frames-1)
        result = apply_normalization(self._raw_action, self._normalization_stats, self._normalize_mode).astype(np.float32)
        result[~self._raw_action_mask] = 0
        return result

    def _proprio_20d(self, win):
        result = apply_normalization(self._raw_state, self._state_stats, self._normalize_mode).astype(np.float32)
        result[~self._raw_state_mask] = 0
        return result

    def _finalize_action(self, values, actual_raw_len):
        action, mask = LeRobotV3Reader._finalize_action(self, values, actual_raw_len - 1)
        dynamic = np.zeros_like(mask)
        n = min(len(self._raw_action_mask), len(mask))
        dynamic[:n, self._scatter] = self._raw_action_mask[:n]
        mask &= dynamic
        action[~mask] = 0
        return action, mask

    def _finalize_proprio(self, values):
        state, mask = LeRobotV3Reader._finalize_proprio(self, values)
        dynamic = np.zeros_like(mask)
        dynamic[:, self._scatter] = self._raw_state_mask
        mask &= dynamic
        state[~mask] = 0
        return state, mask

    def _getitem_impl(self, idx):
        sample = LeRobotV3Reader._getitem_impl(self, idx)
        if self._fk_builder is not None:
            sample["fk_supervision"] = self._fk_payload
        if self._return_trace:
            ep = int(np.searchsorted(self._cum_n_starts, idx, side="right") - 1)
            offset = int(self._ep_valid_start[ep]) + (idx - int(self._cum_n_starts[ep])) * self._window_stride
            raw = np.zeros((self._num_frames - 1, self.representation.raw_dim), np.float32)
            n = min(len(raw), len(self._raw_action))
            raw[:n] = self._raw_action[:n]
            sample["trace"] = {
                **self.representation.contract, "dataset_dir": str(self._dataset_dir.resolve()),
                "dataset_id": self._dataset_id, "episode_index": int(self._eps_df.iloc[ep].episode_index),
                "window_offset": offset, "raw_frame_indices": (offset + np.arange(self._num_frames)).tolist(),
                "video_frame_indices": (offset + self._video_sample_indices).tolist(),
                "action_target_state_key": self._state_key,
                "action_target_frame_indices": [offset + i + 1 if i < n else None for i in range(len(raw))],
                "raw_action_eef20": raw[:, :20], "raw_action_joint14": raw[:, 20:34],
                "normalized_action_eef20": self.gather_unified_eef(sample["action"].numpy()),
                "raw_proprio_eef20": self._raw_state[:, :20].copy(),
                "raw_proprio_joint14": self._raw_state[:, 20:34].copy(),
                "normalized_proprio_eef20": self.gather_unified_eef(sample["proprio"].numpy()),
                f"unified_action{self.UNIFIED_ACTION_DIM}": sample["action"].numpy().copy(),
                f"unified_proprio{self.UNIFIED_ACTION_DIM}": sample["proprio"].numpy().copy(),
                "stats_path": str(self._stats_path), "stats_metadata": self._stats_metadata,
            }
            if self.UNIFIED_ACTION_DIM == 112:
                sample["trace"]["raw_action_head18"] = raw[:, 34:].copy()
                sample["trace"]["raw_proprio_head18"] = self._raw_state[:, 34:].copy()
        return sample

    def gather_unified_eef(self, array):
        values = np.asarray(array)
        if values.shape[-1] != self.UNIFIED_ACTION_DIM:
            raise ValueError(f"expected unified width {self.UNIFIED_ACTION_DIM}, got {values.shape}")
        return values[..., self._scatter[:20]]

    def denormalize_eef(self, array, stream="action"):
        if stream not in {"action", "state"}:
            raise ValueError("stream must be action or state")
        stats = self._normalization_stats if stream == "action" else self._state_stats
        return _inverse_normalization(array, {k: v[:20] for k, v in stats.items()}, self._normalize_mode).astype(np.float32)

    def denormalize_head(self, array, stream="action"):
        """Invert normalization of the left9/right9 head-reference TCP block."""
        if self.UNIFIED_ACTION_DIM != 112 or stream not in {"action", "state"}:
            raise ValueError("head poses require the 112D representation and action/state stream")
        stats = self._normalization_stats if stream == "action" else self._state_stats
        return _inverse_normalization(array, {k: v[34:52] for k, v in stats.items()}, self._normalize_mode).astype(np.float32)

    @classmethod
    def from_config(cls, config, split="train"):
        keys = (*LeRobotV3Reader.CONFIG_KEYS, *SOURCE_KEYS, "normalization_stats_path", "episode_indices",
                "max_episodes", "max_windows", "return_trace", "dataset_id", "head_camera",
                "left_wrist_camera", "right_wrist_camera", "video_root", "max_hours", "subsample_seed",
                "action_representation", "head_world_to_camera_key", "primary_camera", "tcp_reference_to_world_key",
                "fk_supervision")
        missing = object()
        kwargs = {key: value for key in keys if (value := get_cfg(config, key, missing)) is not missing}
        return cls(get_cfg(config, "dataset_dir"), split=split, **kwargs)
