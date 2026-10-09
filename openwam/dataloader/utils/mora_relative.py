"""Explicit source contract for Mora TCP-relative poses and 14 arm joints.

Raw layout is EEF20 + left7 + right7; scatter retains OpenWAM80 positions
and appends the arm joints. No human wrist or robot flange is guessed to be a
TCP. A pose source or a fixed EEF-to-TCP transform must be declared per side.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
from scipy.spatial.transform import Rotation

RAW_DIM = 34
UNIFIED_DIM = 94
SCATTER = np.array([*range(10), *range(34, 44), *range(80, 94)], dtype=np.int64)
SIDES = ("left", "right")


def _rotation(values):
    a, b = values[..., :3], values[..., 3:6]
    na = np.linalg.norm(a, axis=-1, keepdims=True)
    a = a / np.maximum(na, 1e-8)
    b = b - (a * b).sum(axis=-1, keepdims=True) * a
    nb = np.linalg.norm(b, axis=-1, keepdims=True)
    b = b / np.maximum(nb, 1e-8)
    valid = np.isfinite(values).all(axis=-1) & (na[..., 0] > 1e-6) & (nb[..., 0] > 1e-6)
    return np.stack((a, b, np.cross(a, b)), axis=-1), valid


def _rot6d(matrix):
    return np.concatenate((matrix[..., :, 0], matrix[..., :, 1]), axis=-1)


class MoraRelativeRepresentation:
    """Map declared native state columns to masked raw34 targets and proprio.

    TCP columns contain xyz + quaternion(wxyz). Fixed transforms map TCP-local
    coordinates into the source EEF frame. Joint indices exclude grippers and
    are ordered within each arm. Missing optional columns yield masked zeros;
    malformed mappings raise ValueError. Returned arrays are float32/bool.
    """

    raw_dim = RAW_DIM
    unified_dim = UNIFIED_DIM
    scatter = SCATTER

    def __init__(self, info, *, state_key="observation.states.eef",
                 joint_state_key="observation.states.joint", tcp_pose_keys=None,
                 tcp_from_eef=None, joint_indices=None):
        self.features = info.get("features", {})
        self.state_key = state_key
        self.joint_state_key = joint_state_key
        self.tcp_pose_keys = dict(tcp_pose_keys or {})
        self.tcp_from_eef = {k: np.asarray(v, dtype=np.float32) for k, v in (tcp_from_eef or {}).items()}
        for mapping in (self.tcp_pose_keys, self.tcp_from_eef, joint_indices or {}):
            if set(mapping) - set(SIDES):
                raise ValueError("source mappings may contain only left/right sides")
        for side, transform in self.tcp_from_eef.items():
            if side in self.tcp_pose_keys:
                raise ValueError(f"{side}: choose tcp_pose_keys or tcp_from_eef, not both")
            if (transform.shape != (4, 4) or not np.isfinite(transform).all()
                    or not np.allclose(transform[3], [0, 0, 0, 1])
                    or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5)
                    or not np.isclose(np.linalg.det(transform[:3, :3]), 1, atol=1e-5)):
                raise ValueError(f"tcp_from_eef.{side} must be a rigid 4x4 transform")
        self.columns = []
        if state_key in self.features:
            if self.features[state_key].get("shape") not in ([10], [20]):
                raise ValueError(f"{state_key} must contain EEF10 or EEF20")
            self.columns.append(state_key)
        for side, key in self.tcp_pose_keys.items():
            if key not in self.features or self.features[key].get("shape") != [7]:
                raise ValueError(f"tcp_pose_keys.{side}={key!r} must declare shape [7]")
            self.columns.append(key)
        self.joint_indices = {side: [] for side in SIDES}
        if joint_state_key in self.features:
            feature = self.features[joint_state_key]
            shape = feature.get("shape", [])
            if len(shape) != 1:
                raise ValueError(f"{joint_state_key} must be a vector")
            width = shape[0]
            names = feature.get("names", [])
            if isinstance(names, dict):
                names = names.get("motors", [])
            if joint_indices is None:
                if len(names) != width:
                    raise ValueError("joint_indices required when joint schema has no complete motor names")
                joint_indices = {side: [i for i, name in enumerate(names)
                                       if str(name).startswith(side + "_") and "grip" not in str(name).lower()]
                                 for side in SIDES}
                if not any(joint_indices.values()):
                    raise ValueError("joint_indices required: cannot infer left/right arm ordering")
            all_indices = []
            for side in SIDES:
                indices = list(joint_indices.get(side, []))
                if len(indices) > 7 or any(not isinstance(i, int) or i < 0 or i >= width for i in indices):
                    raise ValueError(f"joint_indices.{side} must contain up to seven valid source indices")
                if names and any("grip" in str(names[i]).lower() for i in indices):
                    raise ValueError(f"joint_indices.{side} must exclude grippers")
                self.joint_indices[side] = indices
                all_indices.extend(indices)
            if len(set(all_indices)) != len(all_indices):
                raise ValueError("joint_indices must not repeat source indices")
            self.columns.append(joint_state_key)
        elif joint_indices:
            raise ValueError(f"joint_indices supplied but {joint_state_key} is absent")
        self.columns = list(dict.fromkeys(self.columns))
        self.contract = {
            "representation": "tcp_relative_joint94", "action_target_source": "next_state",
            "action_target_offset": 1, "anchor": "window_start", "state_key": state_key,
            "joint_state_key": joint_state_key, "tcp_pose_keys": self.tcp_pose_keys,
            "tcp_from_eef": {k: v.tolist() for k, v in self.tcp_from_eef.items()},
            "joint_indices": self.joint_indices, "raw_width": RAW_DIM, "unified_width": UNIFIED_DIM,
            "joint_target": "relative", "missing_tcp": "mask_pose", "gripper_target": "absolute",
        }
        schema = {key: self.features[key] for key in self.columns}
        digest = json.dumps({"contract": self.contract, "schema": schema}, sort_keys=True).encode()
        self.contract["schema_fingerprint"] = hashlib.sha256(digest).hexdigest()

    def _absolute_values(self, columns):
        """Return source TCP/gripper/joint values and per-component masks.

        Every future row is compared with row zero, including the second and
        later targets. Non-finite joints and invalid poses are independently
        masked; padding and unified scatter are handled by the reader.
        """
        n = len(next(iter(columns.values())))
        if n < 2:
            raise ValueError("next-state targets require at least two state rows")
        values = np.zeros((n, RAW_DIM), dtype=np.float32)
        valid = np.zeros((n, RAW_DIM), dtype=bool)
        eef = np.asarray(columns[self.state_key], dtype=np.float32) if self.state_key in columns else None
        joints = np.asarray(columns[self.joint_state_key], dtype=np.float32) if self.joint_state_key in columns else None
        for side_id, side in enumerate(SIDES):
            start = side_id * 10
            if eef is not None and eef.shape[1] >= start + 10:
                values[:, start + 9] = eef[:, start + 9]
                valid[:, start + 9] = np.isfinite(eef[:, start + 9])
            key = self.tcp_pose_keys.get(side)
            if key is not None:
                pose = np.asarray(columns[key], dtype=np.float32)
                pose_valid = np.isfinite(pose).all(axis=-1) & (np.linalg.norm(pose[:, 3:], axis=-1) > 1e-6)
                rotation = np.tile(np.eye(3, dtype=np.float32), (n, 1, 1))
                rotation[pose_valid] = Rotation.from_quat(pose[pose_valid][:, [4, 5, 6, 3]]).as_matrix()
                xyz = pose[:, :3]
            elif side in self.tcp_from_eef and eef is not None and eef.shape[1] >= start + 10:
                source = eef[:, start:start + 9]
                rotation, pose_valid = _rotation(source[:, 3:9])
                transform = self.tcp_from_eef[side]
                xyz = source[:, :3] + (rotation @ transform[:3, 3, None])[..., 0]
                rotation = rotation @ transform[:3, :3]
                pose_valid &= np.isfinite(xyz).all(axis=-1)
            else:
                continue
            values[:, start:start + 3] = xyz
            values[:, start + 3:start + 9] = _rot6d(rotation)
            valid[:, start:start + 9] = pose_valid[:, None]
        if joints is not None:
            for side_id, side in enumerate(SIDES):
                indices = self.joint_indices[side]
                dest = slice(20 + side_id * 7, 20 + side_id * 7 + len(indices))
                values[:, dest] = joints[:, indices]
                valid[:, dest] = np.isfinite(joints[:, indices])
        values[~valid] = 0
        return values, valid

    def encode_window(self, columns):
        """Encode all future states relative to the fixed window start."""
        values, valid = self._absolute_values(columns)
        action, action_mask = values[1:].copy(), valid[1:].copy()
        for start in (0, 10):
            pose_valid = valid[0, start] & valid[1:, start]
            action_mask[:, start:start + 9] = pose_valid[:, None]
            if not pose_valid.any():
                continue
            rotations, _ = _rotation(values[:, start + 3:start + 9])
            reference = rotations[0].T
            action[:, start:start + 3] = (reference @ (values[1:, start:start + 3] - values[0, start:start + 3])[..., None])[..., 0]
            action[:, start + 3:start + 9] = _rot6d(reference @ rotations[1:])
        action[:, 20:] -= values[0, 20:]
        action_mask[:, 20:] &= valid[0, 20:]
        action[~action_mask] = 0
        return action, action_mask, values[:1], valid[:1]


class MoraHeadTCPRepresentation(MoraRelativeRepresentation):
    """Append the same TCP deltas expressed in the current head-camera axes.

    Both branches use the declared physical TCP, converted to the common world
    frame before taking differences. A moving robot reference requires its
    per-row reference-to-world transform; a flange is never used as fallback.
    Calibration is row-major world-to-camera, fixed at the window anchor.
    """

    raw_dim = 52
    unified_dim = 112
    scatter = np.concatenate((SCATTER, np.arange(94, 112)))

    def __init__(self, info, *, head_world_to_camera_key="observation.states.camera.world_to_camera",
                 head_camera_available=True, tcp_reference_to_world_key=None, **kwargs):
        super().__init__(info, **kwargs)
        self.tcp_reference_to_world_key = tcp_reference_to_world_key
        if tcp_reference_to_world_key is not None and tcp_reference_to_world_key in self.features:
            if self.features[tcp_reference_to_world_key].get("shape") not in ([7], [16]):
                raise ValueError("tcp_reference_to_world_key must contain pose7 wxyz or row-major matrix16")
            self.columns.append(tcp_reference_to_world_key)
        self.head_calibration_key = head_world_to_camera_key
        self.head_camera_available = bool(head_camera_available)
        if head_world_to_camera_key in self.features and self.head_camera_available:
            if self.features[head_world_to_camera_key].get("shape") != [16]:
                raise ValueError("head world_to_camera must have row-major shape [16]")
            self.columns.append(head_world_to_camera_key)
        self.contract.update({
            "representation": "head_tcp_relative_joint112", "raw_width": 52, "unified_width": 112,
            "head_world_to_camera_key": head_world_to_camera_key,
            "head_camera_available": self.head_camera_available,
            "head_pose_source": "declared_tcp", "head_endpoint": "same_tcp_as_local",
            "tcp_reference_to_world_key": tcp_reference_to_world_key,
            "tcp_reference_frame": "world" if tcp_reference_to_world_key is None else "transformed_to_world",
            "head_target": "reference_relative_in_current_head_camera",
            "tcp_target": "local_relative_in_current_tcp", "missing_head": "mask_head_pose",
        })
        self.contract.pop("schema_fingerprint")
        schema = {key: self.features[key] for key in self.columns}
        digest = json.dumps({"contract": self.contract, "schema": schema}, sort_keys=True).encode()
        self.contract["schema_fingerprint"] = hashlib.sha256(digest).hexdigest()

    def _absolute_values(self, columns):
        values, valid = super()._absolute_values(columns)
        key = self.tcp_reference_to_world_key
        if key is None:
            return values, valid
        pose_dims = [*range(9), *range(10, 19)]
        if key not in columns:
            values[:, pose_dims] = 0
            valid[:, pose_dims] = False
            return values, valid
        source = np.asarray(columns[key], dtype=np.float64)
        transforms = np.tile(np.eye(4), (len(source), 1, 1))
        transform_valid = np.isfinite(source).all(axis=-1)
        if source.shape[-1] == 7:
            transform_valid &= np.linalg.norm(source[:, 3:], axis=-1) > 1e-6
            transforms[:, :3, 3] = source[:, :3]
            transforms[transform_valid, :3, :3] = Rotation.from_quat(
                source[transform_valid][:, [4, 5, 6, 3]]).as_matrix()
        else:
            transforms = source.reshape(-1, 4, 4)
            rotation = transforms[:, :3, :3]
            transform_valid &= np.isclose(transforms[:, 3], [0, 0, 0, 1], atol=1e-5).all(axis=-1)
            transform_valid &= np.isclose(rotation @ rotation.transpose(0, 2, 1), np.eye(3), atol=1e-4).all(axis=(1, 2))
            # Evaluate determinants only for finite inputs to avoid invalid warnings.
            finite_rotation = np.where(np.isfinite(rotation), rotation, 0)
            transform_valid &= np.isclose(np.linalg.det(finite_rotation), 1, atol=1e-4)
        rw, tw = transforms[:, :3, :3], transforms[:, :3, 3]
        for start in (0, 10):
            rotations, _ = _rotation(values[:, start + 3:start + 9])
            values[:, start:start + 3] = (rw @ values[:, start:start + 3, None])[..., 0] + tw
            values[:, start + 3:start + 9] = _rot6d(rw @ rotations)
            valid[:, start:start + 9] &= transform_valid[:, None]
        values[~valid] = 0
        return values, valid

    def encode_window(self, columns):
        base = super().encode_window(columns)
        action, mask, state, state_mask = [np.pad(x, ((0, 0), (0, 18))) for x in base]
        key = self.head_calibration_key
        if not self.head_camera_available or key not in columns:
            return action, mask, state, state_mask
        camera = np.asarray(columns[key][0], dtype=np.float64).reshape(4, 4)
        rc, tc = camera[:3, :3], camera[:3, 3]
        if (not np.isfinite(camera).all() or not np.allclose(camera[3], [0, 0, 0, 1], atol=1e-5)
                or not np.allclose(rc @ rc.T, np.eye(3), atol=1e-4)
                or not np.isclose(np.linalg.det(rc), 1, atol=1e-4)):
            return action, mask, state, state_mask
        tcp, tcp_valid = self._absolute_values(columns)
        for side in range(2):
            src, dst = side * 10, 34 + side * 9
            xyz = tcp[:, src:src + 3]
            rotation, _ = _rotation(tcp[:, src + 3:src + 9])
            valid = tcp_valid[:, src]
            if valid[0]:
                state[0, dst:dst + 3] = rc @ xyz[0] + tc
                state[0, dst + 3:dst + 9] = _rot6d(rc @ rotation[0])
                state_mask[0, dst:dst + 9] = True
                action[:, dst:dst + 3] = (rc @ (xyz[1:] - xyz[0])[..., None])[..., 0]
                action[:, dst + 3:dst + 9] = _rot6d(rc @ rotation[1:] @ rotation[0].T @ rc.T)
                mask[:, dst:dst + 9] = valid[1:, None]
        action[~mask] = 0
        state[~state_mask] = 0
        return action, mask, state, state_mask
