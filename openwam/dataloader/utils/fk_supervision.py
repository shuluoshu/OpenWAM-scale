"""Build optional, frame-aligned FK GT payloads without deriving GT from joints."""

import numpy as np
from scipy.spatial.transform import Rotation

from openwam.kinematics import compile_chain


def joint_affine(stats, mode):
    """Inverse of the reader's normalization before clipping, in physical radians."""
    if mode in (None, "none", "null"):
        return np.ones(14, np.float32), np.zeros(14, np.float32)
    if mode == "z-score":
        return np.maximum(stats["std"][20:34], 1e-6), stats["mean"][20:34].copy()
    lower, upper = ("q01", "q99") if mode == "quantile" else ("min", "max")
    if mode not in ("quantile", "min-max"):
        raise ValueError(f"unsupported FK normalization {mode}")
    lo = np.asarray(stats[lower][20:34])
    scale = np.maximum(np.asarray(stats[upper][20:34]) - lo, 1e-6) / 2
    return scale, lo + scale


def _pose7(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[-1] != 7:
        raise ValueError("FK GT must be per-row xyz + quaternion(wxyz)")
    valid = np.isfinite(values).all(-1) & (np.linalg.norm(values[:, 3:], axis=-1) > 1e-6)
    pose = np.tile(np.eye(4), (len(values), 1, 1))
    pose[valid, :3, 3] = values[valid, :3]
    pose[valid, :3, :3] = Rotation.from_quat(values[valid][:, [4, 5, 6, 3]]).as_matrix()
    return pose, valid


class FKSupervisionBuilder:
    """Explicit URDF-base GT is necessary: a world pose cannot be guessed local."""

    def __init__(self, config, representation, info):
        self.config = config
        self.arms, self.columns = [], []
        if not config.get("arms") or set(config["arms"]) - {"left", "right"}:
            raise ValueError("FK supervision requires declared left/right arms")
        for side, arm in config["arms"].items():
            if arm.get("gt_frame") != "urdf_base":
                raise ValueError("FK GT must explicitly use gt_frame=urdf_base")
            if arm.get("endpoint") not in ("tcp", "eef") or "endpoint_transform" not in arm:
                raise ValueError("FK requires explicit endpoint semantics and endpoint_transform")
            key = arm["gt_pose_key"]
            if info["features"].get(key, {}).get("shape") != [7]:
                raise ValueError(f"FK GT column {key} must declare shape [7]")
            names = list(arm["joint_names"])
            if len(names) != len(representation.joint_indices[side]):
                raise ValueError("URDF joint order must match the reader arm joint order/count")
            chain = compile_chain(
                arm["urdf_path"], arm["base_link"], arm["target_link"], names, arm["endpoint_transform"]
            )
            self.arms.append((0 if side == "left" else 1, key, chain, arm["endpoint"]))
            self.columns.append(key)

    def build(self, columns, raw_action, raw_mask, raw_state, state_mask, stats, mode, horizon):
        scale, offset = joint_affine(stats, mode)
        arms = []
        for side, key, chain, endpoint in self.arms:
            n = min(len(raw_action), horizon)
            start, count = 20 + 7 * side, chain["dof"]
            block = slice(start, start + count)
            affine = slice(7 * side, 7 * side + count)
            pose, pose_valid = _pose7(columns[key])
            if len(pose) != len(raw_action) + 1:
                raise ValueError("FK GT and action frame alignment differs")
            valid = np.zeros(horizon, bool)
            valid[:n] = pose_valid[0] & pose_valid[1 : n + 1] & state_mask[0, block].all() & raw_mask[:n, block].all(-1)
            # FM supervises clipped normalized targets. Avoid contradictory FK GT.
            normalized = (raw_action[:n, block] - offset[affine]) / scale[affine]
            clipped = (np.abs(normalized) > 1 + 1e-5).any(-1) if mode in ("quantile", "min-max") else np.zeros(n, bool)
            clipped_count = int((valid[:n] & clipped).sum())
            valid[:n] &= ~clipped
            relative = np.tile(np.eye(4), (horizon, 1, 1))
            relative[:n] = np.linalg.inv(pose[0]) @ pose[1 : n + 1]
            arms.append(
                dict(
                    chain=chain,
                    slots=list(range(80 + 7 * side, 80 + 7 * side + count)),
                    q_anchor=raw_state[0, block].copy(),
                    scale=scale[affine].copy(),
                    offset=offset[affine].copy(),
                    gt_relative=relative.astype(np.float32),
                    valid=valid,
                    gt_pose_key=key,
                    gt_frame="urdf_base",
                    endpoint=endpoint,
                    normalization_clipped_count=clipped_count,
                )
            )
        return {"arms": arms}
