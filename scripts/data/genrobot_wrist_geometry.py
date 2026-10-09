"""CAD-derived Genrobot wrist optical chain for the audited identity calibration.

Evidence: /mnt/data/cpfs/b5/libinjie/Finger-urdf/urdf/Finger-urdf.urdf,
lines 87-99, fixed joint ``cam`` (base_link -> cam_Link), zero translation,
RPY=(-pi/2, 0, -pi/2). URDF SHA256 is recorded below. Raw finger xyz+xyzw
matches canonical world EEF in shard_0148 episode0, frames30..209; the
camera chain puts the existing TCP-origin candidate at the visible gripper
center in both original wrist videos. Full audit and comparison images:
outputs/fk_consistency/genrobot_wrist_audit/.

This is a CAD-derived visualization chain, not independent optical calibration
or a certified physical TCP. Acquisition serial/CAD revision matching remains
unestablished. It intentionally masks nonidentity T_b_c: its calibration-parent
semantics would need additional evidence. Head MONITOR_OPTICAL_AXIS and the
legacy FRD correction are different chains and must not be substituted here.
"""
from __future__ import annotations

import numpy as np

FINGER_URDF_SOURCE = '/mnt/data/cpfs/b5/libinjie/Finger-urdf/urdf/Finger-urdf.urdf'
FINGER_URDF_SHA256 = '3317e155d66f126b23847a31e514708eeaef3a1a514a760bba9563b328282b4e'
# T_base_camera maps optical coordinates to finger base coordinates.
# Optical axes in base: right=-Y, down=-Z, forward=+X.
FINGER_BASE_TO_CAMERA = np.array([
    [0., 0., 1., 0.],
    [-1., 0., 0., 0.],
    [0., -1., 0., 0.],
    [0., 0., 0., 1.],
])
FINGER_BASE_TO_CAMERA.setflags(write=False)


def genrobot_wrist_world_to_camera(eef_world, raw_t_b_c, source_valid):
    """Return (N,4,4) T_camera_world and (N,) valid for one wrist.

    eef_world is canonical T_world_finger_base for that same hand. raw_t_b_c
    is the same wrist's raw camera0 calibration, xyz+xyzw (N,7).
    source_valid (N,) must combine canonical frame, raw finger pose alignment,
    and wrist video alignment masks. The caller also validates same-view K/D.
    Invalid, unsupported, or masked rows return NaN matrices.

    Formula for the audited identity-T_b_c case:
    T_camera_world = inverse(T_world_finger_base @ T_finger_base_cam_Link).
    """
    poses = np.asarray(eef_world, dtype=np.float64)
    calibration = np.asarray(raw_t_b_c, dtype=np.float64)
    supplied_valid = np.asarray(source_valid, dtype=bool)
    if (poses.ndim != 3 or poses.shape[1:] != (4, 4)
            or calibration.shape != (len(poses), 7)
            or supplied_valid.shape != (len(poses),)):
        raise ValueError('expected EEF (N,4,4), same-wrist xyz+xyzw T_b_c (N,7), validity (N,)')
    finite = np.isfinite(poses).all(axis=(1, 2))
    rotation = np.where(finite[:, None, None], poses[:, :3, :3], np.eye(3))
    rigid = (finite
             & np.isclose(poses[:, 3], [0, 0, 0, 1], atol=1e-8, rtol=0).all(axis=1)
             & np.isclose(np.swapaxes(rotation, -1, -2) @ rotation,
                          np.eye(3), atol=1e-5, rtol=0).all(axis=(1, 2))
             & np.isclose(np.linalg.det(rotation), 1., atol=1e-5, rtol=0))
    identity_calibration = (np.isfinite(calibration).all(axis=1)
                            & np.isclose(calibration[:, :6], 0., atol=1e-8, rtol=0).all(axis=1)
                            & np.isclose(np.abs(calibration[:, 6]), 1., atol=1e-8, rtol=0))
    valid = supplied_valid & rigid & identity_calibration
    result = np.full_like(poses, np.nan)
    result[valid] = np.linalg.inv(poses[valid] @ FINGER_BASE_TO_CAMERA)
    return result, valid
