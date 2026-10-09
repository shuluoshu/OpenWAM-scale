import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from scripts.data.genrobot_wrist_geometry import (
    FINGER_BASE_TO_CAMERA,
    genrobot_wrist_world_to_camera,
)


def inputs(n=1):
    return (np.tile(np.eye(4), (n, 1, 1)),
            np.tile([0., 0, 0, 0, 0, 0, 1], (n, 1)), np.ones(n, bool))


def test_matches_finger_cad_joint_and_maps_tcp_below_center():
    cad = Rotation.from_euler('xyz', [-np.pi / 2, 0, -np.pi / 2]).as_matrix()
    np.testing.assert_allclose(FINGER_BASE_TO_CAMERA[:3, :3], cad, atol=1e-14)
    poses, calibration, valid = inputs()
    poses[0, :3, :3] = Rotation.from_euler('xyz', [.3, -.6, 1.2]).as_matrix()
    poses[0, :3, 3] = [2., -.3, 1.8]
    cameras, accepted = genrobot_wrist_world_to_camera(poses, calibration, valid)
    tcp_world = poses[0] @ [.1015, -.0003, -.0398, 1.]
    np.testing.assert_allclose(cameras[0] @ tcp_world, [.0003, .0398, .1015, 1], atol=1e-14)
    np.testing.assert_allclose(cameras[0] @ poses[0] @ FINGER_BASE_TO_CAMERA, np.eye(4), atol=1e-14)
    assert accepted.all()


def test_masks_unestablished_calibration_and_invalid_source_without_inversion_errors():
    poses, calibration, source_valid = inputs(7)
    source_valid[1] = False
    calibration[2, 0] = .01  # nonidentity needs a separately established chain
    calibration[3] = [0, 0, 0, 1, 0, 0, 0]  # not xyzw identity
    poses[4, 0, 0] = 0  # singular
    poses[5] = np.nan
    poses[6, 3, 0] = .2
    cameras, accepted = genrobot_wrist_world_to_camera(poses, calibration, source_valid)
    np.testing.assert_array_equal(accepted, [True, False, False, False, False, False, False])
    assert np.isnan(cameras[~accepted]).all()
    assert np.isfinite(cameras[accepted]).all()


def test_accepts_equivalent_negative_identity_quaternion_and_rejects_reflection():
    poses, calibration, valid = inputs(2)
    calibration[0, -1] = -1
    poses[1, 0, 0] = -1
    _, accepted = genrobot_wrist_world_to_camera(poses, calibration, valid)
    np.testing.assert_array_equal(accepted, [True, False])


def test_rejects_shape_mismatch_instead_of_broadcasting_sides():
    poses, calibration, valid = inputs(2)
    with pytest.raises(ValueError):
        genrobot_wrist_world_to_camera(poses, calibration[:1], valid)
