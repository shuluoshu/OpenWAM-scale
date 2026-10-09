import numpy as np

from scripts.data.wrist_projection_samples import camera_validity, wrist_keys


def test_robotwin_uses_side_specific_intrinsic_not_head():
    features = {'observation.states.camera.calibration.left_camera.intrinsic_cv': {},
                'observation.states.cameras.cam_left_wrist.world_to_camera': {},
                'observation.states.camera.intrinsic': {}}
    k, e = wrist_keys('left', features)
    assert k == 'observation.states.camera.calibration.left_camera.intrinsic_cv'
    assert e == 'observation.states.cameras.cam_left_wrist.world_to_camera'
    assert wrist_keys('right', features) == (None, None)


def test_invalid_extrinsic_and_intrinsic_mask_projection():
    k = np.tile([[500., 0, 320], [0, 500, 240], [0, 0, 1]], (5, 1, 1))
    e = np.tile(np.eye(4), (5, 1, 1))
    k[1, 0, 0] = 0
    e[2, 0, 0] = 2
    e[3, 3, 0] = 1
    e[4] = np.nan
    np.testing.assert_array_equal(camera_validity(k, e), [True, False, False, False, False])


def test_head_calibration_is_never_a_wrist_fallback():
    features = {'observation.states.camera.intrinsic': {},
                'observation.states.camera.world_to_camera': {}}
    assert wrist_keys('left', features) == (None, None)
