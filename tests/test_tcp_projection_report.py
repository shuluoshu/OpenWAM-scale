"""Frame-selection and source validity checks around the report's projections."""
import numpy as np

from scripts.visualize_tcp_projections import projected


def test_whole_trajectory_is_reprojected_into_selected_camera():
    camera = np.tile(np.eye(4), (3, 1, 1))
    camera[1, 0, 3] = -1
    camera[2, 1, 3] = 3
    poses = np.tile(np.eye(4), (3, 2, 1, 1))
    poses[..., 2, 3] = 2
    poses[:, :, 0, 3] = np.arange(3)[:, None]
    arrays = dict(world_to_camera=camera, K=np.tile(np.diag([100., 100., 1.]), (3, 1, 1)),
                  distortion=np.empty((3, 0)), camera_valid=np.ones(3, bool))
    anchor, valid = projected(arrays, poses, np.ones((3, 2), bool), 0, 'pinhole')
    moving, _ = projected(arrays, poses, np.ones((3, 2), bool), 1, 'pinhole')
    np.testing.assert_allclose(anchor[:, 0], [[0, 0], [50, 0], [100, 0]])
    np.testing.assert_allclose(moving[:, 0], [[-50, 0], [0, 0], [50, 0]])
    assert valid.all()
    # Invalid future source labels are masked, even with a valid anchor camera.
    source_valid = np.ones((3, 2), bool)
    source_valid[2, 1] = False
    uv, valid = projected(arrays, poses, source_valid, 0, 'pinhole')
    assert not valid[2, 1] and np.isnan(uv[2, 1]).all()
    # Unavailable calibration is never substituted with identity.
    arrays['camera_valid'][0] = False
    arrays['world_to_camera'][0] = np.nan
    uv, valid = projected(arrays, poses, source_valid, 0, 'pinhole')
    assert not valid.any() and np.isnan(uv).all()
