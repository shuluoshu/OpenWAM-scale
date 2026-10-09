"""Nonidentity-frame checks for video coordinate explanations."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from scripts.data.video_trajectory_geometry import decode_relative, encode_relative, future_indices


def test_local_and_head_encode_different_axes_and_restore_same_world_point():
    anchor = np.tile(np.eye(4), (2, 1, 1))
    anchor[:, :3, 3] = [4, 5, 6]
    anchor[:, :3, :3] = Rotation.from_euler('z', 90, degrees=True).as_matrix()
    poses = np.tile(anchor, (3, 1, 1, 1))
    poses[:, :, :3, 3] += [1, 0, 0]
    poses[:, :, :3, :3] = Rotation.from_euler('x', 30, degrees=True).as_matrix()
    camera = np.eye(4)
    camera[:3, :3] = Rotation.from_euler('z', 30, degrees=True).as_matrix()
    camera[:3, 3] = [100, 200, 300]  # Translation must not enter delta vectors.
    local = encode_relative(poses, anchor, camera, 'local')
    head = encode_relative(poses, anchor, camera, 'head')
    np.testing.assert_allclose(local[..., :3, 3], np.broadcast_to([0, -1, 0], (3, 2, 3)), atol=1e-14)
    np.testing.assert_allclose(head[..., :3, 3], np.broadcast_to([np.sqrt(3)/2, .5, 0], (3, 2, 3)), atol=1e-14)
    # Both translation and orientation depend on the chosen reference axes.
    assert not np.allclose(local[..., :3, :3], head[..., :3, :3])
    for mode, encoded in [('local', local), ('head', head)]:
        np.testing.assert_allclose(decode_relative(encoded, anchor, camera, mode), poses, atol=1e-14)


@pytest.mark.parametrize('mode', ['head', 'local'])
def test_random_rotation_roundtrip(mode):
    rng = np.random.default_rng(421)
    poses = np.tile(np.eye(4), (12, 2, 1, 1))
    poses[..., :3, :3] = Rotation.random(24, random_state=rng).as_matrix().reshape(12, 2, 3, 3)
    poses[..., :3, 3] = rng.normal(size=(12, 2, 3))
    camera = np.eye(4)
    camera[:3, :3] = Rotation.random(random_state=rng).as_matrix()
    encoded = encode_relative(poses, poses[0], camera, mode)
    np.testing.assert_allclose(encoded[0], np.tile(np.eye(4), (2, 1, 1)), atol=1e-14)
    np.testing.assert_allclose(decode_relative(encoded, poses[0], camera, mode), poses, atol=1e-14)


def test_future_indices_truncate_within_sample_without_repeating_labels():
    np.testing.assert_array_equal(future_indices(0, 40), np.arange(33))
    np.testing.assert_array_equal(future_indices(37, 40), [37, 38, 39])
    np.testing.assert_array_equal(future_indices(39, 40), [39])
    with pytest.raises(ValueError):
        future_indices(40, 40)
