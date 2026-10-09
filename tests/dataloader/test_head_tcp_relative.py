import numpy as np
from scipy.spatial.transform import Rotation

from openwam.dataloader.utils.mora_relative import MoraHeadTCPRepresentation, _rot6d
from tests.dataloader.test_mora_relative import poses, source_info


def test_head_reference_rotation_differs_from_tcp_local_and_uses_current_camera():
    info = source_info()
    info['features']['cal'] = {'shape': [16]}
    rep = MoraHeadTCPRepresentation(info, state_key='eef', joint_state_key='joint',
                                    head_world_to_camera_key='cal', tcp_from_eef={'left': np.eye(4).tolist()})
    eef = poses()
    r0, r1, rc = Rotation.from_euler('xyz', [[20, 40, 70], [60, -10, 30], [10, 80, -30]], degrees=True).as_matrix()
    eef[0, 3:9], eef[1, 3:9] = _rot6d(r0), _rot6d(r1)
    cal = np.tile(np.eye(4), (3, 1, 1))
    cal[0, :3, :3], cal[0, :3, 3] = rc, [3, 8, 10]
    cal[1:] = np.nan  # Future camera motion/calibration must not affect this anchor.
    action, mask, state, _ = rep.encode_window({'eef': eef, 'cal': cal.reshape(3, 16)})
    assert action.shape == (2, 52)
    np.testing.assert_allclose(action[0, 34:37], rc @ (eef[1, :3] - eef[0, :3]), atol=1e-6)
    np.testing.assert_allclose(action[0, 37:43], _rot6d(rc @ r1 @ r0.T @ rc.T), atol=1e-6)
    np.testing.assert_allclose(action[0, 3:9], _rot6d(r0.T @ r1), atol=1e-6)
    np.testing.assert_allclose(state[0, 34:37], rc @ eef[0, :3] + cal[0, :3, 3], atol=2e-6)
    assert mask[:, 34:43].all()
    assert not mask[:, 43:].any()
    assert not mask[:, 10:19].any()  # both right poses are masked without a right TCP


def test_missing_or_invalid_head_calibration_masks_only_head():
    for cal in (None, np.zeros((3, 16)), np.full((3, 16), np.nan)):
        info = source_info()
        columns = {'eef': poses()}
        if cal is not None:
            info['features']['cal'] = {'shape': [16]}
            columns['cal'] = cal
        rep = MoraHeadTCPRepresentation(info, state_key='eef', head_world_to_camera_key='cal',
                                        tcp_from_eef={'left': np.eye(4).tolist()})
        action, mask, _, state_mask = rep.encode_window(columns)
        assert not mask[:, 34:].any() and not action[:, 34:].any()
        assert not state_mask[:, 34:].any()
        assert mask[:, :9].all()


def test_same_tcp_point_and_moving_robot_reference_for_both_branches():
    info = source_info()
    info['features'].update({'cal': {'shape': [16]}, 'tcp': {'shape': [7]}, 'base': {'shape': [7]}})
    rep = MoraHeadTCPRepresentation(info, state_key='eef', tcp_pose_keys={'left': 'tcp'},
                                    head_world_to_camera_key='cal', tcp_reference_to_world_key='base')
    data = {'eef': poses(), 'tcp': [[1, 0, 0, 1, 0, 0, 0]] * 3,
            'base': [[0, 0, 0, 1, 0, 0, 0], [2, 0, 0, 1, 0, 0, 0], [4, 0, 0, 1, 0, 0, 0]],
            'cal': np.tile(np.eye(4).ravel(), (3, 1))}
    action, mask, state, _ = rep.encode_window(data)
    # TCP is static relative to robot, but the robot moves two meters in world.
    np.testing.assert_allclose(action[:, :3], [[2, 0, 0], [4, 0, 0]])
    np.testing.assert_allclose(action[:, 34:37], action[:, :3])
    np.testing.assert_allclose(state[:, :3], state[:, 34:37])
    assert mask[:, :9].all() and mask[:, 34:43].all()
    assert not mask[:, 43:].any()  # canonical right EEF cannot replace a missing TCP
    data['base'][1] = [float('nan')] * 7
    action, mask, _, _ = rep.encode_window(data)
    assert not mask[0, :9].any() and not mask[0, 34:43].any()
    assert not action[0, :9].any() and mask[1, :9].all()


def test_rotating_matrix16_reference_and_missing_reference_are_masked():
    info = source_info()
    info['features'].update({'base': {'shape': [16]}, 'cal': {'shape': [16]}, 'tcp': {'shape': [7]}})
    rep = MoraHeadTCPRepresentation(info, tcp_pose_keys={'left': 'tcp'},
                                    head_world_to_camera_key='cal', tcp_reference_to_world_key='base')
    reference = np.tile(np.eye(4), (2, 1, 1))
    reference[1, :3, :3] = Rotation.from_euler('z', 90, degrees=True).as_matrix()
    data = {'tcp': [[1, 0, 0, 1, 0, 0, 0]] * 2,
            'base': reference.reshape(2, 16), 'cal': np.tile(np.eye(4).ravel(), (2, 1))}
    action, mask, _, _ = rep.encode_window(data)
    np.testing.assert_allclose(action[0, :3], [-1, 1, 0], atol=1e-6)
    np.testing.assert_allclose(action[0, 34:43], action[0, :9], atol=1e-6)
    assert mask[0, :9].all()
    del data['base']
    action, mask, _, _ = rep.encode_window(data)
    assert not mask.any() and not action.any()
