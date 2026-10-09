import numpy as np
import pytest

from openwam.dataloader.utils.mora_relative import MoraRelativeRepresentation

IDENTITY = [1, 0, 0, 0, 1, 0]
RZ90 = [0, 1, 0, -1, 0, 0]


def source_info(joints=7):
    names = [f"{side}_joint_{j}" for side in ("left", "right") for j in range(joints)]
    return {"features": {
        "eef": {"shape": [20]},
        "joint": {"shape": [2*joints], "names": names},
    }}


def poses():
    # The origin is translated and rotated: subtracting world XYZ is wrong.
    return np.array([
        [10, 20, 30, *RZ90, .2, 0, 0, 0, *IDENTITY, .3],
        [11, 20, 30, *IDENTITY, .4, 0, 2, 0, *IDENTITY, .5],
        [12, 20, 30, *RZ90, .6, 0, 3, 0, *IDENTITY, .7],
    ], dtype=np.float32)


def representation(**kwargs):
    return MoraRelativeRepresentation(source_info(), state_key="eef", joint_state_key="joint", **kwargs)


def test_local_pose_and_joint_delta_share_window_start_anchor():
    rep = representation(tcp_from_eef={"left": np.eye(4).tolist(), "right": np.eye(4).tolist()})
    joints = np.arange(42, dtype=np.float32).reshape(3, 14)
    action, mask, state, state_mask = rep.encode_window({"eef": poses(), "joint": joints})
    assert action.shape == (2, 34)
    np.testing.assert_allclose(action[:, :3], [[0, -1, 0], [0, -2, 0]], atol=1e-6)
    np.testing.assert_allclose(action[0, 3:9], [0, -1, 0, 1, 0, 0], atol=1e-6)
    np.testing.assert_array_equal(action[:, 20:], joints[1:] - joints[0])
    np.testing.assert_array_equal(state[0, 20:], joints[0])
    np.testing.assert_array_equal(action[:, 9], np.array([.4, .6], dtype=np.float32))
    assert mask.all() and state_mask.all()


def test_no_tcp_does_not_supervise_pose_but_keeps_joints_and_grippers():
    rep = representation()
    eef = poses()
    eef[:, :9] = np.nan  # An absent pose must not poison independent labels.
    action, mask, state, state_mask = rep.encode_window({"eef": eef, "joint": np.ones((3, 14))})
    pose_dims = list(range(9)) + list(range(10, 19))
    assert not mask[:, pose_dims].any()
    assert not state_mask[:, pose_dims].any()
    assert not action[:, pose_dims].any()
    assert mask[:, [9, 19]].all() and mask[:, 20:].all()
    assert np.isfinite(action).all() and np.isfinite(state).all()


def test_six_joint_arms_leave_seventh_slots_masked():
    rep = MoraRelativeRepresentation(source_info(6), state_key="eef", joint_state_key="joint")
    action, mask, _, _ = rep.encode_window({"eef": poses(), "joint": np.arange(36).reshape(3, 12)})
    assert not mask[:, [26, 33]].any()
    assert not action[:, [26, 33]].any()
    np.testing.assert_array_equal(action[0, 20:26], np.full(6, 12))
    np.testing.assert_array_equal(action[0, 27:33], np.full(6, 12))


def test_fixed_tcp_offset_changes_relative_translation_under_rotation():
    offset = np.eye(4)
    offset[0, 3] = 1
    rep = representation(tcp_from_eef={"left": offset.tolist()})
    action, mask, _, _ = rep.encode_window({"eef": poses(), "joint": np.ones((3, 14))})
    # TCP0=(10,21,30); TCP1=(12,20,30), expressed in Rz90 axes.
    np.testing.assert_allclose(action[0, :3], [-1, -2, 0], atol=1e-6)
    assert mask[:, :9].all() and not mask[:, 10:19].any()


def test_pose_missing_entirely_allows_video_only_samples():
    rep = MoraRelativeRepresentation({"features": {}}, state_key="eef", joint_state_key="joint")
    action, mask, _, state_mask = rep.encode_window({"task_index": [0, 0, 0]})
    assert action.shape == (2, 34)
    assert not action.any() and not mask.any() and not state_mask.any()


def test_explicit_tcp_columns_use_wxyz_and_missing_side_is_masked():
    info = source_info()
    info["features"]["left_tcp"] = {"shape": [7]}
    rep = MoraRelativeRepresentation(info, state_key="eef", joint_state_key="joint", tcp_pose_keys={"left": "left_tcp"})
    data = {"eef": poses(), "joint": np.ones((3, 14)), "left_tcp": [[0, 0, 0, 1, 0, 0, 0], [1, 2, 3, 1, 0, 0, 0], [2, 3, 4, 1, 0, 0, 0]]}
    action, mask, _, _ = rep.encode_window(data)
    np.testing.assert_allclose(action[0, :9], [1, 2, 3, *IDENTITY])
    assert mask[:, :9].all() and not mask[:, 10:19].any()


def test_declared_joint_layout_cannot_silently_include_gripper():
    with pytest.raises(ValueError, match="joint_indices"):
        representation(joint_indices={"left": [0] * 7, "right": list(range(7, 14))})
