"""Prevent a named CAD center or position-only offset becoming a full TCP GT."""
import numpy as np
import pytest

from scripts.data.tcp_endpoint_policy import apply_endpoint_policy


def sample():
    poses = np.tile(np.eye(4), (4, 2, 1, 1))
    poses[..., :3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    poses[..., :3, 3] = [1, 2, 3]
    return {'eef_world': poses, 'eef_valid': np.ones((4, 2), bool),
            'tcp_world': poses.copy(), 'tcp_valid': np.ones((4, 2), bool)}, {}


def test_rejects_agibot_named_urdf_center_as_task_tcp():
    arrays, meta = sample()
    apply_endpoint_policy('agibotworld', arrays, meta)
    assert not arrays['tcp_valid'].any()
    assert np.isnan(arrays['tcp_world']).all()
    assert meta['tcp_status'] == 'unresolved_contact_tcp'
    assert len(meta['origin_diagnostics']) == 2


def test_agibot_uses_own_episode_camera_audit_without_promoting_tcp():
    arrays, meta = sample()
    meta.update(head_calibration_quality='borderline', source_episode_audit={
        'head_extrinsic_audit_classification': 'normal',
        'head_extrinsic_base_fk_classification': 'acceptable',
        'head_extrinsic_repair_status': 'not_required'})
    apply_endpoint_policy('agibotworld', arrays, meta)
    assert meta['head_calibration_quality'] == 'normal'
    assert 'base-FK audit: acceptable' in meta['limitations']
    assert not arrays['tcp_valid'].any()


def test_agibot_missing_camera_audit_is_unknown():
    arrays, meta = sample()
    apply_endpoint_policy('agibotworld', arrays, meta)
    assert meta['head_calibration_quality'] == 'unknown'


def test_selected_agibot_visual_tcp_uses_local_axis_and_preserves_masks():
    arrays, meta = sample()
    meta['dataset_dir'] = '/data/agibotworld/task_362/a2d_gripper__dual_gripper__base__unmodified'
    arrays['eef_valid'][1, 0] = False
    apply_endpoint_policy('agibotworld', arrays, meta)
    np.testing.assert_allclose(arrays['tcp_world'][0, 0, :3, 3], [1, 2.2, 3])
    np.testing.assert_array_equal(arrays['tcp_world'][0, 0, :3, :3], arrays['eef_world'][0, 0, :3, :3])
    assert np.isnan(arrays['tcp_world'][1, 0]).all()
    assert not arrays['tcp_valid'][1, 0]
    assert meta['tcp_status'] == 'visual_tcp_preview'
    assert meta['tcp_orientation_verified'] is False
    assert meta['acceptance_status'] == 'pending_user_visual_review'
    assert meta['tcp_training_supervision_enabled'] is False


def test_other_agibot_embodiment_keeps_tcp_masked():
    arrays, meta = sample()
    meta['dataset_dir'] = '/data/agibotworld/task_362/other_gripper'
    apply_endpoint_policy('agibotworld', arrays, meta)
    assert not arrays['tcp_valid'].any()


@pytest.mark.parametrize('name', ['robodojo', 'robotwin'])
def test_user_preview_offset_follows_local_x_not_world_x(name):
    arrays, meta = sample()
    arrays['eef_valid'][2, 1] = False
    apply_endpoint_policy(name, arrays, meta)
    np.testing.assert_allclose(arrays['tcp_world'][0, 0, :3, 3], [1., 2.12, 3.])
    np.testing.assert_array_equal(arrays['tcp_world'][0, 0, :3, :3], arrays['eef_world'][0, 0, :3, :3])
    assert not arrays['tcp_valid'][2, 1]
    assert np.isnan(arrays['tcp_world'][2, 1]).all()
    assert meta['tcp_status'] == 'user_defined_tcp_preview'
    assert meta['origin_diagnostics'] == []
    assert meta['acceptance_status'] == 'user_visual_accepted'
    first = arrays['tcp_world'].copy()
    apply_endpoint_policy(name, arrays, meta)
    np.testing.assert_allclose(arrays['tcp_world'], first, equal_nan=True)
