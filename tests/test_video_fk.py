import json

import numpy as np
import pytest

from scripts.data.video_fk import load_fk_tracks


def poses(n):
    return np.broadcast_to(np.eye(4), (n, 2, 4, 4)).copy()


def audit(tmp_path, name, **arrays):
    folder = tmp_path / name
    folder.mkdir()
    np.savez(folder / 'arrays.npz', **arrays)
    (folder / 'metadata.json').write_text(json.dumps({'episode_index': 0}))
    return tmp_path


def test_exact_join_preserves_order_and_masks_missing_rows(tmp_path):
    fk = poses(2)
    fk[:, :, 0, 3] = [[10, 11], [30, 31]]
    root = audit(tmp_path, 'robotwin', frame_indices=[1, 3], fk_pose=fk,
                 joint=np.ones((2, 14)))
    result, meta = load_fk_tracks('robotwin', np.array([3, 2, 1]), {}, {}, root)
    np.testing.assert_array_equal(result['fk_eef_valid'], [[1, 1], [0, 0], [1, 1]])
    np.testing.assert_allclose(result['fk_eef_world'][[0, 2], :, 0, 3], [[30, 31], [10, 11]])
    assert np.isnan(result['fk_eef_world'][1]).all()
    assert not result['fk_tcp_valid'].any()
    assert not result['fk_joint_valid'][:, [6, 13]].any()
    assert meta['missing_frame_indices'] == [2]


def test_rejects_sample_frame_mismatch(tmp_path):
    with pytest.raises(ValueError, match='frame_indices'):
        load_fk_tracks('robotwin', np.array([1]), {'frame_indices': np.array([2])}, {}, tmp_path)


@pytest.mark.parametrize('name', ['robodojo', 'robotwin'])
def test_user_preview_tcp_uses_fk_flange_and_preserves_missing_masks(tmp_path, name):
    fk = poses(1)
    fk[..., :3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    root = audit(tmp_path, name, frame_indices=[5], fk_pose=fk)
    transform = np.eye(4)
    transform[0, 3] = .12
    result, _ = load_fk_tracks(name, np.array([5, 6]), {},
                              {'tcp_status': 'user_defined_tcp_preview',
                               'tcp_fixed_transform': transform.tolist()}, root)
    np.testing.assert_allclose(result['fk_tcp_world'][0, 0, :3, 3], [0, .12, 0])
    assert result['fk_tcp_valid'][0].all()
    assert not result['fk_tcp_valid'][1].any()


def test_rejects_duplicate_audit_indices(tmp_path):
    root = audit(tmp_path, 'robotwin', frame_indices=[1, 1], fk_pose=poses(2))
    with pytest.raises(ValueError, match='duplicate'):
        load_fk_tracks('robotwin', np.array([1]), {}, {}, root)


def test_masks_invalid_audit_pose_and_explicit_validity(tmp_path):
    fk = poses(2)
    fk[0, 1, 0, 0] = np.nan
    root = audit(tmp_path, 'robodojo', frame_indices=[1, 2], fk_pose=fk,
                 fk_valid=np.array([[1, 1], [0, 1]], bool))
    result, _ = load_fk_tracks('robodojo', np.array([1, 2]), {}, {}, root)
    np.testing.assert_array_equal(result['fk_eef_valid'], [[1, 0], [0, 1]])
    assert np.isnan(result['fk_eef_world'][~result['fk_eef_valid']]).all()


def test_interndata_tcp_composes_world_from_robot_on_left(tmp_path):
    tcp = poses(2)
    tcp[:, :, 0, 3] = 2
    world = np.broadcast_to(np.eye(4), (2, 4, 4)).copy()
    world[:, :3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    world[:, 0, 3] = [10, 20]
    root = audit(tmp_path, 'interndata', frame_indices=[5, 6], fk_pose=poses(2),
                 canonical_diagnostic_fk_pose=poses(2), tcp_diagnostic_fk_pose=tcp)
    result, meta = load_fk_tracks('interndata', np.array([5, 6]),
                                  {'world_from_robot': world}, {}, root)
    np.testing.assert_allclose(result['fk_tcp_world'][:, 0, :3, 3], [[10, 2, 0], [20, 2, 0]])
    assert result['fk_tcp_valid'].all()
    assert meta['independent_world_calibration'] is False
    assert 'row 0' in meta['world_frame_status']


def test_interndata_missing_robot_transform_masks_tcp_only(tmp_path):
    root = audit(tmp_path, 'interndata', frame_indices=[1],
                 canonical_diagnostic_fk_pose=poses(1), tcp_diagnostic_fk_pose=poses(1))
    result, meta = load_fk_tracks('interndata', np.array([1]), {}, {}, root)
    assert result['fk_eef_valid'].all()
    assert not result['fk_tcp_valid'].any()
    assert any('robot_to_env' in item for item in meta['limitations'])


def test_agibot_uses_interarm_comparison_and_never_projects_torso0(tmp_path):
    fk, gt = poses(2), poses(2)
    fk[:, 0, 0, 3] = 1
    fk[:, 1, 0, 3] = [3, 4]
    gt[:] = fk
    gt[:, :, 0, 3] += 100
    root = audit(tmp_path, 'agibotworld', frame_indices=[1, 2], fk_pose=fk, label_pose=gt)
    result, meta = load_fk_tracks('agibotworld', np.array([1, 2]), {}, {}, root)
    assert not result['fk_eef_valid'].any()
    assert not result['fk_tcp_valid'].any()
    np.testing.assert_allclose(result['fk_local_pose'], result['label_local_pose'])
    np.testing.assert_allclose(result['fk_local_pose'][:, 1, 0, 3], [2, 3])
    assert result['fk_local_valid'].all()
    assert 'left wrist' in meta['local_frame']
    assert 'torso' in meta['world_frame_status']


def test_human_and_missing_audits_return_invalid_tracks(tmp_path):
    for name in ('genrobot', 'lightwheel', 'mecka', 'robotwin'):
        result, meta = load_fk_tracks(name, np.array([4, 5]), {}, {}, tmp_path)
        assert result['fk_eef_world'].shape == (2, 2, 4, 4)
        assert not result['fk_eef_valid'].any()
        assert not result['fk_tcp_valid'].any()
        assert meta['status'] == 'unavailable'


def test_episode_mismatch_rejected(tmp_path):
    root = audit(tmp_path, 'robotwin', frame_indices=[1], fk_pose=poses(1))
    with pytest.raises(ValueError, match='episode'):
        load_fk_tracks('robotwin', np.array([1]), {}, {'episode': 7}, root)


def test_source_robot_to_env_join_is_per_episode_and_exact_frame(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = audit(tmp_path, 'interndata', frame_indices=[2, 7],
                 canonical_diagnostic_fk_pose=poses(2), tcp_diagnostic_fk_pose=poses(2))
    dataset = tmp_path / 'source'
    (dataset / 'data').mkdir(parents=True)
    pq.write_table(pa.table({'episode_index': [0, 1, 0], 'frame_index': [7, 2, 2],
                            'states.robot_to_env_pose': [[70., 0, 0, 1, 0, 0, 0],
                                                        [999., 0, 0, 1, 0, 0, 0],
                                                        [20., 0, 0, 1, 0, 0, 0]]}),
                   dataset / 'data' / 'part.parquet')
    (root / 'interndata' / 'metadata.json').write_text(json.dumps(
        {'episode_index': 0, 'dataset_dir': str(dataset)}))
    result, _ = load_fk_tracks('interndata', np.array([7, 2]), {}, {}, root)
    np.testing.assert_allclose(result['fk_tcp_world'][:, 0, 0, 3], [70, 20])


def test_invalid_robot_pose7_masks_tcp_without_warning(tmp_path):
    root = audit(tmp_path, 'interndata', frame_indices=[1],
                 canonical_diagnostic_fk_pose=poses(1), tcp_diagnostic_fk_pose=poses(1))
    result, _ = load_fk_tracks('interndata', np.array([1]),
                               {'states.robot_to_env_pose': np.zeros((1, 7))}, {}, root)
    assert not result['fk_tcp_valid'].any()
