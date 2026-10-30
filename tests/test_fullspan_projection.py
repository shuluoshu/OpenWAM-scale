import importlib.util
import json

import numpy as np
import pytest


def test_projection_module_exists():
    assert importlib.util.find_spec('scripts.experiments.fullspan_projection') is not None


def fixture():
    state = np.zeros((1, 52)); state[0, [2, 12, 36, 45]] = 2
    raw = np.zeros((32, 52)); raw[:, 34] = np.arange(1, 33) / 100
    sample = dict(trace=dict(anchor=10, raw_state=state, raw_state_mask=np.ones_like(state, bool),
                            raw_action=raw, raw_action_mask=np.ones_like(raw, bool)),
                  action_mask=np.ones((32, 112), bool),
                  _eval=dict(family='interndata_a1', action_stats=dict(q01=[-1.] * 52, q99=[1.] * 52, count=[1] * 52)))
    pred = np.zeros((32, 112)); pred[:, 94] = raw[:, 34]
    camera = dict(K=np.array([[100., 0, 160], [0, 100, 128], [0, 0, 1]]),
                  E=np.eye(4), model='pinhole', D=None, wh=(320, 256))
    base = dict(camera=camera, eligible=True, fk_anchor_world=None, reason='')
    views = {}
    for index, role in enumerate(('head', 'left_wrist', 'right_wrist')):
        cam = dict(camera, E=np.eye(4)); cam['E'][0, 3] = index
        views[role] = dict(camera=cam, eligible=True, image_available=True, reason='', source_frame=10)
    return sample, pred, dict(anchor=10, base_geometry=base, views=views, endpoint={})


def test_all_future_points_use_each_fixed_anchor_camera():
    from scripts.experiments.fullspan_projection import project_action_chunks
    sample, pred, geometry = fixture()
    result = project_action_chunks(sample, pred, geometry)
    assert result['source_frames'].tolist() == list(range(11, 43))
    for index, role in enumerate(('head', 'left_wrist', 'right_wrist')):
        direct = result['views'][role]['direct']
        assert direct['pred_uv'].shape == (32, 2, 2)
        np.testing.assert_allclose(direct['pred_uv'][:, 0, 0], 160 + 50 * (index + np.arange(1, 33) / 100))
        np.testing.assert_allclose(direct['pred_uv'], direct['gt_uv'])
        assert direct['gt_in_frame'].all()
        assert not result['views'][role]['fk']['valid'].any()


def test_invalid_prediction_preserves_gt_and_offscreen_values():
    from scripts.experiments.fullspan_projection import project_action_chunks
    sample, pred, geometry = fixture(); pred[0, 94] = np.nan; pred[1, 94] = 100
    result = project_action_chunks(sample, pred, geometry)['views']['head']['direct']
    assert result['gt_in_frame'][0, 0] and not result['pred_projectable'][0, 0]
    assert result['pred_projectable'][1, 0] and not result['pred_in_frame'][1, 0]
    assert result['pred_uv'][1, 0, 0] > 319
    assert result['pixel_error_valid'][1, 0]


def test_missing_calibration_keeps_image_and_never_borrows_head_camera():
    from scripts.experiments.fullspan_projection import project_action_chunks
    sample, pred, geometry = fixture()
    geometry['views']['left_wrist'].update(camera=None, eligible=False, reason='missing wrist calibration')
    geometry['views']['right_wrist'].update(camera=None, eligible=False, image_available=False, reason='missing image')
    result = project_action_chunks(sample, pred, geometry)
    left = result['views']['left_wrist']; right = result['views']['right_wrist']
    assert left['image_available'] and not left['projection_available']
    assert not right['image_available']
    assert np.isnan(left['direct']['pred_uv']).all() and not left['direct']['gt_in_frame'].any()
    assert result['views']['head']['projection_available']


def test_short_tail_is_nan_padded_and_geometry_anchor_cannot_be_reused():
    from scripts.experiments.fullspan_projection import project_action_chunks
    sample, pred, geometry = fixture()
    sample['trace']['raw_action'] = sample['trace']['raw_action'][:3]
    sample['trace']['raw_action_mask'] = sample['trace']['raw_action_mask'][:3]
    result = project_action_chunks(sample, pred, geometry)['views']['head']['direct']
    assert result['gt_uv'].shape == (32, 2, 2)
    assert result['gt_valid'][:3].all() and not result['gt_valid'][3:].any()
    assert np.isnan(result['gt_uv'][3:]).all()
    geometry['anchor'] = 11
    with pytest.raises(ValueError, match='anchor'):
        project_action_chunks(sample, pred, geometry)


def calibration_fixture(tmp_path, monkeypatch):
    from scripts.experiments import fullspan_projection as module
    sample, pred, geometry = fixture()
    cameras = ['observation.images.cam_high', 'observation.images.cam_left_wrist', 'observation.images.cam_right_wrist']
    (tmp_path / 'meta').mkdir()
    (tmp_path / 'meta/info.json').write_text(json.dumps(dict(features={c: dict(dtype='video', shape=[256, 320, 3]) for c in cameras})))
    sample['trace'].update(source_repo=str(tmp_path), episode=0,
        video_alignment={c: dict(alignment={10: dict(source_frame_index=10, native_wh=[320, 256])}) for c in cameras})
    sample['goal_view_valid'] = np.ones(3, bool)
    values = {'observation.states.camera.intrinsic':geometry['base_geometry']['camera']['K'],
              'observation.states.camera.world_to_camera':np.eye(4)}
    for index, camera in enumerate(cameras[1:], 1):
        stem = 'observation.states.cameras.' + camera.rsplit('.', 1)[-1]
        values[stem + '.intrinsic'] = geometry['base_geometry']['camera']['K']
        e = np.eye(4); e[0, 3] = index
        values[stem + '.world_to_camera'] = e
    monkeypatch.setattr(module, '_source_anchor_row', lambda sample: (values, ['native.parquet']))
    monkeypatch.setattr(module.stage12, 'prepare_geometry', lambda samples: [geometry['base_geometry']])
    return module, sample, values


def test_prepare_requires_native_wrist_calibration_and_current_decode_identity(tmp_path, monkeypatch):
    module, sample, values = calibration_fixture(tmp_path, monkeypatch)
    del values['observation.states.cameras.cam_left_wrist.world_to_camera']
    result = module.prepare_projection_geometry(sample)
    assert result['views']['head']['eligible']
    left = result['views']['left_wrist']; right = result['views']['right_wrist']
    assert left['image_available'] and not left['eligible'] and left['camera'] is None
    assert 'missing intrinsic or world_to_camera' in left['reason']
    assert right['eligible']
    np.testing.assert_array_equal(right['camera']['E'][:3, 3], [2, 0, 0])
    sample['trace']['video_alignment']['observation.images.cam_right_wrist']['alignment'][10]['source_frame_index'] = 11
    result = module.prepare_projection_geometry(sample)
    assert not result['views']['right_wrist']['eligible']


def test_qa_rejected_wrist_still_has_image(tmp_path, monkeypatch):
    module, sample, _ = calibration_fixture(tmp_path, monkeypatch)
    sample['trace']['calibration_eligibility'] = dict(strict_head_camera_eef_eligible=True, strict_wrist_camera_eef_eligible=False)
    result = module.prepare_projection_geometry(sample)
    assert result['views']['head']['eligible']
    assert result['views']['left_wrist']['image_available']
    assert not result['views']['left_wrist']['eligible']
    assert 'QA' in result['views']['left_wrist']['reason']


def test_unreadable_calibration_keeps_existing_images(tmp_path, monkeypatch):
    module, sample, _ = calibration_fixture(tmp_path, monkeypatch)
    def unavailable(sample):
        raise OSError('source calibration row unreadable')
    monkeypatch.setattr(module, '_source_anchor_row', unavailable)
    result = module.prepare_projection_geometry(sample)
    assert all(view['image_available'] for view in result['views'].values())
    assert not any(view['eligible'] for view in result['views'].values())


def human_endpoint_row(frame, valid=True):
    hands = np.zeros((2, 21, 3)); hands[:, :, 2] = 2
    hands[:, 4, 0] = frame / 100; hands[:, 8, 0] = frame / 100 + .02
    return dict(frame_index=frame, episode_index=0, **{
        'observation.states.hand.position': hands.tolist(),
        'observation.states.hand.valid': [float(valid), float(valid)]})


def test_source_human_gt_survives_missing_native_actions_and_task_end(tmp_path, monkeypatch):
    module, sample, _ = calibration_fixture(tmp_path, monkeypatch)
    sample['_eval']['family'] = 'LightWheel'
    # Use this family's real head name with the same synthetic camera geometry.
    old = 'observation.images.cam_high'; head = 'observation.images.head_left'
    info_path = tmp_path / 'meta/info.json'; info = json.loads(info_path.read_text())
    info['features'][head] = info['features'].pop(old); info_path.write_text(json.dumps(info))
    sample['trace']['video_alignment'][head] = sample['trace']['video_alignment'].pop(old)
    sample['action_mask'][:] = False
    sample['trace']['raw_action_mask'][:] = False
    sample['instruction_end_exclusive'] = 14
    calls = []
    def future_rows(s):
        calls.append(s['trace']['anchor'])
        return {frame: human_endpoint_row(frame, valid=frame != 12) for frame in range(11, 43)}, ['source.parquet']
    monkeypatch.setattr(module, '_source_future_rows', future_rows)
    geometry = module.prepare_projection_geometry(sample)
    for _ in range(2):
        result = module.project_action_chunks(sample, np.zeros((32, 112)), geometry)
        direct = result['views']['head']['direct']
        assert direct['gt_in_frame'][[0, 2]].all()
        assert not direct['gt_valid'][1].any() and not direct['gt_valid'][3:].any()
        assert not direct['valid'].any() and not direct['pixel_error_valid'].any()
        assert direct['gt_source'][0, 0] == 'source_endpoint_annotation'
        np.testing.assert_allclose(direct['gt_world_xyz'][0, 0], [.12, 0, 2])
        assert result['source_gt_provenance']['endpoint']['definition_id'] == 'human_thumb_index_midpoint_v1'
    assert calls == [10]
    assert not sample['action_mask'].any()


def test_native_gt_preferred_if_consistent_source_disagreement_is_audited():
    from scripts.experiments.fullspan_projection import project_action_chunks
    sample, pred, geometry = fixture()
    xyz = np.zeros((32, 2, 3)); xyz[..., 2] = 2; xyz[:, 0, 0] = np.arange(1, 33) / 100
    xyz[1, 0, 0] += .05
    geometry['source_gt'] = dict(world_xyz=xyz, valid=np.ones((32, 2), bool), provenance={}, reason='')
    result = project_action_chunks(sample, pred, geometry)['views']['head']['direct']
    assert result['gt_source'][0, 0] == 'native_action_supervision'
    assert result['gt_source'][1, 0] == 'source_endpoint_annotation'
    assert result['gt_native_disagreement'][1, 0]
    np.testing.assert_allclose(result['gt_world_xyz'][1, 0, 0], .07)
    assert not result['pixel_error_valid'][1, 0]
    np.testing.assert_allclose(result['world_xyz'][1, 0, 0], .02)


def test_source_future_rows_reads_one_bounded_batch_and_honors_episode(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from scripts.experiments.fullspan_projection import _source_future_rows
    rows = [human_endpoint_row(frame) for frame in range(9, 45)]
    rows.append(dict(human_endpoint_row(11), episode_index=1))
    path = tmp_path / 'source.parquet'; pq.write_table(pa.Table.from_pylist(rows), path)
    sample = dict(trace=dict(anchor=10, episode=0, source_parquet=str(path), last_source=42),
                  instruction_end_exclusive=14)
    actual, paths = _source_future_rows(sample)
    assert sorted(actual) == [11, 12, 13]
    assert paths == [str(path)]
    assert actual[11]['episode_index'] == 0


@pytest.mark.parametrize('provenance_location', ['camera_status', 'candidate_backfill'])
def test_candidate_wrist_calibration_preserves_image_without_projection(tmp_path, monkeypatch, provenance_location):
    module, sample, _ = calibration_fixture(tmp_path, monkeypatch)
    sample['_eval']['family'] = 'robochallenge_v2'
    info_path = tmp_path / 'meta/info.json'
    info = json.loads(info_path.read_text())
    if provenance_location == 'camera_status':
        info['starvla_metadata'] = dict(coordinate_system=dict(cameras={
            'cam_left_wrist': dict(status='visual_candidate_manual_selected')}))
    else:
        info['starvla_metadata'] = dict(camera_candidate_backfill=dict(
            status='visual_candidate_manual_selected', fields={
                'standard_left_world_to_camera': 'observation.states.cameras.cam_left_wrist.world_to_camera'}))
    info_path.write_text(json.dumps(info))
    geometry = module.prepare_projection_geometry(sample)
    left = geometry['views']['left_wrist']
    assert left['image_available']
    assert not left['eligible']
    assert left['camera'] is None
    assert 'visual_candidate_manual_selected' in left['reason']
    assert 'unverified' in left['reason']
    assert geometry['views']['head']['eligible']
    assert geometry['views']['right_wrist']['eligible']
    result = module.project_action_chunks(sample, np.zeros((32, 112)), geometry)
    assert result['views']['left_wrist']['image_available']
    assert not result['views']['left_wrist']['projection_available']
    assert not result['views']['left_wrist']['direct']['gt_in_frame'].any()
    assert not result['views']['left_wrist']['direct']['pred_in_frame'].any()
