import numpy as np
import pytest
import torch


def verified_info():
    return {'starvla_metadata': {'source': 'genrobot', 'action': {
        'semantic': 'absolute_next_step', 'frame_offset': 1,
        'last_frame_policy': 'copy_last_available_masked'}}}


@pytest.mark.parametrize('canonical', [True, False, float('nan'), float('inf')])
def test_next_action_does_not_override_verified_goal_observation(canonical):
    from openwam.dataloader.goal_validity import stage1_goal_validity
    from openwam.dataloader.native_stage2_io import source_frame_validity
    columns = {'canonical.frame_valid': [True, canonical], 'training.valid': [True, False],
               'source.frame_validity': [True, True]}
    valid, policy = stage1_goal_validity(columns, 2, family='Genrobot', info=verified_info())
    assert valid.tolist() == [True, canonical is True]
    assert 'next-action' in policy
    assert source_frame_validity(columns, 2).tolist() == [True, False]


@pytest.mark.parametrize('change', ['family', 'semantic', 'offset', 'bool_offset', 'last_frame', 'canonical'])
def test_unverified_sources_remain_strict(change):
    from openwam.dataloader.goal_validity import stage1_goal_validity
    columns = {'canonical.frame_valid': [True], 'training.valid': [False]}
    info = verified_info(); family = 'Genrobot'
    if change == 'family': family = 'mecka'
    elif change == 'canonical': columns.pop('canonical.frame_valid')
    else:
        key, value = {'semantic': ('semantic', 'unknown'), 'offset': ('frame_offset', 2),
                      'bool_offset': ('frame_offset', True),
                      'last_frame': ('last_frame_policy', 'unknown')}[change]
        info['starvla_metadata']['action'][key] = value
    valid, policy = stage1_goal_validity(columns, 1, family=family, info=info)
    assert not valid.any()
    assert policy == 'legacy-source-frame-validity-v1'


def test_source_and_camera_flags_still_mask_goal():
    from openwam.dataloader.goal_validity import stage1_goal_validity, goal_camera_validity
    columns = {'canonical.frame_valid': [True, True], 'training.valid': [False, False],
               'source.frame_validity': [True, False]}
    valid, policy = stage1_goal_validity(columns, 2, family='Genrobot', info=verified_info())
    assert valid.tolist() == [True, False]
    row = {'raw.robot0.sensor.camera2.video_alignment_valid': False,
           'raw.robot1.sensor.camera0.video_alignment_valid': True}
    views = goal_camera_validity(row, ['observation.images.cam_high', 'observation.images.cam_left_wrist', None],
                                 observation_valid=True, policy=policy)
    assert views == [False, True, False]


def test_generation_has_no_gt_dependency_and_still_produces_when_gt_invalid():
    from test_goal_memory_core import fixture
    from openwam.train.goal_generation import generate_goal_latents
    model, args = fixture(); model.config.goal_horizon_mode = 'kf1_endpoint'; model.eval()
    batch = {k: v for k, v in args.items() if k not in ('noisy_target_latents', 'timestep')}
    batch['observed_view_valid'] = torch.tensor([[True, False, False], [True, True, False]])
    batch['target_image_valid'] = torch.zeros(2, dtype=torch.bool)
    batch['target_view_valid'] = torch.zeros(2, 3, dtype=torch.bool)
    batch['target_latents'] = torch.full_like(args['noisy_target_latents'], float('nan'))
    def common_noise(template, *, seed, cursor, validation):
        gen = torch.Generator(device=template.device).manual_seed(seed + cursor)
        return None, torch.randn(template.shape, device=template.device, dtype=template.dtype, generator=gen)
    first = generate_goal_latents(model, batch, common_noise=common_noise, seed=42, index=8, steps=2)
    for name in tuple(batch):
        if name.startswith('target_'): del batch[name]
    second = generate_goal_latents(model, batch, common_noise=common_noise, seed=42, index=8, steps=2)
    assert torch.isfinite(first['latents']).all() and first['latents'].abs().sum() > 0
    torch.testing.assert_close(first['latents'], second['latents'], rtol=0, atol=0)
    torch.testing.assert_close(first['uv'], second['uv'], rtol=0, atol=0)


@pytest.mark.parametrize('canonical,camera,expected', [(True, True, True), (False, True, False), (True, False, False)])
def test_reader_preserves_endpoint_stage2_history_and_independent_tcp(tmp_path, canonical, camera, expected):
    import json
    import yaml
    import pyarrow as pa
    import pyarrow.parquet as pq
    from test_curated_goal import fixture
    from openwam.dataloader.curated_goal import CuratedGoalDataset
    av = pytest.importorskip('av')
    config, oldroot = fixture(tmp_path)
    root = tmp_path / 'Genrobot/fixture'; root.parent.mkdir(); oldroot.rename(root)
    config.write_text(yaml.safe_dump({'data_root_dir': str(tmp_path), 'selection': {
        'segments_by_repo': {'Genrobot/fixture': [[0, 1, 3]]}, 'repo_weights': {'Genrobot/fixture': 1}}}))
    info = json.loads((root / 'meta/info.json').read_text())
    info.update(verified_info())
    info['features'] = {'observation.images.cam_high': {'dtype': 'video', 'shape': [24, 32, 3]}}
    (root / 'meta/info.json').write_text(json.dumps(info))
    epfile = root / 'meta/episodes/chunk-000/file-000.parquet'
    ep = pq.read_table(epfile).to_pylist()[0]
    ep = {k.replace('egocentric', 'cam_high'): v for k, v in ep.items()}
    pq.write_table(pa.Table.from_pylist([ep]), epfile)
    rows = pq.read_table(root / 'data.parquet').to_pylist()
    for i, row in enumerate(rows):
        row.update({'canonical.frame_valid': canonical if i == 7 else True,
                    'training.valid': i != 7, 'source.frame_validity': True,
                    'raw.robot0.sensor.camera2.video_alignment_valid': camera})
    pq.write_table(pa.Table.from_pylist(rows), root / 'data.parquet')
    with av.open(str(root / 'video.mp4'), 'w') as container:
        stream = container.add_stream('mpeg4', rate=10)
        stream.width = 32; stream.height = 24; stream.pix_fmt = 'yuv420p'
        for index in range(10):
            frame = av.VideoFrame.from_ndarray(np.full((24, 32, 3), index * 20, np.uint8), format='rgb24')
            for packet in stream.encode(frame): container.mux(packet)
        for packet in stream.encode(): container.mux(packet)
    reader = CuratedGoalDataset(config, tmp_path / 'cache', history={'enabled': True},
                                observation_mode='multiview', materialize_multiview_target=True)
    repo = reader.repos[0]; metadata = reader._episode(repo, 0)
    record, batch = reader._materialize(repo, 0, metadata, metadata['tasks'][0], 1, 7, 'task')
    assert record['goal_source_frame'] == 7 == record['instruction_end_exclusive'] - 1
    assert not metadata['source_valid'][7] and not metadata['history_source_valid'][7]
    assert bool(batch['targets'].image_valid) == expected
    assert bool(batch['target_view_valid'][0, 0]) == expected
    assert not batch['targets'].point_valid.any()  # Missing geometry stays missing even with readable RGB.
    assert record['success_verified'] is False
