from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.experiments.stage12_projection import prepare_geometry, trajectories, prepare_video_geometry, video_trajectories


ROOT = Path('/mnt/data/cpfs/b5/yeqingda/openwam-0911/OpenWAM')
CASE = ROOT / 'outputs/z_action_stage2_comparison_20261027/data_v4/case_018.pt'


def synthetic_sample():
    state = np.zeros((1, 52)); state[0, [36, 45]] = 2
    raw = np.zeros((2, 52)); raw[:, 34] = [.1, .2]
    return dict(trace=dict(anchor=7, raw_state=state, raw_action=raw,
                          raw_state_mask=np.ones_like(state, bool), raw_action_mask=np.ones_like(raw, bool)),
                action_mask=np.ones((2, 112), bool),
                _eval=dict(action_stats=dict(q01=[-1.] * 52, q99=[1.] * 52, count=[1] * 52)))


def test_direct_head_delta_uses_current_camera_and_preserves_masks():
    s = synthetic_sample(); pred = np.zeros((2, 112)); pred[:, 94] = [.1, .2]
    e = np.eye(4); e[0, 3] = 1
    g = dict(camera=dict(K=np.diag([100., 100., 1.]), E=e, model='pinhole', D=None, wh=(640, 360)),
             eligible=True, fk_anchor_world=None)
    result = trajectories(pred, s, g)
    np.testing.assert_allclose(result['direct']['world_xyz'][:, 0, 0], [-.9, -.8])
    np.testing.assert_allclose(result['direct']['position_error_m'], 0, atol=1e-14)
    np.testing.assert_allclose(result['direct']['uv_pixels'][:, 0, 0], [5, 10])
    assert result['source_frames'].tolist() == [8, 9]
    s['action_mask'][1, 94] = False; pred[1, 94] = np.nan
    result = trajectories(pred, s, g)
    assert not result['direct']['valid'][1, 0]
    assert np.isnan(result['direct']['world_xyz'][1, 0]).all()
    assert not result['fk']['valid'].any()


def test_missing_camera_masks_output_without_identity_fallback():
    s = synthetic_sample()
    result = trajectories(np.zeros((2, 112)), s, dict(camera=None, eligible=False, fk_anchor_world=None))
    assert not result['direct']['valid'].any()
    assert np.isnan(result['direct']['world_xyz']).all()


def test_offscreen_prediction_counts_large_pixel_error_and_invalid_count():
    s = synthetic_sample(); pred = np.zeros((2, 112)); pred[0, 94] = 100.; pred[1, 94] = np.nan
    k = np.array([[100., 0., 320.], [0., 100., 180.], [0., 0., 1.]])
    g = dict(camera=dict(K=k, E=np.eye(4), model='pinhole', D=None, wh=(640, 360)),
             eligible=True, fk_anchor_world=None)
    d = trajectories(pred, s, g)['direct']
    assert d['gt_in_frame'].all()
    assert d['pred_projectable'][0, 0] and not d['pred_in_frame'][0, 0]
    assert d['pixel_error_valid'][0, 0] and d['pixel_error'][0, 0] > 2000
    assert d['pred_uv'][0, 0, 0] > 319
    assert d['counts'] == dict(eligible_gt=4, invalid_pred=1, outside_pred=1, scored=3)


@pytest.mark.parametrize('offset,expected', [(0., True), (.01, False)])
def test_prepare_geometry_checks_native_world_head_anchor(tmp_path, monkeypatch, offset, expected):
    import json
    from scripts.experiments import stage12_projection as module
    s = synthetic_sample(); s['trace']['raw_state'][0, [2, 12]] = 2
    head = 'observation.images.cam_high'
    s['_eval']['family'] = 'interndata_a1'
    s['trace'].update(source_repo=str(tmp_path), episode=0,
                      video_alignment={head:dict(alignment={7:dict(native_wh=[640, 360])})})
    (tmp_path / 'meta').mkdir()
    (tmp_path / 'meta/info.json').write_text(json.dumps(dict(features={head:dict(shape=[360, 640, 3])})))
    e = np.eye(4); e[0, 3] = offset
    values = {'observation.states.cameras.cam_high.intrinsic':np.eye(3),
              'observation.states.cameras.cam_high.world_to_camera':e}
    monkeypatch.setattr(module, '_anchor_row', lambda sample:(values, []))
    g = prepare_geometry([s])[0]
    assert g['eligible'] is expected
    if not expected:
        assert g['camera'] is None and 'native head anchor mismatch' in g['reason']


@pytest.mark.skipif(not CASE.exists(), reason='real case 18 fixture unavailable')
def test_real_case18_native_fk_and_registered_world_match_raw_gt():
    import pyarrow.parquet as pq
    from openwam.model.fk_supervision import torch_fk
    from scripts.data.urdf_fk_audit import pose7_to_matrix
    s = torch.load(CASE, map_location='cpu', weights_only=False)
    g = prepare_geometry([s])[0]
    assert g['eligible'] and all(g['fk_eligible'])
    assert 'source-anchor registration' in g['provenance']['fk_label']
    trace = s['trace']; names = ['frame_index']
    for side in ('left', 'right'):
        names += [f'states.{side}_joint.position', f'states.{side}_tcp_to_{side}_armbase_pose']
    rows = pq.read_table(trace['source_parquet'], columns=names,
                        filters=[('episode_index', '=', 10), ('frame_index', '>=', 338),
                                 ('frame_index', '<=', 369)]).to_pandas().sort_values('frame_index')
    pred = s['action'].float().numpy()
    result = trajectories(pred, s, g)
    for side_id, (side, arm) in enumerate(zip(('left', 'right'), s['fk_supervision']['arms'])):
        q = torch.tensor(np.stack(rows[f'states.{side}_joint.position']), dtype=torch.float64)
        fk = torch_fk(q, arm['chain']).numpy()
        native = pose7_to_matrix(np.stack(rows[f'states.{side}_tcp_to_{side}_armbase_pose']))
        np.testing.assert_allclose(fk, native, atol=2e-6, rtol=0)
        world = (g['fk_anchor_world'][side_id] @ fk)[:, :3, 3]
        np.testing.assert_allclose(world, result['direct']['gt_world_xyz'][:, side_id], atol=2e-6, rtol=0)
        valid = result['fk']['metric_valid'][:, side_id]
        assert valid.sum() == (32 if side_id == 0 else 26)
        assert result['fk']['position_error_m'][valid, side_id].max() < 2e-6
    assert not result['direct']['gt_in_frame'][-1, 1]


def moving_video_fixture():
    s=synthetic_sample();s['trace']['raw_state'][0,[2,12]]=2
    s['trace']['raw_action']=np.zeros((32,52));s['trace']['raw_action'][:,34]=np.arange(32)/100
    s['trace']['raw_action_mask']=np.ones((32,52),bool);s['action_mask']=np.ones((32,112),bool)
    pred=np.zeros((32,112));pred[:,94]=np.arange(32)/100
    camera=dict(K=np.array([[100.,0,160],[0,100,128],[0,0,1]]),E=np.eye(4),model='pinhole',D=None,wh=(320,256))
    geometry=dict(camera=camera,eligible=True,fk_anchor_world=None)
    fixed=trajectories(pred,s,geometry)
    frames=[]
    for j in range(9):
        c=dict(camera,E=np.eye(4));c['E'][0,3]=j*.1
        frames.append(dict(source_frame=7+4*j,action_index=4*j-1,camera=c,eligible=True,video_valid=True,reason=''))
    vg=dict(frames=frames,source_frames=np.arange(7,40,4),action_indices=np.arange(-1,32,4),
            anchor_gt_xyz=np.array([[0,0,2],[0,0,2.]]),anchor_gt_valid=np.ones(2,bool),reason='')
    return s,pred,geometry,fixed,vg


def test_video_uses_action_4j_minus1_and_same_frame_moving_camera():
    s,_,_,fixed,vg=moving_video_fixture();v=video_trajectories(fixed,s,vg);d=v['direct']
    assert v['action_indices'].tolist()==[-1,3,7,11,15,19,23,27,31]
    np.testing.assert_allclose(d['pred_xyz'][1:,0,0],np.arange(3,32,4)/100)
    np.testing.assert_allclose(d['pred_uv'][1,0],[166.5,128])
    assert d['pred_uv'][1,0,0]!=fixed['direct']['pred_uv'][3,0,0]
    assert d['gt_valid'][0].all() and not d['valid'][0].any()
    assert np.isnan(d['pred_uv'][0]).all() and not d['metric_valid'][0].any()
    assert d['counts']['eligible_gt']==16 and d['counts']['scored']==16
    assert not v['fk']['gt_valid'].any()


def test_video_offscreen_prediction_scores_without_clamp_and_missing_camera_masks():
    s,pred,g,_,vg=moving_video_fixture();pred[3,94]=20;pred[11,94]=np.nan
    fixed=trajectories(pred,s,g);vg['frames'][2].update(camera=None,eligible=False,reason='missing camera')
    d=video_trajectories(fixed,s,vg)['direct']
    assert d['pred_uv'][1,0,0]>319 and not d['pred_in_frame'][1,0]
    assert d['pixel_error_valid'][1,0] and d['pixel_error'][1,0]>900
    assert not d['gt_in_frame'][2].any() and np.isnan(d['pred_uv'][2]).all()
    assert d['counts']['outside_pred']==1 and d['counts']['eligible_gt']==14
    assert d['counts']['invalid_pred']==1 and d['counts']['scored']==13


def test_video_rejects_misaligned_action_timestamps():
    s,_,_,fixed,vg=moving_video_fixture();fixed['source_frames']=fixed['source_frames']+1
    with pytest.raises(ValueError,match='action/source frame'):
        video_trajectories(fixed,s,vg)


@pytest.mark.parametrize('model',['brown_conrady','double_sphere'])
def test_video_uses_each_frames_distortion_model(model):
    s,_,_,fixed,vg=moving_video_fixture();camera=vg['frames'][1]['camera'];camera['model']=model
    if model=='brown_conrady':
        camera['D']=np.array([.1,.01,.01,.02]);x=.13/2;r2=x*x
        expected=[160+100*(x*(1+.1*r2+.01*r2*r2)+.02*3*r2),128+100*.01*r2]
    else:
        camera['D']=np.array([0.,.5]);expected=[160+100*.13/(.5*np.sqrt(.13**2+4)+1),128]
    result=video_trajectories(fixed,s,vg)
    np.testing.assert_allclose(result['direct']['pred_uv'][1,0],expected,atol=1e-8)


@pytest.mark.parametrize('failure,reason',[('missing_k','missing intrinsic'),('alignment','decoded source frame'),
    ('padded','masked or padded'),('invalid_camera','camera.valid false'),('wrong_dimensions','dimensions mismatch')])
def test_prepare_video_geometry_rejects_missing_or_mismatched_frame_calibration(tmp_path,monkeypatch,failure,reason):
    import json
    from scripts.experiments import stage12_projection as module
    s,_,_,_,_=moving_video_fixture();head='observation.images.cam_high';frames=np.arange(7,40,4)
    s['_eval']['family']='interndata_a1';s['video_mask']=np.ones(9,bool)
    alignment={int(f):dict(source_frame_index=int(f),native_wh=[320,256]) for f in frames}
    s['trace'].update(source_repo=str(tmp_path),episode=0,video_source_indices=frames,
                      video_alignment={head:dict(alignment=alignment)})
    (tmp_path/'meta').mkdir();(tmp_path/'meta/info.json').write_text(json.dumps(dict(features={head:dict(shape=[256,320,3])})))
    kk='observation.states.cameras.cam_high.intrinsic';ek='observation.states.cameras.cam_high.world_to_camera'
    rows={int(f):{kk:np.eye(3),ek:np.eye(4)} for f in frames}
    if failure=='missing_k':del rows[11][kk]
    elif failure=='alignment':alignment[11]['source_frame_index']=12
    elif failure=='padded':s['video_mask'][1]=False
    elif failure=='invalid_camera':rows[11]['observation.states.camera.valid']=False
    elif failure=='wrong_dimensions':alignment[11]['native_wh']=[640,256]
    monkeypatch.setattr(module,'_video_source_rows',lambda sample,indices:(rows,[]))
    monkeypatch.setattr(module,'prepare_geometry',lambda samples:[dict(eligible=True,reason='',provenance={})])
    vg=prepare_video_geometry(s)
    assert vg['frames'][0]['eligible'] and vg['frames'][2]['eligible']
    assert not vg['frames'][1]['eligible'] and vg['frames'][1]['camera'] is None
    assert reason in vg['frames'][1]['reason']


@pytest.mark.skipif(not CASE.exists(),reason='real case 18 fixture unavailable')
def test_real_case18_video_frame0_is_gt_only_and_fk_matches_selected_actions():
    s=torch.load(CASE,map_location='cpu',weights_only=False);g=prepare_geometry([s])[0]
    fixed=trajectories(s['action'],s,g);vg=prepare_video_geometry(s);video=video_trajectories(fixed,s,vg)
    assert all(f['eligible'] for f in vg['frames'])
    assert video['fk']['observed_gt_count']==2 and not video['fk']['valid'][0].any()
    selected=np.arange(3,32,4)
    np.testing.assert_allclose(video['fk']['pred_xyz'][1:],fixed['fk']['pred_xyz'][selected],equal_nan=True)
    metric=video['fk']['metric_valid']
    assert metric.sum()==14
    assert video['fk']['position_error_m'][metric].max()<2e-6
