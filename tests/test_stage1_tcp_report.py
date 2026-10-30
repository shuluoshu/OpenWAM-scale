import numpy as np
import pytest
from PIL import Image

from scripts.experiments.stage1_tcp_report import tcp_overlay_data, draw_tcp_overlay, coverage_rows


def inputs(valid):
    record=dict(family='robot', provenance=dict(endpoint=dict(definition_id='missing_verified_tcp',
        missing_reasons=['No reviewed source TCP chain']), camera=dict(missing_reasons=['missing intrinsic or world_to_camera'])))
    target=dict(uv=np.array([[[0., 0.], [.5, .5]]]), point_valid=np.array([[valid, valid]]),
                in_frame=np.array([[valid, valid]]), image_valid=np.array([True]))
    prediction=dict(uv=[[[.4, .6], [.7, .55]]], valid_tcp=2 if valid else 0)
    return record, target, prediction


def test_missing_gt_keeps_placeholders_hidden_and_draws_untrusted_prediction(tmp_path):
    r,t,p=inputs(False);data=tcp_overlay_data(r,t,p)
    assert data['valid_gt_count']==0
    assert all(a['gt_px'] is None and a['prediction_status']=='unsupervised_untrusted' for a in data['arms'])
    np.testing.assert_allclose(data['arms'][0]['pred_px'],[127.6,153.])
    out=tmp_path/'overlay.png';draw_tcp_overlay(Image.new('RGB',(320,256),'black'),data,out,'test')
    im=np.array(Image.open(out));assert np.any(np.all(im[:256]==[255,159,28],axis=-1))
    assert not np.any(np.all(im[:256]==[0,230,118],axis=-1))
    assert 'missing intrinsic' in data['reason']
    assert data['prediction_coordinate_kind']=='learned_normalized_uv'
    assert data['gt_coordinate_kind']=='source_geometry_projection'
    assert data['physical_contact_verification']=='not_established_by_this_renderer'


def test_valid_gt_is_green_and_prediction_red_without_mutating_input(tmp_path):
    r,t,p=inputs(True);before=t['uv'].copy();data=tcp_overlay_data(r,t,p)
    out=tmp_path/'overlay.png';draw_tcp_overlay(Image.new('RGB',(320,256),'black'),data,out,'test')
    im=np.array(Image.open(out));assert np.any(np.all(im[:256]==[0,230,118],axis=-1))
    assert np.any(np.all(im[:256]==[255,59,48],axis=-1))
    np.testing.assert_array_equal(t['uv'],before)
    assert data['valid_gt_count']==2


def test_outside_prediction_is_not_clamped_or_drawn_and_coverage_is_explicit(tmp_path):
    r,t,p=inputs(False);p['uv']=[[[2.,.5],[float('nan'),.5]]]
    data=tcp_overlay_data(r,t,p)
    assert data['arms'][0]['pred_px']==[638.,127.5]
    assert not any(a['pred_drawable'] for a in data['arms'])
    rows=coverage_rows([dict(record=r,split='fit',tcp=data)])
    assert rows[0]['family']=='robot' and rows[0]['valid_gt_points']==0 and rows[0]['possible_gt_points']==2


@pytest.mark.parametrize('valid_gt', [False, True])
def test_evaluate_writes_markers_on_both_backgrounds_and_preserves_loss_masks(tmp_path, valid_gt):
    import json
    import torch
    from types import SimpleNamespace
    from openwam.model.goal.types import GoalTargets
    from scripts.experiments.overfit_visual_history import evaluate, rgb
    image=torch.zeros(1,3,1,256,320)
    target=GoalTargets(image,torch.tensor([True]),torch.tensor([[[.3,.3],[.7,.7]]]),
                       torch.full((1,2),valid_gt),torch.full((1,2),valid_gt),torch.full((1,2),-1))
    mask_before=target.point_valid.clone()
    class Dummy:
        shift=5.
        codec=SimpleNamespace(decode=lambda latent:image)
        def eval(self):return self
        def condition_tokens(self,*args,**kwargs):return torch.zeros(1,120,8)
        def encode_observation(self,*args,**kwargs):return torch.zeros(1,48,1,24,20)
        def velocity(self,obs,image,tcp,*args,**kwargs):return torch.zeros_like(image),torch.zeros_like(tcp)
    batch=dict(observed=image,observed_view_valid=torch.ones(1,3,dtype=torch.bool),
               targets=target,context=torch.zeros(1,2,8),context_lens=torch.tensor([2]))
    metrics=evaluate(Dummy(),batch,tmp_path,all_targets=[rgb(image)])
    assert (tmp_path/'prediction_tcp.png').exists()
    coords=json.loads((tmp_path/'tcp_coordinates.json').read_text())
    assert coords['valid_gt_count']==(2 if valid_gt else 0)
    expected_color=[255,59,48] if valid_gt else [255,159,28]
    for file in ['tcp.png','prediction_tcp.png']:
        pixels=np.array(Image.open(tmp_path/file))[:256]
        assert np.any(np.all(pixels==expected_color,axis=-1))
    assert metrics['valid_tcp']==(2 if valid_gt else 0)
    if valid_gt:
        expected=np.linalg.norm((np.asarray(metrics['uv'])-target.uv.numpy())*[319,255],axis=-1).mean()
        assert metrics['tcp_error_px']==pytest.approx(expected,abs=1e-4)
    else:assert metrics['tcp_error_px'] is None
    torch.testing.assert_close(target.point_valid,mask_before)
