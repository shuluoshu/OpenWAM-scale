import json
from pathlib import Path
import pytest
from scripts.experiments.report_stage12_contract import checked_stage1, discover_variants, projection_lane


def test_stage1_missing_is_not_unavailable(tmp_path):
    assert checked_stage1(tmp_path, {'trace': {}}, {})['status'] == 'missing'


def test_stage1_rejects_wrong_actual_source_before_normalizing(tmp_path):
    (tmp_path/'metadata.json').write_text(json.dumps({'identity': {'source_repo': 'wrong'}}))
    sample={'trace': {'source_repo': 'actual', 'episode': 1, 'anchor': 2}, 'prompt': 'pick'}
    with pytest.raises(ValueError, match='identity'):
        checked_stage1(tmp_path, sample, {'dataset': 'actual'})


def test_variants_derive_initial_latest_from_eval_directories(tmp_path):
    for mode, steps in [('joint', [0, 25, 530]), ('none', [0, 602])]:
        for step in steps: (tmp_path/mode/f'eval_{step:08d}').mkdir(parents=True)
    assert [(x['mode'], x['step']) for x in discover_variants(tmp_path)] == [('joint',0),('joint',530),('none',0),('none',602)]


def test_prediction_only_lane_is_ready_without_gt(tmp_path):
    (tmp_path/'video_fk_tcp.mp4').write_bytes(b'test')
    p={'fk': {'valid': [[True, False]], 'gt_valid': [[False, False]]}, 'reason': ''}
    got=projection_lane(tmp_path, tmp_path, 'fk', p, {'trace': {'mask_reasons': []}})
    assert got['status']=='ready'
    assert got['assets']==['video_fk_tcp.mp4']


def test_absent_render_is_missing_when_prediction_exists(tmp_path):
    p={'direct': {'valid': [[True,False]]}, 'reason': ''}
    assert projection_lane(tmp_path,tmp_path,'direct',p,{'trace': {'mask_reasons': []}})['status']=='missing'


def test_missing_reviewed_mapping_explains_camera_available(tmp_path):
    p={'camera_eligible':[True]*9,'direct':{'valid':[[False,False]]},'reason':'masked'}
    got=projection_lane(tmp_path,tmp_path,'direct',p,{'trace':{'mask_reasons':['missing_tcp_action_mapping']}})
    assert got['status']=='unavailable'
    assert 'reviewed TCP' in got['reason'] and 'camera available' in got['reason']


def test_stage1_includes_real_target_overlay_and_visible_metric(tmp_path):
    import torch
    from scripts.experiments.infer_stage1_samecase import sample_identity
    sample={'trace':{'source_repo':'actual','identity':['actual',1,2],'episode':1,'anchor':2},
            'prompt':'pick','granularity':'task','instruction_start':0,'instruction_end_exclusive':10}
    for key in ('goal_observed_multiview','goal_view_valid','history_images','history_valid','history_ages','history_episode_id'):
        sample[key]=torch.zeros(1)
    meta={'identity':sample_identity(sample),'goal_frame':9,'anchor_frame':2,
          'checkpoint_sha256':'abc','mean_tcp_error_px':3.5,'tcp':{'valid_gt_count':2}}
    (tmp_path/'metadata.json').write_text(json.dumps(meta))
    for name in ['prediction.png','tcp_overlay.png','target_tcp_overlay.png']:(tmp_path/name).write_bytes(b'image')
    lane=checked_stage1(tmp_path,sample,{})
    assert lane['status']=='ready'
    assert 'target_tcp_overlay.png' in lane['assets']
    assert lane['mean_tcp_error_px']==3.5 and lane['valid_gt_count']==2
    assert lane['coordinate_output_kind']=='stage1_2d_model_output'
    assert lane['gt_coordinate_output_kind']=='physical_gt_projection'
    assert lane['prediction_status']=='compared_with_valid_gt'
    meta['tcp']['valid_gt_count']=0;meta['mean_tcp_error_px']=None
    (tmp_path/'metadata.json').write_text(json.dumps(meta))
    no_gt=checked_stage1(tmp_path,sample,{})
    assert no_gt['status']=='ready' and no_gt['prediction_status']=='unverified_no_gt'
    assert 'artifact availability only' in no_gt['status_semantics']


def frame_sample():
    import torch
    return dict(trace=dict(anchor=169,video_source_indices=[169,173,177,181,185,189,193,197,201],last_source=176),
        instruction_start=108,instruction_end_exclusive=177,granularity='subtask',prompt='Ink stamp on pad',
        report_target=dict(goal_source_frame=176),video_mask=torch.tensor([True,True,False,False,False,False,False,False,False]),
        action=torch.zeros(32,112))


def test_frame_contract_uses_selected_interval_and_distinguishes_padding():
    from scripts.experiments.report_stage12_contract import frame_contract, frame_label_lines
    sample=frame_sample(); frame=frame_contract(sample)
    assert frame['input_frame']==169 and frame['target_frame']==176
    assert frame['instruction_end_exclusive']==177
    assert frame['stage2_horizon_last_frame']==201
    assert frame['stage2_valid_action_last_frame']==176
    assert frame['stage2_last_valid_video_frame']==173
    labels='\n'.join(frame_label_lines(frame))
    for token in ['INPUT f169', 'subtask TARGET f176', 'end_exclusive f177', 'Stage2 horizon f201', 'valid action <= f176', 'video <= f173']:
        assert token in labels


@pytest.mark.parametrize('key,value',[('goal',177),('source_anchor',170)])
def test_frame_contract_rejects_off_by_one_target_or_source_alignment(key,value):
    from scripts.experiments.report_stage12_contract import frame_contract
    sample=frame_sample()
    if key=='goal':sample['report_target']['goal_source_frame']=value
    else:sample['trace']['video_source_indices'][0]=value
    with pytest.raises(ValueError):frame_contract(sample)


def test_label_banner_preserves_coordinate_pixels_and_original_file(tmp_path):
    import numpy as np
    from PIL import Image
    from scripts.experiments.report_stage12_contract import frame_contract,label_display_image
    from scripts.experiments.infer_stage1_samecase import sha
    source=tmp_path/'prediction.png';output=tmp_path/'display.png'
    original=np.zeros((256,320,3),np.uint8);original[32:38,50:56]=[255,0,0]
    Image.fromarray(original).save(source);before=sha(source)
    record=label_display_image(source,output,frame_contract(frame_sample()),role='Stage1 generated image',valid_gt_count=0)
    displayed=np.asarray(Image.open(output))
    assert np.array_equal(displayed[:256],original)
    assert displayed.shape[0]>256 and sha(source)==before
    assert record['image_region_xywh']==[0,0,320,256]
    assert 'NO VALID GT 0/2' in record['labels']
    assert any('Untrusted 2D model output' in line for line in record['labels'])


def test_gt_zero_caption_never_claims_gt_plus_prediction():
    from scripts.experiments.report_stage12_contract import lane_html
    lane=dict(status='ready',assets=['tcp_overlay.png','target_tcp_overlay.png'],valid_gt_count=0,
              goal_frame=176,mean_tcp_error_px=None,prediction_status='unverified_no_gt')
    rendered=lane_html('Stage1',lane)
    assert 'GT + 预测 TCP' not in rendered
    assert '无有效 GT' in rendered
    assert '未验证二维模型输出' in rendered


def test_endpoint_details_preserve_source_definition_without_invented_center():
    from scripts.experiments.report_stage12_contract import endpoint_html
    details=dict(endpoint=dict(definition_id='genrobot_finger_device_tcp_offset_v1',
        used_fields=['observation.states.eef'],transforms=['p_TCP = p_EEF + R_EEF @ [0.1015,-0.0003,-0.0398] m'],
        endpoint_semantics='selected finger-device TCP convention; not human thumb/index midpoint',
        missing_reasons=['left endpoint invalid']),camera=dict(model='double_sphere'),calibration_eligibility=None)
    rendered=endpoint_html(details,'endpoint_audit_v3.json')
    for text in ['物理 TCP 定义','genrobot_finger_device_tcp_offset_v1','observation.states.eef',
                 '0.1015,-0.0003,-0.0398','finger-device TCP convention','left endpoint invalid',
                 'double_sphere','calibration_eligibility','endpoint_audit_v3.json']:
        assert text in rendered
    assert '夹爪中心' not in rendered


def test_source_valid_gt_does_not_clear_physical_contact_review(tmp_path):
    from scripts.experiments.report_stage12_contract import physical_contact_review, physical_review_html, lane_html
    audit={'visual_suspects':{'case018':{'goal_frame':676,'source_repo':'/source/a',
        'visual_observation':'Declared source TCP is beyond visible pad centers.',
        'recommended_report_semantics':'Physical gripper midpoint unverified; no correction justified.'}}}
    sample={'trace':{'source_repo':'/source/a','episode':10},'report_target':{'goal_source_frame':676}}
    review=physical_contact_review(audit,'018',sample,tmp_path)
    assert review['status']=='visual_suspect_unresolved'
    html=physical_review_html(review,'Stage1')
    assert 'beyond visible pad centers' in html and '未认证' in html
    assert '<details' not in html
    lane=dict(status='ready',assets=[],valid_gt_count=2,goal_frame=676,mean_tcp_error_px=0.0)
    visible=lane_html('Stage1',lane)
    assert '源数据有效 GT 2/2' in visible and '物理接触中点未认证' in visible
    other=physical_contact_review(audit,'017',sample,tmp_path)
    assert other['status']=='not_independently_certified'
    assert 'visual_observation' not in other


def test_physical_review_rejects_audit_for_another_target(tmp_path):
    from scripts.experiments.report_stage12_contract import physical_contact_review
    audit={'visual_suspects':{'case012':{'goal_frame':247,'visual_observation':'suspect'}}}
    sample={'trace':{'episode':24},'report_target':{'goal_source_frame':248}}
    with pytest.raises(ValueError,match='physical review target'):
        physical_contact_review(audit,'012',sample,tmp_path)
