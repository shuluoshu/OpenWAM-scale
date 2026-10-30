from copy import deepcopy
import numpy as np
import pytest
import torch
from PIL import Image
from scripts.experiments.infer_stage1_samecase import sample_identity, assert_same_case, make_batch, target_data


def sample():
    return dict(trace=dict(identity=['repo', 4, 10, 'task', [0, 30], [0, 20]], source_repo='/data/repo', episode=4, anchor=10),
        prompt='Pick up the cup', instruction='Pick up the cup', granularity='task', instruction_start=0, instruction_end_exclusive=30,
        goal_observed_multiview=torch.zeros(3, 1, 384, 320), goal_view_valid=torch.tensor([True, False, False]),
        history_images=torch.zeros(10, 3, 1, 256, 320), history_valid=torch.zeros(10,dtype=torch.bool),
        history_ages=torch.zeros(10), history_episode_id=torch.tensor(4),
        report_target=dict(goal_source_frame=29, projection=dict(uv=np.zeros((2,2)), projection_valid=np.array([False,True]), in_frame=np.array([False,True]))),
        _eval=dict(family='test', report_goal=Image.new('RGB',(640,480))))


def test_identity_rejects_same_family_different_anchor_prompt_or_pixels():
    s=sample(); identity=sample_identity(s)
    for mutate in (lambda x:x['trace'].update(anchor=11), lambda x:x.update(prompt='Put down the cup'),
                   lambda x:x['goal_observed_multiview'].__setitem__((0,0,0,0),1),
                   lambda x:x['history_images'].__setitem__((0,0,0,0,0),1)):
        other=deepcopy(s); mutate(other)
        with pytest.raises(ValueError): assert_same_case(other,identity)


def test_stage1_inference_batch_has_only_exact_conditioning_and_no_future():
    s=sample(); contexts={s['prompt']:(torch.ones(1,512,8),torch.tensor([5]))}
    b=make_batch(s,contexts,device='cpu',dtype=torch.float32)
    assert torch.equal(b['observed'][0],s['goal_observed_multiview'])
    assert torch.equal(b['history_images'][0],s['history_images'])
    assert not {'targets','video','action','report_target','report_goal'} & b.keys()


def test_gt_requires_instruction_endpoint_and_masks_placeholder():
    s=sample(); data=target_data(s,[[.3,.4],[.5,.6]])
    assert data['arms'][0]['gt_uv'] is None
    assert data['arms'][1]['gt_valid']
    s['report_target']['goal_source_frame']=19
    with pytest.raises(ValueError,match='instruction endpoint'): target_data(s,[[.3,.4],[.5,.6]])


def test_identity_is_a_snapshot_not_mutable_trace_alias():
    s=sample(); identity=sample_identity(s)
    s['trace']['identity'][0]='another/repo'
    with pytest.raises(ValueError): assert_same_case(s,identity)
