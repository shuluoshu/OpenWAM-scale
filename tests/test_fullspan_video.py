import numpy as np
from PIL import Image
import torch
from scripts.experiments.fullspan_video import compose_frame, complete_anchors


def fixture():
    sample=dict(goal_view_valid=torch.tensor([True,False,True]),
        goal_observed_multiview=torch.zeros(3,1,384,320),
        _eval={'report_goal':Image.new('RGB',(320,256),'blue')},
        action_mask=torch.ones(32,112,dtype=torch.bool),trace={'anchor':10},
        instruction_end_exclusive=45,granularity='subtask',prompt='pick')
    point=dict(gt_valid=True,gt_px=[100.,100.],pred_drawable=True,pred_px=[120.,100.],side='L')
    tcp=dict(arms=[point],valid_gt_count=1)
    views={}
    for role in ['head','left_wrist','right_wrist']:
        points=np.full((32,2,2),np.nan);points[0,0]=[60,70]
        mask=np.zeros((32,2),bool);mask[0,0]=True
        branch=dict(gt_uv=points,pred_uv=points,gt_in_frame=mask,pred_in_frame=mask)
        views[role]=dict(image_available=role!='left_wrist',projection_available=True,direct=branch,reason='')
    return sample,tcp,dict(views=views)


def test_complete_interval_includes_last_frame():
    assert complete_anchors(4,29)==[4,14,24,28]
    assert complete_anchors(4,25)==[4,14,24]


def test_two_rows_current_cameras_and_goal_are_separate():
    sample,tcp,projection=fixture()
    frame,meta=compose_frame(sample,Image.new('RGB',(320,256),'yellow'),tcp,projection,'none')
    assert frame.size==(1280,704)
    pixels=np.asarray(frame)
    assert np.all(pixels[110:340,330:630]==0)  # absent GT wrist remains black
    assert np.all(pixels[414:644,330:630]==0)  # absent predicted wrist remains black
    assert tuple(pixels[150,1000])==(0,0,255) # real goal top
    assert tuple(pixels[454,1000])==(255,255,0) # predicted goal bottom
    assert meta['input_frame']==10 and meta['target_frame']==44
    assert meta['chunk_horizon']==32


def test_projection_marker_is_in_each_current_view_not_keyframe():
    sample,tcp,projection=fixture()
    frame,meta=compose_frame(sample,Image.new('RGB',(320,256),'yellow'),tcp,projection,'joint')
    pixels=np.asarray(frame)
    assert (pixels[96+66:96+75,56:65,1]>180).any()
    assert (pixels[400+66:400+75,56:65,0]>180).any()
    assert meta['views']['left_wrist']['image_available'] is False


def test_absent_goal_image_is_black_without_hiding_valid_wrist_or_prediction():
    sample,tcp,projection=fixture()
    sample['_eval']['report_goal']=None;sample['goal_view_valid'][0]=False
    frame,meta=compose_frame(sample,Image.new('RGB',(320,256),'yellow'),tcp,projection,'none')
    pixels=np.asarray(frame)
    assert not pixels[96:352,960:1280].any()
    assert tuple(pixels[454,1000])==(255,255,0)
    assert pixels[110:340,650:950].any() # real right wrist survives
    assert meta['goal_image_available'] is False
    # A missing target never gives a red/verified learned point, even if caller's mask was stale.
    assert (pixels[496:505,1076:1085,1]>85).any()
