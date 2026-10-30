import numpy as np
from PIL import Image
from scripts.experiments import report_stage12_timed as report


def payload():
    return dict(source_frames=[100,104],action_indices=[-1,3],
        pred_uv=np.array([[[np.nan,np.nan]]*2,[[80,70],[140,110]]]),
        gt_uv=np.array([[[20,30],[40,50]],[[20,30],[40,50]]]),
        gt_in_frame=np.ones((2,2),bool),pred_in_frame=np.array([[False,False],[True,True]]),
        pred_projectable=np.array([[False,False],[True,True]]),
        metric_valid=np.array([[False,False],[True,True]]),
        pixel_error=np.array([[np.nan,np.nan],[72.1,116.6]]),
        reasons=['',''])


def renderer():
    fn=getattr(report,'render_tcp_video_frame',None)
    assert callable(fn), 'the moving video needs an actual TCP overlay renderer'
    return fn


def test_direct_tcp_points_are_on_both_actual_video_panels():
    im=Image.new('RGB',(320,384),'black')
    out=renderer()(im,im,payload(),1,'DIRECT TCP')
    assert out.size==(640,480)
    a=np.asarray(out)
    for offset in (0,320):
        red=a[96+62:96+79,offset+72:offset+89]
        assert ((red[:,:,0]>180)&(red[:,:,1]<100)&(red[:,:,2]<100)).any()
        green=a[96+22:96+39,offset+12:offset+29]
        assert ((green[:,:,1]>150)&(green[:,:,0]<100)).any()


def test_current_observation_has_gt_but_no_invented_prediction():
    im=Image.new('RGB',(320,384),'black')
    a=np.asarray(renderer()(im,None,payload(),0,'DIRECT TCP'))[96:]
    assert a.shape==(384,320,3)
    assert not ((a[:,:,0]>180)&(a[:,:,1]<100)&(a[:,:,2]<100)).any()
    assert ((a[:,:,1]>150)&(a[:,:,0]<100)).any()


def test_outside_predictions_are_not_clamped_to_image_border():
    p=payload();p['pred_uv'][1]=[[400,70],[-20,40]];p['pred_in_frame'][1]=False
    im=Image.new('RGB',(320,384),'black')
    a=np.asarray(renderer()(im,None,p,1,'DIRECT TCP'))[96:]
    assert not ((a[:,:,0]>180)&(a[:,:,1]<100)&(a[:,:,2]<100)).any()
