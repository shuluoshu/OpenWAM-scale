import numpy as np
import pytest


def test_goal_camera_motion_outside_behind_and_scaling():
    from scripts.data.trajectory_projection import project_goal_tcp
    k=np.array([[100.,0,320],[0,100,240],[0,0,1]])
    points=np.array([[0.,0,2],[1.,0,2],[0.,0,-1],[20.,0,1]])
    e=np.eye(4);e[0,3]=-1
    r=project_goal_tcp(points,e,k,(640,480))
    np.testing.assert_allclose(r['uv'][0],[(270.5*.5-.5)/319,(240.5*256/480-.5)/255],atol=1e-5)
    assert r['projection_valid'].tolist()==[True,True,False,True]
    assert r['in_frame'].tolist()==[True,True,False,False]
    assert (r['visibility']==-1).all()
    with pytest.raises(ValueError):project_goal_tcp(points,e,k,(0,480))


@pytest.mark.parametrize('family,key,count,thumb,index,wrist',[
    ('LightWheel','observation.states.hand.position',21,4,8,0),
    ('mecka','observation.states.hand.position',21,4,8,0),
    ('egoverse','observation.states.joints',21,0,1,5),
    ('egodex','observation.states.finger.tip.position',5,4,0,None)])
def test_human_adapter_hand_order_and_tracking(family,key,count,thumb,index,wrist):
    from scripts.data.trajectory_projection import goal_source_tcp
    hand=np.zeros((2,count,3));hand[:,thumb]=[[1,0,2],[3,0,2]];hand[:,index]=[[1,2,2],[3,2,2]]
    row={key:hand.ravel(), 'observation.states.hand.valid':[True,False]}
    if family=='egodex':row['observation.states.hand.position']=np.zeros(6)
    r=goal_source_tcp(family,row)
    np.testing.assert_allclose(r['tcp_world'][0],[1,1,2])
    assert r['tcp_valid'].tolist()==[True,False]
    assert r['definition_id']=='human_thumb_index_midpoint_v1'


def test_morphi_offset_is_released_local_x_and_missing_human_stays_masked():
    from scripts.data.trajectory_projection import goal_source_tcp
    pose=np.array([[1,2,3,0,1,0,0,0,1,0]]*2)
    r=goal_source_tcp('morphirobot0930',{'observation.states.eef':pose.ravel()})
    np.testing.assert_allclose(r['tcp_world'],[[1,2.23116,3]]*2)
    assert not goal_source_tcp('LightWheel',{})['tcp_valid'].any()


def test_genrobot_uses_existing_device_tcp_convention_and_per_hand_alignment():
    from scripts.data.trajectory_projection import goal_source_tcp
    # Left identity; right rotates local X to world Y. Compute expected points
    # independently from the established device-frame offset.
    poses=np.array([[1,2,3,1,0,0,0,1,0,0], [4,5,6,0,1,0,-1,0,0,0.]])
    row={'observation.states.eef':poses.ravel(),
         'observation.states.eef.mask':np.ones(20), 'canonical.frame_valid':[1],
         'raw.robot1.finger.eef_pose.alignment_valid':[1],
         'raw.robot2.finger.eef_pose.alignment_valid':[1]}
    r=goal_source_tcp('Genrobot',row)
    assert r['tcp_valid'].tolist()==[True,True]
    np.testing.assert_allclose(r['tcp_world'],[[1.1015,1.9997,2.9602],[4.0003,5.1015,5.9602]])
    assert r['definition_id']=='genrobot_finger_device_tcp_offset_v1'
    row['raw.robot2.finger.eef_pose.alignment_valid']=[0]
    r=goal_source_tcp('Genrobot',row)
    assert r['tcp_valid'].tolist()==[True,False]
    assert np.isnan(r['tcp_world'][1]).all()


def test_decode_goal_uses_exact_ordinal_with_nonzero_origin_and_nonuniform_pts(tmp_path):
    import av
    from fractions import Fraction
    from scripts.data.prepare_goal_labels import decode_goal
    path=tmp_path/'timing.mp4'
    pts=[90,93,99,102]
    with av.open(str(path),'w') as container:
        stream=container.add_stream('mpeg4',rate=30);stream.width=16;stream.height=16
        stream.pix_fmt='yuv420p';stream.time_base=Fraction(1,90)
        for i,p in enumerate(pts):
            frame=av.VideoFrame.from_ndarray(np.full((16,16,3),i*50,np.uint8),format='rgb24')
            frame.pts=p;frame.time_base=Fraction(1,90)
            for packet in stream.encode(frame):container.mux(packet)
        for packet in stream.encode():container.mux(packet)
    image,metadata=decode_goal(path,0,2)
    assert abs(np.array(image).mean()-100)<5
    assert metadata['video_ordinal']==2
    assert metadata['video_pts_seconds']>1
    assert metadata['decode_alignment']=='sequential_ordinal'
    assert metadata['nonuniform_pts_seen']


def test_decode_nonzero_offset_rejects_vfr_with_coarse_clock(monkeypatch):
    import av
    from fractions import Fraction
    from types import SimpleNamespace
    from PIL import Image
    from scripts.data.prepare_goal_labels import decode_goal
    stream=SimpleNamespace(average_rate=Fraction(30),time_base=Fraction(1,30),start_time=0)
    class Container:
        streams=SimpleNamespace(video=[stream])
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def decode(self,stream):
            for pts in [0,1,3,4]:
                yield SimpleNamespace(pts=pts,to_image=lambda:Image.new('RGB',(8,8)))
    monkeypatch.setattr(av,'open',lambda _:Container())
    with pytest.raises(ValueError,match='VFR time offset'):
        decode_goal('coarse-clock.mp4',1/30,2)
