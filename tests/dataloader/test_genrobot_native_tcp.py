import copy
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from openwam.dataloader.native_stage2_profiles import resolve_native_profile
from openwam.dataloader.utils.mora_relative import MoraHeadTCPRepresentation
from scripts.data.trajectory_projection import goal_source_tcp


def info():
    return {'features':{'observation.states.eef':{'shape':[20]},'actions.eef':{'shape':[20]},'observation.images.cam_high':{'dtype':'video','shape':[32,40,3]}},'starvla_metadata':{'source':'genrobot','device_type':'ego_finger','eef_anchor':'finger_device_base_link','eef_coordinate_frame':'world'}}


def test_genrobot_tcp_matches_existing_projection_and_preserves_device_axes():
    i=info();p=resolve_native_profile('Genrobot/test',i);rep=MoraHeadTCPRepresentation(i,**p['representation'])
    rot=Rotation.from_euler('xyz',[.4,-.2,.7]).as_matrix();r6=rot[:,:2].T.reshape(-1)
    eef=np.array([*[.2,.3,.4],*r6,.5,*[.7,-.2,.8],*r6,.6],np.float32)
    cols={'observation.states.eef':np.stack([eef,eef+.01,eef+.02])};values,valid=rep._absolute_values(cols)
    expected=goal_source_tcp('Genrobot',{'observation.states.eef':eef})
    np.testing.assert_allclose(values[0,[0,1,2,10,11,12]].reshape(2,3),expected['tcp_world'],atol=1e-7)
    np.testing.assert_allclose(values[0,3:9],r6,atol=1e-7)
    assert valid[0,:9].all() and valid[0,10:19].all()
    action,mask,_,_=rep.encode_window(cols)
    assert mask[:,:9].all() and mask[:,10:19].all()
    assert not mask[:,[9,19]].any(), 'measured opening is not a controller command'
    cols['observation.states.eef'][1,0]=np.nan
    _,mask,_,_=rep.encode_window(cols)
    assert not mask[0,:9].any() and mask[0,10:19].all()


@pytest.mark.parametrize('field,value',[('source','other'),('device_type','robot_gripper'),('eef_anchor','wrist'),('eef_coordinate_frame','robot')])
def test_genrobot_rejects_unverified_metadata(field,value):
    i=info();i['starvla_metadata'][field]=value
    with pytest.raises(ValueError,match='unverified_genrobot_tcp'):resolve_native_profile('Genrobot/test',i)


def test_genrobot_rejects_missing_or_nondual_eef_declaration():
    for shape in (None,[10],[19]):
        i=info();i['features']['observation.states.eef']['shape']=shape
        with pytest.raises(ValueError,match='unverified_genrobot_tcp'):resolve_native_profile('Genrobot/test',i)


def test_genrobot_stored_pose_mask_invalidates_full_pose_without_erasing_other_device():
    from openwam.dataloader.native_stage2_io import merge_source_masks
    i=info();p=resolve_native_profile('Genrobot/test',i);rep=MoraHeadTCPRepresentation(i,**p['representation'])
    pose=[0.,0.,0.,1.,0.,0.,0.,1.,0.,.5]
    cols={'observation.states.eef':np.tile(pose*2,(3,1)),
          'observation.states.eef.mask':np.ones((3,20)),
          'actions.eef.mask':np.ones((3,20))}
    cols['observation.states.eef.mask'][1,4]=0
    _,mask,_,sm=rep.encode_window(cols)
    merge_source_masks(rep,cols,mask,sm)
    assert not mask[0,:9].any() and mask[0,10:19].all()
    assert mask[1,:9].all() and mask[1,10:19].all()
    assert not mask[:,[9,19]].any()
