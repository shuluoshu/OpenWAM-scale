"""Native recorded joint/command semantics and verified geometry boundaries."""
import copy
import numpy as np
import pytest
from openwam.dataloader.native_stage2_profiles import resolve_native_profile
from openwam.dataloader.utils.mora_relative import MoraHeadTCPRepresentation

NAMES = [[f'{s}arm{i}_Joint' for i in range(1,8)] + [f'{s}gripper_Joint'] for s in ('L','R')]
NAMES = sum(NAMES, []) + [f'{s}{part}_Joint' for s in ('LF','RF','RR','LR') for part in ('Steer','Hub')] + ['Waist1_Joint','Waist2_Joint','Waist3_Joint','Head1_Joint','Head2_Joint']


def info_fixture(width=29):
    features={k:{'shape':[29],'names':NAMES} for k in ('observation.state.position','action.position')}
    features['observation.state']={'shape':[width],'names':NAMES if width==29 else [f'converted_{i}' for i in range(width)]}
    features['observation.images.head_undistorted']={'shape':[100,100,3],'dtype':'video'}
    features['observation.odom_vehicle_absolute']={'shape':[13]}
    result={'features':features,'fps':30,'robot_type':'generic'}
    if width==34:result['eef_conversion']={'eef_type':'flange','tcp_offset_m':0,'tcp_pose_semantics':'T_base_eef_t','gripper_action_source':'cmd','layout':'joint_eef'}
    if width==64:result['mobile_manipulation_64_conversion']={'source_dimension':29,'head_eef_zero_padding':True,'tcp_pose_reference_frame':'episode_start_tcp'}
    return result


def columns_fixture():
    q=np.zeros((3,29));q[:,0]=[0,.1,.2];q[:,7]=[1.01,.3,.4];q[:,15]=[-.1,.5,.6]
    cmd=q.copy();cmd[:,[7,15]]=[[.2,.8],[.4,.6],[.9,.1]]
    odom=np.zeros((3,13));odom[:,6]=1;odom[:,0]=[10,11,12]
    return {'observation.state.position':q,'action.position':cmd,'observation.odom_vehicle_absolute':odom}


def test_native_profile_keeps_named_joint_and_recorded_gripper_targets():
    info=info_fixture();profile=resolve_native_profile('/native/laundry',info)
    from openwam.dataloader.posttrain_physical import augment_posttrain_episode
    info,cols,profile,_=augment_posttrain_episode(info,{},columns_fixture(),profile)
    rep=MoraHeadTCPRepresentation(info,**profile['representation'])
    action,mask,state,sm=rep.encode_window(cols)
    np.testing.assert_allclose(action[:,20],[.1,.2])
    np.testing.assert_allclose(action[:,[9,19]],[[.2,.8],[.4,.6]])
    np.testing.assert_allclose(state[0,[9,19]],[1,0])
    assert mask[:,20:34].all() and mask[:,[9,19]].all()
    assert not mask[:,:9].any() and not mask[:,34:].any()
    assert rep.unified_dim==112


def test_unverified_native_joint_order_rejected():
    info=info_fixture();info['features']['observation.state.position']['names']=NAMES[:-1]+['invented']
    with pytest.raises(ValueError,match='native_joint_schema'):
        resolve_native_profile('/native/laundry',info)


def calibration_fixture(tmp_path):
    root=tmp_path/'robot/MZ2501_V1.2';device=root/'device/TEST-1';device.mkdir(parents=True)
    (device/'device_extrinsic.yaml').write_text('device: TEST-1\n')
    links=['base_link','head01_link','head02_link'];joints=[]
    for short in ('L','R'):
        parent='base_link'
        for i in range(1,8):
            child=f'{short}arm{i:02}_link';links.append(child)
            joints.append(f'<joint name="{short}arm{i}_Joint" type="revolute"><parent link="{parent}"/><child link="{child}"/><axis xyz="0 0 1"/><origin xyz="0 0 0.1"/></joint>');parent=child
        links.append(f'{short}arm08_link');joints.append(f'<joint name="{short}tool" type="fixed"><parent link="{parent}"/><child link="{short}arm08_link"/></joint>')
    for i in (1,2):joints.append(f'<joint name="Head0{i}_Joint" type="revolute"><parent link="'+('base_link' if i==1 else 'head01_link')+f'"/><child link="head0{i}_link"/><axis xyz="0 0 1"/></joint>')
    (root/'robot_description.urdf').write_text('<robot name="fixture">'+''.join(f'<link name="{x}"/>' for x in links)+''.join(joints)+'</robot>')
    ep={'source_group_id':'TEST-1','calibration_intrinsic':{'channels':{'head':{'dir_name':'head_undistorted','undistortion':{'new_k':[[50,0,50],[0,50,50],[0,0,1]],'view_rotation_applied':False}}}},'calibration_extrinsic':{'sources':{'record':{'extrinsic':{'positions':{'head':{'transforms':[{'target_frame':'camera4','source_frame':'head01_link','T_target_source':np.eye(4).tolist()}]}}}}}}}
    return tmp_path,ep


@pytest.mark.parametrize('width',[29,34,64])
def test_verified_fk_uses_odom_and_ignores_converted_local_delta(tmp_path,width):
    from openwam.dataloader.posttrain_physical import augment_posttrain_episode
    info=info_fixture(width);profile=resolve_native_profile('/native/demo',info);columns=columns_fixture()
    columns['observation.state']=np.full((3,width),12345.)
    calibration,ep=calibration_fixture(tmp_path)
    augmented,cols,profile,geometry=augment_posttrain_episode(info,ep,columns,profile,calibration_root=calibration)
    rep=MoraHeadTCPRepresentation(augmented,**profile['representation']);action,mask,state,sm=rep.encode_window(cols)
    np.testing.assert_allclose(action[:,0],[1,2],atol=1e-6)
    assert mask[:,:9].all() and geometry['reference']=='episode_start_base'
    assert bool(mask[:,34:].any()) is (width!=64)
    assert profile['fk_supervision'] is None  # Independent FK loss GT is not invented from q.


def test_missing_geometry_preserves_joint_and_grip(tmp_path):
    from openwam.dataloader.posttrain_physical import augment_posttrain_episode
    info=info_fixture();profile=resolve_native_profile('/native/demo',info)
    updated,cols,p,g=augment_posttrain_episode(info,{},columns_fixture(),profile,calibration_root=tmp_path)
    action,mask,_,_=MoraHeadTCPRepresentation(updated,**p['representation']).encode_window(cols)
    assert mask[:,20:34].all() and mask[:,[9,19]].all()
    assert not mask[:,:9].any() and not mask[:,34:].any()
    assert 'missing_device_urdf' in p['mask_reasons']


def test_invalid_odom_masks_affected_pose_rows_only(tmp_path):
    from openwam.dataloader.posttrain_physical import augment_posttrain_episode
    info=info_fixture();profile=resolve_native_profile('/native/demo',info);cols=columns_fixture();cols['observation.odom_vehicle_absolute'][1,3:7]=0
    calibration,ep=calibration_fixture(tmp_path)
    info,cols,p,_=augment_posttrain_episode(info,ep,cols,profile,calibration_root=calibration)
    _,mask,_,_=MoraHeadTCPRepresentation(info,**p['representation']).encode_window(cols)
    assert not mask[0,:9].any() and mask[1,:9].all()
    assert mask[:,20:34].all()


def test_projection_requires_verified_calibration_and_does_not_clamp(tmp_path):
    from openwam.dataloader.posttrain_physical import project_posttrain_goal
    info=info_fixture();cols=columns_fixture();calibration,ep=calibration_fixture(tmp_path)
    xy,valid,provenance=project_posttrain_goal(info,ep,cols,0,'observation.images.head_undistorted',calibration_root=calibration)
    assert valid.all();np.testing.assert_allclose(xy,[[50,50],[50,50]])
    _,valid,_=project_posttrain_goal(info,{},cols,0,'observation.images.head_undistorted',calibration_root=calibration)
    assert not valid.any()


def test_episode_geometry_does_not_change_stats_contract(tmp_path):
    from openwam.dataloader.posttrain_physical import augment_posttrain_episode, representation_info
    info=info_fixture();profile=resolve_native_profile('/native/demo',info)
    header_rep=MoraHeadTCPRepresentation(representation_info(info,profile),**profile['representation'])
    calibration,ep=calibration_fixture(tmp_path)
    info,cols,p,_=augment_posttrain_episode(info,ep,columns_fixture(),profile,calibration_root=calibration)
    episode_rep=MoraHeadTCPRepresentation(info,**p['representation'])
    assert p['profile_id']==profile['profile_id']
    assert header_rep.contract==episode_rep.contract


def test_native_physical_dependency_hash_catches_calibration_change(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from openwam.dataloader.posttrain_physical import physical_dependency_hash
    info=info_fixture();root=tmp_path/'repo';(root/'meta/episodes/chunk-000').mkdir(parents=True)
    (root/'meta/info.json').write_text(__import__('json').dumps(info))
    file=root/'meta/episodes/chunk-000/file-000.parquet'
    pq.write_table(pa.Table.from_pylist([{'episode_index':0,'source_group_id':'TEST-1','calibration_intrinsic':'a'}]),file)
    calibration,_=calibration_fixture(tmp_path/'calibration')
    first=physical_dependency_hash(root,calibration_root=calibration)
    pq.write_table(pa.Table.from_pylist([{'episode_index':0,'source_group_id':'TEST-1','calibration_intrinsic':'b'}]),file)
    assert physical_dependency_hash(root,calibration_root=calibration)!=first


def test_invalid_head_calibration_masks_camera_without_erasing_physical_joints(tmp_path):
    from openwam.dataloader.posttrain_physical import augment_posttrain_episode
    info=info_fixture();profile=resolve_native_profile('/native/demo',info);calibration,ep=calibration_fixture(tmp_path)
    ep['calibration_extrinsic']['sources']['record']['extrinsic']['positions']['head']['transforms'][0]['T_target_source'][0][0]=2
    updated,cols,p,g=augment_posttrain_episode(info,ep,columns_fixture(),profile,calibration_root=calibration)
    _,mask,_,_=MoraHeadTCPRepresentation(updated,**p['representation']).encode_window(cols)
    assert mask[:,20:34].all() and mask[:,:9].all()
    assert not mask[:,34:].any()
    assert 'invalid_episode_head_calibration' in p['mask_reasons']
