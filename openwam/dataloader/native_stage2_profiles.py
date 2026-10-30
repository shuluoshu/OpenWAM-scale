"""Audited native source mappings, independent of any fixture or selected clip."""
from __future__ import annotations
import hashlib
import json
import numpy as np
from openwam.dataloader.native_cameras import HEAD_CAMERAS, resolve_native_cameras
from openwam.dataloader.utils.gripper import GripperCommands

VERSION = 'native_mora112_v1'
SIDES = ('left', 'right')
HUMANS = {'LightWheel', 'mecka', 'egodex', 'egoverse'}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def names(feature):
    result=feature.get('names') or []
    return result.get('motors',[]) if isinstance(result,dict) else result


def resolve_native_profile(repo, info):
    from openwam.dataloader.posttrain_physical import is_posttrain_native, resolve_posttrain_profile
    if is_posttrain_native(info):return resolve_posttrain_profile(repo,info)
    family=repo.split('/')[0]; f=info['features']; m=info.get('starvla_metadata',{})
    if family not in HEAD_CAMERAS:raise ValueError(f'{repo}: unsupported_family')
    robot=m.get('robot_type',m.get('canonical_robot_type',m.get('embodiment',family)))
    reasons=[]
    cameras=resolve_native_cameras(repo,info)
    head=cameras[0]
    joint_key='observation.state' if family in {'abc_130k','robomind_v1','robomind_v2'} else 'observation.states.joint'
    if family in HUMANS or family=='Genrobot':joint_key='__unavailable_robot_joint__'
    indices={s:[] for s in SIDES}; j=f.get(joint_key)
    if j:
        unit=j.get('unit',m.get('arm_joint_unit',m.get('joint_unit')))
        if unit not in {'radian','radians','rad'}:raise ValueError(f'{repo}: unverified_joint_unit: {joint_key}={unit}')
        ns=names(j)
        if len(ns)!=j['shape'][0]:raise ValueError(f'{repo}: unverified_joint_order: {joint_key}')
        if robot=='robomind_franka_fr3_dual':
            expected=[*[f'puppet_joint_position_{i}' for i in range(7)],'left_gripper',*[f'puppet_joint_position_{i}' for i in range(8,15)],'right_gripper']
            if ns!=expected:raise ValueError(f'{repo}: unverified_joint_order: FR3')
            indices={'left':list(range(7)),'right':list(range(8,15))}
        else:
            for side,short in [('left','L'),('right','R')]:
                candidates=([f'{side}_joint_{i}' for i in range(7)],
                            [f'puppet_joint_position_{side}_{i}' for i in range(7)],
                            [f'{short}arm{i}_Joint' for i in range(1,8)])
                ordered=next(([n for n in seq if n in ns] for seq in candidates if seq[0] in ns),[])
                if ordered and ordered not in [seq[:len(ordered)] for seq in candidates]:
                    raise ValueError(f'{repo}: unverified_joint_order: {side}')
                indices[side]=[ns.index(n) for n in ordered]
            accounted={i for values in indices.values() for i in values}
            if any(i not in accounted and 'grip' not in str(n).lower() for i,n in enumerate(ns)):
                raise ValueError(f'{repo}: unverified_joint_order: {ns}')
    rep=dict(state_key='observation.states.eef',joint_state_key=joint_key,joint_indices=indices,
             tcp_pose_keys={},tcp_from_eef={},gripper_state_sources={})
    norm=m.get('gripper_normalization',{}); canonical=norm.get('convention')=='0=closed, 1=open'
    # EEF and joint grippers can have different units in the same native file.
    for side_id,side in enumerate(SIDES):
        source=None
        for key in [joint_key,'observation.state','observation.states.eef']:
            feature=f.get(key,{})
            verified=feature.get('gripper_unit')=='openness_0_closed_1_open' or (canonical and key=='observation.states.eef')
            if family=='robomind_v2' and key=='observation.state':verified=canonical
            if robot=='robomind_franka_fr3_dual':verified=False
            if not verified:continue
            ns=names(feature); index=None
            if key=='observation.states.eef' and feature.get('shape') in ([10],[20]):
                index=9+side_id*10
                if index>=feature['shape'][0]:continue
            else:
                matches=[i for i,n in enumerate(ns) if n in (side+'_gripper',('L' if side=='left' else 'R')+'gripper_Joint')]
                if len(matches)==1:index=matches[0]
            if index is not None:
                source=dict(key=key,index=index,provenance='native metadata declares 0=closed,1=open');break
        if source:rep['gripper_state_sources'][side]=source
    if len(rep['gripper_state_sources'])<2:reasons.append('unverified_gripper_state')
    if family=='Genrobot':
        required={'source':'genrobot','device_type':'ego_finger',
                  'eef_anchor':'finger_device_base_link','eef_coordinate_frame':'world'}
        if any(m.get(k)!=v for k,v in required.items()) or f.get('observation.states.eef',{}).get('shape')!=[20]:
            raise ValueError(f'{repo}: unverified_genrobot_tcp: require dual world-frame finger_device_base_link EEF')
        # Audited finger-device endpoint, matching trajectory_projection.goal_source_tcp
        # and outputs/fk_consistency/genrobot_wrist_audit/audit.json. TCP axes inherit
        # the finger-device frame; this is not a human fingertip landmark and does
        # not infer controller-command gripper semantics or physical calibration.
        transform=np.eye(4);transform[:3,3]=[.1015,-.0003,-.0398]
        rep['tcp_from_eef']={side:transform.tolist() for side in SIDES}
    if family=='interndata_a1':
        rep['tcp_pose_keys']={s:f'states.{s}_tcp_to_robot_pose' for s in SIDES if f'states.{s}_tcp_to_robot_pose' in f}
        rep['tcp_reference_to_world_key']='states.robot_to_env_pose'
    if family in {'abc_130k','robochallenge_v2'}:
        hist=m.get('rectification_history',[])
        entry=next((h for h in hist if h.get('schema_version')=='eef_tcp_to_wrist_0804_v1'),None)
        if entry:
            basis=np.eye(4)
            if family=='abc_130k' and any(h.get('schema_version')=='abc_130k_eef_axis_x_forward_z_up_v1' for h in hist):
                basis[:3,:3]=np.eye(3)[:,[2,0,1]]
            for t in entry['transforms']:
                if t.get('source_anchor') not in {'two_fingertip_contact_center','tcp_gripper_center'}:
                    raise ValueError(f'{repo}: unverified_tcp_action_mapping')
                rep['tcp_from_eef'][t['arm']]=(np.linalg.inv(basis)@np.linalg.inv(np.array(t['T_source_wrist_row_major']).reshape(4,4))).tolist()
    if not rep['tcp_pose_keys'] and not rep['tcp_from_eef']:reasons.append('missing_tcp_action_mapping')
    calibration='observation.states.camera.world_to_camera'
    head_ok=head is not None and calibration in f and family not in {'robochallenge_v1','robomind_v1','robomind_v2'}
    rep.update(head_camera_available=head_ok,head_world_to_camera_key=calibration)
    if not head_ok:reasons.append('head_calibration_ineligible')
    commands=GripperCommands(info)
    if not commands.sources:reasons.append('unverified_gripper_command_semantic')
    fk=None
    if family=='interndata_a1' and robot=='Genie-1':
        arms={}
        for side,short,base in [('left','l',21),('right','r',61)]:
            key=f'states.{side}_tcp_to_{side}_armbase_pose'
            if key in f and len(indices[side])==7:
                transform=np.eye(4);transform[2,3]=.22
                arms[side]=dict(urdf_path='/mnt/data/cpfs/b5big/pretrain_urdfs/interndata_a1/G1_120s/G1_120s.urdf',base_link='arm_base_link',target_link=f'arm_{short}_end_link',joint_names=[f'idx{base+i}_arm_{short}_joint{i+1}' for i in range(7)],endpoint='tcp',endpoint_transform=transform.tolist(),gt_pose_key=key,gt_frame='urdf_base')
        if arms:fk=dict(arms=arms)
    if fk is None:reasons.append('missing_fk_profile')
    result=dict(version=VERSION,family=family,robot_type=robot,representation=rep,cameras=cameras,
                fk_supervision=fk,mask_reasons=reasons,tracking_confidence_threshold=.8)
    result['profile_id']=fingerprint(result)
    return result
