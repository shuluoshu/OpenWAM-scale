"""Audited Morphi native 29/34/64 adapter (metres, radians, openness).

Uses the named preserved ROS joint fields and recorded controller commands.
Geometry follows outputs/posttraining_addition_20261023/tcp_geometry.py: device
URDF, arm08 + 0.23116 m, embedded episode head calibration, recorded odometry.
Converted local deltas and visualization-only source relocation are never read.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
from openwam.kinematics import rigid_matrix

CALIBRATION_ROOT = Path('/mnt/data/nas/B_zone_nas/jeffry/calibration_results')
SIDES = ('left', 'right')
NATIVE_NAMES = sum(([f'{s}arm{i}_Joint' for i in range(1,8)]+[f'{s}gripper_Joint'] for s in ('L','R')), []) + [f'{s}{part}_Joint' for s in ('LF','RF','RR','LR') for part in ('Steer','Hub')] + ['Waist1_Joint','Waist2_Joint','Waist3_Joint','Head1_Joint','Head2_Joint']
TCP_KEYS = {s: f'_posttrain_tcp_{s}' for s in SIDES}
HEAD_KEY = '_posttrain_head_world_to_camera'
ALIASES = {'Head01_Joint':'Head1_Joint','Head02_Joint':'Head2_Joint'}


def _names(feature):
    names=feature.get('names') or []
    return names.get('motors',[]) if isinstance(names,dict) else names


def is_posttrain_native(info):
    if info.get('starvla_metadata'):return False
    return info.get('features',{}).get('observation.state',{}).get('shape') in ([29],[34],[64])


def _source_keys(info):
    f=info['features'];q='observation.state.position' if 'observation.state.position' in f else 'observation.state'
    if f.get(q,{}).get('shape') != [29] or _names(f[q]) != NATIVE_NAMES:
        raise ValueError('native_joint_schema: requires audited named Morphi29 preserved joints')
    # Same-tick recorded command, not converted action or next measured state.
    cmd='action.position' if 'action.position' in f else 'action'
    if f.get(cmd,{}).get('shape') != [29] or _names(f[cmd]) != NATIVE_NAMES:
        raise ValueError('native_command_schema: requires named recorded Morphi29 command')
    for key in (q,cmd):
        if f[key].get('unit','radian') not in ('rad','radian','radians'):
            raise ValueError('native_joint_unit: incompatible with audited ROS joint schema')
    return q,cmd


def posttrain_columns(info):
    q,cmd=_source_keys(info)
    return [q,cmd]+[k for k in ('observation.odom_vehicle_absolute','canonical.frame_valid','training.valid') if k in info['features']]


def resolve_posttrain_profile(repo,info):
    from openwam.dataloader.native_cameras import resolve_native_cameras
    from openwam.dataloader.native_stage2_profiles import fingerprint
    q,cmd=_source_keys(info)
    provenance='audited Morphi29 ROS positions rad; gripper source openness 0=closed,1=open; recorded same-tick action.position'
    rep=dict(state_key='__no_converted_eef__',joint_state_key=q,
        joint_indices={s:[NATIVE_NAMES.index(a+f'arm{i}_Joint') for i in range(1,8)] for s,a in zip(SIDES,('L','R'))},
        tcp_pose_keys=dict(TCP_KEYS),tcp_from_eef={},head_camera_available=not bool(info.get('mobile_manipulation_64_conversion',{}).get('head_eef_zero_padding')),head_world_to_camera_key=HEAD_KEY,
        gripper_state_sources={s:dict(key=q,index=NATIVE_NAMES.index(a+'gripper_Joint'),provenance=provenance) for s,a in zip(SIDES,('L','R'))},
        gripper_command_sources={s:dict(key=cmd,index=NATIVE_NAMES.index(a+'gripper_Joint'),provenance=provenance) for s,a in zip(SIDES,('L','R'))})
    result=dict(version='posttrain_morphi112_v1',family='posttrain_native',robot_type='Morphi',representation=rep,
        cameras=resolve_native_cameras(repo,info),fk_supervision=None,tracking_confidence_threshold=.8,
        source_columns=posttrain_columns(info),mask_reasons=['episode_geometry_pending','missing_independent_fk_gt'])
    result['profile_id']=fingerprint(result)
    return result


class _Kinematics:
    """Batched named URDF FK, matching the audited deployment geometry helper."""
    def __init__(self,path):
        content=Path(path).read_bytes();self.sha256=hashlib.sha256(content).hexdigest();xml=ET.fromstring(content)
        self.links={x.get('name') for x in xml.findall('link')};self.joints={}
        for item in xml.findall('joint'):
            child=item.find('child').get('link')
            if child in self.joints:raise ValueError('duplicate URDF child')
            origin=item.find('origin');axis=item.find('axis');fixed=np.eye(4)
            if origin is not None:
                fixed[:3,3]=np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
                fixed[:3,:3]=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix()
            vector=np.fromstring(axis.get('xyz','1 0 0') if axis is not None else '1 0 0',sep=' ')
            if item.get('type')!='fixed' and (item.find('mimic') is not None or np.linalg.norm(vector)==0):raise ValueError('unsupported URDF joint')
            self.joints[child]=dict(name=item.get('name'),kind=item.get('type'),parent=item.find('parent').get('link'),fixed=rigid_matrix(fixed),axis=vector/max(np.linalg.norm(vector),1e-12))

    def link(self,q,names,target):
        lookup={name:i for i,name in enumerate(names)}
        lookup.update({alias:lookup[name] for alias,name in ALIASES.items() if name in lookup})
        chain=[];seen=set()
        while target!='base_link':
            if target in seen or target not in self.joints:raise ValueError('unavailable URDF chain')
            seen.add(target);joint=self.joints[target];chain.append(joint);target=joint['parent']
        out=np.broadcast_to(np.eye(4),(len(q),4,4)).copy();valid=np.ones(len(q),bool)
        for j in reversed(chain):
            out=out@j['fixed']
            if j['kind']=='fixed':continue
            v=q[:,lookup[j['name']]] if j['name'] in lookup else np.full(len(q),np.nan)
            valid &= np.isfinite(v);clean=np.nan_to_num(v,nan=0,posinf=0,neginf=0)
            move=np.broadcast_to(np.eye(4),out.shape).copy()
            if j['kind'] in ('revolute','continuous'):move[:,:3,:3]=Rotation.from_rotvec(clean[:,None]*j['axis']).as_matrix()
            elif j['kind']=='prismatic':move[:,:3,3]=clean[:,None]*j['axis']
            else:raise ValueError('unsupported URDF joint type')
            out=out@move
        out[~valid]=np.nan
        return out


def _json(value):
    return json.loads(value) if isinstance(value,str) and value else value or {}


def _head_calibration(info,ep,fk,q,camera_key):
    """Only explicit embedded calibration for the actual released image."""
    if camera_key not in ('observation.images.head','observation.images.head_undistorted'):
        return None,'unsupported_projection_camera'
    intrinsic=_json(ep.get('calibration_intrinsic'));extrinsic=_json(ep.get('calibration_extrinsic'))
    if not intrinsic or not extrinsic:return None,'missing_episode_head_calibration'
    name=camera_key.split('.')[-1]
    matches=[v for v in intrinsic.get('channels',{}).values() if name in (v.get('dir_name'),v.get('source_dir_name'))]
    sources=list(extrinsic.get('sources',{}).values())
    if len(matches)!=1 or len(sources)!=1:return None,'ambiguous_episode_head_calibration'
    entry=matches[0];source=sources[0]
    if source.get('device_id',ep.get('source_group_id'))!=ep.get('source_group_id'):
        return None,'episode_calibration_device_mismatch'
    ex=source.get('extrinsic',{});tr=[t for t in ex.get('positions',{}).get('head',{}).get('transforms',[]) if t.get('target_frame')=='camera4']
    if len(tr)!=1 or tr[0].get('source_frame') not in fk.links:return None,'unverified_head_mount'
    tr=tr[0];mount=rigid_matrix(tr['T_target_source']);link=fk.link(q,NATIVE_NAMES,tr['source_frame'])
    # np.linalg.inv accepts NaNs; they remain unavailable rather than becoming identity.
    E=mount@np.linalg.inv(link)
    undistorted=camera_key.endswith('_undistorted');und=entry.get('undistortion',{})
    h,w=info['features'][camera_key]['shape'][:2]
    if undistorted:
        if und.get('view_rotation_applied',False):return None,'unverified_rectification_rotation'
        if 'square' in und and (h,w)!=(und['square'],und['square']):return None,'unverified_released_image_resize'
        if 'new_k' not in und:return None,'missing_undistorted_intrinsics'
        K=np.asarray(und['new_k'],float);raw=None
    else:
        raw=entry.get('calibration',{});K=np.asarray(raw.get('k'),float)
        if raw.get('distortion_model')!='omni-radtan6':return None,'unsupported_raw_camera_model'
        if 'omni_xi' not in raw or len(raw.get('d',[]))!=6:return None,'incomplete_raw_intrinsics'
    if K.shape!=(3,3) or not np.isfinite(K).all():return None,'invalid_camera_intrinsics'
    return dict(E_base=E,K=K,raw=raw,shape=[h,w],source='embedded_episode_calibration',mount_link=tr['source_frame']),None


def _geometry(info,ep,columns,calibration_root):
    qkey,_=_source_keys(info);q=np.asarray(columns[qkey],float);n=len(q)
    if q.shape!=(n,29):raise ValueError('native_joint_schema: rows must have width29')
    device=ep.get('source_group_id');matches=[]
    # The device name is a literal path component, never a glob from metadata.
    if isinstance(device,str) and device and Path(device).name==device:
        for version in ('V1.0','V1.2'):
            root=Path(calibration_root)/'robot'/f'MZ2501_{version}'
            if (root/'device'/device/'device_extrinsic.yaml').is_file() and (root/'robot_description.urdf').is_file():matches.append(root/'robot_description.urdf')
    g=dict(reasons=[],device=device,reference=None,cameras={})
    if len(matches)!=1:
        g['reasons'].append('missing_device_urdf' if not matches else 'ambiguous_device_urdf');return g
    fk=_Kinematics(matches[0]);g.update(urdf=str(matches[0]),urdf_sha256=fk.sha256,tcp_offset_m=[0,0,.23116])
    tool=np.eye(4);tool[2,3]=.23116
    tcp=np.stack([fk.link(q,NATIVE_NAMES,a+'arm08_link')@tool for a in ('L','R')],axis=1)
    g['tcp_base']=tcp
    for camera in ('observation.images.head_undistorted','observation.images.head'):
        if camera in info['features']:
            try:
                c,reason=_head_calibration(info,ep,fk,q,camera)
            except (ValueError,KeyError,TypeError) as exc:
                c,reason=None,'invalid_episode_head_calibration'
                g.setdefault('calibration_errors',{})[camera]=str(exc)
            if c is not None:g['cameras'][camera]=c
            elif reason not in g['reasons']:g['reasons'].append(reason)
    odom=columns.get('observation.odom_vehicle_absolute')
    if odom is None:
        g['reasons'].append('missing_odom_for_temporal_tcp');return g
    odom=np.asarray(odom,float)
    if odom.ndim!=2 or odom.shape[0]!=n or odom.shape[1]<7:raise ValueError('invalid native odom shape')
    valid=np.isfinite(odom[:,:7]).all(1)&(np.linalg.norm(odom[:,3:7],axis=1)>1e-6)
    base=np.broadcast_to(np.eye(4),(n,4,4)).copy()
    base[valid,:3,3]=odom[valid,:3];base[valid,:3,:3]=Rotation.from_quat(odom[valid,3:7]).as_matrix()
    # Anchor is a constant rebase for precision only; select first valid source
    # row if episode row zero is invalid, while retaining that row's NaN mask.
    if valid.any():base=np.linalg.inv(base[np.flatnonzero(valid)[0]])@base
    base[~valid]=np.nan
    if not valid.all():g['reasons'].append('invalid_recorded_odom_rows')
    g.update(tcp_reference=base[:,None]@tcp,reference='episode_start_base' if valid[0] else 'first_valid_odom_base',base=base)
    return g


def _pose7(matrix):
    result=np.full((len(matrix),7),np.nan);valid=np.isfinite(matrix).all((1,2))
    result[valid,:3]=matrix[valid,:3,3]
    result[valid,3:]=Rotation.from_matrix(matrix[valid,:3,:3]).as_quat()[:,[3,0,1,2]] if valid.any() else np.empty((0,4))
    return result


def representation_info(info,profile):
    info=copy.deepcopy(info)
    if profile['family']=='posttrain_native':
        for key in TCP_KEYS.values():info['features'][key]={'shape':[7],'unit':'metre_wxyz','provenance':'device_urdf_fk_plus_recorded_odom'}
        info['features'][HEAD_KEY]={'shape':[16]}
    return info


def augment_posttrain_episode(info,ep,columns,profile,*,calibration_root=CALIBRATION_ROOT):
    """Return copied info/columns/profile plus bounded per-episode geometry."""
    from openwam.dataloader.native_stage2_profiles import fingerprint
    info=representation_info(info,profile);columns=dict(columns);profile=copy.deepcopy(profile)
    g=_geometry(info,ep,columns,calibration_root);rep=profile['representation']
    n=len(columns[rep['joint_state_key']])
    for key in TCP_KEYS.values():columns[key]=np.full((n,7),np.nan)
    columns[HEAD_KEY]=np.full((n,16),np.nan)
    profile['mask_reasons']=[r for r in profile['mask_reasons'] if r!='episode_geometry_pending']+g['reasons']
    if 'tcp_reference' in g:
        rep['tcp_pose_keys']=dict(TCP_KEYS)
        for i,side in enumerate(SIDES):
            info['features'][TCP_KEYS[side]]={'shape':[7],'unit':'metre_wxyz','provenance':'device_urdf_fk_plus_recorded_odom'}
            columns[TCP_KEYS[side]]=_pose7(g['tcp_reference'][:,i])
        camera=profile['cameras'][0];cal=g['cameras'].get(camera)
        disabled=bool(info.get('mobile_manipulation_64_conversion',{}).get('head_eef_zero_padding'))
        if disabled:profile['mask_reasons'].append('source_head_eef_zero_padding')
        if cal is not None and not disabled:
            columns[HEAD_KEY]=(cal['E_base']@np.linalg.inv(g['base'])).reshape(-1,16)
            info['features'][HEAD_KEY]={'shape':[16]};rep['head_camera_available']=True
    profile['geometry_provenance']={k:v for k,v in g.items() if k not in ('tcp_base','tcp_reference','base','cameras')}
    profile['geometry_provenance']['episode_calibration_sha256']=hashlib.sha256(json.dumps([_json(ep.get('calibration_intrinsic')),_json(ep.get('calibration_extrinsic'))],sort_keys=True).encode()).hexdigest()
    return info,columns,profile,g


def project_posttrain_goal(info,ep,columns,goal_index,camera_key,*,calibration_root=CALIBRATION_ROOT):
    """Project same-frame base FK into the exact recorded camera (pixel xy).

    No odometry is required for a same-frame camera projection. Returned points
    outside the source image, invalid joints, or unverified calibration are masked.
    """
    g=_geometry(info,ep,columns,calibration_root);xy=np.zeros((2,2),np.float32);valid=np.zeros(2,bool)
    provenance={k:v for k,v in g.items() if k not in ('tcp_base','tcp_reference','base','cameras')}
    provenance['projection_reference']='same_frame_base'
    cal=g['cameras'].get(camera_key)
    if cal is None or 'tcp_base' not in g:return xy,valid,provenance
    xyz=g['tcp_base'][goal_index,:,:3,3];E=cal['E_base'][goal_index];xyz=xyz@E[:3,:3].T+E[:3,3]
    valid=np.isfinite(xyz).all(1)&(xyz[:,2]>1e-6)
    K=cal['K'];raw=cal['raw']
    if raw:
        den=xyz[:,2]+float(raw['omni_xi'])*np.linalg.norm(xyz,axis=1);valid &= den>1e-8
        x,y=(xyz[:,:2]/np.where(valid,den,1)[:,None]).T
        k1,k2,k3,k4,p1,p2=raw['d'];r=x*x+y*y;radial=1+r*(k1+r*(k2+r*(k3+r*k4)))
        uv=np.c_[x*radial+2*p1*x*y+p2*(r+2*x*x),y*radial+p1*(r+2*y*y)+2*p2*x*y,np.ones(2)]@K.T
    else:uv=xyz@K.T
    points=uv[:,:2]/np.where(valid,uv[:,2],1)[:,None]
    h,w=cal['shape'];valid &= np.isfinite(points).all(1)&(points[:,0]>=0)&(points[:,0]<w)&(points[:,1]>=0)&(points[:,1]<h)
    for key in ('canonical.frame_valid','training.valid'):
        if key in columns:
            flag=np.asarray(columns[key][goal_index]);valid &= bool((np.isfinite(flag)&(flag>.5)).all())
    xy[valid]=points[valid]
    return xy,valid,provenance


def physical_dependency_hash(root,*,calibration_root=CALIBRATION_ROOT):
    """Hash projected episode geometry and device/URDF dependencies, bounded RAM.

    Cache by file stat identity; changed inputs are rehashed before stats lookup.
    No image data or unrelated parquet statistics are materialized.
    """
    root=Path(root);calibration_root=Path(calibration_root)
    metadata=sorted((root/'meta/episodes').glob('chunk-*/*.parquet'))
    urdfs=sorted((calibration_root/'robot').glob('MZ2501_V*/robot_description.urdf'))
    devices=sorted(str(p.relative_to(calibration_root)) for p in (calibration_root/'robot').glob('MZ2501_V*/device/*/device_extrinsic.yaml'))
    identity=tuple((str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in metadata+urdfs)
    return _physical_dependency_hash(identity,len(metadata),tuple(devices))


from functools import lru_cache
@lru_cache(maxsize=512)
def _physical_dependency_hash(identity,metadata_count,devices):
    import pyarrow.parquet as pq
    digest=hashlib.sha256(json.dumps(devices).encode())
    wanted=('episode_index','source_group_id','calibration_intrinsic','calibration_extrinsic')
    for index,(path,size,mtime) in enumerate(identity):
        digest.update(str(Path(path).name).encode())
        if index>=metadata_count:
            digest.update(Path(path).read_bytes());continue
        parquet=pq.ParquetFile(path);columns=[k for k in wanted if k in parquet.schema_arrow.names]
        for batch in parquet.iter_batches(columns=columns,batch_size=128):
            for row in batch.to_pylist():digest.update(json.dumps(row,sort_keys=True,separators=(',',':')).encode())
    return digest.hexdigest()
