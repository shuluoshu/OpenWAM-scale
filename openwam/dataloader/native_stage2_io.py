"""Bounded, projected native episode reads and strict source-time validity."""
from collections import OrderedDict
from pathlib import Path
import json
import numpy as np
from openwam.dataloader.curated_goal import language_intervals
from openwam.dataloader.native_stage2_profiles import resolve_native_profile, fingerprint
from openwam.dataloader.utils.mora_relative import MoraHeadTCPRepresentation
from openwam.dataloader.utils.gripper import ALIGNED_COMMAND, ALIGNED_VALID, gripper_state_outliers

VALIDITY_COLUMNS = ('canonical.frame_valid','training.valid','observation.states.eef.mask',
    'source.camera.world_to_camera_valid','observation.states.camera.valid',
    'observation.states.hand.valid','observation.states.hand.confidence',
    'observation.states.eef.confidence','actions.eef.mask','actions.joint.mask')


class ByteLRU(OrderedDict):
    def __init__(self, max_entries, max_bytes):
        super().__init__();self.max_entries=int(max_entries);self.max_bytes=int(max_bytes);self.bytes=0
        if min(self.max_entries,self.max_bytes)<1:raise ValueError('cache limits must be positive')
    def put(self,key,value,size):
        if size>self.max_bytes:raise ValueError(f'cache_byte_budget: episode {key} needs {size}, limit {self.max_bytes}')
        if key in self:self.bytes-=self.pop(key)[1]
        while self and (len(self)>=self.max_entries or self.bytes+size>self.max_bytes):
            _,(_,old_size)=self.popitem(last=False);self.bytes-=old_size
        self[key]=(value,size);self.bytes+=size
    def get_value(self,key):
        self.move_to_end(key);return self[key][0]


def read_episode_metadata(root,episode):
    import pyarrow.dataset as ds
    files=sorted((root/'meta/episodes').glob('chunk-*/*.parquet'))
    if not files:raise ValueError(f'{root}:{episode}: missing episode metadata')
    dataset=ds.dataset([str(p) for p in files],format='parquet')
    columns=[n for n in dataset.schema.names if not n.startswith('stats/')]
    table=dataset.to_table(columns=columns,filter=ds.field('episode_index')==episode)
    if len(table)!=1:raise ValueError(f'{root}:{episode}: nonunique episode metadata')
    return table.to_pylist()[0]


def load_episode(repo,episode,filter_config,max_bytes):
    import pyarrow.parquet as pq
    from scripts.data.prepare_goal_labels import task_text
    from openwam.dataloader.utils.fk_supervision import FKSupervisionBuilder
    root=Path(repo['path']);raw_info=(root/'meta/info.json').read_bytes();info=json.loads(raw_info)
    profile=resolve_native_profile(repo['repo'],info);rep=MoraHeadTCPRepresentation(info,**profile['representation'])
    rep.contract['gripper_state_filter']=filter_config
    rep.contract['native_profile_id']=profile['profile_id']
    fk=FKSupervisionBuilder(profile['fk_supervision'],rep,info) if profile['fk_supervision'] else None
    ep=read_episode_metadata(root,episode);length=int(ep['length'])
    head_eligible,head_qa=calibration_eligibility(root,repo['family'],episode)
    path=root/info['data_path'].format(episode_index=episode,chunk_index=int(ep['data/chunk_index']),file_index=int(ep['data/file_index']))
    available=pq.ParquetFile(path).schema_arrow.names
    ann=root/'meta/sub_tasks.parquet'
    keys=list(dict.fromkeys(['frame_index','task_index',*rep.columns,
        *(fk.columns if fk else []),*[k for k in VALIDITY_COLUMNS if k in info['features']]]))
    if ann.exists():
        if 'subtask_index' not in available:raise ValueError(f'{root}:{episode}: annotations lack subtask_index')
        keys+=['subtask_index']
        if 'subtask_unannotated' in available:keys+=['subtask_unannotated']
    # Reject oversized episodes before allocating their projected arrays. No full parquet/pandas conversion.
    estimate=length*sum(max(1,int(np.prod(info['features'].get(k,{}).get('shape',[1]))))*8 for k in keys)
    if estimate>max_bytes:raise ValueError(f'{root}:{episode}: cache_byte_budget needs >= {estimate}')
    table=pq.read_table(path,columns=keys,filters=[('episode_index','=',episode)]).sort_by('frame_index')
    if table.nbytes>max_bytes:raise ValueError(f'{root}:{episode}: cache_byte_budget')
    columns={k:np.asarray(table[k].to_pylist()) for k in keys}
    if not np.array_equal(columns['frame_index'],np.arange(length)):
        raise ValueError(f'{root}:{episode}: source frame indices must be dense unique complete')
    labels={k:columns[k].tolist() for k in ('frame_index','task_index','subtask_index','subtask_unannotated') if k in columns}
    annotations=None
    if ann.exists():
        ns=pq.ParquetFile(ann).schema_arrow.names
        if not {'episode_index','subtask_index','subtask'}.issubset(ns):raise ValueError('malformed present subtask annotations')
        annotations=pq.read_table(ann,columns=[k for k in ['episode_index','subtask_index','subtask','start_frame','end_frame'] if k in ns],filters=[('episode_index','=',episode)]).to_pylist()
    tasks,subs=language_intervals(labels,{int(i):task_text(root,int(i)) for i in set(labels['task_index'])},annotations,length)
    for e,a,b in repo['ranges']:
        if e==episode and not 0<=a<b<=length:raise ValueError(f'{root}:{episode}: invalid_segment_bounds')
    commands,valid=rep.gripper_commands.align(columns,length)
    bad,reasons=gripper_state_outliers(rep.measured_grippers(columns,length),commands,rep.gripper_commands.controller_valid(valid),filter_config)
    columns[ALIGNED_COMMAND]=commands;columns[ALIGNED_VALID]=valid
    result=dict(root=root,info=info,info_hash=__import__('hashlib').sha256(raw_info).hexdigest(),profile=profile,
        representation=rep,fk=fk,ep=ep,length=length,data=path,columns=columns,tasks=tasks,subtasks=subs,
        head_calibration_eligible=head_eligible,head_calibration_qa=head_qa,calibration_sha256=calibration_file_hash(root),
        bad=bad,bad_prefix=np.r_[0,np.cumsum(bad)],filter_reasons=reasons,annotation_file=str(ann) if ann.exists() else None)
    size=sum(v.nbytes for v in columns.values())+bad.nbytes+result['bad_prefix'].nbytes+len(raw_info)
    return result,size


def window_options(metadata,ranges,level):
    """Compact eligible half-open anchor intervals; never bridge a selected gap."""
    options=[]
    for instruction in metadata['subtasks' if level=='subtask' else 'tasks']:
        for _,a,b in ranges:
            start=max(a,instruction['start']);last=min(b,instruction['end_exclusive'],metadata['length'])-1
            if start<last:options.append(dict(start=start,end_exclusive=last,last_source=last,selected_start=a,
                selected_end_exclusive=b,instruction=instruction))
    return options


def _valid_flags(values, threshold=.5, *, inclusive=False):
    """All stored validity/confidence flags reject NaN and either infinity."""
    values=np.asarray(values)
    return np.isfinite(values)&(values>=threshold if inclusive else values>threshold)


def _eef_mask_flags(rep,columns,key,n):
    """EEF10 declares the left block; absent blocks must not erase other sources."""
    width=rep.features.get(rep.state_key,{}).get('shape',[20])[0]
    source=np.asarray(columns[key])
    if width not in (10,20) or source.shape!=(n,width):
        raise ValueError(f'{key}: declared EEF{width} requires stored mask width {width} and {n} rows; got {source.shape}')
    flags=np.ones((n,20),bool)
    flags[:,:width]=_valid_flags(source)
    return flags


def source_frame_validity(columns,n):
    """Global source flags share finite, >0.5 semantics in selection and labels."""
    valid=np.ones(n,bool)
    for key in ('canonical.frame_valid','training.valid'):
        if key in columns:
            flags=np.asarray(columns[key]).reshape(n,-1)
            valid &= _valid_flags(flags).all(-1)
    return valid


def merge_source_masks(rep,columns,action_mask,state_mask):
    """Merge stored validity without allowing one missing branch to erase another."""
    n=len(action_mask)+1
    global_valid=source_frame_validity(columns,n)
    state_mask &= global_valid[:1,None]
    action_mask &= (global_valid[0]&global_valid[1:])[:,None]
    eef=np.ones((n,20),bool)
    if 'observation.states.eef.mask' in columns:eef &= _eef_mask_flags(rep,columns,'observation.states.eef.mask',n)
    for key,threshold in [('observation.states.hand.valid',.5),('observation.states.hand.confidence',.8)]:
        if key in columns:
            source=np.asarray(columns[key]).reshape(n,2)
            good=_valid_flags(source,threshold,inclusive='confidence' in key)
            for i in range(2):eef[:,i*10:(i+1)*10] &= good[:,i,None]
    if 'observation.states.eef.confidence' in columns:
        source=np.asarray(columns['observation.states.eef.confidence']).reshape(n,4)
        for i in range(2):eef[:,i*10+9] &= _valid_flags(source[:,i+2],.8,inclusive=True)
    for side in range(2):
        start=10*side;pose=eef[:,start:start+9].all(-1)
        state_mask[:,start:start+10] &= eef[:1,start:start+10]
        state_mask[:,34+9*side:43+9*side] &= pose[:1,None]
        action_mask[:,start:start+9] &= (pose[0]&pose[1:])[:,None]
        action_mask[:,34+9*side:43+9*side] &= (pose[0]&pose[1:])[:,None]
    if 'actions.eef.mask' in columns:
        flags=_eef_mask_flags(rep,columns,'actions.eef.mask',n)[:-1]
        for side in range(2):
            start=10*side
            # Both coordinate labels describe the same TCP. A pose is coupled,
            # including its rotation6d components; grip and joints are independent.
            pose=flags[:,start:start+9].all(-1)
            action_mask[:,start:start+9] &= pose[:,None]
            action_mask[:,34+9*side:43+9*side] &= pose[:,None]
            action_mask[:,start+9] &= flags[:,start+9]
    if 'actions.joint.mask' in columns:
        flags=_valid_flags(columns['actions.joint.mask'])[:-1]
        for side,name in enumerate(('left','right')):
            idx=rep.joint_indices[name]
            action_mask[:,20+side*7:20+side*7+len(idx)] &= flags[:,idx]
    for key in ('source.camera.world_to_camera_valid','observation.states.camera.valid'):
        if key in columns and not _valid_flags(columns[key][0]).all():
            action_mask[:,34:]=False;state_mask[:,34:]=False
    return global_valid


def video_frame_base(episode,rows,camera,fps):
    """Cross-check file-local ordinal against merged episode metadata timestamps."""
    selected=[r for r in rows if int(r['episode_index'])==int(episode['episode_index'])]
    if len(selected)!=1:raise ValueError('invalid_video_offset: selected episode missing/duplicated')
    rows=sorted(rows,key=lambda r:int(r['dataset_from_index']))
    base=0
    for row in rows:
        if int(row['episode_index'])==int(episode['episode_index']):break
        base+=int(row['length'])
    timestamp=float(episode['videos/'+camera+'/from_timestamp'])*fps
    if not np.isfinite(timestamp) or abs(timestamp-base)>1e-3:
        raise ValueError(f'invalid_video_offset: {camera} cumsum={base}, timestamp_frames={timestamp}')
    return base


def validate_video_metadata(m,camera):
    import pyarrow.dataset as ds
    prefix='videos/'+camera+'/'
    files=sorted((m['root']/'meta/episodes').glob('chunk-*/*.parquet'))
    dataset=ds.dataset([str(p) for p in files],format='parquet')
    ep=m['ep'];expression=(ds.field(prefix+'chunk_index')==int(ep[prefix+'chunk_index'])) & (ds.field(prefix+'file_index')==int(ep[prefix+'file_index']))
    table=dataset.to_table(columns=['episode_index','length','dataset_from_index'],filter=expression)
    return video_frame_base(ep,table.to_pylist(),camera,float(m['info']['fps']))


def calibration_eligibility(root,family,episode):
    """ABC source camera QA is episode-specific; no source file is fabricated."""
    if family!='abc_130k':return True,None
    path=Path(root)/'meta/camera_geometry_eligibility.jsonl';matches=[]
    if path.exists():
        with path.open() as stream:
            for line in stream:
                if line.strip():
                    row=json.loads(line)
                    if row.get('episode_index')==episode:matches.append(row)
    if len(matches)>1:raise ValueError(f'{root}:{episode}: conflicting camera eligibility')
    record=matches[0] if matches else None
    return bool(record and record.get('broadcast_qa_status')=='passed'),record


def calibration_file_hash(root):
    path=Path(root)/'meta/camera_geometry_eligibility.jsonl'
    if not path.exists():return None
    digest=__import__('hashlib').sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()
