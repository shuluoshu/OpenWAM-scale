"""Subtask-only goal pairs; future image/TCP labels stay outside model conditions."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from openwam.train.goal_overfit import prepare_overfit_images

FORMAT='openwam.goal.subtask.v1'


def contiguous_subtasks(frames,annotations,episode_length):
    """Split on frame-label changes AND explicit annotation occurrence boundaries.

    A subtask ID identifies text, not a unique occurrence. Adjacent occurrences
    can share that ID but must retain their own terminal goal frames.
    """
    if frames['frame_index']!=list(range(episode_length)):
        raise ValueError('source frame indices must be dense, unique and complete')
    if any(len(frames[k])!=episode_length for k in ('task_index','subtask_index')):
        raise ValueError('source frame label length mismatch')
    mapping={};by_index={}
    for annotation in annotations:
        idx=int(annotation['subtask_index']);text=annotation.get('subtask')
        if not isinstance(text,str) or not text.strip():continue
        if idx in mapping and mapping[idx]!=text:raise ValueError('conflicting subtask text')
        mapping[idx]=text;by_index.setdefault(idx,[]).append(annotation)
    boundaries={}
    for idx,rows in by_index.items():
        intervals=sorted((int(a['start_frame']),int(a['end_frame'])) for a in rows
                         if 'start_frame' in a and 'end_frame' in a)
        if any(a[1]>b[0] for a,b in zip(intervals,intervals[1:])):
            raise ValueError(f'overlapping subtask annotation intervals: {idx}')
        boundaries[idx]={edge for interval in intervals for edge in interval}
    unannotated=frames.get('subtask_unannotated',[False]*episode_length)
    if len(unannotated)!=episode_length:raise ValueError('unannotated mask length mismatch')
    keys=[(int(t),int(s) if s is not None else -1,bool(u)) for t,s,u in zip(frames['task_index'],frames['subtask_index'],unannotated)]
    segments=[];start=0
    for end in range(1,episode_length+1):
        if (end!=episode_length and keys[end]==keys[start]
                and end not in boundaries.get(keys[start][1],())):continue
        task,idx,unmarked=keys[start];text=mapping.get(idx);reasons=[]
        if idx<0:reasons.append('negative_or_missing_subtask_index')
        if unmarked:reasons.append('source_subtask_unannotated')
        if not text:reasons.append('missing_subtask_text')
        intervals=[a for a in by_index.get(idx,[]) if 'start_frame' in a and 'end_frame' in a]
        if intervals and not any(int(a['start_frame'])<=start and int(a['end_frame'])>=end for a in intervals):
            reasons.append('annotation_interval_does_not_cover_frame_run')
        segments.append(dict(number=len(segments),start=start,end_exclusive=end,end_frame=end-1,
            task_index=task,subtask_index=idx,text=text,valid=not reasons,exclusion_reasons=reasons))
        start=end
    return segments


def load_subtask_manifest(path):
    path=Path(path);rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids=set();episodes={};occurrences=set()
    if not rows:raise ValueError('empty subtask manifest')
    for r in rows:
        if r.get('format')!=FORMAT:raise ValueError('wrong subtask manifest format')
        if r.get('label_kind')!='subtask_end' or r.get('success_verified') is not False:
            raise ValueError('subtask_end is a weak candidate, never verified success')
        keys=('source_episode','source_episode_length','segment_number','subtask_index','instruction_start',
              'instruction_end_exclusive','anchor_source_frame','goal_source_frame')
        if any(type(r.get(k)) is not int or r[k]<0 for k in keys):raise ValueError('invalid integer subtask coordinates')
        a,b=r['instruction_start'],r['instruction_end_exclusive']
        if not (0<=a==r['anchor_source_frame']<r['goal_source_frame']==b-1<r['source_episode_length']):
            raise ValueError('subtask goal must be interval end-1, after the anchor')
        if not isinstance(r.get('instruction'),str) or not r['instruction'].strip():raise ValueError('missing subtask instruction')
        if r['id'] in ids:raise ValueError('duplicate subtask id')
        ids.add(r['id'])
        identity=(str(Path(r['source_repo']).resolve()),r['source_episode'])
        if r['split'] not in ('diagnostic','train','val','test'):raise ValueError('invalid split')
        if identity in episodes and episodes[identity]!=r['split']:raise ValueError('source episode split leakage')
        episodes[identity]=r['split']
        occurrence=(*identity,r['segment_number'])
        if occurrence in occurrences:raise ValueError('duplicate contiguous subtask occurrence')
        occurrences.add(occurrence)
        if r['provenance'].get('goal_source_frame')!=r['goal_source_frame']:
            raise ValueError('target geometry frame differs from subtask goal')
        for k in ('observed_image_path','goal_image_path','target_path'):
            p=Path(r[k]);p=p if p.is_absolute() else path.parent/p
            if not p.is_file():raise ValueError(f'missing {k}: {p}')
            r[k]=str(p.resolve())
        with np.load(r['target_path'],allow_pickle=False) as t:
            for k,shape in [('uv',(2,2)),('projection_valid',(2,)),('in_frame',(2,)),('visibility',(2,))]:
                if k not in t or t[k].shape!=shape:raise ValueError('invalid TCP target shape')
            if not np.isin(t['projection_valid'],[False,True]).all() or not np.isin(t['in_frame'],[False,True]).all():
                raise ValueError('invalid TCP masks')
            active=t['projection_valid'].astype(bool)&t['in_frame'].astype(bool)
            if not np.isfinite(t['uv'][active]).all() or np.any((t['uv'][active]<0)|(t['uv'][active]>1)):
                raise ValueError('active TCP coordinates must be finite normalized UV')
            if not np.isin(t['visibility'],[-1,0,1]).all():raise ValueError('invalid visibility')
    return rows


class SubtaskGoalDataset(Dataset):
    """Returns singleton batches matching PatchGoalExpert and text-cache helpers.

    Text encoding is deliberately separate: only ``instruction`` is encoded.
    Targets retain UV in float32 even when images use BF16. No task fallback.
    """
    def __init__(self,manifest,*,dtype=torch.bfloat16,device='cpu',allow_unreviewed_geometry=False):
        self.records=load_subtask_manifest(manifest)
        self.dtype=getattr(torch,dtype) if isinstance(dtype,str) else dtype
        self.device=device;self.allow_unreviewed_geometry=allow_unreviewed_geometry

    def __len__(self):return len(self.records)

    def __getitem__(self,index):
        row=self.records[index]
        batch=prepare_overfit_images([row],device=self.device,dtype=self.dtype)[0]
        if not self.allow_unreviewed_geometry and row['provenance'].get('geometry_review_status')!='approved':
            batch['targets'].point_valid[:]=False
        batch['instruction']=row['instruction'];batch['id']=row['id']
        return batch
