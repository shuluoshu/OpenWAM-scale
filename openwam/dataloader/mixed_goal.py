"""Validated task/subtask pairs and deterministic family/level-balanced sampling."""
import json
import random
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image

FORMAT = 'openwam.goal.mixed.v1'


def episode_key(row):
    return str(Path(row['source_repo']).resolve()), row['source_episode']


def validate_mixed_rows(rows):
    if not rows: raise ValueError('empty mixed manifest')
    ids=set(); tasks=set(); splits={}; occurrences=set()
    for r in rows:
        if r['id'] in ids: raise ValueError('duplicate sample id')
        ids.add(r['id']); key=episode_key(r)
        if r.get('split')!='diagnostic': raise ValueError('mixed overfit requires diagnostic split')
        if not isinstance(r['instruction'],str) or not r['instruction'].strip(): raise ValueError('empty instruction')
        if r['success_verified'] is not False: raise ValueError('weak endpoints cannot assert success')
        for k in ('source_episode','source_episode_length','anchor_source_frame','goal_source_frame','instruction_start','instruction_end_exclusive'):
            if type(r[k]) is not int or r[k]<0: raise ValueError('invalid integer frame coordinates')
        a,b=r['instruction_start'],r['instruction_end_exclusive']; goal=r['goal_source_frame']
        if not 0<=a<=r['anchor_source_frame']<goal==b-1<r['source_episode_length']:
            raise ValueError('goal must equal instruction end-1 after observation')
        if r['granularity']=='task':
            if a!=0 or b!=r['source_episode_length'] or r['label_kind']!='episode_end': raise ValueError('task goal must be full episode end')
            if key in tasks: raise ValueError('duplicate task episode')
            tasks.add(key)
        elif r['granularity']=='subtask':
            if r['label_kind']!='subtask_end' or r['anchor_source_frame']!=a: raise ValueError('subtask goal bounds')
        else: raise ValueError('unknown granularity')
        occurrence=(key,r['granularity'],a,b)
        if occurrence in occurrences: raise ValueError('duplicate interval')
        occurrences.add(occurrence)
        p=r.get('provenance',{})
        for k in ('goal_source_frame','source_goal_frame'):
            if k in p and p[k]!=goal: raise ValueError('geometry goal frame mismatch')
    for r in rows:
        if r['granularity']=='subtask' and episode_key(r) not in tasks:
            raise ValueError('subtask requires same-source episode task pair')
    return rows


def load_mixed_manifest(path,*,diagnostic_override=False,allow_unreviewed_geometry=False):
    if not diagnostic_override or not allow_unreviewed_geometry:
        raise ValueError('mixed diagnostics require explicit diagnostic and geometry overrides')
    path=Path(path); rows=json.loads(path.read_text())
    validate_mixed_rows(rows)
    keys=set()
    for r in rows:
        if r.get('format')!=FORMAT: raise ValueError('wrong mixed format')
        if not r.get('metric_key') or r['metric_key'] in keys: raise ValueError('unique metric keys required')
        keys.add(r['metric_key'])
        for k in ('observed_image_path','goal_image_path','target_path'):
            p=Path(r[k]);p=p if p.is_absolute() else path.parent/p
            if not p.is_file(): raise ValueError(f'missing {k}: {p}')
            r[k]=str(p.resolve())
        for k in ('observed_image_path','goal_image_path'):
            with Image.open(r[k]) as im:
                if im.mode!='RGB' or min(im.size)<2: raise ValueError('RGB image required')
                im.load()
        with np.load(r['target_path'],allow_pickle=False) as t:
            for k,shape in [('uv',(2,2)),('projection_valid',(2,)),('in_frame',(2,)),('visibility',(2,))]:
                if k not in t or t[k].shape!=shape: raise ValueError('invalid target shape')
            for k in ('projection_valid','in_frame'):
                if not np.isin(t[k],[0,1]).all(): raise ValueError('invalid mask')
            valid=t['projection_valid'].astype(bool); active=valid&t['in_frame'].astype(bool)
            if not np.isfinite(t['uv'][valid]).all(): raise ValueError('nonfinite valid UV')
            if np.any((t['uv'][active]<0)|(t['uv'][active]>1)): raise ValueError('UV outside normalized bounds')
            if not np.isin(t['visibility'],[-1,0,1]).all(): raise ValueError('invalid visibility')
    return rows


def balanced_schedule(records,steps,seed):
    groups=defaultdict(lambda:defaultdict(list))
    for i,r in enumerate(records): groups[r['family']][r['granularity']].append(i)
    if not groups or any('task' not in levels for levels in groups.values()): raise ValueError('every family needs task samples')
    rng=random.Random(seed); families=sorted(groups); visits=defaultdict(int); cursors=defaultdict(int); result=[]
    while len(result)<steps:
        rng.shuffle(families)
        for family in families:
            levels=groups[family]; level='subtask' if 'subtask' in levels and visits[family]%2 else 'task'
            visits[family]+=1; key=(family,level); pool=levels[level]
            if cursors[key]%len(pool)==0:rng.shuffle(pool)
            result.append(pool[cursors[key]%len(pool)]);cursors[key]+=1
            if len(result)==steps:break
    return result
