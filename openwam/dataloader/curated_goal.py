"""Lazy source-native curated task/subtask weak goal pairs.

Curated ranges constrain observations only. Goals use the original contiguous
instruction interval end, which may lie outside the curated range. No source
geometry or success state is inferred when source metadata is unavailable.
"""
from __future__ import annotations
from collections import OrderedDict
import hashlib
import json
import math
from itertools import islice
from pathlib import Path
import numpy as np
import torch
import yaml

HEAD_CAMERAS = {f: 'observation.images.' + c for f,c in {
    'Genrobot':'cam_high','LightWheel':'head_left','abc_130k':'cam_high',
    'egodex':'egocentric','egoverse':'front_1','interndata_a1':'cam_high',
    'mecka':'egocentric','morphirobot0930':'cam_high','robochallenge_v1':'cam_high',
    'robochallenge_v2':'cam_high','robomind_v1':'cam_front','robomind_v2':'camera_front'}.items()}


def parse_curated_yaml(path, data_root=None):
    """Parse only selection metadata; never enumerate source episode contents."""
    path=Path(path).resolve()
    with path.open() as stream:
        document=yaml.load(stream,Loader=getattr(yaml,'CSafeLoader',yaml.SafeLoader))
    selections=[]; roots=[]
    def walk(value):
        if isinstance(value,dict):
            if 'segments_by_repo' in value:selections.append(value)
            if 'data_root_dir' in value:roots.append(value['data_root_dir'])
            for child in value.values():walk(child)
        elif isinstance(value,list):
            for child in value:walk(child)
    walk(document)
    if len(selections)!=1:raise ValueError('expected exactly one segments_by_repo selection')
    if data_root is None:
        if len(set(roots))!=1:raise ValueError('expected one source data_root_dir or explicit data_root')
        data_root=roots[0]
    root=Path(data_root).resolve();selection=selections[0];repos=[]
    weights=selection.get('repo_weights',{})
    for name,ranges in selection['segments_by_repo'].items():
        weight=float(weights.get(name,0))
        if not math.isfinite(weight) or weight<0:raise ValueError(f'invalid repo weight: {name}')
        if not weight:continue
        family=name.split('/')[0]
        if family not in HEAD_CAMERAS:raise ValueError(f'unsupported source family {family}')
        clean=[]
        for item in ranges:
            if len(item)!=3 or any(type(v) is not int for v in item):raise ValueError('ranges must be integer [episode,start,end]')
            episode,start,end=item
            if episode<0 or not 0<=start<end:raise ValueError('invalid source range')
            clean.append((episode,start,end))
        if not clean:continue
        # Overlap would duplicate observation probability; reject ambiguous inputs.
        ordered=sorted(clean)
        if any(a[0]==b[0] and a[2]>b[1] for a,b in zip(ordered,ordered[1:])):
            raise ValueError(f'overlapping curated ranges: {name}')
        repos.append(dict(repo=name,path=str(root/name),family=family,weight=weight,ranges=clean))
    if not repos:raise ValueError('no positive weighted selected repositories')
    return dict(source_yaml=str(path),data_root=str(root),repos=repos)


def language_intervals(labels, task_mapping, annotations, length):
    """Shared language-only parser. Repeated IDs remain separate contiguous runs."""
    from openwam.dataloader.subtask_goal import contiguous_subtasks
    if labels['frame_index'] != list(range(length)):
        raise ValueError('source frame indices must be dense unique complete')
    tasks=[]; start=0
    for end in range(1,length+1):
        if end==length or labels['task_index'][end]!=labels['task_index'][start]:
            index=int(labels['task_index'][start]); text=task_mapping[index]
            if not isinstance(text,str) or not text.strip():raise ValueError('invalid task text')
            tasks.append(dict(start=start,end_exclusive=end,text=text,task_index=index));start=end
    if annotations is None:return tasks,[]
    for row in annotations:
        if type(row['subtask_index']) is not int or not isinstance(row['subtask'],str) or not row['subtask'].strip():
            raise ValueError('malformed present subtask annotation row')
        if ('start_frame' in row)!=('end_frame' in row):raise ValueError('incomplete annotation interval')
        if 'start_frame' in row and not (type(row['start_frame']) is int and type(row['end_frame']) is int and 0<=row['start_frame']<row['end_frame']<=length):
            raise ValueError('invalid present annotation interval')
    runs=contiguous_subtasks(labels,annotations,length)
    for run in runs:
        reasons=set(run['exclusion_reasons'])
        absent=reasons & {'negative_or_missing_subtask_index','source_subtask_unannotated'}
        conflicts=reasons & {'missing_subtask_text','annotation_interval_does_not_cover_frame_run'}
        if annotations and conflicts and not absent:
            raise ValueError(f'inconsistent subtask annotation: {run["start"]}:{run["end_exclusive"]}: {sorted(conflicts)}')
    return tasks,[s for s in runs if s['valid']]


class CuratedGoalDataset:
    def __init__(self,source_yaml,cache_dir,seed=42,subtask_probability=.5,
                 max_cached_episodes=8,dtype=torch.bfloat16,data_root=None):
        if not 0<=subtask_probability<=1:raise ValueError('invalid subtask probability')
        if max_cached_episodes<1:raise ValueError('max_cached_episodes must be positive')
        self.selection=parse_curated_yaml(source_yaml,data_root);self.repos=self.selection['repos']
        self.cache_dir=Path(cache_dir);self.cache_dir.mkdir(parents=True,exist_ok=True)
        self.seed=int(seed);self.subtask_probability=subtask_probability;self.max_cached_episodes=max_cached_episodes
        self.dtype=getattr(torch,dtype) if isinstance(dtype,str) else dtype
        self._episodes=OrderedDict();self._evaluations=None;self._asset_slot=0;self._evaluating=False
        self._weights=np.array([r['weight'] for r in self.repos]);self._weights/=self._weights.sum()

    def scope_summary(self):
        return dict(source_yaml=self.selection['source_yaml'],repositories=len(self.repos),
                    segments=sum(len(r['ranges']) for r in self.repos),
                    episodes=sum(len({s[0] for s in r['ranges']}) for r in self.repos),
                    selected_frames=sum(e-s for r in self.repos for _,s,e in r['ranges']),
                    families=sorted({r['family'] for r in self.repos}),
                    selected_weight_sum=sum(r['weight'] for r in self.repos),
                    max_cached_episodes=self.max_cached_episodes,
                    goal_policy='original contiguous instruction end_exclusive-1; may extend curated observation range')

    def _episode(self,repo,episode):
        import pyarrow.parquet as pq
        from scripts.data.prepare_goal_labels import read_episode,task_text
        from openwam.dataloader.subtask_goal import contiguous_subtasks
        key=(repo['path'],episode)
        if key in self._episodes:
            self._episodes.move_to_end(key);return self._episodes[key]
        root=Path(repo['path']);info=json.loads((root/'meta/info.json').read_text())
        head=HEAD_CAMERAS[repo['family']]
        if info['features'].get(head,{}).get('dtype')!='video':raise ValueError(f'unverified head camera schema: {head}')
        ep=read_episode(root,episode);length=int(ep['length'])
        data=root/info['data_path'].format(episode_index=episode,chunk_index=int(ep['data/chunk_index']),file_index=int(ep['data/file_index']))
        names=pq.ParquetFile(data).schema_arrow.names
        ann=root/'meta/sub_tasks.parquet';has_annotations=ann.exists()
        columns=['frame_index','task_index']
        if has_annotations:
            if 'subtask_index' not in names:raise ValueError('present annotations lack frame subtask_index')
            columns+=['subtask_index']
            if 'subtask_unannotated' in names:columns+=['subtask_unannotated']
        labels=pq.read_table(data,columns=columns,filters=[('episode_index','=',episode)]).sort_by('frame_index').to_pydict()
        if labels['frame_index']!=list(range(length)):raise ValueError('source frame indices must be dense unique complete')
        annotations=None
        if has_annotations:
            annotation_names=pq.ParquetFile(ann).schema_arrow.names
            if not {'episode_index','subtask_index','subtask'}.issubset(annotation_names):
                raise ValueError('malformed present subtask annotations')
            annotations=pq.read_table(ann,filters=[('episode_index','=',episode)]).to_pylist()
        mapping={int(i):task_text(root,int(i)) for i in set(labels['task_index'])}
        tasks,subtasks=language_intervals(labels,mapping,annotations,length)
        for _,start,end in (s for s in repo['ranges'] if s[0]==episode):
            if end>length:raise ValueError('curated range exceeds original episode')
        result=dict(root=root,info=info,ep=ep,length=length,data=data,head=head,tasks=tasks,subtasks=subtasks,annotation_file=str(ann) if has_annotations else None)
        self._episodes[key]=result
        while len(self._episodes)>self.max_cached_episodes:self._episodes.popitem(last=False)
        return result

    def _options(self,metadata,ranges,level):
        options=[]
        for segment in metadata['subtasks' if level=='subtask' else 'tasks']:
            for _,a,b in ranges:
                start=max(a,segment['start']);end=min(b,segment['end_exclusive']-1)
                if start<end:options.append((segment,start,end))
        return options

    def sample(self,step):
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,int(step)]))
        repo=self.repos[int(rng.choice(len(self.repos),p=self._weights))]
        ranges=repo['ranges'];lengths=np.array([e-s for _,s,e in ranges],float)
        selected=ranges[int(rng.choice(len(ranges),p=lengths/lengths.sum()))];episode=selected[0]
        metadata=self._episode(repo,episode);ranges=[s for s in ranges if s[0]==episode]
        level='subtask' if rng.random()<self.subtask_probability and self._options(metadata,ranges,'subtask') else 'task'
        return self._sample_episode(repo,episode,metadata,ranges,level,rng)

    def _sample_episode(self,repo,episode,metadata,ranges,level,rng):
        options=self._options(metadata,ranges,level)
        if not options:raise ValueError(f'no future {level} goal in curated observation ranges: {repo["repo"]}:{episode}')
        lengths=np.array([b-a for _,a,b in options],float)
        segment,a,b=options[int(rng.choice(len(options),p=lengths/lengths.sum()))]
        anchor=int(rng.integers(a,b));goal=segment['end_exclusive']-1
        return self._materialize(repo,episode,metadata,segment,anchor,goal,level)

    def evaluation_samples(self):
        if self._evaluations is not None:return self._evaluations
        result=[]
        self._evaluating=True
        for family in sorted({r['family'] for r in self.repos}):
            found=set()
            # Bound source reads explicitly. An unavailable subtask in these sources
            # is reported as absent from this diagnostic set, never relabeled.
            for repo,episode in islice(((r,e) for r in self.repos if r['family']==family for e in dict.fromkeys(s[0] for s in r['ranges'])),8):
                metadata=self._episode(repo,episode);ranges=[s for s in repo['ranges'] if s[0]==episode]
                for level in ('task','subtask'):
                    if level not in found and self._options(metadata,ranges,level):
                        result.append(self._sample_episode(repo,episode,metadata,ranges,level,np.random.default_rng(self.seed)));found.add(level)
                if len(found)==2:break
        self._evaluating=False
        self._evaluations=result;return result

    def _materialize(self,repo,episode,m,segment,anchor,goal,level):
        import pyarrow.parquet as pq
        from PIL import Image
        from scripts.data.prepare_subtask_goals import decode_frames,recorded_tcp_transform,camera_eligibility,clean_json
        from scripts.data.prepare_goal_labels import goal_camera
        from scripts.data.trajectory_projection import goal_source_tcp,project_goal_tcp
        from openwam.train.goal_overfit import prepare_overfit_images
        prefix='videos/'+m['head']+'/'
        video=m['root']/m['info']['video_path'].format(video_key=m['head'],chunk_index=int(m['ep'][prefix+'chunk_index']),file_index=int(m['ep'][prefix+'file_index']))
        images,alignment=decode_frames(video,float(m['ep'][prefix+'from_timestamp']),float(m['info']['fps']),[anchor,goal],sequential=repo['family']=='mecka')
        rows=pq.read_table(m['data'],filters=[('episode_index','=',episode),('frame_index','=',goal)]).to_pylist()
        if len(rows)!=1:raise ValueError('goal geometry source row not unique')
        endpoint=goal_source_tcp(repo['family'],rows[0],tcp_from_eef=recorded_tcp_transform(repo['family'],m['info']))
        eligible,qa=camera_eligibility(repo['family'],m['root'],episode)
        camera,cam_meta=goal_camera(repo['family'],rows[0],m['head'],m['info'],images[goal].size,geometry_eligible=eligible)
        projection=dict(uv=np.zeros((2,2),np.float32),projection_valid=np.zeros(2,bool),in_frame=np.zeros(2,bool),visibility=np.full(2,-1,np.int8))
        if camera is not None:
            k,e,model,d,wh=camera
            projection=project_goal_tcp(endpoint['tcp_world'],e,k,wh,camera_model=model,distortion=d,tcp_valid=endpoint['tcp_valid'])
        identity=f'{repo["repo"]}:{episode}:{level}:{segment["start"]}:{anchor}:{goal}'
        ident=hashlib.sha256(identity.encode()).hexdigest()[:24]
        # Training assets occupy a fixed rotating set; evaluation assets persist.
        slot=('evaluation_'+ident if self._evaluating else 'sample_'+str(self._asset_slot % self.max_cached_episodes))
        if not self._evaluating:self._asset_slot+=1
        folder=self.cache_dir/slot;folder.mkdir(exist_ok=True)
        for frame,name in ((anchor,'observed.png'),(goal,'goal.png')):
            images[frame].resize((320,256),Image.Resampling.BILINEAR).save(folder/name)
        np.savez_compressed(folder/'targets.npz',**{k:projection[k] for k in ('uv','projection_valid','in_frame','visibility')})
        record=dict(id=ident,metric_key=ident,family=repo['family'],granularity=level,instruction=segment['text'],
            source_repo=str(m['root']),source_episode=episode,source_episode_length=m['length'],
            instruction_start=segment['start'],instruction_end_exclusive=segment['end_exclusive'],
            anchor_source_frame=anchor,goal_source_frame=goal,head=m['head'],label_kind=level+'_end',success_verified=False,
            observed_image_path=str(folder/'observed.png'),goal_image_path=str(folder/'goal.png'),target_path=str(folder/'targets.npz'),
            provenance=dict(source_parquet=str(m['data']),source_video=str(video),annotation_file=m['annotation_file'],
                subtask_availability=('annotations_absent' if m['annotation_file'] is None else 'available_in_selected_ranges' if self._options(m,[s for s in repo['ranges'] if s[0]==episode],'subtask') else 'no_usable_subtask_intersects_selected_observation_ranges'),
                observed_video=alignment[anchor],goal_video=alignment[goal],goal_source_frame=goal,
                endpoint=endpoint,camera=cam_meta,calibration_eligibility_record=qa,
                goal_policy='original contiguous instruction end_exclusive-1; may extend curated observation range',geometry_review_status='pending_review'))
        record=clean_json(record)
        batch=prepare_overfit_images([record],device='cpu',dtype=self.dtype)[0]
        batch.update(instruction=record['instruction'],id=ident,family=repo['family'],granularity=level)
        return record,batch
