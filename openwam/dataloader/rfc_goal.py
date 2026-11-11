"""Finite proportional task/subtask data and source-native 1 Hz goal memory.

Every memory frame is a single three-view canvas. Missing views stay black;
cache view identity names the ordered composite (the VAE mixes across tiles).
The disk cache is checked before decoding any history video. Source snapshots
are immutable for the lifetime of a dataset; rebuild after a dependency change.
Metadata is content-hashed. Large parquet/video generations use resolved paths,
device/inode/size/mtime/ctime, avoiding a corpus-sized data read at every startup.
This detects normal source replacements/edits, not privileged stat forgery.
"""
from pathlib import Path
from copy import deepcopy
import hashlib
import json
import math
import numpy as np
import torch
from torch.utils.data import Dataset
from openwam.dataloader.curated_goal import CuratedGoalDataset
from openwam.model.goal.goal_memory_cache import select_memory_indices
from openwam.model.goal.observation import clean_multiview


def select_history_frames(anchor, fps, recent_frames=2):
    if not math.isfinite(fps) or fps<=0:raise ValueError('source fps must be positive')
    if anchor<0:raise ValueError('anchor must be nonnegative')
    select_memory_indices([],recent_frames=recent_frames)
    if anchor==0:return []
    if fps<=1:
        # A sub-Hz source has every native frame on the deduplicated 1Hz grid.
        recent=range(max(0,anchor-recent_frames),anchor)
        return sorted({0,*recent})
    last=int(math.ceil(anchor/fps))-1
    frames={0}
    for second in range(max(0,last-recent_frames+1),last+1):
        frame=int(math.floor(second*fps+1e-9))
        if frame<anchor:frames.add(frame)
    return sorted(frames)



def _stamp(path):
    try:
        st=Path(path).stat()
        return (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
    except FileNotFoundError:return None


def _digest_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def _video(metadata,episode,camera):
    prefix='videos/'+camera+'/'
    ep=metadata['ep'];info=metadata['info']
    chunk,file,offset=(ep[prefix+k] for k in ('chunk_index','file_index','from_timestamp'))
    if any(isinstance(v,bool) or int(v)!=v or v<0 for v in (chunk,file)) or not math.isfinite(offset) or offset<0:
        raise ValueError('invalid camera source video metadata')
    return metadata['root']/info['video_path'].format(video_key=camera,episode_index=episode,
        chunk_index=int(chunk),file_index=int(file)),float(offset)


class FrozenEpisodeMetadata:
    """One repository manifest; immutable selected rows, local runtime checks.

    Global uniqueness is established before accepting the population. The full
    manifest is checked at construction and epoch boundaries. Within an epoch,
    only the selected fragment is checked; unselected fragments cannot change
    the frozen rows or inject new duplicate rows into a running reader.
    """
    def __init__(self,root,selected):
        import pyarrow.parquet as pq
        self.root=Path(root).resolve();self.rows={};self.fragments={}
        self.manifest=self._manifest()
        if not self.manifest:raise ValueError(f'{root}: missing episode metadata')
        selected=set(selected);seen=set()
        for path,stamp in self.manifest:
            file=pq.ParquetFile(path)
            ids=file.read(columns=['episode_index'])['episode_index'].to_pylist()
            for episode in ids:
                if episode in seen:raise ValueError(f'{root}:{episode}: nonunique episode metadata')
                seen.add(episode)
            wanted=selected.intersection(ids)
            if wanted:
                columns=[name for name in file.schema_arrow.names if not name.startswith('stats/')]
                rows=pq.read_table(path,columns=columns,filters=[('episode_index','in',sorted(wanted))]).to_pylist()
                for row in rows:
                    episode=int(row['episode_index'])
                    self.rows[episode]=row;self.fragments[episode]=(path,stamp)
        missing=selected-self.rows.keys()
        if missing:raise ValueError(f'{root}:{sorted(missing)}: missing episode metadata')
        self.validate()
        self.identity=hashlib.sha256(json.dumps(self.manifest,sort_keys=True).encode()).hexdigest()

    def _manifest(self):
        return tuple((str(p.resolve()),_stamp(p)) for p in sorted((self.root/'meta/episodes').glob('chunk-*/*.parquet')))

    def validate(self):
        if self._manifest()!=self.manifest:
            raise ValueError(f'{self.root}: episode metadata source changed; rebuild population')

    def read(self,episode):
        path,stamp=self.fragments[episode]
        if _stamp(path)!=stamp:raise ValueError(f'source dependency changed: {path}; rebuild population')
        return deepcopy(self.rows[episode])


class RFCGoalDataset(Dataset):
    def __init__(self, source_yaml, cache_dir, *, codec=None, epoch_size=1000,
                 subtask_probability=.7, seed=42, recent_frames=2, data_root=None,
                 path_aliases=None, dtype=torch.float32, require_cache=True, max_cached_episodes=8):
        if type(epoch_size) is not int or epoch_size<1:raise ValueError('epoch_size must be positive integer')
        if not 0<=subtask_probability<=1:raise ValueError('invalid subtask probability')
        select_history_frames(0,1,recent_frames)
        self.reader=CuratedGoalDataset(source_yaml,cache_dir,seed=seed,
            subtask_probability=subtask_probability,data_root=data_root,path_aliases=path_aliases,
            dtype=dtype,observation_mode='multiview',materialize_multiview_target=True,
            max_cached_episodes=max_cached_episodes)
        self.codec=codec;self.epoch_size=epoch_size;self.seed=int(seed);self.epoch=0
        self.recent_frames=recent_frames;self.require_cache=require_cache
        self.subtask_probability=subtask_probability;self.dtype=self.reader.dtype
        self.pools={'task':[],'subtask':[]};self.sources={};self._file_digests={}
        self.repo_identity={r['repo']:hashlib.sha256(json.dumps(r,sort_keys=True).encode()).hexdigest()
                            for r in self.reader.repos}
        selected={}
        for repo in self.reader.repos:
            selected.setdefault(repo['path'],set()).update(e for e,_,_ in repo['ranges'])
        self.repository_metadata={root:FrozenEpisodeMetadata(root,episodes) for root,episodes in selected.items()}
        self.reader.episode_metadata_reader=self._read_frozen_episode
        for repo in self.reader.repos:
            for episode in dict.fromkeys(e for e,_,_ in repo['ranges']):
                m=self.reader._episode(repo,episode)
                ranges=[r for r in repo['ranges'] if r[0]==episode]
                source=self._snapshot(repo,episode,m)
                self.sources[(repo['repo'],episode)]=source
                for level in self.pools:
                    for segment,a,b in self.reader._options(m,ranges,level):
                        # Compact intervals avoid one Python object per source frame.
                        valid=np.asarray(m['source_valid'][a:b],dtype=bool)
                        edges=np.flatnonzero(np.diff(np.r_[False,valid,False]))
                        for start,stop in zip(edges[::2]+a,edges[1::2]+a):
                            self.pools[level].append(dict(repo=repo,episode=episode,
                                segment=segment,start=int(start),stop=int(stop),
                                level=level,source=source['identity']))
        for level,prob in [('task',1-subtask_probability),('subtask',subtask_probability)]:
            if prob and not self.pools[level]:raise ValueError(f'no eligible {level} population for requested mixture')
        self._rebuild_sampler()
        population=dict(sources=sorted(s['identity'] for s in self.sources.values()),
                        epoch_size=epoch_size,seed=seed,subtask_probability=subtask_probability,
                        recent_frames=recent_frames,policy='strict-past-episode-grid-1Hz-sink-recent-v1',
                        sampler='global-grain-then-repo-marginal-then-uniform-valid-anchor-v2')
        self.population_identity=hashlib.sha256(json.dumps(population,sort_keys=True).encode()).hexdigest()
        for repository in self.repository_metadata.values():repository.validate()
        self.set_epoch(0)

    def _rebuild_sampler(self):
        self.pool_weights={};self.repo_pools={}
        for level,pool in self.pools.items():
            totals={}
            for item in pool:
                name=item['repo']['repo'];totals[name]=totals.get(name,0)+item['stop']-item['start']
            # YAML weights are repository marginals, not per-frame density.
            # Normalize only among repositories eligible for this chosen grain.
            weights=np.array([x['repo']['weight']*(x['stop']-x['start'])/totals[x['repo']['repo']]
                              for x in pool],dtype=float)
            self.pool_weights[level]=weights/weights.sum() if len(weights) else weights
            groups={}
            for index,item in enumerate(pool):groups.setdefault(item['repo']['repo'],[]).append(index)
            by_repo=[]
            for indices in groups.values():
                cumulative=np.cumsum([pool[i]['stop']-pool[i]['start'] for i in indices],dtype=np.int64)
                by_repo.append((np.asarray(indices),cumulative,pool[indices[0]]['repo']['weight']))
            repo_weights=np.array([group[2] for group in by_repo],dtype=float)
            self.repo_pools[level]=(by_repo,repo_weights/repo_weights.sum() if len(repo_weights) else repo_weights)

    def restrict_stage2(self,native):
        """Intersect compact pools with actual native action eligibility once.

        Keeps instruction endpoints and grain/repository marginals. Failed
        restrictions leave the existing population unchanged; no per-step redraw.
        """
        def instruction_key(value):
            return (int(value['start']),int(value['end_exclusive']),str(value['text']))
        options_by_episode={}
        filtered={level:[] for level in self.pools}
        for level,pool in self.pools.items():
            for item in pool:
                self._check_source(item)
                key=(item['repo']['repo'],item['episode'],level)
                if key not in options_by_episode:
                    metadata=native._episode(item['repo'],item['episode'])
                    ranges=[r for r in item['repo']['ranges'] if r[0]==item['episode']]
                    options={}
                    for option in native._options(metadata,ranges,level):
                        options.setdefault(instruction_key(option['instruction']),[]).append(option)
                    options_by_episode[key]=options
                matches=options_by_episode[key].get(instruction_key(item['segment']),[])
                previous_stop=None
                for option in sorted(matches,key=lambda o:o['start']):
                    begin=max(item['start'],int(option['start']))
                    end=min(item['stop'],int(option['end_exclusive']))
                    if begin>=end:continue
                    if previous_stop is not None and begin<previous_stop:
                        raise ValueError('overlapping native eligible action options')
                    previous_stop=end
                    filtered[level].append(dict(item,start=begin,stop=end,stage2_option=option))
        for level,probability in [('task',1-self.subtask_probability),('subtask',self.subtask_probability)]:
            if probability and not filtered[level]:
                raise ValueError(f'no eligible {level} population after native action-window intersection')
        rows=[(level,item['repo']['repo'],item['episode'],instruction_key(item['segment']),
               item['start'],item['stop'],item['stage2_option']['last_source'])
              for level,pool in filtered.items() for item in pool]
        base=getattr(self,'_stage1_population_identity',self.population_identity)
        identity=hashlib.sha256(json.dumps(dict(stage1=base,native_action_runs=rows,
            policy='same-instruction-native-eligible-intersection-v1'),sort_keys=True).encode()).hexdigest()
        self.pools=filtered
        self._rebuild_sampler()
        self._stage1_population_identity=base
        self.population_identity=identity
        self.stage2_restricted=True

    def _read_frozen_episode(self,root,episode):
        return self.repository_metadata[str(Path(root).resolve())].read(episode)

    def _snapshot(self,repo,episode,m):
        repository=self.repository_metadata[repo['path']]
        fragment,fragment_stamp=repository.fragments[episode]
        if m['ep']!=repository.rows[episode]:raise ValueError('episode metadata differs from frozen population')
        root=m['root']
        paths={str((root/'meta'/name).resolve()) for name in ('info.json','tasks.parquet','sub_tasks.parquet')}
        paths.update(str(Path(p).resolve()) for p in m.get('data_files',[]))
        paths.add(fragment)
        for key in ('data','annotation_file'):
            if m.get(key):paths.add(str(Path(m[key]).resolve()))
        paths.add(self.reader.selection['source_yaml'])
        for camera in m['cameras']:
            if camera is not None:paths.add(str(_video(m,episode,camera)[0].resolve()))
        files=[]
        video_paths={str(_video(m,episode,c)[0].resolve()) for c in m['cameras'] if c is not None}
        large_paths=video_paths | {fragment} | {str(Path(p).resolve()) for p in m.get('data_files',[])}
        for path in sorted(paths):
            stamp=_stamp(path);key=(path,stamp)
            if key not in self._file_digests:
                self._file_digests[key]=(_digest_file(path) if stamp is not None and path not in large_paths else None)
            digest=self._file_digests[key]
            if _stamp(path)!=stamp:raise ValueError(f'source dependency changed while hashing: {path}')
            files.append((path,stamp,digest))
        identity=hashlib.sha256(json.dumps(dict(repo=self.repo_identity[repo['repo']],episode=episode,ep=m['ep'],
            cameras=m['cameras'],files=files,repository_metadata=repository.identity),sort_keys=True,default=str).encode()).hexdigest()
        return dict(identity=identity,files=files)

    def _check_source(self,item):
        source=self.sources[(item['repo']['repo'],item['episode'])]
        for path,stamp,_ in source['files']:
            if _stamp(path)!=stamp:raise ValueError(f'source dependency changed: {path}; rebuild dataset/cache')

    def __len__(self):return self.epoch_size

    def set_epoch(self,epoch):
        if type(epoch) is not int or epoch<0:raise ValueError('epoch must be a nonnegative integer')
        if epoch!=self.epoch:
            for repository in self.repository_metadata.values():repository.validate()
        self.epoch=epoch
        count=int(math.floor(self.epoch_size*self.subtask_probability+.5))
        self._levels=np.array(['subtask']*count+['task']*(self.epoch_size-count))
        np.random.default_rng(np.random.SeedSequence([self.seed,epoch])).shuffle(self._levels)

    def selection_at(self,index):
        if not 0<=index<len(self):raise IndexError(index)
        level=str(self._levels[index]);pool=self.pools[level]
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch,int(index),71]))
        groups,weights=self.repo_pools[level]
        indices,cumulative,_=groups[int(rng.choice(len(groups),p=weights))]
        offset=int(rng.integers(int(cumulative[-1])))
        run=int(np.searchsorted(cumulative,offset,side='right'))
        item=dict(pool[int(indices[run])])
        item['anchor']=item['start']+offset-(int(cumulative[run-1]) if run else 0)
        item['goal']=item['segment']['end_exclusive']-1
        return item

    def memory_requests(self,index):
        item=self.selection_at(index);self._check_source(item)
        m=self.reader._episode(item['repo'],item['episode'])
        ids=select_history_frames(item['anchor'],float(m['info']['fps']),self.recent_frames)
        return [dict(source=item['source'],episode=item['episode'],frame=frame,
                     repo=item['repo'],view_valid=[c is not None and bool(m['source_valid'][frame]) for c in m['cameras']],
                     age=(item['anchor']-frame)/float(m['info']['fps'])) for frame in ids]

    def precompute_requests(self):
        """All reachable memory frames, covering every epoch/mixture draw."""
        seen=set()
        for pool in self.pools.values():
            for item in pool:
                self._check_source(item)
                m=self.reader._episode(item['repo'],item['episode'])
                fps=float(m['info']['fps'])
                frames=set(select_history_frames(item['start'],fps,self.recent_frames))
                if item['stop']>1:frames.add(0)
                if self.recent_frames:
                    # Across a contiguous anchor run, every intervening 1Hz frame
                    # can be the latest past frame; no per-anchor enumeration.
                    first=max(0,int(math.floor(item['start']/fps)))
                    last=int(math.ceil((item['stop']-1)/fps))
                    frames.update(int(math.floor(k*fps+1e-9)) for k in range(first,last)
                        if item['start']<=int(math.floor(k*fps+1e-9))<item['stop']-1)
                for frame in sorted(frames):
                    key=(item['source'],item['episode'],frame)
                    if key in seen:continue
                    seen.add(key)
                    yield dict(source=item['source'],episode=item['episode'],frame=frame,repo=item['repo'],
                        view_valid=[c is not None and bool(m['source_valid'][frame]) for c in m['cameras']])

    def memory_source_identity(self,index):return self.selection_at(index)['source']

    def decode_memory(self,request):
        from scripts.data.prepare_subtask_goals import decode_frames
        item=dict(repo=request['repo'],episode=request['episode']);self._check_source(item)
        m=self.reader._episode(request['repo'],request['episode']);frames=[None]*3
        valid=torch.tensor(request['view_valid'],dtype=torch.bool)
        if not valid.any():return torch.full((3,1,384,320),-1.,dtype=self.dtype)
        for i,camera in enumerate(m['cameras']):
            if not valid[i]:continue
            path,offset=_video(m,request['episode'],camera)
            images,_=decode_frames(path,offset,float(m['info']['fps']),[request['frame']],
                sequential=request['repo']['family']=='mecka',video_ordinals=m.get('video_ordinals'))
            frames[i]=images[request['frame']]
        self._check_source(item)
        return clean_multiview(frames,valid).to(dtype=self.dtype)

    @staticmethod
    def cache_key(request):
        return {k:request[k] for k in ('source','episode','frame')}

    def __getitem__(self,index):
        item=self.selection_at(index);requests=self.memory_requests(index)
        count=1+self.recent_frames
        memory=torch.zeros((count,48,1,24,20),dtype=self.dtype)
        images=torch.full((count,3,1,384,320),-1.,dtype=self.dtype) if self.codec is None else None
        valid=torch.zeros(count,dtype=torch.bool);views=torch.zeros(count,3,dtype=torch.bool)
        ages=torch.zeros(count);ids=torch.full((count,),-1,dtype=torch.long)
        for i,request in enumerate(requests):
            views[i]=torch.tensor(request['view_valid']);valid[i]=views[i].any()
            ages[i]=request['age'];ids[i]=request['frame']
            if not valid[i]:continue
            if self.codec is None:images[i]=self.decode_memory(request)
            else:
                key=self.cache_key(request);latent=self.codec.get_cached(**key)
                if latent is None:
                    if self.require_cache:raise FileNotFoundError(f'missing goal memory cache {key}; run scripts/data/precompute_goal_memory.py')
                    latent=self.codec.encode_image(self.decode_memory(request)).detach().cpu()
                    self.codec.put_cached(**key,latent=latent)
                memory[i]=latent.to(dtype=self.dtype)
        m=self.reader._episode(item['repo'],item['episode'])
        record,batch=self.reader._materialize(item['repo'],item['episode'],m,item['segment'],
                                              item['anchor'],item['goal'],item['level'])
        self._check_source(item)
        targets=batch['targets']
        result=dict(observed=batch['observed'][0],observed_view_valid=batch['observed_view_valid'][0],
            memory_latents=memory,memory_valid=valid,memory_view_valid=views,memory_ages=ages,memory_frame_ids=ids,
            target_image=batch['target_image'][0],target_uv=targets.uv[0],
            target_image_valid=batch['target_image_valid'][0],target_view_valid=batch['target_view_valid'][0],
            target_point_valid=targets.point_valid[0],target_in_frame=targets.in_frame[0],
            goal_horizon_s=torch.tensor((item['goal']-item['anchor'])/float(m['info']['fps'])),
            instruction=record['instruction'],granularity=item['level'],record=record,
            memory_source_identity=item['source'],population_identity=self.population_identity)
        if images is not None:result['memory_images']=images
        return result

    def sample(self,step):
        epoch,index=divmod(int(step),len(self))
        if epoch!=self.epoch:self.set_epoch(epoch)
        return self[index]

    def scope_summary(self):
        return dict(self.reader.scope_summary(),epoch_size=self.epoch_size,
            subtask_probability=self.subtask_probability,task_samples=int((self._levels=='task').sum()),
            subtask_samples=int((self._levels=='subtask').sum()),population_identity=self.population_identity,
            memory_policy='1Hz strict past, episode sink plus recent',recent_frames=self.recent_frames,
            independent_vae_t1=True,temporal_compression=False,
            pool_anchors={k:sum(v['stop']-v['start'] for v in pool) for k,pool in self.pools.items()})


def collate_goal_memory(samples):
    if not samples:raise ValueError('cannot collate empty goal memory batch')
    return {key:torch.stack([s[key] for s in samples]) if isinstance(value,torch.Tensor)
            else [s[key] for s in samples] for key,value in samples[0].items()}
