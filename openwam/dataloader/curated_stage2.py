"""Source-native, deterministic curated Stage 2 task/subtask windows."""
from __future__ import annotations
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from openwam.dataloader.head_history import HistorySamplingConfig, materialize_head_history
from openwam.dataloader.curated_goal import parse_curated_yaml
from openwam.dataloader.native_stage2_profiles import fingerprint
from openwam.dataloader.native_stage2_io import ByteLRU, load_episode, window_options, merge_source_masks, source_frame_validity
from openwam.dataloader.utils.gripper import filter_config
from openwam.dataloader.transforms.observation import ObservationAugmentation


class CuratedStage2Dataset(Dataset):
    """Draw IDs address selected source identities, independently of cache state.

    ``normalization_registry=None`` permits metadata/statistics preparation only;
    model samples always require matching immutable native statistics.
    """
    def __init__(self,source_yaml,*,data_root=None,seed=42,normalization_registry=None,
                 split='train',max_cached_episodes=2,max_cache_bytes=256*1024*1024,
                 observation_augmentation=None,gripper_state_filter=None,draws_per_epoch=100000,history=None):
        self.selection=parse_curated_yaml(source_yaml,data_root);self.repos=self.selection['repos']
        self.seed=int(seed);self.split=split;self.draws_per_epoch=int(draws_per_epoch)
        self.history=history if isinstance(history,HistorySamplingConfig) else HistorySamplingConfig(**(history or {}))
        self.source_yaml_sha256=hashlib.sha256(Path(source_yaml).read_bytes()).hexdigest()
        self.filter_config=filter_config(gripper_state_filter)
        self._cache=ByteLRU(max_cached_episodes,max_cache_bytes)
        self._weights=np.array([r['weight'] for r in self.repos],float);self._weights/=self._weights.sum()
        self.augmentation=ObservationAugmentation(observation_augmentation,training=split=='train')
        self.registry=None;self.registry_path=None
        if normalization_registry is not None:
            from openwam.dataloader.native_stage2_stats import read_registry
            self.registry_path=str(Path(normalization_registry).resolve())
            self.registry=read_registry(normalization_registry,self.stats_contract())

    def stats_contract(self):
        return dict(format='openwam.native_stage2_stats.v1',selection_sha256=self.source_yaml_sha256,
                    data_root=self.selection['data_root'],split=self.split,horizon=32,video_stride=4,
                    source_sampling='contiguous native source rows; YAML frame_sample_fps does not rescale indices',
                    source_validity_policy='finite_gt_0.5; valid_anchor_and_future_within_32_v1',
                    source_pose_mask_policy='shared_local_head_all9_finite_gt_0.5_v1',
                    stored_mask_policy='declared_eef_blocks_all_flags_finite_v2',
                    filter_config=self.filter_config)

    @classmethod
    def from_config(cls,config,split='train'):
        from openwam.dataloader.utils import get_cfg
        source=get_cfg(config,'source_yaml') if split=='train' else get_cfg(config,'evaluation_source_yaml')
        if not source:raise ValueError('explicit source_yaml required' if split=='train' else 'independent evaluation_source_yaml required')
        keys=('data_root','seed','normalization_registry','max_cached_episodes','max_cache_bytes',
              'observation_augmentation','gripper_state_filter','draws_per_epoch','history')
        kwargs={k:get_cfg(config,k) for k in keys if get_cfg(config,k) is not None}
        if split!='train':
            stats=get_cfg(config,'evaluation_normalization_registry')
            if stats is not None:kwargs['normalization_registry']=stats
        if kwargs.get('normalization_registry') is None and not get_cfg(config,'metadata_only',False):
            raise ValueError('stats_entry_missing: normalization_registry required; use preflight --prepare-stats')
        return cls(source,split=split,**kwargs)

    def __len__(self):return self.draws_per_epoch
    def __getitem__(self,draw_id):return self.sample_at(draw_id)
    def __getstate__(self):
        state=self.__dict__.copy();state['_cache']=ByteLRU(self._cache.max_entries,self._cache.max_bytes);return state

    def _episode(self,repo,episode):
        key=(repo['repo'],episode)
        if key not in self._cache:
            value,size=load_episode(repo,episode,self.filter_config,self._cache.max_bytes)
            self._cache.put(key,value,size)
        return self._cache.get_value(key)

    def _options(self,m,ranges,level):
        # Temporary masks/prefix sums are bounded by this one cached episode.
        # Only compact allowed intervals survive; there is no full-dataset frame index.
        global_valid=source_frame_validity(m['columns'],m['length'])
        valid_prefix=np.r_[0,np.cumsum(global_valid)]
        bad_prefix=np.r_[0,np.cumsum(m['bad'])]
        options=[]
        for option in window_options(m,ranges,level):
            anchors=np.arange(option['start'],option['end_exclusive'])
            ends=np.minimum(anchors+33,option['last_source']+1)
            keep=global_valid[anchors] & (valid_prefix[ends]>valid_prefix[anchors+1])
            keep &= bad_prefix[ends]==bad_prefix[anchors]
            boundaries=np.diff(np.r_[False,keep,False].astype(np.int8))
            starts=np.flatnonzero(boundaries==1)+option['start']
            stops=np.flatnonzero(boundaries==-1)+option['start']
            options.extend({**option,'start':int(a),'end_exclusive':int(b)} for a,b in zip(starts,stops))
        return options

    def _choose(self,m,ranges,rng,level=None):
        sub=self._options(m,ranges,'subtask')
        if level is None:level='subtask' if sub and rng.random()<.5 else 'task'
        options=sub if level=='subtask' else self._options(m,ranges,'task')
        if not options:raise ValueError(f'{m["root"]}:{m["ep"]["episode_index"]}: no valid future {level} windows (gripper_spike_rejected or interval boundary)')
        weights=np.array([o['end_exclusive']-o['start'] for o in options],float)
        option=options[int(rng.choice(len(options),p=weights/weights.sum()))]
        return option,int(rng.integers(option['start'],option['end_exclusive'])),level

    def _draw(self,draw_id):
        if type(draw_id) is not int or draw_id<0:raise ValueError('draw_id must be a nonnegative integer')
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,draw_id]))
        repo=self.repos[int(rng.choice(len(self.repos),p=self._weights))]
        # Selecting a range first preserves selected-duration episode weights.
        weights=np.array([b-a for _,a,b in repo['ranges']],float)
        episode=repo['ranges'][int(rng.choice(len(weights),p=weights/weights.sum()))][0]
        m=self._episode(repo,episode);ranges=[r for r in repo['ranges'] if r[0]==episode]
        option,anchor,level=self._choose(m,ranges,rng)
        return repo,episode,m,option,anchor,level

    def raw_window(self,m,option,anchor):
        end=min(anchor+33,option['last_source']+1)
        cols={k:v[anchor:end] for k,v in m['columns'].items()}
        raw,mask,state,sm=m['representation'].encode_window(cols)
        frame_valid=merge_source_masks(m['representation'],cols,mask,sm)
        if not m['head_calibration_eligible']:mask[:,34:]=False;sm[:,34:]=False
        raw[~mask]=0;state[~sm]=0
        return cols,raw,mask,state,sm,frame_valid

    def _decode_views(self,m,indices):
        from scripts.data.prepare_subtask_goals import decode_frames
        from openwam.dataloader.native_stage2_io import validate_video_metadata
        views=[{} for _ in indices];trace={}
        for camera in m['profile']['cameras']:
            if camera is None:continue
            checked=m.setdefault('video_offsets',{})
            if camera not in checked:checked[camera]=validate_video_metadata(m,camera)
            prefix='videos/'+camera+'/'
            ep=m['ep'];fps=float(m['info']['fps'])
            path=m['root']/m['info']['video_path'].format(video_key=camera,episode_index=int(ep['episode_index']),chunk_index=int(ep[prefix+'chunk_index']),file_index=int(ep[prefix+'file_index']))
            offset=float(ep[prefix+'from_timestamp'])
            # Native metadata carries the per-file start; decode verifies integral CFR ordinals.
            if prefix+'to_timestamp' in ep and float(ep[prefix+'to_timestamp'])<offset+(max(indices)+1)/fps-1/fps:
                raise ValueError(f'{path}: invalid_segment_bounds/video_offset')
            images,alignment=decode_frames(path,offset,fps,indices,sequential=m['profile']['family']=='mecka',
                video_ordinals=m['video_ordinals'] if m['profile']['family']=='mecka' else None)
            for target,index in zip(views,indices):target[camera]=images[index]
            trace[camera]=dict(path=str(path),alignment=alignment)
        return views,trace

    def sample_at(self,draw_id):
        if self.registry is None:raise ValueError('stats_entry_missing: prepare a native normalization_registry before model sampling')
        repo,episode,m,option,anchor,level=self._draw(draw_id)
        return self._sample_selected(repo,episode,m,option,anchor,level,draw_id=draw_id)

    def _sample_selected(self,repo,episode,m,option,anchor,level,*,draw_id=None):
        from openwam.dataloader.native_stage2_stats import lookup_stats, normalize
        stats=lookup_stats(self.registry,repo,m)
        cols,raw,mask,state,sm,frame_valid=self.raw_window(m,option,anchor)
        norm=normalize(raw,mask,stats['action'],source=f'{repo["repo"]}:{episode}:action')
        norm_state=normalize(state,sm,stats['state'],source=f'{repo["repo"]}:{episode}:state')
        rep=m['representation'];action=np.zeros((32,112),np.float32);action_mask=np.zeros_like(action,bool)
        action[:len(raw),rep.scatter]=norm;action_mask[:len(raw),rep.scatter]=mask
        proprio=np.zeros((1,112),np.float32);proprio_mask=np.zeros_like(proprio,bool)
        proprio[:,rep.scatter]=norm_state;proprio_mask[:,rep.scatter]=sm
        video_indices=anchor+np.arange(9)*4;video_mask=video_indices<=option['last_source']
        real_indices=video_indices[video_mask].tolist();views,alignment=self._decode_views(m,real_indices)
        from PIL import Image
        from openwam.dataloader.transforms.multiview import assemble_multiview_layout
        camera=m['profile']['cameras'][0];head=views[0].get(camera)
        from openwam.model.goal.runtime import clean_goal_head
        clean=None if head is None else clean_goal_head(head)
        canvases=[];missing=[];view_mask=np.zeros((9,3),bool)
        for i,v in enumerate(views):
            canvas,absent=assemble_multiview_layout(v,m['profile']['cameras'],384,320,return_missing_mask=True)
            canvases.append(canvas);missing.append(absent)
            view_mask[i]=[c is not None and c in v for c in m['profile']['cameras']]
            video_mask[i] &= frame_valid[i*4]
        while len(canvases)<9:canvases.append(canvases[-1].copy());missing.append(missing[-1].copy())
        view_mask &= video_mask[:,None]
        from openwam.model.goal.observation import clean_multiview
        current_mask = torch.from_numpy(view_mask[0].copy())
        clean_canvas = (clean_multiview([views[0].get(c) for c in m['profile']['cameras']],current_mask)
                        if current_mask.any() else torch.full((3,1,384,320),-1.))
        instruction=option['instruction']
        seed_material=[self.seed,repo['repo'],episode,anchor,instruction['text']]
        goal_seed=int(fingerprint(seed_material)[:15],16)
        augmentation_seed=goal_seed if draw_id is None else int(fingerprint([goal_seed,draw_id])[:15],16)
        identity=[repo['repo'],episode,anchor,level,[instruction['start'],instruction['end_exclusive']],
                  [option['selected_start'],option['selected_end_exclusive']]]
        sample=dict(video=canvases,vace_video=None,first_frame_image=[canvases[0]],action=torch.from_numpy(action),
            action_mask=torch.from_numpy(action_mask),video_mask=torch.from_numpy(video_mask),
            video_view_mask=torch.from_numpy(view_mask),stage2_strict_video_mask=True,
            proprio=torch.from_numpy(proprio),proprio_mask=torch.from_numpy(proprio_mask),
            prompt=instruction['text'],instruction=instruction['text'],granularity=level,
            instruction_start=instruction['start'],instruction_end_exclusive=instruction['end_exclusive'],
            goal_observed_head=clean,goal_observed_multiview=clean_canvas,goal_view_valid=current_mask,goal_head_valid=head is not None and bool(frame_valid[0]),goal_seed=goal_seed,
            report_target=dict(goal_source_frame=instruction['end_exclusive']-1,label_kind=level+'_end',success_verified=False),
            trace=dict(identity=identity,draw_id=draw_id,source_repo=str(m['root']),episode=episode,anchor=anchor,last_source=option['last_source'],
                source_parquet=str(m['data']),source_parquets=[str(p) for p in m['data_files']],info_sha256=m['info_hash'],profile_id=m['profile']['profile_id'],
                raw_action=raw,raw_action_mask=mask,raw_state=state,raw_state_mask=sm,
                video_source_indices=video_indices.tolist(),video_alignment=alignment,
                mask_reasons=m['profile']['mask_reasons']+(['head_calibration_ineligible'] if not m['head_calibration_eligible'] else []),
                calibration_eligibility=m['head_calibration_qa'],gripper_filter_reasons=m['filter_reasons'],
                selected_range=[option['selected_start'],option['selected_end_exclusive']],
                goal_resize={'source_wh':list(head.size) if head else None,'destination_wh':[320,256]},
                stats_registry=self.registry_path,normalization_population=stats['population']))
        if self.history.enabled:
            history_seed=int(fingerprint([goal_seed,'head_history',draw_id])[:15],16)
            fields,history_trace=materialize_head_history(m['root'],m['info'],m['ep'],camera,anchor,
                option['selected_start'],self.history,np.random.default_rng(history_seed),self.split=='train',
                sequential=m['profile']['family']=='mecka',
                source_valid=source_frame_validity(m['columns'],m['length']),
                video_ordinals=m['video_ordinals'] if m['profile']['family']=='mecka' else None)
            sample.update(fields);sample['trace']['head_history']=history_trace
        if m['fk']:
            sample['fk_supervision']=m['fk'].build(cols,raw,mask,state,sm,
                {k:np.array(v) for k,v in stats['action'].items() if k!='count'},'quantile',32)
        # Deterministic augmentation without perturbing the caller/worker RNG stream.
        old_random=random.getstate();old_np=np.random.get_state();old_torch=torch.random.get_rng_state()
        try:
            random.seed(augmentation_seed);np.random.seed(augmentation_seed%(2**32));torch.random.set_rng_state(torch.Generator(device="cpu").manual_seed(augmentation_seed).get_state())
            sample=self.augmentation.apply(sample,missing)
        finally:
            random.setstate(old_random);np.random.set_state(old_np);torch.random.set_rng_state(old_torch)
        return sample

    def normalization_identity(self):
        """Portable checkpoint identity: immutable bytes and sampling contract."""
        if self.registry_path is None:raise ValueError('stats_entry_missing')
        return dict(registry_path=self.registry_path,
            registry_sha256=hashlib.sha256(Path(self.registry_path).read_bytes()).hexdigest(),
            selection_sha256=self.source_yaml_sha256,contract=self.stats_contract(),
            assets=[self.registry_path,self.selection['source_yaml']])

    def scope_summary(self):
        return dict(repositories=len(self.repos),episodes=sum(len({e for e,_,_ in r['ranges']}) for r in self.repos),
                    selected_frames=sum(b-a for r in self.repos for _,a,b in r['ranges']),
                    families=sorted({r['family'] for r in self.repos}),source_yaml=self.selection['source_yaml'],
                    max_cached_episodes=self._cache.max_entries,max_cache_bytes=self._cache.max_bytes,
                    statistics_ready=self.registry is not None,all_videos_audited=False)
