"""Bounded, all-rank Stage2 generation with immutable rank-zero report publication.

Every rank generates the same ordered cases (including collective-bearing backbones).
Metrics are rank-zero authoritative, never summed across duplicated rank samples.
Only scalar source descriptors are retained; images and predictions are handled one
case at a time. Future labels are used exclusively by reporting and metrics.
"""
from __future__ import annotations
from contextlib import contextmanager
import copy
import hashlib
import html
import json
import logging
import os
from pathlib import Path
import random
import tempfile
import uuid

import numpy as np
import torch
from PIL import Image, ImageDraw
from openwam.deploy.denoise_schedule import make_schedule

LOG = logging.getLogger(__name__)
SCHEMA = 'openwam.stage2.evaluation.v1'
SCATTER = np.r_[0:10,34:44,80:94,94:112]
TILES = ((0,0,320,256),(0,256,160,384),(160,256,320,384))


def _json_safe(value):
    if isinstance(value,(np.ndarray,torch.Tensor)):return _json_safe(_array(value).tolist())
    if isinstance(value,np.generic):return _json_safe(value.item())
    if isinstance(value,Path):return str(value)
    if isinstance(value,float) and not np.isfinite(value):return None
    if isinstance(value,dict):return {str(k):_json_safe(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [_json_safe(v) for v in value]
    return value


def _json(value):
    return json.dumps(_json_safe(value),ensure_ascii=False,sort_keys=True,allow_nan=False)


def _write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False, encoding='utf-8') as stream:
        stream.write(_json(value)+'\n'); temporary = stream.name
    os.replace(temporary, path)


def _sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): digest.update(block)
    return digest.hexdigest()


def _array(value):
    return value.detach().cpu().float().numpy() if torch.is_tensor(value) and value.dtype == torch.bfloat16 else (value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value))


def _distributed():
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _rank(): return torch.distributed.get_rank() if _distributed() else 0


def _agree_error(error, phase):
    """All ranks execute this before the next generation or publication boundary."""
    errors = [error]
    if _distributed():
        errors = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(errors, error)
    if any(errors): raise RuntimeError(f'Stage2 evaluation {phase}: ' + '; '.join(str(e) for e in errors if e))


def _same(value):
    if _distributed():
        values = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(values, value)
        if any(v != values[0] for v in values): raise ValueError('Stage2 fixed cases/conditions differ between ranks')


@contextmanager
def preserve_evaluation_state(architecture):
    """Restore RNG, every module mode and mutable scheduler buffers even on failure."""
    python = random.getstate(); numpy = np.random.get_state(); cpu = torch.random.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    modes = [(module, module.training) for module in architecture.modules()]
    schedulers = [(scheduler, copy.deepcopy(scheduler.__dict__)) for scheduler in
        (architecture.video_scheduler, architecture.action_scheduler)]
    try:
        architecture.eval()
        with torch.no_grad(): yield
    finally:
        for scheduler, state in schedulers:
            scheduler.__dict__.clear(); scheduler.__dict__.update(state)
        # Calling parent.train() would overwrite intentionally frozen child modes.
        for module, training in modes: module.training = training
        random.setstate(python); np.random.set_state(numpy); torch.random.set_rng_state(cpu)
        if cuda is not None: torch.cuda.set_rng_state_all(cuda)


def generation_conditions(sample, *, seed, num_steps, observation_mode='head', observation_dtype=None):
    """An explicit allowlist: no future mask, target, interval or geometry is read."""
    active = torch.as_tensor(sample['_eval']['active_action_mask'], dtype=torch.bool)
    head = sample.get('goal_observed_head') if observation_mode=='head' else None
    result = dict(first_frame_image=sample['first_frame_image'], num_frames=9, action_num_frames=33,
        height=384, width=320, seed=int(seed), num_inference_steps=int(num_steps),
        proprio=sample['proprio'].unsqueeze(0),
        _proprio_sample_mask=sample['proprio_mask'].unsqueeze(0), active_action_mask=active,
        goal_observed_head=None if head is None else head.unsqueeze(0),
        goal_head_valid=bool(sample['goal_head_valid']), goal_seed=int(sample['goal_seed']), decode_video=True)
    from openwam.model.goal.history import history_kwargs
    result.update({key:value.unsqueeze(0) for key,value in history_kwargs(sample).items()})
    if observation_mode == 'multiview':
        observed, valid = _clean_multiview_condition(sample)
        if observation_dtype is not None:observed=observed.to(dtype=observation_dtype)
        result.update(goal_observed_head=None,goal_observed_multiview=observed.unsqueeze(0),
                      goal_view_valid=valid.unsqueeze(0))
    elif observation_mode != 'head':
        raise ValueError('unsupported goal observation mode')
    return result


def _clean_multiview_condition(sample):
    from openwam.model.goal.observation import sanitize_canvas
    if sample.get('goal_view_valid') is None or sample.get('goal_observed_multiview') is None:
        raise ValueError('multiview evaluation requires explicit clean canvas and view mask')
    valid=torch.as_tensor(sample['goal_view_valid'])
    observed=sample['goal_observed_multiview']
    if valid.dtype != torch.bool or valid.shape != (3,):
        raise ValueError('evaluation goal_view_valid must be bool [3]')
    if bool(valid[0]) != bool(sample['goal_head_valid']):
        raise ValueError('evaluation view mask disagrees with head_valid')
    if not isinstance(observed,torch.Tensor) or tuple(observed.shape)!=(3,1,384,320):
        raise ValueError('evaluation clean canvas must be [3,1,384,320]')
    # Missing-head rows never invoke the expert and have wholly invalid memory.
    clean=sanitize_canvas(observed.unsqueeze(0),valid.unsqueeze(0))[0] if valid[0] else torch.full_like(observed,-1.)
    return clean, valid


def _conditions_sha(sample, kwargs):
    digest = hashlib.sha256(_json([sample['prompt'], kwargs['seed'], kwargs['goal_seed'], kwargs['goal_head_valid'],
        kwargs['num_inference_steps']]).encode())
    for key in ('proprio','_proprio_sample_mask','active_action_mask','goal_observed_head'):
        value=kwargs[key]
        if value is not None: digest.update(np.ascontiguousarray(_array(value)).tobytes())
    if 'goal_observed_multiview' in kwargs:
        from openwam.model.goal.runtime import MULTIVIEW_PREPROCESSING_VERSION
        digest.update(_json(['multiview',MULTIVIEW_PREPROCESSING_VERSION,str(kwargs['goal_observed_multiview'].dtype)]).encode())
        for key in ('goal_observed_multiview','goal_view_valid'):
            digest.update(np.ascontiguousarray(_array(kwargs[key])).tobytes())
    from openwam.model.goal.history import history_kwargs
    for key, value in history_kwargs(kwargs).items():
        digest.update(_json([key,list(value.shape),str(value.dtype)]).encode())
        digest.update(np.ascontiguousarray(_array(value)).tobytes())
    for image in kwargs['first_frame_image']: digest.update(image.tobytes())
    return digest.hexdigest()


def _validate_sample(sample, *, observation_mode='head'):
    if not isinstance(sample['prompt'], str) or not sample['prompt']: raise ValueError('empty fixed prompt')
    for key, shape in [('action',(32,112)),('action_mask',(32,112)),('proprio',(1,112)),('proprio_mask',(1,112)),
                       ('video_mask',(9,)),('video_view_mask',(9,3))]:
        if _array(sample[key]).shape != shape: raise ValueError(f'{key} shape must be {shape}')
    if len(sample['video']) != 9 or any(i.size != (320,384) for i in sample['video']):
        raise ValueError('GT video must have 9 native 320x384 canvases')
    if len(sample['first_frame_image'])!=1 or sample['first_frame_image'][0].size!=(320,384):
        raise ValueError('observed canvas shape mismatch')
    if _array(sample['_eval']['active_action_mask']).shape!=(112,): raise ValueError('structural support must be width112')
    if not np.isfinite(_array(sample['proprio'])).all(): raise ValueError('nonfinite observed proprio')
    if observation_mode == 'multiview':_clean_multiview_condition(sample)
    head=sample.get('goal_observed_head')
    if observation_mode == 'head' and head is not None and (tuple(head.shape)!=(3,1,256,320) or not torch.isfinite(head).all()):
        raise ValueError('invalid clean observed goal head')
    if not np.isfinite(_array(sample['action'])[_array(sample['action_mask']).astype(bool)]).all():
        raise ValueError('nonfinite valid action target')


class NativeFixedCases:
    """Frozen scalar descriptors; bounded search/cache, one decoded sample at a time."""
    def __init__(self, dataset, *, run_dir, seed=42, max_cases=64, diagnostic_subset=False, max_search_episodes=16):
        self.dataset=dataset; self.run_dir=Path(run_dir); self.seed=int(seed)
        if max_cases<1 or max_search_episodes<1: raise ValueError('positive fixed case/search budget required')
        self.identity=dict(normalization=dataset.normalization_identity()['registry_sha256'],
            selection=dataset.source_yaml_sha256, seed=self.seed, max_cases=int(max_cases),
            diagnostic_subset=bool(diagnostic_subset), max_search_episodes=int(max_search_episodes))
        path=self.run_dir/'fixed_cases.json'
        if path.exists():
            saved=json.loads(path.read_text())
            if saved.get('schema')!=SCHEMA or saved['identity']!=self.identity: raise ValueError('fixed-case evaluation contract changed')
            self.descriptors=saved['cases']; self.coverage=saved['coverage']; return
        families=sorted({repo['family'] for repo in dataset.repos}); descriptors=[]; coverage=[]
        for family in families:
            found={}; attempts=0
            for repo in [r for r in dataset.repos if r['family']==family]:
                for episode in sorted({e for e,_,_ in repo['ranges']}):
                    if attempts>=max_search_episodes: break
                    attempts+=1; metadata=dataset._episode(repo,episode)
                    ranges=[r for r in repo['ranges'] if r[0]==episode]
                    for level in ('task','subtask'):
                        if level in found or not dataset._options(metadata,ranges,level): continue
                        option,anchor,_=dataset._choose(metadata,ranges,np.random.default_rng(self.seed),level)
                        found[level]=dict(repo=repo['repo'],episode=episode,anchor=anchor,level=level,option=option,family=family)
                    if len(found)==2: break
                if len(found)==2 or attempts>=max_search_episodes: break
            if 'task' not in found: raise ValueError(f'No valid task case for {family} within {attempts} episode search budget')
            descriptors.extend(found[level] for level in ('task','subtask') if level in found)
            coverage.append(dict(family=family,task='available',subtask='available' if 'subtask' in found else 'unavailable within bounded search',searched_episodes=attempts))
        if len(descriptors)>max_cases and not diagnostic_subset:
            raise ValueError(f'max_cases={max_cases} cannot cover all {len(descriptors)} fixed cases; explicit diagnostic_subset required')
        self.descriptors=descriptors[:max_cases] if diagnostic_subset else descriptors
        self.coverage=dict(mode='diagnostic reduced coverage' if diagnostic_subset else 'all-family bounded coverage',
            families=coverage,selected_cases=len(self.descriptors),available_cases=len(descriptors))

    def __len__(self): return len(self.descriptors)

    def __getitem__(self,index):
        from openwam.dataloader.native_stage2_stats import lookup_stats
        from openwam.dataloader.stage2_diagnostics import report_goal_target
        descriptor=self.descriptors[index]
        repo=next(r for r in self.dataset.repos if r['repo']==descriptor['repo'])
        metadata=self.dataset._episode(repo,descriptor['episode'])
        sample=self.dataset._sample_selected(repo,descriptor['episode'],metadata,descriptor['option'],descriptor['anchor'],descriptor['level'])
        stats=lookup_stats(self.dataset.registry,repo,metadata)
        active=np.zeros(112,bool);active[metadata['representation'].scatter]=np.asarray(stats['action']['count'])>0
        if descriptor['level']=='task':
            sample['report_target']=dict(goal_source_frame=metadata['length']-1,label_kind='episode_end',success_verified=False)
        image,target=report_goal_target(self.dataset,repo,descriptor['episode'],metadata,sample)
        sample['report_target']=target
        sample['_eval']=dict(family=descriptor['family'], active_action_mask=active,action_stats=stats['action'],
            report_goal=image,projection=target.get('projection'))
        return sample

    def publish_descriptors(self):
        path=self.run_dir/'fixed_cases.json'
        value=dict(schema=SCHEMA,identity=self.identity,cases=self.descriptors,coverage=self.coverage)
        if path.exists():
            if json.loads(path.read_text())!=json.loads(_json(value)): raise ValueError('fixed descriptors are immutable')
        else: _write_json(path,value)


def build_fixed_cases(config, run_dir):
    from omegaconf import OmegaConf
    from openwam.dataloader.curated_stage2 import CuratedStage2Dataset
    if OmegaConf.is_config(config): config=OmegaConf.to_container(config,resolve=True)
    dl=dict(config['dataloader']); dl['observation_augmentation']={'enabled':False}
    dl['seed']=int(config['project']['seed'])
    dataset=CuratedStage2Dataset.from_config(dl)
    evaluation=config.get('evaluation',{})
    return NativeFixedCases(dataset,run_dir=run_dir,seed=config['project']['seed'],
        max_cases=int(evaluation.get('max_cases',64)),diagnostic_subset=bool(evaluation.get('diagnostic_subset',False)),
        max_search_episodes=int(evaluation.get('max_search_episodes',16)))


def _metric(values, *, sample_count=1):
    values=np.asarray(values,dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all(): raise ValueError('nonfinite valid metric values')
    return dict(value=float(values.mean()) if len(values) else None, sum=float(values.sum()),
        valid_count=len(values),sample_count=int(sample_count if len(values) else 0),status='available' if len(values) else 'unavailable')


def action_metrics(predicted, sample):
    target=_array(sample['action']);mask=_array(sample['action_mask']).astype(bool)
    metrics={'action_mae':_metric(np.abs(predicted[mask]-target[mask]))}
    raw=_array(sample['trace']['raw_action']);n=min(32,len(raw))
    stats=sample['_eval'].get('action_stats');physical=None
    if stats is not None:
        q01=np.asarray(stats['q01']); span=np.maximum(np.asarray(stats['q99'])-q01,1e-6)
        physical=q01+(predicted[:n,SCATTER]+1)*span/2
        physical[:,[9,19]]=np.clip(predicted[:n,SCATTER][:,[9,19]],0,1)
        raw_mask=mask[:n,SCATTER] & (np.asarray(stats['count'])>0)[None,:]
        if 'raw_action_mask' in sample['trace']: raw_mask &= _array(sample['trace']['raw_action_mask'])[:n].astype(bool)
        for name,slots in [('joint_radian_mae',list(range(20,34))),('gripper_openness_mae',[9,19])]:
            valid=raw_mask[:,slots];metrics[name]=_metric(np.abs(physical[:,slots]-raw[:n,slots])[valid])
        clipping=raw_mask & ((raw[:n]<q01) | (raw[:n]>np.asarray(stats['q99'])))
        clipping[:,[9,19]]=False
        metrics['normalization_clipped_fraction']=_metric(clipping[raw_mask].astype(float))
    else:
        for name in ('joint_radian_mae','gripper_openness_mae','normalization_clipped_fraction'):metrics[name]=_metric([])
    positions=[];rotations=[]
    from openwam.model.fk_supervision import torch_fk
    for arm in (sample.get('fk_supervision') or {}).get('arms',[]):
        valid=np.asarray(arm['valid'],bool)&mask[:,arm['slots']].all(-1)
        if not valid.any(): continue
        q0=torch.as_tensor(arm['q_anchor'],dtype=torch.float64)
        delta=torch.as_tensor(predicted[:,arm['slots']][valid],dtype=torch.float64)*torch.as_tensor(arm['scale'])+torch.as_tensor(arm['offset'])
        t0=torch_fk(q0,arm['chain']);tf=torch_fk(q0+delta,arm['chain']);r0=t0[:3,:3].T
        p=(r0@(tf[:,:3,3]-t0[:3,3])[...,None])[...,0];r=r0@tf[:,:3,:3]
        gt=torch.as_tensor(np.asarray(arm['gt_relative'])[valid],dtype=torch.float64)
        positions.extend(torch.linalg.vector_norm(p-gt[:,:3,3],dim=-1).tolist())
        relative=r.transpose(-1,-2)@gt[:,:3,:3]
        angles=torch.acos(((relative.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1))
        rotations.extend(angles.tolist())
    metrics['fk_position_m']=_metric(positions);metrics['fk_rotation_rad']=_metric(rotations)
    return metrics


def _pil_goal(decoded):
    image=_array(decoded)
    if image.shape!=(1,3,1,256,320) or not np.isfinite(image).all():raise ValueError('decoded goal must be finite B3x1x256x320')
    return Image.fromarray(np.clip((image[0,:,0].transpose(1,2,0)+1)*127.5,0,255).astype(np.uint8))


def prediction_metrics(result, diagnostics, goal_image, sample):
    predicted=np.asarray(result['normalized_actions'])
    if predicted.shape!=(32,112) or not np.isfinite(predicted).all():raise ValueError('invalid normalized_actions generation output')
    video=result['video']
    if len(video)!=9 or any(not isinstance(i,Image.Image) or i.size!=(320,384) for i in video):
        raise ValueError('prediction must contain exactly 9 PIL 320x384 canvases')
    metrics=action_metrics(predicted,sample)
    view_mask=_array(sample['video_view_mask']).astype(bool)&_array(sample['video_mask']).astype(bool)[:,None]
    total=0.;count=0
    for t in range(1,9):
        gt=np.asarray(sample['video'][t],dtype=np.float32);pred=np.asarray(video[t],dtype=np.float32)
        for view,(x0,y0,x1,y1) in enumerate(TILES):
            if view_mask[t,view]:
                error=np.abs(pred[y0:y1,x0:x1]-gt[y0:y1,x0:x1]);total+=float(error.sum());count+=error.size
    metrics['future_video_mae']=dict(value=total/count if count else None,sum=total,valid_count=count,
        sample_count=int(count>0),status='available' if count else 'unavailable')
    valid_goal=bool(_array(diagnostics['image_token_valid']).any()) and diagnostics.get('goal_decode_available', True)
    target=sample['_eval'].get('report_goal')
    metrics['goal_image_mae']=_metric(np.abs(np.asarray(goal_image,dtype=float)-np.asarray(target.convert('RGB').resize((320,256),Image.Resampling.BILINEAR),dtype=float)) if target is not None and goal_image is not None and valid_goal else [])
    projection=sample['_eval'].get('projection');uv=_array(diagnostics['predicted_uv'])[0]
    if projection is not None and valid_goal:
        valid=np.asarray(projection['projection_valid'],bool)&np.asarray(projection['in_frame'],bool)
        expected=np.asarray(projection['uv'])*[319,255]
        errors=np.linalg.norm(uv*[319,255]-expected,axis=-1)[valid]
    else:errors=[]
    metrics['tcp_pixel_error']=_metric(errors)
    return metrics


def _save_video(path, frames):
    import av
    with av.open(str(path),'w') as container:
        stream=container.add_stream('libx264',rate=8);stream.width=640;stream.height=384;stream.pix_fmt='yuv420p'
        for frame in frames:
            for packet in stream.encode(av.VideoFrame.from_image(frame)):container.mux(packet)
        for packet in stream.encode():container.mux(packet)


def _asset_hashes(folder):
    return {p.relative_to(folder).as_posix():_sha(p) for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='manifest.json'}


def _fixed_assets(sample, folder, *, observation_mode='head', actual_goal_inputs=None):
    """Input/GT snapshot is written once; later steps must have identical source bytes."""
    folder.mkdir(parents=True,exist_ok=True)
    head=sample.get('goal_observed_head')
    images={'observation.png':sample['first_frame_image'][0]}
    observed=None; valid=None
    if observation_mode=='multiview':
        observed,valid=_clean_multiview_condition(sample)
        if actual_goal_inputs is not None:
            observed=actual_goal_inputs['goal_observed_multiview'][0]
            valid=actual_goal_inputs['goal_view_valid'][0]
        pixels=_array(observed)[:,0].transpose(1,2,0)
        images['goal_input.png']=Image.fromarray(np.clip((pixels+1)*127.5,0,255).astype(np.uint8))
    elif head is not None:images['goal_input.png']=_pil_goal(head.unsqueeze(0))
    if sample['_eval'].get('report_goal') is not None: images['goal_gt.png']=sample['_eval']['report_goal']
    identity=hashlib.sha256()
    for name,image in sorted(images.items()):identity.update(name.encode()+image.tobytes())
    for image in sample['video']:identity.update(image.tobytes())
    for key in ('action','action_mask','video_mask','video_view_mask','proprio','proprio_mask'):identity.update(_array(sample[key]).tobytes())
    trace_identity={k:v for k,v in sample['trace'].items() if k!='stats_registry'}
    identity.update(_json([trace_identity,sample['report_target'],sample['prompt'],sample['goal_seed']]).encode())
    if observation_mode=='multiview':
        identity.update(b'multiview'+str(observed.dtype).encode()+_array(observed).tobytes()+_array(valid).tobytes())
    from openwam.model.goal.history import history_kwargs
    fixed_history=history_kwargs(sample if actual_goal_inputs is None else actual_goal_inputs)
    for key, value in fixed_history.items():
        identity.update(_json([key,list(value.shape),str(value.dtype)]).encode()+_array(value).tobytes())
    signature=identity.hexdigest();manifest=folder/'manifest.json'
    if manifest.exists():
        saved=json.loads(manifest.read_text())
        if saved['identity']!=signature:raise ValueError('fixed input/GT assets changed between evaluations')
        for name,digest in saved['assets'].items():
            if _sha(folder/name)!=digest:raise ValueError('immutable fixed asset corrupted')
        return
    if list(folder.iterdir()):raise ValueError('incomplete fixed assets; use a fresh report run directory')
    for name,image in images.items():image.save(folder/name)
    for i,image in enumerate(sample['video']):image.save(folder/f'video_gt_{i:02d}.png')
    tensors={key:_array(sample[key]) for key in ('action','action_mask','video_mask','video_view_mask','proprio','proprio_mask')}
    if observation_mode=='multiview':
        tensors['goal_observed_multiview']=_array(observed)
        if observed.dtype==torch.bfloat16:
            tensors['goal_observed_multiview_bfloat16_bits']=observed.detach().cpu().contiguous().view(torch.uint16).numpy()
        tensors['goal_view_valid']=_array(valid)
        from openwam.model.goal.observation import observation_token_mask
        tensors['goal_observation_token_valid']=(_array(observation_token_mask(valid[None])[0]) if valid[0] else np.zeros(120,bool))
    elif head is not None:tensors['goal_observed_head']=_array(head)
    tensors.update({key:_array(value) for key,value in fixed_history.items()})
    tensors['active_action_mask']=_array(sample['_eval']['active_action_mask'])
    np.savez_compressed(folder/'supervision.npz',**tensors)
    trace=dict(trace=sample['trace'],report_target=sample['report_target'],prompt=sample['prompt'],goal_seed=sample['goal_seed'])
    if observation_mode=='multiview':trace.update(goal_observation_mode=observation_mode,goal_view_valid=_array(valid).tolist(),goal_input_shape=list(observed.shape),goal_input_dtype=str(observed.dtype))
    _write_json(folder/'trace.json',trace)
    _write_json(manifest,dict(schema=SCHEMA,identity=signature,assets=_asset_hashes(folder)))


def _curves(folder, predicted, sample):
    # Self-contained SVG panels avoid a heavyweight plotting dependency on workers.
    gt=_array(sample['action']);mask=_array(sample['action_mask']).astype(bool)
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1920" height="2520" viewBox="0 0 1920 2520"><rect width="100%" height="100%" fill="white"/>']
    for channel in range(112):
        ox=(channel%8)*240;oy=(channel//8)*180
        values=np.r_[predicted[:,channel],gt[:,channel][mask[:,channel]]]
        lo=float(values.min());hi=float(values.max());span=max(hi-lo, .1)
        svg.append(f'<g transform="translate({ox},{oy})"><text x="25" y="15" font-size="12">channel {channel} [{lo:.3g}, {hi:.3g}]</text><path d="M25 25V150H230" fill="none" stroke="#bbb"/>')
        def point(t,v):return f'{25+t*205/31:.2f},{145-(v-lo)*110/span:.2f}'
        svg.append('<polyline fill="none" stroke="#dd3344" stroke-width="1" points="'+' '.join(point(t,v) for t,v in enumerate(predicted[:,channel]))+'"/>')
        indices=np.flatnonzero(mask[:,channel])
        for group in np.split(indices,np.flatnonzero(np.diff(indices)>1)+1):
            if not len(group):continue
            svg.append('<polyline fill="none" stroke="#116677" stroke-width="2" points="'+' '.join(point(t,gt[t,channel]) for t in group)+'"/>')
            for t in group:
                x,y=point(t,gt[t,channel]).split(',');svg.append(f'<circle cx="{x}" cy="{y}" r="1.5" fill="#116677"/>')
        svg.append('<text x="25" y="165" font-size="10">0</text><text x="215" y="165" font-size="10">31</text></g>')
    svg.append('</svg>');(folder/'action_curves.svg').write_text(''.join(svg))
    Image.fromarray((mask.T*255).astype(np.uint8)).resize((640,448),Image.Resampling.NEAREST).save(folder/'action_mask.png')
    views=_array(sample['video_view_mask']).astype(bool)&_array(sample['video_mask']).astype(bool)[:,None]
    Image.fromarray((views.T*255).astype(np.uint8)).resize((576,96),Image.Resampling.NEAREST).save(folder/'video_mask.png')


def _case_assets(folder, result, diagnostics, goal_image, sample):
    folder.mkdir()
    uv=_array(diagnostics['predicted_uv'])[0]
    if goal_image is not None:
        goal_image.save(folder/'goal_prediction.png')
        overlay=goal_image.copy();draw=ImageDraw.Draw(overlay)
        for point,color in zip(uv,('red','blue')):
            if np.isfinite(point).all():
                x,y=point*[319,255];draw.ellipse((x-4,y-4,x+4,y+4),outline=color,width=2)
        overlay.save(folder/'goal_uv.png')
    projection=sample['_eval'].get('projection')
    _write_json(folder/'goal_points.json',dict(predicted_uv=[p.tolist() if np.isfinite(p).all() else None for p in uv],
        image_token_valid=_array(diagnostics['image_token_valid']),target=projection,
        goal_decode_available=diagnostics.get('goal_decode_available',True),
        representation=diagnostics.get('representation','patch_denoise')))
    side=[]
    for gt,pred in zip(sample['video'],result['video']):
        canvas=Image.new('RGB',(640,384));canvas.paste(gt,(0,0));canvas.paste(pred,(320,0));side.append(canvas)
    _save_video(folder/'future_gt_prediction.mp4',side)
    side[1].save(folder/'future_preview.png')
    np.savez_compressed(folder/'predictions.npz',normalized_actions=result['normalized_actions'],predicted_uv=_array(diagnostics['predicted_uv']),predicted_latent=_array(diagnostics['predicted_latent']))
    _curves(folder,np.asarray(result['normalized_actions']),sample)


def _metric_table(metrics):
    return '<table><tr><th>metric</th><th>value</th><th>valid count</th><th>samples</th></tr>'+''.join(
        f'<tr><td>{html.escape(name)}</td><td>{"unavailable" if m["value"] is None else format(m["value"],".6g")}</td><td>{m["valid_count"]}</td><td>{m["sample_count"]}</td></tr>' for name,m in metrics.items())+'</table>'


def _page(body):
    return '<!doctype html><meta charset="utf-8"><title>Stage2 fixed evaluation</title><style>body{font:15px system-ui;max-width:1400px;margin:30px auto;background:#f8fafc;color:#142030}img,video{max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #bbc;padding:5px}section{background:white;padding:20px;margin:20px 0}.inputs{display:flex;gap:10px;align-items:start}.inputs figure{margin:0;width:24%}.inputs img{width:100%}</style>'+body


def _step_html(folder, report):
    body=f'<h1>Stage2 update {report["step"]}</h1><p>{html.escape(_json(report["coverage"]))}</p>'
    body+='<p>Predicted frozen Stage1 goals stay constant for fixed input/seed. Future video and actions measure Stage2 learning. Task report endpoints are true episode ends; subtask report endpoints are instruction interval ends. Success is not verified. Video: GT left, prediction right. MAE pixels use 0–255; joints rad, gripper openness, FK m/rad. Counts exclude invalid labels. All ranks generate identical cases; counts come from rank zero once.</p>'
    if report.get('goal_representation') == 'query_latent':
        body+='<p>Query latent mode: Stage2 loads only the frozen encoder. Goal image/TCP auxiliary decoders are absent; their predictions and errors are unavailable here. Stage1 evaluation retains those diagnostics. Latent tokens, action and future video predictions are saved.</p>'
    body+=_metric_table(report['aggregate'])
    for group,metrics in report['by_family_granularity'].items():
        body+='<details><summary>'+html.escape(group)+'</summary>'+_metric_table(metrics)+'</details>'
    for case in report['cases']:
        name=case['name'];fixed=f'../fixed_inputs/{name}'
        body+=f'<section><h2>{html.escape(case["family"])} / {html.escape(case["granularity"])}</h2><p>{html.escape(case["prompt"])}</p><div class="inputs">'
        for file,label in [(f'{fixed}/goal_input.png','Actual Stage1 clean '+report.get('goal_observation_mode','head')),(f'{fixed}/observation.png','Stage2 observed canvas'),(f'{fixed}/goal_gt.png','GT instruction endpoint'),(f'{name}/goal_uv.png','Predicted goal + UV')]:
            if (folder/file).exists():body+=f'<figure><figcaption>{label}</figcaption><img src="{file}"></figure>'
            else:body+=f'<figure>{label}: unavailable</figure>'
        body+='</div><p>Endpoint: '+html.escape(str(case['report_target'].get('goal_source_frame')))+'; '+html.escape(case['goal_visibility'])+'</p>'+_metric_table(case['metrics'])
        body+=f'<video controls preload="none" poster="{name}/future_preview.png" src="{name}/future_gt_prediction.mp4"></video><p>GT / prediction video includes observed index0 for context; future MAE excludes it.</p><p>Action curves: GT teal, prediction red; missing GT is omitted. Channel numbers refer to unified112.</p><img loading="lazy" src="{name}/action_curves.svg"><p>Action channel × time validity (white valid)</p><img src="{name}/action_mask.png"><p>Video head / left wrist / right wrist × time validity</p><img src="{name}/video_mask.png">'
        body+=f'<p><a href="{fixed}/trace.json">GT, geometry and source trace</a> · <a href="{name}/goal_points.json">Continuous goal UV and validity</a> · <a href="{name}/predictions.npz">Predicted points, latent and actions</a> · <a href="{fixed}/supervision.npz">All supervision/masks</a></p></section>'
    (folder/'index.html').write_text(_page(body),encoding='utf-8')


def _aggregate(cases):
    result={}
    for case in cases:
        for name,m in case['metrics'].items():
            a=result.setdefault(name,dict(sum=0.,valid_count=0,sample_count=0))
            for key in a:a[key]+=m[key]
    for m in result.values():
        m['value']=m['sum']/m['valid_count'] if m['valid_count'] else None
        m['status']='available' if m['valid_count'] else 'unavailable'
    return result


def render_stage2_history(run_dir):
    """Read completed metrics/assets only; no models, source readers or generation."""
    root=Path(run_dir);steps=[]
    for folder in sorted(root.glob('eval_step_*')):
        marker=folder/'manifest.json'
        if not marker.is_file(): continue
        manifest=json.loads(marker.read_text())
        if manifest.get('schema')!=SCHEMA or manifest.get('complete') is not True:continue
        for name,digest in manifest['assets'].items():
            asset=folder/name
            if asset.resolve().is_relative_to(folder.resolve()) is False or not asset.is_file() or _sha(asset)!=digest:
                raise ValueError(f'invalid immutable evaluation asset: {asset}')
        report=json.loads((folder/'metrics.json').read_text())
        _validate_fixed_inputs(root,report)
        steps.append((folder,report))
    body='<h1>Stage2 immutable prediction history</h1><p>Each update links its original inputs, predictions, video, actions, validity counts and errors. Reports never rerun generation.</p>'
    for folder,report in sorted(steps,key=lambda item:item[1]['step']):
        body+=f'<section><h2><a href="{folder.name}/index.html">Update {report["step"]}</a></h2><p>{html.escape(_json(report["coverage"]))}</p>'+_metric_table(report['aggregate'])+'</section>'
    path=root/'stage2_history.html'
    with tempfile.NamedTemporaryFile('w',dir=root,delete=False,encoding='utf-8') as stream:
        stream.write(_page(body));temporary=stream.name
    os.replace(temporary,path)
    return path


def evaluate_stage2(architecture,cases,output_dir,*,step,seed,num_steps=20):
    """Synchronous all-rank generation, atomic rank-zero publication after all assets.

    A published step is immutable; callers may skip an already complete step
    after verifying its fixed-case contract (trainer resume hook does so).
    """
    root=Path(output_dir);final=root/f'eval_step_{int(step):06d}';rank=_rank()
    if not _distributed() and final.exists():raise FileExistsError(final)
    observation_mode=getattr(architecture.goal_expert,'observation_mode','head')
    error=None
    try:
        if int(step)<0 or int(num_steps)<1 or len(cases)<1:raise ValueError('nonnegative step, positive denoise steps and nonempty cases required')
        if final.exists():raise FileExistsError(final)
    except Exception as exc:error=f'{type(exc).__name__}: {exc}'
    _agree_error(error,'setup')
    descriptors=getattr(cases,'descriptors',None)
    _same(_json(dict(count=len(cases),descriptors=descriptors,step=int(step),seed=int(seed),num_steps=int(num_steps),goal_observation_mode=observation_mode)))
    work=root/f'.eval_step_{int(step):06d}-{uuid.uuid4().hex}' if rank==0 else None
    rows=[];error=None
    if rank==0:
        try:
            root.mkdir(parents=True,exist_ok=True);work.mkdir()
            if hasattr(cases,'publish_descriptors'):cases.publish_descriptors()
        except Exception as exc:error=f'{type(exc).__name__}: {exc}'
    _agree_error(error,'publication setup')
    with preserve_evaluation_state(architecture):
        for index in range(len(cases)):
            sample=None;kwargs=None;error=None
            try:
                sample=cases[index];_validate_sample(sample,observation_mode=observation_mode)
                kwargs=generation_conditions(sample,seed=int(seed)+index,num_steps=num_steps,observation_mode=observation_mode,
                    observation_dtype=getattr(architecture,'dtype',None))
            except Exception as exc:error=f'{type(exc).__name__}: {exc}'
            _agree_error(error,f'case {index} validation')
            condition_sha=_conditions_sha(sample,kwargs);_same(condition_sha)
            random.seed(int(seed)+index);np.random.seed((int(seed)+index)%(2**32))
            torch.random.set_rng_state(torch.Generator(device='cpu').manual_seed(int(seed)+index).get_state())
            if torch.cuda.is_initialized():torch.cuda.manual_seed_all(int(seed)+index)
            # make_schedule mutates training scheduler buffers; protected by outer finally.
            error=None
            try:
                shift=float(getattr(architecture.action_backbone,'shift_action',5.))
                schedule=make_schedule('sync',architecture.video_scheduler,architecture.action_scheduler,
                    num_steps=int(num_steps),shift=shift,shift_video=getattr(architecture.video_backbone,'shift_video',None))
            except Exception as exc:error=f'{type(exc).__name__}: {exc}'
            _agree_error(error,'schedule construction')
            result=None;error=None
            try:result=architecture.generate(schedule,sample['prompt'],shift=shift,**kwargs)
            except Exception as exc:error=f'{type(exc).__name__}: {exc}'
            _agree_error(error,f'case {index} generation')
            error=None
            if rank==0:
                try:
                    diagnostics=architecture.goal_diagnostics
                    goal_image=(_pil_goal(architecture.goal_expert.codec.decode(diagnostics['predicted_latent']))
                                if diagnostics.get('goal_decode_available',True) else None)
                    metrics=prediction_metrics(result,diagnostics,goal_image,sample)
                    name=f'case_{index:03d}';_fixed_assets(sample,root/'fixed_inputs'/name,observation_mode=observation_mode,actual_goal_inputs=kwargs)
                    _case_assets(work/name,result,diagnostics,goal_image,sample)
                    rows.append(dict(name=name,family=sample['_eval']['family'],granularity=sample['granularity'],
                        prompt=sample['prompt'],goal_seed=int(sample['goal_seed']),generation_seed=int(seed)+index,
                        goal_observation_mode=observation_mode,
                        goal_view_valid=(_array(kwargs['goal_view_valid'])[0].tolist() if observation_mode=='multiview' else None),
                        goal_input_dtype=(str(kwargs['goal_observed_multiview'].dtype) if observation_mode=='multiview' else None),
                        goal_input_shape=(list(kwargs['goal_observed_multiview'].shape) if observation_mode=='multiview' else [1,3,1,256,320]),
                        condition_sha256=condition_sha,report_target=sample['report_target'],
                        valid_supervision=dict(action_elements=int(_array(sample['action_mask']).sum()),
                            future_frames=int(_array(sample['video_mask'])[1:].sum()),future_views=int((_array(sample['video_view_mask'])[1:]&_array(sample['video_mask'])[1:,None]).sum())),
                        goal_visibility=('available: at least one projected TCP in frame' if sample['_eval'].get('projection') is not None and np.asarray(sample['_eval']['projection']['in_frame']).any() else sample['report_target'].get('diagnostic_unavailable','unavailable: no projected TCP in frame')),metrics=metrics))
                except Exception as exc:error=f'{type(exc).__name__}: {exc}'
            _agree_error(error,f'case {index} rendering')
            del result,sample
        error=None
        if rank==0:
            try:
                coverage=getattr(cases,'coverage',dict(mode='explicit supplied cases',selected_cases=len(cases)))
                report=dict(schema=SCHEMA,step=int(step),seed=int(seed),num_steps=int(num_steps),coverage=coverage,sample_count=len(rows),
                    goal_representation=getattr(architecture.goal_expert,'representation','patch_denoise'),
                    goal_observation_mode=observation_mode,metric_rank_policy='rank0 authoritative; same cases generated on all ranks',cases=rows,aggregate=_aggregate(rows))
                report['by_family_granularity']={f'{family}/{level}':_aggregate([r for r in rows if r['family']==family and r['granularity']==level])
                    for family,level in sorted({(r['family'],r['granularity']) for r in rows})}
                _write_json(work/'metrics.json',report);_step_html(work,report)
                _write_json(work/'manifest.json',dict(schema=SCHEMA,complete=True,step=int(step),assets=_asset_hashes(work)))
                # Final directory appears with completion marker and all assets in one rename.
                os.rename(work,final)
                render_stage2_history(root)
            except Exception as exc:error=f'{type(exc).__name__}: {exc}'
        _agree_error(error,'final publication')
    if rank==0:LOG.info('report_ready global_step=%d html=%s',step,final/'index.html')
    return final/'index.html'


def completed_evaluation(run_dir, cases, *, step, seed, num_steps, observation_mode='head'):
    """Resume may reuse only a verified, matching published step, never overwrite it."""
    root=Path(run_dir);folder=root/f'eval_step_{int(step):06d}'
    if not folder.exists():return False
    marker=folder/'manifest.json'
    if not marker.is_file():raise ValueError('incomplete evaluation directory cannot be reused')
    manifest=json.loads(marker.read_text())
    if manifest.get('schema')!=SCHEMA or manifest.get('complete') is not True or manifest.get('step')!=int(step):
        raise ValueError('invalid evaluation completion manifest')
    for name,digest in manifest['assets'].items():
        path=folder/name
        if not path.resolve().is_relative_to(folder.resolve()) or not path.is_file() or _sha(path)!=digest:
            raise ValueError('corrupt published evaluation asset')
    report=json.loads((folder/'metrics.json').read_text())
    if report.get('goal_observation_mode','head')!=observation_mode:
        raise ValueError('published evaluation goal observation mode mismatch')
    expected=dict(step=int(step),seed=int(seed),num_steps=int(num_steps))
    if any(report.get(k)!=v for k,v in expected.items()) or len(report['cases'])!=len(cases):
        raise ValueError('published evaluation step/seed/case budget mismatch')
    if isinstance(cases,NativeFixedCases):
        saved=json.loads((root/'fixed_cases.json').read_text())
        if saved.get('identity')!=cases.identity or saved.get('cases')!=json.loads(_json(cases.descriptors)):
            raise ValueError('fixed descriptor contract changed')
    _validate_fixed_inputs(root,report)
    return True


def _validate_fixed_inputs(root,report):
    for row in report['cases']:
        name=row['name']
        if Path(name).name!=name:raise ValueError('invalid fixed case name')
        fixed=root/'fixed_inputs'/name
        marker=fixed/'manifest.json'
        if not marker.is_file():raise ValueError('missing immutable fixed input manifest')
        fixed_manifest=json.loads(marker.read_text())
        for name,digest in fixed_manifest['assets'].items():
            path=fixed/name
            if not path.resolve().is_relative_to(fixed.resolve()) or not path.is_file() or _sha(path)!=digest:
                raise ValueError('corrupt immutable evaluation input (fixed asset)')
