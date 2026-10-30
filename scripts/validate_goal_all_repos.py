#!/usr/bin/env python3
"""One real Stage1 forward per selected source directory (not family).

Preparation records every source, including errors. Training reuses production
decoding, VAE, text encoder, query/history encoder, DiT and losses.
This diagnostic never writes a production checkpoint or a full-epoch marker.
"""
from pathlib import Path
import argparse
from fractions import Fraction
import gc
import json
import os
import subprocess
import sys
import time
import traceback
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from copy import copy

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


class NoLegalPair(ValueError):
    pass


def choose_pair(meta,ranges,sample_fps):
    """Prefer a subtask; use the exact episode-zero grid and anchor validity."""
    import numpy as np
    ratio=max(Fraction(str(meta['info']['fps']))/Fraction(str(sample_fps)),Fraction(1))
    num,den=ratio.numerator,ratio.denominator
    valid=meta['source_valid']
    for level,key in [('subtask','subtasks'),('task','tasks')]:
        for segment in meta[key]:
            goal=segment['end_exclusive']-1
            for _,a,b in ranges:
                start=max(a,segment['start']);end=min(b,goal)
                if start>=end:continue
                lo=(start*den+num-1)//num;hi=(end*den+num-1)//num
                anchors=np.arange(lo,hi,dtype=np.int64)*num//den
                anchors=anchors[np.asarray(valid)[anchors]]
                if len(anchors):return level,segment,int(anchors[len(anchors)//2]),goal
    raise NoLegalPair('no valid sampled anchor with a future goal in selected episode')


def append(path,row):
    with Path(path).open('a') as f:
        f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');f.flush()


def select_replay_entries(entries,previous,current,repos,families,shard_index,shards):
    """Reuse audited sample selection, not the previous model's pass results."""
    for key in ('source_sha256','config','population_version','repositories','sample_fps'):
        if previous.get(key)!=current.get(key):raise ValueError('replay source/config identity changed: '+key)
    if [r['repo_index'] for r in entries]!=list(range(len(repos))):raise ValueError('incomplete replay source coverage')
    for row,repo in zip(entries,repos):
        if any(row[key]!=repo[key] for key in ('repo','path','family')):raise ValueError('replay source identity changed')
    if shards<1 or not 0<=shard_index<shards:raise ValueError('invalid replay shard')
    if not set(families)<=set(r['family'] for r in entries):raise ValueError('unknown replay family')
    chosen=[r for r in entries if not families or r['family'] in families]
    return chosen[shard_index::shards]


def prepare_one(data,index,repo,sample_fps):
    # Each worker owns its metadata LRU; no decoding, assets or CUDA in workers.
    data=copy(data);data._episodes=OrderedDict();data.max_cached_episodes=1
    row=dict(repo_index=index,repo=repo['repo'],path=repo['path'],family=repo['family'])
    try:
        empty=[]
        for ep in sorted({r[0] for r in repo['ranges']}):
            row['episode']=ep
            meta=data._episode(repo,ep)
            ranges=[r for r in repo['ranges'] if r[0]==ep]
            try:
                level,segment,anchor,goal=choose_pair(meta,ranges,sample_fps)
                break
            except NoLegalPair:empty.append(ep)
        else:raise NoLegalPair('no valid sampled anchor in any selected episode')
        row.update(status='prepared',level=level,segment=segment,anchor=anchor,goal=goal,
                   source_length=meta['length'],source_fps=meta['info']['fps'],no_pair_episodes=empty)
    except Exception as exc:
        row.update(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
    return row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--forward',action='store_true')
    p.add_argument('--config',required=True);p.add_argument('--source-yaml',required=True)
    p.add_argument('--output-dir',required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--replay-from',type=Path,help='reuse a verified full source-selection manifest for fresh forwards')
    p.add_argument('--families',nargs='*',default=[])
    p.add_argument('--shards',type=int,default=1);p.add_argument('--shard-index',type=int,default=0)
    args=p.parse_args()
    if args.prepare and (args.replay_from or args.families or args.shards!=1 or args.shard_index):
        p.error('replay/shard options require --forward')
    import torch
    from openwam.train.goal_curated import load_config,prepare_batch,TextContextCache
    from openwam.dataloader.curated_goal import CuratedGoalDataset
    from openwam.dataloader.goal_epoch import epoch_sample_fps,sha256_file,VERSION
    from openwam.train.goal_epoch import atomic_json
    torch.set_num_threads(2)
    if int(os.environ.get('WORLD_SIZE','1'))!=1:raise ValueError('single DSW process required')
    output=Path(args.output_dir).resolve();output.mkdir(parents=True,exist_ok=True)
    config=load_config(args.config,source_yaml=args.source_yaml)
    if config['model']['goal_representation']!='query_latent':raise ValueError('query latent config required')
    dtype=getattr(torch,config.get('dtype','bfloat16'));seed=config.get('seed',42)
    ds=dict(config['dataset']);ds['cache_dir']=str(output/'cache')
    data=CuratedGoalDataset(**ds,seed=seed,dtype=dtype)
    identity=dict(source_yaml=config['dataset']['source_yaml'],source_sha256=sha256_file(args.source_yaml),
                  config=config,git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  population_version=VERSION,repositories=len(data.repos),sample_fps=epoch_sample_fps(config))
    manifest=output/'samples.jsonl'
    if args.prepare:
        if manifest.exists():raise ValueError('choose a fresh output directory')
        atomic_json(output/'identity.json',identity)
        passed=failed=0;started=time.monotonic()
        with ThreadPoolExecutor(max_workers=8) as pool:
            rows=pool.map(lambda pair:prepare_one(data,*pair,identity['sample_fps']),enumerate(data.repos))
            for index,row in enumerate(rows):
                if row['status']=='prepared':passed+=1
                else:
                    append(output/'errors.jsonl',dict(phase='prepare',**row));failed+=1
                append(manifest,row)
                status=dict(phase='prepare',attempted=index+1,total=len(data.repos),passed=passed,failed=failed,
                            elapsed_s=time.monotonic()-started,updated_at=time.time())
                atomic_json(output/'progress.json',status)
                if index%25==0 or row['status']=='failed':print(json.dumps(status|{'repo':row['repo'],'error':row.get('error')}),flush=True)
        atomic_json(output/'prepare_summary.json',status)
        return int(failed>0)
    if args.replay_from:
        previous=json.loads((args.replay_from/'identity.json').read_text())
        entries=[json.loads(line) for line in (args.replay_from/'samples.jsonl').read_text().splitlines()]
        entries=select_replay_entries(entries,previous,identity,data.repos,args.families,args.shard_index,args.shards)
        if manifest.exists():raise ValueError('choose a fresh replay output directory')
        atomic_json(output/'identity.json',identity|{'replay':dict(source=str(args.replay_from.resolve()),
            previous_git_sha=previous.get('git_sha'),families=args.families,shards=args.shards,shard_index=args.shard_index)})
        for row in entries:append(manifest,row)
    else:
        if args.families or args.shards!=1 or args.shard_index:p.error('family/shard options require --replay-from')
        if json.loads((output/'identity.json').read_text())!=identity:
            raise ValueError('preparation identity changed; use a fresh output directory')
        entries=[json.loads(line) for line in manifest.read_text().splitlines()]
        if [r['repo_index'] for r in entries]!=list(range(len(data.repos))):raise ValueError('incomplete source coverage')
    if not entries:raise ValueError('empty forward selection')
    if (output/'results.jsonl').exists():raise ValueError('diagnostic already started; preserve existing results')
    from openwam.train.goal_trainer import bootstrap_components,_seed
    from openwam.model.goal.query_expert import QueryGoalExpert
    from openwam.model.video_backbone.wan.encode import encode_text
    from openwam.model.goal.history import history_kwargs
    device=torch.device(args.device);torch.cuda.set_device(device);_seed(seed)
    atomic_json(output/'progress.json',dict(phase='model_loading',total=len(entries),updated_at=time.time()))
    original,text_encoder,tokenizer=bootstrap_components(config,device='cpu',dtype=dtype)
    expert=QueryGoalExpert(original.dit,original.codec,dit_config=original.dit_config,
        query_config=config['model'].get('query_goal'),history_config=config['model'].get('query_history'),
        conditioning_mode=config['model'].get('query_conditioning','legacy_text'),
        observation_mode=config['observation_mode'],shift=config.get('shift',5.),
        lambda_image=config.get('lambda_image',.5),lambda_tcp=config.get('lambda_tcp',1.),
        use_gradient_checkpointing=config.get('gradient_checkpointing',True)).to(device=device,dtype=dtype)
    del original
    expert.codec.vae.to(device=device,dtype=dtype).eval().requires_grad_(False)
    text_device=config.get('text_device','cuda')
    text_encoder.to(device=text_device,dtype=dtype).eval().requires_grad_(False)
    cache=TextContextCache(lambda text:encode_text([text],tokenizer=tokenizer,text_encoder=text_encoder,device=text_device),
                           max_entries=config.get('text_cache_size',128))
    expert.eval().requires_grad_(False)
    gc.collect();passed=failed=0;started=time.monotonic()
    for ordinal,entry in enumerate(entries):
        index=entry['repo_index'];repo=data.repos[index];row=dict(entry)
        begin=time.monotonic();phase='metadata';batch=cpu=out=None
        torch.cuda.reset_peak_memory_stats(device)
        try:
            if entry['status']!='prepared':raise ValueError('preparation failed: '+entry['error'])
            _seed(seed+index);data._history_draw_id=index
            meta=data._episode(repo,entry['episode'])
            if args.replay_from:
                ranges=[r for r in repo['ranges'] if r[0]==entry['episode']]
                actual=choose_pair(meta,ranges,identity['sample_fps'])
                if actual!=(entry['level'],entry['segment'],entry['anchor'],entry['goal']):
                    raise ValueError('replay sample labels or anchor changed')
            phase='decode'
            record,cpu=data._materialize(repo,entry['episode'],meta,entry['segment'],entry['anchor'],entry['goal'],entry['level'])
            phase='encode';batch=prepare_batch(record,cpu,device=device,dtype=dtype,context_cache=cache)
            row.update(instruction=record['instruction'],point_valid=cpu['targets'].point_valid.tolist(),
                       history_valid=cpu.get('history_valid',torch.empty(0)).tolist(),
                       observed_shape=list(cpu['observed'].shape),target_shape=list(cpu['targets'].image.shape),
                       provenance=record['provenance'])
            phase='forward'
            with torch.no_grad():
                out=expert(batch['observed'],batch['context'],batch['targets'],context_lens=batch.get('context_lens'),
                           observed_view_valid=batch.get('observed_view_valid'),**history_kwargs(batch))
            if not torch.isfinite(out['loss']).item():raise FloatingPointError('nonfinite Stage1 loss')
            torch.cuda.synchronize(device)
            row.update(status='passed',forward_count=1,backward_count=0,
                losses={k:float(v.detach()) for k,v in out.items() if k.endswith('loss') and isinstance(v,torch.Tensor) and v.numel()==1})
            passed+=1
        except Exception as exc:
            row.update(status='failed',failure_phase=phase,error=f'{type(exc).__name__}: {exc}');failed+=1
            append(output/'errors.jsonl',dict(phase='forward',**row,traceback=traceback.format_exc()))
        finally:
            row.update(elapsed_s=time.monotonic()-begin,peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
                       peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30)
            append(output/'results.jsonl',row)
            del batch,cpu,out
            if row['status']=='failed':gc.collect();torch.cuda.empty_cache()
        status=dict(phase='forward',attempted=ordinal+1,total=len(entries),passed=passed,failed=failed,
                    elapsed_s=time.monotonic()-started,updated_at=time.time(),last_repo=repo['repo'],
                    last_loss=row.get('losses'),last_error=row.get('error'),last_step_s=row['elapsed_s'])
        atomic_json(output/'progress.json',status)
        print(json.dumps(status),flush=True)
    atomic_json(output/'summary.json',status|{'complete':True,'all_passed':passed==len(entries)})
    return int(failed>0)


if __name__=='__main__':raise SystemExit(main())
