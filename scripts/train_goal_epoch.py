#!/usr/bin/env python3
"""Build an exact configured-FPS population or train it once with torchrun."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
from datetime import timedelta
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def export_run_artifacts(output,config,dependencies,*,identity,context_cache=None):
    """Native reload inputs with hashes; preserve contexts across complete resumes.

    Text contexts are the bounded primary rank cache, including fixed diagnostic
    cases. A multi-rank run does not claim this is every worker's text population.
    """
    import torch
    from omegaconf import OmegaConf
    from openwam.train.goal_epoch import atomic_json,sha256,_cpu
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    manifest=output/'artifacts.json'
    if manifest.exists() and json.loads(manifest.read_text())['identity']!=identity:
        raise ValueError('reload artifact identity mismatch')
    temporary=output/'resolved_config.yaml.pending'
    OmegaConf.save(OmegaConf.create(config),temporary);os.replace(temporary,output/'resolved_config.yaml')
    atomic_json(output/'dependencies.json',dependencies)
    target=output/'text_contexts.pt'
    contexts=torch.load(target,map_location='cpu',weights_only=True) if target.exists() else {}
    if context_cache is not None:
        contexts.update(_cpu(dict(context_cache.entries)))
        temporary=output/'text_contexts.pt.pending';torch.save(contexts,temporary);os.replace(temporary,target)
    names=['resolved_config.yaml','dependencies.json']+(['text_contexts.pt'] if target.exists() else [])
    atomic_json(manifest,dict(identity=identity,representation=config.get('model',{}).get('goal_representation'),
        files={name:sha256(output/name) for name in names},text_prompts=list(contexts),
        text_context_scope='bounded primary-rank cache including fixed training diagnostics',
        rank=int(os.environ.get('RANK',0)),world_size=int(os.environ.get('WORLD_SIZE',1))))


def validated_population(dataset,index_path,*,sample_fps,device=None):
    """Scan global source dependencies once, then verify index identity per rank."""
    import torch.distributed as dist
    from openwam.dataloader.goal_epoch import FiniteGoalDataset
    if not dist.is_initialized() or dist.get_world_size()==1:
        finite=FiniteGoalDataset(dataset,index_path,sample_fps=sample_fps)
        if not len(finite):raise ValueError('empty population')
        return finite
    rank=dist.get_rank();world=dist.get_world_size();finite=None;message=[None]
    if rank==0:
        try:
            finite=FiniteGoalDataset(dataset,index_path,sample_fps=sample_fps)
            if not len(finite):raise ValueError('empty population')
            message[0]=dict(summary=finite.summary,error=None)
        except Exception as exc:
            message[0]=dict(summary=None,error=f'{type(exc).__name__}: {exc}')
    dist.broadcast_object_list(message,src=0,device=device)
    if message[0]['error']:raise RuntimeError('population validation failed: '+message[0]['error'])
    error=None
    if rank!=0:
        try:
            finite=FiniteGoalDataset(dataset,index_path,sample_fps=sample_fps,
                _validated_summary=message[0]['summary'])
        except Exception as exc:error=f'{type(exc).__name__}: {exc}'
    # A follower's local index/hash mismatch must also stop every rank before
    # model bootstrap enters its collectives; otherwise peers wait indefinitely.
    errors=[None]*world
    dist.all_gather_object(errors,error)
    for failed_rank,error in enumerate(errors):
        if error:raise RuntimeError(f'population validation failed on rank {failed_rank}: {error}')
    return finite


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare',action='store_true')
    modes.add_argument('--train',action='store_true')
    parser.add_argument('--config',required=True)
    parser.add_argument('--index',required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--source-yaml')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--max-updates',type=int,help='bounded optimizer updates; saved identity includes this budget')
    parser.add_argument('--repeat-population',action='store_true',help='repeat the finite population for bounded overfit')
    parser.add_argument('--index-workers',type=int,default=32,help='bounded CPU metadata reader threads during --prepare (default: 32)')
    parser.add_argument('--index-max-pending',type=int,default=64,help='maximum ordered pending episode results (default: 64)')
    parser.add_argument('--index-commit-every',type=int,default=256,help='episodes per durable SQLite transaction (default: 256)')
    parser.add_argument('--index-progress',help='atomic progress JSON path (default: INDEX.progress.json)')
    args=parser.parse_args(argv)
    import torch
    import torch.distributed as dist
    from openwam.train.goal_curated import load_config,TextContextCache
    from openwam.dataloader.curated_goal import CuratedGoalDataset
    from openwam.dataloader.goal_epoch import build_epoch_index,epoch_sample_fps
    from openwam.train.goal_epoch import atomic_json,train_epoch,sha256
    torch.set_num_threads(2)
    config=load_config(args.config,source_yaml=args.source_yaml)
    if args.max_updates is not None:config['max_updates']=args.max_updates
    if args.repeat_population:config['repeat_population']=True
    sample_fps=epoch_sample_fps(config)
    if config['model']['goal_representation'] not in ('query_latent','visual_history_masked_sa'):
        raise ValueError('finite launcher requires query_latent or visual_history_masked_sa')
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    if args.prepare and world!=1:raise ValueError('prepare the index in a single CPU process')
    output=Path(args.output_dir).resolve();output.mkdir(parents=True,exist_ok=True)
    ds=dict(config['dataset']);ds['cache_dir']=str(output/'cache'/f'rank_{rank}')
    seed=config.get('seed',42);dtype=getattr(torch,config.get('dtype','bfloat16'))
    dataset=CuratedGoalDataset(**ds,seed=seed,dtype=dtype)
    if args.prepare:
        preparation={key:value for key,value in dict(workers=args.index_workers,max_pending=args.index_max_pending,
            commit_every=args.index_commit_every,progress_path=args.index_progress).items() if value is not None}
        result=build_epoch_index(dataset,args.index,sample_fps=sample_fps,**preparation)
        atomic_json(output/'population.json',result);print(json.dumps(result,indent=2));return 0
    # Keep the single-process error path before CUDA allocation; distributed
    # validation needs an initialized group so failures can reach every rank.
    finite=validated_population(dataset,args.index,sample_fps=sample_fps) if world==1 else None
    device=torch.device('cuda',local)
    torch.cuda.set_device(device)
    if world>1:dist.init_process_group('nccl',timeout=timedelta(hours=4))
    try:
        if world>1:finite=validated_population(dataset,args.index,sample_fps=sample_fps,device=device)
        from openwam.train.stage12 import epoch_identity
        identity=epoch_identity(config,finite.summary,world)
        if rank==0:atomic_json(output/'identity.json',identity)
        from openwam.model.video_backbone.wan.encode import encode_text
        from openwam.train.bootstrap import run_bootstrap_in_waves,bootstrap_query_components
        local_world=int(os.environ.get('LOCAL_WORLD_SIZE',world))
        concurrency=config.get('bootstrap_workers_per_node',2)
        if type(concurrency) is int and concurrency>0:concurrency=min(concurrency,local_world)
        expert,text_encoder,tokenizer,dependencies=run_bootstrap_in_waves(
            lambda:bootstrap_query_components(config,device=device,dtype=dtype),
            local_rank=local,local_world_size=local_world,workers_per_node=concurrency)
        if config.get('codec_cache_max_bytes',0):
            expert.codec.enable_cache(max_bytes=config['codec_cache_max_bytes'])
        if rank==0:export_run_artifacts(output,config,dependencies,identity=identity)
        text_device=next(text_encoder.parameters()).device
        cache=TextContextCache(lambda prompt:encode_text([prompt],tokenizer=tokenizer,text_encoder=text_encoder,device=text_device),max_entries=config.get('text_cache_size',128))
        result=train_epoch(expert,finite,cache,output,identity=identity,dependencies=dependencies,
            learning_rate=config['learning_rate'],history_learning_rate=config.get('history_learning_rate',1e-4),
            condition_learning_rate=config.get('condition_learning_rate'),
            checkpoint_every=config.get('checkpoint_every',5000),eval_every=config.get('eval_every',500),
            sampling_steps=config.get('sampling_steps',20),resume=args.resume,
            bootstrap_workers_per_node=config.get('bootstrap_workers_per_node',2),
            optimizer_device=config.get('optimizer_device','model'),
            max_updates=config.get('max_updates'),repeat_population=config.get('repeat_population',False),
            sample_cache_size=config.get('sample_cache_size',0))
        if rank==0:
            export_run_artifacts(output,config,dependencies,identity=identity,context_cache=cache)
            print(json.dumps(result,indent=2))
    finally:
        if dist.is_initialized():dist.destroy_process_group()
    return 0


if __name__=='__main__':raise SystemExit(main())
