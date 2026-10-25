#!/usr/bin/env python3
"""Build an exact configured-FPS population or train it once with torchrun."""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import os
import sys
from datetime import timedelta
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


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
    sample_fps=epoch_sample_fps(config)
    if config['model']['goal_representation']!='query_latent':raise ValueError('finite launcher requires query_latent')
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
        from openwam.train.goal_trainer import bootstrap_components,_seed
        from openwam.model.goal.query_expert import QueryGoalExpert
        from openwam.model.goal.dependencies import dependency_identity
        from openwam.model.video_backbone.wan.encode import encode_text
        expert=text_encoder=tokenizer=dependencies=None
        # Bound transient host RAM: bootstrap only one rank at a time.
        for loading_rank in range(world):
            error=None
            if rank==loading_rank:
                try:
                    _seed(seed)
                    original,text_encoder,tokenizer=bootstrap_components(config,device='cpu',dtype=dtype)
                    expert=QueryGoalExpert(original.dit,original.codec,dit_config=original.dit_config,
                        query_config=config['model'].get('query_goal'),history_config=config['model'].get('query_history'),
                        observation_mode=config['observation_mode'],shift=config.get('shift',5.),
                        lambda_image=config.get('lambda_image',.5),lambda_tcp=config.get('lambda_tcp',1.),
                        lambda_query_image=config.get('lambda_query_image',.1),lambda_query_tcp=config.get('lambda_query_tcp',.1),
                        use_gradient_checkpointing=config.get('gradient_checkpointing',True)).to(device=device,dtype=dtype)
                    del original
                    expert.codec.vae.to(device=device,dtype=dtype).eval().requires_grad_(False)
                    text_device=device if config.get('text_device','cuda')=='cuda' else torch.device(config['text_device'])
                    text_encoder.to(device=text_device,dtype=dtype).eval().requires_grad_(False)
                    dependencies=dependency_identity(expert.codec.vae,text_dim=expert.text_dim,text_encoder=text_encoder,tokenizer=tokenizer)
                    gc.collect()
                except Exception as exc:error=f'{type(exc).__name__}: {exc}'
            message=[error]
            if world>1:dist.broadcast_object_list(message,src=loading_rank,device=device)
            if message[0]:raise RuntimeError('bootstrap failed: '+message[0])
        text_device=next(text_encoder.parameters()).device
        cache=TextContextCache(lambda prompt:encode_text([prompt],tokenizer=tokenizer,text_encoder=text_encoder,device=text_device),max_entries=config.get('text_cache_size',128))
        result=train_epoch(expert,finite,cache,output,identity=identity,dependencies=dependencies,
            learning_rate=config['learning_rate'],history_learning_rate=config.get('history_learning_rate',1e-4),
            checkpoint_every=config.get('checkpoint_every',5000),eval_every=config.get('eval_every',500),
            sampling_steps=config.get('sampling_steps',20),resume=args.resume)
        if rank==0:print(json.dumps(result,indent=2))
    finally:
        if dist.is_initialized():dist.destroy_process_group()
    return 0


if __name__=='__main__':raise SystemExit(main())
