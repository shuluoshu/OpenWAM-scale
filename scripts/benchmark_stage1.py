#!/usr/bin/env python3
"""Benchmark real Stage1 updates on a reserved torchrun allocation.

Example: torchrun --nproc_per_node=8 ... scripts/benchmark_stage1.py \
  --config BASE.yaml --index population.sqlite --prepared-runtime PROOF.json \
  --output-dir benchmark-new --case b64_w8_p8

Output is throughput evidence only, never a Stage1 training/checkpoint handoff.
"""
from datetime import timedelta
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,help='exact production Stage1 config bound by proof')
    parser.add_argument('--source-yaml')
    parser.add_argument('--index',required=True)
    parser.add_argument('--prepared-runtime',required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--matrix',help='JSON file: list of named performance-only profiles')
    parser.add_argument('--case',help='run exactly one named profile')
    parser.add_argument('--warmup-steps',type=int,default=3)
    parser.add_argument('--measured-steps',type=int,default=10)
    parser.add_argument('--timeout-seconds',type=int,default=180,help='process group and per-step watchdog, maximum 300')
    parser.add_argument('--startup-timeout-seconds',type=int,default=600,help='CUDA/NCCL/model bootstrap watchdog, maximum 900')
    args=parser.parse_args(argv)
    import torch
    import torch.distributed as dist
    from openwam.train.goal_benchmark import parse_cases,run_benchmark,watchdog,validate_launch_state
    from openwam.train.goal_curated import load_config
    from openwam.train.goal_epoch import atomic_json
    from openwam.train.goal_optimizer import optimizer_config
    from openwam.dataloader.curated_goal import CuratedGoalDataset
    from openwam.dataloader.goal_epoch import epoch_sample_fps
    from scripts.train_goal_epoch import validated_population
    if not 1<=args.timeout_seconds<=300:parser.error('--timeout-seconds must be between 1 and 300')
    if not 1<=args.startup_timeout_seconds<=900:parser.error('--startup-timeout-seconds must be between 1 and 900')
    config=load_config(args.config,source_yaml=args.source_yaml)
    options=optimizer_config(config)
    if options['sharding']!='zero1' or options['device']!='model':
        raise ValueError('production benchmark requires base GPU FP32-master zero1 optimizer')
    if config.get('dtype','bfloat16')!='bfloat16':raise ValueError('production benchmark requires base BF16 dtype')
    spec=json.loads(Path(args.matrix).read_text()) if args.matrix else None
    cases=parse_cases(spec,config,warmup_steps=args.warmup_steps,measured_steps=args.measured_steps)
    if args.case:
        cases=[case for case in cases if case['name']==args.case]
        if not cases:raise ValueError('--case does not match a matrix profile')
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    if world<1:raise ValueError('invalid WORLD_SIZE')
    output=Path(args.output_dir).resolve();output_accepted=False
    timeout=timedelta(seconds=args.timeout_seconds)
    torch.set_num_threads(2)
    # Gloo coordinates proof failures entirely on CPU, before any CUDA device or
    # model bootstrap. Destroy it before creating the normal default NCCL group
    # used by both DDP and native ZeroRedundancyOptimizer.
    singleton=None
    init={}
    if world==1:
        import tempfile
        singleton=tempfile.TemporaryDirectory(prefix='openwam-benchmark-group-')
        init=dict(init_method=Path(singleton.name,'cpu').as_uri(),rank=0,world_size=1)
    try:
        with watchdog(args.timeout_seconds,None,rank,'launch-preflight'):
            dist.init_process_group('gloo',timeout=timeout,**init)
            environment=validate_launch_state(output)
            output.mkdir(parents=True,exist_ok=True);output_accepted=True
        with watchdog(args.timeout_seconds,output,rank,'prepared-runtime-validation'):
            ds=dict(config['dataset']);ds['cache_dir']=str(output/'cache'/f'rank_{rank}')
            dataset=CuratedGoalDataset(**ds,seed=config.get('seed',42),dtype=getattr(torch,config.get('dtype','bfloat16')))
            finite=validated_population(dataset,args.index,sample_fps=epoch_sample_fps(config),
                                        prepared_runtime=args.prepared_runtime,config=config)
            proof_facts=dict(config=config,index=str(Path(args.index).resolve()),
                             proof=str(Path(args.prepared_runtime).resolve()),cases=cases,population=finite.summary,
                             environment=environment,cache_policy=environment['cache_policy'])
            all_facts=[None]*world;dist.all_gather_object(all_facts,proof_facts)
            if any(facts!=proof_facts for facts in all_facts):raise ValueError('base config/proof/population/matrix/environment differs between ranks')
            if rank==0:atomic_json(output/'benchmark_manifest.json',dict(benchmark_only=True,world_size=world,**proof_facts))
            dist.destroy_process_group()
        with watchdog(args.startup_timeout_seconds,output,rank,'cuda-nccl-bootstrap'):
            set_group_timeout=getattr(dist.distributed_c10d,'_set_pg_timeout',None)
            if not callable(set_group_timeout):raise RuntimeError('benchmark requires PyTorch process-group timeout adjustment support')
            device=torch.device('cuda',local);torch.cuda.set_device(device)
            if singleton is not None:init['init_method']=Path(singleton.name,'gpu').as_uri()
            dist.init_process_group('nccl',timeout=timedelta(seconds=args.startup_timeout_seconds),**init)
            from openwam.train.bootstrap import run_bootstrap_in_waves,bootstrap_query_components
            from openwam.model.video_backbone.wan.encode import encode_text
            local_world=int(os.environ.get('LOCAL_WORLD_SIZE',world))
            concurrency=config.get('bootstrap_workers_per_node',2)
            if type(concurrency) is int and concurrency>0:concurrency=min(concurrency,local_world)
            started=time.monotonic()
            expert,text_encoder,tokenizer,dependencies=run_bootstrap_in_waves(
                lambda:bootstrap_query_components(config,device=device,dtype=getattr(torch,config.get('dtype','bfloat16'))),
                local_rank=local,local_world_size=local_world,workers_per_node=concurrency)
            if config.get('codec_cache_max_bytes',0):expert.codec.enable_cache(max_bytes=config['codec_cache_max_bytes'])
            if rank==0:atomic_json(output/'benchmark_bootstrap.json',dict(benchmark_only=True,
                elapsed_s=time.monotonic()-started,dependencies=dependencies))
            text_device=next(text_encoder.parameters()).device
            encode=lambda prompt:encode_text([prompt],tokenizer=tokenizer,text_encoder=text_encoder,device=text_device)
            # Bootstrap may exceed a training-step deadline; restore the strict
            # collective deadline before DDP setup or any benchmark update.
            set_group_timeout(timeout)
        run_benchmark(expert,finite,encode,config,cases,output,timeout_seconds=args.timeout_seconds)
    except BaseException as exc:
        if output_accepted and not (output/f'failure_rank_{rank}.json').exists():
            atomic_json(output/f'failure_rank_{rank}.json',dict(benchmark_only=True,rank=rank,
                error=f'{type(exc).__name__}: {exc}',restart_required=True))
        raise
    finally:
        with watchdog(args.timeout_seconds,output if output_accepted else None,rank,'process-group-cleanup'):
            if dist.is_initialized():dist.destroy_process_group()
            if singleton is not None:singleton.cleanup()
    return 0


if __name__=='__main__':raise SystemExit(main())
