"""Bounded throughput measurements; deliberately no training completion artifacts.

CUDA synchronization around each phase makes GPU timings real elapsed times.
This adds measurement overhead consistently across profiles. File-system/page
caches are outside this runner's control; application caches reset each case.
Any case failure aborts the matrix: restart a selected case in a fresh job.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import gc
import json
import math
import os
from pathlib import Path
import re
import statistics
import threading
import time

TIMINGS = ('load_s', 'prepare_s', 'all_ready_wait_s', 'forward_backward_s', 'optimizer_sync_s', 'total_s')



def benchmark_environment(environ=None):
    """Resolve decoder defaults and forbid case-dependent persistent index builds."""
    source=os.environ if environ is None else environ
    decode=source.get('OPENWAM_MECKA_DECODE_MODE','indexed')
    mode=source.get('OPENWAM_VIDEO_INDEX_MODE','lazy')
    if decode not in ('indexed','sequential'):raise ValueError('invalid OPENWAM_MECKA_DECODE_MODE')
    if mode not in ('lazy','read','off'):raise ValueError('invalid OPENWAM_VIDEO_INDEX_MODE')
    effective='off' if decode=='sequential' else mode
    if effective=='lazy':raise ValueError('benchmark rejects lazy video index builds; prebuild and use read, or use sequential/off')
    try:decode_threads=int(source.get('OPENWAM_VIDEO_DECODE_THREADS','2'))
    except (ValueError,TypeError):raise ValueError('OPENWAM_VIDEO_DECODE_THREADS must be an integer in [1,8]')
    if not 1<=decode_threads<=8:raise ValueError('OPENWAM_VIDEO_DECODE_THREADS must be an integer in [1,8]')
    default_cache=Path(source.get('XDG_CACHE_HOME',Path.home()/'.cache'))/'openwam/video_index'
    cache=Path(source.get('OPENWAM_VIDEO_INDEX_CACHE',default_cache)).expanduser().resolve()
    policy=dict(effective_video_index_mode=effective,video_index_builds_allowed=False,
                application_caches='reset before each case',sample_cache_size=0,
                filesystem_cache='not controlled by benchmark')
    return dict(OPENWAM_MECKA_DECODE_MODE=decode,OPENWAM_VIDEO_INDEX_MODE=mode,
                OPENWAM_VIDEO_INDEX_CACHE=str(cache),OMP_NUM_THREADS=source.get('OMP_NUM_THREADS'),
                MKL_NUM_THREADS=source.get('MKL_NUM_THREADS'),OPENWAM_VIDEO_DECODE_THREADS=str(decode_threads),cache_policy=policy)


def validate_launch_state(output):
    """Collectively reject old evidence or differing optimization environments.

    This function never writes. Only a real control/ directory is allowed in
    an existing output root; it belongs to an external allocation watchdog.
    """
    import torch.distributed as dist
    error=None;environment=None
    try:
        path=Path(output)
        if path.exists():
            conflicts=[p.name for p in path.iterdir()
                       if p.name!='control' or not p.is_dir() or p.is_symlink()]
            if conflicts:raise ValueError('benchmark output must be fresh; existing entries: '+str(sorted(conflicts)))
        environment=benchmark_environment()
    except Exception as exc:error=f'{type(exc).__name__}: {exc}'
    local=dict(error=error,environment=environment);states=[local]
    if dist.is_initialized():
        states=[None]*dist.get_world_size();dist.all_gather_object(states,local)
    errors=[f'rank {rank}: {state["error"]}' for rank,state in enumerate(states) if state['error']]
    if errors:raise ValueError('benchmark launch rejected: '+'; '.join(errors))
    if any(state['environment']!=states[0]['environment'] for state in states):
        raise ValueError('benchmark optimization environment differs between ranks')
    return environment

def parse_cases(spec, base, *, warmup_steps=3, measured_steps=10):
    """Whitelist performance knobs, leaving the proof-bound base config intact."""
    if spec is None:
        spec = [dict(name=f'b{batch}_w8_p{batch//8}', batch_size=batch,
                     num_workers=8, prefetch_factor=batch//8) for batch in (32,64,128)]
    if not isinstance(spec,list) or not spec:
        raise ValueError('matrix must be a nonempty JSON list')
    defaults = dict(batch_size=base.get('batch_size',32),num_workers=base.get('dataset_num_workers',8),
                    prefetch_factor=base.get('prefetch_factor',4),text_cache_size=base.get('text_cache_size',128),
                    warmup_steps=warmup_steps,measured_steps=measured_steps,
                    decode_threads=int(os.environ.get('OPENWAM_VIDEO_DECODE_THREADS','2')))
    result=[]; names=set()
    for entry in spec:
        if not isinstance(entry,dict) or set(entry)-({'name'}|defaults.keys()):
            raise ValueError('matrix accepts only name and performance controls')
        name=entry.get('name')
        if not isinstance(name,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}',name) or name in names:
            raise ValueError('case name must be unique and safe')
        case=dict(defaults,**entry)
        for key in defaults:
            minimum=0 if key in ('num_workers','text_cache_size','warmup_steps') else 1
            if type(case[key]) is not int or case[key]<minimum:
                raise ValueError(f'{key} must be an integer >= {minimum}')
        if case['decode_threads']>8:raise ValueError('decode_threads must be in [1,8]')
        names.add(name);result.append(case)
    return result


def summarize_case(case, rank_results):
    """Use actual rank wall windows, never sum component timers for throughput."""
    samples=valid_tcp=samples_with_tcp=0; timings={key:[] for key in TIMINGS}
    per_rank=[]
    for result in rank_results:
        rows=[row for row in result['rows'] if row['measured']]
        if not rows or not math.isfinite(result['measured_wall_s']) or result['measured_wall_s']<=0:
            raise ValueError('empty or invalid measured window')
        samples+=sum(r['real_samples'] for r in rows)
        valid_tcp+=sum(r['valid_tcp_points'] for r in rows)
        samples_with_tcp+=sum(r['samples_with_tcp'] for r in rows)
        for key in TIMINGS:timings[key].append(statistics.mean(r[key] for r in rows))
        per_rank.append({k:v for k,v in result.items() if k!='rows'})
    elapsed=max(r['measured_wall_s'] for r in rank_results)
    return dict(benchmark_only=True,case=case,measured_global_samples=samples,measured_wall_s=elapsed,
                global_samples_per_second=samples/elapsed,valid_tcp_points=valid_tcp,
                samples_with_tcp=samples_with_tcp,
                timing_s={key:dict(max=max(values),p50=statistics.median(values)) for key,values in timings.items()},
                timing_definition='MAX/P50 across rank mean measured-step seconds; synchronized GPU boundaries',
                per_rank=per_rank)


@contextmanager
def watchdog(seconds, output, rank, phase):
    """Hard bound even Python/DataLoader hangs; launcher must kill peer ranks."""
    def expired():
        from openwam.train.goal_epoch import atomic_json
        try:
            failure=dict(benchmark_only=True,phase=phase,rank=rank,
                         error=f'watchdog exceeded {seconds}s',restart_required=True)
            if output is not None:atomic_json(Path(output)/f'failure_rank_{rank}.json',failure)
            else:
                import sys
                print(json.dumps(failure),file=sys.stderr,flush=True)
        finally:
            os._exit(124)
    timer=threading.Timer(seconds,expired);timer.daemon=True;timer.start()
    try:yield
    finally:timer.cancel()


def run_benchmark(expert,dataset,encode_text,config,cases,output,*,timeout_seconds=180):
    """Run native batches, DDP, losses, clipping and optimizer; return summaries.

    CPU execution is supported for focused tiny-model tests. Production launch
    enforces CUDA, BF16 model/GPU FP32 master ZeRO-1 via the base config.
    """
    import torch
    import torch.distributed as dist
    from openwam.train.epoch_order import EpochOrder,sampling_contract
    from openwam.train.goal_batch import GoalBatchLoader,prepare_goal_batch,rank_batch
    from openwam.train.goal_curated import TextContextCache,goal_parameter_groups
    from openwam.train.goal_epoch import atomic_json
    from openwam.train.goal_optimizer import build_goal_optimizer
    from openwam.train.goal_trainer import _seed
    from openwam.model.goal.history import history_kwargs
    from openwam.model.goal.heads import point_mask
    if type(timeout_seconds) is not int or not 1<=timeout_seconds<=300:
        raise ValueError('timeout_seconds must be between 1 and 300')
    distributed=dist.is_initialized(); rank=dist.get_rank() if distributed else 0
    world=dist.get_world_size() if distributed else 1
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/f'rank_{rank}.jsonl').exists():raise ValueError('benchmark output must be fresh')
    parameter=next(expert.parameters());device,dtype=parameter.device,parameter.dtype
    def sync():
        if device.type=='cuda':torch.cuda.synchronize(device)
    def barrier():
        if distributed:dist.barrier()
    def gather(value):
        values=[None]*world
        if distributed:dist.all_gather_object(values,value)
        else:values=[value]
        return values
    def ensure_finite(value,label):
        flag=torch.tensor(int(bool(torch.isfinite(value).all())),device=device)
        if distributed:dist.all_reduce(flag,op=dist.ReduceOp.MIN)
        if not flag.item():raise FloatingPointError('nonfinite '+label)
    setup=time.monotonic()
    with watchdog(timeout_seconds,output,rank,'order/DDP/snapshot'):
        order=EpochOrder(sampling_contract(config,len(dataset)))
        if order.contract['mode']!='global_shuffle':raise ValueError('benchmark requires global_shuffle')
        for case in cases:
            if (case['warmup_steps']+case['measured_steps']-1)*world*case['batch_size']>=len(dataset):
                raise ValueError(f"case {case['name']} exceeds one finite population")
        expert.train_stage_a()
        model=expert
        if distributed:
            model=torch.nn.parallel.DistributedDataParallel(expert,device_ids=[device.index] if device.type=='cuda' else None,
                gradient_as_bucket_view=True,find_unused_parameters=True)
        initial={name:p.detach().cpu().clone() for name,p in expert.named_parameters() if p.requires_grad}
        buffers={name:b.detach().cpu().clone() for name,b in expert.named_buffers()}
        sync()
    common_setup_s=time.monotonic()-setup
    environment=benchmark_environment()
    summaries=[]
    with (output/f'rank_{rank}.jsonl').open('x',buffering=65536) as log:
        def event(kind,**values):
            log.write(json.dumps(dict(event=kind,benchmark_only=True,rank=rank,**values),allow_nan=False)+'\n')
        event('setup',common_setup_s=common_setup_s,sampling=order.contract,sampling_sha256=order.sha256,
              anchor_prefix=[order[i] for i in range(min(32,len(dataset)))],environment=environment,
              world_size=world,base_config=config)
        log.flush()
        for case in cases:
            loader=optimizer=cache=None;case_start=time.monotonic(); rows=[];cursor=0
            prior_decode_threads=os.environ.get('OPENWAM_VIDEO_DECODE_THREADS')
            try:
                with watchdog(timeout_seconds,output,rank,case['name']+':setup'):
                    os.environ['OPENWAM_VIDEO_DECODE_THREADS']=str(case['decode_threads'])
                    case_environment=benchmark_environment()
                    # Fresh readers and metadata state; no cached sample replay.
                    current=dataset
                    while current is not None:
                        for name in ('_episodes','_validated'):
                            value=getattr(current,name,None)
                            if value is not None:value.clear()
                        connection=getattr(current,'_connection',None)
                        if connection is not None:connection.close();current._connection=None
                        current=getattr(current,'dataset',None)
                    with torch.no_grad():
                        for name,p in expert.named_parameters():
                            if name in initial:p.copy_(initial[name])
                        for name,b in expert.named_buffers():b.copy_(buffers[name])
                    expert.zero_grad(set_to_none=True)
                    if hasattr(expert.codec,'clear_cache'):expert.codec.clear_cache()
                    _seed(config.get('seed',42))
                    cache=TextContextCache(encode_text,max_entries=case['text_cache_size']) if encode_text is not None else None
                    optimizer=build_goal_optimizer(goal_parameter_groups(expert,config['learning_rate'],
                        config.get('history_learning_rate',1e-4),config.get('condition_learning_rate')),config)
                    loader=GoalBatchLoader(dataset,order,cursor=0,rank=rank,world=world,batch_size=case['batch_size'],
                        num_workers=case['num_workers'],prefetch_factor=case['prefetch_factor'],seed=config.get('seed',42),
                        max_batches=case['warmup_steps']+case['measured_steps'],sample_cache_size=0,
                        pin_memory=config.get('dataset_pin_memory',False))
                    sync();barrier()
                    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
                setup_s=time.monotonic()-case_start
                event('case_setup',case=case['name'],setup_s=setup_s,profile=case);log.flush()
                measured_start=None
                for step in range(1,case['warmup_steps']+case['measured_steps']+1):
                    with watchdog(timeout_seconds,output,rank,f"{case['name']}:step:{step}"):
                        if step==case['warmup_steps']+1:
                            sync();barrier();measured_start=time.monotonic()
                        start=time.monotonic()
                        positions,real_count,weight,consumed=rank_batch(cursor,len(dataset),rank,world,case['batch_size'])
                        index=order[positions[0]];seed=config.get('seed',42)+index
                        _seed(seed);expert.train_stage_a();optimizer.zero_grad()
                        t=time.monotonic();pairs=next(loader);load_s=time.monotonic()-t
                        t=time.monotonic()
                        batch=prepare_goal_batch(pairs,device=device,dtype=dtype,context_cache=cache)
                        if not real_count:
                            target=batch['targets']
                            batch['targets']=replace(target,image_valid=torch.zeros_like(target.image_valid),point_valid=torch.zeros_like(target.point_valid))
                        sync();prepare_s=time.monotonic()-t
                        # Timing before rendezvous preserves each rank's own input cost.
                        t=time.monotonic();barrier();all_ready_wait_s=time.monotonic()-t
                        _seed(seed);t=time.monotonic()
                        out=model(batch['observed'],batch['context'],batch['targets'],context_lens=batch.get('context_lens'),
                                  observed_view_valid=batch.get('observed_view_valid'),**history_kwargs(batch))
                        ensure_finite(out['loss'],'Stage1 loss')
                        (out['loss']*weight).backward();sync();forward_backward_s=time.monotonic()-t
                        t=time.monotonic()
                        norm=torch.nn.utils.clip_grad_norm_(expert.parameters(),1.,error_if_nonfinite=False,foreach=False)
                        ensure_finite(norm,'Stage1 gradient norm')
                        optimizer.step();sync();optimizer_sync_s=time.monotonic()-t
                        row=dict(case=case['name'],step=step,measured=step>case['warmup_steps'],anchor_index=index,
                                 cursor=cursor,real_samples=real_count,global_real_samples=consumed,load_s=load_s,prepare_s=prepare_s,
                                 all_ready_wait_s=all_ready_wait_s,forward_backward_s=forward_backward_s,
                                 optimizer_sync_s=optimizer_sync_s,total_s=time.monotonic()-start,
                                 loss=float(out['loss'].detach()),gradient_norm=float(norm),
                                 valid_tcp_points=int(out['valid_points']) if real_count else 0,
                                 samples_with_tcp=int(point_mask(batch['targets']).any(dim=1).sum()) if real_count else 0)
                        rows.append(row);event('update',**row);log.flush();cursor+=consumed
                        del out,batch,pairs,norm
                sync();barrier();elapsed=time.monotonic()-measured_start
                local=dict(rank=rank,rows=rows,measured_wall_s=elapsed,setup_s=setup_s,
                           peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0,
                           peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else 0)
                with watchdog(timeout_seconds,output,rank,case['name']+':summarize'):
                    results=gather(local)
                    summary=summarize_case(case,results)
                    summary.update(environment=case_environment,cache_policy=case_environment['cache_policy'])
                    summaries.append(summary)
                    if rank==0:
                        atomic_json(output/(case['name']+'.benchmark.json'),summary)
                        print(json.dumps(summary,allow_nan=False),flush=True)
                event('case_end',case=case['name'],measured_wall_s=elapsed);log.flush()
            except BaseException as exc:
                atomic_json(output/f'failure_rank_{rank}.json',dict(benchmark_only=True,case=case,
                    rank=rank,error=f'{type(exc).__name__}: {exc}',restart_required=True))
                log.flush()
                # Do not attempt CUDA/NCCL recovery or start another case.
                raise
            finally:
                if prior_decode_threads is None:os.environ.pop('OPENWAM_VIDEO_DECODE_THREADS',None)
                else:os.environ['OPENWAM_VIDEO_DECODE_THREADS']=prior_decode_threads
                with watchdog(timeout_seconds,output,rank,case['name']+':cleanup'):
                    if loader is not None:loader.close()
                    loader=optimizer=cache=None
                    expert.zero_grad(set_to_none=True)
                    if hasattr(expert.codec,'clear_cache'):expert.codec.clear_cache()
                    gc.collect()
                    if device.type=='cuda':torch.cuda.empty_cache()
        if rank==0:atomic_json(output/'benchmark_summary.json',dict(benchmark_only=True,cases=summaries,environment=environment,cache_policy=environment['cache_policy']))
    return summaries
