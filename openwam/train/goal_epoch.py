"""Finite Stage1 updates, distributed tail weighting, atomic resumable handoff."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
import uuid
from collections import Counter
import torch
import torch.distributed as dist


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.pending')
    with temporary.open('w') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def rank_sample(cursor,total,rank,world):
    if not 0<=cursor<total or not 0<=rank<world:raise ValueError('invalid finite cursor/rank')
    active=min(world,total-cursor)
    return (cursor+rank,world/active) if rank<active else (total-1,0.)


def _cpu(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:_cpu(v) for k,v in value.items()}
    if isinstance(value,list):return [_cpu(v) for v in value]
    if isinstance(value,tuple):return tuple(_cpu(v) for v in value)
    return value


def optimizer_payload(optimizer):
    return dict(masters=[p.detach().cpu().clone() for p in optimizer.master_parameters],
                adam=_cpu(optimizer.optimizer.state_dict()))


def restore_optimizer(optimizer,payload):
    if len(payload['masters'])!=len(optimizer.master_parameters):raise ValueError('optimizer parameter count mismatch')
    with torch.no_grad():
        for (model,master),saved in zip(optimizer.parameter_pairs,payload['masters']):
            if saved.shape!=master.shape or saved.dtype!=torch.float32:raise ValueError('invalid FP32 master')
            master.copy_(saved);model.copy_(master)
    optimizer.optimizer.load_state_dict(payload['adam'])


def save_epoch_checkpoint(expert,optimizer,output,*,identity,cursor,update,dependencies):
    from openwam.train.goal_query import save_query_bundle
    output=Path(output);root=output/'checkpoints';root.mkdir(parents=True,exist_ok=True)
    if not 0<=cursor<=identity['total_samples']:raise ValueError('invalid checkpoint cursor')
    temporary=root/('.pending-'+uuid.uuid4().hex);temporary.mkdir()
    destination=root/f'update_{update:09d}_{uuid.uuid4().hex[:8]}'
    try:
        save_query_bundle(expert,temporary/'query_goal.safetensors',
            training_metadata=dict(identity=identity,cursor=cursor,update=update),dependency_identities=dependencies)
        torch.save(dict(identity=identity,cursor=cursor,update=update,optimizer=optimizer_payload(optimizer)),temporary/'training.pt')
        files={p.name:sha256(p) for p in temporary.iterdir() if p.is_file()}
        atomic_json(temporary/'manifest.json',dict(identity=identity,cursor=cursor,update=update,files=files))
        for p in temporary.iterdir():
            with p.open('rb') as f:os.fsync(f.fileno())
        temporary.rename(destination)
        fd=os.open(root,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
        atomic_json(output/'latest.json',dict(directory=str(destination.resolve())))
    except BaseException:
        shutil.rmtree(temporary,ignore_errors=True);raise
    # Keep two durable generations. Never remove pending/orphan directories.
    complete=sorted((p for p in root.glob('update_*') if (p/'manifest.json').is_file()),key=lambda p:p.name)
    for old in complete[:-2]:shutil.rmtree(old)
    return destination


def _checkpoint(output,identity):
    directory=Path(json.loads((Path(output)/'latest.json').read_text())['directory'])
    manifest=json.loads((directory/'manifest.json').read_text())
    if manifest['identity']!=identity:raise ValueError('checkpoint identity mismatch')
    for name,digest in manifest['files'].items():
        if sha256(directory/name)!=digest:raise ValueError('checkpoint checksum mismatch: '+name)
    return directory,manifest


def restore_epoch_checkpoint(expert,optimizer,output,identity):
    from safetensors.torch import load_file
    directory,manifest=_checkpoint(output,identity)
    expert.load_state_dict(load_file(str(directory/'query_goal.safetensors')),strict=True)
    saved=torch.load(directory/'training.pt',map_location='cpu',weights_only=True)
    if saved['identity']!=identity or any(saved[k]!=manifest[k] for k in ('cursor','update')):
        raise ValueError('training identity/cursor mismatch')
    restore_optimizer(optimizer,saved['optimizer'])
    return saved['cursor'],saved['update']


def verify_completion(output,identity):
    proof=json.loads((Path(output)/'epoch_complete.json').read_text())
    if proof['identity']!=identity or proof['seen_samples']!=identity['total_samples']:
        raise ValueError('incomplete epoch or identity mismatch')
    directory,manifest=_checkpoint(output,identity)
    if manifest['cursor']!=identity['total_samples']:raise ValueError('partial checkpoint cannot hand off')
    checkpoint=directory/'query_goal.safetensors'
    if str(checkpoint)!=proof['checkpoint'] or sha256(checkpoint)!=proof['checkpoint_sha256']:
        raise ValueError('handoff checkpoint mismatch')
    return proof


def train_epoch(expert,dataset,context_cache,output,*,identity,dependencies,
                learning_rate=1e-5,history_learning_rate=1e-4,checkpoint_every=5000,
                eval_every=500,sampling_steps=20,resume=False,bootstrap_workers_per_node=2):
    from openwam.train.goal_curated import prepare_batch,goal_parameter_groups,_export_cases
    from openwam.train.goal_overfit import FP32MasterAdamW
    from openwam.train.goal_trainer import _seed
    from openwam.train.goal_patch import evaluate_patch
    from openwam.train.goal_history import render_goal_history
    from openwam.model.goal.history import history_kwargs
    from openwam.train.epoch_order import EpochOrder
    distributed=dist.is_initialized();rank=dist.get_rank() if distributed else 0;world=dist.get_world_size() if distributed else 1
    if world!=identity['world_size'] or len(dataset)!=identity['total_samples'] or not len(dataset):raise ValueError('population/world identity mismatch')
    if any(not isinstance(v,int) or v<1 for v in (checkpoint_every,eval_every,sampling_steps)):raise ValueError('invalid training intervals')
    if not math.isfinite(learning_rate) or learning_rate<=0:raise ValueError('invalid learning rate')
    # The cursor counts permutation positions, not source-contiguous dataset IDs.
    # Validate on every rank and propagate failures before any rank starts DDP.
    order=None;order_error=None
    try:
        order=EpochOrder(identity.get('sampling'))
        if order.total!=len(dataset) or order.contract['seed']!=identity['seed']:
            raise ValueError('sampling contract population/seed mismatch')
    except Exception as exc:order_error=f'{type(exc).__name__}: {exc}'
    order_state=dict(error=order_error,contract=order.contract if order is not None else None,
                     sha256=order.sha256 if order is not None else None)
    order_states=[order_state]
    if distributed:
        order_states=[None]*world
        dist.all_gather_object(order_states,order_state)
    if any(s['error'] for s in order_states):
        raise ValueError('sampling initialization failed: '+str([s['error'] for s in order_states]))
    if any(s!=order_states[0] for s in order_states):raise ValueError('sampling order differs between ranks')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    p=next(expert.parameters());device,dtype=p.device,p.dtype
    def primary(call):
        result=None;error=None
        if rank==0:
            try:result=call()
            except Exception as exc:error=f'{type(exc).__name__}: {exc}'
        message=[error]
        if distributed:dist.broadcast_object_list(message,src=0,device=device)
        if message[0]:raise RuntimeError(message[0])
        return result
    def event(kind,**values):
        if rank==0:
            row=dict(event=kind,wall_time=time.time(),**values)
            with (output/'progress.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
            print(json.dumps(row,allow_nan=False),flush=True)
    primary(lambda: (_ for _ in ()).throw(ValueError('existing run requires --resume')) if (output/'latest.json').exists() and not resume else None)
    if (output/'epoch_complete.json').exists():
        proof=verify_completion(output,identity);expert.freeze();return proof
    expert.train_stage_a()
    optimizer=FP32MasterAdamW(goal_parameter_groups(expert,learning_rate,history_learning_rate))
    cursor=update=0
    if resume and (output/'latest.json').exists():
        # Bound transient host RAM per node while other nodes restore in parallel.
        from openwam.train.bootstrap import run_bootstrap_in_waves
        def restore():
            from safetensors import safe_open
            directory,_=_checkpoint(output,identity)
            with safe_open(str(directory/'query_goal.safetensors'),framework='pt',device='cpu') as bundle:
                stored=json.loads(bundle.metadata()['query_goal_bundle'])['dependencies']
            if any(stored.get(k)!=v for k,v in dependencies.items()):
                raise ValueError('resume frozen dependency identity mismatch')
            return restore_epoch_checkpoint(expert,optimizer,output,identity)
        local_world=int(os.environ.get('LOCAL_WORLD_SIZE',world))
        local_rank=int(os.environ.get('LOCAL_RANK',rank))
        concurrency=bootstrap_workers_per_node
        if type(concurrency) is int and concurrency>0:concurrency=min(concurrency,local_world)
        cursor,update=run_bootstrap_in_waves(restore,local_rank=local_rank,
            local_world_size=local_world,workers_per_node=concurrency)
    event('resume' if resume else 'started',cursor=cursor,step=update,identity=identity,
          sampling=order.contract,sampling_sha256=order.sha256,sampling_host_bytes=order.nbytes)
    model=expert
    if distributed:
        from torch.nn.parallel import DistributedDataParallel
        model=DistributedDataParallel(expert,device_ids=[device.index] if device.type=='cuda' else None,
            gradient_as_bucket_view=True,find_unused_parameters=True)
    summary=dict(mode='finite_stage1',identity=identity,scope=dataset.scope_summary(),evaluations=[],
        completed_steps=update,steps=math.ceil(len(dataset)/world),evaluation_split='train',success_verified=False)
    fixed=[];cards=[]
    def setup():
        nonlocal fixed,cards
        fixed=list(dataset.evaluation_samples());cards=_export_cases(output,fixed)
        old=output/'summary.json'
        if resume and old.exists():summary['evaluations']=[e for e in json.loads(old.read_text()).get('evaluations',[]) if e['step']<=update]
        (output/'index.html').write_text('<meta charset="utf-8"><h1>Finite Stage1</h1><a href="timeline.html">All evaluations</a> · <a href="inputs/index.html">Training inputs</a>')
    primary(setup)
    def evaluate():
        directory=f'eval_step_{update:09d}'
        dest=output/directory
        # A crash can leave an incomplete evaluation; rewrite only that uncommitted step.
        if dest.exists():shutil.rmtree(dest)
        batches=(prepare_batch(r,b,device=device,dtype=dtype,context_cache=context_cache) for r,b in fixed)
        result=evaluate_patch(expert,batches,dest,step=update,seed=identity['seed']+100000,num_steps=sampling_steps)
        summary['evaluations']=[e for e in summary['evaluations'] if e['step']!=update]
        summary['evaluations'].append(dict(step=update,directory=directory,**{k:result[k] for k in ('point_metrics','group_metrics','image_mae_uint8')}))
        summary.update(completed_steps=update,seen_samples=cursor)
        atomic_json(output/'summary.json',summary);render_goal_history(output,summary,cards)
        timeline=output/'timeline.html'
        timeline.write_text(timeline.read_text().replace('href="../index.html"','href="index.html"').replace('href="../g1_concept_and_plan.html"','href="summary.json"'))
        event('report_ready',step=update,cursor=cursor,directory=directory)
    started=time.monotonic()
    while cursor<len(dataset):
        position,weight=rank_sample(cursor,len(dataset),rank,world)
        index=order[position]
        _seed(identity['seed']+index);expert.train_stage_a();optimizer.zero_grad()
        error=None;batch=None;composition=None
        try:
            record,cpu=dataset.sample(index)
            batch=prepare_batch(record,cpu,device=device,dtype=dtype,context_cache=context_cache)
            if weight>0:composition=dict(family=record.get('family','unknown'),granularity=record.get('granularity','unknown'))
        except Exception as exc:error=f'rank {rank}: {type(exc).__name__}: {exc}'
        prepared=dict(error=error,composition=composition);preparations=[prepared]
        if distributed:
            preparations=[None]*world
            dist.all_gather_object(preparations,prepared)
        errors=[r['error'] for r in preparations if r['error']]
        if any(errors):raise RuntimeError('sample preparation failed: '+str(errors))
        family_counts=dict(Counter(r['composition']['family'] for r in preparations if r['composition']))
        granularity_counts=dict(Counter(r['composition']['granularity'] for r in preparations if r['composition']))
        out=model(batch['observed'],batch['context'],batch['targets'],context_lens=batch.get('context_lens'),
                  observed_view_valid=batch.get('observed_view_valid'),**history_kwargs(batch))
        finite=torch.tensor(int(torch.isfinite(out['loss']).item()),device=device)
        if distributed:dist.all_reduce(finite,op=dist.ReduceOp.MIN)
        if not finite.item():raise FloatingPointError('nonfinite Stage1 loss')
        (out['loss']*weight).backward()
        norm=torch.nn.utils.clip_grad_norm_(expert.parameters(),1.,error_if_nonfinite=True,foreach=False)
        optimizer.step()
        losses={k:float(v.detach())*weight/world for k,v in out.items() if k.endswith('loss') and isinstance(v,torch.Tensor) and v.numel()==1}
        valid_tcp=int(out['valid_points']) if weight>0 else 0
        values=torch.tensor([*losses.values(),valid_tcp,int(valid_tcp>0)],dtype=torch.float64,device=device)
        if distributed:dist.all_reduce(values)
        reduced=values.tolist();losses=dict(zip(losses,reduced[:-2]))
        cursor+=min(world,len(dataset)-cursor);update+=1
        del out,batch,cpu
        event('update',step=update,cursor=cursor,total_samples=len(dataset),gradient_norm=float(norm),elapsed_s=time.monotonic()-started,
              family_counts=family_counts,granularity_counts=granularity_counts,
              valid_tcp_points=int(reduced[-2]),samples_with_tcp=int(reduced[-1]),**losses)
        if update%eval_every==0 or cursor==len(dataset):primary(evaluate)
        if update%checkpoint_every==0 or cursor==len(dataset):
            primary(lambda:save_epoch_checkpoint(expert,optimizer,output,identity=identity,cursor=cursor,update=update,dependencies=dependencies))
            event('checkpoint_saved',step=update,cursor=cursor)
    expert.freeze()
    def finish():
        directory,manifest=_checkpoint(output,identity)
        checkpoint=directory/'query_goal.safetensors'
        proof=dict(identity=identity,total_samples=len(dataset),seen_samples=cursor,updates=update,
            checkpoint=str(checkpoint),checkpoint_sha256=manifest['files']['query_goal.safetensors'])
        atomic_json(output/'epoch_complete.json',proof)
    primary(finish)
    return verify_completion(output,identity)
