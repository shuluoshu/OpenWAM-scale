"""Finite Stage1 updates, distributed tail weighting, atomic resumable handoff."""
from __future__ import annotations
import hashlib
import errno
from dataclasses import dataclass,replace
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


def _file_stamp(path):
    stat=Path(path).stat()
    return dict(device=stat.st_dev,inode=stat.st_ino,mode=stat.st_mode,size=stat.st_size,
                mtime_ns=stat.st_mtime_ns,ctime_ns=stat.st_ctime_ns,nlink=stat.st_nlink)


@dataclass
class _SaveReceipt:
    """Bounded local reuse; metadata guards cannot detect metadata-preserving damage.

    Public verification and resume always hash fresh, including on filesystems
    whose ctime granularity can hide rapid writes with restored mtime.
    """
    directory: Path
    manifest: dict
    stamps: dict
    pid: int

    def matches(self,directory,manifest):
        return (self.pid==os.getpid() and self.directory==directory and self.manifest==manifest
                and all(_file_stamp(directory/name)==stamp for name,stamp in self.stamps.items()))


def save_epoch_checkpoint(expert,optimizer,output,*,identity,cursor,update,dependencies,convergence_state=None,
                          _return_receipt=False):
    from openwam.train.goal_curated import goal_bundle_spec
    name,_,save_bundle=goal_bundle_spec(getattr(expert,'representation','query_latent'))
    output=Path(output);root=output/'checkpoints';root.mkdir(parents=True,exist_ok=True)
    if cursor<0 or (identity.get('training',{}).get('mode')!='bounded_updates' and cursor>identity['total_samples']):
        raise ValueError('invalid checkpoint cursor')
    temporary=root/('.pending-'+uuid.uuid4().hex);temporary.mkdir()
    destination=root/f'update_{update:09d}_{uuid.uuid4().hex[:8]}'
    try:
        save_bundle(expert,temporary/name,
            training_metadata=dict(identity=identity,cursor=cursor,update=update),dependency_identities=dependencies)
        torch.save(dict(identity=identity,cursor=cursor,update=update,optimizer=optimizer_payload(optimizer)),temporary/'training.pt')
        if convergence_state is not None:
            atomic_json(temporary/'convergence_state.json',convergence_state)
        stamps={p.name:_file_stamp(p) for p in temporary.iterdir() if p.is_file()}
        files={name:sha256(temporary/name) for name in stamps}
        if any(_file_stamp(temporary/name)!=stamp for name,stamp in stamps.items()):
            raise ValueError('checkpoint changed during save verification')
        manifest=dict(identity=identity,cursor=cursor,update=update,files=files)
        atomic_json(temporary/'manifest.json',manifest)
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
    if _return_receipt:
        # Round-trip small metadata to detach the receipt from caller-owned dicts.
        return _SaveReceipt(destination.resolve(),json.loads(json.dumps(manifest)),stamps,os.getpid())
    return destination


def _checkpoint(output,identity,*,receipt=None):
    directory=Path(json.loads((Path(output)/'latest.json').read_text())['directory'])
    manifest=json.loads((directory/'manifest.json').read_text())
    if manifest['identity']!=identity:raise ValueError('checkpoint identity mismatch')
    if isinstance(receipt,_SaveReceipt) and receipt.matches(directory,manifest):
        return directory,manifest
    for name,digest in manifest['files'].items():
        if sha256(directory/name)!=digest:raise ValueError('checkpoint checksum mismatch: '+name)
    return directory,manifest


def restore_epoch_checkpoint(expert,optimizer,output,identity):
    from safetensors.torch import load_file
    from openwam.train.goal_curated import goal_bundle_spec
    name,_,_=goal_bundle_spec(getattr(expert,'representation','query_latent'))
    directory,manifest=_checkpoint(output,identity)
    state=load_file(str(directory/name))
    if any(k.startswith('readout.') for k in state) and not hasattr(expert,'readout'):
        raise ValueError('legacy readout checkpoint cannot resume after head removal; '
                         'load model weights with load_query_bundle and start a new optimizer/run')
    expert.load_state_dict(state,strict=True)
    saved=torch.load(directory/'training.pt',map_location='cpu',weights_only=True)
    if saved['identity']!=identity or any(saved[k]!=manifest[k] for k in ('cursor','update')):
        raise ValueError('training identity/cursor mismatch')
    restore_optimizer(optimizer,saved['optimizer'])
    return saved['cursor'],saved['update']


def load_convergence_state(directory,manifest,identity,*,evaluation_identity=None):
    from openwam.train.convergence import validate_state
    if 'convergence_state.json' not in manifest['files']:
        raise ValueError('missing checkpoint convergence state')
    state=json.loads((directory/'convergence_state.json').read_text())
    return validate_state(state,identity['training']['convergence'],
        evaluation_identity if evaluation_identity is not None else state['evaluation_identity'],
        manifest['update'],identity['training']['max_updates'])


def verify_completion(output,identity,*,_receipt=None):
    from openwam.train.goal_curated import goal_bundle_spec
    bounded=identity.get('training',{}).get('mode')=='bounded_updates'
    proof=json.loads((Path(output)/('training_complete.json' if bounded else 'epoch_complete.json')).read_text())
    if proof['identity']!=identity:
        raise ValueError('incomplete epoch or identity mismatch')
    directory,manifest=_checkpoint(output,identity,receipt=_receipt)
    if identity.get('training',{}).get('convergence') is not None:
        gate=load_convergence_state(directory,manifest,identity)
        if (not gate['converged'] or proof.get('completion_kind')!='convergence'
                or proof.get('finite_epoch_complete') is not False
                or proof.get('convergence')!=gate
                or proof.get('convergence_sha256')!=manifest['files']['convergence_state.json']
                or proof.get('updates')!=manifest['update'] or proof.get('completed_steps')!=manifest['update']
                or proof.get('seen_samples')!=manifest['cursor']):
            raise ValueError('convergence completion mismatch')
    elif bounded:
        if (proof.get('completion_kind')!='bounded_updates' or proof.get('finite_epoch_complete') is not False
                or manifest['update']!=identity['training']['max_updates']
                or proof['updates']!=manifest['update'] or proof.get('completed_steps')!=manifest['update']
                or proof['seen_samples']!=manifest['cursor']):
            raise ValueError('bounded update completion mismatch')
    elif manifest['cursor']!=identity['total_samples'] or proof['seen_samples']!=identity['total_samples']:
        raise ValueError('partial checkpoint cannot hand off')
    name,_,_=goal_bundle_spec(identity.get('representation','query_latent'))
    checkpoint=directory/name
    if str(checkpoint)!=proof['checkpoint'] or manifest['files'][name]!=proof['checkpoint_sha256']:
        raise ValueError('handoff checkpoint mismatch')
    return proof



def _publish_bundle(source,target,*,receipt=None):
    """Atomically publish a conventional alias; the proof names the generation."""
    source,target=Path(source),Path(target)
    temporary=target.with_name(target.name+'.pending-'+uuid.uuid4().hex)
    before=_file_stamp(source)
    old=target.lstat() if target.exists() or target.is_symlink() else None
    same_old=old is not None and (old.st_dev,old.st_ino)==(before['device'],before['inode'])
    linked=False
    try:
        try:
            os.link(source,temporary);linked=True
        except OSError as exc:
            if exc.errno not in (errno.EXDEV,errno.EPERM,errno.EOPNOTSUPP,errno.ENOSYS,errno.EMLINK):raise
            shutil.copyfile(source,temporary)
            with temporary.open('rb') as stream:os.fsync(stream.fileno())
        os.replace(temporary,target)
    finally:
        temporary.unlink(missing_ok=True)
    if linked and isinstance(receipt,_SaveReceipt) and receipt.stamps.get(source.name)==before:
        after=_file_stamp(source)
        # link/replace intentionally changes only ctime and link count. Never
        # bless changed data/inode/mtime; a mismatch forces full verification.
        stable=('device','inode','mode','size','mtime_ns')
        if (all(before[k]==after[k] for k in stable)
                and after['nlink']==before['nlink']+1-int(same_old)):
            receipt.stamps[source.name]=after

def train_epoch(expert,dataset,context_cache,output,*,identity,dependencies,
                learning_rate=1e-5,history_learning_rate=1e-4,checkpoint_every=5000,
                eval_every=500,sampling_steps=20,resume=False,bootstrap_workers_per_node=2,optimizer_device='model',
                condition_learning_rate=None,max_updates=None,repeat_population=False,sample_cache_size=0,
                convergence=None,batch_size=1,dataset_num_workers=0,prefetch_factor=2):
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
    if optimizer_device not in ('model','cpu'):raise ValueError('optimizer_device must be model or cpu')
    if type(sample_cache_size) is not int or sample_cache_size<0:raise ValueError('sample_cache_size must be nonnegative')
    from openwam.train.goal_batch import GoalBatchLoader,prepare_goal_batch,rank_batch
    loading=dict(batch_size=batch_size,dataset_num_workers=dataset_num_workers,prefetch_factor=prefetch_factor)
    defaults=dict(batch_size=1,dataset_num_workers=0,prefetch_factor=2)
    bounded=max_updates is not None
    if bounded and (type(max_updates) is not int or max_updates<1):raise ValueError('max_updates must be a positive integer')
    if type(repeat_population) is not bool or (repeat_population and not bounded):
        raise ValueError('repeat_population requires max_updates')
    training=dict(mode='bounded_updates',max_updates=max_updates,repeat_population=repeat_population) if bounded else None
    from openwam.train.convergence import (convergence_config,new_state,observe,fixed_flow_metric,
        evaluation_identity,flow_contract,preserve_rng)
    convergence=convergence_config(convergence,max_updates)
    if convergence is not None:training['convergence']=convergence
    if identity.get('training')!=training:raise ValueError('training budget identity mismatch')
    representation=getattr(expert,'representation','query_latent')
    if identity.get('representation',representation)!=representation:raise ValueError('representation identity mismatch')
    from openwam.train.goal_curated import goal_bundle_spec
    bundle_name,metadata_key,_=goal_bundle_spec(representation)
    # The cursor counts permutation positions, not source-contiguous dataset IDs.
    # Validate on every rank and propagate failures before any rank starts DDP.
    order=None;order_error=None
    try:
        for name,value,minimum in (('batch_size',batch_size,1),('dataset_num_workers',dataset_num_workers,0),
                                  ('prefetch_factor',prefetch_factor,1)):
            if type(value) is not int or value<minimum:raise ValueError(name+' has invalid value')
        if identity.get('data_loading',defaults)!=loading:raise ValueError('data loading identity mismatch')
        if bounded and not repeat_population and max_updates>math.ceil(len(dataset)/(world*batch_size)):
            raise ValueError('max_updates exceeds one population; enable repeat_population')
        order=EpochOrder(identity.get('sampling'))
        if order.total!=len(dataset) or order.contract['seed']!=identity['seed']:
            raise ValueError('sampling contract population/seed mismatch')
    except Exception as exc:order_error=f'{type(exc).__name__}: {exc}'
    order_state=dict(error=order_error,contract=order.contract if order is not None else None,
                     sha256=order.sha256 if order is not None else None,data_loading=loading)
    order_states=[order_state]
    if distributed:
        order_states=[None]*world
        dist.all_gather_object(order_states,order_state)
    if any(s['error'] for s in order_states):
        raise ValueError('sampling initialization failed: '+str([s['error'] for s in order_states]))
    if any(s!=order_states[0] for s in order_states):raise ValueError('sampling order or data loading differs between ranks')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    p=next(expert.parameters());device,dtype=p.device,p.dtype
    def primary(call,*,broadcast_result=False):
        result=None;error=None
        if rank==0:
            try:result=call()
            except Exception as exc:error=f'{type(exc).__name__}: {exc}'
        message=[error,result if broadcast_result else None]
        if distributed:dist.broadcast_object_list(message,src=0,device=device)
        if message[0]:raise RuntimeError(message[0])
        return message[1] if broadcast_result else result
    def event(kind,**values):
        if rank==0:
            row=dict(event=kind,wall_time=time.time(),**values)
            with (output/'progress.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
            print(json.dumps(row,allow_nan=False),flush=True)
    primary(lambda: (_ for _ in ()).throw(ValueError('existing run requires --resume')) if (output/'latest.json').exists() and not resume else None)
    completion_file=output/('training_complete.json' if bounded else 'epoch_complete.json')
    def fingerprint(pairs):
        with preserve_rng():
            prepared=((r,prepare_batch(r,b,device=device,dtype=dtype,context_cache=context_cache)) for r,b in pairs)
            return evaluation_identity(prepared,seed=identity['seed']+200000,expert_contract=flow_contract(expert))
    if completion_file.exists():
        def completed():
            proof=verify_completion(output,identity)
            if convergence is not None:
                with preserve_rng():fixed_inputs=list(dataset.evaluation_samples())
                directory,manifest=_checkpoint(output,identity)
                load_convergence_state(directory,manifest,identity,evaluation_identity=fingerprint(fixed_inputs))
            return proof
        proof=primary(completed,broadcast_result=True);expert.freeze();return proof
    expert.train_stage_a()
    optimizer=FP32MasterAdamW(goal_parameter_groups(expert,learning_rate,history_learning_rate,condition_learning_rate),
                            master_device='cpu' if optimizer_device=='cpu' else None)
    cursor=update=0
    if resume and (output/'latest.json').exists():
        # Bound transient host RAM per node while other nodes restore in parallel.
        from openwam.train.bootstrap import run_bootstrap_in_waves
        def restore():
            from safetensors import safe_open
            directory,_=_checkpoint(output,identity)
            with safe_open(str(directory/bundle_name),framework='pt',device='cpu') as bundle:
                stored=json.loads(bundle.metadata()[metadata_key])['dependencies']
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
          sampling=order.contract,sampling_sha256=order.sha256,sampling_host_bytes=order.nbytes,
          optimizer_device=optimizer_device,**loading)
    model=expert
    if distributed:
        from torch.nn.parallel import DistributedDataParallel
        model=DistributedDataParallel(expert,device_ids=[device.index] if device.type=='cuda' else None,
            gradient_as_bucket_view=True,find_unused_parameters=True)
    summary=dict(mode='bounded_stage1' if bounded else 'finite_stage1',status='running',identity=identity,scope=dataset.scope_summary(),evaluations=[],
        representation=representation,finite_epoch_complete=False,
        completed_steps=update,steps=max_updates if bounded else math.ceil(len(dataset)/(world*batch_size)),evaluation_split='train',success_verified=False)
    fixed=[];cards=[]
    def setup():
        nonlocal fixed,cards
        fixed=list(dataset.evaluation_samples());cards=_export_cases(output,fixed)
        old=output/'summary.json'
        if resume and old.exists():summary['evaluations']=[e for e in json.loads(old.read_text()).get('evaluations',[]) if e['step']<=update]
        (output/'index.html').write_text('<meta charset="utf-8"><h1>Finite Stage1</h1><a href="timeline.html">All evaluations</a> · <a href="inputs/index.html">Training inputs</a>')
        atomic_json(output/'summary.json',summary)
    if convergence is not None:
        with preserve_rng():primary(setup)
    else:primary(setup)
    gate=None
    def prepared_fixed():
        return (prepare_batch(r,b,device=device,dtype=dtype,context_cache=context_cache) for r,b in fixed)
    def setup_gate():
        fixed_identity=fingerprint(fixed)
        if resume and (output/'latest.json').exists():
            directory,manifest=_checkpoint(output,identity)
            return load_convergence_state(directory,manifest,identity,evaluation_identity=fixed_identity)
        return new_state(convergence,fixed_identity)
    def gate_evaluate():
        metric=fixed_flow_metric(expert,prepared_fixed(),seed=identity['seed']+200000)
        result=observe(gate,update,metric,max_updates=max_updates)
        invalid_reason=None
        if result['history'][-1]['metric'] is None:
            missing=[name for name,count,weight in (('image','valid_images','lambda_image'),('TCP','valid_points','lambda_tcp'))
                     if isinstance(metric,dict) and metric.get(weight,0)>0 and metric.get(count)==0]
            invalid_reason=('no valid '+' or '.join(missing)+' supervision in fixed diagnostic pool' if missing
                            else 'empty, nonfinite, invalid, or changed fixed diagnostic supervision')
        summary['convergence']=result
        summary.update(evaluation_split='fixed_train_population_diagnostic',formal_generalization=False,
                       convergence_last_invalid_reason=invalid_reason)
        atomic_json(output/'summary.json',summary)
        event('convergence_evaluation',step=update,metric=result['history'][-1]['metric'],
              best_loss=result['best_loss'],reference_loss=result['reference_loss'],
              streak=result['streak'],converged=result['converged'],invalid_reason=invalid_reason)
        return result
    last_saved=None
    def save():
        nonlocal last_saved
        extra={'convergence_state':gate} if convergence is not None else {}
        last_saved=save_epoch_checkpoint(expert,optimizer,output,identity=identity,cursor=cursor,
                                     update=update,dependencies=dependencies,_return_receipt=True,**extra)
        return last_saved.directory
    def update_gate():
        nonlocal gate
        try:
            gate=primary(gate_evaluate,broadcast_result=True)
        except RuntimeError:
            # Commit the actual current optimizer and an invalid evidence row;
            # every rank received the same error before entering this checkpoint.
            gate=primary(lambda:observe(gate,update,None,max_updates=max_updates),broadcast_result=True)
            primary(save)
            raise
    if convergence is not None:
        gate=primary(setup_gate,broadcast_result=True)
        if not gate['history']:
            update_gate()
            primary(save)
        summary['convergence']=gate
        summary.update(evaluation_split='fixed_train_population_diagnostic',formal_generalization=False)
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
    loader=GoalBatchLoader(dataset,order,cursor=cursor,rank=rank,world=world,batch_size=batch_size,
        num_workers=dataset_num_workers,prefetch_factor=prefetch_factor,seed=identity['seed'],
        repeat_population=repeat_population,max_batches=max_updates-update if bounded else None,
        sample_cache_size=sample_cache_size)
    started=time.monotonic()
    try:
        while (update<max_updates if bounded else cursor<len(dataset)) and not (gate and gate['converged']):
            update_started=time.monotonic()
            population_cursor=cursor%len(dataset) if repeat_population else cursor
            positions,real_count,weight,consumed=rank_batch(population_cursor,len(dataset),rank,world,batch_size)
            index=order[positions[0]]
            _seed(identity['seed']+index+(cursor//len(dataset))*len(dataset));expert.train_stage_a();optimizer.zero_grad()
            error=None;batch=None;composition=None
            try:
                wait_started=time.monotonic()
                pairs=next(loader)
                loader_wait_s=time.monotonic()-wait_started
                prepare_started=time.monotonic()
                batch=prepare_goal_batch(pairs,device=device,dtype=dtype,context_cache=context_cache)
                if real_count:
                    composition=[dict(family=record.get('family','unknown'),granularity=record.get('granularity','unknown'))
                                 for record,_ in pairs]
                else:
                    # An empty rank participates in DDP without supervising a duplicate.
                    target=batch['targets']
                    batch['targets']=replace(target,image_valid=torch.zeros_like(target.image_valid),
                                             point_valid=torch.zeros_like(target.point_valid))
                batch_prepare_host_s=time.monotonic()-prepare_started
            except Exception as exc:error=f'rank {rank}: {type(exc).__name__}: {exc}'
            prepared=dict(error=error,composition=composition);preparations=[prepared]
            if distributed:
                preparations=[None]*world
                dist.all_gather_object(preparations,prepared)
            errors=[r['error'] for r in preparations if r['error']]
            if any(errors):raise RuntimeError('sample preparation failed: '+str(errors))
            family_counts=dict(Counter(c['family'] for r in preparations for c in (r['composition'] or [])))
            granularity_counts=dict(Counter(c['granularity'] for r in preparations for c in (r['composition'] or [])))
            _seed(identity['seed']+index+(cursor//len(dataset))*len(dataset))
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
            from openwam.model.goal.heads import point_mask
            samples_with_tcp=int(point_mask(batch['targets']).any(dim=1).sum()) if real_count else 0
            values=torch.tensor([*losses.values(),valid_tcp,samples_with_tcp],dtype=torch.float64,device=device)
            if distributed:dist.all_reduce(values)
            reduced=values.tolist();losses=dict(zip(losses,reduced[:-2]))
            cursor+=consumed;update+=1
            del out,batch,pairs
            memory={} if device.type!='cuda' else dict(
                cuda_max_memory_allocated_bytes=torch.cuda.max_memory_allocated(device),
                cuda_max_memory_reserved_bytes=torch.cuda.max_memory_reserved(device))
            event('update',step=update,cursor=cursor,total_samples=len(dataset),gradient_norm=float(norm),elapsed_s=time.monotonic()-started,
                  batch_size=batch_size,global_batch_size=batch_size*world,real_samples=consumed,local_real_samples=real_count,
                  dataset_num_workers=dataset_num_workers,prefetch_factor=prefetch_factor,
                  loader_wait_s=loader_wait_s,batch_prepare_host_s=batch_prepare_host_s,
                  update_host_s=time.monotonic()-update_started,**memory,
                  family_counts=family_counts,granularity_counts=granularity_counts,
                  valid_tcp_points=int(reduced[-2]),samples_with_tcp=int(reduced[-1]),**losses)
            summary.update(completed_steps=update,seen_samples=cursor)
            if rank==0:atomic_json(output/'summary.json',summary)
            final=update==max_updates if bounded else cursor==len(dataset)
            visual_due=update%eval_every==0 or final
            if convergence is None and visual_due:primary(evaluate)
            gate_due=convergence is not None and (update%convergence['eval_every']==0 or final)
            if gate_due:
                update_gate()
            if update%checkpoint_every==0 or final or gate_due:
                primary(save)
                event('checkpoint_saved',step=update,cursor=cursor)
            if convergence is not None and visual_due:primary(evaluate)
    finally:
        loader.close()
    if convergence is not None and not gate['converged']:
        def exhausted():
            summary.update(status='budget_exhausted_without_convergence',convergence=gate,
                           completed_steps=update,seen_samples=cursor)
            atomic_json(output/'summary.json',summary)
            event('budget_exhausted_without_convergence',step=update,cursor=cursor)
        primary(exhausted)
        raise RuntimeError('Stage1 max_updates exhausted without convergence; Stage2 handoff refused')
    expert.freeze()
    def finish():
        directory,manifest=_checkpoint(output,identity,receipt=last_saved)
        checkpoint=directory/bundle_name
        proof=dict(identity=identity,total_samples=len(dataset),seen_samples=cursor,updates=update,
            checkpoint=str(checkpoint),checkpoint_sha256=manifest['files'][bundle_name])
        if bounded:
            proof.update(completion_kind='bounded_updates',finite_epoch_complete=False,completed_steps=update,
                         unique_samples_seen=min(cursor,len(dataset)),representation=representation)
        if convergence is not None:
            proof.update(completion_kind='convergence',convergence=gate,
                         convergence_sha256=manifest['files']['convergence_state.json'])
        # Stable conventional path for inference tools; the proof always names
        # the hashed durable generation rather than this convenience copy.
        _publish_bundle(checkpoint,output/bundle_name,receipt=last_saved)
        summary.update(status='complete',completed_steps=update,seen_samples=cursor,
            finite_epoch_complete=not bounded,completion_kind='convergence' if convergence is not None else 'bounded_updates' if bounded else 'finite_epoch')
        atomic_json(output/'summary.json',summary)
        atomic_json(completion_file,proof)
    primary(finish)
    return primary(lambda:verify_completion(output,identity,_receipt=last_saved),broadcast_result=True)
