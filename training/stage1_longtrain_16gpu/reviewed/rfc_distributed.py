"""Isolated distributed execution helpers for the matched RFC comparison.

Model/source files remain unmodified. Only execution, objective normalization
and checkpoint storage are adapted to one process per GPU.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import time
import uuid

import numpy as np
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('rfc_prepared_v1', ROOT/'frozen_v1/rfc_compare.py')
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
STATE_FORMAT = 'openwam.rfc_stage1.zero1.named_shards.v1'


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def normalized_objective(terms, local_counts, global_counts, world, image_weight, uv_weight):
    """Undo native local means, divide by global counts, undo DDP mean.

    Native FM divides by max(values,1); UV divides by 2*max(points,1).
    No numerator is detached, so averaging DDP gradients recovers the exact
    valid-element-normalized global-batch objective, including empty ranks.
    """
    fm_scale = max(int(local_counts[0]), 1)/max(int(global_counts[0]), 1)
    uv_scale = max(int(local_counts[1]), 1)/max(int(global_counts[1]), 1)
    return world*(image_weight*terms['fm_loss']*fm_scale + uv_weight*terms['uv_loss']*uv_scale)


def rank_noise(target, *, seed, cursor, rank, world):
    full = torch.empty((world, *target.shape[1:]), dtype=target.dtype, device='cpu')
    sigma, noise = base.common_noise(full, seed=seed, cursor=cursor, validation=False)
    return sigma[rank:rank+1].to(target.device), noise[rank:rank+1].to(target)


def configure_no_text(model):
    """Remove every instruction edge without invalid all-masked attention.

    The native preparation still runs with one constant zero context token;
    all G/Y text-query gates are then zeroed before any transformer block.
    Parameters, architecture and state_dict keys remain source-native.
    """
    original = model.prepare_state
    def prepare(current_latents,memory_latents,noisy_target_latents,timestep,text,**kwargs):
        text = torch.zeros(text.shape[0],1,text.shape[-1],device=text.device,dtype=text.dtype)
        kwargs.update(text_mask=torch.ones(text.shape[:2],device=text.device,dtype=torch.bool),warmup_y_text=False)
        state = original(current_latents,memory_latents,noisy_target_latents,timestep,text,**kwargs)
        state.text_query_gate.zero_()
        return state
    model.prepare_state = prepare


def load_portable(path,*,dependencies,device='cpu',dtype=torch.float32):
    """Load a native model bundle and reapply the recorded no_text intervention."""
    from openwam.train.goal_memory_checkpoint import load_memory_bundle
    model=load_memory_bundle(path,dependencies=dependencies,device=device,dtype=dtype)
    contract=model.bundle_metadata.get('training',{}).get('contract',{})
    if contract.get('scientific',{}).get('comparison',{}).get('arm')=='no_text':
        configure_no_text(model)
    return model


class DistributedLossAdapter(torch.nn.Module):
    def __init__(self, model, arm='default'):
        super().__init__()
        self.model = model
        self.arm = arm

    def forward(self, batch, *, update, total_updates, sigma, noise):
        result = self.model.training_loss(**batch, update=update, total_updates=total_updates, sigma=sigma, noise=noise)
        pixels, points = base.validity(batch)
        local = (int(pixels.sum())*batch['target_latents'].shape[1], int(points.sum()))
        counts = torch.tensor(local, dtype=torch.long, device=batch['target_latents'].device)
        world = dist.get_world_size() if dist.is_initialized() else 1
        if world > 1:
            dist.all_reduce(counts)
        global_counts = tuple(int(v) for v in counts)
        result['loss'] = normalized_objective(result, local, global_counts, world,
                                             self.model.config.lambda_image, self.model.config.lambda_uv)
        # Detached logging values are a separate reduction of true numerators.
        statistics = torch.stack((result['fm_loss'].detach()*max(local[0],1),
                                  result['uv_loss'].detach()*(2*max(local[1],1))))
        if world > 1:
            dist.all_reduce(statistics)
        fm = statistics[0]/max(global_counts[0],1)
        uv = statistics[1]/(2*max(global_counts[1],1))
        # Return just the used differentiable objective. Returning unused native
        # auxiliary tensors confuses DDP's unused-parameter output traversal.
        return dict(loss=result['loss'], fm_loss=fm, uv_loss=uv,
                    valid_latent_values=global_counts[0], valid_points=global_counts[1])


def parameter_names(model, groups):
    names = {id(p):name for name,p in model.named_parameters()}
    return [[names[id(p)] for p in group['params']] for group in groups]


def legacy_parameter_state(model, groups, payload, *, betas, eps, weight_decay, validate_names=None):
    """Map trusted legacy FP32 masters/moments through exact global group order."""
    saved_groups = payload['optimizer']['param_groups']
    if len(saved_groups) != len(groups):
        raise ValueError('legacy optimizer parameter-group count differs')
    names = parameter_names(model, groups)
    masters = payload.get('masters')
    if masters is None or len(masters) != sum(map(len,names)):
        raise ValueError('legacy FP32 master count differs')
    mapped = {}; cursor = 0
    for group, saved, group_names in zip(groups, saved_groups, names):
        if len(saved['params']) != len(group_names):
            raise ValueError('legacy parameter group order/count differs')
        expected = dict(lr=group['lr'], betas=tuple(betas), eps=eps, weight_decay=weight_decay)
        for key,value in expected.items():
            actual = tuple(saved[key]) if key == 'betas' else saved[key]
            if actual != value:
                raise ValueError('legacy optimizer hyperparameter differs: '+key)
        for parameter, name, index in zip(group['params'], group_names, saved['params']):
            master = masters[cursor]; cursor += 1
            state = payload['optimizer']['state'].get(index)
            if state is None or not all(k in state for k in ('step','exp_avg','exp_avg_sq')):
                raise ValueError('legacy optimizer state is incomplete: '+name)
            if validate_names is not None and name not in validate_names:
                continue
            step=state['step']
            if not torch.is_tensor(step) or step.numel()!=1 or not torch.isfinite(step).all() or float(step)<0 or (payload.get('update') is not None and float(step)>payload['update']):
                raise ValueError('legacy optimizer step is invalid: '+name)
            for value in (master,state['exp_avg'],state['exp_avg_sq']):
                if value.dtype != torch.float32 or value.shape != parameter.shape or not torch.isfinite(value).all():
                    raise ValueError('legacy FP32 state dtype/shape/finite differs: '+name)
            if not torch.equal(master.to(dtype=payload['model'][name].dtype), payload['model'][name]):
                raise ValueError('legacy master/model integrity differs: '+name)
            mapped[name] = dict(master=master, **state)
    return mapped


def import_legacy(model,optimizer,groups,path,*,legacy_contract,output,betas,eps,weight_decay):
    """Explicit world-one CPU-master to world-eight GPU-shard migration."""
    path=Path(path); before=path.stat()
    payload=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
    if payload.get('format')!='openwam.goal_memory.training.v1' or payload.get('contract')!=canonical(legacy_contract):
        raise ValueError('legacy scientific contract differs (data, mixture, schedule, model or arm)')
    if payload['update'] != 500:
        raise ValueError('authorized legacy migration requires update 500')
    current=model.state_dict()
    if current.keys()!=payload['model'].keys() or any(p.shape!=payload['model'][n].shape or p.dtype!=payload['model'][n].dtype for n,p in current.items()):
        raise ValueError('legacy architecture/model dtype differs')
    names={id(p):name for name,p in model.named_parameters()}
    owned={names[id(p)] for group in optimizer.local_optimizer.param_groups for p in group['params']}
    states=legacy_parameter_state(model,groups,payload,betas=betas,eps=eps,weight_decay=weight_decay,validate_names=owned)
    model.load_state_dict(payload['model'],strict=True)
    install_owned_state(model,optimizer,states)
    source_fingerprints=state_fingerprints(states)
    destination_states={names[id(p)]:state for p,state in optimizer.local_optimizer.state.items()}
    destination_fingerprints=state_fingerprints(destination_states)
    source_weights={name:tensor_digest(payload['model'][name]) for name in owned}
    destination_weights={name:tensor_digest(dict(model.named_parameters())[name]) for name in owned}
    if source_fingerprints!=destination_fingerprints or source_weights!=destination_weights:
        raise ValueError('legacy migration source/destination precision integrity differs')
    if (path.stat().st_size,path.stat().st_mtime_ns)!=(before.st_size,before.st_mtime_ns):
        raise ValueError('legacy checkpoint changed during migration')
    receipt=dict(format='openwam.rfc_stage1.execution_migration.v1',source=str(path),source_bytes=before.st_size,
        source_mtime_ns=before.st_mtime_ns,update=500,rank=dist.get_rank(),world=dist.get_world_size(),
        old_execution=payload['execution'],new_execution=dict(optimizer='native-zero1-gpu-fp32',world=dist.get_world_size(),
        torch=torch.__version__,device=str(next(model.parameters()).device)),scientific_contract=legacy_contract,
        owned_state_fingerprints=source_fingerprints,destination_state_fingerprints=destination_fingerprints,
        owned_weight_fingerprints=source_weights,destination_weight_fingerprints=destination_weights,
        stochastic_execution='Per-update and rank branch RNG; global-eight sigma/noise and draw cursor unchanged. Reduction/branch RNG execution changes, so continuation is not bit-exact to world one.')
    base.atomic_json(Path(output)/f'legacy_migration_rank_{dist.get_rank():02d}.json',receipt)
    receipts=[None]*dist.get_world_size(); dist.all_gather_object(receipts,dict(rank=dist.get_rank(),names=sorted(owned)))
    if dist.get_rank()==0:
        all_names=[name for item in receipts for name in item['names']]
        expected={name for name,p in model.named_parameters() if p.requires_grad}
        if len(all_names)!=len(set(all_names)) or set(all_names)!=expected:
            raise ValueError('legacy migration ownership does not exactly cover full model')
        base.atomic_json(Path(output)/'legacy_migration.json',dict(source=str(path),source_sha256=base.sha256(path),
            source_bytes=before.st_size,source_mtime_ns=before.st_mtime_ns,update=500,world=dist.get_world_size(),
            shard_receipts=[f'legacy_migration_rank_{i:02d}.json' for i in range(dist.get_world_size())],coverage=receipts))
    dist.barrier()
    return 500


def install_owned_state(model, optimizer, states):
    """Restore directly; Optimizer.load_state_dict would round FP32 via BF16."""
    names = {id(p):name for name,p in model.named_parameters()}
    local = optimizer.local_optimizer
    owned = {names[id(p)] for group in local.param_groups for p in group['params']}
    if not owned <= states.keys():
        raise ValueError('missing owned optimizer parameter states')
    for group in local.param_groups:
        for p in group['params']:
            name = names[id(p)]; entry = states[name]
            if not all(k in entry for k in ('master','step','exp_avg','exp_avg_sq')):
                raise ValueError('incomplete optimizer state: '+name)
            restored = {}
            for key,value in entry.items():
                if key in ('master','exp_avg','exp_avg_sq'):
                    if value.dtype != torch.float32 or value.shape != p.shape or not torch.isfinite(value).all():
                        raise ValueError('invalid FP32 optimizer state: '+name)
                    restored[key] = value.to(device=p.device,dtype=torch.float32,copy=True)
                elif key == 'step':
                    if not torch.is_tensor(value) or value.numel()!=1 or not torch.isfinite(value).all() or float(value)<0:
                        raise ValueError('invalid optimizer step: '+name)
                    restored[key] = value.detach().to(device='cpu',copy=True) if torch.is_tensor(value) else torch.tensor(float(value))
                else:
                    raise ValueError('unsupported optimizer state key: '+key)
            local.state[p] = restored
            with torch.no_grad():
                p.copy_(restored['master'])
    return sorted(owned)


def tensor_digest(tensor):
    raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
    return hashlib.sha256(raw.tobytes()).hexdigest()


def owned_snapshot(model, optimizer):
    names = {id(p):name for name,p in model.named_parameters()}
    return {names[id(p)]:{key:(v.detach().cpu().clone() if torch.is_tensor(v) else v)
                          for key,v in state.items()} for p,state in optimizer.local_optimizer.state.items()}


def state_fingerprints(states):
    return {name:{key:tensor_digest(value) for key,value in state.items() if torch.is_tensor(value)}
            for name,state in states.items()}


def atomic_torch(path, payload):
    from openwam.train.goal_memory_checkpoint import _atomic
    _atomic(path, lambda filename:torch.save(payload,filename))


def checkpoint_generation(model, optimizer, output, *, update, contract, dependencies):
    """Commit a complete immutable generation only after all rank writes finish."""
    from openwam.train.goal_memory_checkpoint import save_memory_bundle
    rank,world = dist.get_rank(),dist.get_world_size()
    output = Path(output)
    chosen=[f'step_{update:06d}_{uuid.uuid4().hex[:12]}' if rank==0 else None]
    dist.broadcast_object_list(chosen,src=0)
    generation = output/'checkpoints'/chosen[0]
    if rank == 0:
        if generation.exists():
            raise ValueError('checkpoint generation already exists: '+str(generation))
        generation.mkdir(parents=True)
    dist.barrier()
    states = owned_snapshot(model,optimizer)
    filename = f'optimizer_rank_{rank:02d}.pt'
    device = next(model.parameters()).device
    payload = dict(format=STATE_FORMAT,update=update,rank=rank,world=world,contract=canonical(contract),
                   states=states, torch_rng=torch.get_rng_state(),python_rng=random.getstate(),
                   numpy_rng=np.random.get_state(),cuda_rng=torch.cuda.get_rng_state(device) if device.type=='cuda' else None)
    atomic_torch(generation/filename,payload)
    receipt = dict(rank=rank,file=filename,sha256=base.sha256(generation/filename),bytes=(generation/filename).stat().st_size,
                   names=sorted(states),fingerprints=state_fingerprints(states))
    receipts = [None]*world
    dist.all_gather_object(receipts,receipt)
    if rank == 0:
        save_memory_bundle(model,generation/'goal_memory.safetensors',dit_config=model.dit_config,
                           dependencies=dependencies,training=dict(update=update,contract=contract))
        names = [name for item in receipts for name in item['names']]
        expected = {name for name,p in model.named_parameters() if p.requires_grad}
        if len(names)!=len(set(names)) or set(names)!=expected:
            raise ValueError('optimizer shards do not cover every trainable parameter exactly once')
        weights = generation/'goal_memory.safetensors'
        manifest = dict(format=STATE_FORMAT,update=update,world=world,contract=contract,shards=receipts,
                        weights=dict(file=weights.name,sha256=base.sha256(weights),bytes=weights.stat().st_size))
        base.atomic_json(generation/'manifest.json',manifest)
        base.atomic_json(output/'checkpoint.json',dict(update=update,samples=update*world,
                         generation=str(generation),manifest_sha256=base.sha256(generation/'manifest.json'),
                         weights=str(weights),resume=str(generation/'manifest.json')))
    dist.barrier()
    return generation


def restore_generation(model, optimizer, resume, *, contract):
    from safetensors import safe_open
    rank,world = dist.get_rank(),dist.get_world_size()
    path = Path(resume)
    if path.is_dir(): path = path/'manifest.json'
    pointer = json.loads(path.read_text())
    if 'generation' in pointer:
        path = Path(pointer['generation'])/'manifest.json'
        if base.sha256(path) != pointer['manifest_sha256']:
            raise ValueError('checkpoint manifest SHA256 differs')
    manifest = json.loads(path.read_text()); directory=path.parent
    if manifest['format']!=STATE_FORMAT or manifest['world']!=world or canonical(manifest['contract'])!=canonical(contract):
        raise ValueError('distributed resume contract/world differs')
    weights = directory/manifest['weights']['file']
    # Rank zero verifies the large shared file; per-rank shard reads are independent.
    if rank == 0 and base.sha256(weights)!=manifest['weights']['sha256']:
        raise ValueError('portable checkpoint weights SHA256 differs')
    dist.barrier()
    with safe_open(str(weights),framework='pt',device='cpu') as source:
        current=model.state_dict()
        if set(source.keys())!=set(current): raise ValueError('resume model keys differ')
        with torch.no_grad():
            for name,p in current.items():
                saved=source.get_tensor(name)
                if saved.shape!=p.shape or saved.dtype!=p.dtype: raise ValueError('resume model tensor differs: '+name)
                p.copy_(saved)
    receipt = next(s for s in manifest['shards'] if s['rank']==rank)
    shard = directory/receipt['file']
    if shard.stat().st_size!=receipt['bytes'] or base.sha256(shard)!=receipt['sha256']:
        raise ValueError('optimizer shard checksum differs')
    payload=torch.load(shard,map_location='cpu',mmap=True,weights_only=False)
    if payload['format']!=STATE_FORMAT or payload['update']!=manifest['update'] or payload['rank']!=rank or payload['world']!=world or payload['contract']!=canonical(contract):
        raise ValueError('optimizer shard contract differs')
    for name,state in payload['states'].items():
        if name not in current or not torch.equal(state['master'].to(dtype=current[name].dtype),current[name].cpu()):
            raise ValueError('portable weights/FP32 master integrity differs: '+name)
        if float(state['step'])>manifest['update']:
            raise ValueError('optimizer step exceeds committed update')
    owned=install_owned_state(model,optimizer,payload['states'])
    if owned!=receipt['names'] or state_fingerprints(payload['states'])!=receipt['fingerprints']:
        raise ValueError('optimizer shard ownership/content differs')
    torch.set_rng_state(payload['torch_rng']); random.setstate(payload['python_rng']); np.random.set_state(payload['numpy_rng'])
    device=next(model.parameters()).device
    if payload['cuda_rng'] is not None:
        if device.type!='cuda': raise ValueError('CUDA RNG execution differs')
        torch.cuda.set_rng_state(payload['cuda_rng'],device)
    return manifest['update']
