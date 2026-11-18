"""Exact whole-update DDP objectives and bounded atomic native checkpoints."""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import shutil
import tempfile
import time
import uuid
import numpy as np
import torch
import torch.distributed as dist

ROOT=Path(__file__).resolve().parent
REVIEWED=Path('/mnt/data/cpfs/b5/yeqingda/CodeX/experiments/rfc_stage1_compare_8gpu_20261111')
STATE_FORMAT='openwam.stage1_longtrain.zero1.named_shards.v1'


def canonical(value):return json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False)

def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix=path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as f:
            json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)

def atomic_torch(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix=path.name+'.',dir=path.parent);os.close(fd)
    try:
        torch.save(value,name)
        with open(name,'rb') as f:os.fsync(f.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)

def append_jsonl(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:f.write(canonical(value)+'\n');f.flush()

def load_reviewed():
    import sys
    if str(REVIEWED) not in sys.path:sys.path.insert(0,str(REVIEWED))
    import rfc_distributed
    return rfc_distributed

def load_cstar():
    import sys
    directory=str(REVIEWED/'baseline_cstar')
    if directory not in sys.path:sys.path.insert(0,directory)
    import cstar
    return cstar

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_initialized():torch.cuda.manual_seed_all(seed)

@contextmanager
def preserved_rng():
    cpu=torch.get_rng_state();py=random.getstate();npstate=np.random.get_state()
    gpu=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:yield
    finally:
        torch.set_rng_state(cpu);random.setstate(py);np.random.set_state(npstate)
        if gpu is not None:torch.cuda.set_rng_state_all(gpu)

def normalized_objective(terms,local_counts,global_counts,world,image_weight,uv_weight,*,second='uv_loss'):
    """Global counts include every microstep. DDP mean is compensated once.

    Native terms are means over max(local denominator,1); no /accumulation
    is permitted because global_counts already covers the entire update.
    """
    return world*(image_weight*terms['fm_loss']*max(int(local_counts[0]),1)/max(int(global_counts[0]),1)
                 +uv_weight*terms[second]*max(int(local_counts[1]),1)/max(int(global_counts[1]),1))

def batch_counts(batch,arch):
    if arch=='G':
        pixels,points=load_reviewed().base.validity(batch)
        return int(pixels.sum())*batch['target_latents'].shape[1],int(points.sum())
    return load_cstar().target_counts(batch['targets'])

def whole_update_counts(batches,arch,device):
    local=[batch_counts(batch,arch) for batch in batches]
    counts=torch.tensor([sum(c[i] for c in local) for i in (0,1)],device=device,dtype=torch.long)
    if dist.is_initialized():dist.all_reduce(counts)
    return local,tuple(int(v) for v in counts)

def global_noise_slice(target,*,arch,seed,update,rank,world,accumulation,micro):
    """Native RNG distribution; arbitrary B, microstep-major global partition."""
    batch=target.shape[0];total=world*batch*accumulation
    if not 0<=micro<accumulation or not 0<=rank<world:raise ValueError('noise partition differs')
    salt=773 if arch=='G' else 3551
    derived=int(np.random.SeedSequence([seed,int(update),salt]).generate_state(1)[0])
    generator=torch.Generator(device='cpu').manual_seed(derived)
    start=(micro*world+rank)*batch;sl=slice(start,start+batch)
    if arch=='G':
        times=torch.rand(total,generator=generator,dtype=torch.float32)
        noise=torch.randn((total,*target.shape[1:]),generator=generator,dtype=torch.float32)
        return times[sl].to(target.device),noise[sl].to(target)
    times=torch.randint(0,1000,(total,),generator=generator,dtype=torch.long)
    noise=torch.randn((total,*target.shape[1:]),generator=generator,dtype=torch.float32).to(dtype=target.dtype)
    tcp=torch.randn((total,192),generator=generator,dtype=torch.float32)
    return times[sl].to(target.device),noise[sl].to(target),tcp[sl].to(target.device)

class LossAdapter(torch.nn.Module):
    def __init__(self,model,arch):super().__init__();self.model=model;self.arch=arch
    def forward(self,batch,*,local_counts,global_counts,noise,update,schedule_reference_updates):
        world=dist.get_world_size() if dist.is_initialized() else 1
        if self.arch=='G':
            result=self.model.training_loss(**batch,update=update,total_updates=schedule_reference_updates,sigma=noise[0],noise=noise[1])
            image_weight,other_weight=self.model.config.lambda_image,self.model.config.lambda_uv;second='uv_loss'
        else:
            result=self.model(**batch,timestep_indices=noise[0],noise=noise[1],tcp_noise=noise[2])
            image_weight,other_weight=self.model.lambda_image,self.model.lambda_tcp;second='tcp_loss'
        loss=normalized_objective(result,local_counts,global_counts,world,image_weight,other_weight,second=second)
        numerators=torch.stack((result['fm_loss'].detach()*max(local_counts[0],1),result[second].detach()*max(local_counts[1],1)))
        return dict(loss=loss,numerators=numerators)

def reduce_statistics(numerators,counts):
    value=sum(numerators)
    if dist.is_initialized():dist.all_reduce(value)
    return [float(value[i])/max(counts[i],1) for i in (0,1)]

def tensor_digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()

def owned_snapshot(model,optimizer):
    names={id(p):n for n,p in model.named_parameters()}
    return {names[id(p)]:{k:v.detach().cpu().clone() for k,v in state.items()} for p,state in optimizer.local_optimizer.state.items()}

def install_owned_state(model,optimizer,states):
    names={id(p):n for n,p in model.named_parameters()};owned=[]
    for group in optimizer.local_optimizer.param_groups:
        for p in group['params']:
            name=names[id(p)];owned.append(name)
            if name not in states:raise ValueError('missing owned optimizer state: '+name)
            state=states[name]
            if set(state) not in ({'master'},{'master','step','exp_avg','exp_avg_sq'}):raise ValueError('invalid optimizer state keys')
            restored={}
            for key,value in state.items():
                if not torch.is_tensor(value) or not torch.isfinite(value).all():raise ValueError('nonfinite optimizer state: '+name)
                if key=='step':
                    if value.numel()!=1 or float(value)<0:raise ValueError('invalid Adam step')
                    restored[key]=value.detach().cpu().clone()
                else:
                    if value.dtype!=torch.float32 or value.shape!=p.shape:raise ValueError('invalid FP32 optimizer state: '+name)
                    restored[key]=value.to(device=p.device,dtype=torch.float32,copy=True)
            optimizer.local_optimizer.state[p]=restored
            with torch.no_grad():p.copy_(restored['master'])
    if set(owned)!=set(states):raise ValueError('optimizer ownership differs')
    return sorted(owned)

def _rank_world():return (dist.get_rank(),dist.get_world_size()) if dist.is_initialized() else (0,1)
def _barrier():
    if dist.is_initialized():dist.barrier()
def _collect(items):
    if not dist.is_initialized():return [items]
    out=[None]*dist.get_world_size();dist.all_gather_object(out,items);return out

def collective_status(error,device):
    flag=torch.tensor(int(error is None),device=device)
    if dist.is_initialized():dist.all_reduce(flag,op=dist.ReduceOp.MIN)
    if not bool(flag):
        errors=_collect(str(error) if error else None)
        raise RuntimeError('coordinated rank failure: '+canonical(errors))

def _save_native(model,path,arch,dependencies,training):
    if arch=='G':
        from openwam.train.goal_memory_checkpoint import save_memory_bundle
        save_memory_bundle(model,path,dit_config=model.dit_config,dependencies=dependencies,training=training)
    else:
        from openwam.train.goal_visual_history import save_visual_history_bundle
        save_visual_history_bundle(model,path,dependency_identities=dependencies,training_metadata=training)

def checkpoint_generation(model,optimizer,output,*,update,global_cursor,contract,arch,dependencies,retain=2,save_weights=None):
    """Only a complete manifest/pointer is resumable; two committed generations."""
    if retain<1:raise ValueError('retain must be positive')
    rank,world=_rank_world();output=Path(output);device=next(model.parameters()).device
    chosen=[f'step_{update:09d}_{uuid.uuid4().hex[:12]}' if rank==0 else None]
    if world>1:dist.broadcast_object_list(chosen,src=0)
    directory=output/'checkpoints'/chosen[0];error=None
    if rank==0:
        try:
            cleaned=[]
            for old in (output/'checkpoints').glob('step_*'):
                if old.is_dir() and re.fullmatch(r'step_[0-9]{9}_[0-9a-f]{12}',old.name) and not (old/'manifest.json').exists():
                    cleaned.append(dict(generation=old.name,files=[dict(file=str(p.relative_to(old)),bytes=p.stat().st_size) for p in old.rglob('*') if p.is_file()]))
                    shutil.rmtree(old)
            if cleaned:atomic_json(output/'recovery'/f'cleaned_incomplete_before_{update:09d}_{int(time.time())}.json',dict(reason='incomplete immutable generations never committed; retain bounded checkpoint storage',generations=cleaned))
            directory.mkdir(parents=True,exist_ok=False)
            parameter_bytes=sum(p.numel()*4 for p in model.parameters() if p.requires_grad)
            # Weight copy, three FP32 states, and bounded temporary file margin.
            required=parameter_bytes*4+1024**3
            if shutil.disk_usage(directory).free<required:raise OSError(f'insufficient checkpoint space: require {required} free bytes')
        except Exception as exc:error=repr(exc)
    collective_status(error,device);_barrier()
    states=owned_snapshot(model,optimizer);filename=f'optimizer_rank_{rank:02d}.pt';error=None;receipt=None
    try:
        payload=dict(format=STATE_FORMAT,update=update,global_cursor=global_cursor,rank=rank,world=world,contract=canonical(contract),states=states,
                     torch_rng=torch.get_rng_state(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),
                     cuda_rng=torch.cuda.get_rng_state(device) if device.type=='cuda' else None)
        atomic_torch(directory/filename,payload)
        receipt=dict(rank=rank,file=filename,bytes=(directory/filename).stat().st_size,sha256=sha256(directory/filename),names=sorted(states))
    except Exception as exc:error=repr(exc)
    collective_status(error,device);receipts=_collect(receipt);del states,payload
    error=None
    if rank==0:
        try:
            names=[n for r in receipts for n in r['names']]
            if len(names)!=len(set(names)) or set(names)!={n for n,p in model.named_parameters() if p.requires_grad}:raise ValueError('optimizer ownership does not cover trainable parameters once')
            weights=directory/('goal_memory.safetensors' if arch=='G' else 'visual_history.safetensors')
            if save_weights:save_weights(model,weights)
            else:_save_native(model,weights,arch,dependencies,dict(update=update,global_cursor=global_cursor,contract=contract))
            manifest=dict(format=STATE_FORMAT,update=update,global_cursor=global_cursor,world=world,arch=arch,contract=contract,shards=receipts,
                          weights=dict(file=weights.name,sha256=sha256(weights),bytes=weights.stat().st_size))
            atomic_json(directory/'manifest.json',manifest)
            atomic_json(output/'checkpoint.json',dict(update=update,samples=global_cursor,generation=str(directory),manifest_sha256=sha256(directory/'manifest.json'),weights=str(weights),resume=str(directory/'manifest.json')))
            committed=sorted([p for p in (output/'checkpoints').glob('step_*') if (p/'manifest.json').is_file()],key=lambda p:json.loads((p/'manifest.json').read_text())['update'])
            for old in committed[:-retain]:shutil.rmtree(old)
        except Exception as exc:error=repr(exc)
    collective_status(error,device);_barrier();return directory

def restore_generation(model,optimizer,resume,*,contract,load_weights=None):
    """Every integrity stage rejects collectively before the next collective."""
    rank,world=_rank_world();device=next(model.parameters()).device
    error=None;manifest=None;directory=None
    try:
        path=Path(resume)
        if path.is_dir():path=path/'manifest.json'
        pointer=json.loads(path.read_text())
        if 'generation' in pointer:
            path=Path(pointer['generation'])/'manifest.json'
            if sha256(path)!=pointer['manifest_sha256']:raise ValueError('checkpoint manifest SHA256 differs')
        manifest=json.loads(path.read_text());directory=path.parent
        if manifest['format']!=STATE_FORMAT or manifest['world']!=world or canonical(manifest['contract'])!=canonical(contract):raise ValueError('resume contract/world differs')
        payload_contract = canonical(contract)
        if 'optimizer_contract' in manifest:
            from openwam.train.goal_memory_kf1_migration import optimizer_contract_for_restore
            payload_contract = optimizer_contract_for_restore(manifest, contract)
        weights=directory/manifest['weights']['file']
        if weights.stat().st_size!=manifest['weights']['bytes'] or (rank==0 and sha256(weights)!=manifest['weights']['sha256']):raise ValueError('portable weights checksum differs')
    except Exception as exc:error=repr(exc)
    collective_status(error,device)
    error=None
    try:
        if load_weights:load_weights(model,weights)
        else:
            from safetensors import safe_open
            with safe_open(str(weights),framework='pt',device='cpu') as src:
                current=model.state_dict()
                if set(src.keys())!=set(current):raise ValueError('model keys differ')
                with torch.no_grad():
                    for name,value in current.items():
                        saved=src.get_tensor(name)
                        if saved.shape!=value.shape or saved.dtype!=value.dtype:raise ValueError('model tensor differs: '+name)
                        value.copy_(saved)
    except Exception as exc:error=repr(exc)
    collective_status(error,device)
    error=None
    try:
        receipt=next(s for s in manifest['shards'] if s['rank']==rank);shard=directory/receipt['file']
        if shard.stat().st_size!=receipt['bytes'] or sha256(shard)!=receipt['sha256']:raise ValueError('optimizer shard checksum differs')
        payload=torch.load(shard,map_location='cpu',mmap=True,weights_only=False)
        for key,expected in dict(format=STATE_FORMAT,update=manifest['update'],global_cursor=manifest['global_cursor'],rank=rank,world=world,contract=payload_contract).items():
            if payload[key]!=expected:raise ValueError('optimizer shard identity differs: '+key)
        current=model.state_dict()
        for name,state in payload['states'].items():
            if name not in current or not torch.equal(state['master'].to(current[name].dtype),current[name].cpu()):raise ValueError('model/FP32 master differs: '+name)
            if 'step' in state and float(state['step'])>manifest['update']:raise ValueError('Adam step exceeds update')
        if install_owned_state(model,optimizer,payload['states'])!=receipt['names']:raise ValueError('optimizer shard ownership differs')
    except Exception as exc:error=repr(exc)
    collective_status(error,device)
    torch.set_rng_state(payload['torch_rng']);random.setstate(payload['python_rng']);np.random.set_state(payload['numpy_rng'])
    if payload['cuda_rng'] is not None:torch.cuda.set_rng_state(payload['cuda_rng'],device)
    return manifest['update'],dict(global_cursor=manifest['global_cursor'],manifest=str(path))
