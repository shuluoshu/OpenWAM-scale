"""Native causal C* execution helpers; frozen three-arm runner stays untouched."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys
import uuid

import numpy as np
import torch
import torch.distributed as dist

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
import rfc_distributed as shared
base=shared.base

from openwam.model.goal.codec import GoalCodec,GoalCodecSpec
from openwam.model.goal.visual_history import VisualHistoryGoalExpert


class PreparedGoalCodec(GoalCodec):
    """Native head codec with a content-verified preencoded target input."""
    def encode(self,image,*,device=None):
        self.validate_latent(image)
        return image.detach().to(device=device or image.device)


class PreparedVisualHistoryExpert(VisualHistoryGoalExpert):
    """Override only frozen observation encoding; all native model math stays."""
    def encode_observation(self,observed,*,observed_view_valid=None):
        from openwam.model.goal.observation import validate_view_valid
        self.observation_codec.validate_latent(observed)
        validate_view_valid(observed_view_valid,batch_size=len(observed),device=observed.device)
        return observed.detach()


def full_noise(target,*,seed,cursor,world,validation=False):
    derived=int(np.random.SeedSequence([seed,int(cursor),3559 if validation else 3551]).generate_state(1)[0])
    generator=torch.Generator(device='cpu').manual_seed(derived)
    indices=torch.randint(0,1000,(world,),generator=generator,dtype=torch.long)
    image=torch.randn((world,*target.shape[1:]),generator=generator,dtype=torch.float32).to(dtype=target.dtype)
    tcp=torch.randn((world,192),generator=generator,dtype=torch.float32)
    return indices,image,tcp


def target_counts(targets):
    from openwam.model.goal.heads import point_mask
    return int(targets.image_valid.sum()),96*int(point_mask(targets).sum())


def global_loss(result,local,global_counts,world,image_weight,tcp_weight):
    return world*(image_weight*result['fm_loss']*max(local[0],1)/max(global_counts[0],1)
                 +tcp_weight*result['tcp_loss']*max(local[1],1)/max(global_counts[1],1))


class CstarLossAdapter(torch.nn.Module):
    def __init__(self,expert):
        super().__init__();self.expert=expert

    def forward(self,batch,*,timestep_indices,noise,tcp_noise):
        result=self.expert(**batch,timestep_indices=timestep_indices,noise=noise,tcp_noise=tcp_noise)
        local=target_counts(batch['targets']);device=batch['observed'].device
        counts=torch.tensor(local,dtype=torch.long,device=device)
        world=dist.get_world_size() if dist.is_initialized() else 1
        if world>1:dist.all_reduce(counts)
        global_counts=tuple(int(v) for v in counts)
        loss=global_loss(result,local,global_counts,world,self.expert.lambda_image,self.expert.lambda_tcp)
        numerators=torch.stack((result['fm_loss'].detach()*max(local[0],1),result['tcp_loss'].detach()*max(local[1],1)))
        if world>1:dist.all_reduce(numerators)
        return dict(loss=loss,fm_loss=numerators[0]/max(global_counts[0],1),
                    tcp_loss=numerators[1]/max(global_counts[1],1),valid_images=global_counts[0],valid_points=global_counts[1]//96)


def collate(rows,data,*,device,dtype):
    from openwam.model.goal.types import GoalTargets
    def tensor(name):
        value=torch.stack([r[name] for r in rows])
        return value.to(device=device,dtype=dtype if name.endswith('latent') or name=='history_latents' else value.dtype)
    contexts=[]
    for row in rows:
        path=Path(row['context_file']);path=path if path.is_absolute() else Path(data)/path
        contexts.append(torch.load(path,map_location='cpu',weights_only=True))
    lens=torch.tensor([len(t) for t in contexts],device=device,dtype=torch.long)
    if not (lens>0).all():raise ValueError('C* native causal text needs a real instruction context')
    context=torch.zeros(len(rows),int(lens.max()),contexts[0].shape[-1],device=device,dtype=dtype)
    for i,value in enumerate(contexts):context[i,:len(value)]=value.to(device=device,dtype=dtype)
    targets=GoalTargets(image=tensor('target_latent'),image_valid=tensor('target_image_valid'),
        uv=tensor('target_uv').float(),point_valid=tensor('target_point_valid'),in_frame=tensor('target_in_frame'),
        visibility=torch.full((len(rows),2),-1,device=device,dtype=torch.int64))
    return dict(observed=tensor('observed_latent'),context=context,context_lens=lens,targets=targets,
        observed_view_valid=tensor('observed_view_valid'),history_latents=tensor('history_latents'),
        history_ages=tensor('history_ages'),history_valid=tensor('history_valid'),history_episode_id=tensor('history_episode_id'))


def checkpoint(expert,optimizer,output,*,update,contract,dependencies):
    """Native portable v3 weights plus immutable named FP32 optimizer shards."""
    from openwam.train.goal_visual_history import save_visual_history_bundle
    rank,world=dist.get_rank(),dist.get_world_size();output=Path(output)
    chosen=[f'step_{update:06d}_{uuid.uuid4().hex[:12]}' if rank==0 else None]
    dist.broadcast_object_list(chosen,src=0);directory=output/'checkpoints'/chosen[0]
    if rank==0:directory.mkdir(parents=True)
    dist.barrier()
    states=shared.owned_snapshot(expert,optimizer);device=next(expert.parameters()).device
    filename=f'optimizer_rank_{rank:02d}.pt'
    payload=dict(format=shared.STATE_FORMAT,update=update,rank=rank,world=world,contract=shared.canonical(contract),
        states=states,torch_rng=torch.get_rng_state(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),
        cuda_rng=torch.cuda.get_rng_state(device) if device.type=='cuda' else None)
    shared.atomic_torch(directory/filename,payload)
    receipt=dict(rank=rank,file=filename,sha256=base.sha256(directory/filename),bytes=(directory/filename).stat().st_size,
                 names=sorted(states),fingerprints=shared.state_fingerprints(states))
    receipts=[None]*world;dist.all_gather_object(receipts,receipt)
    if rank==0:
        names=[n for item in receipts for n in item['names']]
        if len(names)!=len(set(names)) or set(names)!={n for n,p in expert.named_parameters() if p.requires_grad}:
            raise ValueError('C* optimizer ownership does not cover every trainable parameter exactly once')
        path=directory/'visual_history.safetensors'
        save_visual_history_bundle(expert,path,dependency_identities=dependencies,
            training_metadata=dict(update=update,contract=contract,cached_inputs='independent-native-T1-VAE-latents'))
        manifest=dict(format=shared.STATE_FORMAT,update=update,world=world,contract=contract,shards=receipts,
            weights=dict(file=path.name,sha256=base.sha256(path),bytes=path.stat().st_size))
        base.atomic_json(directory/'manifest.json',manifest)
        base.atomic_json(output/'checkpoint.json',dict(update=update,samples=update*world,generation=str(directory),
            manifest_sha256=base.sha256(directory/'manifest.json'),weights=str(path),resume=str(directory/'manifest.json')))
    dist.barrier();return directory


@torch.no_grad()
def evaluate_record(expert,batch,*,index,seed=42):
    indices,noise,tcp=full_noise(batch['targets'].image,seed=seed,cursor=index,world=1,validation=True)
    result=expert(**batch,timestep_indices=indices.to(batch['observed'].device),noise=noise.to(batch['observed'].device),
                  tcp_noise=tcp.to(batch['observed'].device))
    images,coordinates=target_counts(batch['targets'])
    return dict(fm_sum=float(result['fm_loss'])*max(images,1),image_den=images,
                tcp_fm_sum=float(result['tcp_loss'])*max(coordinates,1),tcp_den=coordinates,
                records=1,timestep_index=int(indices[0]),sigma=float(result['sigma'][0]))


def summarize(sums):
    return dict(metrics=dict(native_head_fm=sums.get('fm_sum',0)/sums['image_den'] if sums.get('image_den') else None,
        native_tcp_patch_fm=sums.get('tcp_fm_sum',0)/sums['tcp_den'] if sums.get('tcp_den') else None),
        counts=dict(valid_images=int(sums.get('image_den',0)),valid_points=int(sums.get('tcp_den',0)//96),
                    records=int(sums.get('records',0))),sums=dict(sums))
