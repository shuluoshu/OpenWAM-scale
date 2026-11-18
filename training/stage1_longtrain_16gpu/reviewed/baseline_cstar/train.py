#!/usr/bin/env python3
"""Eight-GPU fresh native causal C* matched fourth-arm training."""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
import rfc_distributed as shared
base=shared.base


def identity():
    return {name:base.sha256(ROOT/name) for name in ('train.py','cstar.py','prepare.py','report.py')}


def load_manifest(data,*,verify=True):
    data=Path(data);m=json.loads((data/'manifest.json').read_text())
    if m['format']!='openwam.causal_cstar.prepared.v1':raise ValueError('wrong C* prepared format')
    if m['parent_manifest_sha256']!='987b385a32586300e4c1ad6f49e0d1d5a49d7cbfaa0a3917992e74b983e10704':
        raise ValueError('C* parent comparison identity differs')
    if m.get('seed')!=42 or m.get('dtype')!='bfloat16' or {k:len(v) for k,v in m['records'].items()}!={'train':256,'val':64}:
        raise ValueError('C* prepared seed/dtype/population counts differ')
    if verify:
        for name,digest in {**m['files'],**m.get('external_files',{})}.items():
            path=Path(name);path=path if path.is_absolute() else data/path
            if base.sha256(path)!=digest:raise ValueError('C* prepared SHA256 mismatch: '+str(path))
    return m


def visualize(expert,data,manifest,config,output,step,*,device):
    import cstar as c
    from openwam.model.goal.codec import GoalCodec
    from openwam.model.goal.dependencies import module_identity
    from openwam.model.goal.heads import point_mask
    from PIL import Image,ImageDraw
    started=time.monotonic();entries=manifest['records']['val'];selected=[]
    for grain in ('subtask','task'):
        selected.append(next(i for i,e in enumerate(entries) if e['granularity']==grain and i not in selected))
    selected += [i for i in range(len(entries)) if i not in selected];selected=selected[:4]
    vae=base.construct_component(config,'vae',device=device,dtype=torch.bfloat16).eval().requires_grad_(False)
    if module_identity(vae)!=manifest['dependencies']['vae']:raise ValueError('C* visual VAE identity differs')
    codec=GoalCodec(vae);directory=Path(output)/'visuals'/f'step_{step:06d}';directory.mkdir(parents=True,exist_ok=True)
    total=defaultdict(float);records=[];cards=[]
    for index in selected:
        entry=entries[index];row=torch.load(Path(data)/entry['file'],map_location='cpu',weights_only=False)
        batch=c.collate([row],data,device=device,dtype=torch.bfloat16)
        args={k:v for k,v in batch.items() if k!='targets'}
        generated=expert.generate(**args,num_steps=20,seed=42+index+1000000)
        decoded=(codec.decode(generated.latent)[0].detach().float().cpu().clamp(-1,1)+1)/2
        target=row['raw_target_rgb'].float()/255;valid=point_mask(batch['targets'])[0].cpu()
        uv=generated.uv[0].detach().float().cpu();truth=row['target_uv'].float()
        points=int(valid.sum());tcp_sum=float(((uv[valid]-truth[valid])*torch.tensor([319.,255.])).norm(dim=-1).sum()) if points else 0.
        rgb_den=3*256*320 if bool(row['target_image_valid']) else 0
        rgb_sum=float((decoded-target).abs().sum()) if rgb_den else 0.
        sums=dict(points=points,tcp_sum=tcp_sum,rgb_den=rgb_den,rgb_sum=rgb_sum,records=1)
        for key,value in sums.items():total[key]+=value
        def pil(value):return Image.fromarray(value[:,0].permute(1,2,0).contiguous().numpy())
        current=pil(row['raw_current_rgb'][:,:,:256]);gt=pil(row['raw_target_rgb']);prediction=pil((decoded*255).round().byte())
        for image,coords in ((gt,truth),(prediction,uv)):
            draw=ImageDraw.Draw(image)
            for hand,color in ((0,'cyan'),(1,'magenta')):
                if valid[hand]:
                    x,y=(coords[hand]*torch.tensor([319.,255.])).tolist();draw.ellipse((x-4,y-4,x+4,y+4),outline=color,width=2)
        canvas=Image.new('RGB',(960,296),'white');draw=ImageDraw.Draw(canvas)
        draw.text((5,5),f'causal_cstar update{step}: current head | original terminal head | native joint20-step image/TCP',fill='black')
        draw.text((5,21),f'{entry["id"]} {entry["granularity"]}',fill='black')
        for x,image in ((0,current),(320,gt),(640,prediction)):canvas.paste(image,(x,40))
        filename=f'{index:04d}.png';canvas.save(directory/filename)
        record=dict(index=index,id=entry['id'],family=entry['family'],grain=entry['granularity'],
                    protocol='native-joint20-step',**base.summarize_sums(sums));records.append(record)
        cards.append(f'<article><h3>{entry["id"]}</h3><img width="960" src="{filename}"></article>')
    result=dict(update=step,arm='causal_cstar',protocol='native-joint20-step',steps=20,records=records,
                seconds=time.monotonic()-started,**base.summarize_sums(total))
    base.atomic_json(directory/'metrics.json',result);base.append_jsonl(Path(output)/'visual.jsonl',result)
    base.atomic_text(directory/'index.html','<!doctype html><meta charset="utf-8"><h1>causal_cstar original head RGB and native20-step predictions</h1>'+''.join(cards))
    del vae,codec;torch.cuda.empty_cache()


def train(args):
    if int(os.environ.get('WORLD_SIZE','1'))!=8:raise ValueError('C* production needs8-rank torchrun')
    if args.updates!=1000 or args.seed!=42:raise ValueError('C* matched budget/seed must be1000/42')
    rank=int(os.environ['RANK']);local_rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local_rank)
    device=torch.device('cuda',local_rank);dist.init_process_group('nccl',timeout=timedelta(minutes=60))
    try:
        code=base.bind_code(args.code_dir)
        import cstar as c
        from openwam.model.video_backbone.wan.models.dit import WanModel
        from openwam.train.goal_curated import goal_parameter_groups
        from openwam.train.goal_optimizer import ShardedFP32MasterAdamW
        data=Path(args.data);manifest=load_manifest(data,verify=rank==0)
        parent=Path(args.parent_data);parent_manifest=base.load_manifest(parent,verify=False)
        if base.sha256(parent/'manifest.json')!=manifest['parent_manifest_sha256'] or code['python_tree_sha256']!=parent_manifest['code']['python_tree_sha256']:
            raise ValueError('C* pinned source/parent content differs')
        config=parent_manifest['config']
        if rank==0 and base.sha256(config['baseline_checkpoint'])!=parent_manifest['baseline']['sha256']:
            raise ValueError('C* foundation checkpoint differs')
        dist.barrier();base.seed_all(42)
        dit=base.construct_component(config,'dit',device='cpu',dtype=torch.bfloat16)
        options=dict(num_history_tokens=32,pool_dim=256,pool_heads=8,adapter_block=0,
                     language_conditioning=True,causal_history=True)
        expert=c.PreparedVisualHistoryExpert(dit,c.PreparedGoalCodec(None),dit_config=config['components']['dit']['extra_kwargs'],
            shift=5.,lambda_image=.5,lambda_tcp=1.,use_gradient_checkpointing=True,**options).to(device=device,dtype=torch.bfloat16)
        if expert.dit_config.get('num_layers')!=30 or expert.dit_config.get('dim')!=3072:raise ValueError('full native Wan model required')
        expert.train_stage_a()
        model=DDP(c.CstarLossAdapter(expert),device_ids=[local_rank],output_device=local_rank,
                  find_unused_parameters=True,gradient_as_bucket_view=True,broadcast_buffers=False,bucket_cap_mb=50)
        groups=goal_parameter_groups(expert,learning_rate=1e-5,history_learning_rate=1e-4)
        optimizer=ShardedFP32MasterAdamW(groups,betas=(.9,.999),eps=1e-8,weight_decay=.01)
        entries=manifest['records']['train'];stream=base.sample_stream(entries,8000,seed=42)
        parent_stream=base.sample_stream(parent_manifest['records']['train'],8000,seed=42)
        if [entries[i]['id'] for i in stream]!=[parent_manifest['records']['train'][i]['id'] for i in parent_stream]:
            raise ValueError('C* finite draw identities differ from G arms')
        contract=dict(arm='causal_cstar',seed=42,updates=1000,world=8,local_batch=1,dtype='bfloat16',
            torch=torch.__version__,source=code,runner=identity(),prepared_manifest_sha256=base.sha256(data/'manifest.json'),
            parent_manifest_sha256=manifest['parent_manifest_sha256'],stream_sha256=hashlib.sha256(np.asarray(stream,dtype='<i8').tobytes()).hexdigest(),
            model=dict(dit=expert.dit_config,history=options,format='openwam.visual_history.v3'),
            objective=dict(shift=5.,lambda_image=.5,lambda_tcp=1.,scheduler='native-uniform-shifted1000-weighted-grid',
                normalization='global-valid-image-count-and96-times-valid-point-count',noise='CPU-SeedSequence3551/3559-global8-v1'),
            optimizer=dict(type='native-zero1-gpu-FP32',betas=[.9,.999],eps=1e-8,weight_decay=.01,
                groups=[dict(name=g['name'],lr=g['lr']) for g in groups]),
            evaluation=dict(every=100,checkpoint_every=500,visual_updates=[0,100,500,1000],visual_count=4,sampling_steps=20))
        output=Path(args.output)
        if rank==0:
            if not args.resume and output.exists() and any(output.iterdir()):raise ValueError('C* output nonempty; use fresh directory')
            output.mkdir(parents=True,exist_ok=True);base.atomic_json(output/'contract.json',contract)
        dist.barrier();update=shared.restore_generation(expert,optimizer,args.resume,contract=contract) if args.resume else 0
        rows=base.read_jsonl(output/'train.jsonl') if rank==0 else []
        if rows and rows[-1]['update']>update:raise ValueError('C* logs exceed checkpoint; resume into fresh output')
        offset=rows[-1].get('wall_seconds',0.) if rows else 0.;clock=time.monotonic()
        def assess(step):
            with base.preserved_rng(),torch.no_grad():
                expert.eval();records=[];start=time.monotonic()
                for index in range(rank,len(manifest['records']['val']),8):
                    entry=manifest['records']['val'][index]
                    row=torch.load(data/entry['file'],map_location='cpu',weights_only=False)
                    batch=c.collate([row],data,device=device,dtype=torch.bfloat16)
                    stats=c.evaluate_record(expert,batch,index=index)
                    records.append(dict(index=index,id=entry['id'],grain=entry['granularity'],family=entry['family'],sums=stats))
                collected=[None]*8;dist.all_gather_object(collected,records)
                if rank==0:
                    records=sorted([r for part in collected for r in part],key=lambda r:r['index'])
                    sums=defaultdict(float);groupsums=defaultdict(lambda:defaultdict(float))
                    for r in records:
                        for k in ('fm_sum','image_den','tcp_fm_sum','tcp_den','records'):
                            sums[k]+=r['sums'][k]
                            for group in ('grain:'+r['grain'],'family:'+r['family']):groupsums[group][k]+=r['sums'][k]
                    result=dict(update=step,samples=step*8,wall_seconds=offset+time.monotonic()-clock,arm='causal_cstar',
                        protocol='native-fixed-noise-one-step-weighted-FM; objective-only, separate from generated metrics',
                        groups={k:c.summarize(v) for k,v in groupsums.items()},records=records,**c.summarize(sums))
                    base.append_jsonl(output/'eval.jsonl',result)
                    base.append_jsonl(output/'phases.jsonl',dict(update=step,phase='evaluation',seconds=time.monotonic()-start))
                    if step in {0,100,500,1000}:
                        start=time.monotonic();visualize(expert,data,manifest,config,output,step,device=device)
                        base.append_jsonl(output/'phases.jsonl',dict(update=step,phase='visualization',seconds=time.monotonic()-start))
                    from report import report
                    report(args.runs_root,Path(args.runs_root)/'comparison_report')
                dist.barrier();expert.train_stage_a()
        if not args.resume:assess(0)
        end=min(1000,update+args.stop_after) if args.stop_after else 1000
        while update<end:
            started=time.monotonic();indices=stream[update*8:(update+1)*8]
            row=torch.load(data/entries[indices[rank]]['file'],map_location='cpu',weights_only=False)
            batch=c.collate([row],data,device=device,dtype=torch.bfloat16)
            base.seed_all(42+100003*(update+1)+1009*rank)
            times,noise,tcp=c.full_noise(batch['targets'].image,seed=42,cursor=update,world=8)
            optimizer.zero_grad()
            out=model(batch,timestep_indices=times[rank:rank+1].to(device),noise=noise[rank:rank+1].to(device),tcp_noise=tcp[rank:rank+1].to(device))
            if not torch.isfinite(out['loss']):raise FloatingPointError('nonfinite C* objective')
            out['loss'].backward();norm=torch.nn.utils.clip_grad_norm_(expert.parameters(),1.,error_if_nonfinite=True,foreach=False)
            optimizer.step();torch.cuda.synchronize(device);update+=1
            if rank==0:
                fm=float(out['fm_loss']);tcp_loss=float(out['tcp_loss'])
                row=dict(update=update,samples=update*8,arm='causal_cstar',seconds=time.monotonic()-started,
                    wall_seconds=offset+time.monotonic()-clock,loss=.5*fm+tcp_loss,fm_loss=fm,tcp_loss=tcp_loss,
                    gradient_norm=float(norm),valid_images=out['valid_images'],valid_points=out['valid_points'],
                    sample_ids=[entries[i]['id'] for i in indices],
                    grain_counts={g:sum(entries[i]['granularity']==g for i in indices) for g in ('task','subtask')},
                    native_noise_sha256=dict(indices=shared.tensor_digest(times),image=shared.tensor_digest(noise),tcp=shared.tensor_digest(tcp)))
                base.append_jsonl(output/'train.jsonl',row);print(json.dumps(row,allow_nan=False),flush=True)
            del out,batch,noise,tcp
            stop=torch.tensor(int((output/'STOP_REQUESTED').exists()) if rank==0 else 0,device=device)
            dist.broadcast(stop,src=0);requested=bool(stop.item())
            if update%100==0 or update==end or requested:assess(update)
            if update%500==0 or update==end or requested:
                start=time.monotonic();c.checkpoint(expert,optimizer,output,update=update,contract=contract,dependencies=manifest['dependencies'])
                if rank==0:base.append_jsonl(output/'phases.jsonl',dict(update=update,phase='checkpoint',seconds=time.monotonic()-start))
            if requested:break
        result=dict(update=update,budget=1000,status='complete' if update==1000 else 'interrupted',resume=str(output/'checkpoint.json'))
        if rank==0:base.atomic_json(output/'summary.json',result)
        return result
    finally:dist.destroy_process_group()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--code-dir',required=True);p.add_argument('--data',required=True);p.add_argument('--parent-data',required=True)
    p.add_argument('--output',required=True);p.add_argument('--runs-root',required=True)
    p.add_argument('--updates',type=int,default=1000);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--resume');p.add_argument('--stop-after',type=int)
    args=p.parse_args(argv)
    if args.stop_after is not None and args.stop_after<1:p.error('--stop-after must be positive')
    result=train(args)
    if int(os.environ.get('RANK','0'))==0:print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
