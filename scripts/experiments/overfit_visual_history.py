"""Bounded real-checkpoint overfit and same-noise conditioning interventions.

Run from the isolated checkout. Output is intentionally outside the checkout.
This is a train-set diagnostic, not a generalization benchmark.
"""
import argparse
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml
from PIL import Image
from safetensors.torch import save_file, load_file

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(CODE))
from openwam.model.goal.codec import GoalCodec
from openwam.model.goal.history import history_kwargs
from openwam.model.goal.visual_history import VisualHistoryGoalExpert
from openwam.model.goal.patch_expert import decode_tcp_patch
from openwam.model.video_backbone.wan.shared.diffusion.flow_match import FlowMatchScheduler
from openwam.train.goal_trainer import _construct, load_component_weights, _seed
from openwam.train.goal_curated import prepare_batch
from openwam.train.goal_overfit import FP32MasterAdamW, _write_json, _display_image, export_inputs
from scripts.experiments.stage1_tcp_report import tcp_overlay_data, draw_tcp_overlay


def perturb(batch, other, mode):
    b=dict(batch)
    if mode in ('wrong_visual','wrong_current'):
        for k in ('observed','observed_view_valid'):b[k]=other[k]
    if mode in ('wrong_visual','wrong_history'):
        for k in ('history_images','history_latents','history_ages','history_valid','history_episode_id'):
            if k in other:b[k]=other[k]
            else:b.pop(k,None)
    if mode in ('no_history','blank_visual'):
        b['history_valid']=torch.zeros_like(b['history_valid'])
        b['history_ages']=torch.zeros_like(b['history_ages'])
    if mode=='blank_visual':b['observed']=torch.zeros_like(b['observed'])
    return b


def rgb(t):return np.asarray(_display_image(t),dtype=np.float32)


@torch.no_grad()
def evaluate(expert, batch, directory, *, all_targets, seed=100042, trace=False):
    expert.eval();directory.mkdir(parents=True,exist_ok=True)
    c=expert.condition_tokens(batch['observed'],observed_view_valid=batch['observed_view_valid'],**history_kwargs(batch))
    obs=expert.encode_observation(batch['observed'],observed_view_valid=batch['observed_view_valid'])
    rng=torch.Generator(device=obs.device).manual_seed(seed)
    image=torch.randn((1,48,1,16,20),device=obs.device,dtype=obs.dtype,generator=rng)
    tcp=torch.randn((1,192),device=obs.device,dtype=torch.float32,generator=rng)
    scheduler=FlowMatchScheduler(template='Wan');scheduler.set_timesteps(20,shift=expert.shift)
    gt=rgb(batch['targets'].image);traces=[]
    for i,t in enumerate(scheduler.timesteps):
        velocity,tcp_v=expert.velocity(obs,image,tcp,t.to(obs).expand(1),batch['context'],
            context_lens=batch['context_lens'],observed_view_valid=batch['observed_view_valid'],observation_tokens=c)
        if trace and i in (0,1,4,9,14,19):
            x0=image-float(scheduler.sigmas[i])*velocity
            pic=_display_image(expert.codec.decode(x0));pic.save(directory/f'x0_{i+1:02d}.png')
            traces.append(dict(iteration=i+1,sigma=float(scheduler.sigmas[i]),mae=float(np.abs(np.asarray(pic,np.float32)-gt).mean())))
        image=scheduler.step(velocity,t,image);tcp=scheduler.step(tcp_v,t,tcp)
    prediction=_display_image(expert.codec.decode(image));prediction.save(directory/'prediction.png')
    arr=np.asarray(prediction,np.float32);uv=decode_tcp_patch(tcp).float()
    target=batch['targets'];valid=target.point_valid & target.in_frame & torch.isfinite(target.uv).all(-1)
    scale=torch.tensor([319,255],device=uv.device);err=((uv-target.uv)*scale).square().sum(-1).sqrt()
    # Display finite predictions independently of GT availability. Orange is an
    # untrusted output for this unsupervised sample/arm; loss/metric masks above
    # remain unchanged. Both backgrounds use the same 320x256 pixel convention.
    coordinates=tcp_overlay_data({},target,dict(uv=uv.cpu().tolist()))
    draw_tcp_overlay(_display_image(target.image),coordinates,directory/'tcp.png','TCP | GT background')
    draw_tcp_overlay(prediction,coordinates,directory/'prediction_tcp.png','TCP | generated background')
    _write_json(directory/'tcp_coordinates.json',coordinates)
    distances=[float(np.abs(arr-g).mean()) for g in all_targets]
    metrics=dict(mae=float(np.abs(arr-gt).mean()),target_distances=distances,
        nearest_scene=int(np.argmin(distances)),tcp_error_px=float(err[valid].mean()) if valid.any() else None,
        valid_tcp=int(valid.sum()),uv=uv.cpu().tolist(),seed=seed,steps=20,trace=traces,
        condition_rms=float(c.float().square().mean().sqrt()))
    _write_json(directory/'metrics.json',metrics)
    return metrics


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,nargs=3,default=[200,600,800])
    args=parser.parse_args();root=args.root;out=args.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():raise ValueError('output already contains a run')
    report=out/'report';report.mkdir();start=time.monotonic()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    def event(kind,**kwargs):
        row=dict(event=kind,seconds=time.monotonic()-start,**kwargs)
        print(json.dumps(row),flush=True)
        with (out/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    smoke=root/'outputs/stage1_smoke_history_20261029'
    cfg=yaml.safe_load((smoke/'config.yaml').read_text())
    ckpt=root/'assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model/checkpoint_step_154000.safetensors'
    source_ids=[8,12,4,26]
    items=[torch.load(smoke/'samples'/f'{i:02d}_2.pt',weights_only=False,map_location='cpu') for i in source_ids]
    contexts=torch.load(smoke/'text_contexts.pt',weights_only=True,map_location='cpu')
    common_instruction=items[0][0]['instruction'];common=contexts[common_instruction]
    for r,b in items:b['context'],b['context_lens']=common
    protocol=dict(base_commit='5cd534c',code=str(CODE),checkpoint=str(ckpt),checkpoint_bytes=ckpt.stat().st_size,
        source_ids=source_ids,records=[r for r,b in items],common_instruction=common_instruction,
        text_control='All four scenes deliberately share the same cached text, diagnostic visual routing only.',
        target='80 image patches + 1 TCP patch; original TCP validity masks',
        history=dict(num_history_tokens=32,pool_dim=256,pool_heads=8,adapter_block=0),
        initialization='Official154000 Wan; copied block0 visual adapter, gate0; VAE/T5 frozen.',
        stages=[dict(name=n,steps=s,cases=c) for n,s,c in zip(['singleton','two_scene','four_scene'],args.steps,[1,2,4])],
        learning_rate=1e-5,history_learning_rate=1e-4,history_dropout=.2,
        seeds=dict(model=42,history=777,train='10000+global_step',evaluation=100042),
        optimizer='FP32 master AdamW, weight_decay .01, clip1, BF16 forward',
        evaluation='Training anchors, plus same-episode held anchors at end; no unseen-episode generalization claim.',
        source_hashes={str(p.relative_to(CODE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
            Path(__file__),CODE/'openwam/model/goal/visual_history.py',CODE/'openwam/model/goal/patch_expert.py']})
    _write_json(out/'protocol.json',protocol)
    event('loading_vae')
    vae=_construct(cfg['components']['vae']);load_component_weights(vae,ckpt,'video_backbone.vae.')
    vae.to(device='cuda',dtype=torch.bfloat16).eval().requires_grad_(False);codec=GoalCodec(vae)
    # Exact-byte bounded input cache for this fixed-data diagnostic, frozen VAE only.
    original_encode=vae.encode;latent_cache={}
    def cached(videos,device,**kwargs):
        videos=list(videos);key=tuple((tuple(x.shape),hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()) for x in videos)
        if key not in latent_cache:latent_cache[key]=original_encode(videos,device=device,**kwargs).detach()
        return latent_cache[key]
    vae.encode=cached
    batches=[prepare_batch(r,b,device='cuda',dtype=torch.bfloat16) for r,b in items]
    cards=export_inputs(report,[r for r,b in items],batches)
    for i,b in enumerate(batches):
        _display_image(codec.decode(codec.encode(b['targets'].image))).save(report/'inputs'/f'case_{i:04d}_vae.png')
        h=history_kwargs(b);latents=[]
        for j in range(10):
            latents.append(codec.encode(b['history_images'][:,j]) if bool(b['history_valid'][0,j]) else torch.zeros(1,48,1,16,20,device='cuda',dtype=torch.bfloat16))
        b['history_latents']=torch.stack(latents,dim=1);del b['history_images']
    all_targets=[rgb(b['targets'].image) for b in batches]
    event('loading_dit');_seed(42)
    dit=_construct(cfg['components']['dit']);load_component_weights(dit,ckpt,'video_backbone.dit.');_seed(777)
    expert=VisualHistoryGoalExpert(dit,codec,dit_config=cfg['components']['dit']['extra_kwargs'],
        use_gradient_checkpointing=True,**protocol['history']).to(device='cuda',dtype=torch.bfloat16)
    del dit;expert.train_stage_a()
    summary=dict(status='running',completed_steps=0,evaluations=[],cards=cards,
        copy_maes=[float(np.abs(rgb(b['observed'][:,:,:,:256])-g).mean()) for b,g in zip(batches,all_targets)])
    def eval_all(step,count):
        vals=[evaluate(expert,b,report/f'step_{step:06d}'/f'case_{i}',all_targets=all_targets,trace=True) for i,b in enumerate(batches[:count])]
        row=dict(step=step,cases=count,values=vals,gate=float(expert.history_gate))
        summary['evaluations'].append(row);_write_json(out/'summary.json',summary)
        event('evaluation',step=step,maes=[v['mae'] for v in vals],gate=row['gate'])
    eval_all(0,4)
    params=list(expert.named_parameters())
    groups=[dict(name='wan',params=[p for n,p in params if n.startswith('dit.')],lr=1e-5),
            dict(name='history',params=[p for n,p in params if not n.startswith('dit.')],lr=1e-4)]
    optimizer=FP32MasterAdamW(groups)
    step=0
    for stage in protocol['stages']:
        event('stage_start',**stage)
        for local in range(stage['steps']):
            step+=1;_seed(10000+step);expert.train_stage_a();optimizer.zero_grad()
            case=local%stage['cases'];b=dict(batches[case])
            b['history_valid']=b['history_valid'] & (torch.rand_like(b['history_ages'])>=.2)
            b['history_ages']=torch.where(b['history_valid'],b['history_ages'],0.)
            result=expert(b['observed'],b['context'],b['targets'],context_lens=b['context_lens'],
                observed_view_valid=b['observed_view_valid'],**history_kwargs(b))
            result['loss'].backward()
            norm=torch.nn.utils.clip_grad_norm_(expert.parameters(),1.,error_if_nonfinite=True,foreach=False)
            row=dict(step=step,stage=stage['name'],case=case,loss=float(result['loss']),image_loss=float(result['fm_loss']),
                tcp_loss=float(result['tcp_loss']),valid_tcp=result['valid_points'],grad_norm=float(norm),gate=float(expert.history_gate))
            if step in (1,2,10,100):
                row['gradient_audit']={name:sum(float(p.grad.float().square().sum()) for p in module.parameters() if p.grad is not None)**.5
                    for name,module in [('history_pool',expert.history_pool),('visual_adapter',expert.visual_adapter),('native_text',expert.dit.text_embedding)]}
            optimizer.step();del result
            with (out/'train.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            summary['completed_steps']=step
            if step%50==0:event('progress',**row)
            if (local+1)%200==0 or local+1==stage['steps']:eval_all(step,stage['cases'])
    del optimizer;gc.collect();torch.cuda.empty_cache()
    summary['interventions']={}
    for i,b in enumerate(batches):
        other=batches[(i+1)%len(batches)];values={}
        for mode in ('real','no_history','wrong_history','wrong_current','wrong_visual','blank_visual'):
            altered=perturb(b,other,mode)
            values[mode]=evaluate(expert,altered,report/'final'/f'case_{i}'/mode,all_targets=all_targets,trace=mode in ('real','wrong_visual'))
        values['wrong_visual_expected_scene']=(i+1)%len(batches)
        summary['interventions'][str(i)]=values
        _write_json(out/'summary.json',summary);event('interventions',case=i,real_mae=values['real']['mae'],wrong_nearest=values['wrong_visual']['nearest_scene'])
    # Previously unused anchors in the same episodes; these are not held episodes.
    held_items=[torch.load(smoke/'samples'/f'{i:02d}_0.pt',weights_only=False,map_location='cpu') for i in source_ids]
    held_batches=[]
    for r,b in held_items:
        b['context'],b['context_lens']=common
        held_batches.append(prepare_batch(r,b,device='cuda',dtype=torch.bfloat16))
    held_root=report/'held_anchor';held_root.mkdir();export_inputs(held_root,[r for r,b in held_items],held_batches)
    summary['held_anchor']=[evaluate(expert,b,held_root/f'case_{i}',all_targets=all_targets,trace=False) for i,b in enumerate(held_batches)]
    # Fixed-noise boundary sensitivity independently of iterative image sampling.
    expert.eval();summary['history_effect']={}
    with torch.no_grad():
        for i,b in enumerate(batches):
            c=expert.condition_tokens(b['observed'],observed_view_valid=b['observed_view_valid'],**history_kwargs(b))
            no=perturb(b,b,'no_history');c0=expert.condition_tokens(no['observed'],observed_view_valid=no['observed_view_valid'],**history_kwargs(no))
            wrong=perturb(b,batches[(i+1)%4],'wrong_history');cw=expert.condition_tokens(wrong['observed'],observed_view_valid=wrong['observed_view_valid'],**history_kwargs(wrong))
            summary['history_effect'][str(i)]=dict(no_history_delta_rms=float((c-c0).float().square().mean().sqrt()),wrong_history_delta_rms=float((c-cw).float().square().mean().sqrt()))
    event('saving_checkpoint')
    weights={k:v.detach().cpu().contiguous() for k,v in expert.state_dict().items()}
    metadata=dict(format='openwam.visual_history.v1',dit_config=expert.dit_config,history=expert.history_options,
        shift=expert.shift,lambda_image=expert.lambda_image,lambda_tcp=expert.lambda_tcp,
        official_checkpoint=str(ckpt),dependencies=json.loads((smoke/'dependencies.json').read_text()),protocol=protocol)
    checkpoint_path=out/'visual_history.safetensors'
    save_file(weights,str(checkpoint_path),metadata={'visual_history':json.dumps(metadata)});del weights
    _write_json(out/'visual_history.json',metadata)
    # Reload from disk into the real model, then recompute a deterministic sample.
    restored=load_file(str(checkpoint_path),device='cpu');expert.load_state_dict(restored,strict=True);del restored
    reloaded=evaluate(expert,batches[0],report/'reload_check',all_targets=all_targets)
    a=np.asarray(Image.open(report/'final/case_0/real/prediction.png'));b=np.asarray(Image.open(report/'reload_check/prediction.png'))
    summary['checkpoint_reload']=dict(pixel_identical=bool(np.array_equal(a,b)),mae=reloaded['mae'])
    assert summary['checkpoint_reload']['pixel_identical']
    summary.update(status='complete',elapsed_s=time.monotonic()-start,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
        final_gate=float(expert.history_gate),checkpoint=str(checkpoint_path))
    _write_json(out/'summary.json',summary);event('complete',elapsed_s=summary['elapsed_s'])


if __name__=='__main__':main()
