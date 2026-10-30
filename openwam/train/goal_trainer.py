"""Bounded Stage A G0/G1 training, portable goal-only assets and diagnostics.

The VAE/text modules are caller-owned frozen dependencies. Goal checkpoints
contain the independent DiT, role embeddings and point head, with reconstruction
metadata embedded in safetensors so relocation needs no training directory.
"""
from __future__ import annotations
import hashlib
import importlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from safetensors import safe_open
from safetensors.torch import save_file
from openwam.model.goal.dependencies import (module_identity,tokenizer_identity,dependency_identity,
    precomputed_identity,resolve_tokenizer_config,text_identity)
from openwam.model.goal.codec import GoalCodec
from openwam.model.goal.expert import GoalExpert
from openwam.model.goal.heads import heatmap_uv, point_mask, target_uv_pixels, overlay_points
from openwam.model.goal.types import GoalTargets,collate_goal_targets
from openwam.model.video_backbone.wan.models.dit import WanModel


TINY_DIT_CONFIG = dict(dim=48,in_dim=48,ffn_dim=64,out_dim=48,text_dim=16,freq_dim=16,
    eps=1e-6,patch_size=[1,2,2],num_heads=2,num_layers=1,has_image_input=False,
    seperated_timestep=True,require_vae_embedding=False,require_clip_embedding=False,
    fuse_vae_embedding_in_latents=True)


def source_hashes():
    root=Path(__file__).resolve().parents[1]
    names=['model/goal/dependencies.py','model/goal/codec.py','model/goal/expert.py','model/goal/heads.py','model/goal/types.py','train/goal_trainer.py']
    return {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}



def save_goal_bundle(expert, checkpoint, *, training_metadata=None, dependency_metadata=None,
                     text_encoder=None,tokenizer=None,precomputed_text_fingerprint=None,dependency_identities=None):
    """Save a single relocatable file; adjacent JSON is a human-readable copy."""
    checkpoint=Path(checkpoint);checkpoint.parent.mkdir(parents=True,exist_ok=True)
    identities=dependency_identities or dependency_identity(expert.codec.vae,
        text_dim=expert.dit.text_embedding[0].in_features,text_encoder=text_encoder,tokenizer=tokenizer,
        precomputed_text_fingerprint=precomputed_text_fingerprint)
    metadata=dict(format='openwam.goal.v2',dit_config=expert.dit_config,
                  codec=expert.codec.metadata(),shift=expert.shift,lambda_kp=expert.lambda_kp,
                  point_head=dict(hidden_dim=expert.point_head.hidden_dim,heatmap_size=expert.point_head.heatmap_size),
                  dependencies={**identities,'reconstruction':dependency_metadata or {}},
                  training=training_metadata or {},source_hashes=source_hashes(),
                  sampling=dict(method='Wan Euler flow',shift=expert.shift,return_latent='final_z0'),
                  frozen_on_load=True,point_order=['left','right'])
    serialized=json.dumps(metadata,sort_keys=True)
    weights={k:v.detach().cpu().contiguous() for k,v in expert.state_dict().items()}
    save_file(weights,str(checkpoint),metadata={'goal_bundle':serialized})
    checkpoint.with_suffix('.json').write_text(json.dumps(metadata,indent=2,sort_keys=True)+'\n')
    return metadata


def load_goal_bundle(checkpoint, *, vae, text_encoder=None, tokenizer=None, text_dim=None,
                     device='cpu',dtype=torch.float32,precomputed_text_fingerprint=None):
    """Explicit shared dependencies; no baseline/training path is opened.

    Precomputed contexts must declare both text_dim and the matching text
    fingerprint. Otherwise actual shared encoder/tokenizer identities are checked.
    No dependency is registered or reopened through historical asset paths.
    """
    with safe_open(str(checkpoint),framework='pt',device='cpu') as file:
        raw=(file.metadata() or {}).get('goal_bundle')
        if raw is None:raise ValueError('missing embedded goal bundle metadata')
        metadata=json.loads(raw)
        if metadata.get('format')!='openwam.goal.v2':raise ValueError('unsupported goal checkpoint format: dependency identities required')
        actual_text_dim=getattr(text_encoder,'dim',text_dim)
        if actual_text_dim!=metadata['dependencies']['text_dim']:
            raise ValueError('goal text encoder/context dimension incompatible with checkpoint')
        if module_identity(vae)!=metadata['dependencies']['vae']:
            raise ValueError('goal VAE weights/class/config incompatible with checkpoint')
        actual_text=(text_identity(text_encoder,tokenizer)
                     if text_encoder is not None else precomputed_identity(precomputed_text_fingerprint))
        expected_text=metadata['dependencies']['text']
        if (actual_text['fingerprint']!=expected_text['fingerprint'] or
                (text_encoder is not None and actual_text!=expected_text)):
            raise ValueError('goal text encoder/tokenizer dependency incompatible with checkpoint')
        if metadata['codec']!=json.loads(json.dumps(GoalCodec(vae).metadata())):
            raise ValueError('goal codec metadata incompatible with runtime')
        config=metadata['dit_config']
        expert=GoalExpert(WanModel(**config),GoalCodec(vae),dit_config=config,shift=metadata['shift'],
                          lambda_kp=metadata['lambda_kp'],point_hidden_dim=metadata['point_head']['hidden_dim'],
                          heatmap_size=metadata['point_head']['heatmap_size'])
        expert.load_state_dict({key:file.get_tensor(key) for key in file.keys()},strict=True)
    expert.to(device=device,dtype=dtype).freeze()
    expert.bundle_metadata=metadata
    # Shared dependencies remain external, but their required placement must be explicit.
    for name,module in [('VAE',vae),('text encoder',text_encoder)]:
        if isinstance(module,nn.Module):
            tensors=list(module.parameters())+list(module.buffers())
            if any(t.device!=next(expert.parameters()).device for t in tensors):
                raise ValueError(f'shared goal {name} must already be on {device}')
            if any(t.is_floating_point() and t.dtype!=dtype for t in tensors):
                raise ValueError(f'shared goal {name} must already use {dtype}')
            module.eval().requires_grad_(False)
    return expert


def load_component_weights(module, checkpoint, prefix):
    """Read only one component through safetensors; never materialize action tensors."""
    with safe_open(str(checkpoint),framework='pt',device='cpu') as file:
        keys=[key for key in file.keys() if key.startswith(prefix)]
        expected=set(module.state_dict());available={key[len(prefix):] for key in keys}
        if expected!=available:
            raise ValueError(f'component {prefix}: missing={sorted(expected-available)[:8]}, unexpected={sorted(available-expected)[:8]}')
        state={key[len(prefix):]:file.get_tensor(key) for key in keys}
    module.load_state_dict(state,strict=True)


def _construct(spec):
    module,name=spec['model_class'].rsplit('.',1)
    return getattr(importlib.import_module(module),name)(**spec.get('extra_kwargs',{}))


def bootstrap_components(config, *, device, dtype, validate_codec=True):
    """Build exactly DiT/VAE/text and tokenizer from baseline + explicit config."""
    components=config['components'];modules={}
    if set(components)!={'dit','vae','text_encoder'}:
        raise ValueError('Stage A components must be exactly dit, vae, text_encoder')
    # Sequential loading bounds peak host RAM to one component + loaded weights.
    for name in ('dit','vae','text_encoder'):
        started=time.monotonic()
        print(json.dumps(dict(event='component_loading',component=name)),flush=True)
        module=_construct(components[name])
        load_component_weights(module,config['baseline_checkpoint'],f'video_backbone.{name}.')
        module.to(device=device,dtype=dtype)
        if name!='dit':module.eval().requires_grad_(False)
        modules[name]=module
        print(json.dumps(dict(event='component_loaded',component=name,elapsed_s=time.monotonic()-started)),flush=True)
    token_cfg=resolve_tokenizer_config(config);token_module,token_name=token_cfg['model_class'].rsplit('.',1)
    tokenizer=getattr(importlib.import_module(token_module),token_name)(**token_cfg['kwargs'])
    expert=GoalExpert(modules['dit'],GoalCodec(modules['vae']),dit_config=components['dit'].get('extra_kwargs',{}),
                      shift=config.get('shift',5.),use_gradient_checkpointing=config.get('gradient_checkpointing',True)).to(device=device,dtype=dtype)
    # Validate architecture and codec immediately, before any training data loop.
    if validate_codec:
        with torch.no_grad():
            expert.codec.encode(torch.zeros(1,3,1,256,320,device=device,dtype=dtype))
    if getattr(modules['text_encoder'],'dim',None)!=expert.dit.text_embedding[0].in_features:
        raise ValueError('bootstrap text encoder is incompatible with goal DiT')
    return expert,modules['text_encoder'],tokenizer


def validate_training_manifest(manifest, *, allow_episode_end=False,diagnostic_override=False):
    from scripts.data.prepare_goal_labels import validate_goal_label,validate_splits
    rows=[json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip()]
    if not rows:raise ValueError('empty goal manifest')
    validate_splits(rows,diagnostic_override=diagnostic_override)
    for row in rows:validate_goal_label(row,allow_episode_end=allow_episode_end)
    return dict(pairs=len(rows),verified_success_pairs=sum(bool(row['success_verified']) for row in rows),
                episode_end_pairs=sum(row.get('label_kind')=='episode_end' for row in rows),
                heldout_contaminated=any(row.get('baseline_seen_episode') and row['split']!='train' for row in rows),
                manifest_sha256=hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
                diagnostic_override=diagnostic_override)


def point_metrics(logits,targets,sources,pck_threshold_px=16.):
    valid=point_mask(targets);uv=heatmap_uv(logits).float()
    error=(uv-target_uv_pixels(targets,valid)).square().sum(-1).sqrt()
    groups={'all':torch.ones_like(valid)}
    for source in sorted(set(sources)):
        groups[f'source:{source}']=torch.tensor([s==source for s in sources],device=valid.device)[:,None].expand_as(valid)
    for number,name in [(-1,'unknown'),(0,'occluded'),(1,'visible')]:
        groups[f'visibility:{name}']=targets.visibility==number
    result={}
    for group,mask in groups.items():
        result[group]={}
        for hand,name in enumerate(('left','right')):
            selected=error[:,hand][valid[:,hand]&mask[:,hand]];count=selected.numel()
            result[group][name]=dict(count=count,pixel_error=float(selected.mean()) if count else None,
                pck=float((selected<=pck_threshold_px).float().mean()) if count else None,pck_threshold_px=pck_threshold_px)
    return result


class _DiagnosticVAE(nn.Module):
    """Smoke-only codec stand-in; never exported as a production Wan VAE."""
    def encode(self,videos,device,**kwargs):
        x=torch.stack(list(videos)).to(device)
        if x.shape[2]!=1:raise ValueError('smoke codec only supports isolated single frames')
        return torch.nn.functional.avg_pool3d(x,(1,16,16)).repeat(1,16,1,1,1)
    def decode(self,z,device,**kwargs):
        return torch.nn.functional.interpolate(z[:,:3].to(device),size=(1,256,320)).clamp(-1,1)


def synthetic_smoke_batch(device='cpu'):
    yy,xx=torch.meshgrid(torch.arange(256),torch.arange(320),indexing='ij')
    left=torch.exp(-((xx-80.)**2+(yy-90.)**2)/(2*24**2))
    right=torch.exp(-((xx-240.)**2+(yy-160.)**2)/(2*24**2))
    image=torch.stack((left,right,(left+right)*.5))[None,:,None].to(device)*2-1
    observed=image.flip(-1)
    context=torch.linspace(-1,1,64,device=device).reshape(1,4,16)
    target=GoalTargets(image,torch.ones(1,device=device,dtype=torch.bool),
        torch.tensor([[[80/319,90/255],[240/319,160/255]]],device=device),torch.ones(1,2,device=device,dtype=torch.bool),
        torch.ones(1,2,device=device,dtype=torch.bool),torch.ones(1,2,device=device,dtype=torch.int8))
    return observed,context,target


def run_smoke(*,steps=20,device='cpu',seed=42,lambda_kp=.1):
    """Actual tiny Wan training; synthetic image/geometry, fixed-noise overfit only."""
    torch.manual_seed(seed);start=time.monotonic()
    if torch.device(device).type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    expert=GoalExpert(WanModel(**TINY_DIT_CONFIG),GoalCodec(_DiagnosticVAE()),dit_config=TINY_DIT_CONFIG,
                      lambda_kp=lambda_kp,point_hidden_dim=32).to(device)
    obs,context,target=synthetic_smoke_batch(device)
    noise=torch.randn(1,48,1,16,20,device=device);idx=torch.tensor([800],device=device)
    optimizer=torch.optim.AdamW(expert.parameters(),lr=.003,weight_decay=0.)
    initial=expert(obs,context,target,timestep_indices=idx,noise=noise)
    gradients=torch.autograd.grad(initial['kp_loss'],expert.dit.parameters(),retain_graph=True,allow_unused=True)
    gradient_norm=float(sum(g.float().square().sum() for g in gradients if g is not None).sqrt())
    values=lambda out:{key:float(out[key].detach()) for key in ('loss','fm_loss','kp_loss')}
    report=dict(seed=seed,steps=steps,lambda_kp=lambda_kp,device=str(device),synthetic_diagnostic=True,
                real_wan_blocks=True,initial=values(initial),valid_points=initial['valid_points'],
                point_generator_gradient_norm=gradient_norm,history=[])
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        out=expert(obs,context,target,timestep_indices=idx,noise=noise)
        out['loss'].backward();torch.nn.utils.clip_grad_norm_(expert.parameters(),1.)
        optimizer.step();report['history'].append(dict(step=step+1,**values(out)))
    with torch.no_grad():final=expert(obs,context,target,timestep_indices=idx,noise=noise)
    report.update(final=values(final),elapsed_s=time.monotonic()-start,
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if torch.device(device).type=='cuda' else None,
                  semantic_success='not evaluated; synthetic diagnostic only',
                  precomputed_text_fingerprint=hashlib.sha256(context.detach().cpu().numpy().tobytes()).hexdigest())
    return report,expert.freeze()


def _seed(seed):
    random.seed(seed);np.random.seed(seed%(2**32));torch.manual_seed(seed)


@torch.no_grad()
def validation_score(expert,batches,seed=1234):
    """Same fixed val criterion for both variants; never select on train/test."""
    expert.eval();fm_sum=0.;kp_sum=0.;image_count=0;point_count=0;count=0
    for index,batch in enumerate(batches):
        _seed(seed+index)
        out=expert(batch['observed'],batch['context'],batch['targets'],context_lens=batch.get('context_lens'))
        n_images=int(batch['targets'].image_valid.sum());n_points=out['valid_points']
        fm_sum+=float(out['fm_loss'])*n_images;kp_sum+=float(out['kp_loss'])*n_points
        image_count+=n_images;point_count+=n_points;count+=1
    if count==0:raise ValueError('Stage A requires nonempty validation data for model selection')
    if image_count==0:raise ValueError('Stage A validation has no valid goal images')
    fm=fm_sum/image_count;kp=kp_sum/max(1,point_count)
    if not np.isfinite(fm+kp):raise FloatingPointError('nonfinite validation loss')
    return dict(fm_loss=fm,kp_loss=kp,score=fm,batches=count,valid_images=image_count,valid_points=point_count)


def train_paired(expert, *, train_batches,val_batches,reset_weights,output_dir,steps=1000,
                 eval_every=100,learning_rate=1e-5,seed=42,training_metadata=None,dependency_metadata=None,
                 text_encoder=None,tokenizer=None,precomputed_text_fingerprint=None):
    """Identical reset, seeds, sample sequence, budget and val selection for G0/G1.

    Batch providers are called lazily; callers can use real strict readers
    without preloading image/video datasets or registering shared text modules.
    """
    if steps<1 or eval_every<1:raise ValueError('Stage A steps/eval_every must be positive')
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    run_started=time.monotonic()
    def event(kind,**values):
        line=json.dumps(dict(event=kind,elapsed_s=time.monotonic()-run_started,**values))
        with (output_dir/'events.jsonl').open('a') as stream:
            stream.write(line+'\n')
        print(line,flush=True)
    event('dependency_identity_started')
    identities=dependency_identity(expert.codec.vae,text_dim=expert.dit.text_embedding[0].in_features,
        text_encoder=text_encoder,tokenizer=tokenizer,precomputed_text_fingerprint=precomputed_text_fingerprint)
    event('dependency_identity_finished')
    if isinstance(text_encoder,nn.Module):text_encoder.eval().requires_grad_(False)
    summary=dict(selection_rule='val fm_loss',seed=seed)
    for variant,coefficient in [('G0',0.),('G1',.1)]:
        variant_started=time.monotonic();samples=0
        event('variant_started',variant=variant,steps=steps,lambda_kp=coefficient)
        _seed(seed);reset_weights();expert.train_stage_a();expert.lambda_kp=coefficient
        expert.point_head.requires_grad_(coefficient>0)
        optimizer=torch.optim.AdamW([p for p in expert.parameters() if p.requires_grad],lr=learning_rate,weight_decay=.01)
        best=float('inf');points=0;history=[];path=output_dir/variant/'goal.safetensors'
        if next(expert.parameters()).is_cuda:torch.cuda.reset_peak_memory_stats()
        for step in range(steps):
            step_started=time.monotonic()
            _seed(seed+step);expert.train();batch=train_batches(step)
            optimizer.zero_grad(set_to_none=True)
            out=expert(batch['observed'],batch['context'],batch['targets'],context_lens=batch.get('context_lens'))
            if not torch.isfinite(out['loss']):raise FloatingPointError(f'{variant} nonfinite loss at step {step+1}')
            out['loss'].backward();torch.nn.utils.clip_grad_norm_(expert.parameters(),1.)
            optimizer.step();points+=out['valid_points']
            entry=dict(step=step+1,loss=float(out['loss'].detach()),fm_loss=float(out['fm_loss'].detach()),
                       kp_loss=float(out['kp_loss'].detach()),valid_points=out['valid_points'])
            samples+=int(batch['observed'].shape[0])
            entry['step_seconds']=time.monotonic()-step_started
            event('train_step',variant=variant,**entry)
            if (step+1)%eval_every==0 or step+1==steps:
                event('validation_started',variant=variant,step=step+1)
                score=validation_score(expert,val_batches(),seed=seed+100000)
                entry['val']=score
                event('validation_finished',variant=variant,step=step+1,metrics=score)
                if score['score']<best:
                    best=score['score']
                    save_goal_bundle(expert,path,training_metadata={**(training_metadata or {}),
                        'variant':variant,'selected_step':step+1,'selection_split':'val','selection_rule':summary['selection_rule'],
                        'selection_metrics':score,'seed':seed,'max_steps':steps,'valid_points_seen_at_selection':points},
                        dependency_metadata=dependency_metadata,dependency_identities=identities)
                    event('checkpoint_saved',variant=variant,step=step+1,checkpoint=str(path),selection_rule=summary['selection_rule'])
            history.append(entry)
        expert.freeze()
        summary[variant]=dict(steps=steps,checkpoint=str(path),selection_split='val',selection_rule=summary['selection_rule'],best_val_score=best,
            valid_points_seen=points,point_supervision_effective=bool(coefficient>0 and points>0),history=history,
            elapsed_s=time.monotonic()-variant_started,training_samples=samples,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if next(expert.parameters()).is_cuda else None)
        if coefficient>0 and points==0:
            print('G1: zero valid points; point supervision was ineffective. Image training only.',flush=True)
        (output_dir/variant/'training_report.json').write_text(json.dumps(summary[variant],indent=2)+'\n')
        event('variant_finished',variant=variant,variant_seconds=summary[variant]['elapsed_s'],training_samples=samples)
        del optimizer
    (output_dir/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def prepare_goal_batch(samples, *, text_encoder,tokenizer,device,dtype):
    from openwam.model.video_backbone.wan.encode import encode_text
    images=[]
    for sample in samples:
        image=np.array(sample['goal_observed_head'].convert('RGB'),copy=True)
        images.append(torch.from_numpy(image).permute(2,0,1).unsqueeze(1).float()/127.5-1)
    with torch.no_grad():
        context,lens=encode_text([sample['prompt'] for sample in samples],tokenizer=tokenizer,text_encoder=text_encoder,device=device)
    return dict(observed=torch.stack(images).to(device=device,dtype=dtype),context=context.to(device=device,dtype=dtype),
        context_lens=lens,targets=collate_goal_targets([sample['goal_targets'] for sample in samples]).to(device=device,dtype=dtype),
        sources=[sample.get('_curated_id',sample.get('goal_source_identity','unknown')).split(':')[0] for sample in samples])


def overlay_goal_targets(image, uv_pixels, valid):
    """Draw only eligible geometry labels on the exact 320x256 target pixels."""
    from PIL import ImageDraw
    if image.size!=(320,256):raise ValueError('goal target overlay requires 320x256 image')
    result=image.copy();draw=ImageDraw.Draw(result)
    for side,xy,enabled,color in zip(('L','R'),uv_pixels,valid,('red','blue')):
        if not enabled:continue
        x,y=map(float,xy)
        if not (np.isfinite([x,y]).all() and 0<=x<=319 and 0<=y<=255):
            raise ValueError('eligible target must be inside head image')
        draw.rectangle((x-4,y-4,x+4,y+4),outline=color,width=2)
        draw.text((x+5,y-8),side,fill=color)
    return result


def render_goal_report(output_dir,report):
    """Keep geometric supervision separate from learned heatmap predictions."""
    import html
    output_dir=Path(output_dir)
    status=report.get('point_training_status','unknown')
    message={'no_effective_point_supervision':'有效点监督为 0：预测点头未学到定位，红蓝圆点不能视为正确 TCP 投影。',
             'disabled':'G0 未训练点头：红蓝圆点只是未训练预测。',
             'has_point_supervision':'点头接受过监督；预测精度需用对照和指标验证。',
             'unknown':'无法从 checkpoint 确认点头监督状态，不能假定预测可靠。'}[status]
    cards=[]
    for c in report['review']:
        name=c['id']
        valid=c.get('target_point_valid',[False,False])
        supported=all(valid)
        note=('左右 TCP GT 均有效；误差仅衡量对这些标签的拟合。' if supported else
              '本样例无有效 TCP GT；不能评估 TCP 预测。' if not any(valid) else
              '本样例仅部分 TCP GT 有效；缺失侧不能评估。')
        note += ' ' + html.escape(c.get('audit_note',''))
        raw_link=('' if supported else f'<p><a href="{name}_generated.png">查看未经完整 GT 验证的原始预测点（诊断）</a></p>')
        details={k:c[k] for k in ('target_point_valid','target_uv_px','predicted_uv_px','geometry_notes','provenance') if k in c}
        panels=[('真实目标图像',f'{name}_target.png'),('真实目标图像 + 几何 GT 候选（方框；仅有效点）' if any(valid) else '真实目标图像（无有效 TCP GT）',f'{name}_target_points.png'),('生成图像 + 模型预测点（圆圈；不是几何投影）' if supported else '生成图像（GT 不完整，默认不叠加预测点）',f'{name}_generated.png' if supported else f'{name}_generated_clean.png')]
        figures=''.join(f'<figure><figcaption>{html.escape(title)}</figcaption><img src="{path}" width="320" height="256"></figure>' for title,path in panels)
        cards.append(f'<article><h2>{html.escape(c["source"])} {name}</h2><p class="warning">{note}</p><div class="panels">{figures}</div>{raw_link}<pre>{html.escape(json.dumps(details,ensure_ascii=False,indent=2))}</pre></article>')
    if 'step' in report:
        message=f'第 {report["step"]} 步评估。'+message
    output_dir.joinpath('index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Goal 图像与 TCP：GT / 预测分开展示</title><style>body{font:16px/1.6 system-ui;max-width:1150px;margin:24px auto;padding:0 16px}.warning{background:#fff1da;padding:16px}article{border-top:1px solid #bbb;margin-top:24px}.panels{display:flex;flex-wrap:wrap;gap:12px}figure{margin:0;width:320px}figcaption{min-height:52px}img{max-width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><h1>目标图像、几何 GT 候选与模型预测</h1><p class="warning">'+html.escape(message)+'</p><p>左手红色，右手蓝色。GT 坐标以实际目标图像 320×256 为准；无有效标签的手不画点。几何 GT 候选仍需核验相机标定与 TCP 定义。模型预测来自热图峰值，不能代替标定投影。</p>'+''.join(cards)+'</html>')


@torch.no_grad()
def evaluate_goals(expert,batches,output_dir, *, seed=42,num_steps=20):
    """Numerical heldout diagnostics + pending manual semantic review cards.

    Object/relation correctness and image-point agreement require review. No
    automatic metric here claims these checks passed.
    """
    import html
    from PIL import Image
    expert.freeze();output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    logits=[];targets=[];sources=[];cards=[]
    for index,batch in enumerate(batches):
        prediction=expert.generate(batch['observed'],batch['context'],seed=seed+index,num_steps=num_steps,context_lens=batch.get('context_lens'))
        decoded=expert.codec.decode(prediction.latent).detach().float().cpu()
        for item in range(decoded.shape[0]):
            image=Image.fromarray(((decoded[item,:,0].clamp(-1,1)+1)*127.5).byte().permute(1,2,0).numpy())
            target=Image.fromarray(((batch['targets'].image[item,:,0].detach().float().cpu().clamp(-1,1)+1)*127.5).byte().permute(1,2,0).numpy())
            name=f'{index:04d}_{item:02d}'
            image.save(output_dir/f'{name}_generated_clean.png')
            overlay_points(image,prediction.point_logits[item]).save(output_dir/f'{name}_generated.png')
            target.save(output_dir/f'{name}_target.png')
            valid=point_mask(batch['targets'])[item].cpu().tolist()
            uv=target_uv_pixels(batch['targets'])[item].cpu().tolist()
            overlay_goal_targets(target,uv,valid).save(output_dir/f'{name}_target_points.png')
            cards.append(dict(id=name,source=batch['sources'][item],object_correct='pending_review',
                              relation_correct='pending_review',image_point_consistent='pending_review',
                              target_point_valid=valid,target_uv_px=[p if v else None for p,v in zip(uv,valid)],
                              predicted_uv_px=heatmap_uv(prediction.point_logits)[item].float().cpu().tolist()))
        logits.append(prediction.point_logits.detach().cpu());targets.append(batch['targets'].to(device='cpu'));sources.extend(batch['sources'])
    if not targets:raise ValueError('no heldout goal batches')
    from dataclasses import fields
    combined=GoalTargets(**{f.name:torch.cat([getattr(t,f.name) for t in targets]) for f in fields(GoalTargets)})
    seen=getattr(expert,'bundle_metadata',{}).get('training',{}).get('valid_points_seen_at_selection')
    status=('disabled' if expert.lambda_kp==0 else 'no_effective_point_supervision' if seen==0
            else 'has_point_supervision' if seen is not None and seen>0 else 'unknown')
    report=dict(point_metrics=point_metrics(torch.cat(logits),combined,sources),review=cards,
                point_training_status=status,
                semantic_success='pending human review',point_order=['left','right'],samples=len(cards))
    (output_dir/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    render_goal_report(output_dir,report)
    return report
