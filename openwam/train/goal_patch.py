"""Independent terminal-image / TCP patch joint-flow training and diagnostics."""
from __future__ import annotations
import hashlib
import html
import json
import time
from pathlib import Path
import numpy as np
import torch
from PIL import ImageDraw
from safetensors import safe_open
from safetensors.torch import save_file
from openwam.model.goal.codec import GoalCodec
from openwam.model.goal.dependencies import dependency_identity, module_identity, precomputed_identity, text_identity
from openwam.model.goal.heads import point_mask, target_uv_pixels
from openwam.model.goal.patch_expert import PatchGoalExpert
from openwam.model.goal.observation import OBSERVATION_MODES, observation_metadata
from openwam.model.video_backbone.wan.models.dit import WanModel
from openwam.train.goal_overfit import FP32MasterAdamW, _write_json, _display_image, export_inputs
from openwam.train.goal_trainer import _seed, overlay_goal_targets

FORMAT = 'openwam.goal.patch.v1'
MULTIVIEW_FORMAT = 'openwam.goal.patch.v2'
REFERENCE = dict(repository='https://github.com/Awomo-WestlakeDI/Awomo-0.5',
    commit='d22dda9aae0f9ae8096e95580a140a784a17b513', paper='https://arxiv.org/html/2609.25961v1',
    adaptation='Wan latent patch width 192; one TCP patch, 48 repeats per coordinate; independent encoded-space Gaussian noise as paper, unlike released 05I low-dimensional noisy-action encoding')


def save_patch_bundle(expert, checkpoint, *, training_metadata=None, dependency_metadata=None,
                      dependency_identities=None, precomputed_text_fingerprint=None):
    checkpoint=Path(checkpoint); checkpoint.parent.mkdir(parents=True,exist_ok=True)
    identities=dependency_identities or dependency_identity(expert.codec.vae,
        text_dim=expert.dit.text_embedding[0].in_features,
        precomputed_text_fingerprint=precomputed_text_fingerprint)
    root=Path(__file__).resolve().parents[2]
    files=['openwam/model/goal/observation.py','openwam/model/goal/patch_expert.py','openwam/train/goal_patch.py','scripts/train_goal_patch.py','openwam/dataloader/mixed_goal.py','openwam/train/mixed_metrics.py','openwam/train/goal_history.py']
    metadata=dict(format=FORMAT if expert.observation_mode == 'head' else MULTIVIEW_FORMAT,
        observation_mode=expert.observation_mode, observation=observation_metadata(expert.observation_mode),
        dit_config=expert.dit_config,codec=expert.codec.metadata(),
        shift=expert.shift,lambda_image=expert.lambda_image,lambda_tcp=expert.lambda_tcp,
        dependencies={**identities,'reconstruction':dependency_metadata or {}},
        training=training_metadata or {},reference=REFERENCE,
        source_hashes={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files},
        tcp_codec=dict(order=['uL','vL','uR','vR'],normalization='2*uv-1',repeats=48,
                       patch_width=192,decode='group mean; (x+1)/2; no clamp'),
        sampling=dict(method='joint Euler',noise='independent image / 192D TCP',return_latent='final_z0'))
    if hasattr(expert, 'initialization_metadata'):
        metadata['initialization'] = dict(expert.initialization_metadata)
    temporary=checkpoint.with_suffix('.pending.safetensors')
    save_file({k:v.detach().cpu().contiguous() for k,v in expert.state_dict().items()},str(temporary),
              metadata={'patch_goal_bundle':json.dumps(metadata,sort_keys=True)})
    temporary.replace(checkpoint)
    _write_json(checkpoint.with_suffix('.json'),metadata)
    return metadata


def load_patch_bundle(checkpoint, *, vae, text_encoder=None,tokenizer=None,text_dim=None,
                      precomputed_text_fingerprint=None,device='cpu',dtype=torch.float32,
                      observation_mode=None,warm_start=False):
    """Load the saved mode, or explicitly initialize multiview from a v1 bundle.

    warm_start=True only permits v1 head -> multiview, preserves all DiT weights,
    and applies the same strict frozen dependency identity checks as normal load.
    The returned expert stays frozen until train_stage_a() is requested.
    """
    if observation_mode is not None and observation_mode not in OBSERVATION_MODES:
        raise ValueError(f'unsupported observation_mode: {observation_mode}')
    with safe_open(str(checkpoint),framework='pt',device='cpu') as file:
        raw=(file.metadata() or {}).get('patch_goal_bundle')
        if raw is None: raise ValueError('missing patch goal bundle metadata')
        meta=json.loads(raw)
        if meta.get('format') not in (FORMAT, MULTIVIEW_FORMAT):
            raise ValueError('unsupported patch goal format')
        saved_mode = 'head' if meta['format'] == FORMAT else 'multiview'
        if meta.get('observation_mode', 'head') != saved_mode:
            raise ValueError('patch bundle format/observation mode mismatch')
        if saved_mode == 'multiview' or 'observation' in meta:
            expected = json.loads(json.dumps(observation_metadata(saved_mode)))
            if meta.get('observation') != expected:
                raise ValueError('incompatible patch observation contract')
        requested_mode = saved_mode if observation_mode is None else observation_mode
        if warm_start:
            if saved_mode != 'head' or requested_mode != 'multiview':
                raise ValueError('warm_start requires explicit v1 head -> multiview mode')
        elif requested_mode != saved_mode:
            raise ValueError('observation mode mismatch; v1 -> multiview requires explicit warm_start=True')
        dep=meta['dependencies']
        if getattr(text_encoder,'dim',text_dim)!=dep['text_dim']: raise ValueError('incompatible text dimension')
        actual=text_identity(text_encoder,tokenizer) if text_encoder is not None else precomputed_identity(precomputed_text_fingerprint)
        if actual['fingerprint']!=dep['text']['fingerprint']: raise ValueError('incompatible text dependency')
        if module_identity(vae)!=dep['vae']: raise ValueError('incompatible VAE dependency')
        if json.loads(json.dumps(GoalCodec(vae).metadata()))!=meta['codec']: raise ValueError('incompatible codec')
        expert=PatchGoalExpert(WanModel(**meta['dit_config']),GoalCodec(vae),dit_config=meta['dit_config'],
            shift=meta['shift'],lambda_image=meta['lambda_image'],lambda_tcp=meta['lambda_tcp'],
            observation_mode=requested_mode)
        expert.load_state_dict({k:file.get_tensor(k) for k in file.keys()},strict=True)
    expert.to(device=device,dtype=dtype).freeze(); expert.bundle_metadata=meta
    if warm_start:
        expert.initialization_metadata = dict(source_format=meta['format'], source_mode=saved_mode,
                                              observation_mode=requested_mode, trained_multiview=False)
    elif 'initialization' in meta:
        expert.initialization_metadata = dict(meta['initialization'])
    for module in (vae,text_encoder):
        if isinstance(module,torch.nn.Module):
            for tensor in list(module.parameters())+list(module.buffers()):
                if tensor.device!=next(expert.parameters()).device or (tensor.is_floating_point() and tensor.dtype!=dtype):
                    raise ValueError('place shared dependencies on requested device/dtype before load')
            module.eval().requires_grad_(False)
    return expert


def continuous_point_metrics(predicted_uv,targets,sources,threshold=16.):
    valid=point_mask(targets)
    if predicted_uv.shape!=targets.uv.shape or len(sources)!=len(predicted_uv): raise ValueError('UV/source shape mismatch')
    if not torch.isfinite(predicted_uv).all(): raise FloatingPointError('nonfinite predicted TCP')
    scale=predicted_uv.new_tensor([319.,255.])
    error=((predicted_uv.float()-torch.nan_to_num(targets.uv.float()))*scale).norm(dim=-1)
    groups={'overall':torch.ones_like(valid)}
    groups.update({'source:'+s:torch.tensor([x==s for x in sources],device=valid.device)[:,None].expand_as(valid) for s in sorted(set(sources))})
    metrics={}
    for name,mask in groups.items():
        metrics[name]={}
        for hand,side in enumerate(('left','right')):
            selected=error[:,hand][valid[:,hand]&mask[:,hand]]; count=selected.numel()
            metrics[name][side]=dict(count=count,pixel_error=float(selected.mean()) if count else None,
                pck=float((selected<=threshold).float().mean()) if count else None,pck_threshold_px=threshold)
    return metrics


def _prediction_overlay(image,uv_px):
    result=image.copy(); draw=ImageDraw.Draw(result)
    for (x,y),color in zip(uv_px,('red','blue')):
        # Out-of-frame predictions remain out of frame; never clamp to a border.
        if 0<=x<=319 and 0<=y<=255:
            draw.ellipse((x-4,y-4,x+4,y+4),outline=color,width=2)
    return result


@torch.no_grad()
def evaluate_patch(expert,batches,output_dir,*,step=0,seed=42,num_steps=20):
    expert.eval(); output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=False)
    cards=[]; metrics={}; aggregate=[[],[]]; maes=[]
    for index,b in enumerate(batches):
        if b['observed'].shape[0]!=1: raise ValueError('one case per batch required')
        prediction=expert.generate(b['observed'],b['context'],seed=seed+index,num_steps=num_steps,context_lens=b.get('context_lens'),observed_view_valid=b.get('observed_view_valid'))
        decoded=expert.codec.decode(prediction.latent)
        if not torch.isfinite(decoded).all(): raise FloatingPointError('nonfinite generated image')
        uv=prediction.uv.float(); target=b['targets']; valid=point_mask(target)[0].cpu().tolist()
        values=continuous_point_metrics(uv,target,b['sources'])
        metrics.update({k:v for k,v in values.items() if k!='overall'})
        for hand,side in enumerate(('left','right')):
            if values['overall'][side]['count']: aggregate[hand].append(values['overall'][side]['pixel_error'])
        pred_px=(uv[0]*uv.new_tensor([319.,255.])).cpu().tolist()
        gt_px=target_uv_pixels(target)[0].cpu().tolist()
        name=f'{index:04d}_00'; image=_display_image(decoded); goal=_display_image(target.image)
        image.save(output_dir/f'{name}_generated_clean.png'); goal.save(output_dir/f'{name}_target.png')
        _prediction_overlay(image,pred_px).save(output_dir/f'{name}_generated.png')
        overlay_goal_targets(goal,gt_px,valid).save(output_dir/f'{name}_target_points.png')
        mae=float(np.abs(np.asarray(image,dtype=np.float32)-np.asarray(goal,dtype=np.float32)).mean()); maes.append(mae)
        cards.append(dict(id=name,source=b['sources'][0],target_point_valid=valid,
            observed_shape=list(b['observed'].shape),
            observed_view_valid=b['observed_view_valid'].detach().cpu().tolist() if 'observed_view_valid' in b else None,
            target_uv_px=[p if v else None for p,v in zip(gt_px,valid)],predicted_uv_px=pred_px,
            predicted_in_frame=[bool(0<=x<=319 and 0<=y<=255) for x,y in pred_px],image_mae_uint8=mae,
            family=b.get('family',b['sources'][0]),granularity=b.get('granularity','task'),instruction=b.get('instruction',''),
            tcp_pixel_errors=[values['overall'][side]['pixel_error'] for side in ('left','right')]))
    metrics['overall']={side:dict(count=len(errors),pixel_error=float(np.mean(errors)) if errors else None,
        pck=float(np.mean(np.asarray(errors)<=16)) if errors else None,pck_threshold_px=16.)
        for side,errors in zip(('left','right'),aggregate)}
    result=dict(step=step,point_metrics=metrics,image_mae_uint8=float(np.mean(maes)),review=cards,
        evaluation_split='train',semantic_success='not measured',prediction='joint flow sampled TCP; continuous UV; no clamp')
    from openwam.train.mixed_metrics import aggregate_mixed_metrics, render_group_table
    result['group_metrics']=aggregate_mixed_metrics(cards)
    _write_json(output_dir/'metrics.json',result)
    articles=[]
    for card in cards:
        name=card['id']; supported=all(card['target_point_valid'])
        suffix='generated.png' if supported else 'generated_clean.png'
        sides=result['point_metrics'].get('source:'+card['source'],{})
        eligible=[v for v in sides.values() if v.get('count',0)>0 and v.get('pixel_error') is not None]
        count=sum(v['count'] for v in eligible)
        mean=(f"{sum(v['pixel_error']*v['count'] for v in eligible)/count:.2f} px" if count else '—（无有效 GT）')
        labels=f'<p data-case-error-metrics="true"><strong>图像 MAE：{card["image_mae_uint8"]:.2f} / 255 · TCP 平均误差：{mean}</strong></p>'
        figures=''.join(f'<figure><figcaption>{label}</figcaption><img width="320" height="256" src="{name}_{file}"></figure>' for label,file in [('实际目标 + 有效 GT 方框','target_points.png'),('联合生成图像 + TCP 圆圈' if supported else '生成图像；缺失 TCP GT，原始点仅供诊断',suffix)])
        articles.append(f'<article><h2>{html.escape(card["source"])}</h2><p>{html.escape(card.get("instruction",""))}</p>{labels}<div class="panels">{figures}</div><a href="{name}_generated.png">原始预测点诊断</a><pre>{html.escape(json.dumps(card,ensure_ascii=False,indent=2))}</pre></article>')
    (output_dir/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>Joint patch denoising</title><style>body{font:16px system-ui;margin:24px}.panels{display:flex;gap:16px}figure{margin:0}pre{white-space:pre-wrap}</style>'+f'<h1>联合去噪 · 第 {step} 步</h1><a href="../timeline.html">全部历史</a><p>TCP 来自去噪 patch 的固定分组平均解码；连续坐标，不裁剪到图像边界。GT 左红右蓝方框，预测圆圈。缺失 GT 不计算点误差。</p>'+render_group_table(result['group_metrics'])+''.join(articles)+'</html>')
    return result


def train_patch(expert,*,batches,records,output_dir,steps=5000,eval_every=500,sampling_steps=20,
                learning_rate=1e-5,seed=42,training_metadata=None,dependency_metadata=None,
                dependency_identities=None,precomputed_text_fingerprint=None):
    output_dir=Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()): raise ValueError('nonempty output directory; select a new run')
    if min(steps,eval_every,sampling_steps)<1 or learning_rate<=0: raise ValueError('positive budgets/LR required')
    if not batches or len(batches)!=len(records): raise ValueError('one prepared batch per record required')
    output_dir.mkdir(parents=True,exist_ok=True); started=time.monotonic()
    cards=export_inputs(output_dir,records,batches)
    schedule=None
    if (training_metadata or {}).get('sampling_policy')=='family_level_balanced':
        from openwam.dataloader.mixed_goal import balanced_schedule
        schedule=balanced_schedule(records,steps,seed)
        _write_json(output_dir/'sample_schedule.json',dict(indices=schedule,ids=[records[i]['id'] for i in schedule]))
    summary=dict(mode='G1_patch_joint_overfit',status='running',steps=steps,completed_steps=0,
        eval_every=eval_every,sampling_steps=sampling_steps,learning_rate=learning_rate,
        lambda_image=expert.lambda_image,lambda_tcp=expert.lambda_tcp,seed=seed,
        families=[r['family'] for r in records],evaluations=[],valid_points_seen=0,
        reference=REFERENCE,evaluation_split='train',formal_generalization=False,
        optimizer='FP32-master AdamW',trainable_parameters=sum(p.numel() for p in expert.parameters()),
        **(training_metadata or {}))
    def event(kind,**values):
        entry=dict(event=kind,elapsed_s=time.monotonic()-started,**values)
        with (output_dir/'progress.jsonl').open('a') as f: f.write(json.dumps(entry,allow_nan=False)+'\n')
        print(json.dumps(entry),flush=True)
    def publish(render=False):
        _write_json(output_dir/'summary.json',summary)
        if render:
            from openwam.train.goal_history import render_goal_history
            render_goal_history(output_dir,summary,cards)
            # Existing reusable history has parent links; adapt only this new artifact.
            path=output_dir/'timeline.html'; content=path.read_text().replace('href="../index.html"','href="index.html"').replace('href="../g1_concept_and_plan.html"','href="method.html"')
            path.write_text(content)
            (output_dir/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>TCP / image joint flow</title><h1>图像 + TCP patch 联合去噪</h1><a href="timeline.html">按数据集连续对比全部步数</a><p>独立 Stage A；原热图版本保留。训练集过拟合诊断，不代表泛化或任务成功。</p><a href="method.html">方法与配置</a></html>')
    (output_dir/'method.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>Joint patch method</title><h1>联合 patch 去噪</h1><p>干净观测 + 指令 → 目标图像噪声 patch / TCP 噪声 patch → 共享 Wan DiT → 图像 VAE 解码 / TCP 固定分组平均。TCP 顺序 uL,vL,uR,vR；归一化 2uv−1，各重复48次。每个192维噪声独立采样；两模态共享 sigma。无独立点预测头。</p><p>只训练 Stage A；后续 action/video expert 未训练。</p><pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre></html>')
    def evaluate(step):
        event('evaluation_started',step=step)
        directory=f'eval_step_{step:06d}'
        result=evaluate_patch(expert,batches,output_dir/directory,step=step,seed=seed+100000,num_steps=sampling_steps)
        summary['evaluations'].append(dict(step=step,directory=directory,point_metrics=result['point_metrics'],image_mae_uint8=result['image_mae_uint8'],group_metrics=result['group_metrics']))
        publish(True);event('report_ready',step=step,directory=directory,point_metrics=result['point_metrics'],image_mae_uint8=result['image_mae_uint8'])
    publish(True);event('training_started',steps=steps)
    try:
        evaluate(0);expert.train_stage_a()
        optimizer=FP32MasterAdamW([dict(params=list(expert.parameters()),lr=learning_rate)])
        for step in range(1,steps+1):
            _seed(seed+step);expert.train_stage_a();optimizer.zero_grad()
            index=schedule[step-1] if schedule is not None else (step-1)%len(batches);b=batches[index]; tick=time.monotonic()
            out=expert(b['observed'],b['context'],b['targets'],context_lens=b.get('context_lens'),observed_view_valid=b.get('observed_view_valid'))
            if not torch.isfinite(out['loss']):raise FloatingPointError('nonfinite training loss')
            out['loss'].backward()
            norm=torch.nn.utils.clip_grad_norm_(expert.parameters(),1.,error_if_nonfinite=True,foreach=False)
            if not any(p.grad is not None for p in expert.dit.parameters()):raise RuntimeError('no shared DiT gradients')
            optimizer.step()
            summary.update(completed_steps=step)
            summary['valid_points_seen']+=int(out['valid_points'])
            summary['last_step']=dict(step=step,family=records[index]['family'],granularity=records[index].get('granularity','task'),sample_id=records[index].get('id',str(index)),loss=float(out['loss'].detach()),
                fm_loss=float(out['fm_loss'].detach()),tcp_loss=float(out['tcp_loss'].detach()),
                valid_points=int(out['valid_points']),gradient_norm=float(norm),step_seconds=time.monotonic()-tick)
            del out;event('train_step',**summary['last_step']);publish()
            if step%eval_every==0 or step==steps:
                evaluate(step)
                save_patch_bundle(expert,output_dir/'patch_goal.safetensors',training_metadata={**summary,'selected_step':step},
                    dependency_metadata=dependency_metadata,dependency_identities=dependency_identities,
                    precomputed_text_fingerprint=precomputed_text_fingerprint)
                event('checkpoint_saved',step=step)
        summary.update(status='complete',elapsed_s=time.monotonic()-started);publish(True);event('training_finished',completed_steps=steps)
    except Exception as exc:
        summary.update(status='failed',error=f'{type(exc).__name__}: {exc}');publish(True);event('training_failed',error=summary['error']);raise
    return summary


def make_smoke_data(device='cpu',*,observation_mode='head'):
    from openwam.train.goal_trainer import TINY_DIT_CONFIG,_DiagnosticVAE,synthetic_smoke_batch
    _seed(42)
    expert=PatchGoalExpert(WanModel(**TINY_DIT_CONFIG),GoalCodec(_DiagnosticVAE()),dit_config=TINY_DIT_CONFIG,
                           observation_mode=observation_mode).to(device)
    obs,ctx,target=synthetic_smoke_batch(device)
    record=dict(family='synthetic',source_repo='synthetic',source_episode=0,source_episode_length=2,
        anchor_source_frame=0,goal_source_frame=1,clip_start=0,clip_end_exclusive=2,instruction='synthetic smoke',provenance={})
    batch=dict(observed=obs,context=ctx,targets=target,sources=['synthetic'])
    if observation_mode=='multiview':
        from openwam.model.goal.observation import sanitize_canvas
        mask=torch.tensor([[True,True,False]],device=device)
        batch['observed']=sanitize_canvas(torch.cat((obs,obs[:,:,:,:128]),dim=3),mask)
        batch['observed_view_valid']=mask
        record.update(observation_mode=observation_mode,view_valid=mask[0].cpu().tolist())
    return expert,[batch],[record]
