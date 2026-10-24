"""Real G1 training-set overfit on one source segment per dataset family.

This diagnostic does not estimate heldout performance or task success. Geometry
validity, frame inclusion and visibility are deliberately independent fields.
"""
from __future__ import annotations

import hashlib
import html
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from openwam.model.goal.types import GoalTargets
from openwam.model.goal.heads import point_mask, target_uv_pixels
from openwam.train.goal_trainer import (
    _seed, evaluate_goals, overlay_goal_targets, save_goal_bundle,
)


class FP32MasterAdamW:
    """Accumulate small updates in FP32 while the forward model stays BF16.

    Only caller-supplied owned parameters get master copies. Sequential gradient
    transfer releases each model gradient immediately; foreach=False avoids a
    whole-model intermediate allocation. Both gradient sets are absent at return
    from step(), including before generation/checkpointing.
    """
    def __init__(self, parameter_groups, *, weight_decay=.01):
        self.parameter_pairs = []
        groups = []
        for group in parameter_groups:
            masters = []
            for parameter in group['params']:
                master = torch.nn.Parameter(parameter.detach().to(dtype=torch.float32, copy=True))
                self.parameter_pairs.append((parameter, master))
                masters.append(master)
            groups.append({**group, 'params': masters})
        self.master_parameters = [master for _, master in self.parameter_pairs]
        self.optimizer = torch.optim.AdamW(groups, weight_decay=weight_decay, foreach=False)

    def zero_grad(self, set_to_none=True):
        if not set_to_none:
            raise ValueError('FP32 master optimizer requires set_to_none=True to bound memory')
        for parameter, _ in self.parameter_pairs:
            parameter.grad = None
        self.optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def step(self):
        for parameter, master in self.parameter_pairs:
            gradient = parameter.grad
            master.grad = None if gradient is None else gradient.detach().to(dtype=torch.float32)
            parameter.grad = None
            del gradient
        self.optimizer.step()
        for parameter, master in self.parameter_pairs:
            parameter.copy_(master)
        self.zero_grad(set_to_none=True)


def overfit_eval_every(config):
    """Overfit has its own budget; generic Stage A settings cannot change it."""
    return config.get('overfit_eval_every', 120)


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def validate_overfit_manifest(manifest, *, expected_families=12,
                              diagnostic_override=False, allow_unreviewed_geometry=False):
    """Validate all input assets before allocating the baseline model."""
    if not diagnostic_override:
        raise ValueError('overfit requires --diagnostic-override; training-set evaluation only')
    if not allow_unreviewed_geometry:
        raise ValueError('overfit numerical geometry requires --allow-unreviewed-geometry')
    path = Path(manifest)
    rows = json.loads(path.read_text())
    if not isinstance(rows, list) or len(rows) != expected_families:
        raise ValueError(f'overfit requires a JSON list with exactly {expected_families} families')
    families = set()
    required = {'family', 'source_repo', 'source_episode', 'source_episode_length',
                'anchor_source_frame', 'goal_source_frame', 'clip_start', 'clip_end_exclusive',
                'observed_image_path', 'goal_image_path', 'target_path', 'instruction',
                'label_kind', 'success_verified', 'provenance'}
    for row in rows:
        if not isinstance(row, dict) or required - row.keys():
            raise ValueError(f'manifest missing required fields: {required - row.keys() if isinstance(row, dict) else required}')
        family = row['family']
        if not isinstance(family, str) or not family or family in families:
            raise ValueError('exactly one case per family required; duplicate/invalid family')
        families.add(family)
        for key in ('source_episode', 'source_episode_length', 'anchor_source_frame',
                    'goal_source_frame', 'clip_start', 'clip_end_exclusive'):
            if type(row[key]) is not int or row[key] < 0:
                raise ValueError(f'{family}: {key} must be a nonnegative integer')
        if row['source_episode_length'] < 2 or row['goal_source_frame'] != row['source_episode_length'] - 1:
            raise ValueError(f'{family}: goal must be the full source episode last frame')
        if not (0 <= row['clip_start'] <= row['anchor_source_frame'] < row['clip_end_exclusive'] <= row['source_episode_length']):
            raise ValueError(f'{family}: invalid source clip/anchor bounds')
        if row['anchor_source_frame'] >= row['goal_source_frame']:
            raise ValueError(f'{family}: observation must precede episode-terminal goal')
        if row['label_kind'] != 'episode_end' or row['success_verified'] is not False:
            raise ValueError('episode_end labels must never claim verified success')
        if not isinstance(row['instruction'], str) or not row['instruction'].strip():
            raise ValueError(f'{family}: nonempty instruction required')
        if not Path(row['source_repo']).is_absolute() or not isinstance(row['provenance'], dict):
            raise ValueError(f'{family}: absolute source_repo and provenance object required')
        for key in ('observed_image_path', 'goal_image_path', 'target_path'):
            asset = Path(row[key])
            if not asset.is_absolute():
                asset = path.parent / asset
            row[key] = str(asset.resolve())
            if not asset.is_file():
                raise ValueError(f'{family}: missing {key}: {asset}')
        for key in ('observed_image_path', 'goal_image_path'):
            with Image.open(row[key]) as image:
                if image.mode != 'RGB' or min(image.size) < 2:
                    raise ValueError(f'{family}: native RGB image required for {key}')
                image.load()
        with np.load(row['target_path'], allow_pickle=False) as data:
            for key, shape in [('uv', (2, 2)), ('projection_valid', (2,)), ('in_frame', (2,)), ('visibility', (2,))]:
                if key not in data or data[key].shape != shape:
                    raise ValueError(f'{family}: {key} must have shape {shape}')
            uv = data['uv']; valid = data['projection_valid']; inside = data['in_frame']
            if not np.isin(valid, [False, True]).all() or not np.isin(inside, [False, True]).all():
                raise ValueError(f'{family}: validity and inclusion masks must be boolean')
            if not np.isin(data['visibility'], [-1, 0, 1]).all():
                raise ValueError(f'{family}: visibility must be -1/0/1')
            if not np.isfinite(uv[valid.astype(bool)]).all():
                raise ValueError(f'{family}: valid projection coordinates must be finite')
            active = valid.astype(bool) & inside.astype(bool)
            if np.any((uv[active] < 0) | (uv[active] > 1)):
                raise ValueError(f'{family}: declared in-frame UV must be normalized to [0,1]')
    return rows


def _image_tensor(path, device, dtype, *, size=(320,256)):
    with Image.open(path) as image:
        array = np.array(image.convert('RGB').resize(size, Image.Resampling.BILINEAR), copy=True)
    return (torch.from_numpy(array).permute(2, 0, 1)[None, :, None].float() / 127.5 - 1).to(device=device, dtype=dtype)


def prepare_overfit_images(records, *, device, dtype):
    """Cache actual, unaugmented model images; UV remains normalized float32."""
    batches = []
    for row in records:
        with np.load(row['target_path'], allow_pickle=False) as data:
            targets = GoalTargets(
                _image_tensor(row['goal_image_path'], device, dtype),
                torch.ones(1, device=device, dtype=torch.bool),
                torch.as_tensor(data['uv'].copy(), device=device, dtype=torch.float32)[None],
                torch.as_tensor(data['projection_valid'].copy(), device=device, dtype=torch.bool)[None],
                torch.as_tensor(data['in_frame'].copy(), device=device, dtype=torch.bool)[None],
                torch.as_tensor(data['visibility'].copy(), device=device, dtype=torch.int8)[None])
        mode=row.get('observation_mode','head')
        if mode=='multiview':
            from openwam.model.goal.observation import sanitize_canvas, validate_view_valid
            valid=validate_view_valid(torch.as_tensor(row['view_valid'],device=device)[None],batch_size=1)
            with Image.open(row['observed_image_path']) as image:
                if image.size!=(320,384):raise ValueError('multiview observation asset must be W320 H384')
            observed=sanitize_canvas(_image_tensor(row['observed_image_path'],device,dtype,size=(320,384)),valid)
            batches.append(dict(observed=observed,observed_view_valid=valid,targets=targets,sources=[row['family']]))
        elif mode=='head':
            batches.append(dict(observed=_image_tensor(row['observed_image_path'], device, dtype),
                                targets=targets, sources=[row['family']]))
        else:raise ValueError(f'unsupported observation_mode: {mode}')
    return batches


def cache_text_contexts(batches, records, *, text_encoder, tokenizer, device, dtype):
    from openwam.model.video_backbone.wan.encode import encode_text
    text_encoder.eval().requires_grad_(False)
    contexts = {}
    with torch.no_grad():
        for batch, row in zip(batches, records):
            prompt = row['instruction']
            if prompt not in contexts:
                context, lens = encode_text([prompt], tokenizer=tokenizer,
                                           text_encoder=text_encoder, device=device)
                contexts[prompt] = (context.detach().to(device=device, dtype=dtype), lens)
            batch['context'], batch['context_lens'] = contexts[prompt]
    return len(contexts)


def _display_image(tensor):
    array = ((tensor[0, :, 0].detach().float().cpu().clamp(-1, 1) + 1) * 127.5).byte()
    return Image.fromarray(array.permute(1, 2, 0).numpy())


def input_report_labels(card):
    """Visible dimensions/view availability shared by preflight and input audit."""
    multiview=card.get('observation_mode','head')=='multiview'
    label='Current three-view input (H384 × W320)' if multiview else 'Current head input (H256 × W320)'
    status=''
    if multiview:
        from openwam.model.goal.observation import VIEW_NAMES
        valid=card['observed_view_valid'][0]
        status=' · '.join(f'{name}: {"valid" if present else "missing"}' for name,present in zip(VIEW_NAMES,valid))
    anchor=card.get('anchor_source_frame')
    note='<p>'+html.escape(status)+'</p>' if status else ''
    if anchor is not None:note+='<p>Same anchor source frame: '+html.escape(str(anchor))+'</p>'
    return label,'Head-only endpoint GT (H256 × W320)',note


def export_inputs(output_dir, records, batches):
    destination = Path(output_dir) / 'inputs'; destination.mkdir(exist_ok=True)
    cards = []
    for index, (row, batch) in enumerate(zip(records, batches)):
        name = f'case_{index:04d}'
        target = batch['targets']; valid = point_mask(target)[0].cpu().tolist()
        uv = target_uv_pixels(target)[0].cpu().tolist()
        _display_image(batch['observed']).save(destination / f'{name}_observed.png')
        goal = _display_image(target.image)
        goal.save(destination / f'{name}_target.png')
        overlay_goal_targets(goal, uv, valid).save(destination / f'{name}_target_points.png')
        arrays = dict(observed=batch['observed'].detach().float().cpu().numpy(),
                      target=target.image.detach().float().cpu().numpy(),
                      uv=target.uv.detach().float().cpu().numpy(),
                      projection_valid=target.point_valid.cpu().numpy(),
                      in_frame=target.in_frame.cpu().numpy(), visibility=target.visibility.cpu().numpy(),
                      context=batch['context'].detach().float().cpu().numpy(),
                      original_image_dtype=str(target.image.dtype))
        view_metadata={}
        if 'history_images' in batch:
            from openwam.model.goal.history import history_kwargs
            for key,value in history_kwargs(batch).items():
                tensor=value.detach().cpu()
                arrays[key]=tensor.float().numpy() if tensor.is_floating_point() else tensor.numpy()
                if tensor.dtype==torch.bfloat16:
                    arrays[key+'_bfloat16_bits']=tensor.view(torch.uint16).numpy()
            view_metadata['history_ages']=batch['history_ages'].cpu().tolist()
            view_metadata['history_valid']=batch['history_valid'].cpu().tolist()
            for slot in range(batch['history_images'].shape[1]):
                _display_image(batch['history_images'][:,slot]).save(destination/f'{name}_history_{slot:02d}.png')
        if 'observed_view_valid' in batch:
            from openwam.model.goal.observation import observation_token_mask
            mask=batch['observed_view_valid'].detach().cpu()
            arrays['observed_view_valid']=mask.numpy()
            arrays['observation_token_valid']=observation_token_mask(mask).numpy()
            view_metadata.update(observation_mode='multiview',observed_view_valid=mask.tolist(),
                               observed_shape=list(batch['observed'].shape),
                               valid_observation_tokens=arrays['observation_token_valid'].sum(axis=1).tolist())
        # Float32 conversion is exact for every BF16 value. Raw bits are also retained.
        if batch['observed'].dtype == torch.bfloat16:
            arrays['observed_bfloat16_bits'] = batch['observed'].detach().cpu().view(torch.uint16).numpy()
        if target.image.dtype == torch.bfloat16:
            arrays['target_bfloat16_bits'] = target.image.detach().cpu().view(torch.uint16).numpy()
        np.savez_compressed(destination / f'{name}.npz', **arrays)
        reasons = []
        for side in range(2):
            reason = []
            if not bool(target.point_valid[0, side]): reason.append('projection_invalid')
            if not bool(target.in_frame[0, side]): reason.append('outside_image')
            if not bool(torch.isfinite(target.uv[0, side]).all()): reason.append('nonfinite_coordinate')
            reasons.append(reason)
        provenance = dict(row.get('provenance', {}))
        if 'id' in row:
            provenance['manifest_id'] = row['id']
        # Real manifests may already carry id, visibility, or other report fields.
        # Computed report values win without duplicate keyword arguments.
        cards.append({**row, **view_metadata, 'id': name, 'valid_points': valid, 'mask_reasons': reasons,
                      'visibility': target.visibility[0].cpu().tolist(), 'provenance': provenance,
                      'target_uv_pixels': [point if ok else None for point, ok in zip(uv, valid)]})
    _write_json(destination / 'manifest.json', cards)
    panels=[]
    for card in cards:
        name=card['id']
        observed_label,target_label,note=input_report_labels(card)
        if 'history_ages' in card:
            note+='<p>Past head frames, oldest first; ages are realized seconds before the current frame. Invalid slots are masked.</p><div class="images" style="overflow-x:auto">'
            for slot,(age,ok) in enumerate(zip(card['history_ages'][0],card['history_valid'][0])):
                label=f'{age:.3f} s ago' if ok else 'PAD — masked'
                note+=f'<figure><figcaption>{label}</figcaption><img style="width:160px" src="{name}_history_{slot:02d}.png"></figure>'
            note+='</div>'
        panels.append('<article><h2>'+html.escape(card['family'])+'</h2><p>'+html.escape(card['instruction'])+
            '</p>'+note+'<div class="images"><figure><figcaption>'+observed_label+'</figcaption><img src="'+name+'_observed.png"></figure>'+
            '<figure><figcaption>'+target_label+'</figcaption><img src="'+name+'_target_points.png"></figure></div>'+
            '<a href="'+name+'.npz">Exact input tensors and masks (NPZ)</a><details><summary>Source and same-anchor provenance</summary><pre>'+html.escape(json.dumps(card,ensure_ascii=False,indent=2))+'</pre></details></article>')
    (destination/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Actual model inputs</title>'+
        '<style>body{font:16px system-ui;margin:24px}.images{display:flex;gap:16px;align-items:flex-start}figure{margin:0}img{width:320px;height:auto}pre{white-space:pre-wrap}</style>'+
        '<h1>Actual model inputs and view masks</h1><p>Unaugmented current frames; goals and TCP remain in the head view. Float32 NPZ values exactly preserve BF16 values; raw BF16 bits are included when used.</p>'+''.join(panels))
    return cards


def render_overfit_report(output_dir, summary, cards):
    output_dir = Path(output_dir)
    from openwam.train.goal_history import render_goal_history
    render_goal_history(output_dir, summary, cards)
    evaluations = summary['evaluations']
    latest = evaluations[-1]['directory'] if evaluations else None
    links = ' · '.join(f'<a href="{html.escape(item["directory"])}/index.html">step {item["step"]}</a>' for item in evaluations)
    articles = []
    for index, card in enumerate(cards):
        name = card['id']
        valid = card['valid_points']; supported = all(valid)
        note = ('左右 TCP GT 均有效。' if supported else
                '本样例无有效 TCP GT：本轮点损失为零，不能评估点预测。' if not any(valid) else
                '本样例仅部分 TCP GT 有效：缺失侧点损失被屏蔽。')
        note += ' ' + html.escape(card.get('audit_note',''))
        panels = [('实际模型输入（pre-VAE）', f'inputs/{name}_observed.png'),
                  ('实际目标 + 几何 GT 候选' if any(valid) else '实际目标（无有效 TCP GT）', f'inputs/{name}_target_points.png')]
        raw_link = ''
        if latest:
            panels.append(('真实生成目标 + 模型预测点' if supported else '真实生成目标（GT 不完整，默认不叠加预测点）', f'{latest}/{index:04d}_00_generated.png' if supported else f'{latest}/{index:04d}_00_generated_clean.png'))
            if not supported:
                raw_link = f'<p><a href="{latest}/{index:04d}_00_generated.png">查看未经完整 GT 验证的原始预测点（诊断）</a></p>'
        figures = ''.join(f'<figure><figcaption>{title}</figcaption><img src="{src}" width="320" height="256"></figure>' for title, src in panels)
        details = {key: card[key] for key in ('source_repo', 'source_episode', 'source_episode_length',
                   'anchor_source_frame', 'goal_source_frame', 'clip_start', 'clip_end_exclusive',
                   'instruction', 'valid_points', 'mask_reasons', 'visibility', 'target_uv_pixels', 'provenance')}
        articles.append(f'<article><h2>{html.escape(card["family"])}</h2><p>{note}</p><div class="panels">{figures}</div>{raw_link}'
                        f'<a href="inputs/{name}.npz">实际输入与目标张量 NPZ</a><pre>{html.escape(json.dumps(details, ensure_ascii=False, indent=2))}</pre></article>')
    history = []
    for evaluation in reversed(evaluations):
        directory = html.escape(evaluation['directory'])
        rows = []
        for index, card in enumerate(cards):
            supported = all(card['valid_points'])
            suffix = 'generated.png' if supported else 'generated_clean.png'
            note = '预测圆圈；GT 方框' if supported else 'GT 不完整：仅展示生成图像；原始预测点见诊断链接'
            rows.append(f'<section><h3>{html.escape(card["family"])}</h3><p>{note}</p><div class="panels">'
                        f'<figure><figcaption>实际目标 + 有效 GT</figcaption><img loading="lazy" width="320" height="256" src="inputs/{card["id"]}_target_points.png"></figure>'
                        f'<figure><figcaption>第 {evaluation["step"]} 步预测</figcaption><img loading="lazy" width="320" height="256" src="{directory}/{index:04d}_00_{suffix}"></figure>'
                        f'</div><a href="{directory}/index.html">本轮完整指标与诊断</a></section>')
        metrics = {key: evaluation[key] for key in ('point_metrics', 'image_mae_uint8') if key in evaluation}
        history.append(f'<details id="history-step-{evaluation["step"]}"><summary>第 {evaluation["step"]} 步 · {len(cards)} 类数据预测</summary>'
                       f'<pre>{html.escape(json.dumps(metrics, ensure_ascii=False, indent=2))}</pre>{"".join(rows)}</details>')
    display = {key: value for key, value in summary.items() if key not in ('evaluations', 'config', 'records')}
    if evaluations: display['latest_training_set_evaluation'] = evaluations[-1]
    page = ('<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<meta http-equiv="refresh" content="30"><title>G1 per-dataset overfit</title>'
            '<style>body{font:16px/1.5 system-ui;max-width:1100px;margin:24px auto;padding:0 16px}.panels{display:flex;flex-wrap:wrap;gap:12px}figure{margin:0}img{max-width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}article{border-top:1px solid #aaa;margin-top:24px}</style>'
            '<h1>G1：每个数据集一个片段的真实训练集 overfit</h1>'
            '<p><a href="timeline.html"><strong>按数据集连续对比所有步数：GT → 500 → 1000 → …</strong></a></p>'
            '<p>共享 G1 expert；无增强。目标始终是完整 source episode 的最后一帧。这里评估训练案例，不能声称 heldout 泛化或任务成功。'
            '几何是未人工核验的数值候选；GT 左红右蓝方框，预测为圆圈。visibility 独立记录，不替代投影有效性或画面内掩码。</p>'
            f'<p>{links}</p><h2>历史预测（展开查看，保留每一轮）</h2>{"".join(history)}'
            f'<pre>{html.escape(json.dumps(display, ensure_ascii=False, indent=2))}</pre>' + ''.join(articles) + '</html>')
    temporary = output_dir / 'index.html.tmp'; temporary.write_text(page); temporary.replace(output_dir / 'index.html')


def train_overfit(expert, *, batches, records, output_dir, steps=1200, eval_every=120,
                  sampling_steps=20, backbone_lr=1e-5, point_head_lr=1e-3, seed=42,
                  training_metadata=None, dependency_metadata=None, dependency_identities=None,
                  precomputed_text_fingerprint=None):
    if min(steps, eval_every, sampling_steps) < 1 or min(backbone_lr, point_head_lr) <= 0:
        raise ValueError('positive training/evaluation budgets and learning rates required')
    if not batches or len(batches) != len(records):
        raise ValueError('one prepared batch per manifest case required')
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / 'progress.jsonl').exists():
        raise ValueError('output already has progress.jsonl; choose a new run directory')
    started = time.monotonic(); expert.lambda_kp = .1
    summary = dict(mode='G1_per_dataset_overfit', status='running', steps=steps, completed_steps=0,
                   families=[row['family'] for row in records], seed=seed, lambda_kp=.1,
                   backbone_learning_rate=backbone_lr, point_head_learning_rate=point_head_lr,
                   eval_every=eval_every, sampling_steps=sampling_steps, evaluation_split='train',
                   formal_generalization=False, task_success='not measured', verified_success_pairs=0,
                   representation='RGB head 320x256; isolated-frame VAE latent [48,1,16,20]; normalized left/right UV',
                   augmentation=False, sampling='round robin; one fixed observed anchor per representative segment',
                   valid_points_seen=0, supported_points_per_case=[int(point_mask(b['targets']).sum()) for b in batches],
                   evaluations=[], **(training_metadata or {}))
    owned = list(expert.parameters())
    count = sum(p.numel() for p in owned)
    model_bytes = sum(p.numel()*p.element_size() for p in owned)
    summary['optimizer'] = dict(name='AdamW with FP32 master weights', model_dtype=str(owned[0].dtype),
        master_dtype='torch.float32', moment_dtype='torch.float32', owned_parameters=count,
        model_bytes=model_bytes, master_bytes=4*count, moment_bytes=8*count,
        fp32_gradient_bytes=4*count, estimated_parameter_peak_bytes=model_bytes+16*count,
        estimate_excludes='VAE, cached inputs, activations, allocator reserve, per-tensor optimizer temporary buffers',
        gradient_lifetime='BF16 gradients released during sequential transfer; FP32 gradients released after step')
    del owned
    cards = export_inputs(output_dir, records, batches)
    def event(kind, **values):
        entry = dict(event=kind, elapsed_s=time.monotonic()-started, **values)
        with (output_dir/'progress.jsonl').open('a') as stream:
            stream.write(json.dumps(entry, allow_nan=False)+'\n'); stream.flush()
        print(json.dumps(entry), flush=True)
    def publish():
        _write_json(output_dir/'summary.json', summary)
        render_overfit_report(output_dir, summary, cards)
    def evaluate(step):
        event('evaluation_started', step=step)
        expert.bundle_metadata = {'training': {'valid_points_seen_at_selection': summary['valid_points_seen']}}
        directory = f'eval_step_{step:06d}'
        result = evaluate_goals(expert, batches, output_dir/directory, seed=seed+100000, num_steps=sampling_steps)
        result.update(evaluation_split='train', formal_generalization=False, task_success='not measured', step=step)
        for card, record in zip(result['review'], cards):
            card['provenance'] = record
            card['geometry_notes'] = dict(mask_reasons=record['mask_reasons'], visibility=record['visibility'])
            with Image.open(output_dir/directory/f'{card["id"]}_generated_clean.png') as generated:
                prediction_pixels = np.asarray(generated, dtype=np.float32)
            with Image.open(output_dir/directory/f'{card["id"]}_target.png') as target_image:
                target_pixels = np.asarray(target_image, dtype=np.float32)
            # Pixel metric on rendered, uncluttered images; no semantic success inference.
            card['image_mae_uint8'] = float(np.abs(prediction_pixels-target_pixels).mean())
        result['image_mae_uint8'] = float(np.mean([c['image_mae_uint8'] for c in result['review']]))
        _write_json(output_dir/directory/'metrics.json', result)
        from openwam.train.goal_trainer import render_goal_report
        render_goal_report(output_dir/directory, result)
        summary['evaluations'].append(dict(step=step, directory=directory, point_metrics=result['point_metrics'], image_mae_uint8=result['image_mae_uint8']))
        event('evaluation_finished', step=step, point_metrics=result['point_metrics'], image_mae_uint8=result['image_mae_uint8'])
        publish()
        event('report_ready', step=step, directory=directory, report=str(output_dir/'index.html'),
              point_metrics=result['point_metrics'], image_mae_uint8=result['image_mae_uint8'])
    def save(step):
        event('checkpoint_started', step=step)
        path = output_dir/'goal.safetensors'
        temporary = output_dir/'goal.pending.safetensors'
        save_goal_bundle(expert, temporary, training_metadata={**summary, 'selected_step': step,
                         'selection_split': 'train', 'selection_rule': 'latest diagnostic step',
                         'valid_points_seen_at_selection': summary['valid_points_seen'], 'records': records},
                         dependency_metadata=dependency_metadata, dependency_identities=dependency_identities,
                         precomputed_text_fingerprint=precomputed_text_fingerprint)
        temporary.replace(path); temporary.with_suffix('.json').replace(path.with_suffix('.json'))
        summary['checkpoint'] = str(path)
        event('checkpoint_saved', step=step, checkpoint=str(path))
    publish(); event('training_started', families=summary['families'], steps=steps)
    try:
        evaluate(0)
        expert.train_stage_a()
        optimizer = FP32MasterAdamW([
            {'params': list(expert.dit.parameters()) + [expert.role_embedding], 'lr': backbone_lr},
            {'params': list(expert.point_head.parameters()), 'lr': point_head_lr}], weight_decay=.01)
        for step in range(1, steps+1):
            step_started = time.monotonic(); _seed(seed+step)
            # evaluate_goals calls freeze(); train() alone cannot re-enable gradients.
            expert.train_stage_a()
            index = (step-1) % len(batches); batch = batches[index]
            optimizer.zero_grad(set_to_none=True)
            out = expert(batch['observed'], batch['context'], batch['targets'], context_lens=batch.get('context_lens'))
            if not bool(torch.isfinite(out['loss'])):
                raise FloatingPointError(f'nonfinite loss at step {step}')
            out['loss'].backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(expert.parameters(), 1., error_if_nonfinite=True, foreach=False)
            if not any(p.grad is not None for p in expert.dit.parameters()):
                raise RuntimeError('DiT gradients absent: learning was frozen')
            optimizer.step()
            summary['completed_steps'] = step
            summary['valid_points_seen'] += int(out['valid_points'])
            summary['last_step'] = dict(step=step, family=records[index]['family'],
                fm_loss=float(out['fm_loss'].detach()), kp_loss=float(out['kp_loss'].detach()),
                loss=float(out['loss'].detach()), valid_points=int(out['valid_points']),
                gradient_finite=bool(torch.isfinite(gradient_norm)), gradient_norm=float(gradient_norm),
                step_seconds=time.monotonic()-step_started)
            event('train_step', **summary['last_step']); publish()
            del out
            if step % eval_every == 0 or step == steps:
                optimizer.zero_grad(set_to_none=True)
                evaluate(step); save(step); publish()
        summary.update(status='complete', elapsed_s=time.monotonic()-started)
        publish(); event('training_finished', completed_steps=steps)
    except Exception as exc:
        summary.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        publish(); event('training_failed', error=summary['error']); raise
    return summary


def run_overfit_cli(args, *, root):
    from omegaconf import OmegaConf
    from openwam.train.goal_trainer import bootstrap_components, resolve_tokenizer_config
    from openwam.model.goal.dependencies import dependency_identity
    if args.batch_size not in (None, 1):
        raise ValueError('--overfit uses exactly one case per step; --batch-size must be 1')
    if not args.manifest:
        raise ValueError('--overfit requires --manifest (JSON list)')
    records = validate_overfit_manifest(args.manifest, diagnostic_override=args.diagnostic_override,
                                       allow_unreviewed_geometry=args.allow_unreviewed_geometry)
    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    if args.baseline_checkpoint: config['baseline_checkpoint'] = args.baseline_checkpoint
    if args.tokenizer_path: config['tokenizer']['kwargs']['name'] = args.tokenizer_path
    config['tokenizer'] = resolve_tokenizer_config(config)
    dtype = getattr(torch, args.dtype or config.get('dtype', 'bfloat16'))
    _seed(args.seed)
    output = Path(args.output_dir or 'outputs/g1_per_dataset_overfit')
    if (output/'progress.jsonl').exists():
        raise ValueError('output already has progress.jsonl; choose a new run directory')
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output/'validated_manifest.json', records)
    expert, text_encoder, tokenizer = bootstrap_components(config, device=args.device, dtype=dtype)
    batches = prepare_overfit_images(records, device=args.device, dtype=dtype)
    unique_prompts = cache_text_contexts(batches, records, text_encoder=text_encoder, tokenizer=tokenizer,
                                        device=args.device, dtype=dtype)
    # No future text encoding is required for this fixed training set.
    text_encoder.to('cpu')
    identities = dependency_identity(expert.codec.vae, text_dim=expert.dit.text_embedding[0].in_features,
                                     text_encoder=text_encoder, tokenizer=tokenizer)
    del text_encoder, tokenizer
    if torch.device(args.device).type == 'cuda': torch.cuda.empty_cache()
    report = train_overfit(expert, batches=batches, records=records, output_dir=output,
        steps=args.steps if args.steps is not None else config.get('overfit_steps', 1200),
        eval_every=overfit_eval_every(config),
        sampling_steps=config.get('sampling_steps', 20),
        backbone_lr=config.get('overfit_backbone_lr', 1e-5),
        point_head_lr=config.get('overfit_point_head_lr', 1e-3), seed=args.seed,
        training_metadata=dict(config=config, cached_unique_prompts=unique_prompts,
                               diagnostic_override=True, geometry_review='unreviewed numerical candidates',
                               manifest_sha256=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
                               runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
        dependency_metadata={'components': config['components'], 'tokenizer': config['tokenizer']},
        dependency_identities=identities)
    print(json.dumps(report, indent=2), flush=True)
    return 0
