"""Single-process streaming Stage A joint image/TCP training.

Only the current sample and one evaluation case are transferred to the model
device. Frozen text contexts live in a bounded CPU LRU; evaluation anchors are
fixed training-set diagnostics, never a held-out success estimate.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import replace
import hashlib
import html
import json
import math
import os
from pathlib import Path
import time

import torch

ROOT = Path(__file__).resolve().parents[2]


def load_config(path, *, source_yaml=None, data_root=None):
    """Merge relative base configs before interpolation; data paths use repo root."""
    from omegaconf import OmegaConf
    def read(filename, seen):
        filename=Path(filename).resolve()
        if filename in seen: raise ValueError(f'cyclic base_config: {filename}')
        config=OmegaConf.load(filename)
        base=config.pop('base_config',None)
        if base:
            config=OmegaConf.merge(read(filename.parent/base,seen|{filename}),config)
        return config
    config=OmegaConf.to_container(read(path,set()),resolve=True)
    representation = config.get('model', {}).get('goal_representation', 'patch_denoise')
    if representation not in ('patch_denoise', 'query_latent'):
        raise ValueError('unsupported model.goal_representation')
    if representation == 'query_latent':
        from openwam.model.goal.query_expert import QueryConfig
        QueryConfig(**config.get('model', {}).get('query_goal', {}))
    from openwam.model.goal.observation import OBSERVATION_MODES
    mode=config.get('observation_mode','head')
    if mode not in OBSERVATION_MODES or config.get('dataset',{}).get('observation_mode','head')!=mode:
        raise ValueError('Stage1 observation_mode must agree with dataset.observation_mode')
    if source_yaml is not None:config.setdefault('dataset',{})['source_yaml']=str(source_yaml)
    if data_root is not None:config.setdefault('dataset',{})['data_root']=str(data_root)
    for section,keys in ((config,('baseline_checkpoint',)),(config.get('dataset',{}),('source_yaml','cache_dir','data_root'))):
        for key in keys:
            if section.get(key):
                value=Path(os.path.expandvars(os.path.expanduser(str(section[key]))))
                section[key]=str(value if value.is_absolute() else ROOT/value)
    token=config.get('tokenizer',{}).get('kwargs',{})
    if token.get('name'):
        value=Path(os.path.expandvars(os.path.expanduser(token['name'])))
        token['name']=str(value if value.is_absolute() else ROOT/value)
    return config


def initialize_goal_checkpoint(checkpoint,*,vae,dit_config,observation_mode,
                               text_encoder=None,tokenizer=None,text_dim=None,
                               precomputed_text_fingerprint=None,device='cpu',dtype=torch.float32,
                               use_gradient_checkpointing=True,shift=None,lambda_image=None,lambda_tcp=None):
    """Checked v1 -> multiview initialization with fresh optimizer and flow settings.

    Omitted flow settings retain the source bundle values. Overrides rebuild the
    validated scheduler through PatchGoalExpert without copying the loaded DiT.
    """
    from safetensors import safe_open
    from openwam.train.goal_patch import load_patch_bundle
    from openwam.model.goal.patch_expert import PatchGoalExpert
    if observation_mode!='multiview':
        raise ValueError('--init-goal-checkpoint requires observation_mode=multiview')
    with safe_open(str(checkpoint),framework='pt',device='cpu') as file:
        meta=json.loads((file.metadata() or {}).get('patch_goal_bundle','{}'))
    if meta.get('dit_config')!=json.loads(json.dumps(dit_config)):
        raise ValueError('initialization checkpoint architecture differs from configured baseline')
    expert=load_patch_bundle(checkpoint,vae=vae,text_encoder=text_encoder,tokenizer=tokenizer,text_dim=text_dim,
        precomputed_text_fingerprint=precomputed_text_fingerprint,device=device,dtype=dtype,
        observation_mode=observation_mode,warm_start=True)
    loaded=expert
    expert=PatchGoalExpert(loaded.dit,loaded.codec,dit_config=loaded.dit_config,
        observation_mode=loaded.observation_mode,
        shift=loaded.shift if shift is None else shift,
        lambda_image=loaded.lambda_image if lambda_image is None else lambda_image,
        lambda_tcp=loaded.lambda_tcp if lambda_tcp is None else lambda_tcp,
        use_gradient_checkpointing=use_gradient_checkpointing).freeze()
    expert.bundle_metadata=loaded.bundle_metadata
    expert.initialization_metadata=loaded.initialization_metadata
    digest=hashlib.sha256()
    with Path(checkpoint).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    expert.initialization_metadata.update(source_checkpoint=str(Path(checkpoint).resolve()),
        source_checkpoint_sha256=digest.hexdigest(),optimizer='fresh FP32-master AdamW')
    return expert


def require_single_process():
    if int(os.environ.get('WORLD_SIZE','1'))!=1:
        raise ValueError('curated Stage A supports one process only; WORLD_SIZE must be 1')


class TextContextCache:
    """Cache at most max_entries CPU encodings (zero disables caching)."""
    def __init__(self,encode,*,max_entries=32):
        if not isinstance(max_entries,int) or max_entries<0:raise ValueError('text_cache_size must be a nonnegative integer')
        self.encode=encode;self.max_entries=max_entries;self.entries=OrderedDict()
    def __len__(self):return len(self.entries)
    @torch.no_grad()
    def get(self,prompt):
        if prompt in self.entries:
            value=self.entries.pop(prompt);self.entries[prompt]=value;return value
        context,lens=self.encode(prompt)
        value=(context.detach().cpu(),lens.detach().cpu() if isinstance(lens,torch.Tensor) else deepcopy(lens))
        if self.max_entries:
            self.entries[prompt]=value
            while len(self.entries)>self.max_entries:self.entries.popitem(last=False)
        return value


def prepare_batch(record,batch,*,device,dtype,context_cache=None):
    """Copy metadata and transfer this singleton only; retain FP32 UV precision."""
    result=dict(batch)
    result['observed']=batch['observed'].to(device=device,dtype=dtype)
    if 'observed_view_valid' in batch:
        result['observed_view_valid']=batch['observed_view_valid'].to(device=device)
    target=batch['targets'].to(device=device)
    result['targets']=replace(target,image=target.image.to(dtype=dtype))
    if context_cache is not None:
        context,lens=context_cache.get(record['instruction'])
    else:
        context,lens=batch['context'],batch.get('context_lens')
    result['context']=context.to(device=device,dtype=dtype)
    if lens is not None:result['context_lens']=lens.to(device) if isinstance(lens,torch.Tensor) else lens
    result.update(sources=[record.get('metric_key',record['family'])],family=record['family'],
                  granularity=record.get('granularity','task'),instruction=record['instruction'])
    return result


def _new_output(output_dir):
    output=Path(output_dir)
    if output.exists() and any(output.iterdir()):raise ValueError('nonempty output directory; choose a new run')
    output.mkdir(parents=True,exist_ok=True)
    return output


def _export_cases(output,pairs):
    from openwam.train.goal_overfit import export_inputs
    # export_inputs retains the precise pre-VAE BF16 bits. An empty context
    # explicitly indicates this data audit performs no text-model inference.
    batches=[]
    for _,batch in pairs:
        copy=dict(batch);copy.setdefault('context',torch.empty(0));batches.append(copy)
    return export_inputs(output,[r for r,_ in pairs],batches)


def preflight(dataset,output_dir):
    """Decode bounded real cases and export geometry/images, without any model."""
    from openwam.train.goal_overfit import _write_json
    require_single_process();output=_new_output(output_dir)
    pairs=list(dataset.evaluation_samples())
    if not pairs:raise ValueError('no evaluation samples selected')
    keys=[r.get('metric_key',r['family']) for r,_ in pairs]
    if len(keys)!=len(set(keys)):raise ValueError('evaluation metric keys must be unique')
    cards=_export_cases(output,pairs)
    summary=dict(mode='curated_preflight',model_loaded=False,scope=dataset.scope_summary(),
                 cases=len(cards),evaluation_split='train',success_verified=False)
    _write_json(output/'summary.json',summary)
    from openwam.train.goal_overfit import input_report_labels
    articles=[]
    for card in cards:
        observed_label,target_label,note=input_report_labels(card)
        name=card['id']
        articles.append('<article><h2>'+html.escape(str(card['family']))+' · '+html.escape(card.get('granularity','task'))+
            '</h2><p>'+html.escape(card['instruction'])+'</p>'+note+
            '<div class="images"><figure><figcaption>'+observed_label+'</figcaption><img src="inputs/'+name+'_observed.png"></figure>'+
            '<figure><figcaption>'+target_label+'</figcaption><img src="inputs/'+name+'_target_points.png"></figure></div>'+
            '<a href="inputs/'+name+'.npz">Exact input tensors and masks (NPZ)</a><details><summary>Source and same-anchor provenance</summary><pre>'+html.escape(json.dumps(card,ensure_ascii=False,indent=2))+'</pre></details></article>')
    body='<style>body{font:16px system-ui;margin:24px}.images{display:flex;gap:16px;align-items:flex-start}figure{margin:0}img{width:320px;height:auto}pre{white-space:pre-wrap}</style>'+''.join(articles)
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Curated preflight</title><h1>Actual model inputs and weak image/TCP targets</h1><a href="inputs/index.html">Exact inputs, view masks and NPZ</a><p>CPU data audit; no model weights loaded. Labels are weak and success is not verified. Left/right GT red/blue; invalid points masked. Evaluation cases are bounded training examples.</p><pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre>'+body)
    return summary


def train_curated(expert,*,dataset,context_cache,output_dir,steps=5000,eval_every=500,
                  sampling_steps=20,learning_rate=1e-5,seed=42,training_metadata=None,
                  dependency_metadata=None,dependency_identities=None,precomputed_text_fingerprint=None):
    """Stream indexed samples; keep immutable evaluation directories and history."""
    from openwam.train.goal_overfit import FP32MasterAdamW,_write_json
    from openwam.train.goal_trainer import _seed
    from openwam.train.goal_patch import evaluate_patch,save_patch_bundle
    from openwam.train.goal_history import render_goal_history
    require_single_process()
    query = getattr(expert, 'representation', 'patch_denoise') == 'query_latent'
    save_bundle = save_patch_bundle
    if query:
        from openwam.train.goal_query import save_query_bundle
        save_bundle = save_query_bundle
    checkpoint_name = 'query_goal.safetensors' if query else 'patch_goal.safetensors'
    if any(not isinstance(v,int) or v<1 for v in (steps,eval_every,sampling_steps)):raise ValueError('positive integer budgets required')
    if not math.isfinite(learning_rate) or learning_rate<=0:raise ValueError('positive finite learning rate required')
    output=_new_output(output_dir);started=time.monotonic()
    fixed=list(dataset.evaluation_samples())
    if not fixed:raise ValueError('no evaluation samples selected')
    keys=[r.get('metric_key',r['family']) for r,_ in fixed]
    if len(set(keys))!=len(keys):raise ValueError('evaluation metric keys must be unique')
    cards=_export_cases(output,fixed)
    parameter=next(expert.parameters());device,dtype=parameter.device,parameter.dtype
    _seed(seed)
    summary=dict(**(training_metadata or {}))
    summary.update(mode='stage1_joint_curated_streaming',status='running',steps=steps,completed_steps=0,
        representation='query_latent' if query else 'patch_denoise', checkpoint=checkpoint_name,
        observation_mode=expert.observation_mode,initialization=getattr(expert,'initialization_metadata',None),
        eval_every=eval_every,sampling_steps=sampling_steps,seed=seed,learning_rate=learning_rate,
        evaluations=[],scope=dataset.scope_summary(),evaluation_split='train',formal_generalization=False,
        success_verified=False,optimizer='FP32-master AdamW',single_process=True,
        records=[r for r,_ in fixed],valid_points_seen=0,
        sampling_policy='selected_repo_weight / selected_duration / available_language_level',
        source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                       (Path(__file__),ROOT/'scripts/train_goal_curated.py',ROOT/'openwam/dataloader/curated_goal.py') if p.is_file()})
    def event(kind,**values):
        entry=dict(event=kind,elapsed_s=time.monotonic()-started,**values)
        with (output/'progress.jsonl').open('a') as f:f.write(json.dumps(entry,allow_nan=False)+'\n')
        print(json.dumps(entry,allow_nan=False),flush=True)
    def publish(render=False):
        _write_json(output/'summary.json',summary)
        if render:
            render_goal_history(output,summary,cards)
            timeline=output/'timeline.html'
            timeline.write_text(timeline.read_text().replace('href="../index.html"','href="index.html"').replace('href="../g1_concept_and_plan.html"','href="method.html"'))
    (output/'method.html').write_text('<!doctype html><meta charset="utf-8"><h1>Stage A joint flow</h1><p>Clean observation and instruction condition shared Wan blocks. Terminal image and ordered TCP patch receive independent Gaussian noise and share sigma. TCP coordinates repeat 48 times; group means decode continuous UV without clamping. Both task/subtask supervision is weak. Frozen text uses a bounded CPU LRU. Single-process only; checkpoint reload/resume training is not implemented.</p>')
    if query:
        (output/'method.html').write_text('<!doctype html><meta charset="utf-8"><h1>Query latent Stage1</h1><p>Current observation and instruction condition learned image/TCP queries in one encoder forward. Only these tokens condition a training-only joint image/TCP flow decoder. Group readouts add masked image-latent and TCP regression supervision. All images and sampled points shown here are auxiliary joint-flow predictions. Stage2 loads only the frozen encoder, with no auxiliary generation. Single-process; Stage1 optimizer resume is not implemented.</p>')
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Curated Stage A diagnostics</h1><a href="timeline.html">All evaluation steps</a> · <a href="inputs/index.html">Actual inputs and view masks</a> · <a href="inputs/manifest.json">Input provenance</a> · <a href="summary.json">Scope/config</a> · <a href="method.html">Method</a><p>Fixed bounded training examples; no held-out performance or task-success claim.</p>')
    def evaluate(step):
        event('evaluation_started',step=step)
        directory=f'eval_step_{step:06d}'
        # A generator ensures evaluate_patch only owns one accelerator case.
        batches=(prepare_batch(r,b,device=device,dtype=dtype,context_cache=context_cache) for r,b in fixed)
        result=evaluate_patch(expert,batches,output/directory,step=step,seed=seed+100000,num_steps=sampling_steps)
        summary['evaluations'].append(dict(step=step,directory=directory,
            **{k:result[k] for k in ('point_metrics','group_metrics','image_mae_uint8')}))
        publish(True);event('report_ready',step=step,directory=directory)
    publish(True)
    try:
        evaluate(0)
        optimizer=FP32MasterAdamW([dict(params=list(expert.parameters()),lr=learning_rate)])
        for step in range(1,steps+1):
            _seed(seed+step);expert.train_stage_a();optimizer.zero_grad()
            record,cpu_batch=dataset.sample(step-1)
            batch=prepare_batch(record,cpu_batch,device=device,dtype=dtype,context_cache=context_cache)
            out=expert(batch['observed'],batch['context'],batch['targets'],context_lens=batch.get('context_lens'),
                       observed_view_valid=batch.get('observed_view_valid'))
            if not torch.isfinite(out['loss']):raise FloatingPointError('nonfinite training loss')
            out['loss'].backward()
            norm=torch.nn.utils.clip_grad_norm_(expert.parameters(),1.,error_if_nonfinite=True,foreach=False)
            if not any(p.grad is not None for p in expert.dit.parameters()):raise RuntimeError('no shared DiT gradients')
            optimizer.step()
            summary['completed_steps']=step;summary['valid_points_seen']+=int(out['valid_points'])
            last=dict(step=step,sample_id=record.get('id'),metric_key=record.get('metric_key'),family=record['family'],
                granularity=record.get('granularity','task'),
                source_repo=record.get('source_repo'),source_episode=record.get('source_episode'),
                anchor_source_frame=record.get('anchor_source_frame'),goal_source_frame=record.get('goal_source_frame'),
                instruction_start=record.get('instruction_start'),instruction_end_exclusive=record.get('instruction_end_exclusive'),
                provenance=record.get('provenance',{}),
                loss=float(out['loss'].detach()),fm_loss=float(out['fm_loss'].detach()),tcp_loss=float(out['tcp_loss'].detach()),
                valid_points=int(out['valid_points']),gradient_norm=float(norm))
            summary['last_step']=last
            if query:
                last.update(query_image_loss=float(out['query_image_loss'].detach()),
                            query_tcp_loss=float(out['query_tcp_loss'].detach()))
            del out,batch,cpu_batch,record
            event('train_step',**last);publish()
            if step%eval_every==0 or step==steps:
                evaluate(step)
                save_bundle(expert,output/checkpoint_name,training_metadata={**summary,'selected_step':step},
                    dependency_metadata=dependency_metadata,dependency_identities=dependency_identities,
                    precomputed_text_fingerprint=precomputed_text_fingerprint)
                event('checkpoint_saved',step=step)
        summary.update(status='complete',elapsed_s=time.monotonic()-started);publish(True)
        event('training_finished',completed_steps=steps)
    except Exception as exc:
        summary.update(status='failed',error=f'{type(exc).__name__}: {exc}');publish(True)
        event('training_failed',error=summary['error']);raise
    return summary
