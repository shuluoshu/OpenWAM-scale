"""Frozen Stage1 inference on the exact Stage2 sample inputs, never index joins.

GT images/geometry are report-only and never enter the model batch. Stage1's
instruction endpoint is deliberately distinct from Stage2's 32-step horizon.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from PIL import Image

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE))
from scripts.experiments.stage1_tcp_report import tcp_overlay_data, draw_tcp_overlay


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def tensor_sha(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(str((str(value.dtype), tuple(value.shape))).encode() + value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def safe(value):
    if isinstance(value, torch.Tensor): return safe(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray): return safe(value.tolist())
    if isinstance(value, np.generic): return safe(value.item())
    if isinstance(value, dict): return {str(k):safe(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def dump(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(safe(value), ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def sample_identity(sample):
    trace = sample['trace']
    return dict(trace_identity=safe(trace['identity']), source_repo=trace['source_repo'], episode=trace['episode'],
        anchor=trace['anchor'], prompt=sample['prompt'], granularity=sample['granularity'],
        instruction_start=sample['instruction_start'], instruction_end_exclusive=sample['instruction_end_exclusive'],
        conditioning_sha256={k:tensor_sha(sample[k]) for k in ('goal_observed_multiview','goal_view_valid',
            'history_images','history_valid','history_ages','history_episode_id')})


def assert_same_case(sample, identity):
    if sample_identity(sample) != identity: raise ValueError('exact Stage2 sample identity/conditioning mismatch')


def make_batch(sample, contexts, *, device, dtype):
    """Allowlist conditioning fields; no future video, action or target accepted."""
    context, lengths = contexts[sample['prompt']]
    b = dict(observed=sample['goal_observed_multiview'][None].to(device=device,dtype=dtype),
             observed_view_valid=sample['goal_view_valid'][None].to(device=device),
             context=context.to(device=device,dtype=dtype), context_lens=lengths.to(device=device))
    for key in ('history_images','history_ages','history_valid','history_episode_id'):
        typ = torch.bool if key=='history_valid' else torch.int64 if key=='history_episode_id' else torch.float32 if key=='history_ages' else dtype
        b[key] = sample[key][None].to(device=device,dtype=typ)
    return b


def target_data(sample, uv):
    report = sample['report_target']
    if report['goal_source_frame'] != sample['instruction_end_exclusive']-1:
        raise ValueError('Stage1 GT must match exact instruction endpoint')
    projection = report.get('projection', {})
    target = dict(uv=projection.get('uv',np.zeros((2,2))),
        point_valid=projection.get('projection_valid',np.zeros(2,bool)),
        in_frame=projection.get('in_frame',np.zeros(2,bool)),
        image_valid=sample['_eval'].get('report_goal') is not None)
    return tcp_overlay_data(dict(provenance=report),target,dict(uv=uv))


@torch.no_grad()
def infer(expert, batch, *, seed, steps):
    from openwam.model.goal.history import history_kwargs
    from openwam.model.goal.patch_expert import decode_tcp_patch
    from openwam.model.video_backbone.wan.shared.diffusion.flow_match import FlowMatchScheduler
    from openwam.train.goal_overfit import _display_image
    c = expert.condition_tokens(batch['observed'],observed_view_valid=batch['observed_view_valid'],**history_kwargs(batch))
    obs = expert.encode_observation(batch['observed'],observed_view_valid=batch['observed_view_valid'])
    rng = torch.Generator(device=obs.device).manual_seed(seed)
    image = torch.randn((1,48,1,16,20),device=obs.device,dtype=obs.dtype,generator=rng)
    tcp = torch.randn((1,192),device=obs.device,dtype=torch.float32,generator=rng)
    scheduler = FlowMatchScheduler(template='Wan');scheduler.set_timesteps(steps,shift=expert.shift)
    for t in scheduler.timesteps:
        velocity,tcp_v=expert.velocity(obs,image,tcp,t.to(obs).expand(1),batch['context'],
            context_lens=batch['context_lens'],observed_view_valid=batch['observed_view_valid'],observation_tokens=c)
        image=scheduler.step(velocity,t,image);tcp=scheduler.step(tcp_v,t,tcp)
    return _display_image(expert.codec.decode(image)), decode_tcp_patch(tcp).float().cpu()[0].tolist()


def prepare_contexts(samples, smoke, cfg, deps, official, output):
    """Reuse exact-prompt contexts; encode missing prompts with verified real T5."""
    from openwam.train.goal_trainer import _construct,load_component_weights
    from openwam.model.goal.dependencies import text_identity,resolve_tokenizer_config
    from openwam.model.video_backbone.wan.encode import encode_text
    contexts=torch.load(smoke/'text_contexts.pt',map_location='cpu',weights_only=True)
    missing=list(dict.fromkeys(s['prompt'] for s in samples if s['prompt'] not in contexts))
    source=dict(existing_cache=str(smoke/'text_contexts.pt'),existing_cache_sha256=sha(smoke/'text_contexts.pt'),
                text_dependency_fingerprint=deps['text']['fingerprint'],new_prompts=missing)
    if missing:
        print(json.dumps(dict(event='loading_frozen_t5',missing_prompts=len(missing))),flush=True)
        text_encoder=_construct(cfg['components']['text_encoder'])
        load_component_weights(text_encoder,official,'video_backbone.text_encoder.')
        text_encoder.to(device='cuda',dtype=torch.bfloat16).eval().requires_grad_(False)
        token_cfg=resolve_tokenizer_config(cfg)
        tokenizer=_construct(dict(model_class=token_cfg['model_class'],extra_kwargs=token_cfg['kwargs']))
        actual=text_identity(text_encoder,tokenizer)
        if actual['fingerprint'] != deps['text']['fingerprint']: raise ValueError('T5 dependency fingerprint mismatch')
        with torch.no_grad():
            for prompt in missing:
                context,lens=encode_text([prompt],tokenizer=tokenizer,text_encoder=text_encoder,device='cuda')
                contexts[prompt]=(context.cpu(),lens.cpu())
        del text_encoder,tokenizer;gc.collect();torch.cuda.empty_cache()
    selected={s['prompt']:contexts[s['prompt']] for s in samples}
    torch.save(selected,output/'text_contexts.pt')
    source['output_sha256']=sha(output/'text_contexts.pt');dump(output/'text_contexts_source.json',source)
    return selected


def main():
    import yaml
    from openwam.train.goal_trainer import _construct,load_component_weights
    from openwam.train.goal_visual_history import load_visual_history_bundle
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--steps',type=int,default=20)
    a=p.parse_args();run=a.run;out=run/'report/stage1_samecase';out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    samples_path=run/'data/samples.pt';checkpoint=run/'stage1/visual_history.safetensors'
    samples=torch.load(samples_path,map_location='cpu',weights_only=False)
    protocol=json.loads((run/'stage1/protocol.json').read_text());smoke=Path(protocol['smoke_data'])
    cfg=yaml.safe_load((smoke/'config.yaml').read_text());deps=json.loads((smoke/'dependencies.json').read_text())
    official=Path(protocol['official_dependencies']);start=time.monotonic()
    manifest=dict(schema='openwam.stage1_samecase.v1',status='running',samples=str(samples_path),samples_sha256=sha(samples_path),
        checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),steps=a.steps,seed=100042,
        target_semantics='instruction_end_exclusive - 1; not Stage2 action horizon endpoint',cases=[])
    dump(out/'summary.json',manifest)
    identities=[sample_identity(s) for s in samples]
    for s in samples: target_data(s,[[0.,0.],[0.,0.]])
    contexts=prepare_contexts(samples,smoke,cfg,deps,official,out)
    print(json.dumps(dict(event='loading_frozen_stage1')),flush=True)
    vae=_construct(cfg['components']['vae']);load_component_weights(vae,official,'video_backbone.vae.')
    vae.to(device='cuda',dtype=torch.bfloat16).eval().requires_grad_(False)
    expert=load_visual_history_bundle(checkpoint,vae=vae,text_dim=deps['text_dim'],
        precomputed_text_fingerprint=deps['text']['fingerprint'],device='cuda',dtype=torch.bfloat16)
    expert.eval().requires_grad_(False)
    for i,s in enumerate(samples):
        assert_same_case(s,identities[i]);folder=out/f'case_{i:03d}';folder.mkdir(exist_ok=True)
        batch=make_batch(s,contexts,device='cuda',dtype=torch.bfloat16)
        prediction,uv=infer(expert,batch,seed=manifest['seed'],steps=a.steps)
        prediction.save(folder/'prediction.png');tcp=target_data(s,uv)
        draw_tcp_overlay(prediction,tcp,folder/'tcp_overlay.png','Stage1 | exact Stage2 current/history + prompt')
        target=s['_eval'].get('report_goal')
        if target is not None:
            target=target.convert('RGB').resize((320,256),Image.Resampling.BILINEAR)
            target.save(folder/'target.png');draw_tcp_overlay(target,tcp,folder/'target_tcp_overlay.png','Stage1 | exact instruction-end GT')
        current=s['goal_observed_multiview'][:,0].float().permute(1,2,0).numpy()
        Image.fromarray(np.rint((current+1)*127.5).clip(0,255).astype(np.uint8)).save(folder/'current.png')
        errors=[float(np.linalg.norm(np.asarray(x['pred_px'])-x['gt_px'])) for x in tcp['arms'] if x['gt_valid'] and x['pred_finite']]
        meta=dict(case=i,family=s['_eval']['family'],identity=identities[i],checkpoint_sha256=manifest['checkpoint_sha256'],
            samples_sha256=manifest['samples_sha256'],prompt=s['prompt'],granularity=s['granularity'],
            anchor_frame=s['trace']['anchor'],goal_frame=s['report_target']['goal_source_frame'],
            stage2_last_source=s['trace'].get('last_source'),predicted_uv=uv,tcp=tcp,
            mean_tcp_error_px=float(np.mean(errors)) if errors else None,report_target=s['report_target'],
            goal_image_provenance='exact sample _eval.report_goal from source episode at report_target.goal_source_frame',
            context_sha256=tensor_sha(contexts[s['prompt']][0]),history_trace=s['trace'].get('head_history'),
            current_source_trace=s['trace'].get('video_alignment'),seed=manifest['seed'],steps=a.steps,
            prediction_sha256=sha(folder/'prediction.png'),input_fields=list(batch),
            caveat='Same-episode diagnostic, not generalization or task-success evidence; GT is report-only, never conditioning.')
        dump(folder/'metadata.json',meta)
        manifest['cases'].append(dict(case=i,family=meta['family'],identity=identities[i],metadata=f'case_{i:03d}/metadata.json',
            valid_gt_count=tcp['valid_gt_count'],mean_tcp_error_px=meta['mean_tcp_error_px']))
        dump(out/'summary.json',manifest)
        print(json.dumps(dict(event='case_complete',case=i,family=meta['family'],elapsed_seconds=time.monotonic()-start)),flush=True)
    if sha(checkpoint)!=manifest['checkpoint_sha256']: raise ValueError('frozen checkpoint changed during inference')
    if sha(samples_path)!=manifest['samples_sha256']: raise ValueError('fixed samples changed during inference')
    manifest.update(status='complete',elapsed_seconds=time.monotonic()-start,checkpoint_unchanged=True,samples_unchanged=True)
    dump(out/'summary.json',manifest)
    print(json.dumps(dict(event='complete',cases=len(samples),output=str(out))),flush=True)


if __name__=='__main__': main()
