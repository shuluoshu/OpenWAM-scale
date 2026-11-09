#!/usr/bin/env python3
"""Alternate old/new cache paths using shared resident real Stage1 weights."""
import argparse
import ast
import hashlib
import importlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import types


def load_baseline_goal(repo):
    """Isolate relative imports and the old patch expert's absolute codec names."""
    name='_history_vae_baseline_'+hashlib.sha256(str(repo.resolve()).encode()).hexdigest()[:12]
    package=types.ModuleType(name)
    package.__path__=[str(repo/'openwam/model/goal')]
    sys.modules[name]=package
    codec=importlib.import_module(name+'.codec')
    history=importlib.import_module(name+'.history')
    patch=importlib.import_module(name+'.patch_expert')
    # This historical module used absolute imports, which a package alias alone
    # does not redirect. Never bind a candidate codec to an old LatentCache.
    patch.GoalCodec=codec.GoalCodec
    patch.GoalCodecSpec=codec.GoalCodecSpec
    visual=importlib.import_module(name+'.visual_history')
    return types.SimpleNamespace(codec=codec,history=history,visual=visual)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline-repo', type=Path, required=True)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--fixture', type=Path, required=True)
    ap.add_argument('--bundle', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--count', type=int, default=8)
    ap.add_argument('--encode-batch-size', type=int, default=4)
    ap.add_argument('--warmup', type=int, default=1)
    ap.add_argument('--repeats', type=int, default=5)
    a = ap.parse_args()
    if min(a.count, a.encode_batch_size, a.repeats) < 1 or a.warmup < 0:ap.error('invalid counts')
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    from openwam.model.goal.codec import GoalCodec
    from openwam.model.goal.history import HeadFeatureCache
    from openwam.model.goal.types import GoalTargets
    from openwam.train.goal_curated import load_config
    from openwam.train.goal_trainer import _construct, load_component_weights
    from openwam.train.goal_visual_history import load_visual_history_bundle
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device, dtype = torch.device('cuda'), torch.bfloat16

    def emit(event, **kw):print(json.dumps(dict(event=event, **kw)), flush=True)
    def gpu():return subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
    def digest(path):
        h=hashlib.sha256()
        with Path(path).open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
        return h.hexdigest()

    # Relative goal imports resolve to the pinned old checkout; unchanged native
    # Wan modules are shared. No sys.modules swapping occurs during a forward.
    old=load_baseline_goal(a.baseline_repo)
    old_codec,old_visual=old.codec,old.visual
    cfg = load_config(a.config)
    for native in ['openwam/model/video_backbone/wan/models/vae.py',
                   'openwam/model/video_backbone/wan/models/dit.py']:
        if digest(a.baseline_repo/native)!=digest(repo/native):
            raise ValueError('paired measurement requires identical native backends: '+native)
    fixture = {k:v[:a.count] for k,v in load_file(str(a.fixture)).items()}
    if len(fixture['observed'])!=a.count:raise ValueError('fixture too small')
    emit('loading_shared_vae', gpu=gpu())
    vae = _construct(cfg['components']['vae'])
    load_component_weights(vae,cfg['baseline_checkpoint'],'video_backbone.vae.')
    vae.to(device=device,dtype=dtype).eval().requires_grad_(False)
    with safe_open(str(a.bundle),framework='pt',device='cpu') as f:
        meta=json.loads(f.metadata()['visual_history'])
    emit('loading_shared_stage1')
    candidate=load_visual_history_bundle(a.bundle,vae=vae,text_dim=meta['dependencies']['text_dim'],
        precomputed_text_fingerprint=meta['dependencies']['text']['fingerprint'],device=device,dtype=dtype)
    baseline=old_visual.VisualHistoryGoalExpert(candidate.dit,old_codec.GoalCodec(vae),
        dit_config=candidate.dit_config,**candidate.history_options,
        shift=candidate.shift,lambda_image=candidate.lambda_image,lambda_tcp=candidate.lambda_tcp)
    # All trainable parameters, not just initial random seeds, are identical.
    baseline.history_pool=candidate.history_pool
    baseline.visual_adapter=candidate.visual_adapter
    baseline.history_gate=candidate.history_gate
    assert type(baseline.observation_codec) is old_codec.GoalCodec
    assert type(candidate.observation_codec) is GoalCodec
    models={'baseline':baseline,'candidate':candidate}
    codecs={'baseline':baseline.codec,'candidate':candidate.codec}
    heads={'baseline':baseline._history_cache,'candidate':candidate._history_cache}
    for model in models.values():
        model.train_stage_a();model.use_gradient_checkpointing=True
        model.codec.enable_cache(max_bytes=536870912)
    candidate.codec.max_encode_batch_size=a.encode_batch_size
    for name,p in baseline.named_parameters():
        assert p is dict(candidate.named_parameters())[name],name
    context=torch.zeros(a.count,512,meta['dependencies']['text_dim'],device=device,dtype=dtype)
    pixels=fixture['history_images'].to(dtype=dtype)
    valid=fixture['history_valid'].to(device);ids=fixture['history_episode_id'].to(device)
    last={}
    result=dict(scope='paired single-GPU real 30-layer forward/backward; no optimizer/T5/loader/DDP',
        count=a.count,valid_history_frames=int(valid.sum()),dtype=str(dtype),gpu=torch.cuda.get_device_name(),
        warmup_pairs=a.warmup,measured_pairs=a.repeats,encode_batch_size=a.encode_batch_size,
        fixture_sha256=digest(a.fixture),bundle=str(a.bundle),bundle_sha256=digest(a.bundle),
        config=str(a.config),config_sha256=digest(a.config),
        baseline_head=subprocess.check_output(['git','-C',str(a.baseline_repo),'rev-parse','HEAD'],text=True).strip(),
        candidate_head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
        candidate_diff_sha256=hashlib.sha256(subprocess.check_output(['git','-C',str(repo),'diff','10dc74e','--','openwam'])).hexdigest(),
        parameters_shared_by_identity=True,native_backends_identical=True,gpu_before=gpu())
    pool=lambda path: next(ast.dump(n) for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='HistoryTokenPool')
    result['time_pe_source_identical']=pool(a.baseline_repo/'openwam/model/goal/visual_history.py')==pool(repo/'openwam/model/goal/visual_history.py')

    def history(name):
        if name=='baseline':return heads[name].encode(codecs[name],pixels.to(device),valid,episode_ids=ids)
        return heads[name].encode(codecs[name],pixels,valid,episode_ids=ids,device=device)

    def train(name):
        model=models[name]
        model.zero_grad(set_to_none=True);torch.manual_seed(42)
        targets=GoalTargets(**{k:fixture['target_'+k].to(device) for k in
            ['image','image_valid','uv','point_valid','in_frame','visibility']})
        out=model(fixture['observed'].to(device),context,targets,
            observed_view_valid=fixture['observed_view_valid'].to(device),
            history_images=pixels.to(device) if name=='baseline' else pixels,
            history_valid=valid,history_ages=fixture['history_ages'].to(device),history_episode_id=ids)
        out['loss'].backward()
        if not torch.isfinite(out['loss']):raise FloatingPointError('nonfinite loss')
        last[name]=dict(loss=float(out['loss'].detach()),condition=out['condition_tokens'].detach())

    for phase,call,cold in [('history_cold',history,True),('history_warm',history,False),
                            ('stage1_forward_backward_cold',train,True)]:
        times={name:[] for name in models};peak={name:[] for name in models}
        if not cold:
            for name in models:call(name)
        for pair in range(a.warmup+a.repeats):
            # Reverse order every pair to distribute drift between variants.
            for name in (list(models) if pair%2==0 else list(models)[::-1]):
                if cold:heads[name].clear();codecs[name].clear_cache()
                models[name].zero_grad(set_to_none=True)
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                started=time.perf_counter();value=call(name);torch.cuda.synchronize()
                elapsed=time.perf_counter()-started
                if phase=='history_cold':last[name+'_history']=value.detach()
                if pair>=a.warmup:
                    times[name].append(elapsed);peak[name].append(torch.cuda.max_memory_allocated())
                emit('paired_measurement',phase=phase,pair=pair,variant=name,seconds=elapsed)
        result[phase]={name:dict(seconds=times[name],median_s=statistics.median(times[name]),
            samples_per_s=a.count/statistics.median(times[name]),peak_allocated_bytes=max(peak[name])) for name in models}
        result[phase]['speedup']=statistics.median(times['baseline'])/statistics.median(times['candidate'])
        result[phase]['median_paired_speedup']=statistics.median(b/c for b,c in zip(times['baseline'],times['candidate']))
        result[phase]['gpu_after']=gpu()
    for key,baseline_tensor,candidate_tensor in [
        ('history_numerics',last['baseline_history'],last['candidate_history']),
        ('condition_numerics',last['baseline']['condition'],last['candidate']['condition'])]:
        expected=baseline_tensor.float();actual=candidate_tensor.float();delta=actual-expected
        rel=float(delta.norm()/expected.norm().clamp_min(1e-12))
        result[key]=dict(max_abs=float(delta.abs().max()),mean_abs=float(delta.abs().mean()),relative_l2=rel,
            passed=rel<=.01 and bool(torch.isfinite(actual).all()),acceptance_relative_l2=.01)
    result['loss']={name:last[name]['loss'] for name in models}
    result['gpu_after']=gpu()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    emit('complete',output=str(a.output),result=result)
    assert result['history_numerics']['passed'] and result['condition_numerics']['passed']


if __name__=='__main__':main()
