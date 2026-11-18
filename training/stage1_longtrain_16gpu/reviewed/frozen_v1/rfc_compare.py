"""Isolated matched RFC Stage1 comparison using the source-native implementation.

Production never substitutes a small network. Native tiny blocks are test fixtures.
The shared snapshot, explicit draw cursor and independent noise stream are the
scientific comparison contract; branch dropout has its own RNG stream.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import copy
import hashlib
import html
import importlib
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time

import numpy as np
import torch

ARMS = ('default', 'no_history', 'no_g')
FORMAT = 'openwam.rfc_stage1_comparison.prepared.v1'
FIELDS = ('memory_valid','memory_view_valid','memory_ages','memory_frame_ids',
          'observed_view_valid','target_uv','target_image_valid','target_view_valid',
          'target_point_valid','target_in_frame','goal_horizon_s')
PREPARE_FIELDS = ('text_mask','memory_valid','observed_view_valid','memory_view_valid',
                  'memory_ages','memory_frame_ids','goal_horizon_s','target_image_valid','target_view_valid')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def runner_identity():
    root=Path(__file__).resolve().parent
    return {name:sha256(root/name) for name in ('compare.py','rfc_compare.py')}


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    atomic_text(path,json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def atomic_text(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix=path.name+'.',dir=path.parent)
    os.close(fd); temp=Path(name)
    try:
        temp.write_text(value); temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def jsonable(value):
    if torch.is_tensor(value): return value.detach().cpu().tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    if isinstance(value, dict): return {str(k):jsonable(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [jsonable(v) for v in value]
    return value


def bind_code(code_dir):
    code = Path(code_dir).resolve()
    if not (code/'openwam/model/goal/goal_memory.py').is_file():
        raise ValueError('explicit --code-dir must contain native RFC model code')
    sys.path.insert(0,str(code))
    import openwam
    if Path(openwam.__file__).resolve().parent != code/'openwam':
        raise ValueError('another openwam was imported before --code-dir')
    files = sorted((code/'openwam').rglob('*.py'))
    digest = hashlib.sha256()
    for p in files:
        digest.update(str(p.relative_to(code)).encode()); digest.update(sha256(p).encode())
    return dict(directory=str(code), python_tree_sha256=digest.hexdigest())


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_initialized(): torch.cuda.manual_seed_all(seed)


def sample_stream(records, count, *, seed):
    """One finite globally shuffled 3:7 stream; restarting uses its absolute cursor."""
    if count < 1 or count % 10:
        raise ValueError('total sample draws must be a positive multiple of 10 for exact 3:7')
    pools = {g:[i for i,r in enumerate(records) if r['granularity']==g] for g in ('task','subtask')}
    if not all(pools.values()): raise ValueError('no eligible task/subtask prepared pool')
    rng = np.random.default_rng(np.random.SeedSequence([seed,71]))
    grains = np.array(['task']*(count*3//10)+['subtask']*(count*7//10))
    rng.shuffle(grains)
    return [int(rng.choice(pools[g])) for g in grains]


def common_noise(target, *, seed, cursor, validation):
    """CPU generator avoids arm RNG consumption and device-specific sample streams."""
    derived = int(np.random.SeedSequence([seed, int(cursor), 881 if validation else 773]).generate_state(1)[0])
    generator = torch.Generator(device='cpu').manual_seed(derived)
    sigma = torch.rand(target.shape[0],generator=generator,dtype=torch.float32)
    noise = torch.randn(target.shape,generator=generator,dtype=torch.float32)
    return sigma.to(target.device), noise.to(target)


def make_state(model, batch, noisy_target, timestep, *, arm, warmup_y_text=False):
    from openwam.model.goal.goal_memory_masks import group_attention_mask
    if arm not in ARMS: raise ValueError('unknown ablation')
    state = model.prepare_state(batch['current_latents'], batch['memory_latents'], noisy_target,
        timestep, batch['text'], warmup_y_text=warmup_y_text,
        **{k:batch[k] for k in PREPARE_FIELDS if k in batch})
    if arm == 'no_g': state.key_valid[:,state.goal_slice] = False
    if arm == 'no_history': state.key_valid[:,:state.current_slice.start] = False
    state.attention_mask = group_attention_mask(state.memory_slices,state.current_slice,
                                               state.goal_slice,state.target_slice,state.key_valid)
    return state


def validity(batch):
    from openwam.model.goal.goal_memory_masks import view_token_mask
    target = batch['target_latents']; b,_,_,h,w = target.shape
    image = batch['target_image_valid'].bool()
    views = batch.get('target_view_valid',torch.ones(b,3,dtype=torch.bool,device=target.device)).bool()
    tokens = view_token_mask(views,h//2,w//2) & image[:,None]
    pixels = tokens.reshape(b,h//2,w//2).repeat_interleave(2,-2).repeat_interleave(2,-1)[:,None,None]
    points = batch['target_point_valid'].bool() & batch['target_in_frame'].bool()
    points = points & image[:,None] & views[:,0,None] & torch.isfinite(batch['target_uv']).all(-1)
    return pixels,points


def summarize_sums(sums):
    values = dict(sums)
    counts = dict(valid_latent_values=int(values.get('fm_den',0)),valid_points=int(values.get('points',0)),
                  records=int(values.get('records',0)), head_rgb_values=int(values.get('rgb_den',0)))
    metrics = dict(fm_mse=values.get('fm_sum',0)/values['fm_den'] if values.get('fm_den',0) else None,
                   uv_mse=values.get('uv_sum',0)/(2*values['points']) if values.get('points',0) else None,
                   tcp_px=values.get('tcp_sum',0)/values['points'] if values.get('points',0) else None,
                   head_image_mae=values.get('rgb_sum',0)/values['rgb_den'] if values.get('rgb_den',0) else None)
    return dict(metrics=metrics,counts=counts,sums=values)


@torch.no_grad()
def evaluate_batch(model, batch, *, arm, sigma, noise):
    pixels,points = validity(batch)
    clean = torch.where(pixels,batch['target_latents'],0.)
    noise = torch.where(pixels,noise.to(clean),0.)
    s = sigma.to(clean)[:,None,None,None,None]
    state = make_state(model,batch,(1-s)*clean+s*noise,sigma*1000,arm=arm)
    outputs = model.outputs(model.run_state(state))
    fm_error = torch.where(pixels, outputs['velocity'].float()-(noise-clean).float(),0.)
    uv_error = torch.where(points[...,None], outputs['uv']-torch.where(points[...,None],batch['target_uv'],0.),0.)
    scale = torch.tensor([319.,255.],device=uv_error.device)
    return summarize_sums(dict(fm_sum=float(fm_error.square().sum()),fm_den=int(pixels.sum())*clean.shape[1],
        uv_sum=float(uv_error.square().sum()),points=int(points.sum()),
        tcp_sum=float((uv_error*scale).norm(dim=-1).sum()),records=clean.shape[0]))


def ols(rows, x, y):
    pairs = [(float(r[x]),float(r[y])) for r in rows if r.get(x) is not None and r.get(y) is not None
             and math.isfinite(float(r[x])) and math.isfinite(float(r[y]))]
    result = dict(n=len(pairs),x=x,y=y,slope=None,x_min=None,x_max=None)
    if not pairs: return result
    a,b = np.array(pairs,dtype=float).T
    result.update(x_min=float(a.min()),x_max=float(a.max()))
    denominator = float(((a-a.mean())**2).sum())
    if len(pairs)>=2 and denominator:
        result['slope'] = float(((a-a.mean())*(b-b.mean())).sum()/denominator)
    return result


def read_jsonl(path):
    result = []
    if not Path(path).exists(): return result
    for line in Path(path).read_text().splitlines():
        try: result.append(json.loads(line))
        except json.JSONDecodeError: pass  # A writer may be appending the final line.
    return result


def append_jsonl(path, row):
    with Path(path).open('a') as f:
        f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n'); f.flush()


@contextmanager
def preserved_rng():
    cpu = torch.get_rng_state(); py = random.getstate(); npstate = np.random.get_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try: yield
    finally:
        torch.set_rng_state(cpu); random.setstate(py); np.random.set_state(npstate)
        if cuda is not None: torch.cuda.set_rng_state_all(cuda)


def run_updates(model, *, output, config, dependencies, population_identity, provider, evaluator,
                arm, seed, batch_size, eval_every, resume=None, stop_after=None, report_enabled=True,
                visualizer=None):
    from openwam.train.goal_memory import parameter_groups, _publish_training
    from openwam.train.goal_memory_checkpoint import restore_training_state, _execution_contract
    from openwam.train.goal_overfit import FP32MasterAdamW
    if int(os.environ.get('WORLD_SIZE','1')) != 1: raise ValueError('WORLD_SIZE must be 1')
    output = Path(output)
    if not resume and output.exists() and any(output.iterdir()): raise ValueError('nonempty output; use --resume')
    output.mkdir(parents=True,exist_ok=True)
    budget = config['max_updates']
    contract = dict(config=config,dependencies=dependencies,population=population_identity,dit_config=model.dit_config,
        comparison=dict(arm=arm,seed=seed,batch_size=batch_size,eval_every=eval_every,
                        noise='CPU-fp32-SeedSequence-773/881-v1',draws='finite-global-3:7-v1'))
    groups = parameter_groups(model,learning_rate=config['learning_rate'],new_learning_rate=config['new_learning_rate'])
    for group in groups: group['betas'] = tuple(config.get('adam_betas',(.9,.999)))
    optimizer = FP32MasterAdamW(groups,weight_decay=config.get('weight_decay',.01),master_device='cpu')
    update = restore_training_state(resume,model,optimizer,contract=contract) if resume else 0
    if update > budget: raise ValueError('resume exceeds update budget')
    end = min(budget,update+stop_after) if stop_after is not None else budget
    if stop_after is not None and stop_after <= 0: raise ValueError('stop_after must be positive')
    atomic_json(output/'contract.json',contract)
    old_rows = read_jsonl(output/'train.jsonl')
    # Refuse logs from a cursor beyond the checkpoint; never silently double-append.
    if old_rows and old_rows[-1]['update'] > update: raise ValueError('training log cursor exceeds resume; recover into a fresh output directory')
    wall_offset = old_rows[-1].get('wall_seconds',0.) if old_rows else 0.
    clock = time.monotonic()
    visual_updates = {0,100,500,budget}
    def assess(step):
        if evaluator is None and visualizer is None: return
        with preserved_rng(), torch.no_grad():
            model.eval()
            if evaluator:
                phase_started=time.monotonic()
                result = evaluator(model,step)
                append_jsonl(output/'eval.jsonl',dict(update=step,samples=step*batch_size,
                    wall_seconds=wall_offset+time.monotonic()-clock,arm=arm,**result))
                append_jsonl(output/'phases.jsonl',dict(update=step,phase='evaluation',seconds=time.monotonic()-phase_started))
            if visualizer and step in visual_updates:
                phase_started=time.monotonic(); visualizer(model,step)
                append_jsonl(output/'phases.jsonl',dict(update=step,phase='visualization',seconds=time.monotonic()-phase_started))
        model.train()
        if report_enabled: report(output.parent,output.parent/'report')
    if not resume: assess(0)
    model.train()
    while update < end:
        started = time.monotonic()
        batch,metadata = provider(update)
        # Branch RNG is reproducible from the absolute update; common noise is explicit.
        seed_all(seed+100003*(update+1))
        sigma,noise = common_noise(batch['target_latents'],seed=seed,cursor=update,validation=False)
        optimizer.zero_grad()
        result = model.training_loss(**batch,update=update,total_updates=budget,sigma=sigma,noise=noise)
        if not torch.isfinite(result['loss']): raise FloatingPointError('nonfinite training objective')
        result['loss'].backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),config.get('max_grad_norm',1.),
                                             error_if_nonfinite=True,foreach=False)
        optimizer.step()
        if next(model.parameters()).device.type=='cuda': torch.cuda.synchronize()
        update += 1
        pixels,points = validity(batch)
        row = dict(update=update,samples=update*batch_size,wall_seconds=wall_offset+time.monotonic()-clock,
                   seconds=time.monotonic()-started,arm=arm,gradient_norm=float(norm),
                   valid_points=int(points.sum()),valid_latent_values=int(pixels.sum())*batch['target_latents'].shape[1],
                   grain_counts={g:sum(r['granularity']==g for r in metadata) for g in ('task','subtask')},
                   sample_ids=[r.get('id') for r in metadata],
                   common_noise_sha256=hashlib.sha256(noise.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
                   **{k:float(result[k].detach()) for k in ('loss','fm_loss','uv_loss')})
        append_jsonl(output/'train.jsonl',row); print(json.dumps(row,allow_nan=False),flush=True)
        del result,batch,noise,sigma
        if update%eval_every==0 or update==end: assess(update)
        if update%config['checkpoint_every']==0 or update==end:
            phase_started=time.monotonic()
            _publish_training(model,optimizer,output,update=update,contract=contract,config=config,
                              dependencies=dependencies,dit_config=model.dit_config)
            append_jsonl(output/'phases.jsonl',dict(update=update,phase='checkpoint',seconds=time.monotonic()-phase_started))
            atomic_json(output/'checkpoint.json',dict(update=update,samples=update*batch_size,
                        weights=str(output/'goal_memory.safetensors'),resume=str(output/'resume.pt')))
            if report_enabled: report(output.parent,output.parent/'report')
    if update==0 or (resume and update==budget):
        _publish_training(model,optimizer,output,update=update,contract=contract,config=config,
                          dependencies=dependencies,dit_config=model.dit_config)
    summary = dict(update=update,budget=budget,status='complete' if update==budget else 'interrupted',
                   resume=str(output/'resume.pt'),execution=_execution_contract(model))
    atomic_json(output/'summary.json',summary)
    return summary


def validate_record(sample):
    record=sample['record']; grain=sample['granularity']
    if grain not in ('task','subtask') or record.get('granularity')!=grain or record.get('label_kind')!=grain+'_end':
        raise ValueError('invalid matching-grain terminal label')
    if record['goal_source_frame']!=record['instruction_end_exclusive']-1:
        raise ValueError('target is not the matching-grain terminal')
    valid=sample['memory_valid'].bool()
    if (sample['memory_frame_ids'][valid]>=record['anchor_source_frame']).any():
        raise ValueError('memory violates strict past source frames')


def stratified_validation_selections(dataset,count,*,seed):
    """Balanced family visits, alternating only the grains really in each pool.

    Within a family/grain preserve native RFC repo/run weights and draw a
    uniform eligible anchor. This deliberately changes validation allocation,
    never the train population, labels, endpoints or history policy.
    """
    groups=defaultdict(dict)
    for grain,pool in dataset.pools.items():
        for index,item in enumerate(pool):
            groups[item['repo']['family']].setdefault(grain,[]).append(index)
    families=sorted(groups)
    if not families or count<len(families):
        raise ValueError('validation count must cover every eligible family')
    rng=np.random.default_rng(np.random.SeedSequence([seed,909]))
    rng.shuffle(families); visits=defaultdict(int); result=[]
    for i in range(count):
        family=families[i%len(families)]
        grains=sorted(groups[family]); grain=grains[visits[family]%len(grains)]
        visits[family]+=1
        indices=groups[family][grain]
        weights=np.asarray([dataset.pool_weights[grain][j] for j in indices],dtype=float)
        weights=weights/weights.sum()
        selected=int(rng.choice(indices,p=weights)); item=dict(dataset.pools[grain][selected])
        item['anchor']=int(rng.integers(item['start'],item['stop']))
        item['goal']=item['segment']['end_exclusive']-1
        result.append(item)
    rng.shuffle(result)
    return result


def physical_episode_ids(dataset):
    return {(str(Path(repo['path']).resolve()),int(episode)) for repo in dataset.reader.repos
            for episode,_,_ in repo['ranges']}


def grain_weighted_metrics(groups):
    weights={'task':.3,'subtask':.7}; metrics={}
    for metric in ('fm_mse','uv_mse','tcp_px','head_image_mae'):
        values={g:groups.get('grain:'+g,{}).get('metrics',{}).get(metric) for g in weights}
        metrics[metric]=sum(weights[g]*v for g,v in values.items()) if all(v is not None for v in values.values()) else None
    return dict(weights=weights,metrics=metrics,
                grain_counts={g:groups.get('grain:'+g,{}).get('counts',{}) for g in weights},
                policy='weighted grain means; null unless each real grain has a valid metric denominator')


def load_manifest(data, *, verify=True):
    data=Path(data); manifest=json.loads((data/'manifest.json').read_text())
    if manifest.get('format')!=FORMAT: raise ValueError('unsupported prepared data format')
    if verify:
        for name,digest in manifest['files'].items():
            path=data/name
            if not path.is_file() or sha256(path)!=digest: raise ValueError('prepared SHA256 mismatch: '+str(path))
    return manifest


def collate_prepared(rows, data, *, device, dtype):
    data=Path(data)
    fields=('current_latents','target_latents','memory_latents')+FIELDS
    batch={k:torch.stack([r[k] for r in rows]).to(device=device,
               dtype=dtype if k.endswith('latents') else rows[0][k].dtype) for k in fields if k in rows[0]}
    contexts=[torch.load(data/r['context_file'],map_location='cpu',weights_only=True) for r in rows]
    maxlen=max(1,max(len(t) for t in contexts)); width=contexts[0].shape[-1]
    text=torch.zeros(len(rows),maxlen,width,dtype=dtype,device=device)
    text_mask=torch.zeros(len(rows),maxlen,dtype=torch.bool,device=device)
    for i,context in enumerate(contexts):
        text[i,:len(context)]=context.to(device=device,dtype=dtype); text_mask[i,:len(context)]=True
    batch.update(text=text,text_mask=text_mask)
    return batch,[{**r['record'],'granularity':r['granularity']} for r in rows]


@torch.no_grad()
def encode_valid_images(codec,images,views):
    """Skip empty/padded canvases; native preprocessing requires a real view.

    Invalid slots retain zero latents and their original false masks. Valid
    canvases still pass through the native sanitizer and independent T=1 VAE.
    """
    if views.dtype!=torch.bool or views.shape!=(len(images),3):
        raise ValueError('prepare view masks must be boolean [N,3]')
    eligible=views.any(-1)
    latents=torch.zeros((len(images),*codec.latent_shape),device=images.device,
                        dtype=codec.dtype or images.dtype)
    if eligible.any():
        encoded=codec.encode_images(images[eligible],view_valid=views[eligible])
        latents[eligible]=encoded.detach().to(latents)
    return latents


@torch.no_grad()
def generate_latents(model, batch, *, arm, seed, index, steps=20):
    if steps<1: raise ValueError('positive generation steps required')
    _,noise=common_noise(batch['target_latents'],seed=seed,cursor=index+1000000,validation=True)
    pixels,_=validity(batch); latent=torch.where(pixels,noise,0.)
    outputs=None
    for n in range(steps):
        sigma=1.-n/steps
        timestep=torch.full((latent.shape[0],),sigma*1000,device=latent.device)
        state=make_state(model,batch,latent,timestep,arm=arm,warmup_y_text=False)
        outputs=model.outputs(model.run_state(state))
        latent=torch.where(pixels,latent-outputs['velocity']/steps,0.)
    return dict(latents=latent,uv=outputs['uv'],steps=steps,noise_seed_index=index+1000000)


def report(runs_root, output):
    root=Path(runs_root); output=Path(output); output.mkdir(parents=True,exist_ok=True)
    arms={}; galleries=[]
    for arm in ARMS:
        run=root/arm
        if not run.is_dir(): continue
        train=read_jsonl(run/'train.jsonl'); evaluations=read_jsonl(run/'eval.jsonl')
        slopes={}
        eval_slopes={}
        for window in (None,100,500):
            subset=train if window is None else train[-window:]
            slopes['all' if window is None else f'last_{window}']={
                metric:{axis:ols(subset,axis,metric) for axis in ('update','samples','wall_seconds')}
                for metric in ('loss','fm_loss','uv_loss')}
            evaluation_rows=[{**r,**r.get('metrics',{})} for r in evaluations]
            evaluation_rows=evaluation_rows if window is None else evaluation_rows[-window:]
            eval_slopes['all' if window is None else f'last_{window}']={
                metric:{axis:ols(evaluation_rows,axis,metric) for axis in ('update','samples','wall_seconds')}
                for metric in ('fm_mse','uv_mse','tcp_px')}
        visual_rows=read_jsonl(run/'visual.jsonl')
        arms[arm]=dict(train_rows=len(train),latest_update=train[-1]['update'] if train else 0,
                       latest_train_update=train[-1]['update'] if train else 0,
                       latest_eval_update=evaluations[-1]['update'] if evaluations else None,
                       slopes=slopes,eval_slopes=eval_slopes,latest_eval=evaluations[-1] if evaluations else None,
                       latest_visual=visual_rows[-1] if visual_rows else None,
                       phase_timings=read_jsonl(run/'phases.jsonl'),
                       train=train,evaluations=evaluations)
        for path in sorted((run/'visuals').glob('step_*/index.html')):
            galleries.append((arm,os.path.relpath(path,output)))
    completed=[{r['update'] for r in value['evaluations']} for value in arms.values()]
    common=set.intersection(*completed) if completed else set()
    common_update=max(common) if common else None
    matched=dict(update=common_update,participants=list(arms),expected_arms=list(ARMS),
                 all_three_present=set(arms)==set(ARMS),evaluations={})
    if common_update is not None:
        matched['evaluations']={arm:next(r for r in reversed(value['evaluations']) if r['update']==common_update)
                                for arm,value in arms.items()}
    payload=dict(format='openwam.rfc_stage1_comparison.report.v1',generated_unix=time.time(),arms=arms,
                 matched_validation=matched,
                 wall_time_scope='Training loop elapsed time including evaluation/visualization/checkpoints, excluding preparation/model bootstrap. Fresh-output resume resets its elapsed-time origin; not complete GPU cost.',
                 scope='Matched source-native RFC objective pilot; validation episode-disjoint from this pilot only; foundation exposure unknown. TCP counts are required; missing points yield null metrics.',
                 slopes='Ordinary least squares over each recorded window, with count and x range; early slopes are descriptive, not success estimates.')
    atomic_json(output/'metrics.json',payload)
    def table_row(arm,train_update,e):
        e=e or {}; m=e.get('metrics',{}); counts=e.get('counts',{})
        def number(x): return 'n/a' if x is None else f'{x:.6g}'
        weighted=e.get('grain_3_7',{}).get('metrics',{})
        return '<tr>'+''.join(f'<td>{html.escape(str(x))}</td>' for x in
            (arm,train_update,e.get('update','n/a'),number(m.get('fm_mse')),number(weighted.get('fm_mse')),
             number(m.get('uv_mse')),number(m.get('tcp_px')),counts.get('valid_points',0)))+'</tr>'
    rows=[table_row(arm,value['latest_train_update'],value['latest_eval']) for arm,value in arms.items()]
    matched_rows=[table_row(arm,arms[arm]['latest_train_update'],e) for arm,e in matched['evaluations'].items()]
    groups=''.join('<h3>'+arm+' — validation update '+str(common_update)+'</h3><pre>'+html.escape(json.dumps(
        dict(objective=e.get('groups',{}),grain_3_7=e.get('grain_3_7',{})),indent=2))+'</pre>'
        for arm,e in matched['evaluations'].items())
    visual_groups=''.join('<h3>'+arm+' — image diagnostic update '+str(v['latest_visual']['update'])+'</h3><pre>'+html.escape(
        json.dumps(v['latest_visual'].get('groups',{}),indent=2))+'</pre>' for arm,v in arms.items() if v['latest_visual'])
    links=''.join(f'<li><a href="{html.escape(url)}">{html.escape(arm+" "+url)}</a></li>' for arm,url in galleries)
    # Embedded data and no CDN make the report usable on DSW or as an offline file.
    chart=json.dumps({a:v['train'] for a,v in arms.items()},allow_nan=False).replace('</','<\\/')
    page='''<!doctype html><meta charset="utf-8"><title>RFC Stage1 matched pilot</title>
<style>body{font:15px system-ui;max-width:1200px;margin:30px auto;background:#fafafa;color:#17212b}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccc}canvas{width:100%;height:400px;background:white;border:1px solid #ddd}pre{white-space:pre-wrap}select{margin:10px}</style>
<h1>RFC Stage1 matched three-arm pilot</h1><p>Same prepared input bytes, finite 3:7 draws, foundation initialization and explicit noise. Evaluation keeps ablation masks active and keyframe text off. Validation is episode-disjoint from this pilot; foundation exposure is unknown. Missing TCP points are n/a, never zero-error successes. Slopes describe the recorded windows and require their counts.</p>
<p>Wall seconds measure training-loop elapsed time including evaluation, visualization and checkpoints, excluding preparation/model bootstrap. Fresh-output resume resets the elapsed-time origin. This is not complete GPU cost.</p>
<p><a href="metrics.json">Machine metrics and all slopes</a></p>
<h2>Matched completed validation across available arms</h2><p>Latest common completed validation update: COMMON_UPDATE. Participants: PARTICIPANTS. All three arms present: ALL_THREE.</p>
<p>Pooled FM uses valid latent values in the balanced-family validation allocation. 3:7 grain-weighted FM weights real task/subtask means .3/.7 and is n/a unless both denominators exist. Train and evaluation cursors are separate.</p>
<table><tr><th>Arm</th><th>Train update</th><th>Eval update</th><th>Pooled balanced-family FM</th><th>3:7 grain-weighted FM</th><th>UV MSE</th><th>TCP px</th><th>Valid points</th></tr>MATCHED_ROWS</table>
<h2>Latest per-arm measurements (evaluation updates may differ)</h2>
<table><tr><th>Arm</th><th>Train update</th><th>Eval update</th><th>Pooled balanced-family FM</th><th>3:7 grain-weighted FM</th><th>UV MSE</th><th>TCP px</th><th>Valid points</th></tr>LATEST_ROWS</table>
<label>Horizontal axis<select id="axis"><option>update</option><option>samples</option><option>wall_seconds</option></select></label><label>Metric<select id="metric"><option>loss</option><option>fm_loss</option><option>uv_loss</option></select></label><canvas id="chart" width="1100" height="400"></canvas>
<p>Blue: default; orange: no_history; green: no_g. Plot includes every available update. OLS windows and counts are in the JSON.</p><h2>Fixed 20-step image samples</h2><ul>LINKS</ul><h2>Matched completed validation by grain and family</h2>GROUPS<h2>Latest per-arm image diagnostics (updates may differ)</h2>VISUAL_GROUPS
<script>const data=DATA;function draw(){let c=document.getElementById('chart'),g=c.getContext('2d'),x=document.getElementById('axis').value,y=document.getElementById('metric').value,p=Object.values(data).flat().filter(r=>Number.isFinite(r[x])&&Number.isFinite(r[y]));g.clearRect(0,0,c.width,c.height);if(!p.length)return;let xa=Math.min(...p.map(r=>r[x])),xb=Math.max(...p.map(r=>r[x])),ya=Math.min(...p.map(r=>r[y])),yb=Math.max(...p.map(r=>r[y]));g.strokeStyle='#ccc';g.strokeRect(60,20,1000,330);g.fillStyle='#333';g.fillText(xa,60,375);g.fillText(xb,1000,375);g.fillText(ya.toPrecision(4),2,350);g.fillText(yb.toPrecision(4),2,25);Object.entries(data).forEach(([a,rs],i)=>{g.strokeStyle=['#1677bb','#e67e22','#249647'][i];g.beginPath();rs.forEach((r,j)=>{let u=60+(r[x]-xa)/(xb-xa||1)*1000,v=350-(r[y]-ya)/(yb-ya||1)*330;j?g.lineTo(u,v):g.moveTo(u,v)});g.stroke()})}document.getElementById('axis').onchange=draw;document.getElementById('metric').onchange=draw;draw();</script>'''
    page=page.replace('MATCHED_ROWS',''.join(matched_rows)).replace('LATEST_ROWS',''.join(rows))
    page=page.replace('COMMON_UPDATE',str(common_update) if common_update is not None else 'none yet')
    page=page.replace('PARTICIPANTS',html.escape(', '.join(arms))).replace('ALL_THREE',str(matched['all_three_present']))
    page=page.replace('VISUAL_GROUPS',visual_groups).replace('LINKS',links).replace('GROUPS',groups).replace('DATA',chart)
    atomic_text(output/'index.html',page)
    return payload


def construct_component(config, name, *, device, dtype):
    from openwam.train.goal_trainer import _construct,load_component_weights
    module=_construct(config['components'][name])
    load_component_weights(module,config['baseline_checkpoint'],f'video_backbone.{name}.')
    return module.to(device=device,dtype=dtype)


def prepare(args):
    code_identity=bind_code(args.code_dir)
    from openwam.train.goal_memory import load_config
    from openwam.train.goal_memory_data import make_dataset
    from openwam.model.goal.dependencies import dependency_identity,resolve_tokenizer_config
    from openwam.model.goal.memory_codec import GoalMemoryCodec
    from openwam.model.video_backbone.wan.encode import encode_text
    config=load_config(args.base_config); output=Path(args.output)
    if output.exists() and any(output.iterdir()): raise ValueError('prepare requires empty output')
    if args.train_count<10 or args.val_count<10: raise ValueError('at least 10 train/validation records required')
    output.mkdir(parents=True,exist_ok=True)
    dtype=getattr(torch,args.dtype); seed_all(args.seed)
    datasets={}; split_config={}
    # Freeze and validate source populations before loading any GPU module.
    for split,source,count in (('train',args.train_source,args.train_count),('val',args.val_source,args.val_count)):
        cfg=copy.deepcopy(config)
        cfg['dataset'].update(source_yaml=str(Path(source).resolve()),cache_dir=str(output/'reader_cache'/split),
                              epoch_size=count,subtask_probability=.7,require_cache=False)
        cfg['seed']=args.seed; datasets[split]=make_dataset(cfg,codec=None); split_config[split]=cfg
    # Resolve physical identities so different YAML aliases cannot hide overlap.
    overlap=physical_episode_ids(datasets['train'])&physical_episode_ids(datasets['val'])
    if overlap: raise ValueError('train/validation episode overlap: '+str(sorted(overlap)[:10]))
    selections=stratified_validation_selections(datasets['val'],args.val_count,seed=args.seed)
    datasets['val'].selection_at=lambda index:dict(selections[index])
    validation_selection=[dict(family=s['repo']['family'],repo=s['repo']['repo'],episode=s['episode'],
        grain=s['level'],anchor=s['anchor'],goal=s['goal'],source=s['source']) for s in selections]
    atomic_json(output/'validation_selection.json',dict(policy='balanced-family-alternating-real-grain-native-weighted-run-uniform-anchor-v1',
        seed=args.seed,selections=validation_selection))
    config['dataset'].update(source_yaml=str(Path(args.train_source).resolve()),cache_dir=str(output/'reader_cache/train'))
    print(json.dumps(dict(event='metadata_frozen',scopes={s:jsonable(d.scope_summary()) for s,d in datasets.items()})),flush=True)
    vae=construct_component(config,'vae',device=args.device,dtype=dtype).eval().requires_grad_(False)
    text_encoder=construct_component(config,'text_encoder',device=args.device,dtype=dtype).eval().requires_grad_(False)
    token_cfg=resolve_tokenizer_config(config); mod,name=token_cfg['model_class'].rsplit('.',1)
    tokenizer=getattr(importlib.import_module(mod),name)(**token_cfg['kwargs'])
    width=config['components']['dit']['extra_kwargs']['text_dim']
    dependencies=dependency_identity(vae,text_dim=width,text_encoder=text_encoder,tokenizer=tokenizer)
    codec=GoalMemoryCodec(vae,output/'frame_cache',vae_identity=dependencies['vae']['fingerprint'],
                         device=args.device,dtype=dtype,max_encode_batch_size=args.vae_batch_size)
    codec.codec.enable_cache(max_bytes=args.latent_cache_mb*1024*1024)
    dependencies['memory_contract']=dict(fps=1,temporal_vae=False,views='head+left_wrist+right_wrist',
        labels='matching-granularity terminal',preprocess=codec.preprocess_identity)
    contexts={}; records={'train':[],'val':[]}; file_hashes={'validation_selection.json':sha256(output/'validation_selection.json')}
    for split,dataset in datasets.items():
        (output/split).mkdir(exist_ok=True); (output/'contexts').mkdir(exist_ok=True)
        dataset.reader._evaluating=(split=='val')
        for index in range(len(dataset)):
            sample=dataset.sample(index); validate_record(sample)
            prompt=sample['instruction']
            if prompt not in contexts:
                key=hashlib.sha256(prompt.encode()).hexdigest(); relative=f'contexts/{key}.pt'
                with torch.no_grad():
                    context,length=encode_text([prompt],tokenizer=tokenizer,text_encoder=text_encoder,device=args.device)
                actual=int(length[0]); torch.save(context[0,:actual].detach().cpu().contiguous(),output/relative)
                contexts[prompt]=relative; file_hashes[relative]=sha256(output/relative)
                del context,length
            images=torch.cat((sample['observed'][None],sample['target_image'][None],sample['memory_images']),0)
            views=torch.cat((sample['observed_view_valid'][None],sample['target_view_valid'][None],sample['memory_view_valid']),0)
            latents=encode_valid_images(codec,images,views).detach().cpu()
            row={key:sample[key].detach().cpu() for key in FIELDS}
            row.update(current_latents=latents[0],target_latents=latents[1],memory_latents=latents[2:],
                       context_file=contexts[prompt],instruction=prompt,granularity=sample['granularity'],
                       record=jsonable(sample['record']),memory_source_identity=sample['memory_source_identity'],
                       population_identity=sample['population_identity'])
            if split=='val':
                row['raw_target_rgb']=((sample['target_image'].float().clamp(-1,1)+1)*127.5).round().byte()
                row['raw_current_rgb']=((sample['observed'].float().clamp(-1,1)+1)*127.5).round().byte()
            relative=f'{split}/{index:06d}.pt'; torch.save(row,output/relative)
            file_hashes[relative]=sha256(output/relative)
            records[split].append(dict(file=relative,granularity=sample['granularity'],family=sample['record']['family'],
                                       id=sample['record']['id'],source_episode=sample['record']['source_episode'],
                                       source_repo=sample['record']['source_repo']))
            print(json.dumps(dict(event='prepared_record',split=split,index=index,id=sample['record']['id'],
                                  grain=sample['granularity'])),flush=True)
            del images,views,latents,sample,row
    atomic_json(output/'config.json',config); file_hashes['config.json']=sha256(output/'config.json')
    for split,dataset in datasets.items():
        atomic_json(output/f'{split}_sources.json',jsonable(dict(scope=dataset.scope_summary(),sources=list(dataset.sources.values()))))
        file_hashes[f'{split}_sources.json']=sha256(output/f'{split}_sources.json')
    # The actual frozen module identities above are authoritative for representation bytes.
    manifest=dict(format=FORMAT,code=code_identity,runner=runner_identity(),seed=args.seed,dtype=args.dtype,dependencies=dependencies,
        config=config,records=records,files=file_hashes,contexts=dict(instructions=len(contexts),trimmed=True),
        sources={s:dict(path=str(Path(p).resolve()),sha256=sha256(p)) for s,p in
                 (('train',args.train_source),('val',args.val_source))},
        baseline=dict(path=config['baseline_checkpoint'],sha256=sha256(config['baseline_checkpoint'])),
        preparation=dict(vae_batch_size=args.vae_batch_size,device=args.device,dtype=args.dtype,
                         independent_t1=True,temporal_compression=False,train_val_episode_disjoint=True,
                         validation_selection='balanced-family-alternating-real-grain-native-weighted-run-uniform-anchor-v1'))
    atomic_json(output/'manifest.json',manifest)
    return dict(output=str(output),manifest_sha256=sha256(output/'manifest.json'),
                counts={k:len(v) for k,v in records.items()})


def make_evaluator(data, manifest, *, device, dtype, seed, arm):
    entries=manifest['records']['val']
    def evaluate(model,update):
        sums=defaultdict(float); group_sums=defaultdict(lambda:defaultdict(float)); per_record=[]
        for index,entry in enumerate(entries):
            row=torch.load(Path(data)/entry['file'],map_location='cpu',weights_only=False)
            batch,meta=collate_prepared([row],data,device=device,dtype=dtype)
            sigma,noise=common_noise(batch['target_latents'],seed=seed,cursor=index,validation=True)
            result=evaluate_batch(model,batch,arm=arm,sigma=sigma,noise=noise)
            for key,value in result['sums'].items(): sums[key]+=value
            for name in ('grain:'+entry['granularity'],'family:'+entry['family'],
                         'family_grain:'+entry['family']+'/'+entry['granularity']):
                for key,value in result['sums'].items(): group_sums[name][key]+=value
            per_record.append(dict(id=entry['id'],grain=entry['granularity'],family=entry['family'],
                                   sigma=float(sigma[0]),**result))
            del row,batch,noise,result
        groups={k:summarize_sums(v) for k,v in group_sums.items()}
        return {**summarize_sums(sums), 'groups':groups,'grain_3_7':grain_weighted_metrics(groups),
                'records':per_record,'keyframe_text':False,'ablation_mask_forced':True}
    return evaluate


def make_visualizer(data, manifest, config, *, output, device, dtype, seed, arm, count, steps):
    entries=manifest['records']['val']
    # One fixed balanced subset, including both grains when available.
    selected=[]
    for grain in ('subtask','task'):
        for i,e in enumerate(entries):
            if e['granularity']==grain and i not in selected:
                selected.append(i);break
    selected+= [i for i in range(len(entries)) if i not in selected]
    selected=selected[:count]
    def visualize(model,update):
        from PIL import Image,ImageDraw
        from openwam.model.goal.codec import GoalCodec,GoalCodecSpec
        from openwam.model.goal.dependencies import module_identity
        started=time.monotonic()
        vae=construct_component(config,'vae',device=device,dtype=dtype).eval().requires_grad_(False)
        if module_identity(vae)['fingerprint']!=manifest['dependencies']['vae']['fingerprint']:
            raise ValueError('visual decoder VAE identity differs from prepared representation')
        codec=GoalCodec(vae,GoalCodecSpec(name='wan22_multiview_t1',image_shape=(3,1,384,320),
                                         latent_shape=(48,1,24,20)))
        directory=Path(output)/'visuals'/f'step_{update:06d}'; directory.mkdir(parents=True,exist_ok=True)
        records=[]; items=[]
        for index in selected:
            entry=entries[index]; row=torch.load(Path(data)/entry['file'],map_location='cpu',weights_only=False)
            batch,_=collate_prepared([row],data,device=device,dtype=dtype)
            generated=generate_latents(model,batch,arm=arm,seed=seed,index=index,steps=steps)
            decoded=codec.decode(generated['latents'])[0].detach().float().cpu()
            prediction=(decoded.clamp(-1,1)+1)/2
            target=row['raw_target_rgb'].float()/255
            current=row['raw_current_rgb']
            _,points=validity(batch); pointmask=points[0].cpu()
            uv=generated['uv'][0].cpu(); target_uv=row['target_uv'].float()
            tcp=float(((uv[pointmask]-target_uv[pointmask])*torch.tensor([319.,255.])).norm(dim=-1).sum()) if pointmask.any() else 0.
            head_valid=bool(row['target_image_valid'] and row['target_view_valid'][0])
            rgb_sum=float((prediction[:,:,:256]-target[:,:,:256]).abs().sum()) if head_valid else 0.
            rgb_den=3*256*320 if head_valid else 0
            stats=summarize_sums(dict(rgb_sum=rgb_sum,rgb_den=rgb_den,tcp_sum=tcp,points=int(pointmask.sum()),records=1))
            def pil(tensor):
                return Image.fromarray(tensor[:,0].permute(1,2,0).contiguous().numpy())
            cur=pil(current); gt=pil(row['raw_target_rgb']); pred=pil((prediction*255).round().byte())
            for image,coords in ((gt,target_uv),(pred,uv)):
                draw=ImageDraw.Draw(image)
                for p in range(2):
                    if pointmask[p]:
                        x,y=(coords[p]*torch.tensor([319.,255.])).tolist()
                        draw.ellipse((x-4,y-4,x+4,y+4),outline=('cyan' if p==0 else 'magenta'),width=2)
            canvas=Image.new('RGB',(960,424),'white')
            canvas.paste(cur,(0,40));canvas.paste(gt,(320,40));canvas.paste(pred,(640,40))
            d=ImageDraw.Draw(canvas);d.text((5,5),f'Current | matching-grain terminal GT | generated ({steps}-step Y; clean G)',fill='black')
            d.text((5,21),f'{arm} update {update} {entry["granularity"]} anchor {row["record"]["anchor_source_frame"]} goal {row["record"]["goal_source_frame"]}',fill='black')
            filename=f'{index:04d}.png';canvas.save(directory/filename)
            rec=dict(index=index,id=entry['id'],family=entry['family'],grain=entry['granularity'],
                goal_horizon_s=float(row['goal_horizon_s']),record=row['record'],
                keyframe_text=False,ablation_mask_forced=True,steps=steps,**stats)
            records.append(rec)
            items.append(f'<article><h3>{html.escape(entry["id"])}</h3><p>{html.escape(row["instruction"])}</p><img width="960" src="{filename}"><pre>{html.escape(json.dumps({k:v for k,v in rec.items() if k!="record"},indent=2))}</pre></article>')
            del batch,generated,decoded,prediction,row
        total=defaultdict(float); group_sums=defaultdict(lambda:defaultdict(float))
        for row in records:
            for k,v in row['sums'].items():total[k]+=v
            for name in ('grain:'+row['grain'],'family:'+row['family'],'family_grain:'+row['family']+'/'+row['grain']):
                for k,v in row['sums'].items():group_sums[name][k]+=v
        result=dict(update=update,arm=arm,records=records,groups={k:summarize_sums(v) for k,v in group_sums.items()},
                    **summarize_sums(total),seconds=time.monotonic()-started)
        atomic_json(directory/'metrics.json',result)
        (directory/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>RFC samples</title><style>body{font:15px system-ui;max-width:1000px;margin:auto}img{max-width:100%}pre{white-space:pre-wrap}</style><h1>Fixed source images and current-weight generation</h1><p>Head image MAE is [0,1] RGB versus original terminal RGB. TCP is counted only for valid in-frame GT points; no valid point means n/a. All generation uses the current arm weights, keyframe text off, and explicit ablation masks.</p>'+''.join(items))
        append_jsonl(Path(output)/'visual.jsonl',result)
        del vae,codec
        if torch.cuda.is_initialized(): torch.cuda.empty_cache()
    return visualize


def train(args):
    identity=bind_code(args.code_dir)
    from openwam.model.goal.goal_memory import GoalMemoryExpert,GoalMemoryConfig
    data=Path(args.data);manifest=load_manifest(data)
    if runner_identity()!=manifest['runner']:
        raise ValueError('comparison runner scripts differ from prepared snapshot')
    if identity['python_tree_sha256']!=manifest['code']['python_tree_sha256']:
        raise ValueError('model source code differs from prepared snapshot')
    dtype=getattr(torch,args.dtype)
    if args.dtype!=manifest['dtype']: raise ValueError('train dtype differs from frozen representation')
    if args.seed!=manifest['seed']: raise ValueError('seed differs from prepared comparison contract')
    config=copy.deepcopy(manifest['config'])
    if sha256(config['baseline_checkpoint'])!=manifest['baseline']['sha256']:
        raise ValueError('foundation checkpoint SHA256 differs from preparation')
    config.update(stage=1,max_updates=args.updates,checkpoint_every=args.checkpoint_every,
                  optimizer_device='cpu',dtype=args.dtype,seed=args.seed)
    if config['learning_rate']!=1e-5 or config['new_learning_rate']!=1e-4:
        raise ValueError('matched pilot requires native DiT/new-module learning rates 1e-5/1e-4')
    memory=config.setdefault('memory',{})
    if args.arm=='no_history': memory['history_dropout']=1.
    if args.arm=='no_g': memory['goal_dropout']=1.
    seed_all(args.seed)
    dit=construct_component(config,'dit',device='cpu',dtype=dtype)
    model=GoalMemoryExpert(dit,GoalMemoryConfig(**memory),
                           dit_config=config['components']['dit']['extra_kwargs']).to(device=args.device,dtype=dtype)
    entries=manifest['records']['train']
    stream=sample_stream(entries,args.updates*args.batch_size,seed=args.seed)
    def provider(update):
        indices=stream[update*args.batch_size:(update+1)*args.batch_size]
        rows=[torch.load(data/entries[i]['file'],map_location='cpu',weights_only=False) for i in indices]
        return collate_prepared(rows,data,device=args.device,dtype=dtype)
    evaluator=make_evaluator(data,manifest,device=args.device,dtype=dtype,seed=args.seed,arm=args.arm)
    visualizer=make_visualizer(data,manifest,config,output=args.output,device=args.device,dtype=dtype,
        seed=args.seed,arm=args.arm,count=args.visual_count,steps=args.sampling_steps) if args.visual_count else None
    population=dict(manifest_sha256=sha256(data/'manifest.json'),
                    stream_sha256=hashlib.sha256(np.asarray(stream,dtype='<i8').tobytes()).hexdigest(),
                    prepared_files=manifest['files'],source_code=identity['python_tree_sha256'])
    return run_updates(model,output=args.output,config=config,dependencies=manifest['dependencies'],
        population_identity=population,provider=provider,evaluator=evaluator,visualizer=visualizer,
        arm=args.arm,seed=args.seed,batch_size=args.batch_size,eval_every=args.eval_every,
        resume=args.resume,stop_after=args.stop_after)
