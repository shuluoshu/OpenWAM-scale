"""Continue the verified masked-SA Stage1 on all smoke families for a time budget."""
import argparse
from collections import defaultdict, OrderedDict
import gc
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import torch
import yaml

CODE=Path(__file__).resolve().parents[2];sys.path.insert(0,str(CODE))
from openwam.train.goal_visual_history import load_visual_history_bundle, save_visual_history_bundle
from openwam.train.goal_trainer import _construct, load_component_weights, _seed
from openwam.train.goal_curated import prepare_batch
from openwam.train.goal_overfit import FP32MasterAdamW, _write_json, export_inputs
from openwam.model.goal.history import history_kwargs
from overfit_visual_history import evaluate, rgb, perturb


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=5400);p.add_argument('--max-steps',type=int,default=100000)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'status.json').exists():raise FileExistsError(a.output)
    start=time.time();step=0
    def status(state,**kw):
        row=dict(status=state,step=step,started_unix=start,elapsed_seconds=time.time()-start,pid=__import__('os').getpid(),**kw)
        _write_json(a.output/'status.json',row);print(json.dumps(row),flush=True)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    smoke=a.root/'outputs/stage1_smoke_history_20261029'
    cfg=yaml.safe_load((smoke/'config.yaml').read_text());deps=json.loads((smoke/'dependencies.json').read_text())
    contexts=torch.load(smoke/'text_contexts.pt',weights_only=True,map_location='cpu')
    official=a.root/'assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model/checkpoint_step_154000.safetensors'
    status('loading')
    vae=_construct(cfg['components']['vae']);load_component_weights(vae,official,'video_backbone.vae.')
    vae.to(device='cuda',dtype=torch.bfloat16).eval().requires_grad_(False)
    expert=load_visual_history_bundle(a.checkpoint,vae=vae,text_dim=deps['text_dim'],
        precomputed_text_fingerprint=deps['text']['fingerprint'],device='cuda',dtype=torch.bfloat16)
    expert.use_gradient_checkpointing=True
    # Cache only frozen VAE outputs, bounded and keyed by exact normalized RGB bytes.
    encode=vae.encode;cache=OrderedDict()
    def cached(videos,device,**kw):
        videos=list(videos);key=tuple((tuple(x.shape),hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()) for x in videos)
        if key not in cache:
            cache[key]=encode(videos,device=device,**kw).detach()
            if len(cache)>512:cache.popitem(last=False)
        else:cache.move_to_end(key)
        return cache[key]
    vae.encode=cached
    train=[];held=[];groups=defaultdict(list)
    for path in sorted((smoke/'samples').glob('*.pt')):
        r,b=torch.load(path,weights_only=False,map_location='cpu');b['context'],b['context_lens']=contexts[r['instruction']]
        b=prepare_batch(r,b,device='cuda',dtype=torch.bfloat16)
        with torch.no_grad():
            b['history_latents']=torch.stack([expert.codec.encode(b['history_images'][:,j]) if bool(b['history_valid'][0,j])
                else torch.zeros(1,48,1,16,20,device='cuda',dtype=torch.bfloat16) for j in range(10)],dim=1)
        del b['history_images']
        if r['anchor_index']<2:
            groups[r['family']].append(len(train));train.append((r,b))
        else:held.append((r,b))
    protocol=dict(initial_checkpoint=str(a.checkpoint),official_dependencies=str(official),code=str(CODE),
        seconds=a.seconds,train_cases=len(train),families=sorted(groups),train_records=[r for r,b in train],
        eval_records=[r for r,b in held],split_note='Same smoke episodes. Four anchor_2 samples were used in prior overfit; not unseen episodes.',
        learning_rate=2e-5,history_learning_rate=2e-4,warmup_updates=100,history_dropout=.2,
        optimizer='FP32 master AdamW, BF16 forward, clip1',text='actual per-case instruction; native text CA',
        frozen_handoff='single final visual_history.safetensors shared by both Stage2 arms')
    _write_json(a.output/'protocol.json',protocol)
    expert.train_stage_a()
    groups_opt=[dict(name='wan',params=[p for n,p in expert.named_parameters() if n.startswith('dit.')],lr=2e-5),
        dict(name='history',params=[p for n,p in expert.named_parameters() if not n.startswith('dit.')],lr=2e-4)]
    optimizer=FP32MasterAdamW(groups_opt)
    families=sorted(groups);rng=random.Random(42);last_save=time.time()
    status('training',train_cases=len(train),families=len(families))
    while time.time()-start<a.seconds and step<a.max_steps:
        rng.shuffle(families)
        for family in families:
            index=rng.choice(groups[family]);b=dict(train[index][1]);step+=1;_seed(10000+step)
            expert.train_stage_a();optimizer.zero_grad()
            b['history_valid']=b['history_valid'] & (torch.rand_like(b['history_ages'])>=.2)
            b['history_ages']=torch.where(b['history_valid'],b['history_ages'],0.)
            # FP32MasterAdamW exposes its underlying groups through optimizer.
            for g,base in zip(optimizer.optimizer.param_groups,[2e-5,2e-4]):g['lr']=base*min(1.,step/100)
            before=time.time();result=expert(b['observed'],b['context'],b['targets'],context_lens=b['context_lens'],
                observed_view_valid=b['observed_view_valid'],**history_kwargs(b))
            if not torch.isfinite(result['loss']):raise FloatingPointError('nonfinite Stage1 loss')
            result['loss'].backward();norm=torch.nn.utils.clip_grad_norm_(expert.parameters(),1.,error_if_nonfinite=True,foreach=False)
            row=dict(step=step,family=family,case=index,loss=float(result['loss']),image_loss=float(result['fm_loss']),
                tcp_loss=float(result['tcp_loss']),valid_tcp=result['valid_points'],grad_norm=float(norm),gate=float(expert.history_gate),
                step_seconds=time.time()-before,elapsed_seconds=time.time()-start)
            optimizer.step();del result
            with (a.output/'updates.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            if step%50==0:status('training',last_update=row)
            if time.time()-last_save>1800:
                save_visual_history_bundle(expert,a.output/'latest.safetensors',dependency_identities=deps,training_metadata=dict(step=step,protocol=str(a.output/'protocol.json')))
                last_save=time.time()
            if time.time()-start>=a.seconds or step>=a.max_steps:break
    del optimizer;gc.collect();torch.cuda.empty_cache()
    status('saving')
    save_visual_history_bundle(expert,a.output/'visual_history.safetensors',dependency_identities=deps,
        training_metadata=dict(step=step,initial_checkpoint=str(a.checkpoint),protocol=str(a.output/'protocol.json')))
    expert.freeze();status('evaluating',teacher_frozen=True)
    report=a.output/'report';report.mkdir(exist_ok=True)
    # One fitted and one alternate anchor per family; same model/noise toggles history.
    selected=[train[groups[f][0]] for f in sorted(groups)]
    selected += [next(x for x in held if x[0]['family']==f) for f in sorted(groups)]
    cards=export_inputs(report,[r for r,b in selected],[b for r,b in selected]);results=[]
    for i,(r,b) in enumerate(selected):
        gt=[rgb(b['targets'].image)]
        values={mode:evaluate(expert,b if mode=='history' else perturb(b,b,'no_history'),report/f'case_{i:03d}'/mode,all_targets=gt)
            for mode in ['history','single']}
        results.append(dict(case=i,split='fit' if i<len(groups) else 'alternate_anchor',record=r,values=values))
        _write_json(report/'metrics.json',dict(cards=cards,cases=results))
    status('complete',teacher_frozen=True,checkpoint=str(a.output/'visual_history.safetensors'),evaluated=len(results))


if __name__=='__main__':main()
