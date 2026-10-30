#!/usr/bin/env python3
"""Full-size matched Stage2 overfit. Supervisor launches isolated sequential arms.

Uses the native architecture loader, Stage2 optimizer groups, loss and generation.
Default placement follows the configured Accelerate/DeepSpeed path. The optional
single-GPU mode uses sequential FP32-master AdamW with native parameter ownership.
Finite-case sampling and timed control differ from the production trainer. Run
supervisor directly; it invokes single-rank torchrun.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from openwam.train.z_action_report import atomic_json, render


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def validate_inputs(cfg, args):
    """Reject mismatched source/normalization identities before allocating models."""
    from omegaconf import OmegaConf
    manifest_path=Path(args.samples).parent/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    for key,hash_key,config_key in [('source_yaml','source_sha256','dataloader.source_yaml'),('normalization_registry','normalization_sha256','dataloader.normalization_registry')]:
        configured=OmegaConf.select(cfg,config_key)
        if not configured or sha(configured)!=manifest[hash_key] or sha(manifest[key])!=manifest[hash_key]:
            raise ValueError('prepared data identity mismatch: '+key)
    if (manifest['action_horizon'],manifest['video_stride'],manifest['video_frames'])!=(32,4,9):
        raise ValueError('unsupported cached sample geometry')
    for key in ('goal.checkpoint','training.finetune_ckpt_path'):
        value=OmegaConf.select(cfg,key)
        if not value or not Path(value).exists():raise ValueError('missing required shared initialization: '+key)
    return manifest


def shuffled_order(count, seed, epoch):
    order=list(range(count));random.Random(seed+epoch).shuffle(order);return order


def balanced_order(samples, seed, epoch):
    """Equal draws per family; rotate all cases in independently shuffled cycles."""
    groups={}
    for i,sample in enumerate(samples):groups.setdefault(sample['_eval']['family'],[]).append(i)
    rng=random.Random(seed+epoch);size=max(map(len,groups.values()));draws=[]
    for family in sorted(groups):
        values=groups[family][:];rng.shuffle(values)
        draws.extend(values[j%len(values)] for j in range(size))
    rng.shuffle(draws);return draws


def training_budget_remaining(step, elapsed, args):
    maximum = getattr(args, 'max_updates', None)
    if args.smoke:
        maximum = min(maximum, args.smoke_updates) if maximum is not None else args.smoke_updates
    return elapsed < args.seconds_per_arm and (maximum is None or step < maximum)


def evaluation_indices(samples, indices):
    if indices is None:
        return list(range(len(samples)))
    if not indices or len(set(indices)) != len(indices) or any(i < 0 or i >= len(samples) for i in indices):
        raise ValueError('evaluation indices must be unique valid sample indices')
    return list(indices)


def physical_action_metrics(predicted, sample):
    """Native metrics plus valid EEF delta translation and rotation-6D errors."""
    import numpy as np
    from openwam.train.stage2_eval import action_metrics, _array, _metric, SCATTER
    metrics=action_metrics(predicted,sample)
    stats=sample['_eval'].get('action_stats')
    if stats is None:return metrics
    raw=_array(sample['trace']['raw_action']);n=min(32,len(raw))
    lo=np.asarray(stats['q01']);span=np.maximum(np.asarray(stats['q99'])-lo,1e-6)
    physical=lo+(predicted[:n,SCATTER]+1)*span/2
    mask=_array(sample['action_mask'])[:n,SCATTER].astype(bool)&(np.asarray(stats['count'])>0)[None,:]
    if 'raw_action_mask' in sample['trace']:mask&=_array(sample['trace']['raw_action_mask'])[:n].astype(bool)
    for name,slots in [('eef_delta_translation_m_mae',[0,1,2,10,11,12]),('eef_rotation_6d_component_mae',[3,4,5,6,7,8,13,14,15,16,17,18])]:
        metrics[name]=_metric(np.abs(physical[:,slots]-raw[:n,slots])[mask[:,slots]])
    return metrics


def evaluate(arch, samples, folder, step, elapsed, args):
    import numpy as np
    import torch
    from openwam.train.stage2_eval import generation_conditions, preserve_evaluation_state, action_metrics, _curves, _aggregate
    from openwam.deploy.denoise_schedule import make_schedule
    final=folder/f'eval_{step:08d}'
    if final.exists():raise FileExistsError(final)
    work=folder/f'.eval_{step:08d}';work.mkdir()
    rows=[]
    with preserve_evaluation_state(arch):
        for i in evaluation_indices(samples, getattr(args, 'eval_case_indices', None)):
            sample=samples[i]
            name=f'case_{i:03d}';dest=work/name;dest.mkdir()
            torch.manual_seed(args.seed+i);torch.cuda.manual_seed_all(args.seed+i)
            kwargs=generation_conditions(sample,seed=args.seed+i,num_steps=args.denoise_steps,observation_mode=arch.goal_expert.observation_mode,observation_dtype=arch.dtype)
            kwargs['decode_video']=args.arm=='joint'
            shift=float(getattr(arch.action_backbone,'shift_action',5.))
            schedule=make_schedule('sync',arch.video_scheduler,arch.action_scheduler,num_steps=args.denoise_steps,shift=shift,shift_video=getattr(arch.video_backbone,'shift_video',None))
            result=arch.generate(schedule,sample['prompt'],shift=shift,**kwargs)
            predicted=np.asarray(result['normalized_actions'])
            if predicted.shape!=(32,112) or not np.isfinite(predicted).all():raise ValueError('nonfinite or wrong prediction shape')
            if args.arm=='none' and result.get('video') is not None:raise ValueError('action-only returned video')
            torch.save(dict(normalized_actions=torch.as_tensor(predicted),target=sample['action'],mask=sample['action_mask']),dest/'prediction.pt')
            _curves(dest,predicted,sample)
            video=result.get('video')
            if video is not None:
                for j,image in enumerate(video):image.save(dest/f'video_{j:02d}.png')
            sample['video'][0].save(dest/'current.png')
            sample['video'][-1].save(dest/'ground_truth_future.png')
            rows.append(dict(name=name,family=sample['_eval']['family'],granularity=sample.get('granularity'),metrics=physical_action_metrics(predicted,sample)))
    atomic_json(work/'metrics.json',dict(step=step,elapsed_seconds=elapsed,cases=rows,aggregate=_aggregate(rows),video_status='not_applicable' if args.arm=='none' else 'predicted'))
    os.replace(work,final);render(folder.parent)



def build_master_optimizer(architecture, training):
    """Keep native Stage2 ownership/rates while updating BF16 weights in FP32."""
    from openwam.train.goal_overfit import FP32MasterAdamW
    from openwam.train.utils.stage2 import build_stage2_parameter_groups
    groups = build_stage2_parameter_groups(architecture, training)
    # Per-group values preserve the native optimizer's configured betas; the
    # reusable master optimizer otherwise uses torch AdamW's default betas.
    betas = tuple(training.get('adam_betas', [0.9, 0.95]))
    for group in groups:
        group['betas'] = betas
    parameters = [p for group in groups for p in group['params']]
    optimizer = FP32MasterAdamW(groups, weight_decay=float(training.weight_decay))
    return optimizer, parameters


def build_single_gpu_accelerator():
    """No model/optimizer wrapping or partition-sized DeepSpeed buffers."""
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('--single-gpu-master requires exactly one process')
    # A caller can launch under an Accelerate environment as well as torchrun.
    # This explicit mode owns its optimizer and must not inherit DeepSpeed.
    os.environ['ACCELERATE_USE_DEEPSPEED'] = 'false'
    from accelerate import Accelerator
    accelerator = Accelerator(mixed_precision='bf16', device_placement=False,
                              gradient_accumulation_steps=1)
    if accelerator.num_processes != 1 or accelerator.distributed_type.name == 'DEEPSPEED':
        raise ValueError('--single-gpu-master requires a basic single-process Accelerator')
    if accelerator.device.type != 'cuda':
        raise ValueError('--single-gpu-master requires a CUDA device')
    return accelerator


def worker(args):
    import torch
    from omegaconf import OmegaConf
    from openwam.train.openwam_trainer import OpenWAMTrainer
    from openwam.train.utils.checkpointing import save_weights,save_config
    import importlib.util
    spec=importlib.util.spec_from_file_location('native_train',ROOT/'scripts/train.py');native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
    folder=Path(args.output)/args.arm;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'status.json').exists():raise FileExistsError('refusing to overwrite an existing arm; select a fresh output')
    cfg=OmegaConf.load(args.config)
    OmegaConf.update(cfg,'model.architecture.future_video_mode',args.arm,force_add=True)
    cfg.project.seed=args.seed;cfg.training.batch_size=1;cfg.training.gradient_accumulation_steps=1
    cfg.training.initialize_model_on_cpu=True
    cfg.training.lambda_video=1. if args.arm=='joint' else 0.
    manifest=validate_inputs(cfg,args)
    samples=torch.load(args.samples,map_location='cpu',weights_only=False)
    from openwam.dataloader.utils.video_color import validate_stage2_color
    for sample in samples:validate_stage2_color(sample)
    if not samples:raise ValueError('no samples')
    if args.smoke and args.smoke_case_indices:
        samples=[samples[i] for i in args.smoke_case_indices]
    elif args.smoke and args.smoke_case_limit:
        samples=[sample for sample in samples if bool(sample['action_mask'].any())][:args.smoke_case_limit]
        if not samples:raise ValueError('no valid smoke samples')
    if any(not bool(s['action_mask'].any()) for s in samples):raise ValueError('empty action supervision')
    atomic_json(folder/'status.json',dict(status='setup',mode=args.arm,samples=len(samples),samples_sha256=sha(args.samples),seed=args.seed,pid=os.getpid()))
    render(folder.parent)
    accelerator=build_single_gpu_accelerator() if args.single_gpu_master else native._build_accelerator(cfg)
    trainer=OpenWAMTrainer(cfg,accelerator=accelerator,dataset=samples)
    if args.single_gpu_master:
        # The native CPU initialization and frozen dependency checks complete
        # first. Place once, before making device-local FP32 master weights.
        trainer.architecture.set_dtype_device(torch.bfloat16, accelerator.device)
        trainer.architecture.move_frozen_to_device(accelerator.device)
        optimizer,params=build_master_optimizer(trainer.architecture,cfg.training)
        optimizer_groups=optimizer.optimizer.param_groups
    else:
        optimizer=trainer.build_optimizer()
        optimizer_groups=optimizer.param_groups
        params=[p for group in optimizer_groups for p in group['params']]
    optimizer_mode='single_gpu_fp32_master_adamw' if args.single_gpu_master else 'native_deepspeed'
    OmegaConf.update(cfg,'training.comparison_optimizer',optimizer_mode,force_add=True)
    # Explicit experiment-wide multiplier; preserve native parameter ownership.
    base_lrs=[float(group['lr'])*args.lr_scale for group in optimizer_groups]
    for group,lr in zip(optimizer_groups,base_lrs):group['lr']=lr
    teacher=trainer.architecture.goal_expert
    teacher_ids={id(p) for p in teacher.parameters()}
    if any(id(p) in teacher_ids for p in params):raise ValueError('teacher in optimizer')
    if any(p.requires_grad for p in teacher.parameters()):raise ValueError('teacher not frozen')
    teacher_versions=[p._version for p in teacher.parameters()]
    loader=torch.utils.data.DataLoader(samples,batch_size=1,collate_fn=lambda x:x)
    if not args.single_gpu_master:
        optimizer,loader,_=trainer.prepare_accelerate(optimizer,loader,None)
        optimizer_groups=optimizer.param_groups
        params=[p for group in optimizer_groups for p in group['params']]
    arch=accelerator.unwrap_model(trainer.architecture)
    checkpoint_dir=folder/'checkpoints';checkpoint_dir.mkdir()
    arch.save_assets_for_deployment(str(checkpoint_dir),cfg)
    if trainer._ckpt_source_dir:
        from openwam.train.utils.ckpt_model_loader import copy_ckpt_artifacts
        copy_ckpt_artifacts(trainer._ckpt_source_dir,str(checkpoint_dir))
    from openwam.dataloader.registry import build_dataset
    from openwam.train.utils.stage2_assets import save_native_normalization
    stats_dataset=build_dataset(cfg.dataloader,split='train')
    save_native_normalization(checkpoint_dir,stats_dataset)
    atomic_json(checkpoint_dir/'fixed_cases_manifest.json',manifest)
    atomic_json(checkpoint_dir/'comparison_provenance.json',dict(samples_sha256=sha(args.samples),config_sha256=sha(args.config),future_video_mode=args.arm,seed=args.seed,smoke=args.smoke,lr_scale=args.lr_scale,warmup_updates=args.warmup_updates,peak_learning_rates=base_lrs,initial_checkpoint=str(cfg.training.finetune_ckpt_path),optimizer=dict(mode=optimizer_mode,state='fresh',betas=list(cfg.training.adam_betas),weight_decay=float(cfg.training.weight_decay),master_dtype='float32' if args.single_gpu_master else None,foreach=False if args.single_gpu_master else None),rng_note='Same per-update seed; joint video sampling consumes RNG before action sampling, so action noise draws need not be identical.'))
    save_config(str(checkpoint_dir),cfg)
    step=0;epoch=0;started=time.monotonic();started_unix=time.time();last_eval=-1
    optimizer_step_in_progress=False
    atomic_json(folder/'status.json',dict(status='evaluating',step=0,started_unix=started_unix,budget_seconds=args.seconds_per_arm,deadline_unix=started_unix+args.seconds_per_arm))
    def snapshot():
        nonlocal last_eval
        if last_eval==step:return
        if step>0:save_weights(accelerator,trainer.architecture,str(folder/'checkpoints'),step,final=False)
        evaluate(arch,samples,folder,step,time.monotonic()-started,args)
        last_eval=step
    try:
        snapshot()
        while training_budget_remaining(step,time.monotonic()-started,args):
            for index in (shuffled_order(len(samples),args.seed,epoch) if args.smoke else balanced_order(samples,args.seed,epoch)):
                before=time.monotonic();torch.manual_seed(args.seed+step);torch.cuda.manual_seed_all(args.seed+step)
                warmup=min(1.,(step+1)/max(1,args.warmup_updates))
                for group,lr in zip(optimizer_groups,base_lrs):group['lr']=lr*warmup
                with accelerator.accumulate(trainer.architecture):
                    losses=trainer.compute_loss([copy.deepcopy(samples[index])])
                    if not bool(torch.isfinite(losses['total'])):raise FloatingPointError('nonfinite loss')
                    accelerator.backward(losses['total'])
                    if args.single_gpu_master:
                        torch.nn.utils.clip_grad_norm_(params,float(cfg.training.max_grad_norm),
                            foreach=False,error_if_nonfinite=True)
                    else:
                        accelerator.clip_grad_norm_(params,float(cfg.training.max_grad_norm))
                    optimizer_step_in_progress=True
                    optimizer.step();optimizer.zero_grad()
                    optimizer_step_in_progress=False
                if not args.single_gpu_master and accelerator.optimizer_step_was_skipped:
                    raise RuntimeError('optimizer skipped update')
                step+=1
                if any(p._version!=v for p,v in zip(teacher.parameters(),teacher_versions)):raise RuntimeError('frozen teacher changed')
                row=dict(step=step,sample=index,epoch=epoch,learning_rates=[float(g['lr']) for g in optimizer_groups],elapsed_seconds=time.monotonic()-started,step_seconds=time.monotonic()-before,losses={k:float(v.detach()) for k,v in losses.items()},host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
                with open(folder/'updates.jsonl','a') as f:f.write(json.dumps(row)+'\n');f.flush()
                atomic_json(folder/'status.json',dict(status='training',started_unix=started_unix,budget_seconds=args.seconds_per_arm,deadline_unix=started_unix+args.seconds_per_arm,**row));render(folder.parent)
                if step%args.eval_every==0:snapshot()
                if not training_budget_remaining(step,time.monotonic()-started,args):break
            epoch+=1
        snapshot()
    finally:
        # An optimizer exception can leave some parameters partly updated. Only
        # publish the latest weights when a whole optimizer update completed.
        if optimizer_step_in_progress:
            print('Skipping final checkpoint: optimizer update did not complete',flush=True)
        elif step>0 and not (checkpoint_dir/f'checkpoint_step_{step}.safetensors').exists():
            try:save_weights(accelerator,trainer.architecture,str(checkpoint_dir),step,final=True)
            except BaseException:traceback.print_exc()
    requested_updates=getattr(args,'max_updates',None)
    if requested_updates is not None and step != requested_updates:
        raise TimeoutError(f'completed {step} updates, requested {requested_updates}; wall-clock safety budget exhausted')
    atomic_json(folder/'status.json',dict(status='complete',requested_updates=requested_updates,step=step,started_unix=started_unix,finished_unix=time.time(),budget_seconds=args.seconds_per_arm,elapsed_seconds=time.monotonic()-started,teacher_frozen=True));render(folder.parent)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True,help='fully resolved native Stage2 query config YAML')
    p.add_argument('--samples',required=True);p.add_argument('--output',required=True)
    p.add_argument('--seconds-per-arm',type=float,default=18000);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--eval-every',type=int,default=500);p.add_argument('--denoise-steps',type=int,default=20)
    p.add_argument('--max-updates',type=int,default=None,help='Exact optimizer updates; time budget remains a safety deadline, early expiry is failure')
    p.add_argument('--eval-case-indices',type=int,nargs='+',default=None,help='Explicit diagnostic subset; all cached samples remain in training')
    p.add_argument('--smoke-updates',type=int,default=2)
    p.add_argument('--smoke-case-limit',type=int,default=1,help='smoke-only valid-case limit; 0 uses full set')
    p.add_argument('--smoke-case-indices',type=int,nargs='+',default=None)
    p.add_argument('--smoke',action='store_true');p.add_argument('--arm',choices=['joint','none'])
    p.add_argument('--single-gpu-master',action='store_true',help='Basic Accelerator with native ownership and sequential FP32-master AdamW; no DeepSpeed')
    p.add_argument('--lr-scale',type=float,default=1.)
    p.add_argument('--warmup-updates',type=int,default=0)
    p.add_argument('--no-legacy-report',action='store_true',help='External supervisor owns the current visual-history report')
    args=p.parse_args()
    if args.max_updates is not None and args.max_updates<1:p.error('max-updates must be positive')
    if args.smoke and args.max_updates is not None and args.max_updates!=args.smoke_updates:p.error('smoke and exact update budgets must agree')
    if not math.isfinite(args.lr_scale) or args.lr_scale<=0 or args.warmup_updates<0:p.error('invalid learning rate multiplier/warmup')
    if args.no_legacy_report:
        global render
        render=lambda root:None
    if args.seconds_per_arm<=0 or args.eval_every<1 or args.denoise_steps<1 or args.smoke_updates<1 or args.smoke_case_limit<0:p.error('positive budget/evaluation settings required')
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    if args.arm:
        if (root/args.arm/'status.json').exists():
            raise FileExistsError('existing arm output is immutable; select a fresh output')
        try:worker(args)
        except BaseException:
            atomic_json(root/args.arm/'status.json',dict(status='failed',traceback=traceback.format_exc(),pid=os.getpid()));render(root);raise
    else:
        if (root/'supervisor.json').exists() or any((root/f'{m}.log').exists() or (root/m/'status.json').exists() for m in ('joint','none')):
            raise FileExistsError('existing comparison output is immutable; select a fresh output')
        identity=dict(status='running',pid=os.getpid(),argv=sys.argv,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),config_sha256=sha(args.config),samples_sha256=sha(args.samples))
        atomic_json(root/'supervisor.json',identity)
        render(root)
        for mode in ('joint','none'):
            cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=1',str(Path(__file__).resolve()),*sys.argv[1:],'--arm',mode]
            with open(root/f'{mode}.log','x') as log:
                result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                atomic_json(root/'supervisor.json',dict(identity,status='failed',mode=mode,returncode=result.returncode));render(root);raise SystemExit(result.returncode)
        atomic_json(root/'supervisor.json',dict(identity,status='complete'));render(root)

if __name__=='__main__':main()
