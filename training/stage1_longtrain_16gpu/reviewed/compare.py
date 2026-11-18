#!/usr/bin/env python3
"""Content-bound eight-GPU native RFC Stage1 comparison and legacy continuation."""
from __future__ import annotations

import argparse
import copy
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import rfc_distributed as d

base=d.base
ARMS=('default','no_history','no_g','no_text')
base.ARMS=ARMS
EXPECTED_MANIFEST='987b385a32586300e4c1ad6f49e0d1d5a49d7cbfaa0a3917992e74b983e10704'


def runner_identity():
    return {name:base.sha256(d.ROOT/name) for name in ('compare.py','rfc_distributed.py','frozen_v1/compare.py','frozen_v1/rfc_compare.py')}


def migrate_data(args):
    identity=base.bind_code(args.code_dir)
    data=Path(args.data).resolve()
    if base.sha256(data/'manifest.json')!=args.expected_manifest_sha256:
        raise ValueError('explicit source prepared-manifest SHA256 differs')
    manifest=base.load_manifest(data,verify=True)
    frozen={name:base.sha256(d.ROOT/'frozen_v1'/name) for name in ('compare.py','rfc_compare.py')}
    if frozen!=manifest['runner'] or identity['python_tree_sha256']!=manifest['code']['python_tree_sha256']:
        raise ValueError('frozen prepared-runner/model source identity differs')
    receipt=dict(format='openwam.rfc_stage1.content_identity_migration.v1',data=str(data),
        manifest_sha256=args.expected_manifest_sha256,prepared_files=manifest['files'],code=identity,
        original_runner=frozen,accelerated_runner=runner_identity(),
        policy='Identical prepared bytes; no VAE/T5 rerun, no manifest rewrite; explicit execution migration only.')
    base.atomic_json(args.output,receipt)
    return dict(receipt=args.output,manifest_sha256=receipt['manifest_sha256'],files=len(manifest['files']))


def evaluation_arm(arm):
    return arm


def report_four(runs_root,output):
    payload=base.report(runs_root,output)
    matched=payload['matched_validation']
    matched['all_four_present']=matched.pop('all_three_present')
    matched['participant_count']=len(matched['participants'])
    page=Path(output)/'index.html'
    text=page.read_text().replace('three-arm','four-arm').replace('All three arms present','All four arms present')
    text=text.replace('green: no_g.','green: no_g. Purple: no_text.')
    text=text.replace("['#1677bb','#e67e22','#249647']","['#1677bb','#e67e22','#249647','#8e44ad']")
    base.atomic_text(page,text)
    base.atomic_json(Path(output)/'metrics.json',payload)
    return payload


def train(args):
    if int(os.environ.get('WORLD_SIZE','1'))!=8 or args.updates!=1000 or args.seed!=42:
        raise ValueError('production requires torchrun world=8, 1000 updates and seed42')
    if args.eval_every!=100 or args.checkpoint_every!=500 or args.visual_count!=4 or args.sampling_steps!=20:
        raise ValueError('matched contract requires eval100/checkpoint500/four images/20 sampling steps')
    rank=int(os.environ['RANK']);local_rank=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank);device=torch.device('cuda',local_rank)
    dist.init_process_group('nccl',timeout=timedelta(minutes=60))
    try:
        identity=base.bind_code(args.code_dir)
        receipt=json.loads(Path(args.identity).read_text());data=Path(args.data).resolve()
        if receipt['format']!='openwam.rfc_stage1.content_identity_migration.v1' or receipt['data']!=str(data) or receipt['accelerated_runner']!=runner_identity():
            raise ValueError('content migration receipt/accelerated runner differs')
        if receipt['manifest_sha256']!=EXPECTED_MANIFEST or base.sha256(data/'manifest.json')!=receipt['manifest_sha256']:
            raise ValueError('frozen prepared manifest differs')
        manifest=base.load_manifest(data,verify=rank==0)
        if receipt['code']!=identity or receipt['prepared_files']!=manifest['files']:
            raise ValueError('source/frozen prepared files differ')
        if manifest['dtype']!='bfloat16' or manifest['seed']!=42:
            raise ValueError('prepared dtype/seed differs')
        config=copy.deepcopy(manifest['config'])
        if rank==0 and base.sha256(config['baseline_checkpoint'])!=manifest['baseline']['sha256']:
            raise ValueError('foundation checkpoint SHA256 differs')
        dist.barrier()
        # Preserve the original scientific contract verbatim for strict import.
        config.update(stage=1,max_updates=1000,checkpoint_every=500,optimizer_device='cpu',dtype='bfloat16',seed=42)
        if (config['learning_rate'],config['new_learning_rate'])!=(1e-5,1e-4):
            raise ValueError('native LR contract differs')
        memory=config.setdefault('memory',{})
        if args.arm=='no_history':memory['history_dropout']=1.
        if args.arm=='no_g':memory['goal_dropout']=1.
        if memory.get('warmup_fraction',.2)!=.2 or memory.get('warmup_y_text_probability',.5)!=.5:
            raise ValueError('native keyframe-text warmup contract differs')
        from openwam.model.goal.goal_memory import GoalMemoryExpert,GoalMemoryConfig
        from openwam.train.goal_memory import parameter_groups
        from openwam.train.goal_optimizer import ShardedFP32MasterAdamW
        base.seed_all(42)
        dit=base.construct_component(config,'dit',device='cpu',dtype=torch.bfloat16)
        model=GoalMemoryExpert(dit,GoalMemoryConfig(**memory),dit_config=config['components']['dit']['extra_kwargs']).to(device=device,dtype=torch.bfloat16)
        if model.dit_config.get('num_layers')!=30 or model.dit_config.get('dim')!=3072:
            raise ValueError('production must retain full thirty-layer Wan 5b 3072 backbone')
        if args.arm=='no_text':d.configure_no_text(model)
        adapter=d.DistributedLossAdapter(model,args.arm)
        ddp=DDP(adapter,device_ids=[local_rank],output_device=local_rank,find_unused_parameters=True,
                gradient_as_bucket_view=True,broadcast_buffers=False,bucket_cap_mb=50)
        groups=parameter_groups(model,learning_rate=config['learning_rate'],new_learning_rate=config['new_learning_rate'])
        betas=tuple(config.get('adam_betas',(.9,.999)));eps=float(config.get('adam_eps',1e-8));wd=config.get('weight_decay',.01)
        optimizer=ShardedFP32MasterAdamW(groups,betas=betas,eps=eps,weight_decay=wd)
        entries=manifest['records']['train'];stream=base.sample_stream(entries,8000,seed=42)
        population=dict(manifest_sha256=base.sha256(data/'manifest.json'),
            stream_sha256=hashlib.sha256(np.asarray(stream,dtype='<i8').tobytes()).hexdigest(),
            prepared_files=manifest['files'],source_code=identity['python_tree_sha256'])
        scientific=dict(config=config,dependencies=manifest['dependencies'],population=population,dit_config=model.dit_config,
            comparison=dict(arm=args.arm,seed=42,batch_size=8,eval_every=100,
                noise='CPU-fp32-SeedSequence-773/881-v1',draws='finite-global-3:7-v1'))
        contract=dict(scientific=scientific,execution=dict(format=d.STATE_FORMAT,world=8,local_batch=1,
            optimizer='native-zero1-gpu-fp32',torch=torch.__version__,dtype='bfloat16',
            runner=runner_identity(),branch_rng='seed42+100003*(absolute_update+1)+1009*rank',
            loss='global-valid-value-and-point-denominators-ddp-compensated-v1',
            no_text='constant-one-token-all-G/Y-text-query-gates-zero' if args.arm=='no_text' else None))
        output=Path(args.output)
        if rank==0:
            if not args.resume and output.exists() and any(output.iterdir()):
                raise ValueError('nonempty output; use fresh output or distributed --resume')
            output.mkdir(parents=True,exist_ok=True)
            base.atomic_json(output/'contract.json',contract)
            base.atomic_json(output/'execution.json',dict(torch=torch.__version__,world=8,local_batch=1,global_batch=8,
                optimizer='native ShardedFP32MasterAdamW',data_identity=str(Path(args.identity).resolve()),
                initialized_from=args.legacy_resume or args.resume or 'foundation-seed42'))
        dist.barrier()
        update=0
        if args.legacy_resume:
            if args.arm=='no_text':raise ValueError('no_text fourth arm must cold-start')
            update=d.import_legacy(model,optimizer,groups,args.legacy_resume,legacy_contract=scientific,output=output,betas=betas,eps=eps,weight_decay=wd)
        if args.resume:update=d.restore_generation(model,optimizer,args.resume,contract=contract)
        rows=base.read_jsonl(output/'train.jsonl') if rank==0 else []
        if rows and rows[-1]['update']>update:raise ValueError('log cursor exceeds resume: recover to a fresh output')
        offset=rows[-1].get('wall_seconds',0.) if rows else 0.;clock=time.monotonic()
        evaluator=base.make_evaluator(data,manifest,device=device,dtype=torch.bfloat16,seed=42,arm=evaluation_arm(args.arm)) if rank==0 else None
        visualizer=base.make_visualizer(data,manifest,config,output=output,device=device,dtype=torch.bfloat16,seed=42,
            arm=evaluation_arm(args.arm),count=4,steps=20) if rank==0 else None
        def assess(step):
            dist.barrier()
            if rank==0:
                with base.preserved_rng(),torch.no_grad():
                    model.eval();start=time.monotonic();result=evaluator(model,step)
                    result.update(arm=args.arm,text_conditioning=args.arm!='no_text')
                    base.append_jsonl(output/'eval.jsonl',dict(update=step,samples=step*8,
                        wall_seconds=offset+time.monotonic()-clock,**result))
                    base.append_jsonl(output/'phases.jsonl',dict(update=step,phase='evaluation',seconds=time.monotonic()-start))
                    if step in {0,100,500,1000}:
                        start=time.monotonic();visualizer(model,step)
                        base.append_jsonl(output/'phases.jsonl',dict(update=step,phase='visualization',seconds=time.monotonic()-start))
                model.train()
                report_four(output.parent,output.parent/'report')
            dist.barrier()
        # Imported update500 gets fresh fixed evaluation/images at its exact weights.
        if not args.resume:assess(update)
        end=min(1000,update+args.stop_after) if args.stop_after else 1000
        ddp.train()
        while update<end:
            started=time.monotonic();indices=stream[update*8:(update+1)*8]
            row=torch.load(data/entries[indices[rank]]['file'],map_location='cpu',weights_only=False)
            batch,_=base.collate_prepared([row],data,device=device,dtype=torch.bfloat16)
            base.seed_all(42+100003*(update+1)+1009*rank)
            sigma,noise=d.rank_noise(batch['target_latents'],seed=42,cursor=update,rank=rank,world=8)
            optimizer.zero_grad()
            result=ddp(batch,update=update,total_updates=1000,sigma=sigma,noise=noise)
            if not torch.isfinite(result['loss']):raise FloatingPointError('nonfinite global objective')
            result['loss'].backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config.get('max_grad_norm',1.),error_if_nonfinite=True,foreach=False)
            optimizer.step();torch.cuda.synchronize(device);update+=1
            if rank==0:
                target=torch.empty((8,*batch['target_latents'].shape[1:]),dtype=torch.bfloat16)
                _,full_noise=base.common_noise(target,seed=42,cursor=update-1,validation=False)
                fm=float(result['fm_loss']);uv=float(result['uv_loss'])
                record=dict(update=update,samples=update*8,arm=args.arm,wall_seconds=offset+time.monotonic()-clock,
                    seconds=time.monotonic()-started,gradient_norm=float(norm),valid_points=result['valid_points'],
                    valid_latent_values=result['valid_latent_values'],fm_loss=fm,uv_loss=uv,
                    loss=model.config.lambda_image*fm+model.config.lambda_uv*uv,
                    grain_counts={g:sum(entries[i]['granularity']==g for i in indices) for g in ('task','subtask')},
                    sample_ids=[entries[i]['id'] for i in indices],common_noise_sha256=d.tensor_digest(full_noise))
                base.append_jsonl(output/'train.jsonl',record);print(json.dumps(record,allow_nan=False),flush=True)
            del result,batch,noise,sigma,row
            stop=torch.tensor(int((output/'STOP_REQUESTED').exists()) if rank==0 else 0,device=device)
            dist.broadcast(stop,src=0);requested=bool(stop.item())
            if update%100==0 or update==end or requested:assess(update)
            if update%500==0 or update==end or requested:
                start=time.monotonic();generation=d.checkpoint_generation(model,optimizer,output,update=update,contract=contract,dependencies=manifest['dependencies'])
                if rank==0:base.append_jsonl(output/'phases.jsonl',dict(update=update,phase='checkpoint',seconds=time.monotonic()-start))
                if rank==0:report_four(output.parent,output.parent/'report')
            if requested:break
        summary=dict(update=update,budget=1000,status='complete' if update==1000 else 'interrupted',resume=str(output/'checkpoint.json'))
        if rank==0:base.atomic_json(output/'summary.json',summary)
        return summary
    finally:
        dist.destroy_process_group()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='mode',required=True)
    p=sub.add_parser('migrate-data');p.add_argument('--code-dir',required=True);p.add_argument('--data',required=True)
    p.add_argument('--output',required=True);p.add_argument('--expected-manifest-sha256',default=EXPECTED_MANIFEST)
    p=sub.add_parser('train');p.add_argument('--code-dir',required=True);p.add_argument('--data',required=True)
    p.add_argument('--identity',required=True);p.add_argument('--output',required=True);p.add_argument('--arm',choices=ARMS,required=True)
    p.add_argument('--updates',type=int,default=1000);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--eval-every',type=int,default=100);p.add_argument('--checkpoint-every',type=int,default=500)
    p.add_argument('--visual-count',type=int,default=4);p.add_argument('--sampling-steps',type=int,default=20)
    p.add_argument('--stop-after',type=int);group=p.add_mutually_exclusive_group();group.add_argument('--legacy-resume');group.add_argument('--resume')
    p=sub.add_parser('report');p.add_argument('--runs-root',required=True);p.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    if args.mode=='migrate-data':result=migrate_data(args)
    elif args.mode=='train':
        if args.stop_after is not None and args.stop_after<1:parser.error('--stop-after must be positive')
        result=train(args)
    else:
        result=report_four(args.runs_root,args.output);result=dict(output=args.output,arms=list(result['arms']))
    if int(os.environ.get('RANK','0'))==0:print(json.dumps(result,indent=2,allow_nan=False),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
