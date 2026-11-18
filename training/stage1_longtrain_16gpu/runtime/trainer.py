#!/usr/bin/env python3
"""Fresh native 16-GPU Stage1 longtraining and bounded real optimizer profiles."""
from __future__ import annotations
import argparse
from contextlib import nullcontext
from datetime import timedelta
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time
import traceback
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import distributed16 as d

PINNED='/mnt/data/cpfs/b5/yeqingda/CodeX/runtime/openwam-rfc-goal-memory-16e1180-dlc'
DEFAULT_G='/mnt/data/cpfs/b5/yeqingda/CodeX/experiments/rfc_stage1_compare_20261111/data'
DEFAULT_CSTAR=str(d.REVIEWED/'baseline_cstar/data')


def read_config(path):
    path=Path(path)
    if path.suffix=='.json':return json.loads(path.read_text())
    import yaml
    return yaml.safe_load(path.read_text())

def validate_execution(config,*,world,local_batch,accumulation,production):
    if config.get('seed',42)!=42:raise ValueError('fresh comparison requires seed42')
    if min(local_batch,accumulation)<1:raise ValueError('local batch and accumulation must be positive')
    if production and world!=16:raise ValueError('production requires world16 (two nodes, eight GPUs each)')
    batch=world*local_batch*accumulation
    if config.get('global_batch_size',batch)!=batch:raise ValueError('global_batch_size must equal world*local_batch*accumulation')
    if config.get('schedule_reference_updates',50000)<1:raise ValueError('positive independent warmup schedule reference required')
    return dict(world=world,local_batch_size=local_batch,accumulation_steps=accumulation,global_batch_size=batch,
                schedule_reference_updates=config.get('schedule_reference_updates',50000))

def native_config(config):
    value=config.get('model_config')
    if value is not None:return read_config(value) if isinstance(value,str) else value
    return json.loads((Path(config.get('prepared_g',DEFAULT_G))/'manifest.json').read_text())['config']

def runner_identity():
    files=['trainer.py','distributed16.py','validation16.py','data_pipeline.py','raw_metadata_overlay.py','frozen_metadata_view.py','report_publish.py']
    return {name:d.sha256(d.ROOT/name) for name in files if (d.ROOT/name).is_file()}

def source_identity(code_dir):
    reviewed=d.load_reviewed();identity=reviewed.base.bind_code(code_dir)
    helpers={str(p.relative_to(d.REVIEWED)):d.sha256(p) for p in [d.REVIEWED/'rfc_distributed.py',d.REVIEWED/'frozen_v1/rfc_compare.py',d.REVIEWED/'baseline_cstar/cstar.py',d.REVIEWED/'full_generation_eval.py',d.REVIEWED/'reports_zh/build_reports.py',d.REVIEWED/'reports_zh/charts.js']}
    return dict(pinned=identity,reviewed_helpers=helpers)

def build_native(config,arch,device):
    cfg=native_config(config);base=d.load_reviewed().base;d.seed_all(42)
    if (cfg.get('learning_rate',1e-5),cfg.get('new_learning_rate',1e-4))!=(1e-5,1e-4):raise ValueError('native fresh LR must remain1e-5/1e-4')
    dit=base.construct_component(cfg,'dit',device='cpu',dtype=torch.bfloat16)
    if arch=='G':
        from openwam.model.goal.goal_memory import GoalMemoryExpert,GoalMemoryConfig
        from openwam.train.goal_memory import parameter_groups
        options=dict(cfg.get('memory',{}))
        if options.get('warmup_fraction',.2)!=.2 or options.get('warmup_y_text_probability',.5)!=.5:raise ValueError('default G native warmup differs')
        model=GoalMemoryExpert(dit,GoalMemoryConfig(**options),dit_config=cfg['components']['dit']['extra_kwargs']).to(device=device,dtype=torch.bfloat16)
        groups=parameter_groups(model,learning_rate=1e-5,new_learning_rate=1e-4)
    else:
        c=d.load_cstar()
        from openwam.train.goal_curated import goal_parameter_groups
        options=dict(num_history_tokens=32,pool_dim=256,pool_heads=8,adapter_block=0,language_conditioning=True,causal_history=True)
        model=c.PreparedVisualHistoryExpert(dit,c.PreparedGoalCodec(None),dit_config=cfg['components']['dit']['extra_kwargs'],shift=5.,lambda_image=.5,lambda_tcp=1.,use_gradient_checkpointing=True,**options).to(device=device,dtype=torch.bfloat16)
        model.train_stage_a();groups=goal_parameter_groups(model,learning_rate=1e-5,history_learning_rate=1e-4)
    if model.dit_config.get('num_layers')!=30 or model.dit_config.get('dim')!=3072:raise ValueError('full native thirty-layer3072 model required')
    return model,groups,cfg

class PreparedProfileProvider:
    """Only bounded preflight may draw from the existing real frozen256 pool."""
    def __init__(self,config,arch,rank,world,device,dtype):
        self.arch=arch;self.rank=rank;self.world=world;self.device=device;self.dtype=dtype
        self.data=Path(config.get('prepared_g',DEFAULT_G) if arch=='G' else config.get('prepared_cstar',DEFAULT_CSTAR))
        self.manifest=json.loads((self.data/'manifest.json').read_text());self.entries=self.manifest['records']['train']
        self.dependencies=self.manifest['dependencies'];self.identity=dict(mode='bounded-real-fixtures-only',data=str(self.data),manifest_sha256=d.sha256(self.data/'manifest.json'))
        self.checked=set()
        lengths={}
        for entry in self.entries:
            row=self._load(entry);context=Path(row['context_file']);context=context if context.is_absolute() else self.data/context
            lengths[entry['id']]=len(torch.load(context,map_location='cpu',weights_only=True))
        self.entries=sorted(self.entries,key=lambda e:lengths[e['id']],reverse=True)
        self.vae=None;self.text_encoder=None;self.tokenizer=None;self.warm_receipt=None
    def _load(self,entry):
        path=self.data/entry['file']
        if entry['file'] not in self.checked:
            expected=self.manifest['files'].get(entry['file'])
            if expected is None or d.sha256(path)!=expected:raise ValueError('profile fixture bytes differ: '+str(path))
            self.checked.add(entry['file'])
        row=torch.load(path,map_location='cpu',weights_only=False)
        cp=Path(row['context_file']);cp=cp if cp.is_absolute() else self.data/cp
        hashes={**self.manifest['files'],**self.manifest.get('external_files',{})}
        key=str(cp) if str(cp) in hashes else row['context_file']
        if key in hashes and d.sha256(cp)!=hashes[key]:raise ValueError('profile context bytes differ')
        return row
    def get_local_microbatches(self,update,global_batch_size,accumulation=1):
        local=global_batch_size//(self.world*accumulation);out=[]
        # Include the longest real contexts first to make the preflight bound
        # representative of the worst fixed real instruction sequence length.
        for micro in range(accumulation):
            start=update*global_batch_size+(micro*self.world+self.rank)*local
            entries=[self.entries[(start+i)%len(self.entries)] for i in range(local)]
            rows=[self._load(e) for e in entries]
            if self.arch=='G':batch,_=d.load_reviewed().base.collate_prepared(rows,self.data,device=self.device,dtype=self.dtype)
            else:batch=d.load_cstar().collate(rows,self.data,device=self.device,dtype=self.dtype)
            if self.arch=='G' and batch['text'].shape[1]<512:
                extra=512-batch['text'].shape[1];batch['text']=torch.nn.functional.pad(batch['text'],(0,0,0,extra));batch['text_mask']=torch.nn.functional.pad(batch['text_mask'],(0,extra),value=False)
            if self.arch=='Cstar' and batch['context'].shape[1]<512:
                batch['context']=torch.nn.functional.pad(batch['context'],(0,0,0,512-batch['context'].shape[1]))
            out.append((batch,dict(samples=[dict(id=e['id'],grain=e['granularity']) for e in entries],global_cursor=update*global_batch_size,micro=micro)))
        return out

def resolved_provider_config(config):
    resolved=dict(native_config(config));resolved.update(config)
    resolved['data_pipeline']=dict(config.get('data_pipeline',{}))
    return resolved

def build_provider(config,arch,rank,world,device,profile):
    if profile:return PreparedProfileProvider(config,arch,rank,world,device,torch.bfloat16)
    from data_pipeline import build_provider
    provider=build_provider(resolved_provider_config(config),arch,rank,world,device=device,dtype=torch.bfloat16)
    if getattr(provider,'identity',{}).get('mode')=='bounded-real-fixtures-only':raise ValueError('production forbids finite profile pool')
    return provider

def health(output,rank,phase,update,**extra):
    d.atomic_json(Path(output)/'health'/f'rank_{rank:02d}.json',dict(rank=rank,host=socket.gethostname(),pid=os.getpid(),phase=phase,update=update,unix_time=time.time(),**extra))

def _collect_metadata(micros):
    samples=[]
    for _,metadata in micros:
        values=metadata.get('samples',metadata.get('records',[])) if isinstance(metadata,dict) else metadata
        samples.extend(values)
    collected=[None]*dist.get_world_size();dist.all_gather_object(collected,samples)
    return collected

def publish_report(run_output,device):
    # Each architecture has one writer; no cross-node shared report lock.
    from validation16 import build_report
    run_output=Path(run_output)
    error=None
    if dist.get_rank()==0:
        try:build_report(run_output.parent,run_output/'report')
        except Exception as exc:error=repr(exc)
    d.collective_status(error,device)

def execute_update(ddp,model,optimizer,micros,*,arch,update,execution,device):
    batches=[b for b,_ in micros];counts,global_counts=d.whole_update_counts(batches,arch,device)
    d.seed_all(42+100003*(update+1)+1009*dist.get_rank())
    optimizer.zero_grad();numerators=[]
    for micro,(batch,_) in enumerate(micros):
        target=batch['target_latents'] if arch=='G' else batch['targets'].image
        noise=d.global_noise_slice(target,arch=arch,seed=42,update=update,rank=dist.get_rank(),world=execution['world'],accumulation=execution['accumulation_steps'],micro=micro)
        # Every micro synchronizes so reducer hooks retain gradient bucket views.
        with nullcontext():
            error=None
            try:
                result=ddp(batch,local_counts=counts[micro],global_counts=global_counts,noise=noise,update=update,schedule_reference_updates=execution['schedule_reference_updates'])
                if not torch.isfinite(result['loss']):raise FloatingPointError('nonfinite whole-update objective')
            except Exception as exc:error=repr(exc)
            d.collective_status(error,device)
            result['loss'].backward();numerators.append(result['numerators'])
        del result,noise
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True,foreach=False)
    optimizer.step();torch.cuda.synchronize(device)
    values=d.reduce_statistics(numerators,global_counts)
    return dict(fm_loss=values[0],**({'uv_loss':values[1]} if arch=='G' else {'tcp_loss':values[1]}),loss=(model.config.lambda_image*values[0]+model.config.lambda_uv*values[1] if arch=='G' else model.lambda_image*values[0]+model.lambda_tcp*values[1]),gradient_norm=float(norm),global_denominators=list(global_counts))

def recover_training_log(output,committed_update):
    """Preserve crash-tail evidence and resume from the committed optimizer cursor."""
    output=Path(output);path=output/'train.jsonl'
    if not path.exists():return 0.
    recovery=output/'recovery';recovery.mkdir(exist_ok=True)
    archive=recovery/f'uncommitted_after_{committed_update:09d}_{time.time_ns()}.jsonl'
    temporary=output/'train.jsonl.recovery.tmp';last=None;changed=False
    try:
        with path.open() as source,temporary.open('w') as target,archive.open('w') as tail:
            for line in source:
                if not line.strip():continue
                try:row=json.loads(line)
                except json.JSONDecodeError:
                    if line.endswith('\n'):raise
                    tail.write(line);changed=True;break
                if row['update']>committed_update:tail.write(line);changed=True
                else:target.write(d.canonical(row)+'\n');last=row
            target.flush();os.fsync(target.fileno());tail.flush();os.fsync(tail.fileno())
        if changed:os.replace(temporary,path)
        else:temporary.unlink();archive.unlink()
    finally:
        temporary.unlink(missing_ok=True)
    return last.get('wall_seconds',0.) if last else 0.

def run(args):
    config=read_config(args.config);rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    execution=validate_execution(config,world=world,local_batch=args.local_batch,accumulation=args.accumulation,production=args.mode=='train')
    if args.mode=='train' and args.dry_input:raise ValueError('--dry-input profile fixtures are forbidden in production')
    if args.resume and args.mode!='train':raise ValueError('profile always fresh')
    torch.cuda.set_device(local);device=torch.device('cuda',local);output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('TORCH_NCCL_ASYNC_ERROR_HANDLING','1')
    dist.init_process_group('nccl',timeout=timedelta(seconds=int(config.get('distributed_timeout_seconds',900))))
    update=0;stopping=False
    def request_stop(signum,frame):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    try:
        health(output,rank,'binding-source',0);error=None
        try:
            identity=source_identity(config.get('code_dir',PINNED))
            cfg=native_config(config);base=d.load_reviewed().base
            gmanifest=json.loads((Path(config.get('prepared_g',DEFAULT_G))/'manifest.json').read_text())
            if rank==0 and d.sha256(cfg['baseline_checkpoint'])!=gmanifest['baseline']['sha256']:raise ValueError('foundation checksum differs')
        except Exception as exc:error=repr(exc)
        d.collective_status(error,device);health(output,rank,'constructing-native-model',0);error=None
        try:model,groups,cfg=build_native(config,args.arch,device)
        except Exception as exc:error=repr(exc)
        d.collective_status(error,device)
        adapter=d.LossAdapter(model,args.arch)
        ddp=DDP(adapter,device_ids=[local],output_device=local,find_unused_parameters=True,gradient_as_bucket_view=True,broadcast_buffers=False,bucket_cap_mb=50)
        from openwam.train.goal_optimizer import ShardedFP32MasterAdamW
        optimizer=ShardedFP32MasterAdamW(groups,betas=(.9,.999),eps=1e-8,weight_decay=.01)
        error=None
        try:provider=build_provider(config,args.arch,rank,world,device,args.mode=='profile' and not args.profile_full_source)
        except Exception as exc:error=repr(exc)
        d.collective_status(error,device)
        if args.mode=='profile' and not args.profile_full_source:
            from data_pipeline import warm_frozen_dependencies
            health(output,rank,'warming-real-frozen-dependencies',0)
            frozen=warm_frozen_dependencies(resolved_provider_config(config),args.arch,device=device,dtype=torch.bfloat16,local_batch_size=args.local_batch,accumulation=args.accumulation)
            provider.vae=frozen['vae'];provider.text_encoder=frozen['text_encoder'];provider.tokenizer=frozen['tokenizer'];provider.warm_receipt=frozen['receipt'];provider.frozen_provider=frozen['provider']
            provider.identity['frozen_warmup']=frozen['receipt']
            for key in ('vae','text'):
                if frozen['dependencies'][key]['fingerprint']!=provider.dependencies[key]['fingerprint']:raise ValueError('profile frozen dependencies differ: '+key)
        if args.mode=='profile' and args.profile_full_source:
            provider.frozen_provider=provider
            health(output,rank,'warming-real-frozen-dependencies',0)
            provider.warm_receipt=provider.warm_dependencies(args.local_batch,args.accumulation)
        dependencies=provider.dependencies
        contract=dict(format='openwam.stage1_longtrain.contract.v1',arch=args.arch,fresh_foundation_seed=42,foundation=dict(path=cfg['baseline_checkpoint'],sha256=gmanifest['baseline']['sha256'] if rank==0 else None),source=identity,runner=runner_identity(),data=provider.identity,model_config=cfg,execution={**execution,'dtype':'bfloat16','optimizer':'native-zero1-gpu-FP32','gradient_sync':'each-micro-ddp-bucket-views','betas':[.9,.999],'eps':1e-8,'weight_decay':.01,'torch':torch.__version__},objective='native-global-whole-update-valid-denominators-ddp-compensated-v1',branch_rng='seed42+100003*(absolute_update+1)+1009*rank',noise='native-global-CPU-SeedSequence-microstep-major-v1',checkpoint_every=1000,report_every=1000,retained_generations=args.retain,validation_inputs={str(Path(config.get('prepared_g',DEFAULT_G))/'manifest.json'):d.sha256(Path(config.get('prepared_g',DEFAULT_G))/'manifest.json'),str(Path(config.get('prepared_cstar',DEFAULT_CSTAR))/'manifest.json'):d.sha256(Path(config.get('prepared_cstar',DEFAULT_CSTAR))/'manifest.json')})
        # Every rank binds the same rank-zero foundation digest.
        foundation=[contract['foundation'] if rank==0 else None];dist.broadcast_object_list(foundation,src=0);contract['foundation']=foundation[0]
        error=None
        if rank==0:
            try:
                old=output/'contract.json'
                if old.exists() and not args.resume:raise ValueError('output already contains a run contract; use a fresh directory')
                if old.exists() and d.canonical(json.loads(old.read_text()))!=d.canonical(contract):raise ValueError('saved run contract differs')
                d.atomic_json(old,contract)
            except Exception as exc:error=repr(exc)
        d.collective_status(error,device)
        if args.resume:
            update,resumed=d.restore_generation(model,optimizer,args.resume,contract=contract)
            if resumed['global_cursor']!=update*execution['global_batch_size']:raise ValueError('resumed data cursor differs')
        if args.mode=='profile':
            return profile(args,config,output,model,ddp,optimizer,provider,execution,device,rank)
        from validation16 import assess,build_report
        started=time.monotonic();wall_offset=0.
        error=None
        if rank==0:
            try:wall_offset=recover_training_log(output,update)
            except Exception as exc:error=repr(exc)
        d.collective_status(error,device)
        if args.resume and update%1000==0:
            expected=output/'validation'/f'step_{update:09d}'/'manifest.json'
            if not expected.is_file():
                health(output,rank,'recovering-pending-validation',update)
                assess(model,args.arch,config,output,update,device=device,rank=rank,world=world,vae=provider.vae)
            publish_report(output,device)
            dist.barrier()
        ddp.train();requested=False;last_saved_update=update if args.resume else None
        while True:
            stop_before=torch.tensor(int(stopping or (output/'STOP_REQUESTED').exists()),device=device)
            dist.all_reduce(stop_before,op=dist.ReduceOp.MAX)
            if bool(stop_before):
                requested=True
                if last_saved_update!=update:
                    health(output,rank,'checkpoint-stop-boundary',update)
                    d.checkpoint_generation(model,optimizer,output,update=update,global_cursor=update*execution['global_batch_size'],contract=contract,arch=args.arch,dependencies=dependencies,retain=args.retain)
                    assess(model,args.arch,config,output,update,device=device,rank=rank,world=world,vae=provider.vae)
                    publish_report(output,device)
                    dist.barrier()
                break
            health(output,rank,'preparing-data',update);begin=time.monotonic();micros=None;error=None
            try:
                micros=list(provider.get_local_microbatches(update,execution['global_batch_size'],execution['accumulation_steps']))
                if len(micros)!=execution['accumulation_steps']:raise ValueError('provider microbatch count differs')
            except Exception as exc:error=repr(exc)
            d.collective_status(error,device)
            data_seconds=time.monotonic()-begin;health(output,rank,'forward-backward-optimizer',update)
            result=execute_update(ddp,model,optimizer,micros,arch=args.arch,update=update,execution=execution,device=device);update+=1
            samples=_collect_metadata(micros);del micros
            flattened=[item for part in samples for item in part]
            sample_ids=[r.get('sample_id',r.get('id')) for r in flattened]
            sample_digest=__import__('hashlib').sha256(d.canonical(samples).encode()).hexdigest()
            grain_counts={grain:sum(r.get('granularity',r.get('grain'))==grain for r in flattened) for grain in ('task','subtask')}
            seconds=time.monotonic()-begin;max_seconds=torch.tensor(seconds,device=device);dist.all_reduce(max_seconds,op=dist.ReduceOp.MAX)
            if rank==0:
                row=dict(update=update,samples=update*execution['global_batch_size'],arm=args.arch,wall_seconds=wall_offset+time.monotonic()-started,seconds=float(max_seconds),samples_per_second=execution['global_batch_size']/float(max_seconds),data_seconds=data_seconds,sample_ids=sample_ids,grain_counts=grain_counts,global_draw_start=(update-1)*execution['global_batch_size'],global_draw_end=update*execution['global_batch_size'],sample_metadata_sha256=sample_digest,**result)
                d.append_jsonl(output/'train.jsonl',row);print(d.canonical(row),flush=True)
            health(output,rank,'optimizer-boundary',update)
            stop=torch.tensor(int(stopping or (output/'STOP_REQUESTED').exists()),device=device);dist.all_reduce(stop,op=dist.ReduceOp.MAX);requested=bool(stop)
            if update%1000==0 or requested or (args.stop_after is not None and update>=args.stop_after):
                health(output,rank,'checkpoint',update)
                generation=d.checkpoint_generation(model,optimizer,output,update=update,global_cursor=update*execution['global_batch_size'],contract=contract,arch=args.arch,dependencies=dependencies,retain=args.retain)
                last_saved_update=update
                d.append_jsonl(output/f'phases_rank_{rank:02d}.jsonl',dict(update=update,phase='checkpoint',generation=str(generation),unix_time=time.time()))
                health(output,rank,'validation',update)
                assess(model,args.arch,config,output,update,device=device,rank=rank,world=world,vae=provider.vae)
                publish_report(output,device)
                dist.barrier()
            if requested or (args.stop_after is not None and update>=args.stop_after):break
        summary=dict(status='stopped-at-optimizer-boundary',update=update,samples=update*execution['global_batch_size'],checkpoint=str(output/'checkpoint.json'),stop_requested=requested)
        if rank==0:d.atomic_json(output/'summary.json',summary)
        health(output,rank,'stopped',update);return summary
    except BaseException as exc:
        health(output,rank,'failed',update,error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        if dist.is_initialized():dist.destroy_process_group()


def profile(args,config,output,model,ddp,optimizer,provider,execution,device,rank):
    if args.profile_steps<1 or args.profile_warmup<1:raise ValueError('real profile requires warmup and measured optimizer steps')
    records=[];ddp.train();torch.cuda.reset_peak_memory_stats(device)
    for update in range(args.profile_warmup+args.profile_steps):
        health(output,rank,'real-profile',update);torch.cuda.synchronize(device);begin=time.monotonic()
        micros=provider.get_local_microbatches(update,execution['global_batch_size'],execution['accumulation_steps'])
        result=execute_update(ddp,model,optimizer,micros,arch=args.arch,update=update,execution=execution,device=device)
        torch.cuda.synchronize(device);elapsed=time.monotonic()-begin;del micros
        maximum=torch.tensor(elapsed,device=device);dist.all_reduce(maximum,op=dist.ReduceOp.MAX)
        free,total=torch.cuda.mem_get_info(device)
        row=dict(update=update+1,warmup=update<args.profile_warmup,seconds=float(maximum),samples_per_second=execution['global_batch_size']/float(maximum),peak_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_reserved_bytes=torch.cuda.max_memory_reserved(device),free_bytes=free,total_bytes=total,**result)
        records.append(row)
        if update==0:
            health(output,rank,'real-encode-with-Adam-moments-resident',update+1)
            provider.after_optimizer_warm_receipt=provider.frozen_provider.warm_dependencies(args.local_batch,args.accumulation)
        if rank==0:print(d.canonical(dict(event='real-profile-update',arch=args.arch,**row)),flush=True)
    measured=[r for r in records if not r['warmup']];free,total=torch.cuda.mem_get_info(device)
    receipt=dict(rank=rank,arch=args.arch,execution=execution,device=torch.cuda.get_device_name(device),peak_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_reserved_bytes=torch.cuda.max_memory_reserved(device),free_bytes=free,total_bytes=total,headroom_bytes=total-torch.cuda.max_memory_reserved(device),records=records)
    d.atomic_json(output/f'profile_rank_{rank:02d}.json',receipt);ranks=[None]*execution['world'];dist.all_gather_object(ranks,receipt)
    health(output,rank,'profile-validation-probe',len(records))
    from validation16 import assess
    assess(model,args.arch,config,output,len(records),device=device,rank=rank,world=execution['world'],vae=provider.vae,limit=execution['world'])
    torch.cuda.synchronize(device)
    free,total=torch.cuda.mem_get_info(device)
    probe=dict(rank=rank,peak_allocated_bytes=max(torch.cuda.max_memory_allocated(device),max(r['peak_allocated_bytes'] for r in records)),peak_reserved_bytes=max(torch.cuda.max_memory_reserved(device),max(r['peak_reserved_bytes'] for r in records)),free_bytes=free,total_bytes=total)
    probes=[None]*execution['world'];dist.all_gather_object(probes,probe)
    summary=dict(validation_probe_ranks=probes,frozen_dependency_warmup=provider.warm_receipt,frozen_dependency_after_optimizer_warmup=provider.after_optimizer_warm_receipt,status='real-native-optimizer-profile-complete',arch=args.arch,execution=execution,seconds=sum(r['seconds'] for r in measured),samples=len(measured)*execution['global_batch_size'],samples_per_second=len(measured)*execution['global_batch_size']/sum(r['seconds'] for r in measured),minimum_headroom_bytes=min(r['total_bytes']-r['peak_reserved_bytes'] for r in probes),ranks=ranks)
    if rank==0:d.atomic_json(output/'profile.json',summary)
    health(output,rank,'profile-complete',len(records));return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('train','profile'))
    p.add_argument('--arch',choices=('G','Cstar'),required=True);p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--local-batch',type=int,required=True);p.add_argument('--accumulation',type=int,default=1)
    p.add_argument('--profile-warmup',type=int,default=1);p.add_argument('--profile-steps',type=int,default=3);p.add_argument('--dry-input',action='store_true')
    p.add_argument('--profile-full-source',action='store_true',help='bounded real full-source decode/cache/encode/train timing; requires completed source audit');p.add_argument('--retain',type=int,default=2);p.add_argument('--resume');p.add_argument('--stop-after',type=int)
    args=p.parse_args(argv)
    if args.retain<1:p.error('--retain must be positive')
    if args.stop_after is not None and args.stop_after<1:p.error('--stop-after must be positive')
    result=run(args)
    if int(os.environ.get('RANK',0))==0:print(d.canonical(result),flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
