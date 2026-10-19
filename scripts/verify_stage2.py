#!/usr/bin/env python3
"""Bounded evidence collection around the REAL Hydra/DeepSpeed Stage2 trainer.

This script never implements an optimizer loop. Run under single-rank torchrun;
arguments after ``--`` are forwarded verbatim to scripts/train.py. JSON survives
Python exceptions and intentional post-checkpoint stops, but cannot survive a
host/process kill. --resolve-assets prints a launch plan without loading models.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import runpy
import sys
import time
import traceback


class VerificationStop(BaseException):
    """Escape Hydra's Exception handler only AFTER a successful paired save."""


class VerificationError(RuntimeError):
    pass


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def tensor_hash(value):
    import torch
    return hashlib.sha256(value.detach().to('cpu').contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def scalar(value):
    if hasattr(value, 'detach'):
        value = value.detach().item()
    value = float(value)
    return value if math.isfinite(value) else str(value)


class Probe:
    """Read-only hooks; snapshots are whole representative tensors on CPU."""
    def __init__(self, report, stop_after_checkpoint=None):
        self.path = Path(report)
        self.stop_after_checkpoint = stop_after_checkpoint
        self.started = time.monotonic()
        self.last_update = self.started
        self.trainer = None
        self.parameters = {}
        self.baselines = {}
        self.teacher_parameters = []
        self.handles = []
        self.data = dict(format='openwam.stage2.verification.v1', status='running',
            scope='actual trainer instrumentation; single-process evidence; no GPU bitwise-equivalence claim',
            argv=list(sys.argv), batches=[], updates=[], checkpoints=[], representatives={})

    def resources(self):
        import torch
        result = dict(wall_seconds=time.monotonic()-self.started,
            host_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024)
        device = torch.device(self.trainer.accelerator.device) if self.trainer else None
        if (device is None or device.type=='cuda') and torch.cuda.is_initialized():
            device = device if device is not None else torch.cuda.current_device()
            result.update(cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
                          cuda_peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)))
        return result

    def write(self):
        self.data['resources'] = self.resources()
        atomic_json(self.path, self.data)

    def ownership(self, trainer, optimizer):
        self.trainer = trainer
        if int(trainer.accelerator.num_processes) != 1:
            raise ValueError('verification harness requires a single process; distributed verification is pending')
        architecture = trainer.accelerator.unwrap_model(trainer.architecture)
        goal = getattr(architecture, 'goal_expert', None)
        if goal is None:
            raise VerificationError('verification requires the actual frozen goal expert')
        goal_ids = {id(p) for p in goal.parameters()}
        optimizer_ids = {id(p) for g in optimizer.param_groups for p in g['params']}
        self.data['optimizer'] = dict(parameter_tensors=len(optimizer_ids),
            groups=[dict(name=g.get('name'), lr=float(g['lr']), parameter_tensors=len(g['params']))
                    for g in optimizer.param_groups], goal_parameter_tensors=len(goal_ids),
            goal_overlap=len(goal_ids & optimizer_ids), ownership_before_deepspeed=True)
        self.data['teacher'] = dict(optimizer_excluded=not bool(goal_ids & optimizer_ids))
        if goal_ids & optimizer_ids:
            raise VerificationError('frozen goal expert entered optimizer')

    def capture_baseline(self):
        architecture = self.trainer.accelerator.unwrap_model(self.trainer.architecture)
        named = dict(architecture.named_parameters())
        predicates = {
            'video': lambda n: n.startswith('video_backbone.dit.patch_embedding.'),
            'action': lambda n: n.startswith('action_backbone.blocks.'),
            'proprio': lambda n: n.startswith('proprio_encoder.'),
            'goal_adapter': lambda n: n.startswith('goal_adapter.patch.'),
        }
        self.parameters = {}
        self.data['proprio_conditioning'] = dict(observed_samples=0,active_samples=0,inactive_samples=0,unknown_samples=0)
        for category, predicate in predicates.items():
            selected = next(((n,p) for n,p in named.items() if predicate(n) and p.requires_grad and p.ndim >= 2
                and (category!='action' or (p.ndim==2 and n.endswith('.weight')))), None)
            if selected is None:
                raise VerificationError(f'no whole representative tensor for {category}')
            name, parameter = selected
            self.parameters[category] = parameter
            self.baselines[category] = parameter.detach().cpu().clone()
            self.data['representatives'][category] = dict(name=name, shape=list(parameter.shape),
                dtype=str(parameter.dtype), numel=parameter.numel(), baseline_sha256=tensor_hash(parameter),
                gradient=dict(calls=0, finite=True, nonzero=False, norm=0.))
        goal = architecture.goal_expert
        self.teacher_parameters = list(goal.parameters())
        self.teacher_versions = [p._version for p in self.teacher_parameters]
        self.teacher_hash_parameter = min(self.teacher_parameters, key=lambda p:p.numel())
        self.teacher_hash = tensor_hash(self.teacher_hash_parameter)
        self.data['teacher'].update(parameter_tensors=len(self.teacher_parameters),
            parameter_count=sum(p.numel() for p in self.teacher_parameters),
            baseline_representative_sha256=self.teacher_hash)
        self.last_update = time.monotonic()

    def prepared(self, trainer):
        self.trainer = trainer
        self.capture_baseline()
        for category, parameter in self.parameters.items():
            def hook(gradient, category=category):
                import torch
                values = gradient.detach()
                if values.is_sparse:
                    values = values.coalesce().values()
                item = self.data['representatives'][category]['gradient']
                item['calls'] += 1
                item['finite'] = item['finite'] and bool(torch.isfinite(values).all().item())
                item['nonzero'] = item['nonzero'] or bool(torch.count_nonzero(values).item())
                norm = float(values.float().norm().item())
                item['norm'] = max(item['norm'], norm) if math.isfinite(norm) else item['norm']
                # Returning None preserves the exact incoming gradient.
            self.handles.append(parameter.register_hook(hook))
        self.data['backend'] = dict(model_type=type(trainer.architecture).__name__,
            accelerator_type=type(trainer.accelerator).__name__,
            device=str(trainer.accelerator.device), world_size=int(trainer.accelerator.num_processes))
        self.write()

    def batch(self, batch, losses):
        import torch
        sources = []
        allowed = ('identity','draw_id','source_repo','source_parquet','episode','anchor',
                   'profile_id','selected_range','last_source','info_sha256','video_alignment','video_source_indices')
        for sample in batch if isinstance(batch,list) else [batch]:
            trace = sample.get('trace', {})
            source = {key:trace[key] for key in allowed if key in trace}
            source['granularity'] = sample.get('granularity')
            # A disabled observed token has zero gradient by design. Only an
            # explicit, correctly shaped boolean mask proves inactivity; missing
            # or malformed evidence must retain the ordinary gradient checks.
            observed=dict(active=None,valid_dimensions=None)
            mask=sample.get('proprio_mask');proprio=sample.get('proprio')
            if mask is not None and proprio is not None:
                flags=torch.as_tensor(mask)
                if flags.dtype==torch.bool and flags.shape==torch.as_tensor(proprio).shape:
                    observed.update(active=bool(flags.any()),valid_dimensions=int(flags.sum()))
            counts=self.data['proprio_conditioning'];counts['observed_samples']+=1
            key='unknown_samples' if observed['active'] is None else ('active_samples' if observed['active'] else 'inactive_samples')
            counts[key]+=1
            observed['state_dropped']=bool(sample.get('_observation_augmentation',{}).get('state_dropped',False))
            source['observed_proprio']=observed
            sources.append(source)
        self.data['batches'].append(dict(sources=sources,
            losses={key:scalar(value) for key,value in losses.items()}))

    def inspect(self):
        import torch
        if not self.parameters:
            self.data['checks'] = dict(passed=False, failures=['no prepared training model'])
            return
        teacher = self.data['teacher']
        teacher.update(all_frozen=all(not p.requires_grad for p in self.teacher_parameters),
            all_grad_none=all(p.grad is None for p in self.teacher_parameters),
            versions_unchanged=all(p._version==v for p,v in zip(self.teacher_parameters,self.teacher_versions)),
            representative_sha256=tensor_hash(self.teacher_hash_parameter))
        teacher['representative_unchanged'] = teacher['representative_sha256']==self.teacher_hash
        failures = []
        for category, parameter in self.parameters.items():
            current = parameter.detach().cpu()
            baseline = self.baselines[category]
            item = self.data['representatives'][category]
            item.update(changed_elements=int(torch.count_nonzero(current != baseline)),
                max_abs_change=scalar((current.float()-baseline.float()).abs().max()),
                current_sha256=tensor_hash(current), parameter_finite=bool(torch.isfinite(current).all()))
            gradient = item['gradient']
            condition=self.data['proprio_conditioning']
            inactive=(category=='proprio' and condition['observed_samples']>0
                and condition['inactive_samples']==condition['observed_samples'])
            if inactive:
                item['gradient_update_status']='not_applicable_inactive_observation'
                if not(item['parameter_finite'] and gradient['finite']):
                    failures.append(f'{category}: nonfinite parameter or gradient despite inactive observation')
            else:
                item['gradient_update_status']='required'
                if not(item['changed_elements'] and item['parameter_finite'] and gradient['finite'] and gradient['nonzero']):
                    failures.append(f'{category}: finite nonzero gradient and whole tensor update required')
        for check in ('optimizer_excluded','all_frozen','all_grad_none','versions_unchanged','representative_unchanged'):
            if not teacher[check]:
                failures.append('teacher: '+check)
        finite_losses = all(isinstance(value,(float,int)) and math.isfinite(value)
            for batch in self.data['batches'] for value in batch['losses'].values())
        finite_metrics = all(isinstance(value,(float,int)) and math.isfinite(value)
            for update in self.data['updates'] for value in update['metrics'].values())
        if not finite_metrics:
            failures.append('nonfinite metric/global gradient norm')
        if not finite_losses:
            failures.append('nonfinite loss')
        if not self.data['updates']:
            failures.append('no optimizer update recorded')
        self.data['checks'] = dict(passed=not failures, failures=failures, finite_losses=finite_losses, finite_metrics=finite_metrics,
            nonzero_fk_observed=any(b['losses'].get('fk_valid_count',0)>0
                for b in self.data['batches'] if isinstance(b['losses'].get('fk_valid_count',0),(int,float))))

    @contextmanager
    def instrument(self, module):
        trainer_class = module.OpenWAMTrainer
        patches = []
        def replace(owner, name, replacement):
            original = getattr(owner,name)
            patches.append((owner,name,original))
            setattr(owner,name,replacement(original))
        def build(original):
            def wrapped(trainer,*args,**kwargs):
                optimizer = original(trainer,*args,**kwargs)
                self.ownership(trainer,optimizer)
                return optimizer
            return wrapped
        def prepare(original):
            def wrapped(trainer,*args,**kwargs):
                if int(trainer.accelerator.num_processes)!=1:
                    raise ValueError('verification early stop requires a single process')
                result = original(trainer,*args,**kwargs)
                self.prepared(trainer)
                return result
            return wrapped
        def compute(original):
            def wrapped(trainer,batch,*args,**kwargs):
                self.batch_started = time.monotonic()
                result = original(trainer,batch,*args,**kwargs)
                self.batch(batch,result)
                return result
            return wrapped
        def log(original):
            def wrapped(trainer,*args,**kwargs):
                result = original(trainer,*args,**kwargs)
                now = time.monotonic()
                self.data['updates'].append(dict(opt_step=int(kwargs['opt_step']),
                    micro_step=int(kwargs['global_step']), interval_wall_seconds=now-self.last_update,
                    training_update_wall_seconds=now-self.batch_started,
                    metrics={k:scalar(v) for k,v in kwargs['metrics'].items()}))
                self.data['run_dir'] = str(kwargs['output_path'])
                self.last_update = now
                self.inspect()
                self.write()
                return result
            return wrapped
        def load(original):
            def wrapped(*args,**kwargs):
                metadata = original(*args,**kwargs)
                self.data['resume'] = metadata
                # Full-state restore legitimately rewrites ALL parameter versions.
                self.capture_baseline()
                self.write()
                return metadata
            return wrapped
        def save(original):
            def wrapped(accelerator, output_path, global_step, opt_step, epoch, **kwargs):
                if self.stop_after_checkpoint is not None and int(accelerator.num_processes)!=1:
                    raise ValueError('verification early stop requires a single process')
                result = original(accelerator,output_path,global_step,opt_step,epoch,**kwargs)
                accelerator.wait_for_everyone()
                state = Path(output_path)/f'accel_state_step_{global_step}'
                record = dict(step=int(global_step), state_dir=str(state),
                    weights=str(Path(output_path)/f'checkpoint_step_{global_step}.safetensors'))
                self.data['checkpoints'].append(record)
                self.write()
                if self.stop_after_checkpoint is not None and int(global_step)==self.stop_after_checkpoint:
                    marker = json.loads((state/'trainer_state.json').read_text())
                    if marker.get('global_step')!=global_step or not Path(record['weights']).is_file():
                        raise VerificationError('refusing stop without a completed paired checkpoint')
                    self.data['stop_checkpoint'] = record
                    raise VerificationStop()
                return result
            return wrapped
        try:
            replace(trainer_class,'build_optimizer',build)
            replace(trainer_class,'prepare_accelerate',prepare)
            replace(trainer_class,'compute_loss',compute)
            replace(trainer_class,'log_step',log)
            replace(module,'load_full_state',load)
            replace(module,'save_full_state',save)
            yield
        finally:
            for handle in self.handles:
                handle.remove()
            for owner,name,original in reversed(patches):
                setattr(owner,name,original)


def run_verification(report, trainer_module, run_training, *, stop_after_checkpoint=None):
    probe = Probe(report,stop_after_checkpoint)
    probe.write()
    try:
        with probe.instrument(trainer_module):
            try:
                run_training()
                probe.data['status'] = 'completed'
            except VerificationStop:
                probe.data['status'] = 'checkpoint_stop'
            probe.inspect()
            if not probe.data['checks']['passed']:
                raise VerificationError('Stage2 verification failed: '+ '; '.join(probe.data['checks']['failures']))
            if stop_after_checkpoint is not None and 'stop_checkpoint' not in probe.data:
                raise VerificationError('requested checkpoint stop was not reached')
    except BaseException as error:
        probe.data['status'] = 'failed'
        probe.data['error'] = dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
        raise
    finally:
        probe.write()
    return probe.data


def validate_smoke_config(cfg):
    """Fail before full model construction if an override exceeds this smoke."""
    from omegaconf import OmegaConf
    required = {'goal.enabled':True, 'project.seed':1, 'training.max_steps':3,
        'training.batch_size':1, 'training.gradient_accumulation_steps':1,
        'training.save_steps':1, 'training.keep_last_k_ckpts':3,
        'training.save_full_states_for_resume':True, 'training.debug':False,
        'evaluation.diagnostic_subset':True, 'evaluation.max_cases':2,
        'evaluation.num_steps':20}
    for key,value in required.items():
        if OmegaConf.select(cfg,key)!=value:
            raise ValueError(f'bounded Stage2 smoke requires {key}={value!r}')
    if int(os.environ.get('WORLD_SIZE','1'))!=1:
        raise ValueError('verification harness requires a single process')


def resolve_smoke_plan(asset_dir, output_dir):
    asset_dir, output_dir = Path(asset_dir), Path(output_dir).resolve()
    contract = json.loads((asset_dir/'stage1_asset_contract.json').read_text())
    summary = json.loads((asset_dir/'preflight_sample_v4'/'summary.json').read_text())
    common = ['--config-name', 'goal/stage2_joint_sample', 'project.seed=1', 'training.max_steps=3',
        'training.batch_size=1','training.gradient_accumulation_steps=1',
        'training.save_steps=1','training.keep_last_k_ckpts=3', 'training.save_full_states_for_resume=true',
        'evaluation.diagnostic_subset=true','evaluation.max_cases=2','evaluation.num_steps=20',
        'training.dataset_num_workers=0','project.wandb=null',
        'goal.checkpoint='+str(contract['checkpoint']),
        'dataloader.source_yaml='+str(summary['source_yaml']),
        'dataloader.normalization_registry='+str(summary['normalization_registry']),
        'project.output_dir='+str(output_dir)]
    fresh = common+['training.finetune_ckpt_path='+str(contract['training']['baseline_checkpoint']),
                    'training.resume_ckpt_path=null']
    resume = common+['training.finetune_ckpt_path=null','training.resume_ckpt_path=RUN_DIR']
    prefix = [sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=1',
              'scripts/verify_stage2.py']
    return dict(fresh=fresh,resume_template=resume,
        preflight_command=[sys.executable,'scripts/train.py',*fresh,'goal.preflight=true'],
        fresh_command=prefix+['--report',str(output_dir/'fresh.json'),'--stop-after-checkpoint','2','--',*fresh],
        resume_command_template=prefix+['--report',str(output_dir/'resume.json'),'--',*resume],
        resume_note='Replace RUN_DIR with fresh.json run_dir; use a fresh OS process and unchanged max_steps=3.')


def run_training_entry(train_script):
    """Validate every composed config before the real training dispatch.

    Load the original script without its __main__ call, wrap only its _train
    dispatch, then invoke its unchanged Hydra main. The Stage2-only preflight
    branch cannot guard a config which disables or omits goal.enabled.
    """
    namespace = runpy.run_path(str(train_script), run_name='stage2_verified_train')
    original_train = namespace['_train']
    train_globals = original_train.__globals__
    def checked_train(cfg):
        validate_smoke_config(cfg)
        return original_train(cfg)
    train_globals['_train'] = checked_train
    try:
        return namespace['main']()
    finally:
        train_globals['_train'] = original_train


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path)
    parser.add_argument('--stop-after-checkpoint',type=int)
    parser.add_argument('--resolve-assets',type=Path)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('overrides',nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.resolve_assets:
        if not args.output_dir:
            parser.error('--resolve-assets requires --output-dir')
        print(json.dumps(resolve_smoke_plan(args.resolve_assets,args.output_dir),indent=2))
        return
    if not args.report:
        parser.error('--report is required')
    if args.stop_after_checkpoint is not None and args.stop_after_checkpoint<1:
        parser.error('--stop-after-checkpoint must be positive')
    overrides = args.overrides[1:] if args.overrides[:1]==['--'] else args.overrides
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root))
    os.environ['HYDRA_FULL_ERROR']='1'
    # Record import/startup failures as well as failures inside the trainer.
    atomic_json(args.report,dict(format='openwam.stage2.verification.v1',status='starting'))
    try:
        from openwam.train import openwam_trainer
        saved_argv = sys.argv
        try:
            sys.argv = [str(root/'scripts/train.py'),*overrides]
            run_verification(args.report,openwam_trainer,
                lambda:run_training_entry(root/'scripts/train.py'),
                stop_after_checkpoint=args.stop_after_checkpoint)
        finally:
            sys.argv = saved_argv
    except BaseException as error:
        if json.loads(args.report.read_text()).get('status')!='failed':
            atomic_json(args.report,dict(format='openwam.stage2.verification.v1',status='failed',
                error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())))
        raise


if __name__=='__main__':
    main()
