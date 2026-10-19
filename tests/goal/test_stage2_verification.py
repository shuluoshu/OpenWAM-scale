"""CPU contract tests for the read-only real-trainer verification wrapper."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from torch import nn

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts' / 'verify_stage2.py'


def harness():
    assert SCRIPT.is_file(), 'real-trainer verification harness has not been implemented'
    spec = importlib.util.spec_from_file_location('stage2_verify_under_test', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Architecture(nn.Module):
    def __init__(self):
        super().__init__()
        self.video_backbone = nn.Module()
        self.video_backbone.dit = nn.Module()
        self.video_backbone.dit.patch_embedding = nn.Linear(2, 2)
        self.action_backbone = nn.Module()
        self.action_backbone.blocks = nn.ModuleList([nn.Linear(2, 2)])
        self.action_backbone.action_encoder = nn.Linear(2, 2)
        self.proprio_encoder = nn.Linear(2, 2)
        self.goal_adapter = nn.Module()
        self.goal_adapter.patch = nn.Linear(2, 2)
        self.goal_expert = nn.Linear(2, 2).requires_grad_(False)


def trainer_module(tmp_path, world=1):
    arch = Architecture()
    accelerator = SimpleNamespace(num_processes=world, unwrap_model=lambda x: x,
        wait_for_everyone=lambda: None, device=torch.device('cpu'))
    class Trainer:
        def __init__(self):
            self.architecture = arch
            self.accelerator = accelerator
        def build_optimizer(self):
            return torch.optim.AdamW([p for p in arch.parameters() if p.requires_grad], lr=.05)
        def prepare_accelerate(self, optimizer, dataloader, scheduler):
            return optimizer, dataloader, scheduler
        def compute_loss(self, batch):
            x = torch.ones(1, 2)
            outputs = [arch.video_backbone.dit.patch_embedding(x),
                       arch.action_backbone.blocks[0](x), arch.proprio_encoder(x), arch.goal_adapter.patch(x)]
            return {'total': sum(o.square().sum() for o in outputs), 'fk_valid_count': torch.tensor(18.)}
        def log_step(self, **kwargs):
            return None
    def save(accelerator, output_path, global_step, opt_step, epoch, **kwargs):
        path = Path(output_path)
        (path / f'accel_state_step_{global_step}').mkdir(exist_ok=True)
        (path / f'checkpoint_step_{global_step}.safetensors').write_bytes(b'paired')
        (path / f'accel_state_step_{global_step}' / 'trainer_state.json').write_text(json.dumps({'global_step':global_step,'opt_step':opt_step}))
    return SimpleNamespace(OpenWAMTrainer=Trainer, save_full_state=save,
                           load_full_state=lambda *a, **k: {'opt_step':2}), arch


def run_update(module, tmp_path, *, checkpoint=1, batch=None):
    trainer = module.OpenWAMTrainer()
    optimizer = trainer.build_optimizer()
    trainer.prepare_accelerate(optimizer, None, None)
    losses = trainer.compute_loss(batch if batch is not None else [{'trace': {'draw_id':0, 'source_repo':'native',
        'source_parquet':'source.parquet','episode':8,'anchor':225,'raw_action':torch.ones(1000)},'granularity':'task'}])
    losses['total'].backward()
    optimizer.step()
    optimizer.zero_grad()
    trainer.log_step(metrics={'loss_total':float(losses['total']), 'fk_valid_count':18.},
                     opt_step=1, global_step=1, output_path=str(tmp_path))
    module.save_full_state(trainer.accelerator, str(tmp_path), checkpoint, checkpoint, 0)


def test_probe_records_actual_gradient_update_frozen_teacher_source_and_fk(tmp_path):
    h = harness()
    module, arch = trainer_module(tmp_path)
    report = tmp_path / 'evidence.json'
    h.run_verification(report, module, lambda: run_update(module, tmp_path))
    data = json.loads(report.read_text())
    assert data['status'] == 'completed'
    assert data['checks']['passed']
    assert set(data['representatives']) == {'video','action','proprio','goal_adapter'}
    for item in data['representatives'].values():
        assert item['changed_elements'] > 0
        assert item['gradient']['finite'] and item['gradient']['nonzero']
        assert item['gradient']['norm'] > 0
    assert data['teacher']['all_frozen'] and data['teacher']['all_grad_none']
    assert data['teacher']['versions_unchanged'] and data['teacher']['optimizer_excluded']
    assert data['batches'][0]['sources'][0]['draw_id'] == 0
    assert 'raw_action' not in data['batches'][0]['sources'][0]
    assert data['batches'][0]['losses']['fk_valid_count'] == 18
    assert data['resources']['host_peak_rss_bytes'] > 0
    assert len(data['updates']) == 1


def test_stop_runs_only_after_complete_original_checkpoint_and_restores_patches(tmp_path):
    h = harness()
    module, _ = trainer_module(tmp_path)
    original = module.save_full_state
    report = tmp_path/'stop.json'
    h.run_verification(report, module, lambda:run_update(module,tmp_path,checkpoint=2), stop_after_checkpoint=2)
    data = json.loads(report.read_text())
    assert data['status'] == 'checkpoint_stop'
    assert data['stop_checkpoint']['step'] == 2
    assert Path(data['stop_checkpoint']['state_dir'],'trainer_state.json').is_file()
    assert module.save_full_state is original


def test_save_failure_records_failure_without_intentional_stop(tmp_path):
    h = harness()
    module, _ = trainer_module(tmp_path)
    def broken(*args, **kwargs):
        raise OSError('checkpoint disk failure')
    module.save_full_state=broken
    report=tmp_path/'failed.json'
    with pytest.raises(OSError,match='checkpoint disk failure'):
        h.run_verification(report,module,lambda:run_update(module,tmp_path,checkpoint=2),stop_after_checkpoint=2)
    data=json.loads(report.read_text())
    assert data['status']=='failed' and 'checkpoint disk failure' in data['error']['message']
    assert 'stop_checkpoint' not in data


def test_early_stop_rejects_multiple_ranks_before_prepare(tmp_path):
    h=harness()
    module,_=trainer_module(tmp_path,world=2)
    with pytest.raises(ValueError,match='single'):
        h.run_verification(tmp_path/'world.json',module,lambda:run_update(module,tmp_path),stop_after_checkpoint=2)
    assert json.loads((tmp_path/'world.json').read_text())['status']=='failed'


def test_failure_before_model_construction_still_writes_json(tmp_path):
    h=harness()
    module,_=trainer_module(tmp_path)
    with pytest.raises(RuntimeError,match='preflight failed'):
        h.run_verification(tmp_path/'preflight.json',module,lambda:(_ for _ in ()).throw(RuntimeError('preflight failed')))
    assert json.loads((tmp_path/'preflight.json').read_text())['status']=='failed'


def test_teacher_parameter_change_is_failure_even_with_finite_losses(tmp_path):
    h=harness()
    module,arch=trainer_module(tmp_path)
    def run():
        run_update(module,tmp_path)
        with torch.no_grad():
            arch.goal_expert.weight.add_(1)
    with pytest.raises(h.VerificationError,match='verification'):
        h.run_verification(tmp_path/'teacher.json',module,run)
    data=json.loads((tmp_path/'teacher.json').read_text())
    assert not data['teacher']['versions_unchanged']
    assert not data['checks']['passed']


def test_post_resume_baseline_excludes_restoration_changes(tmp_path):
    h=harness()
    module,arch=trainer_module(tmp_path)
    def restore(*args,**kwargs):
        with torch.no_grad():
            for p in arch.parameters(): p.add_(2)
        return {'opt_step':2}
    module.load_full_state=restore
    def run():
        trainer=module.OpenWAMTrainer(); opt=trainer.build_optimizer()
        trainer.prepare_accelerate(opt,None,None)
        module.load_full_state(trainer.accelerator,'state',require_stage2=True)
    with pytest.raises(h.VerificationError):
        h.run_verification(tmp_path/'resume.json',module,run)
    data=json.loads((tmp_path/'resume.json').read_text())
    assert data['resume']['opt_step']==2
    assert data['teacher']['versions_unchanged']
    assert all(item['changed_elements']==0 for item in data['representatives'].values())


def test_assets_resolver_has_exact_bounded_fresh_and_resume_commands(tmp_path):
    h=harness()
    (tmp_path/'preflight_sample_v4').mkdir()
    (tmp_path/'stage1_asset_contract.json').write_text(json.dumps({'checkpoint':'/goal','training':{'baseline_checkpoint':'/baseline'}}))
    (tmp_path/'preflight_sample_v4/summary.json').write_text(json.dumps({'source_yaml':'/yaml','normalization_registry':'/registry'}))
    (tmp_path/'preflight_sample_v3').mkdir()
    (tmp_path/'preflight_sample_v3/summary.json').write_text(json.dumps({'source_yaml':'/old-yaml','normalization_registry':'/old-registry'}))
    plan=h.resolve_smoke_plan(tmp_path, tmp_path/'output')
    fresh=plan['fresh'];resume=plan['resume_template']
    for overrides in (fresh,resume):
        assert 'training.max_steps=3' in overrides and 'project.seed=1' in overrides
        assert 'evaluation.num_steps=20' in overrides
        assert 'training.save_steps=1' in overrides and 'training.keep_last_k_ckpts=3' in overrides
    assert 'dataloader.normalization_registry=/registry' in fresh
    assert 'dataloader.source_yaml=/yaml' in fresh
    assert not any('old-registry' in value or 'old-yaml' in value for value in fresh)
    assert 'training.finetune_ckpt_path=/baseline' in fresh
    assert 'training.finetune_ckpt_path=null' in resume
    assert 'training.resume_ckpt_path=RUN_DIR' in resume


def test_bounded_policy_refuses_long_or_changed_smoke_before_models():
    h=harness()
    from omegaconf import OmegaConf
    cfg=OmegaConf.create({'goal':{'enabled':True},'project':{'seed':1},
        'training':{'max_steps':3,'batch_size':1,'gradient_accumulation_steps':1,
        'save_steps':1,'keep_last_k_ckpts':3,'save_full_states_for_resume':True,'debug':False},
        'evaluation':{'diagnostic_subset':True,'max_cases':2,'num_steps':20}})
    h.validate_smoke_config(cfg)
    for field,value in [('training.max_steps',500),('project.seed',42),('evaluation.max_cases',64),
                         ('training.save_full_states_for_resume',False)]:
        broken=OmegaConf.create(OmegaConf.to_container(cfg))
        OmegaConf.update(broken,field,value)
        with pytest.raises(ValueError,match=field):
            h.validate_smoke_config(broken)


def test_incomplete_checkpoint_never_produces_clean_stop(tmp_path):
    h=harness()
    module,_=trainer_module(tmp_path)
    module.save_full_state=lambda *a,**kw:None
    with pytest.raises(FileNotFoundError):
        h.run_verification(tmp_path/'partial.json',module,lambda:run_update(module,tmp_path,checkpoint=2),stop_after_checkpoint=2)
    assert json.loads((tmp_path/'partial.json').read_text())['status']=='failed'


def test_nonfinite_gradient_is_reported_even_when_parameter_update_is_finite(tmp_path):
    h=harness()
    module,arch=trainer_module(tmp_path)
    def run():
        run_update(module,tmp_path)
        # A later read-only hook observation must preserve earlier finite evidence
        # while rejecting this gradient; no optimizer operation uses the NaN here.
        (arch.proprio_encoder.weight.sum()*float('nan')).backward()
    with pytest.raises(h.VerificationError):
        h.run_verification(tmp_path/'nan.json',module,run)
    data=json.loads((tmp_path/'nan.json').read_text())
    assert not data['representatives']['proprio']['gradient']['finite']


def test_launch_plan_composes_native_config_with_global_batch_one(tmp_path):
    h=harness()
    import os
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    repo=Path(os.environ.get('STAGE2_TEST_REPO',Path(__file__).resolve().parents[2]))
    (tmp_path/'preflight_sample_v4').mkdir()
    (tmp_path/'stage1_asset_contract.json').write_text(json.dumps({'checkpoint':'/goal','training':{'baseline_checkpoint':'/baseline'}}))
    (tmp_path/'preflight_sample_v4/summary.json').write_text(json.dumps({'source_yaml':'/yaml','normalization_registry':'/registry'}))
    plan=h.resolve_smoke_plan(tmp_path,tmp_path/'output')
    assert plan['fresh'][:2]==['--config-name','goal/stage2_joint_sample']
    with initialize_config_dir(config_dir=str(repo/'configs'),version_base=None):
        cfg=compose(config_name=plan['fresh'][1],overrides=plan['fresh'][2:])
    h.validate_smoke_config(cfg)
    assert cfg.training.offload_optimizer_device=='cpu'
    assert cfg.training.zero_stage==2
    assert cfg.training.batch_size*cfg.training.gradient_accumulation_steps==1
    assert OmegaConf.select(cfg,'dataloader.normalization_registry')=='/registry'


def test_scalar_teacher_hash_and_global_nonfinite_gradient_metric_are_supported(tmp_path):
    h=harness()
    assert h.tensor_hash(torch.tensor(2.,dtype=torch.bfloat16))
    module,_=trainer_module(tmp_path)
    original=module.OpenWAMTrainer.log_step
    def invalid_log(trainer, **kwargs):
        kwargs['metrics']['grad_norm']=float('nan')
        return original(trainer,**kwargs)
    module.OpenWAMTrainer.log_step=invalid_log
    with pytest.raises(h.VerificationError):
        h.run_verification(tmp_path/'metric.json',module,lambda:run_update(module,tmp_path))
    assert not json.loads((tmp_path/'metric.json').read_text())['checks']['finite_metrics']


def test_action_representative_is_transformer_linear_matrix_not_modulation(tmp_path):
    h=harness()
    module,arch=trainer_module(tmp_path)
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.modulation=nn.Parameter(torch.ones(1,6,2))
            self.proj=nn.Linear(2,2)
        def forward(self,x): return self.proj(x)
    arch.action_backbone.blocks[0]=Block()
    h.run_verification(tmp_path/'matrix.json',module,lambda:run_update(module,tmp_path))
    item=json.loads((tmp_path/'matrix.json').read_text())['representatives']['action']
    assert item['name'].endswith('.proj.weight') and len(item['shape'])==2


@pytest.mark.parametrize('preset,extra', [
    ('goal/stage2_joint_sample', ['goal.enabled=false']),
    ('train', []),
])
def test_actual_hydra_dispatch_rejects_disabled_or_missing_goal_before_construction(tmp_path,monkeypatch,preset,extra):
    h=harness()
    import os
    import sys
    repo=Path(os.environ.get('STAGE2_TEST_REPO',Path(__file__).resolve().parents[2]))
    actual_run_path=h.runpy.run_path
    reached=[]
    def forbidden(*args,**kwargs):
        reached.append('Accelerator/model construction')
        raise AssertionError('non-Stage2 config reached construction')
    def load_actual_train(*args,**kwargs):
        namespace=actual_run_path(*args,**kwargs)
        globals_=namespace['_train'].__globals__
        globals_['_build_accelerator']=forbidden
        globals_['_train_openwam']=forbidden
        return namespace
    monkeypatch.setattr(h.runpy,'run_path',load_actual_train)
    monkeypatch.setenv('HYDRA_FULL_ERROR','1')
    monkeypatch.setattr(sys,'argv',[str(repo/'scripts/train.py'),'--config-name',preset,
        'hydra.run.dir='+str(tmp_path),'hydra.output_subdir=null',*extra])
    with pytest.raises(ValueError,match='goal.enabled'):
        h.run_training_entry(repo/'scripts/train.py')
    assert reached==[]


def test_valid_composed_config_still_calls_original_actual_train_dispatch(tmp_path,monkeypatch):
    h=harness()
    import os
    import sys
    repo=Path(os.environ.get('STAGE2_TEST_REPO',Path(__file__).resolve().parents[2]))
    actual_run_path=h.runpy.run_path
    reached=[]
    # The instrumented original dispatch keeps the real script's global scope.
    def load_with_actual_globals(*args,**kwargs):
        namespace=actual_run_path(*args,**kwargs)
        original=namespace['_train']
        def marker(cfg):
            reached.append(cfg.training.max_steps)
            raise RuntimeError('original training dispatch reached')
        import types
        wrapped=types.FunctionType(marker.__code__,original.__globals__,closure=marker.__closure__)
        namespace['_train']=wrapped
        original.__globals__['_train']=wrapped
        return namespace
    monkeypatch.setattr(h.runpy,'run_path',load_with_actual_globals)
    monkeypatch.setenv('HYDRA_FULL_ERROR','1')
    monkeypatch.setattr(sys,'argv',[str(repo/'scripts/train.py'),'--config-name','goal/stage2_joint_sample',
        'hydra.run.dir='+str(tmp_path),'hydra.output_subdir=null','project.seed=1','training.max_steps=3',
        'training.save_steps=1','training.keep_last_k_ckpts=3',
        'evaluation.diagnostic_subset=true','evaluation.max_cases=2'])
    with pytest.raises(RuntimeError,match='original training dispatch reached'):
        h.run_training_entry(repo/'scripts/train.py')
    assert reached==[3]


@pytest.mark.parametrize('kind,expected_pass', [
    ('inactive',True),('inactive_no_update',True),('partial',False),('missing',False),
    ('malformed',False),('stale_dropout_trace',False),('inactive_nonfinite',False),
])
def test_proprio_nonzero_requirement_uses_actual_observed_mask_only(tmp_path,kind,expected_pass):
    h=harness();module,arch=trainer_module(tmp_path)
    sample={'proprio':torch.ones(1,2),'proprio_mask':torch.zeros(1,2,dtype=torch.bool),
        '_observation_augmentation':{'state_dropped':True}}
    if kind=='partial':sample['proprio_mask'][0,1]=True
    elif kind=='missing':del sample['proprio_mask']
    elif kind=='malformed':sample['proprio_mask']=torch.zeros(2,dtype=torch.bool)
    elif kind=='stale_dropout_trace':sample['proprio_mask'].fill_(True)
    def zero_proprio_loss(trainer,batch):
        x=torch.ones(1,2)
        outputs=[arch.video_backbone.dit.patch_embedding(x),arch.action_backbone.blocks[0](x),arch.goal_adapter.patch(x)]
        # Hook fires with finite zero gradient when the observed token is disabled.
        return {'total':sum(o.square().sum() for o in outputs)+arch.proprio_encoder(x).square().sum()*0.,
                'fk_valid_count':torch.tensor(18.)}
    module.OpenWAMTrainer.compute_loss=zero_proprio_loss
    if kind=='inactive_no_update':
        module.OpenWAMTrainer.build_optimizer=lambda trainer:torch.optim.AdamW(
            [p for p in arch.parameters() if p.requires_grad],lr=.05,weight_decay=0.)
    def run():
        run_update(module,tmp_path,batch=[sample])
        if kind=='inactive_nonfinite':(arch.proprio_encoder.weight.sum()*float('nan')).backward()
    path=tmp_path/'conditioned.json'
    if expected_pass:h.run_verification(path,module,run)
    else:
        with pytest.raises(h.VerificationError):h.run_verification(path,module,run)
    data=json.loads(path.read_text());item=data['representatives']['proprio']
    assert data['checks']['passed']==expected_pass
    if expected_pass:
        assert item['gradient_update_status']=='not_applicable_inactive_observation'
        assert item['gradient']['finite'] and not item['gradient']['nonzero']
        assert data['proprio_conditioning']==dict(observed_samples=1,active_samples=0,inactive_samples=1,unknown_samples=0)
        assert data['batches'][0]['sources'][0]['observed_proprio']['valid_dimensions']==0
        if kind=='inactive_no_update':assert item['changed_elements']==0
    else:assert any('proprio' in failure for failure in data['checks']['failures'])


def test_inactive_proprio_still_requires_finite_parameters(tmp_path):
    h=harness();module,arch=trainer_module(tmp_path)
    def run():
        run_update(module,tmp_path,batch=[{'proprio':torch.ones(1,2),'proprio_mask':torch.zeros(1,2,dtype=torch.bool)}])
        with torch.no_grad():arch.proprio_encoder.weight.fill_(float('inf'))
    with pytest.raises(h.VerificationError):h.run_verification(tmp_path/'infinite.json',module,run)
    assert not json.loads((tmp_path/'infinite.json').read_text())['checks']['passed']



def test_assets_resolver_does_not_fall_back_to_legacy_v3_summary(tmp_path):
    h=harness()
    (tmp_path/'preflight_sample_v3').mkdir()
    (tmp_path/'stage1_asset_contract.json').write_text(json.dumps({'checkpoint':'/goal','training':{'baseline_checkpoint':'/baseline'}}))
    (tmp_path/'preflight_sample_v3/summary.json').write_text(json.dumps({'source_yaml':'/old-yaml','normalization_registry':'/old-registry'}))
    with pytest.raises(FileNotFoundError,match='preflight_sample_v4'):
        h.resolve_smoke_plan(tmp_path,tmp_path/'output')



def test_cpu_probe_skips_cuda_resources_after_another_suite_initializes_cuda(tmp_path,monkeypatch):
    h=harness();module,_=trainer_module(tmp_path)
    monkeypatch.setattr(torch.cuda,'is_initialized',lambda:True)
    def forbidden(*args,**kwargs):raise AssertionError('CPU trainer queried CUDA memory/current device')
    for name in ('current_device','max_memory_allocated','max_memory_reserved'):
        monkeypatch.setattr(torch.cuda,name,forbidden)
    probe=h.Probe(tmp_path/'cpu_after_cuda.json');probe.trainer=module.OpenWAMTrainer()
    probe.write()
    data=json.loads((tmp_path/'cpu_after_cuda.json').read_text())
    assert data['resources']['host_peak_rss_bytes']>0
    assert not any(key.startswith('cuda_') for key in data['resources'])


@pytest.mark.parametrize('device,initialized,expect_cuda',[
    ('cpu',True,False),('cuda:2',False,False),('cuda:2',True,True),
    (None,False,False),(None,True,True),
])
def test_resource_probe_uses_trainer_device_and_safe_pretrainer_fallback(tmp_path,monkeypatch,device,initialized,expect_cuda):
    h=harness();probe=h.Probe(tmp_path/'resources.json')
    if device is not None:probe.trainer=SimpleNamespace(accelerator=SimpleNamespace(device=torch.device(device)))
    monkeypatch.setattr(torch.cuda,'is_initialized',lambda:initialized)
    expected=torch.device(device) if device=='cuda:2' else 3
    def current():
        assert device is None and initialized
        return 3
    def allocated(selected):
        assert expect_cuda and selected==expected
        return 100
    def reserved(selected):
        assert expect_cuda and selected==expected
        return 200
    monkeypatch.setattr(torch.cuda,'current_device',current)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',allocated)
    monkeypatch.setattr(torch.cuda,'max_memory_reserved',reserved)
    resources=probe.resources()
    assert ('cuda_peak_allocated_bytes' in resources)==expect_cuda
    if expect_cuda:
        assert resources['cuda_peak_allocated_bytes']==100 and resources['cuda_peak_reserved_bytes']==200
