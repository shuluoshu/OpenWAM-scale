from pathlib import Path
import json
import os
import subprocess
import sys
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from openwam.train.utils.stage2 import build_stage2_parameter_groups, validate_stage2_config, validate_stage2_checkpoint

ROOT=Path(__file__).resolve().parents[2]


def config(preset='sample'):
    with initialize_config_dir(config_dir=str(ROOT/'configs'),version_base=None):
        return compose(config_name='goal/stage2_joint_'+preset)


@pytest.mark.parametrize('preset',['sample','3000h'])
def test_hydra_presets_keep_training_contract_without_gpu(preset,monkeypatch):
    monkeypatch.setattr(torch.cuda,'_lazy_init',lambda:pytest.fail('GPU initialized'))
    cfg=config(preset)
    validate_stage2_config(cfg)
    assert 'base_config' not in cfg
    assert cfg.goal.enabled and cfg.goal.conditioning=='image_memory' and cfg.goal.input_policy=='clean_head'
    assert cfg.goal.num_steps==20 and cfg.goal.cache_max_bytes<=268435456
    assert cfg.training.step_unit=='optimizer_updates' and cfg.training.batch_size==1
    assert cfg.training.offload_optimizer_device=='cpu' and cfg.training.initialize_model_on_cpu
    assert cfg.dataloader.max_cache_bytes<=268435456 and cfg.training.dataset_num_workers==0
    assert cfg.training.lambda_fk_position==cfg.training.lambda_fk_rotation==1
    assert cfg.dataloader.observation_augmentation.patch_mask.area_range==[.05,.2]
    assert cfg.model.architecture.action_dim==cfg.model.architecture.state_dim==112
    assert cfg.evaluation.every_updates==500


def test_groups_are_actual_exhaustive_trainable_ownership_and_config_rates():
    class Architecture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.video_backbone=torch.nn.Linear(2,2)
            self.proprio_encoder=torch.nn.Linear(2,2)
            self.action_backbone=torch.nn.Module()
            self.action_backbone.body=torch.nn.Linear(2,2)
            self.action_backbone.action_encoder=torch.nn.Linear(2,2)
            self.action_backbone.action_decoder=torch.nn.Linear(2,2)
            self.goal_adapter=torch.nn.Linear(2,2)
            self.goal_expert=torch.nn.Linear(2,2).requires_grad_(False)
    model=Architecture();cfg=config()
    groups=build_stage2_parameter_groups(model,cfg.training)
    assert {g['name']:g['lr'] for g in groups}==dict(cfg.training.parameter_lrs)
    params=[id(p) for g in groups for p in g['params']]
    assert len(params)==len(set(params))
    assert set(params)=={id(p) for p in model.parameters() if p.requires_grad}
    cfg.training.parameter_lrs.video=.5
    with pytest.raises(ValueError,match='learning rate'):build_stage2_parameter_groups(model,cfg.training)


def test_structural_checkpoint_contract_checks_projection_width_without_path_lock(tmp_path):
    from safetensors.torch import save_file
    cfg=config();path=tmp_path/'weights.safetensors'
    values={'action_backbone.action_encoder.weight':torch.zeros(8,112),
            'action_backbone.action_decoder.weight':torch.zeros(112,8),
            'action_backbone.action_decoder.bias':torch.zeros(112),
            'proprio_encoder.weight':torch.zeros(8,112)}
    save_file(values,str(path))
    validate_stage2_checkpoint(cfg,path)
    values['proprio_encoder.weight']=torch.zeros(8,80)
    save_file(values,str(path))
    with pytest.raises(ValueError,match='projection'):validate_stage2_checkpoint(cfg,path)


def test_launcher_rejects_missing_paths_before_torchrun(tmp_path):
    env={k:v for k,v in os.environ.items() if not k.startswith('STAGE')}
    result=subprocess.run(['bash',str(ROOT/'scripts/launch_stage2.sh'),'sample','--preflight'],env=env,text=True,capture_output=True)
    assert result.returncode!=0 and 'STAGE1_GOAL_CHECKPOINT' in result.stderr


def test_stage2_loop_uses_successful_update_horizon_and_committed_metadata(tmp_path,monkeypatch):
    import contextlib
    from types import SimpleNamespace
    import openwam.train.openwam_trainer as module
    trainer=module.OpenWAMTrainer.__new__(module.OpenWAMTrainer)
    cfg=config();cfg.training.max_steps=2;cfg.training.gradient_accumulation_steps=2
    cfg.training.save_steps=2
    trainer.cfg=cfg;trainer._run_seed=None;trainer._rank=0
    trainer.architecture=torch.nn.Linear(1,1)
    trainer.architecture.goal_asset_identity="tiny"
    trainer.architecture.goal_config={"num_steps":20}
    trainer.dataset=SimpleNamespace(seed=42,normalization_identity=lambda:dict(selection_sha256='selection',registry_sha256='registry'))
    class Accelerator:
        num_processes=2;is_main_process=True;sync_gradients=False;optimizer_step_was_skipped=False
        micro=0
        @contextlib.contextmanager
        def accumulate(self,architecture):
            self.micro+=1;self.sync_gradients=self.micro%2==0
            self.optimizer_step_was_skipped=self.micro==2
            yield
        def gather(self,value):return value
        def backward(self,loss):loss.backward()
        def clip_grad_norm_(self,params,norm):return 0.
        def wait_for_everyone(self):pass
    trainer.accelerator=Accelerator()
    from openwam.train.utils.stage2 import GlobalDrawSampler
    def loader(batch_size):
        trainer._stage2_sampler=GlobalDrawSampler(8,seed=42,selection_identity='selection')
        return [[dict(x=1)] for i in range(4)]
    trainer.build_dataloader=loader
    trainer.build_optimizer=lambda:torch.optim.SGD(trainer.architecture.parameters(),lr=.01)
    horizons=[]
    trainer.build_lr_scheduler=lambda optimizer,total_opt_steps,debug=False:horizons.append(total_opt_steps)
    trainer.setup_output_dir=lambda *args:(str(tmp_path),None)
    trainer.prepare_accelerate=lambda optimizer,dataloader,scheduler:(optimizer,dataloader,scheduler)
    trainer.compute_loss=lambda batch:dict(total=trainer.architecture.weight.square().sum(),video=0.,action=0.)
    trainer.log_step=lambda **kwargs:None
    evaluations=[]
    trainer._evaluate_stage2=lambda path,step,**kwargs:evaluations.append(step)
    saved=[]
    monkeypatch.setattr(module,'save_weights',lambda *args,**kwargs:None)
    monkeypatch.setattr(module,'save_full_state',lambda *args,**kwargs:saved.append((args[3],kwargs['trainer_metadata'])))
    monkeypatch.setattr(module,'manage_checkpoints',lambda *args:None)
    monkeypatch.setattr(module,'init_wandb',lambda cfg:None)
    trainer.train_stage2()
    assert horizons==[2]
    assert trainer._stage2_progress.opt_step==2 and trainer._stage2_progress.micro_step==6
    assert trainer._stage2_progress.consumed_draws==12
    assert evaluations==[0,2]
    assert saved[-1][1]['stage2']['progress']['opt_step']==2
    assert saved[-1][1]['stage2']['sampler']['offset']==4


def test_cpu_preflight_exits_before_accelerator_and_requires_real_stats_for_training(tmp_path,monkeypatch):
    from safetensors.torch import save_file
    from tests.dataloader.test_curated_stage2 import make_native_source
    import scripts.train as entry
    cfg=config();cfg.goal.preflight=True
    selection=make_native_source(tmp_path)
    goal=tmp_path/'goal.safetensors'
    save_file({'dit.dummy':torch.zeros(1)},str(goal),metadata={'patch_goal_bundle':json.dumps(dict(
        format='openwam.goal.patch.v1',codec={'latent_shape':[48,1,16,20]},dependencies={'test':'identity'}))})
    baseline=tmp_path/'baseline';baseline.mkdir()
    OmegaConf.save({'model':cfg.model},baseline/'config.yaml')
    save_file({'action_backbone.action_encoder.weight':torch.zeros(8,112),
        'action_backbone.action_decoder.weight':torch.zeros(112,8),
        'action_backbone.action_decoder.bias':torch.zeros(112),
        'proprio_encoder.weight':torch.zeros(8,112)},str(baseline/'checkpoint_step_1.safetensors'))
    cfg.goal.checkpoint=str(goal);cfg.training.finetune_ckpt_path=str(baseline)
    cfg.dataloader.source_yaml=str(selection);cfg.project.output_dir=str(tmp_path/'run')
    monkeypatch.setattr(entry,'_build_accelerator',lambda cfg:pytest.fail('Accelerator constructed'))
    monkeypatch.setattr(torch.cuda,'_lazy_init',lambda:pytest.fail('GPU initialized'))
    entry._train(cfg)
    cfg.goal.preflight=False
    with pytest.raises(ValueError,match='normalization_registry required'):entry._train(cfg)


def test_full_state_marker_records_committed_stage2_metadata(tmp_path):
    from openwam.train.utils.checkpointing import save_full_state
    from types import SimpleNamespace
    accelerator=SimpleNamespace(is_main_process=True,process_index=0,wait_for_everyone=lambda:None,save_state=lambda path:None)
    (tmp_path/"checkpoint_step_2.safetensors").write_bytes(b"tiny marker fixture")
    state={'stage2':{'progress':{'micro_step':8,'opt_step':2},'sampler':{'offset':16}}}
    save_full_state(accelerator,str(tmp_path),2,2,0,trainer_metadata=state)
    saved=json.loads((tmp_path/'accel_state_step_2/trainer_state.json').read_text())
    assert saved['stage2']==state['stage2'] and saved['opt_step']==2


def test_stage2_resume_rejects_missing_state_and_changed_normalization():
    from types import SimpleNamespace
    from openwam.train.openwam_trainer import OpenWAMTrainer
    from openwam.train.utils.stage2 import Stage2Progress,GlobalDrawSampler
    trainer=OpenWAMTrainer.__new__(OpenWAMTrainer)
    trainer.dataset=SimpleNamespace(normalization_identity=lambda:dict(selection_sha256='selection',registry_sha256='registry'))
    trainer.cfg=config()
    trainer.architecture=SimpleNamespace(goal_asset_identity='tiny',goal_config={'num_steps':20})
    trainer._stage2_progress=Stage2Progress(1,2,2)
    trainer._stage2_sampler=GlobalDrawSampler(8,seed=42,selection_identity='selection')
    with pytest.raises(ValueError,match='sampler/counter'):trainer.restore_stage2_sampling_state({'global_step':8})
    state=trainer.stage2_sampling_state()
    trainer.restore_stage2_sampling_state({'stage2':state})
    state['normalization']['registry_sha256']='wrong'
    with pytest.raises(ValueError,match='normalization identity'):trainer.restore_stage2_sampling_state({'stage2':state})



@pytest.mark.parametrize('mode,expected_calls',[('--preflight',1),('--train',2)])
def test_launcher_uses_configured_python_for_preflight_and_distributed_training(tmp_path,mode,expected_calls):
    chosen=tmp_path/'chosen python';chosen.write_text('#!'+sys.executable+'\n'+
        'import json,os,sys\n'+
        'with open(os.environ["LAUNCH_CAPTURE"],"a") as stream:stream.write(json.dumps(sys.argv[1:])+"\\n")\n')
    chosen.chmod(0o755)
    trap=tmp_path/'torchrun';trap.write_text('#!/bin/sh\nexit 97\n');trap.chmod(0o755)
    capture=tmp_path/'calls.jsonl'
    env={**os.environ,'PYTHON':str(chosen),'LAUNCH_CAPTURE':str(capture),
        'PATH':str(tmp_path)+os.pathsep+os.environ['PATH'],
        'STAGE1_GOAL_CHECKPOINT':'/fixture/goal','STAGE2_BASELINE_CHECKPOINT':'/fixture/base',
        'STAGE2_OUTPUT_DIR':'/fixture/output','STAGE2_SOURCE_YAML':'/fixture/source',
        'STAGE2_STATS_REGISTRY':'/fixture/registry','NPROC_PER_NODE':'2'}
    completed=subprocess.run(['bash',str(ROOT/'scripts/launch_stage2.sh'),'sample',mode,'training.max_steps=3'],
        env=env,text=True,capture_output=True)
    assert completed.returncode==0,completed.stderr
    calls=[json.loads(line) for line in capture.read_text().splitlines()]
    assert len(calls)==expected_calls and calls[0][0]=='scripts/train.py'
    assert calls[0][-1]=='goal.preflight=true'
    for args in calls:assert 'training.max_steps=3' in args and 'goal.checkpoint=/fixture/goal' in args
    if mode=='--train':
        assert calls[1][:5]==['-m','torch.distributed.run','--standalone','--nproc_per_node=2','scripts/train.py']
        assert calls[1][-1]=='goal.preflight=false'
