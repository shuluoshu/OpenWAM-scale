"""CPU report regressions; generation is replaced by a tiny deterministic model."""
import copy
import hashlib
import json
import random
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


def sample():
    mask=torch.zeros(32,112,dtype=torch.bool);mask[:2,80]=True
    action=torch.zeros(32,112);action[~mask]=999
    return dict(prompt='move cup',granularity='task',goal_seed=17,goal_head_valid=True,
        goal_observed_head=torch.zeros(3,1,256,320),first_frame_image=[Image.new('RGB',(320,384),20)],
        proprio=torch.zeros(1,112),proprio_mask=torch.ones(1,112,dtype=torch.bool),action=action,action_mask=mask,
        video=[Image.new('RGB',(320,384),20) for _ in range(9)],
        video_mask=torch.tensor([True,True]+[False]*7),
        video_view_mask=torch.tensor([[True,False,False],[True,False,False]]+[[False]*3]*7),
        report_target={'goal_source_frame':40,'success_verified':False},
        trace={'identity':['repo',0,1,'task'],'raw_action':np.zeros((2,52)), 'raw_action_mask':np.ones((2,52),bool)},
        _eval=dict(family='unit',active_action_mask=np.ones(112,bool),
            report_goal=Image.new('RGB',(320,256),127),projection=None,
            action_stats={'q01':[-1.]*52,'q99':[1.]*52,'count':[1]*52}))


class Tiny(torch.nn.Module):
    def __init__(self, fail=False):
        super().__init__();self.child=torch.nn.Linear(1,1);self.child.eval();self.fail=fail;self.calls=[]
        self.video_scheduler=SimpleNamespace(timesteps=torch.tensor([99.]))
        self.action_scheduler=SimpleNamespace(timesteps=torch.tensor([77.]))
        self.video_backbone=SimpleNamespace(shift_video=5.)
        self.action_backbone=SimpleNamespace(shift_action=5.)
        self.goal_expert=SimpleNamespace(codec=SimpleNamespace(decode=lambda x:torch.zeros(1,3,1,256,320)))
    def generate(self,schedule,prompt,**kw):
        random.random();np.random.rand();torch.rand(1);self.eval();self.calls.append((prompt,kw))
        if self.fail:raise RuntimeError('generation failed')
        self.goal_diagnostics=dict(predicted_latent=torch.zeros(1,48,1,16,20),predicted_uv=torch.full((1,2,2),.5),image_token_valid=torch.ones(1,80,dtype=torch.bool))
        return dict(normalized_actions=np.zeros((32,112),np.float32),video=[Image.new('RGB',(320,384),20) for _ in range(9)])


def schedule(*args,**kwargs):
    args[1].timesteps=torch.tensor([1.]);args[2].timesteps=torch.tensor([2.]);return [(1.,1.),(0.,0.)]


def test_history_three_steps_is_immutable_and_valid_only(tmp_path,monkeypatch):
    from openwam.train import stage2_eval as e
    monkeypatch.setattr(e,'make_schedule',schedule)
    model=Tiny();data=sample()
    for step in [0,500]:e.evaluate_stage2(model,[data],tmp_path,step=step,seed=10,num_steps=1)
    old=tmp_path/'eval_step_000500/case_000/goal_prediction.png';before=hashlib.sha256(old.read_bytes()).hexdigest()
    e.evaluate_stage2(model,[data],tmp_path,step=1000,seed=10,num_steps=1)
    assert hashlib.sha256(old.read_bytes()).hexdigest()==before
    metrics=json.loads((tmp_path/'eval_step_001000/metrics.json').read_text())
    assert metrics['cases'][0]['metrics']['action_mae']['valid_count']==2
    assert metrics['cases'][0]['metrics']['action_mae']['value']==0
    assert metrics['cases'][0]['metrics']['future_video_mae']['valid_count']==256*320*3
    assert metrics['cases'][0]['metrics']['tcp_pixel_error']['value'] is None
    html=(tmp_path/'stage2_history.html').read_text()
    for step in [0,500,1000]:assert f'eval_step_{step:06d}/index.html' in html
    for p in tmp_path.rglob('manifest.json'):
        m=json.loads(p.read_text())
        if 'assets' in m:
            for name,sha in m['assets'].items():assert hashlib.sha256((p.parent/name).read_bytes()).hexdigest()==sha
    with pytest.raises(FileExistsError):e.evaluate_stage2(model,[data],tmp_path,step=500,seed=10,num_steps=1)


def test_future_labels_never_change_conditions_or_seed():
    from openwam.train.stage2_eval import generation_conditions
    a=sample();b=copy.deepcopy(a)
    b['action_mask'].zero_();b['action'].fill_(25);b['video_mask'].zero_();b['video_view_mask'].zero_()
    b['report_target']['goal_source_frame']=999;b['video']=[Image.new('RGB',(320,384),222)]*9
    x=generation_conditions(a,seed=44,num_steps=3);y=generation_conditions(b,seed=44,num_steps=3)
    assert x.keys()==y.keys()
    for k in x:
        if torch.is_tensor(x[k]) or isinstance(x[k],np.ndarray):np.testing.assert_array_equal(x[k],y[k])
        elif k=='first_frame_image':assert x[k][0].tobytes()==y[k][0].tobytes()
        else:assert x[k]==y[k]


@pytest.mark.parametrize('fail',[False,True])
def test_rng_modes_scheduler_restored_even_failure(tmp_path,monkeypatch,fail):
    from openwam.train import stage2_eval as e
    monkeypatch.setattr(e,'make_schedule',schedule)
    model=Tiny(fail);py=random.getstate();n=np.random.get_state();t=torch.random.get_rng_state().clone()
    if fail:
        with pytest.raises(RuntimeError,match='generation failed'):e.evaluate_stage2(model,[sample()],tmp_path,step=0,seed=4,num_steps=1)
        assert not (tmp_path/'eval_step_000000/manifest.json').exists()
    else:e.evaluate_stage2(model,[sample()],tmp_path,step=0,seed=4,num_steps=1)
    assert random.getstate()==py
    assert np.array_equal(np.random.get_state()[1],n[1]);assert torch.equal(torch.random.get_rng_state(),t)
    assert model.training and not model.child.training
    assert model.video_scheduler.timesteps.tolist()==[99.] and model.action_scheduler.timesteps.tolist()==[77.]


def test_physical_fk_metrics_use_counts_and_raw_targets():
    from openwam.train.stage2_eval import action_metrics
    a=sample();a['action_mask'].zero_();a['action_mask'][:2,[80,9]]=True
    a['trace']['raw_action'][:,20]=2.;a['trace']['raw_action'][:,9]=.7
    result=action_metrics(np.zeros((32,112)),a)
    assert result['joint_radian_mae']['value']==2.
    assert result['gripper_openness_mae']['value']==pytest.approx(.7)
    assert result['fk_position_m']['value'] is None


def test_unavailable_geometry_is_json_null(tmp_path,monkeypatch):
    from openwam.train import stage2_eval as e
    monkeypatch.setattr(e,'make_schedule',schedule)
    a=sample();a['report_target']['endpoint']={'tcp_world':np.full((2,3),np.nan)}
    e.evaluate_stage2(Tiny(),[a],tmp_path,step=0,seed=3,num_steps=1)
    text=(tmp_path/'eval_step_000000/metrics.json').read_text()
    assert 'NaN' not in text and 'null' in text


def test_shared_geometry_projection_uses_pixel_centers_once(monkeypatch):
    from openwam.train.stage2_eval import prediction_metrics
    a=sample();a['_eval']['projection']=dict(uv=np.array([[.25,.75],[0,0]]),projection_valid=[True,False],in_frame=[True,False])
    d=dict(predicted_uv=torch.tensor([[[.25,.75],[9.,9.]]]),image_token_valid=torch.ones(1,80,dtype=torch.bool))
    r=dict(normalized_actions=np.zeros((32,112)),video=a['video'])
    m=prediction_metrics(r,d,Image.new('RGB',(320,256)),a)['tcp_pixel_error']
    assert m['value']==0 and m['valid_count']==1


def test_rank_zero_render_failure_reaches_other_ranks(monkeypatch):
    from openwam.train import stage2_eval as e
    monkeypatch.setattr(e,'_distributed',lambda:True)
    monkeypatch.setattr(torch.distributed,'get_world_size',lambda:2)
    def gather(output,value):output[:]=['RuntimeError: failed to render',None]
    monkeypatch.setattr(torch.distributed,'all_gather_object',gather)
    with pytest.raises(RuntimeError,match='failed to render'):e._agree_error(None,'publication')


def test_native_fixedcases_freeze_scalars_disable_augmentation_and_cap(tmp_path,monkeypatch):
    import importlib.util
    spec=importlib.util.spec_from_file_location('native_fixture',Path(__file__).parents[1]/'dataloader/test_curated_stage2.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    from openwam.dataloader.curated_stage2 import CuratedStage2Dataset
    from openwam.dataloader.native_stage2_stats import prepare_native_stats
    from openwam.train.stage2_eval import build_fixed_cases
    selection=module.make_native_source(tmp_path)
    dataset=CuratedStage2Dataset(selection)
    registry=prepare_native_stats(dataset,tmp_path/'stats',max_episodes_per_profile=1,max_windows_per_profile=8)
    config=dict(dataloader=dict(source_yaml=str(selection),normalization_registry=str(registry),seed=4,
        observation_augmentation={'color_jitter':{'p':1,'brightness':.4}}),project={'seed':4},evaluation={'max_cases':1})
    with pytest.raises(ValueError,match='cannot cover'):build_fixed_cases(config,tmp_path/'report')
    config['evaluation']['diagnostic_subset']=True
    provider=build_fixed_cases(config,tmp_path/'report')
    assert len(provider)==1 and provider.coverage['mode']=='diagnostic reduced coverage'
    provider.publish_descriptors()
    again=build_fixed_cases(config,tmp_path/'report')
    assert again.descriptors==provider.descriptors
    assert not any(torch.is_tensor(v) or isinstance(v,np.ndarray) for d in again.descriptors for v in d.values())
    monkeypatch.setattr(provider.dataset,'_decode_views',lambda m,indices:([{'observation.images.cam_high':Image.new('RGB',(40,32),100)} for _ in indices],{}))
    import openwam.dataloader.stage2_diagnostics as diagnostic
    monkeypatch.setattr(diagnostic,'report_goal_target',lambda d,r,e,m,s:(None,{**s['report_target'],'diagnostic_unavailable':'fixture has no camera'}))
    a=provider[0]
    assert a['report_target']['goal_source_frame']==23, 'task report uses true episode end, beyond selected interval'
    assert not provider.dataset.augmentation.enabled
    assert '_observation_augmentation' not in a
    active=a['_eval']['active_action_mask'].copy();a['action_mask'].zero_()
    from openwam.train.stage2_eval import generation_conditions
    np.testing.assert_array_equal(generation_conditions(a,seed=4,num_steps=1)['active_action_mask'],active)


def test_completed_step_reuse_checks_contract_and_assets(tmp_path,monkeypatch):
    from openwam.train import stage2_eval as e
    monkeypatch.setattr(e,'make_schedule',schedule)
    data=[sample()]
    assert not e.completed_evaluation(tmp_path,data,step=0,seed=4,num_steps=1)
    e.evaluate_stage2(Tiny(),data,tmp_path,step=0,seed=4,num_steps=1)
    assert e.completed_evaluation(tmp_path,data,step=0,seed=4,num_steps=1)
    with pytest.raises(ValueError,match='mismatch'):e.completed_evaluation(tmp_path,data,step=0,seed=8,num_steps=1)
    (tmp_path/'fixed_inputs/case_000/observation.png').write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='immutable evaluation input'):e.completed_evaluation(tmp_path,data,step=0,seed=4,num_steps=1)


def test_trainer_hook_generates_and_resume_dedupes(tmp_path,monkeypatch):
    from openwam.train import stage2_eval as e
    from openwam.train.openwam_trainer import OpenWAMTrainer
    from omegaconf import OmegaConf
    monkeypatch.setattr(e,'make_schedule',schedule)
    monkeypatch.setattr(e,'build_fixed_cases',lambda config,run:[sample()])
    trainer=object.__new__(OpenWAMTrainer);trainer.cfg=OmegaConf.create({'project':{'seed':4},'evaluation':{'num_steps':1}})
    trainer.architecture=Tiny();trainer.accelerator=SimpleNamespace(unwrap_model=lambda m:m)
    trainer._evaluate_stage2(str(tmp_path),0);trainer._evaluate_stage2(str(tmp_path),0,final=True)
    assert len(trainer.architecture.calls)==1
    assert (tmp_path/'eval_step_000000/manifest.json').is_file()


def test_cli_exact_steps_and_history_only(tmp_path,monkeypatch,capsys):
    from scripts import eval_stage2 as cli
    from openwam.train import stage2_eval as e
    from openwam.train.utils import ckpt_model_loader,stage2_assets
    from omegaconf import OmegaConf
    run=tmp_path/'run';run.mkdir()
    OmegaConf.save(OmegaConf.create(dict(goal={'enabled':True},project={'seed':9},training={'mixed_precision':'no'},evaluation={'num_steps':1})),run/'config.yaml')
    for step in [0,500,1000]:(run/f'checkpoint_step_{step}.safetensors').write_bytes(b'fixture')
    model=Tiny();model.set_dtype_device=lambda dtype,device:None;model.move_frozen_to_device=lambda device:None
    model.load_checkpoint=lambda path:None
    monkeypatch.setattr(ckpt_model_loader,'build_architecture_from_ckpt_dir',lambda path,weights_required:(None,model,None))
    monkeypatch.setattr(stage2_assets,'resolve_resume_normalization',lambda config,run:None)
    monkeypatch.setattr(e,'build_fixed_cases',lambda config,run:[sample()]);monkeypatch.setattr(e,'make_schedule',schedule)
    cli.main(['--run-dir',str(run),'--device','cpu','--steps','0','500','1000'])
    assert len(model.calls)==3
    cli.main(['--run-dir',str(run),'--history-only'])
    assert (run/'stage2_history.html').exists()
    assert 'stage2_history.html' in capsys.readouterr().out


def test_prediction_shape_failure_never_publishes(tmp_path,monkeypatch):
    from openwam.train import stage2_eval as e
    monkeypatch.setattr(e,'make_schedule',schedule)
    model=Tiny();generate=model.generate
    def malformed(*args,**kwargs):
        output=generate(*args,**kwargs);output['video'][-1]=Image.new('RGB',(64,64));return output
    model.generate=malformed
    with pytest.raises(RuntimeError,match='9 PIL'):e.evaluate_stage2(model,[sample()],tmp_path,step=0,seed=4,num_steps=1)
    assert not (tmp_path/'eval_step_000000').exists()
    assert model.training


def test_resume_registry_relocation_preserves_fixed_assets(tmp_path,monkeypatch):
    from openwam.train import stage2_eval as e
    monkeypatch.setattr(e,'make_schedule',schedule)
    a=sample();a['trace']['stats_registry']='/old/run/registry.json'
    e.evaluate_stage2(Tiny(),[a],tmp_path,step=0,seed=4,num_steps=1)
    b=copy.deepcopy(a);b['trace']['stats_registry']='/relocated/run/registry.json'
    e.evaluate_stage2(Tiny(),[b],tmp_path,step=500,seed=4,num_steps=1)
    assert (tmp_path/'eval_step_000500/manifest.json').is_file()


def test_fk_metric_is_euclidean_meters_valid_arm_endpoints_only():
    from openwam.train.stage2_eval import action_metrics
    a=sample();a['action_mask'].zero_();a['action_mask'][0,80]=True
    identity=np.eye(4);chain=dict(dof=1,joints=[dict(kind='prismatic',fixed=identity,index=0,axis=[1,0,0])],tool=identity)
    arm=dict(chain=chain,slots=[80],q_anchor=[0.],scale=[2.],offset=[0.],
        valid=[True,True]+[False]*30,gt_relative=np.tile(identity,(32,1,1)))
    a['fk_supervision']={'arms':[arm]}
    pred=np.zeros((32,112));pred[:,80]=.25
    m=action_metrics(pred,a)
    assert m['fk_position_m']['value']==.5 and m['fk_position_m']['valid_count']==1
    assert m['fk_rotation_rad']['value']==0


def test_history_rejects_missing_fixed_asset(tmp_path,monkeypatch):
    from openwam.train import stage2_eval as e
    monkeypatch.setattr(e,'make_schedule',schedule)
    e.evaluate_stage2(Tiny(),[sample()],tmp_path,step=0,seed=4,num_steps=1)
    (tmp_path/'fixed_inputs/case_000/goal_input.png').unlink()
    with pytest.raises(ValueError,match='fixed'):e.render_stage2_history(tmp_path)


@pytest.mark.parametrize('partial', [False, True])
def test_observed_proprio_mask_crosses_actual_generate_conditioning_boundary(partial, monkeypatch):
    from tests.goal.test_patch_stage2_memory import make_arch
    from openwam.train.stage2_eval import generation_conditions, _conditions_sha
    arch=make_arch()
    arch._use_proprioception_context=True;arch.proprio_dim=112
    arch.proprio_encoder=torch.nn.Linear(112,16)
    torch.nn.init.zeros_(arch.proprio_encoder.weight);torch.nn.init.ones_(arch.proprio_encoder.bias)
    arch.video_backbone._scheduler=SimpleNamespace(num_train_timesteps=1000);arch.action_backbone.scheduler=SimpleNamespace(num_train_timesteps=1000)
    monkeypatch.setattr(arch.video_backbone,'preprocess_input_for_inference',lambda **kw:
        dict(context=torch.zeros(1,2,16),latents=torch.zeros(1,48,1,2,2)))
    class ReachedConditioning(Exception):pass
    seen=[]
    def forward(actions, action_timestep, **pipeline):
        result=arch._append_proprio_context_token(pipeline,pipeline['proprio'])
        seen.append(result)
        raise ReachedConditioning
    monkeypatch.setattr(arch,'forward',forward)
    a=sample();a['proprio_mask'].zero_()
    if partial:a['proprio_mask'][0,80]=True
    kwargs=generation_conditions(a,seed=4,num_steps=1)
    assert torch.equal(kwargs['_proprio_sample_mask'],a['proprio_mask'][None])
    with pytest.raises(ReachedConditioning):arch.generate([(1000.,1000.),(0.,0.)],a['prompt'],**kwargs)
    assert bool(seen[0]['context_mask'][0,-1])==partial
    torch.testing.assert_close(seen[0]['context'][0,-1],torch.full((16,),float(partial)))
    changed=copy.deepcopy(a);changed['proprio_mask'].logical_not_()
    assert _conditions_sha(a,kwargs)!=_conditions_sha(changed,generation_conditions(changed,seed=4,num_steps=1))
    changed=copy.deepcopy(a);changed['action_mask'].zero_();changed['video_mask'].zero_();changed['video_view_mask'].zero_()
    assert _conditions_sha(a,kwargs)==_conditions_sha(changed,generation_conditions(changed,seed=4,num_steps=1))


def test_proprio_mask_is_validated_and_immutable_snapshot_input(tmp_path):
    from openwam.train.stage2_eval import _validate_sample, _fixed_assets
    data=sample();_validate_sample(data);_fixed_assets(data,tmp_path/'fixed')
    with np.load(tmp_path/'fixed/supervision.npz') as snapshot:
        np.testing.assert_array_equal(snapshot['proprio_mask'],data['proprio_mask'])
    changed=copy.deepcopy(data);changed['proprio_mask'][0,0]=False
    with pytest.raises(ValueError,match='assets changed'):_fixed_assets(changed,tmp_path/'fixed')
    changed=copy.deepcopy(data);changed['proprio'][0,0]=1
    with pytest.raises(ValueError,match='assets changed'):_fixed_assets(changed,tmp_path/'fixed')
    data['proprio_mask']=torch.ones(112,dtype=torch.bool)
    with pytest.raises(ValueError,match='proprio_mask shape'):_validate_sample(data)
