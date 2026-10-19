"""Multiview Stage2 uses clean actual observations through every public route."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from openwam.model.goal import runtime
from openwam.train.goal_patch import make_smoke_data, save_patch_bundle
from tests.goal.test_patch_stage2_memory import make_arch
from tests.goal.test_stage2_config import config


def configured(tmp_path):
    torch.set_num_threads(1)
    arch=make_arch();teacher,batches,_=make_smoke_data(observation_mode='multiview')
    teacher.codec.vae=arch.video_backbone.vae
    asset=tmp_path/'v2.safetensors';save_patch_bundle(teacher,asset,precomputed_text_fingerprint='a'*64)
    cfg=OmegaConf.create({'goal':dict(enabled=True,checkpoint=str(asset),format='openwam.goal.patch.v2',
        input_policy='clean_multiview',observation_mode='multiview',precomputed_text_fingerprint='a'*64,num_steps=1)})
    runtime.configure_goal(arch,cfg)
    return arch,cfg,batches[0]


def condition(arch,batch,mask=None):
    return runtime.prepare_goal_inputs(arch,None,['move'],batch['context'],head_valid=torch.tensor([True]),
        seeds=[17],goal_observed_multiview=batch['observed'],
        goal_view_valid=batch['observed_view_valid'] if mask is None else mask)


def test_v2_actual_teacher_cache_mask_identity_and_missing_pixels(tmp_path):
    arch,cfg,b=configured(tmp_path)
    first=condition(arch,b);assert first['goal_token_valid'].shape==(1,80)
    assert len(arch.goal_cache.entries)==1
    changed=copy.copy(b);changed['observed']=b['observed'].clone()
    changed['observed'][:,:,:,256:,160:]=float('nan')
    same=condition(arch,changed)
    torch.testing.assert_close(first['goal_latent'],same['goal_latent'],rtol=0,atol=0)
    assert len(arch.goal_cache.entries)==1
    condition(arch,b,torch.tensor([[True,False,False]]))
    assert len(arch.goal_cache.entries)==2
    assert not arch.goal_expert.training and all(p.grad is None for p in arch.goal_expert.parameters())
    for mask in [None,torch.ones(1,3),torch.tensor([[False,True,False]])]:
        with pytest.raises(ValueError,match='mask|valid|bool|head'):
            runtime.prepare_goal_inputs(arch,None,['move'],b['context'],head_valid=[True],seeds=[17],
                goal_observed_multiview=b['observed'],goal_view_valid=mask)
    empty=runtime.prepare_goal_inputs(arch,None,['move'],b['context'],head_valid=[False],seeds=[17],
        goal_observed_multiview=None,goal_view_valid=torch.zeros(1,3,dtype=torch.bool))
    assert not empty['goal_token_valid'].any()


def test_training_and_public_generate_forward_clean_canvas_and_bool_mask(tmp_path,monkeypatch):
    arch,cfg,b=configured(tmp_path)
    arch._pipeline_transform_instance=SimpleNamespace(apply=lambda sample:{**sample,'video':'augmented'})
    arch.preprocess=lambda **kwargs:{'context':b['context']}
    sample=dict(video='raw',prompt='move',goal_observed_head=b['observed'][0,:,:,:256],
        goal_observed_multiview=b['observed'][0],goal_view_valid=b['observed_view_valid'][0],goal_head_valid=True,goal_seed=17)
    class Reached(Exception):pass
    original=runtime.prepare_goal_inputs
    def capture(architecture,head,prompts,ctx,**kwargs):
        assert torch.equal(kwargs['goal_observed_multiview'],b['observed'])
        assert kwargs['goal_view_valid'].dtype==torch.bool
        got=original(architecture,head,prompts,ctx,**kwargs)
        assert got['goal_token_valid'].all()
        raise Reached
    monkeypatch.setattr(runtime,'prepare_goal_inputs',capture)
    with pytest.raises(Reached):arch.prepare_inputs([sample])
    arch.video_backbone.preprocess_input_for_inference=lambda **kwargs:{'context':b['context']}
    with pytest.raises(Reached):arch.generate(None,'move',goal_observed_multiview=b['observed'],
        goal_view_valid=b['observed_view_valid'],goal_seed=17)


def test_strict_config_actual_mode_and_portable_v2_loaders(tmp_path,monkeypatch):
    arch,cfg,b=configured(tmp_path)
    from openwam.train.utils.stage2 import validate_stage2_config
    from hydra import compose,initialize_config_dir
    root=Path(__file__).resolve().parents[2]
    for preset in ('sample','3000h'):
        with initialize_config_dir(config_dir=str(root/'configs'),version_base=None):
            preset_cfg=compose(config_name='goal/stage2_multiview_'+preset)
        validate_stage2_config(preset_cfg)
        assert preset_cfg.goal.input_policy=='clean_multiview' and preset_cfg.goal.format.endswith('v2')
        preset_cfg.goal.format='openwam.goal.patch.v1'
        with pytest.raises(ValueError,match='goal|contract'):validate_stage2_config(preset_cfg)
    bad=copy.deepcopy(cfg);bad.goal.input_policy='clean_head'
    with pytest.raises(ValueError,match='goal|contract|mode'):runtime.configure_goal(make_arch(),bad)
    bad=copy.deepcopy(cfg);bad.goal.format='openwam.goal.patch.v1';bad.goal.input_policy='clean_head';bad.goal.observation_mode='head'
    with pytest.raises(ValueError,match='goal|contract|mode'):runtime.configure_goal(make_arch(),bad)
    expected=condition(arch,b);out=tmp_path/'run'
    cfg.model={'video_backbone':{'name':'wan22_ti2v_5b'}};cfg.training={'mixed_precision':'no'}
    runtime.save_goal_assets(arch,out,cfg);OmegaConf.save(cfg,out/'config.yaml')
    arch.save_checkpoint(str(out/'checkpoint_step_1.safetensors'))
    import openwam.model as model
    resolved=SimpleNamespace(registry_name='tiny',params={'video_backbone':{}},canonical=SimpleNamespace(framework='dual_system',variant='joint_self_attn'))
    monkeypatch.setattr(model,'resolve_architecture_config',lambda cfg:resolved)
    monkeypatch.setattr(model,'build_architecture',lambda *args:make_arch())
    from openwam.train.utils.ckpt_model_loader import build_architecture_from_ckpt_dir
    from openwam.deploy.model_loader import load_from_checkpoint_dir
    for restored in (build_architecture_from_ckpt_dir(str(out),weights_required=True)[1],load_from_checkpoint_dir(str(out),device='cpu')[1]):
        assert restored.goal_expert.observation_mode=='multiview'
        assert restored.goal_expert.codec.vae is restored.video_backbone.vae
        torch.testing.assert_close(condition(restored,b)['goal_latent'],expected['goal_latent'],rtol=0,atol=0)


def test_fixed_eval_actual_input_mask_digest_and_legacy_unused_invariance(tmp_path,monkeypatch):
    from openwam.train import stage2_eval as e
    from tests.goal.test_stage2_history import sample,Tiny,schedule
    a=sample();a['goal_observed_multiview']=torch.zeros(3,1,384,320);a['goal_view_valid']=torch.tensor([True,True,False])
    head=e.generation_conditions(a,seed=4,num_steps=1)
    kw=e.generation_conditions(a,seed=4,num_steps=1,observation_mode='multiview')
    assert kw['goal_view_valid'].shape==(1,3) and kw['goal_observed_multiview'].shape==(1,3,1,384,320)
    changed=copy.deepcopy(a);changed['goal_view_valid'][1]=False
    assert e._conditions_sha(a,kw)!=e._conditions_sha(changed,e.generation_conditions(changed,seed=4,num_steps=1,observation_mode='multiview'))
    assert e._conditions_sha(a,head)==e._conditions_sha(changed,e.generation_conditions(changed,seed=4,num_steps=1))
    changed=copy.deepcopy(a);changed['goal_observed_multiview'][:,:,256:,:160]=.9
    assert e._conditions_sha(a,kw)!=e._conditions_sha(changed,e.generation_conditions(changed,seed=4,num_steps=1,observation_mode='multiview'))
    monkeypatch.setattr(e,'make_schedule',schedule);model=Tiny();model.goal_expert.observation_mode='multiview'
    e.evaluate_stage2(model,[a],tmp_path,step=0,seed=4,num_steps=1)
    fixed=tmp_path/'fixed_inputs/case_000'
    data=np.load(fixed/'supervision.npz')
    np.testing.assert_array_equal(data['goal_view_valid'],a['goal_view_valid'])
    assert data['goal_observed_multiview'].shape==(3,1,384,320)
    assert json.loads((tmp_path/'eval_step_000000/metrics.json').read_text())['goal_observation_mode']=='multiview'
    assert e.completed_evaluation(tmp_path,[a],step=0,seed=4,num_steps=1,observation_mode='multiview')
    with pytest.raises(ValueError,match='mode|contract|mismatch'):
        e.completed_evaluation(tmp_path,[a],step=0,seed=4,num_steps=1,observation_mode='head')


def test_native_dataset_keeps_clean_preaugmentation_canvas(tmp_path,monkeypatch):
    from tests.dataloader.test_curated_stage2 import make_native_source
    from openwam.dataloader.curated_stage2 import CuratedStage2Dataset
    from openwam.dataloader.native_stage2_stats import prepare_native_stats
    from PIL import Image
    selection=make_native_source(tmp_path);d=CuratedStage2Dataset(selection,seed=4)
    registry=prepare_native_stats(d,tmp_path/'stats',max_episodes_per_profile=1,max_windows_per_profile=4)
    d=CuratedStage2Dataset(selection,seed=4,normalization_registry=registry)
    monkeypatch.setattr(d,'_decode_views',lambda m,indices:([{'observation.images.cam_high':Image.new('RGB',(40,32),(100,120,140))} for _ in indices],{}))
    def augment(sample,missing):
        sample['video']=[Image.new('RGB',(320,384),(255,255,255))]*9
        sample['first_frame_image']=[sample['video'][0]]
        return sample
    d.augmentation.apply=augment
    a=d.sample_at(0)
    assert a['goal_observed_multiview'].shape==(3,1,384,320)
    assert a['goal_view_valid'].dtype==torch.bool and a['goal_view_valid'].tolist()==[True,False,False]
    assert torch.equal(a['goal_observed_multiview'][:,:,:256],a['goal_observed_head'])
    assert a['goal_observed_multiview'][:,:,256:].eq(-1).all()
    assert a['first_frame_image'][0].getpixel((0,0))==(255,255,255)


def test_v2_resume_serializes_actual_policy_and_rejects_change(tmp_path):
    from openwam.train.openwam_trainer import OpenWAMTrainer
    from openwam.train.utils.stage2 import Stage2Progress,GlobalDrawSampler
    arch,cfg,b=configured(tmp_path)
    trainer=OpenWAMTrainer.__new__(OpenWAMTrainer);trainer.architecture=arch;trainer.cfg=config()
    trainer.dataset=SimpleNamespace(normalization_identity=lambda:dict(selection_sha256='selection',registry_sha256='registry'))
    trainer._stage2_progress=Stage2Progress(1,1,1)
    trainer._stage2_sampler=GlobalDrawSampler(8,seed=42,selection_identity='selection')
    state=trainer.stage2_sampling_state()
    assert state['goal_prediction']['input_policy']=='clean_multiview'
    assert state['goal_prediction']['observation_mode']=='multiview'
    trainer.restore_stage2_sampling_state({'stage2':state})
    state['goal_prediction']['input_policy']='clean_head'
    with pytest.raises(ValueError,match='prediction policy'):trainer.restore_stage2_sampling_state({'stage2':state})


def test_saved_v2_loader_rejects_explicit_head_override_before_model(tmp_path,monkeypatch):
    arch,cfg,b=configured(tmp_path)
    out=tmp_path/'run';runtime.save_goal_assets(arch,out,cfg)
    cfg.model={'video_backbone':{'name':'wan22_ti2v_5b'}};OmegaConf.save(cfg,out/'config.yaml')
    import openwam.train.utils.stage2 as stage2
    monkeypatch.setattr(stage2,'validate_stage2_config',lambda cfg:None)
    monkeypatch.setattr(stage2,'validate_stage2_checkpoint',lambda cfg,path:None)
    import openwam.model as model
    monkeypatch.setattr(model,'resolve_architecture_config',lambda cfg:pytest.fail('model construction reached'))
    from openwam.train.utils.ckpt_model_loader import build_architecture_from_ckpt_dir
    with pytest.raises(ValueError,match='goal|mode|policy'):
        build_architecture_from_ckpt_dir(str(out),override_cfg=config(),weights_required=False)


@pytest.mark.parametrize('choice',['multiview','invalid'])
def test_launcher_explicit_goal_input_selection(tmp_path,choice):
    import os,subprocess,sys
    root=Path(__file__).resolve().parents[2]
    chosen=tmp_path/'python';capture=tmp_path/'args.jsonl'
    chosen.write_text('#!'+sys.executable+'\nimport os,sys,json\nwith open(os.environ["LAUNCH_CAPTURE"],"a") as f:f.write(json.dumps(sys.argv[1:])+"\\n")\n')
    chosen.chmod(0o755)
    env={**os.environ,'PYTHON':str(chosen),'LAUNCH_CAPTURE':str(capture),'STAGE2_GOAL_INPUT':choice,
         'STAGE1_GOAL_CHECKPOINT':'/fixture/v2','STAGE2_BASELINE_CHECKPOINT':'/fixture/base','STAGE2_OUTPUT_DIR':'/fixture/out',
         'STAGE2_STATS_REGISTRY':'/fixture/stats','NPROC_PER_NODE':'1'}
    result=subprocess.run(['bash',str(root/'scripts/launch_stage2.sh'),'sample','--train'],env=env,capture_output=True,text=True)
    if choice=='invalid':
        assert result.returncode!=0 and 'STAGE2_GOAL_INPUT' in result.stderr
        assert not capture.exists()
    else:
        assert result.returncode==0,result.stderr
        calls=[json.loads(line) for line in capture.read_text().splitlines()]
        assert len(calls)==2 and all('goal/stage2_multiview_sample' in args for args in calls)


def test_v2_launch_preflight_checks_metadata_contract_on_cpu(tmp_path):
    from safetensors.torch import save_file
    from tests.dataloader.test_curated_stage2 import make_native_source
    from openwam.train.utils.stage2 import validate_stage2_launch
    from omegaconf import open_dict
    arch,teacher_cfg,b=configured(tmp_path)
    cfg=config()
    with open_dict(cfg.goal):
        cfg.goal.format='openwam.goal.patch.v2';cfg.goal.input_policy='clean_multiview';cfg.goal.observation_mode='multiview'
    cfg.goal.checkpoint=teacher_cfg.goal.checkpoint
    baseline=tmp_path/'baseline';baseline.mkdir();OmegaConf.save({'model':cfg.model},baseline/'config.yaml')
    save_file({'action_backbone.action_encoder.weight':torch.zeros(8,112),
        'action_backbone.action_decoder.weight':torch.zeros(112,8),'action_backbone.action_decoder.bias':torch.zeros(112),
        'proprio_encoder.weight':torch.zeros(8,112)},str(baseline/'checkpoint_step_1.safetensors'))
    cfg.training.finetune_ckpt_path=str(baseline);cfg.dataloader.source_yaml=str(make_native_source(tmp_path))
    cfg.project.output_dir=str(tmp_path/'run')
    validate_stage2_launch(cfg,metadata_only=True)
    from safetensors import safe_open
    with safe_open(str(teacher_cfg.goal.checkpoint),framework='pt') as f:
        metadata=json.loads(f.metadata()['patch_goal_bundle'])
    metadata.pop('observation')
    bad=tmp_path/'broken.safetensors';save_file({'dummy':torch.zeros(1)},str(bad),metadata={'patch_goal_bundle':json.dumps(metadata)})
    cfg.goal.checkpoint=str(bad)
    with pytest.raises(ValueError,match='observation contract'):validate_stage2_launch(cfg,metadata_only=True)
    metadata['format']='openwam.goal.patch.v1'
    save_file({'dummy':torch.zeros(1)},str(bad),metadata={'patch_goal_bundle':json.dumps(metadata)})
    with pytest.raises(ValueError,match='format mismatch'):validate_stage2_launch(cfg,metadata_only=True)


def test_v2_fixed_input_records_actual_bf16_prevae_bytes(tmp_path,monkeypatch):
    from openwam.train import stage2_eval as e
    from tests.goal.test_stage2_history import sample,Tiny,schedule
    a=sample();a['goal_observed_multiview']=torch.full((3,1,384,320),.12345)
    a['goal_view_valid']=torch.tensor([True,True,False])
    model=Tiny();model.goal_expert.observation_mode='multiview';model.dtype=torch.bfloat16
    monkeypatch.setattr(e,'make_schedule',schedule)
    e.evaluate_stage2(model,[a],tmp_path,step=0,seed=4,num_steps=1)
    actual=model.calls[0][1]['goal_observed_multiview']
    assert actual.dtype==torch.bfloat16
    data=np.load(tmp_path/'fixed_inputs/case_000/supervision.npz')
    np.testing.assert_array_equal(data['goal_observed_multiview'],actual[0].float().numpy())
    np.testing.assert_array_equal(data['goal_observed_multiview_bfloat16_bits'],actual[0].view(torch.uint16).numpy())


def test_mask_cache_separates_identical_black_canvas_and_headless_rows_bypass(tmp_path):
    arch,cfg,b=configured(tmp_path);b=copy.copy(b);b['observed']=torch.full_like(b['observed'],-1.)
    condition(arch,b,torch.tensor([[True,True,False]]))
    condition(arch,b,torch.tensor([[True,False,False]]))
    assert len(arch.goal_cache.entries)==2
    got=runtime.prepare_goal_inputs(arch,None,['move'],b['context'],head_valid=[False],seeds=[17],
        goal_observed_multiview=torch.full_like(b['observed'],float('nan')),
        goal_view_valid=torch.tensor([[False,True,True]]))
    assert not got['goal_token_valid'].any() and len(arch.goal_cache.entries)==2


def test_stage2_preflight_export_shows_actual_clean_multiview(tmp_path,monkeypatch):
    from tests.dataloader.test_curated_stage2 import make_native_source
    from openwam.dataloader.curated_stage2 import CuratedStage2Dataset
    from openwam.dataloader.native_stage2_stats import prepare_native_stats
    from scripts.data.preflight_stage2 import export_sample
    from PIL import Image
    d=CuratedStage2Dataset(make_native_source(tmp_path),seed=4)
    registry=prepare_native_stats(d,tmp_path/'stats',max_episodes_per_profile=1,max_windows_per_profile=4)
    d=CuratedStage2Dataset(d.selection['source_yaml'],seed=4,normalization_registry=registry)
    monkeypatch.setattr(d,'_decode_views',lambda m,indices:([{'observation.images.cam_high':Image.new('RGB',(40,32),(100,120,140))} for _ in indices],{}))
    selected=d._draw(0);folder=tmp_path/'export'
    export_sample(d,*selected,folder,observation_mode='multiview')
    assert Image.open(folder/'goal_input.png').size==(320,384)
    data=np.load(folder/'supervision.npz')
    assert data['goal_observed_multiview'].shape==(3,1,384,320)
    np.testing.assert_array_equal(data['goal_view_valid'],[True,False,False])
    trace=json.loads((folder/'trace.json').read_text())
    assert trace['preprocessing']['goal']['shape']==[3,1,384,320]
    assert trace['goal_observation_mode']=='multiview'


def test_v2_real_goal_memory_gradients_and_full_public_generation(tmp_path):
    from tests.goal.test_patch_stage2_memory import inputs
    from openwam.model.video_backbone.wan.shared.diffusion.flow_match import FlowMatchScheduler
    arch,cfg,b=configured(tmp_path)
    goal=condition(arch,b);x=inputs()
    arch.train();before={name:p.detach().clone() for name,p in arch.goal_expert.named_parameters()}
    result=arch(**x,**goal)
    sum(value.square().mean() for value in result).backward()
    for p in (arch.video_backbone.dit.patch_embedding.weight,arch.action_backbone.action_encoder.weight,arch.goal_adapter.patch.weight):
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
    assert all(p.grad is None and torch.equal(p,before[name]) for name,p in arch.goal_expert.named_parameters())
    arch.video_backbone._scheduler=FlowMatchScheduler(template='Wan');arch.init_training_schedulers()
    arch.video_backbone.preprocess_input_for_inference=lambda **kwargs:{'context':b['context'],'latents':x['latents'].clone()}
    output=arch.generate([(1000.,1000.),(0.,0.)],'move',num_frames=9,action_num_frames=33,
        goal_observed_multiview=b['observed'],goal_view_valid=b['observed_view_valid'],goal_seed=17,decode_video=False)
    assert output['normalized_actions'].shape==(32,112)
    assert arch.goal_diagnostics['predicted_latent'].shape==(1,48,1,16,20)
    assert arch.goal_diagnostics['image_token_valid'].shape==(1,80)
    assert arch.goal_diagnostics['observed_view_valid'].tolist()==[[True,True,False]]
