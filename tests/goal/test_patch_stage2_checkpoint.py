"""Real tiny Wan/action/patch expert persistence and next-update equivalence."""
import copy
import json
import random
import shutil
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file, save_file
from tests.goal.test_patch_stage2_memory import make_arch, inputs
from openwam.train.goal_patch import make_smoke_data, save_patch_bundle
from openwam.model.goal import runtime


def configured(tmp_path):
    torch.set_num_threads(1)
    arch=make_arch();teacher,batches,_=make_smoke_data()
    teacher.codec.vae=arch.video_backbone.vae
    asset=tmp_path/'original.safetensors'
    save_patch_bundle(teacher,asset,precomputed_text_fingerprint='a'*64)
    cfg=OmegaConf.create({'goal':{'enabled':True,'checkpoint':str(asset),
        'precomputed_text_fingerprint':'a'*64,'num_steps':1}})
    runtime.configure_goal(arch,cfg)
    return arch,cfg,batches[0]


def conditioned(arch,batch):
    return runtime.prepare_goal_inputs(arch,batch['observed'],['move'],batch['context'],
        head_valid=torch.tensor([True]),seeds=[17])


def test_actual_prefixes_export_relocation_and_strict_adapter(tmp_path):
    arch,cfg,batch=configured(tmp_path)
    state=arch.state_dict()
    assert any(k.startswith('goal_expert.dit.') for k in state)
    assert not any(k.startswith('goal_expert.codec.') for k in state)
    before=conditioned(arch,batch)
    saved=tmp_path/'saved';saved.mkdir()
    runtime.save_goal_assets(arch,saved,cfg)
    arch.save_checkpoint(str(saved/'checkpoint_step_1.safetensors'))
    unified=load_file(str(saved/'checkpoint_step_1.safetensors'))
    assert not any(k.startswith('goal_expert.') for k in unified)
    assert {k for k in unified if k.startswith('goal_adapter.')} == {k for k in state if k.startswith('goal_adapter.')}
    assert cfg.goal.checkpoint=='goal/expert.safetensors'
    assert cfg.goal.asset_sha256==runtime.goal_asset_sha256(saved/cfg.goal.checkpoint)
    (tmp_path/'original.safetensors').unlink()
    moved=tmp_path/'moved';shutil.move(saved,moved)
    restored=make_arch();runtime.configure_goal_adapter(restored)
    restored.load_checkpoint(str(moved/'checkpoint_step_1.safetensors'))
    runtime.configure_goal(restored,cfg,checkpoint_dir=moved)
    after=conditioned(restored,batch)
    for k in before:torch.testing.assert_close(before[k],after[k],rtol=0,atol=0)
    x=inputs()
    for a,b in zip(arch(**x,**before),restored(**x,**after)):torch.testing.assert_close(a,b,rtol=0,atol=0)
    missing=next(k for k in unified if k.startswith('goal_adapter.'));unified.pop(missing)
    save_file(unified,str(moved/'broken.safetensors'))
    with pytest.raises(RuntimeError,match='goal_adapter'):restored.load_checkpoint(str(moved/'broken.safetensors'))
    with (moved/cfg.goal.checkpoint).open('ab') as stream:stream.write(b'corrupt')
    with pytest.raises(ValueError,match='SHA256'):runtime.configure_goal(make_arch(),cfg,checkpoint_dir=moved)


def test_frozen_dependencies_mismatch_and_injected_teacher_rejected(tmp_path):
    arch,cfg,_=configured(tmp_path)
    runtime.save_goal_assets(arch,tmp_path/'saved',cfg)
    altered=make_arch();altered.video_backbone.vae.marker.fill_(1)
    with pytest.raises(ValueError,match='VAE|vae'):runtime.configure_goal(altered,cfg,checkpoint_dir=tmp_path/'saved')
    arch.save_checkpoint(str(tmp_path/'weights.safetensors'))
    payload=load_file(str(tmp_path/'weights.safetensors'));payload['goal_expert.dit.fake']=torch.zeros(1)
    save_file(payload,str(tmp_path/'injected.safetensors'))
    with pytest.raises(RuntimeError,match='goal_expert'):arch.load_checkpoint(str(tmp_path/'injected.safetensors'))


def test_native_registry_is_portable_and_deploy_requires_selection(tmp_path):
    from openwam.train.utils.checkpointing import save_normalization_stats, verify_resume_normalization_stats
    from openwam.deploy.model_loader import _build_normalizer
    from openwam.dataloader.utils.mora_relative import MoraHeadTCPRepresentation
    from openwam.dataloader.native_stage2_stats import normalize
    rep=MoraHeadTCPRepresentation({'features':{}})
    import hashlib
    selection_text='selected: robot\n'
    contract={'selection_sha256':hashlib.sha256(selection_text.encode()).hexdigest(),'data_root':'/source','split':'train'}
    stats={k:np.full(52,v).tolist() for k,v in [('q01',-2),('q99',4),('count',10)]}
    state_stats={**stats,'q01':np.full(52,-3).tolist()}
    registry={'contract':contract,'sources':{'robot':{'entry':'key'}},'entries':{'key':{'contract':rep.contract,'action':stats,'state':state_stats}}}
    raw=json.dumps(registry,sort_keys=True,separators=(',',':')).encode()
    import hashlib
    src=tmp_path/'source';src.mkdir();path=src/('registry-'+hashlib.sha256(raw).hexdigest()+'.json');path.write_bytes(raw)
    selection=src/'selection.yaml';selection.write_text(selection_text)
    dataset=SimpleNamespace(registry_path=str(path),registry=registry,selection={'source_yaml':str(selection)},stats_contract=lambda:contract)
    out=tmp_path/'out';save_normalization_stats(str(out),dataset)
    verify_resume_normalization_stats(str(out),dataset)
    shutil.rmtree(src)
    cfg=OmegaConf.create({'goal':{'enabled':True},'dataloader':{'type':'curated_stage2'}})
    with pytest.raises(ValueError,match='normalization_profile'): _build_normalizer(cfg,str(out))
    cfg.dataloader.normalization_profile='key';normalizer=_build_normalizer(cfg,str(out))
    values=np.zeros((1,52),np.float32);values[:,[9,19]]=[.2,.8]
    expected=normalize(values,np.ones_like(values,bool),state_stats,source='test')
    np.testing.assert_allclose(normalizer.normalize(values)[:,rep.scatter],expected)
    model=np.zeros((1,112),np.float32);model[:,rep.scatter]=normalize(values,np.ones_like(values,bool),stats,source='test')
    np.testing.assert_allclose(normalizer.unnormalize(model),values,atol=1e-6)
    metadata=json.loads((out/'normalization/manifest.json').read_text())
    copied=out/metadata['registry'];copied.write_text('{}')
    with pytest.raises(ValueError,match='SHA256|fingerprint'): _build_normalizer(cfg,str(out))


def test_full_state_real_accelerate_next_update_rng_scheduler_and_sampler(tmp_path,monkeypatch):
    # No DeepSpeed engine participates in this real CPU Accelerate test. Avoid
    # importing its optional CPU accelerator, which queries inaccessible /proc.
    import accelerate.utils.other
    monkeypatch.setattr(accelerate.utils.other, "is_deepspeed_available", lambda: False)
    from accelerate import Accelerator
    from openwam.train.utils.checkpointing import save_full_state,load_full_state
    from openwam.train.utils.stage2 import GlobalDrawSampler
    arch,cfg,batch=configured(tmp_path);goal=conditioned(arch,batch);x=inputs()
    accelerator=Accelerator(cpu=True)
    optimizer=torch.optim.AdamW([p for p in arch.parameters() if p.requires_grad],lr=1e-4)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=.9)
    arch,optimizer,scheduler=accelerator.prepare(arch,optimizer,scheduler)
    sampler=GlobalDrawSampler(16,seed=4,selection_identity='selection')
    from torch.utils.data import DataLoader, TensorDataset
    population=TensorDataset(torch.arange(16,dtype=torch.float32).reshape(-1,1))
    def next_batch():
        return next(iter(DataLoader(population,batch_size=2,sampler=sampler)))[0]
    def update(batch):
        factor=random.random()+float(np.random.rand())+torch.rand(())+batch.mean()
        optimizer.zero_grad()
        result=arch(**x,**goal)
        loss=sum(y.square().mean() for y in result)*factor
        accelerator.backward(loss);optimizer.step();scheduler.step()
        return loss.detach().clone()
    update(next_batch());runtime.save_goal_assets(accelerator.unwrap_model(arch),tmp_path,cfg)
    cfg.model={'video_backbone':{'name':'wan22_ti2v_5b'}}
    OmegaConf.save(cfg,tmp_path/'config.yaml')
    metadata={'version':'openwam.stage2.trainer.v1','sampler':sampler.state_dict(committed_offset=2),
        'goal_asset_sha256':cfg.goal.asset_sha256}
    accelerator.unwrap_model(arch).save_checkpoint(str(tmp_path/'checkpoint_step_1.safetensors'))
    save_full_state(accelerator,str(tmp_path),1,1,0,trainer_metadata={'stage2':metadata})
    saved_optimizer=copy.deepcopy(optimizer.state_dict());saved_scheduler=copy.deepcopy(scheduler.state_dict())
    sampler.load_state_dict(metadata['sampler']);expected_batch=next_batch()
    expected_loss=update(expected_batch);expected={k:v.detach().clone() for k,v in arch.state_dict().items()}
    expected_optimizer=copy.deepcopy(optimizer.state_dict());expected_rng=(random.random(),float(np.random.rand()),torch.rand(3))
    # Reconstruct all objects from a relocated run after deleting the source asset.
    (tmp_path/'original.safetensors').unlink()
    moved=tmp_path.parent/(tmp_path.name+'-moved');shutil.move(tmp_path,moved)
    import openwam.model as model
    from openwam.train.utils.ckpt_model_loader import build_architecture_from_ckpt_dir
    resolved=SimpleNamespace(registry_name='tiny',params={'video_backbone':{}},
        canonical=SimpleNamespace(framework='dual_system',variant='joint_self_attn'))
    monkeypatch.setattr(model,'resolve_architecture_config',lambda cfg:resolved)
    monkeypatch.setattr(model,'build_architecture',lambda *args:make_arch())
    _,arch,_=build_architecture_from_ckpt_dir(str(moved),weights_required=False)
    accelerator=Accelerator(cpu=True)
    optimizer=torch.optim.AdamW([p for p in arch.parameters() if p.requires_grad],lr=1e-4)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=.9)
    arch,optimizer,scheduler=accelerator.prepare(arch,optimizer,scheduler)
    state_dir=moved/'accel_state_step_1'
    restored=load_full_state(accelerator,str(state_dir),require_stage2=True)
    assert scheduler.state_dict()==saved_scheduler
    for index,entry in saved_optimizer['state'].items():
        for key,value in entry.items():torch.testing.assert_close(optimizer.state_dict()['state'][index][key],value,rtol=0,atol=0)
    sampler.load_state_dict(restored['stage2']['sampler']);assert list(sampler)[:3]==[2,3,4]
    actual_batch=next_batch();torch.testing.assert_close(actual_batch,expected_batch,rtol=0,atol=0)
    actual_loss=update(actual_batch);torch.testing.assert_close(actual_loss,expected_loss,rtol=0,atol=0)
    for k,v in arch.state_dict().items():torch.testing.assert_close(v,expected[k],rtol=0,atol=0)
    for index,entry in expected_optimizer['state'].items():
        for key,value in entry.items():torch.testing.assert_close(optimizer.state_dict()['state'][index][key],value,rtol=0,atol=0)
    actual_rng=(random.random(),float(np.random.rand()),torch.rand(3))
    assert actual_rng[:2]==expected_rng[:2];torch.testing.assert_close(actual_rng[2],expected_rng[2],rtol=0,atol=0)
    marker=state_dir/'trainer_state.json'
    persisted=json.loads(marker.read_text())
    changed=copy.deepcopy(persisted);changed['accelerate_objects']['schedulers']=0
    marker.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='scheduler registration'):
        load_full_state(accelerator,str(state_dir),require_stage2=True)
    marker.write_text(json.dumps(persisted))
    optimizer_file=state_dir/'optimizer.bin';original_optimizer=optimizer_file.read_bytes()
    optimizer_file.write_bytes(original_optimizer[:32])
    with pytest.raises(ValueError,match='missing/truncated'):
        load_full_state(accelerator,str(state_dir),require_stage2=True)
    optimizer_file.write_bytes(original_optimizer)
    marker.unlink()
    with pytest.raises(ValueError,match='full|marker'):load_full_state(accelerator,str(state_dir),require_stage2=True)


def test_generate_uses_explicit_clean_head_and_shared_prediction(tmp_path,monkeypatch):
    arch,cfg,batch=configured(tmp_path)
    lengths=torch.tensor([2])
    expected=runtime.prepare_goal_inputs(arch,batch['observed'],['move'],batch['context'],
        head_valid=torch.tensor([True]),seeds=[17],context_lens=lengths)
    arch.video_backbone.preprocess_input_for_inference=lambda **kwargs:{'context':batch['context'],'seq_lens':lengths}
    class Reached(Exception):pass
    def capture(*args,**kwargs):
        assert args[0] is arch
        result=runtime.prepare_goal_inputs(*args,**kwargs)
        for key in expected:torch.testing.assert_close(result[key],expected[key],rtol=0,atol=0)
        raise Reached
    monkeypatch.setattr(runtime,'prepare_inference_goal_inputs',capture)
    with pytest.raises(Reached):
        arch.generate(None,'move',goal_observed_head=batch['observed'],goal_head_valid=True,goal_seed=17)


def test_complete_generate_returns_normalized_actions_and_goal_diagnostics(tmp_path):
    arch,cfg,batch=configured(tmp_path)
    from openwam.model.video_backbone.wan.shared.diffusion.flow_match import FlowMatchScheduler
    arch.video_backbone._scheduler=FlowMatchScheduler(template="Wan")
    arch.init_training_schedulers()
    latent=inputs()['latents'];ctx=batch['context']
    arch.video_backbone.preprocess_input_for_inference=lambda **kwargs:{'context':ctx,'latents':latent.clone()}
    arch.normalizer=SimpleNamespace(unnormalize=lambda x:x+10)
    result=arch.generate([(1000.,1000.),(0.,0.)],'move',num_frames=9,action_num_frames=33,
        goal_observed_head=batch['observed'],goal_head_valid=True,goal_seed=17,decode_video=False)
    assert result['normalized_actions'].shape==(32,112)
    np.testing.assert_allclose(result['actions'],result['normalized_actions']+10)
    assert arch.goal_diagnostics['predicted_latent'].shape==(1,48,1,16,20)
    assert arch.goal_diagnostics['predicted_uv'].shape==(1,2,2)
    with pytest.raises(ValueError,match='explicit clean'):
        arch.generate([], 'move',decode_video=False)
    with pytest.raises(ValueError,match='predicted goals'):
        arch.generate([], 'move',goal_latent=torch.zeros(1),decode_video=False)


def test_shared_clean_head_preprocessing():
    from PIL import Image
    image=Image.fromarray(np.arange(20*30*3,dtype=np.uint8).reshape(20,30,3))
    expected=torch.from_numpy(np.array(image.resize((320,256),Image.Resampling.BILINEAR),copy=True)).permute(2,0,1).unsqueeze(1).float()/127.5-1
    torch.testing.assert_close(runtime.clean_goal_head(image),expected,rtol=0,atol=0)


def test_train_and_deploy_loaders_restore_real_adapter_before_goal(tmp_path,monkeypatch):
    import openwam.model as model
    from openwam.train.utils.ckpt_model_loader import build_architecture_from_ckpt_dir
    from openwam.deploy.model_loader import load_from_checkpoint_dir
    arch,cfg,batch=configured(tmp_path)
    cfg.model={'video_backbone':{'name':'wan22_ti2v_5b'}}
    cfg.training={'mixed_precision':'no'}
    runtime.save_goal_assets(arch,tmp_path/'run',cfg)
    OmegaConf.save(cfg,tmp_path/'run/config.yaml')
    arch.save_checkpoint(str(tmp_path/'run/checkpoint_step_1.safetensors'))
    expected=conditioned(arch,batch)
    # Replace only expensive skeleton factory; all weights, adapter, expert,
    # dependency checks and public train/deploy loading code remain real.
    resolved=SimpleNamespace(registry_name='tiny',params={'video_backbone':{}},
        canonical=SimpleNamespace(framework='dual_system',variant='joint_self_attn'))
    monkeypatch.setattr(model,'resolve_architecture_config',lambda cfg:resolved)
    monkeypatch.setattr(model,'build_architecture',lambda *args:make_arch())
    _,trained,_=build_architecture_from_ckpt_dir(str(tmp_path/'run'),weights_required=True)
    _,deployed=load_from_checkpoint_dir(str(tmp_path/'run'),device='cpu')
    for restored in (trained,deployed):
        for k,v in expected.items():torch.testing.assert_close(conditioned(restored,batch)[k],v,rtol=0,atol=0)
        for key,value in arch.goal_adapter.state_dict().items():torch.testing.assert_close(restored.goal_adapter.state_dict()[key],value,rtol=0,atol=0)


def test_resume_normalization_uses_packaged_selection_and_checks_data_identity(tmp_path):
    from openwam.train.utils.stage2_assets import resolve_resume_normalization
    # Exercise the production package builder/reader using the native fixture.
    from tests.dataloader.test_curated_stage2 import make_native_source
    from openwam.dataloader.curated_stage2 import CuratedStage2Dataset
    from openwam.dataloader.native_stage2_stats import prepare_native_stats
    from openwam.train.utils.checkpointing import save_normalization_stats, verify_resume_normalization_stats
    selection=make_native_source(tmp_path)
    dataset=CuratedStage2Dataset(selection,seed=4)
    registry=prepare_native_stats(dataset,tmp_path/'stats',max_episodes_per_profile=1,max_windows_per_profile=4)
    dataset=CuratedStage2Dataset(selection,seed=4,normalization_registry=registry)
    saved=tmp_path/'saved';save_normalization_stats(str(saved),dataset)
    selection.unlink();shutil.rmtree(tmp_path/'stats')
    moved=tmp_path/'moved';shutil.move(saved,moved)
    cfg=OmegaConf.create({'dataloader':{'source_yaml':'missing','normalization_registry':'missing'}})
    resolve_resume_normalization(cfg,moved)
    restored=CuratedStage2Dataset.from_config(OmegaConf.to_container(cfg.dataloader))
    verify_resume_normalization_stats(str(moved),restored)
    assert restored.registry==dataset.registry
    cfg.dataloader.data_root=str(tmp_path/'different_root')
    with pytest.raises(ValueError,match='data_root identity'):
        resolve_resume_normalization(cfg,moved)


def test_failed_full_state_rewrite_invalidates_completion_marker(tmp_path):
    from openwam.train.utils.checkpointing import save_full_state,find_latest_accel_state
    directory=tmp_path/'accel_state_step_1';directory.mkdir()
    (directory/'trainer_state.json').write_text('{}')
    def fail(path):raise RuntimeError('interrupted save')
    accelerator=SimpleNamespace(is_main_process=True,wait_for_everyone=lambda:None,save_state=fail)
    with pytest.raises(RuntimeError,match='interrupted'):
        save_full_state(accelerator,str(tmp_path),1,1,0)
    assert find_latest_accel_state(str(tmp_path)) is None


def test_training_resume_contract_detects_scheduler_and_augmentation_changes():
    from openwam.train.utils.stage2 import stage2_resume_contract
    cfg=OmegaConf.create({'training':{'lr_scheduler':'cosine','mixed_precision':'bf16','max_steps':100},
        'dataloader':{'observation_augmentation':{'enabled':True}}})
    original=stage2_resume_contract(cfg)
    cfg.training.lr_scheduler=None
    assert stage2_resume_contract(cfg)!=original
    cfg.training.lr_scheduler='cosine';cfg.dataloader.observation_augmentation.enabled=False
    assert stage2_resume_contract(cfg)!=original


def test_native_deploy_masks_unavailable_profile_dimensions(tmp_path):
    from tests.dataloader.test_curated_stage2 import make_native_source
    from openwam.dataloader.curated_stage2 import CuratedStage2Dataset
    from openwam.dataloader.native_stage2_stats import prepare_native_stats
    from openwam.train.utils.checkpointing import save_normalization_stats
    from openwam.deploy.model_loader import _build_normalizer
    from openwam.dataloader.utils.mora_relative import MoraHeadTCPRepresentation
    selection=make_native_source(tmp_path)
    dataset=CuratedStage2Dataset(selection)
    path=prepare_native_stats(dataset,tmp_path/'stats',max_episodes_per_profile=1,max_windows_per_profile=4)
    dataset=CuratedStage2Dataset(selection,normalization_registry=path)
    save_normalization_stats(str(tmp_path/'saved'),dataset)
    key=next(iter(dataset.registry['entries']));entry=dataset.registry['entries'][key]
    cfg=OmegaConf.create({'dataloader':{'type':'curated_stage2','normalization_profile':key}})
    normalizer=_build_normalizer(cfg,str(tmp_path/'saved'))
    active=np.asarray(entry['action']['count'])>0
    np.testing.assert_array_equal(normalizer._dst_index,MoraHeadTCPRepresentation.scatter[active])
    assert np.all(normalizer.unnormalize(np.ones((1,112),np.float32))[:,~active]==0)


def test_stage2_warmstart_preflight_resolves_saved_relative_goal(tmp_path):
    from tests.goal.test_stage2_config import config
    from tests.dataloader.test_curated_stage2 import make_native_source
    from openwam.dataloader.curated_stage2 import CuratedStage2Dataset
    from openwam.dataloader.native_stage2_stats import prepare_native_stats
    from openwam.train.utils.stage2 import validate_stage2_launch
    cfg=config();selection=make_native_source(tmp_path)
    dataset=CuratedStage2Dataset(selection)
    registry=prepare_native_stats(dataset,tmp_path/'stats',max_episodes_per_profile=1,max_windows_per_profile=4)
    arch,goal_cfg,_=configured(tmp_path);saved=tmp_path/'saved'
    runtime.save_goal_assets(arch,saved,goal_cfg)
    cfg.goal.checkpoint=goal_cfg.goal.checkpoint
    OmegaConf.update(cfg,'goal.asset_sha256',goal_cfg.goal.asset_sha256,force_add=True)
    OmegaConf.save(cfg,saved/'config.yaml')
    save_file({'action_backbone.action_encoder.weight':torch.zeros(8,112),
        'action_backbone.action_decoder.weight':torch.zeros(112,8),
        'action_backbone.action_decoder.bias':torch.zeros(112),
        'proprio_encoder.weight':torch.zeros(8,112)},str(saved/'checkpoint_step_1.safetensors'))
    (tmp_path/'original.safetensors').unlink()
    cfg.goal.checkpoint=None
    cfg.training.finetune_ckpt_path=str(saved)
    cfg.dataloader.source_yaml=str(selection);cfg.dataloader.normalization_registry=registry
    cfg.project.output_dir=str(tmp_path/'run')
    result=validate_stage2_launch(cfg)
    assert result['state'].startswith('warm-start')
