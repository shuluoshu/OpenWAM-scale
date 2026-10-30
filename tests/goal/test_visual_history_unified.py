"""Unified Stage1 visual history uses honest bounded completion and portable assets."""
import json
from pathlib import Path
import pytest
import torch


def test_visual_history_preset_and_bounded_identity():
    from openwam.train.goal_curated import load_config
    from openwam.train.stage12 import epoch_identity
    root=Path(__file__).resolve().parents[2]
    config=load_config(root/'configs/goal/stage1_visual_history_overfit.yaml')
    assert config['model']['goal_representation']=='visual_history_masked_sa'
    assert config['dataset']['history']['enabled']
    population=dict(source_yaml_sha256='a'*64,index_sha256='b'*64,total_samples=3)
    identity=epoch_identity(config,population,1)
    assert identity['training']==dict(mode='bounded_updates',max_updates=1000,repeat_population=True)
    assert identity['representation']=='visual_history_masked_sa'


def test_bounded_visual_history_repeats_exactly_and_portable_resume(tmp_path):
    from openwam.train.goal_epoch import train_epoch,verify_completion,restore_epoch_checkpoint
    from openwam.train.goal_overfit import FP32MasterAdamW
    from openwam.train.epoch_order import sampling_contract
    from openwam.model.goal.dependencies import dependency_identity
    from openwam.train.goal_visual_history import load_visual_history_bundle
    from test_visual_history import make_model,history
    from test_multiview_patch import multiview_batch
    torch.set_num_threads(2)
    expert=make_model();obs,ctx,target,views=multiview_batch(1)
    class Data:
        indices=[]
        def __len__(self):return 2
        def sample(self,index):
            self.indices.append(index)
            return dict(id=str(index),family='tiny',instruction='move',granularity='task'),dict(
                observed=obs,context=ctx,targets=target,observed_view_valid=views,**history())
        def evaluation_samples(self):return [self.sample(0)]
        def scope_summary(self):return dict(total_samples=2)
    data=Data()
    identity=dict(total_samples=2,world_size=1,seed=42,representation=expert.representation,
        sampling=sampling_contract({'seed':42},2),training=dict(mode='bounded_updates',max_updates=5,repeat_population=True))
    deps=dependency_identity(expert.codec.vae,text_dim=expert.context_width,precomputed_text_fingerprint='d'*64)
    result=train_epoch(expert,data,None,tmp_path,identity=identity,dependencies=deps,
        checkpoint_every=3,eval_every=5,sampling_steps=1,max_updates=5,repeat_population=True,sample_cache_size=2)
    assert len(data.indices)==3  # one evaluation read and two distinct cached training reads
    assert result['updates']==result['completed_steps']==5
    assert result['seen_samples']==5 and result['unique_samples_seen']==2
    assert result['completion_kind']=='bounded_updates' and not result['finite_epoch_complete']
    assert not (tmp_path/'epoch_complete.json').exists()
    assert (tmp_path/'training_complete.json').is_file()
    assert verify_completion(tmp_path,identity)==result
    bundle=Path(result['checkpoint'])
    assert bundle.name=='visual_history.safetensors'
    loaded=load_visual_history_bundle(bundle,vae=expert.codec.vae,text_dim=expert.context_width,
        precomputed_text_fingerprint='d'*64)
    for key,value in expert.state_dict().items():torch.testing.assert_close(value,loaded.state_dict()[key],rtol=0,atol=0)
    clone=make_model().train_stage_a()
    from openwam.train.goal_curated import goal_parameter_groups
    optimizer=FP32MasterAdamW(goal_parameter_groups(clone,1e-5,1e-4))
    assert restore_epoch_checkpoint(clone,optimizer,tmp_path,identity)==(5,5)
    count=len(data.indices)
    assert train_epoch(expert,data,None,tmp_path,identity=identity,dependencies=deps,resume=True,max_updates=5,repeat_population=True)==result
    assert len(data.indices)==count
    proof=json.loads((tmp_path/'training_complete.json').read_text());proof['updates']=4
    (tmp_path/'training_complete.json').write_text(json.dumps(proof))
    with pytest.raises(ValueError,match='update|completion'):verify_completion(tmp_path,identity)


def test_visual_history_masked_nan_targets_have_finite_loss_and_gradients():
    from test_visual_history import make_model,history
    from test_multiview_patch import multiview_batch
    expert=make_model().train_stage_a();obs,ctx,target,views=multiview_batch(2)
    target.image_valid[0]=False;target.image[0]=float('nan');target.uv[0]=float('nan')
    target.point_valid[1,0]=False;target.uv[1,0]=float('nan')
    out=expert(obs,ctx,target,observed_view_valid=views,**history(2))
    assert out['valid_points']==1 and torch.isfinite(out['loss'])
    out['loss'].backward()
    assert all(torch.isfinite(p.grad).all() for p in expert.parameters() if p.grad is not None)


def test_missing_head_keeps_wrists_and_masks_head_supervision(tmp_path):
    from test_multiview_curated import native_fixture
    from openwam.dataloader.curated_goal import CuratedGoalDataset
    source,root=native_fixture(tmp_path)
    info=json.loads((root/'meta/info.json').read_text());del info['features']['observation.images.head_left']
    (root/'meta/info.json').write_text(json.dumps(info))
    data=CuratedGoalDataset(source,tmp_path/'cache',observation_mode='multiview',history={'enabled':True})
    record,batch=data.sample(0)
    assert batch['observed_view_valid'].tolist()==[[False,True,True]]
    assert not batch['targets'].image_valid.any()
    assert not batch['targets'].point_valid.any() and not batch['targets'].in_frame.any()
    assert not batch['history_valid'].any()
    assert (batch['observed'][...,:256,:]==-1).all()
    assert record['provenance']['observed_views']['head']['mask_reason']=='missing_camera_feature'
    from openwam.train.goal_overfit import prepare_overfit_images
    assert not prepare_overfit_images([record],device='cpu',dtype=torch.float32)[0]['targets'].image_valid.any()


def test_bounded_codec_cache_reuses_content_and_respects_limit():
    from test_visual_history import make_model
    from openwam.model.goal.codec import GoalCodec,GoalCodecSpec
    from openwam.model.goal.observation import MULTIVIEW_IMAGE_SHAPE,MULTIVIEW_LATENT_SHAPE
    expert=make_model();codec=expert.codec
    calls=[];original=codec.vae.encode
    def counted(*args,**kwargs):calls.append(1);return original(*args,**kwargs)
    codec.vae.encode=counted
    codec.enable_cache(max_bytes=200000)
    image=torch.zeros(1,3,1,256,320)
    expected=codec.encode(image);codec.encode(image.clone())
    assert len(calls)==1
    changed=image.clone();changed[0,0,0,0,0]=1
    codec.encode(changed)
    assert len(calls)==2
    # Observation codecs must share the same bounded run-local cache.
    assert expert.observation_codec.cache is codec.cache
    assert codec.cache.nbytes<=200000
    codec.clear_cache();codec.encode(image)
    assert len(calls)==3
    torch.testing.assert_close(codec.encode(image),expected,rtol=0,atol=0)


def test_visual_history_stage12_handshake_has_no_query_dependency():
    from openwam.train.stage12 import validate_stage1_stage2_contract
    sampling=dict(enabled=True,num_frames=10,mean_interval_s=1.,jitter_fraction=0.,frame_dropout_prob=0.)
    stage1=dict(model=dict(goal_representation='visual_history_masked_sa'),observation_mode='multiview',
        dataset=dict(history=sampling),components=dict(dit=dict(extra_kwargs=dict(dim=32))))
    stage2=dict(model=dict(goal_representation='visual_history_masked_sa'),goal=dict(
        observation_mode='multiview',visual_history=dict(dim=32,num_tokens=120)),dataloader=dict(history=sampling))
    validate_stage1_stage2_contract(stage1,stage2)
    stage2['goal']['visual_history']['dim']=64
    with pytest.raises(ValueError,match='token contract'):validate_stage1_stage2_contract(stage1,stage2)


def test_visual_history_optimizer_groups_pool_adapter_and_gate():
    from test_visual_history import make_model
    from openwam.train.goal_curated import goal_parameter_groups
    expert=make_model().train_stage_a()
    groups=goal_parameter_groups(expert,1e-5,1e-4)
    history=next(group for group in groups if group['name']=='history')
    expected={id(p) for module in (expert.history_pool,expert.visual_adapter) for p in module.parameters() if p.requires_grad}
    expected.add(id(expert.history_gate))
    assert {id(p) for p in history['params']}==expected and history['lr']==1e-4
    all_params=[id(p) for group in groups for p in group['params']]
    assert len(all_params)==len(set(all_params))
    assert set(all_params)=={id(p) for p in expert.parameters() if p.requires_grad}


def test_interrupted_bounded_visual_history_resume_matches_uninterrupted(tmp_path,monkeypatch):
    import openwam.train.goal_epoch as epoch
    from openwam.train.epoch_order import sampling_contract
    from openwam.model.goal.dependencies import dependency_identity
    from test_visual_history import make_model,history
    from test_multiview_patch import multiview_batch
    torch.set_num_threads(2)
    original=make_model();obs,ctx,target,views=multiview_batch(1);past=history()
    class Data:
        def __len__(self):return 2
        def sample(self,index):
            return dict(id=str(index),family='tiny',instruction='move',granularity='task'),dict(
                observed=obs,context=ctx,targets=target,observed_view_valid=views,**past)
        def evaluation_samples(self):return [self.sample(0)]
        def scope_summary(self):return dict(total_samples=2)
    identity=dict(total_samples=2,world_size=1,seed=42,representation=original.representation,
        sampling=sampling_contract({'seed':42},2),training=dict(mode='bounded_updates',max_updates=5,repeat_population=True))
    dependencies=dependency_identity(original.codec.vae,text_dim=original.context_width,precomputed_text_fingerprint='d'*64)
    kwargs=dict(identity=identity,dependencies=dependencies,max_updates=5,repeat_population=True,
        checkpoint_every=2,eval_every=5,sampling_steps=1,sample_cache_size=2)
    epoch.train_epoch(original,Data(),None,tmp_path/'continuous',**kwargs)
    interrupted=make_model();save=epoch.save_epoch_checkpoint
    def interrupt(*args,**kw):
        result=save(*args,**kw)
        if kw['update']==2:raise RuntimeError('simulated stop after durable update 2')
        return result
    monkeypatch.setattr(epoch,'save_epoch_checkpoint',interrupt)
    with pytest.raises(RuntimeError,match='simulated stop'):
        epoch.train_epoch(interrupted,Data(),None,tmp_path/'interrupted',**kwargs)
    assert not (tmp_path/'interrupted/training_complete.json').exists()
    monkeypatch.setattr(epoch,'save_epoch_checkpoint',save)
    resumed=make_model()
    proof=epoch.train_epoch(resumed,Data(),None,tmp_path/'interrupted',resume=True,**kwargs)
    assert proof['updates']==5
    events=[json.loads(line) for line in (tmp_path/'interrupted/progress.jsonl').read_text().splitlines()]
    assert [row['step'] for row in events if row['event']=='update']==[1,2,3,4,5]
    for key,value in original.state_dict().items():torch.testing.assert_close(value,resumed.state_dict()[key],rtol=0,atol=0)


def test_missing_head_image_is_excluded_from_diagnostic_mae(tmp_path):
    from test_visual_history import make_model,history
    from test_multiview_patch import multiview_batch
    from openwam.train.goal_patch import evaluate_patch
    obs,ctx,target,views=multiview_batch(1);target.image_valid.zero_()
    result=evaluate_patch(make_model(),[dict(observed=obs,context=ctx,targets=target,
        observed_view_valid=views,sources=['missing-head'],**history())],tmp_path/'eval',num_steps=1)
    assert result['image_mae_uint8'] is None
    assert result['review'][0]['image_mae_uint8'] is None
    assert not result['review'][0]['target_image_valid']


def test_absolute_native_wrist_only_camera_layout():
    from openwam.dataloader.native_cameras import resolve_native_cameras
    info=dict(features={'observation.images.left':dict(dtype='video',shape=[24,32,3]),
        'observation.images.right':dict(dtype='video',shape=[24,32,3])})
    assert resolve_native_cameras('/external/native-wrist-only',info)==[
        None,'observation.images.left','observation.images.right']


def test_launcher_exports_native_resolved_dependencies_and_text_contexts(tmp_path):
    from scripts.train_goal_epoch import export_run_artifacts
    from openwam.train.goal_curated import TextContextCache
    from openwam.train.goal_epoch import sha256
    from omegaconf import OmegaConf
    config=dict(model=dict(goal_representation='visual_history_masked_sa'),max_updates=1000,repeat_population=True)
    dependencies=dict(text=dict(fingerprint='a'*64),vae=dict(fingerprint='b'*64))
    cache=TextContextCache(lambda prompt:(torch.ones(1,2,3),torch.tensor([2])))
    cache.get('move')
    export_run_artifacts(tmp_path,config,dependencies,identity={'seed':42},context_cache=cache)
    assert OmegaConf.to_container(OmegaConf.load(tmp_path/'resolved_config.yaml'))==config
    assert json.loads((tmp_path/'dependencies.json').read_text())==dependencies
    contexts=torch.load(tmp_path/'text_contexts.pt',map_location='cpu',weights_only=True)
    torch.testing.assert_close(contexts['move'][0],cache.entries['move'][0],rtol=0,atol=0)
    metadata=json.loads((tmp_path/'artifacts.json').read_text())
    assert metadata['identity']=={'seed':42} and metadata['text_prompts']==['move']
    assert all(sha256(tmp_path/name)==digest for name,digest in metadata['files'].items())
