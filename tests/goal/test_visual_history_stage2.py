"""Portable native visual-history conditioning through real Stage2 blocks."""
import copy

import pytest
import torch
from omegaconf import OmegaConf, open_dict

from test_visual_history import make_model, history
from test_multiview_patch import multiview_batch
from test_patch_stage2_memory import make_arch, inputs
from openwam.model.goal import runtime
from openwam.train.goal_visual_history import save_visual_history_bundle


def config():
    return dict(enabled=True, representation='visual_history_masked_sa',
        format='openwam.visual_history.v1', input_policy='clean_multiview',
        observation_mode='multiview', mode='predicted', conditioning='visual_history_memory',
        visual_history=dict(dim=48,num_tokens=120), history=dict(enabled=True),
        num_steps=1, microbatch_size=1, cache_size=8, cache_max_bytes=1000000)


def load_encoder(tmp_path):
    expert=make_model().freeze();expert.history_gate.data.fill_(.2)
    path=tmp_path/'visual.safetensors'
    save_visual_history_bundle(expert,path,precomputed_text_fingerprint='a'*64)
    encoder=runtime.load_stage2_goal(path,vae=expert.codec.vae,text_encoder=None,
        tokenizer=None,text_dim=16,device='cpu',dtype=torch.float32,
        precomputed_text_fingerprint='a'*64)
    return expert,encoder,path


def attach(encoder):
    a=make_arch();a.goal_expert=encoder;a.goal_config=config()
    runtime.configure_goal_adapter(a,config());a.goal_cache=runtime.GoalPredictionCache()
    a.goal_asset_identity='visual-fixture'
    return a


def test_encoder_only_bundle_prunes_decoder_and_preserves_condition_exactly(tmp_path):
    full,encoder,path=load_encoder(tmp_path)
    assert not hasattr(encoder.dit,'blocks') and not hasattr(encoder.dit,'head')
    assert not hasattr(encoder.dit,'text_embedding')
    assert hasattr(encoder.dit,'patch_embedding') and hasattr(encoder.dit,'time_embedding')
    assert hasattr(encoder.dit,'time_projection') and hasattr(encoder.dit,'freqs')
    assert not encoder.training and not any(p.requires_grad for p in encoder.parameters())
    assert not any('vae' in key for key in encoder.state_dict())
    assert encoder.history_config.enabled and encoder.history_options['num_history_tokens']==5
    obs,ctx,_,views=multiview_batch(1);h=history()
    expected=full.condition_tokens(obs,observed_view_valid=views,**h)
    actual=encoder.condition_tokens(obs,observed_view_valid=views,**h)
    assert actual.shape==(1,120,48)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    encoder.train();assert not encoder.training
    with pytest.raises(RuntimeError,match='no goal decoder'):encoder.generate(obs,ctx)
    with pytest.raises(ValueError,match='text dependency'):
        runtime.load_stage2_goal(path,vae=encoder.codec.vae,text_encoder=None,tokenizer=None,
            text_dim=16,device='cpu',dtype=torch.float32,precomputed_text_fingerprint='b'*64)


def test_runtime_uses_visual_mask_and_never_reads_text_or_decodes(tmp_path,monkeypatch):
    _,encoder,_=load_encoder(tmp_path);a=attach(encoder)
    monkeypatch.setattr(encoder,'generate',lambda *a,**kw:pytest.fail('decoder invoked'))
    obs,ctx,_,views=multiview_batch(2);views[0,1]=False;views[1,0]=False
    h=history(2)
    kwargs=dict(head_valid=views[:,0],seeds=[1,2],goal_observed_multiview=obs,
        goal_view_valid=views,**h)
    result=runtime.prepare_goal_inputs(a,None,['a','b'],ctx,**kwargs)
    from openwam.model.goal.observation import observation_token_mask
    assert result['goal_latent'].shape==(2,120,48)
    assert torch.equal(result['goal_token_valid'],observation_token_mask(views))
    assert not result['goal_latent'].requires_grad
    changed=runtime.prepare_goal_inputs(a,None,['new','text'],ctx+123,**kwargs)
    torch.testing.assert_close(result['goal_latent'],changed['goal_latent'],rtol=0,atol=0)
    assert a.goal_diagnostics['representation']=='visual_history_masked_sa'
    assert not a.goal_diagnostics['goal_decode_available']
    assert torch.isnan(a.goal_diagnostics['predicted_uv']).all()
    policy=runtime.goal_prediction_policy(a)
    assert policy['format']=='openwam.visual_history.v1' and policy['num_steps']==1
    assert policy['visual_history']==dict(dim=48,num_tokens=120)
    # Complete missing rows skip encoder and never validate an empty camera set.
    kwargs.update(head_valid=[False,False],goal_view_valid=torch.zeros_like(views),
                  goal_observed_multiview=None)
    empty=runtime.prepare_goal_inputs(a,None,['a','b'],ctx,**kwargs)
    assert not empty['goal_token_valid'].any() and not empty['goal_latent'].any()


@pytest.mark.parametrize('mode',['joint','none'])
def test_native_memory_gradients_masking_and_frozen_encoder(tmp_path,mode):
    _,encoder,_=load_encoder(tmp_path);a=attach(encoder);a.cfg['future_video_mode']=mode
    x=inputs();x['first_frame_latents']=x['latents'][:,:,:1].clone()
    z=torch.randn(1,120,48,requires_grad=True);mask=torch.ones(1,120,dtype=torch.bool);mask[:,80:]=False
    output=a(**x,goal_latent=z,goal_token_valid=mask)
    invisible=z.detach().clone();invisible[:,80:]+=50
    masked=a(**x,goal_latent=invisible,goal_token_valid=mask)
    torch.testing.assert_close(output[1],masked[1])
    visible=z.detach().clone();visible[:,:80]+=5
    changed=a(**x,goal_latent=visible,goal_token_valid=mask)
    assert not torch.allclose(output[1],changed[1])
    if mode=='joint':
        torch.testing.assert_close(output[0][:,:,:1],changed[0][:,:,:1])
        assert not torch.allclose(output[0][:,:,1:],changed[0][:,:,1:])
    else:assert output[0] is None
    output[1].square().mean().backward()
    assert z.grad is None and a.goal_adapter.projection.weight.grad.abs().sum()>0
    assert a.goal_adapter.row_position.shape==(1,12,1,512)
    assert a.goal_adapter.column_position.shape==(1,1,10,512)
    for position in (a.goal_adapter.row_position,a.goal_adapter.column_position):
        assert position.grad is not None and torch.isfinite(position.grad).all()
        assert position.grad.abs().sum()>0
    assert not a.goal_adapter.row_position.grad[:,8:].any()
    assert all(p.grad is None for p in encoder.parameters())
    assert not hasattr(a.goal_adapter,'type_embedding')
    state=a.goal_adapter.state_dict();clone=copy.deepcopy(a.goal_adapter)
    clone.load_state_dict(state,strict=True)
    for u,v in zip(a.goal_adapter(z,a.video_backbone),clone(z,a.video_backbone)):
        torch.testing.assert_close(u,v,rtol=0,atol=0)
    assert not any('norm_k' in key for key in state)


def test_visual_contracts_and_native_configure(tmp_path):
    full,encoder,path=load_encoder(tmp_path);a=make_arch();a.video_backbone.vae=full.codec.vae
    cfg=config();cfg.update(checkpoint=str(path),precomputed_text_fingerprint='a'*64)
    runtime.configure_goal(a,dict(goal=cfg,model=dict(goal_representation=cfg['representation'],query_history={'enabled':True})))
    assert a.goal_adapter.projection.in_features==48
    assert not a.goal_expert.training
    runtime.validate_goal_compatibility(cfg,cfg)
    with pytest.raises(ValueError,match='dimensions'):
        runtime.validate_goal_compatibility(cfg,{**cfg,'visual_history':dict(dim=96)})
    with pytest.raises(ValueError,match='representation|format'):
        runtime.goal_representation({**cfg,'format':'openwam.goal.query.v1'})
    with pytest.raises(ValueError,match='representation'):
        runtime.configure_goal_adapter(a,dict(representation='query_latent',format='openwam.goal.query.v1'))


@pytest.mark.parametrize('mode',['joint','none'])
def test_stage2_visual_preflight_config(mode):
    from hydra import compose, initialize_config_dir
    from pathlib import Path
    from openwam.train.utils.stage2 import validate_stage2_config
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[2]/'configs'),version_base=None):
        cfg=compose(config_name='goal/stage2_query_sample')
    with open_dict(cfg):
        cfg.goal=config()
        cfg.model.goal_representation='visual_history_masked_sa'
        cfg.model.query_history=dict(enabled=True)
        cfg.model.architecture.future_video_mode=mode
        cfg.dataloader.history=dict(enabled=True,num_frames=10,mean_interval_s=1.,jitter_fraction=0.)
    validate_stage2_config(cfg)
    cfg.goal.conditioning='query_memory'
    with pytest.raises(ValueError,match='conditioning'):validate_stage2_config(cfg)


def test_visual_unified_checkpoint_and_asset_relocation(tmp_path):
    import shutil
    from safetensors.torch import load_file
    full,encoder,path=load_encoder(tmp_path)
    a=make_arch();a.video_backbone.vae=full.codec.vae
    cfg=OmegaConf.create(dict(goal={**config(),'checkpoint':str(path),'precomputed_text_fingerprint':'a'*64},
        model=dict(query_history={'enabled':True},goal_representation='visual_history_masked_sa')))
    runtime.configure_goal(a,cfg)
    obs,ctx,_,views=multiview_batch(1);h=history()
    def condition(arch):
        return runtime.prepare_goal_inputs(arch,None,['move'],ctx,head_valid=[True],seeds=[17],
            goal_observed_multiview=obs,goal_view_valid=views,**h)
    before=condition(a);saved=tmp_path/'saved';saved.mkdir()
    runtime.save_goal_assets(a,saved,cfg)
    a.save_checkpoint(str(saved/'checkpoint_step_1.safetensors'))
    unified=load_file(str(saved/'checkpoint_step_1.safetensors'))
    assert not any(key.startswith('goal_expert.') for key in unified)
    assert any(key.startswith('goal_adapter.projection.') for key in unified)
    path.unlink();moved=tmp_path/'relocated';shutil.move(saved,moved)
    b=make_arch();b.video_backbone.vae=type(full.codec.vae)()
    runtime.configure_goal_adapter(b,dict(cfg.goal))
    b.load_checkpoint(str(moved/'checkpoint_step_1.safetensors'))
    runtime.configure_goal(b,cfg,checkpoint_dir=moved)
    after=condition(b)
    for key in before:torch.testing.assert_close(before[key],after[key],rtol=0,atol=0)
    x=inputs()
    for u,v in zip(a(**x,**before),b(**x,**after)):
        torch.testing.assert_close(u,v,rtol=0,atol=0)
    assert runtime.goal_prediction_policy(b)['format']=='openwam.visual_history.v1'


@pytest.mark.parametrize('codec_mode',['absent','correct','bad_shape','bad_range'])
def test_visual_preflight_accepts_legacy_missing_codec_and_validates_present_codec(tmp_path,monkeypatch,codec_mode):
    import json
    from dataclasses import asdict
    from safetensors.torch import save_file
    from tests.dataloader.test_curated_stage2 import make_native_source
    from hydra import compose, initialize_config_dir
    from pathlib import Path
    from openwam.model.goal.codec import GoalCodecSpec
    from openwam.train.utils.stage2 import validate_stage2_launch
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[2]/'configs'),version_base=None):
        cfg=compose(config_name='goal/stage2_query_sample')
    with open_dict(cfg):
        cfg.goal=config();cfg.goal.preflight=True
        cfg.model.goal_representation='visual_history_masked_sa'
        cfg.model.query_history=dict(enabled=True)
        cfg.dataloader.history=dict(enabled=True,num_frames=10,mean_interval_s=1.,jitter_fraction=0.)
    selection=make_native_source(tmp_path)
    metadata=dict(format='openwam.visual_history.v1',dit_config=dict(dim=48,patch_size=[1,2,2]),
        history=dict(num_history_tokens=32),dependencies={'test':'identity'})
    if codec_mode!='absent':metadata['codec']=asdict(GoalCodecSpec())
    if codec_mode=='bad_shape':metadata['codec']['latent_shape']=(48,1,24,20)
    if codec_mode=='bad_range':metadata['codec']['pixel_range']=(0.,1.)
    goal=tmp_path/'goal.safetensors'
    save_file({'dit.dummy':torch.zeros(1)},str(goal),metadata={'visual_history':json.dumps(metadata)})
    baseline=tmp_path/'baseline';baseline.mkdir()
    OmegaConf.save({'model':cfg.model},baseline/'config.yaml')
    save_file({'action_backbone.action_encoder.weight':torch.zeros(8,112),
        'action_backbone.action_decoder.weight':torch.zeros(112,8),
        'action_backbone.action_decoder.bias':torch.zeros(112),
        'proprio_encoder.weight':torch.zeros(8,112)},str(baseline/'checkpoint_step_1.safetensors'))
    with open_dict(cfg):
        cfg.goal.checkpoint=str(goal)
    cfg.training.finetune_ckpt_path=str(baseline)
    cfg.dataloader.source_yaml=str(selection);cfg.project.output_dir=str(tmp_path/'run')
    monkeypatch.setattr(torch.cuda,'_lazy_init',lambda:pytest.fail('GPU initialized'))
    if codec_mode.startswith('bad'):
        with pytest.raises(ValueError,match='codec mismatch'):validate_stage2_launch(cfg,metadata_only=True)
    else:
        report=validate_stage2_launch(cfg,metadata_only=True)
        assert report['metadata_only']


def test_visual_memory_positions_preserve_canvas_layout_under_patch_permutation():
    from openwam.model.goal.conditioning import VisualHistoryMemoryAdapter
    a=make_arch();adapter=VisualHistoryMemoryAdapter(a.video_backbone,dim=48)
    tokens=torch.randn(1,120,48)
    keys,values=adapter(tokens,a.video_backbone)
    permuted_keys,permuted_values=adapter(tokens.flip(1),a.video_backbone)
    query=torch.randn(1,3,48)
    def read(k,v):
        return (query@k[0].transpose(-1,-2)/48**.5).softmax(-1)@v[0]
    # Without explicit positions this read is invariant to patch permutation.
    assert not torch.allclose(read(keys,values),read(permuted_keys,permuted_values),rtol=1e-5,atol=1e-6)
    with torch.no_grad():adapter.row_position.zero_();adapter.column_position.zero_()
    keys,values=adapter(tokens,a.video_backbone)
    permuted_keys,permuted_values=adapter(tokens.flip(1),a.video_backbone)
    torch.testing.assert_close(read(keys,values),read(permuted_keys,permuted_values),rtol=1e-5,atol=1e-6)
