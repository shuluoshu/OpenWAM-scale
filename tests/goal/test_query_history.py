import copy
from dataclasses import asdict

import pytest
import torch

from test_query_goal import make_query, QUERY
from test_multiview_patch import multiview_batch


def history_expert():
    from openwam.model.goal.query_expert import QueryGoalExpert
    base = make_query()
    return QueryGoalExpert(base.dit, base.codec, dit_config=base.dit_config,
        query_config=QUERY, observation_mode='multiview',
        history_config=dict(enabled=True, compressor_dim=32))


def history(batch=1):
    return dict(history_latents=torch.randn(batch,10,48,1,16,20),
        history_ages=torch.arange(10,0,-1).float().expand(batch,-1).clone(),
        history_valid=torch.ones(batch,10,dtype=torch.bool))


def test_history_affects_fixed_queries_and_encodes_real_time():
    e = history_expert().eval(); obs,ctx,_,views = multiview_batch(1); h=history()
    a=e.extract_tokens(obs,ctx,observed_view_valid=views,**h)
    assert a.shape == (1,10,32)
    h['history_ages'] *= 1.7
    b=e.extract_tokens(obs,ctx,observed_view_valid=views,**h)
    assert not torch.allclose(a,b)
    h['history_latents'] += 2
    assert not torch.allclose(b,e.extract_tokens(obs,ctx,observed_view_valid=views,**h))


def test_padding_nan_and_empty_history_cannot_affect_queries():
    e=history_expert().eval(); obs,ctx,_,views=multiview_batch(1); h=history()
    h['history_valid'][:,:4]=False
    a=e.extract_tokens(obs,ctx,observed_view_valid=views,**h)
    h['history_latents'][:,:4]=float('nan'); h['history_ages'][:,:4]=float('nan')
    torch.testing.assert_close(a,e.extract_tokens(obs,ctx,observed_view_valid=views,**h),rtol=0,atol=0)
    h['history_valid'][:]=False
    assert torch.isfinite(e.extract_tokens(obs,ctx,observed_view_valid=views,**h)).all()
    with pytest.raises(ValueError,match='history'):
        e.extract_tokens(obs,ctx,observed_view_valid=views)


def test_bad_times_rejected_and_disabled_model_rejects_unconsumed_history():
    e=history_expert(); obs,ctx,_,views=multiview_batch(1); h=history()
    for ages in (torch.zeros(1,10),torch.arange(10).float()[None]):
        with pytest.raises(ValueError,match='history'):
            e.extract_tokens(obs,ctx,observed_view_valid=views,**{**h,'history_ages':ages})
    with pytest.raises(ValueError,match='disabled'):
        make_query().extract_tokens(obs,ctx,observed_view_valid=views,**h)


def test_detached_features_compressor_fp32_and_nonzero_gradients():
    e=history_expert().to(dtype=torch.bfloat16).train_stage_a()
    assert all(p.dtype==torch.float32 for p in e.encoder.history.parameters())
    # The test-only TinyVAE uses CPU avg_pool3d, which has no BF16 kernel.
    e.float()
    obs,ctx,_,views=multiview_batch(1); h=history(); h['history_latents'].requires_grad_(True)
    z=e.extract_tokens(obs,ctx,observed_view_valid=views,**h)
    z.float().square().mean().backward()
    assert h['history_latents'].grad is None
    grads=[p.grad for p in e.encoder.history.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum() for g in grads)>0
    assert not any(p.grad is not None for p in e.codec.vae.parameters())


def test_cached_images_equal_live_features_and_cache_bounded():
    e=history_expert().eval(); obs,ctx,_,views=multiview_batch(1)
    images=torch.randn(1,10,3,1,256,320).clamp(-1,1); h=history(); h.pop('history_latents')
    calls=[]; encode=e.codec.encode
    def wrapped(x): calls.append(len(x)); return encode(x)
    e.codec.encode=wrapped
    a=e.extract_tokens(obs,ctx,observed_view_valid=views,history_images=images,**h)
    first=sum(calls)
    b=e.extract_tokens(obs,ctx,observed_view_valid=views,history_images=images,**h)
    assert sum(calls)==first
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    latents=torch.stack([encode(images[:,i]) for i in range(10)],dim=1)
    c=e.extract_tokens(obs,ctx,observed_view_valid=views,history_latents=latents,**h)
    torch.testing.assert_close(a,c,rtol=0,atol=0)
    e.clear_history_cache()
    assert len(e._history_cache)==0


def test_history_bundle_roundtrip_encoder_only_and_full(tmp_path):
    from openwam.train.goal_query import save_query_bundle,load_query_bundle
    e=history_expert().eval(); obs,ctx,_,views=multiview_batch(1); h=history()
    path=tmp_path/'query.safetensors'
    save_query_bundle(e,path,precomputed_text_fingerprint='a'*64)
    expected=e.freeze().extract_tokens(obs,ctx,observed_view_valid=views,**h)
    for encoder_only in (True,False):
        restored=load_query_bundle(path,vae=e.codec.vae,text_dim=16,
            precomputed_text_fingerprint='a'*64,encoder_only=encoder_only)
        assert restored.history_config==e.history_config
        assert all(not p.requires_grad for p in restored.parameters())
        torch.testing.assert_close(expected,restored.extract_tokens(obs,ctx,observed_view_valid=views,**h),rtol=0,atol=0)


def test_forward_joint_loss_trains_compressor_and_generate_accepts_history():
    e=history_expert().train_stage_a(); obs,ctx,targets,views=multiview_batch(1); h=history()
    out=e(obs,ctx,targets,observed_view_valid=views,**h)
    out['loss'].backward()
    assert e.encoder.history.output.weight.grad.abs().sum()>0
    pred=e.eval().generate(obs,ctx,observed_view_valid=views,num_steps=2,seed=3,**h)
    assert torch.isfinite(pred.uv).all()


def test_history_optimizer_group_updates_first_step():
    from openwam.train.goal_curated import goal_parameter_groups
    from openwam.train.goal_overfit import FP32MasterAdamW
    e=history_expert().train_stage_a(); obs,ctx,targets,views=multiview_batch(1)
    groups=goal_parameter_groups(e,1e-5,1e-4)
    assert [(g['name'],g['lr']) for g in groups]==[('goal',1e-5),('history',1e-4)]
    ids=[id(p) for g in groups for p in g['params']]
    assert len(ids)==len(set(ids))==len(list(e.parameters()))
    before=e.encoder.history.input.weight.detach().clone()
    optimizer=FP32MasterAdamW(groups)
    e(obs,ctx,targets,observed_view_valid=views,**history())['loss'].backward()
    optimizer.step()
    assert not torch.equal(before,e.encoder.history.input.weight)


def test_history_export_preserves_actual_pre_vae_pixels_and_ages(tmp_path):
    from openwam.train.goal_overfit import export_inputs
    import numpy as np
    obs,ctx,target,views=multiview_batch(1)
    h=history(); h.pop('history_latents')
    h['history_images']=torch.zeros(1,10,3,1,256,320,dtype=torch.bfloat16)
    batch=dict(observed=obs,context=ctx,targets=target,observed_view_valid=views,**h)
    export_inputs(tmp_path,[dict(family='test',instruction='a subtask')],[batch])
    data=np.load(tmp_path/'inputs/case_0000.npz')
    assert data['history_images'].shape==(1,10,3,1,256,320)
    np.testing.assert_array_equal(data['history_ages'],h['history_ages'].numpy())
    assert 'history_images_bfloat16_bits' in data
    assert 'history_09.png' in (tmp_path/'inputs/index.html').read_text()


def test_compressor_parent_bfloat16_preserves_exact_fp32_values():
    e=history_expert()
    before={k:v.clone() for k,v in e.encoder.history.state_dict().items()}
    e.bfloat16().to(device='cpu')
    for key,value in e.encoder.history.state_dict().items():
        assert value.dtype==torch.float32
        assert torch.equal(value,before[key]), key


def test_history_bundle_bfloat16_load_preserves_compressor_weights(tmp_path):
    from openwam.train.goal_query import save_query_bundle,load_query_bundle
    e=history_expert(); before={k:v.clone() for k,v in e.encoder.history.state_dict().items()}
    path=tmp_path/'history.safetensors'
    e.codec.vae.bfloat16()
    save_query_bundle(e,path,precomputed_text_fingerprint='a'*64)
    loaded=load_query_bundle(path,vae=e.codec.vae,text_dim=16,dtype=torch.bfloat16,
        precomputed_text_fingerprint='a'*64)
    for key,value in loaded.encoder.history.state_dict().items():
        assert value.dtype==torch.float32
        assert torch.equal(value,before[key]), key


def test_head_feature_cache_eviction_and_vae_parameter_version_invalidation():
    from types import SimpleNamespace
    from openwam.model.goal.history import HeadFeatureCache
    vae=torch.nn.Linear(1,1); calls=[]
    def encode(images):
        calls.append(1)
        return images.new_full((len(images),48,1,16,20),float(vae.weight.detach().item()))
    codec=SimpleNamespace(vae=vae,encode=encode); cache=HeadFeatureCache(2)
    valid=torch.ones(1,1,dtype=torch.bool)
    images=[torch.full((1,1,3,1,256,320),v) for v in (0.,.2,.4)]
    for image in images: cache.encode(codec,image,valid)
    assert len(cache)==2 and len(calls)==3
    cache.encode(codec,images[2],valid)
    assert len(calls)==3
    cache.encode(codec,images[0],valid)
    assert len(cache)==2 and len(calls)==4
    before=cache.encode(codec,images[0],valid)
    with torch.no_grad(): vae.weight.add_(1)
    after=cache.encode(codec,images[0],valid)
    assert len(cache)==1 and len(calls)==5
    assert not torch.equal(before,after)
