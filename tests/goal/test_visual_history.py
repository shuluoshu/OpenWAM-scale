import copy
import pytest
import torch

from test_expert import TINY_DIT, TinyVAE
from test_multiview_patch import multiview_batch
from openwam.model.goal.codec import GoalCodec
from openwam.model.goal.patch_expert import PatchGoalExpert
from openwam.model.video_backbone.wan.models.dit import WanModel


def make_model():
    from openwam.model.goal.visual_history import VisualHistoryGoalExpert
    torch.manual_seed(123)
    return VisualHistoryGoalExpert(WanModel(**TINY_DIT), GoalCodec(TinyVAE()),
        dit_config=TINY_DIT, num_history_tokens=5, pool_dim=32, pool_heads=4)


def history(b=1):
    return dict(history_latents=torch.randn(b,10,48,1,16,20),
        history_ages=torch.arange(10,0,-1).float()[None].expand(b,-1).clone(),
        history_valid=torch.ones(b,10,dtype=torch.bool))


def test_zero_gate_exact_native_equivalence_and_total_history_budget():
    e=make_model().eval(); obs,ctx,_,views=multiview_batch(1); h=history()
    base=PatchGoalExpert(copy.deepcopy(e.dit), e.codec, dit_config=TINY_DIT, observation_mode='multiview').eval()
    c=e.condition_tokens(obs,observed_view_valid=views,**h)
    z=e.encode_observation(obs,observed_view_valid=views)
    torch.testing.assert_close(c,e.dit.patchify(z).flatten(2).transpose(1,2),rtol=0,atol=0)
    captured=[]
    handle=e.history_pool.register_forward_hook(lambda m,i,o:captured.append(o))
    e.condition_tokens(obs,observed_view_valid=views,**h);handle.remove()
    assert captured[0].shape==(1,5,TINY_DIT['dim'])
    x=torch.randn(1,48,1,16,20); p=torch.randn(1,192);t=torch.tensor([500.])
    a=e.velocity(z,x,p,t,ctx,observed_view_valid=views,observation_tokens=c)
    b=base.velocity(z,x,p,t,ctx,observed_view_valid=views)
    for aa,bb in zip(a,b):torch.testing.assert_close(aa,bb,rtol=0,atol=0)


def test_valid_history_age_sensitive_invalid_nan_invariant_empty_fallback():
    e=make_model().eval();e.history_gate.data.fill_(.2)
    obs,_,_,views=multiview_batch(1);h=history();h['history_valid'][:,:3]=False
    a=e.condition_tokens(obs,observed_view_valid=views,**h)
    h['history_latents'][:,:3]=float('nan');h['history_ages'][:,:3]=float('nan')
    torch.testing.assert_close(a,e.condition_tokens(obs,observed_view_valid=views,**h),rtol=0,atol=0)
    h['history_ages'][:,3:]*=2
    b=e.condition_tokens(obs,observed_view_valid=views,**h)
    assert not torch.allclose(a,b)
    h['history_latents'][:,3:]+=1
    assert not torch.allclose(b,e.condition_tokens(obs,observed_view_valid=views,**h))
    h['history_valid'].zero_();h['history_latents'].fill_(float('nan'))
    c=e.condition_tokens(obs,observed_view_valid=views,**h)
    native=e.dit.patchify(e.encode_observation(obs,observed_view_valid=views)).flatten(2).transpose(1,2)
    torch.testing.assert_close(c,native,rtol=0,atol=0)


def test_mask_isolates_current_from_target_and_time_text_padding_safe():
    e=make_model().eval();e.history_gate.data.fill_(.2)
    obs,ctx,_,views=multiview_batch(1);c=e.condition_tokens(obs,observed_view_valid=views,**history())
    z=e.encode_observation(obs,observed_view_valid=views)
    def run(x,t,context):
        return e.velocity(z,x,torch.zeros(1,192),torch.tensor([t]),context,
            observed_view_valid=views,observation_tokens=c,context_lens=torch.tensor([2]),return_observation=True)
    x=torch.randn(1,48,1,16,20);a=run(x,500.,ctx);b=run(x+2,100.,ctx)
    torch.testing.assert_close(a[2],b[2],rtol=0,atol=0)
    padded=ctx.clone();padded[:,2:]=float('nan')
    for aa,bb in zip(a,run(x,500.,padded)):torch.testing.assert_close(aa,bb,rtol=0,atol=0)
    padded[:,:2]+=1
    assert not torch.allclose(a[0],run(x,500.,padded)[0])


def test_gradients_native_text_pool_adapter_and_strict_roundtrip():
    e=make_model().train_stage_a();e.history_gate.data.fill_(.2)
    obs,ctx,target,views=multiview_batch(1);h=history()
    e(obs,ctx,target,observed_view_valid=views,**h)['loss'].backward()
    for name,params in [('pool',e.history_pool.parameters()),('adapter',e.visual_adapter.parameters()),('text',e.dit.text_embedding.parameters())]:
        grads=[p.grad for p in params if p.requires_grad and p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads),name
        assert sum(float(g.abs().sum()) for g in grads)>0,name
    e.eval();pred=e.generate(obs,ctx,observed_view_valid=views,num_steps=2,seed=7,**h)
    clone=make_model().eval();clone.load_state_dict(e.state_dict(),strict=True)
    restored=clone.generate(obs,ctx,observed_view_valid=views,num_steps=2,seed=7,**h)
    torch.testing.assert_close(pred.latent,restored.latent,rtol=0,atol=0)
    assert torch.isfinite(pred.uv).all()


def test_history_contract_rejects_future_time_and_unsupported_text():
    e=make_model();obs,ctx,_,views=multiview_batch(1);h=history();h['history_ages'][:,-1]=0
    with pytest.raises(ValueError,match='history'):
        e.condition_tokens(obs,observed_view_valid=views,**h)
    with pytest.raises(TypeError):
        e.condition_tokens(obs,context=ctx,observed_view_valid=views,**history())


def test_rgb_history_matches_cached_latents_with_and_without_episode_id():
    e=make_model().eval();e.history_gate.data.fill_(.2)
    obs,_,_,views=multiview_batch(1);h=history();h.pop('history_latents')
    images=torch.rand(1,10,3,1,256,320)*2-1
    latents=torch.stack([e.codec.encode(images[:,j]) for j in range(10)],dim=1)
    expected=e.condition_tokens(obs,observed_view_valid=views,history_latents=latents,**h)
    for episode in (None,torch.tensor([123],dtype=torch.int64)):
        actual=e.condition_tokens(obs,observed_view_valid=views,history_images=images,history_episode_id=episode,**h)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)


def test_mixed_empty_batch_and_checkpoint_gradients():
    e=make_model().train_stage_a();e.history_gate.data.fill_(.2);e.use_gradient_checkpointing=True
    obs,ctx,t,views=multiview_batch(2);h=history(2)
    h['history_valid'][0]=False;h['history_latents'][0]=float('nan');h['history_ages'][0]=float('nan')
    c=e.condition_tokens(obs,observed_view_valid=views,**h)
    native=e.dit.patchify(e.encode_observation(obs,observed_view_valid=views)).flatten(2).transpose(1,2)
    torch.testing.assert_close(c[0],native[0],rtol=0,atol=0)
    assert not torch.allclose(c[1],native[1])
    out=e(obs,ctx,t,observed_view_valid=views,**h);out['loss'].backward()
    assert e.history_pool.queries.grad.abs().sum()>0


def test_portable_bundle_checks_dependencies_and_freezes_on_load(tmp_path):
    from openwam.train.goal_visual_history import save_visual_history_bundle, load_visual_history_bundle
    e=make_model().eval();e.history_gate.data.fill_(.2)
    path=tmp_path/'model.safetensors'
    save_visual_history_bundle(e,path,precomputed_text_fingerprint='a'*64)
    restored=load_visual_history_bundle(path,vae=e.codec.vae,text_dim=16,precomputed_text_fingerprint='a'*64)
    assert not restored.training and not any(p.requires_grad for p in restored.parameters())
    for k,v in e.state_dict().items():torch.testing.assert_close(v,restored.state_dict()[k],rtol=0,atol=0)
    with pytest.raises(ValueError,match='text dependency'):
        load_visual_history_bundle(path,vae=e.codec.vae,text_dim=16,precomputed_text_fingerprint='b'*64)
    bad_vae=TinyVAE();bad_vae.marker+=1
    with pytest.raises(ValueError,match='VAE dependency'):
        load_visual_history_bundle(path,vae=bad_vae,text_dim=16,precomputed_text_fingerprint='a'*64)
