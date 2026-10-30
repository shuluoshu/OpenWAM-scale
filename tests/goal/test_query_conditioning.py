"""Condition interface ablation contracts on real tiny Wan blocks."""
import pytest
import torch
from test_query_goal import QUERY
from test_patch_expert import make_expert
from test_multiview_patch import multiview_batch

MODES = ('legacy_text', 'normalized_cross_attention', 'prefix_self_attention')

def build(mode):
    from openwam.model.goal.query_expert import QueryGoalExpert
    b = make_expert()
    return QueryGoalExpert(b.dit, b.codec, dit_config=b.dit_config, query_config=QUERY,
                           conditioning_mode=mode)

@pytest.mark.parametrize('mode', MODES)
def test_finite_joint_gradients_and_no_unused_trainable_parameters(mode):
    e = build(mode).train_stage_a()
    e.encoder.use_gradient_checkpointing = True
    e.decoder.use_gradient_checkpointing = True
    obs, ctx, target, valid = multiview_batch(1)
    result = e(obs, ctx, target, observed_view_valid=valid)
    result['loss'].backward()
    for name, p in e.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert e.encoder.image_queries.grad.abs().sum() > 0
    assert e.encoder.tcp_queries.grad.abs().sum() > 0
    assert not hasattr(e, 'readout')

@pytest.mark.parametrize('mode', MODES[1:])
def test_new_modes_bypass_wan_text_mlp(mode, monkeypatch):
    e = build(mode).eval()
    monkeypatch.setattr(e.dit.text_embedding, 'forward', lambda *a: pytest.fail('old text MLP used'))
    if mode == 'prefix_self_attention':
        for b in e.dit.blocks:
            monkeypatch.setattr(b.cross_attn, 'forward', lambda *a, **k: pytest.fail('cross-attention used'))
    obs, ctx, target, valid = multiview_batch(1)
    z, blank, condition = e._condition(obs, ctx, observed_view_valid=valid)
    assert condition.shape == (1, 10, e.dit.dim)
    a = e.generate(obs, ctx, observed_view_valid=valid, num_steps=2, seed=7)
    b = e.generate(obs * .1, ctx + .5, observed_view_valid=valid, num_steps=2, seed=7)
    assert torch.isfinite(a.latent).all() and torch.isfinite(a.uv).all()
    assert not torch.allclose(a.latent, b.latent)


def test_prefix_is_clean_causal_and_goal_tcp_are_mutual():
    e = build('prefix_self_attention').eval()
    obs, ctx, _, valid = multiview_batch(1)
    _, blank, c = e._condition(obs, ctx, observed_view_valid=valid)
    ob = e.decoder.encode_observation(blank)
    captured = []
    hook = e.dit.blocks[-1].register_forward_hook(lambda m,a,o: captured.append(o.detach()))
    times = torch.tensor([700.])
    x = torch.zeros_like(ob); tcp = torch.zeros(1,192)
    a = e.decoder.velocity(ob, x, tcp, times, c)
    b = e.decoder.velocity(ob, x+1, tcp, times, c)
    d = e.decoder.velocity(ob, x, tcp+1, times, c)
    hook.remove()
    assert captured[0].shape[1] == 10+80+1
    torch.testing.assert_close(captured[0][:,:10], captured[1][:,:10], rtol=0, atol=0)
    torch.testing.assert_close(captured[0][:,:10], captured[2][:,:10], rtol=0, atol=0)
    assert not torch.allclose(a[1], b[1])
    assert not torch.allclose(a[0], d[0])

@pytest.mark.parametrize('mode', MODES)
def test_bundle_mode_roundtrip_and_stage2_encoder_only(mode, tmp_path):
    from openwam.train.goal_query import save_query_bundle, load_query_bundle
    e=build(mode).eval(); obs,ctx,_,valid=multiview_batch(1)
    p=tmp_path/'model.safetensors'
    meta=save_query_bundle(e,p,precomputed_text_fingerprint='a'*64)
    assert meta['conditioning_mode']==mode
    full=load_query_bundle(p,vae=e.codec.vae,text_dim=16,precomputed_text_fingerprint='a'*64,encoder_only=False)
    assert full.conditioning_mode==mode
    a=e.generate(obs,ctx,observed_view_valid=valid,num_steps=2,seed=2)
    b=full.generate(obs,ctx,observed_view_valid=valid,num_steps=2,seed=2)
    torch.testing.assert_close(a.latent,b.latent,rtol=0,atol=0)
    torch.testing.assert_close(a.uv,b.uv,rtol=0,atol=0)
    enc=load_query_bundle(p,vae=e.codec.vae,text_dim=16,precomputed_text_fingerprint='a'*64)
    with torch.no_grad():
        torch.testing.assert_close(e.extract_tokens(obs,ctx,observed_view_valid=valid),
                                   enc.extract_tokens(obs,ctx,observed_view_valid=valid),rtol=0,atol=0)


def test_invalid_mode_rejected():
    with pytest.raises(ValueError, match='conditioning'):
        build('unknown')

@pytest.mark.parametrize('mode', MODES)
def test_curated_trainer_thaws_before_optimizer_and_excludes_inactive_weights(mode, tmp_path, monkeypatch):
    from openwam.train.goal_curated import train_curated
    import openwam.train.goal_overfit as overfit
    import openwam.train.goal_patch  # Bind its module-level optimizer before this scoped patch.
    e = build(mode).freeze()
    obs,ctx,target,valid = multiview_batch(1)
    before = next(e.condition_projection.parameters()).detach().clone()
    class Data:
        def sample(self, step):
            return dict(id='subtask0',family='tiny',instruction='put back',granularity='subtask'),dict(
                observed=obs,context=ctx,targets=target,observed_view_valid=valid,sources=['tiny'])
        def evaluation_samples(self): return [self.sample(0)]
        def scope_summary(self): return dict(single_subtask=True)
    real = overfit.FP32MasterAdamW
    def checked(groups):
        params = {id(p) for g in groups for p in g['params']}
        expected = {id(p) for p in e.parameters() if p.requires_grad}
        assert params and params == expected, 'thaw and freeze inactive branches before optimizer'
        assert next(g for g in groups if g['name']=='condition')['lr']==1e-4
        return real(groups)
    monkeypatch.setattr(overfit,'FP32MasterAdamW',checked)
    result=train_curated(e,dataset=Data(),context_cache=None,output_dir=tmp_path/'run',
        steps=1,eval_every=1,sampling_steps=2,learning_rate=.001,condition_learning_rate=1e-4,precomputed_text_fingerprint='a'*64)
    assert result['completed_steps']==1
    assert not torch.equal(before,next(e.condition_projection.parameters()).detach())


def test_explicit_condition_lr_groups_cover_only_active_weights_once():
    from openwam.train.goal_curated import goal_parameter_groups
    e=build('prefix_self_attention').train_stage_a()
    groups=goal_parameter_groups(e,1e-5,1e-4,condition_learning_rate=1e-4)
    condition=next(g for g in groups if g['name']=='condition')
    assert condition['lr']==1e-4
    assert {id(p) for p in condition['params']}=={id(p) for p in e.condition_projection.parameters()}
    params=[p for g in groups for p in g['params']]
    assert len(params)==len({id(p) for p in params})
    assert {id(p) for p in params}=={id(p) for p in e.parameters() if p.requires_grad}
    legacy=goal_parameter_groups(e,1e-5,1e-4)
    assert all(g['name']!='condition' for g in legacy)
    assert all(g['lr']==1e-5 for g in legacy if any(id(p) in {id(q) for q in e.condition_projection.parameters()} for p in g['params']))


@pytest.mark.parametrize('value',[0.,-1.,float('nan'),float('inf'),True])
def test_invalid_condition_lr_rejected(value):
    from openwam.train.goal_curated import goal_parameter_groups
    with pytest.raises(ValueError,match='condition_learning_rate'):
        goal_parameter_groups(build('prefix_self_attention'),1e-5,1e-4,condition_learning_rate=value)


@pytest.mark.parametrize('override',[
    {'model':{'query_conditioning':'typo'}},
    {'condition_learning_rate':0},
    {'condition_learning_rate':True},
    {'model':{'goal_representation':'patch_denoise'},'condition_learning_rate':1e-4},
])
def test_config_rejects_invalid_condition_contract_before_loading_weights(tmp_path,override):
    from pathlib import Path
    import yaml
    from openwam.train.goal_curated import load_config
    base=Path(__file__).resolve().parents[2]/'configs/goal/stage1_query_sample.yaml'
    cfg=tmp_path/'bad.yaml';cfg.write_text(yaml.safe_dump(dict(base_config=str(base),**override)))
    with pytest.raises(ValueError,match='condition'):
        load_config(cfg)
