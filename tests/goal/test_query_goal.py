"""Query goal contract: input-dependent tokens, no future leakage, removable heads."""
import copy
import json

import pytest
import torch

from test_patch_expert import make_expert
from test_multiview_patch import multiview_batch


QUERY = dict(num_image_tokens=8, num_tcp_tokens=2, dim=32, num_heads=4,
             num_layers=2, ffn_dim=64)


def make_query():
    from openwam.model.goal.query_expert import QueryGoalExpert
    base = make_expert()
    return QueryGoalExpert(base.dit, base.codec, dit_config=base.dit_config,
                           query_config=QUERY, observation_mode='multiview')


def test_queries_depend_on_observation_and_language_but_ignore_missing_data():
    e = make_query().eval()
    obs, ctx, _, valid = multiview_batch()
    lens = torch.tensor([2, 3])
    z = e.extract_tokens(obs, ctx, context_lens=lens, observed_view_valid=valid)
    assert z.shape == (2, 10, 32)
    assert not torch.allclose(z[0], z[1])
    changed = obs.clone(); changed[:, :, :, :256] *= .3
    assert not torch.allclose(z, e.extract_tokens(changed, ctx, context_lens=lens, observed_view_valid=valid))
    assert not torch.allclose(z, e.extract_tokens(obs, ctx + .5, context_lens=lens, observed_view_valid=valid))
    dirty = obs.clone(); dirty[0, :, :, 256:, 160:] = float('nan')
    padded = ctx.clone(); padded[0, 2:] = float('nan'); padded[1, 3:] = float('nan')
    actual = e.extract_tokens(dirty, padded, context_lens=lens, observed_view_valid=valid)
    torch.testing.assert_close(z, actual, rtol=0, atol=0)


def test_auxiliary_flow_trains_queries_without_future_leakage():
    e = make_query().train_stage_a()
    obs, ctx, target, valid = multiview_batch()
    z_before = e.extract_tokens(obs, ctx, observed_view_valid=valid).detach()
    captured = []
    h = e.encoder.register_forward_hook(lambda m, a, out: captured.append(out.detach()))
    out = e(obs, ctx, target, observed_view_valid=valid)
    h.remove()
    torch.testing.assert_close(z_before, captured[0], rtol=0, atol=0)
    assert len(captured) == 1
    out['loss'].backward()
    assert e.encoder.image_queries.grad.abs().sum() > 0
    assert e.encoder.tcp_queries.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in e.dit.parameters())
    assert all(p.grad is None for p in e.codec.vae.parameters())
    assert out['valid_points'] == 4 and torch.isfinite(out['loss'])
    # One current observation encode and one target encode; no blank-prefix VAE.
    assert e.codec.vae.calls[-2:] == [(2, 3, 1, 384, 320), (2, 3, 1, 256, 320)]
    assert len(e.codec.vae.calls) == 3  # includes z_before's observation encode


def test_invalid_tcp_nan_does_not_poison_loss():
    e = make_query()
    obs, ctx, target, valid = multiview_batch()
    target.uv[:] = float('nan'); target.point_valid[:] = False
    out = e(obs, ctx, target, observed_view_valid=valid)
    assert torch.isfinite(out['loss']) and out['valid_points'] == 0
    assert out['tcp_loss'] == 0 and out['query_tcp_loss'] == 0


def test_encoder_only_bundle_is_exact_and_never_constructs_decoder(tmp_path, monkeypatch):
    from openwam.train.goal_query import save_query_bundle, load_query_bundle
    e = make_query().freeze()
    obs, ctx, _, valid = multiview_batch()
    expected = e.extract_tokens(obs, ctx, observed_view_valid=valid).detach()
    path = tmp_path / 'query.safetensors'
    save_query_bundle(e, path, precomputed_text_fingerprint='a' * 64)
    import openwam.train.goal_query as bundle
    monkeypatch.setattr(bundle, 'WanModel', lambda **kwargs: pytest.fail('decoder loaded for Stage2'))
    loaded = load_query_bundle(path, vae=e.codec.vae, text_dim=16,
        precomputed_text_fingerprint='a' * 64, encoder_only=True)
    assert not hasattr(loaded, 'decoder') and not hasattr(loaded, 'dit')
    assert set(loaded.state_dict()) == {'encoder.' + n for n in e.encoder.state_dict()}
    assert not any(p.requires_grad for p in loaded.parameters())
    torch.testing.assert_close(expected, loaded.extract_tokens(obs, ctx, observed_view_valid=valid), rtol=0, atol=0)
    with pytest.raises(ValueError, match='text'):
        load_query_bundle(path, vae=e.codec.vae, text_dim=16,
            precomputed_text_fingerprint='b' * 64, encoder_only=True)


def test_full_bundle_roundtrip_preserves_joint_predictions(tmp_path):
    from openwam.train.goal_query import save_query_bundle, load_query_bundle
    e = make_query().eval(); obs, ctx, _, valid = multiview_batch(1)
    path = tmp_path / 'query.safetensors'
    save_query_bundle(e, path, precomputed_text_fingerprint='a' * 64)
    restored = load_query_bundle(path, vae=e.codec.vae, text_dim=16,
        precomputed_text_fingerprint='a' * 64, encoder_only=False)
    a = e.generate(obs, ctx, seed=7, num_steps=2, observed_view_valid=valid)
    b = restored.generate(obs, ctx, seed=7, num_steps=2, observed_view_valid=valid)
    torch.testing.assert_close(a.latent, b.latent, rtol=0, atol=0)
    torch.testing.assert_close(a.uv, b.uv, rtol=0, atol=0)


@pytest.mark.parametrize('changes', [dict(num_image_tokens=0), dict(num_tcp_tokens=0),
    dict(dim=31), dict(num_layers=0), dict(unrecognized=1)])
def test_bad_query_settings_rejected(changes):
    from openwam.model.goal.query_expert import QueryConfig
    with pytest.raises((ValueError, TypeError)):
        QueryConfig(**{**QUERY, **changes})
