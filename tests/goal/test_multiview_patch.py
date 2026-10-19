"""Multiview observation uses real tiny Wan blocks and the head-only goal codec."""
import json
import pytest
import torch
from PIL import Image

from test_expert import batch, TinyVAE
from test_patch_expert import make_expert


def multiview_batch(n=2):
    head, context, targets = batch(n)
    canvas = torch.cat((head, head[:, :, :, :128].clone()), dim=3)
    valid = torch.tensor([[True, True, False], [True, False, True]])[:n]
    return canvas, context, targets, valid


def test_clean_multiview_layout_bilinear_and_explicit_missing_slots():
    from openwam.model.goal.observation import clean_multiview, sanitize_canvas, observation_token_mask
    frames = [Image.new('RGB', (19, 17), c) for c in ('red', 'green', 'blue')]
    mask = torch.tensor([True, True, False])
    clean = clean_multiview(frames, mask)
    assert clean.shape == (3, 1, 384, 320)
    assert torch.equal(clean[:, 0, 0, 0], torch.tensor([1., -1., -1.]))
    assert (clean[:, :, 256:, 160:] == -1).all()
    dirty = clean[None].clone(); dirty[:, :, :, 256:, 160:] = float('nan')
    assert torch.equal(sanitize_canvas(dirty, mask[None]), clean[None])
    tokens = observation_token_mask(mask[None])
    expected = torch.ones(12, 10, dtype=torch.bool); expected[8:, 5:] = False
    assert torch.equal(tokens, expected.flatten()[None])
    with pytest.raises(ValueError, match='bool'):
        observation_token_mask(torch.ones(1, 3))
    with pytest.raises(ValueError, match='valid'):
        sanitize_canvas(clean[None], torch.zeros(1, 3, dtype=torch.bool))
    with pytest.raises(ValueError, match='finite'):
        sanitize_canvas(dirty, torch.ones(1, 3, dtype=torch.bool))


def test_unequal_grids_mask_rope_and_head_only_outputs():
    e = make_expert(observation_mode='multiview')
    obs, ctx, targets, valid = multiview_batch()
    calls, block_args, outputs = [], [], []
    ph = e.dit.patch_embedding.register_forward_pre_hook(lambda m, args: calls.append(args[0].shape))
    bh = e.dit.blocks[0].register_forward_pre_hook(lambda m, args, kwargs: block_args.append((args, kwargs)), with_kwargs=True)
    hh = e.dit.head.register_forward_hook(lambda m, args, out: outputs.append(out))
    out = e(obs, ctx, targets, observed_view_valid=valid, timestep_indices=torch.tensor([200, 700]))
    ph.remove(); bh.remove(); hh.remove()
    assert e.codec.vae.calls == [(2, 3, 1, 384, 320), (2, 3, 1, 256, 320)]
    assert [tuple(s) for s in calls] == [(2, 48, 1, 24, 20), (2, 48, 1, 16, 20), (2, 48, 1, 2, 2)]
    assert outputs[0].shape == (2, 201, 192)
    assert out['velocity'].shape == (2, 48, 1, 16, 20)
    assert out['observed_latent'].shape == (2, 48, 1, 24, 20)
    args, kwargs = block_args[0]
    frequencies = args[3]
    def rope(t, h, w):
        return torch.cat((e.dit.freqs[0][t], e.dit.freqs[1][h], e.dit.freqs[2][w]))
    assert torch.equal(frequencies[119, 0], rope(0, 11, 9))
    assert torch.equal(frequencies[120, 0], rope(1, 0, 0))
    assert torch.equal(frequencies[199, 0], rope(1, 7, 9))
    assert torch.equal(frequencies[200, 0], rope(2, 0, 0))
    mask = kwargs['self_attn_mask']
    assert mask.shape == (2, 1, 201, 201) and mask.dtype == torch.bool
    assert not mask[:, :, :120, 120:].any()
    from openwam.model.goal.observation import observation_token_mask
    expected = observation_token_mask(valid)
    assert torch.equal(mask[:, 0, 120, :120], expected)
    assert mask[:, :, 120:, 120:].all()
    out['loss'].backward()
    for weight in (e.dit.patch_embedding.weight, e.dit.blocks[0].self_attn.q.weight, e.dit.head.head.weight):
        assert weight.grad.abs().sum() > 0 and torch.isfinite(weight.grad).all()
    assert not any('vae' in k for k in e.state_dict())


def test_missing_nan_pixels_invariant_valid_wrist_sensitive_batch_masks_and_seed_stable():
    e = make_expert(observation_mode='multiview').eval()
    obs, ctx, _, valid = multiview_batch()
    dirty = obs.clone()
    dirty[0, :, :, 256:, 160:] = float('nan')
    dirty[1, :, :, 256:, :160] = float('nan')
    a = e.generate(obs, ctx, observed_view_valid=valid, num_steps=2, seed=[23, 24])
    b = e.generate(dirty, ctx, observed_view_valid=valid, num_steps=2, seed=[23, 24])
    assert torch.equal(a.latent, b.latent) and torch.equal(a.tcp_patch, b.tcp_patch)
    assert a.latent.shape == (2, 48, 1, 16, 20) and a.token_valid.shape == (2, 81)
    assert a.token_valid.all() and a.uv.shape == (2, 2, 2)
    changed = obs.clone(); changed[0, :, :, 256:, :160] = 1
    c = e.generate(changed, ctx, observed_view_valid=valid, num_steps=2, seed=[23, 24])
    assert not torch.allclose(a.latent[0], c.latent[0])
    assert not torch.allclose(a.tcp_patch[0], c.tcp_patch[0])
    assert torch.equal(a.latent[1], c.latent[1])
    right_changed = obs.clone(); right_changed[1, :, :, 256:, 160:] = -1
    right = e.generate(right_changed, ctx, observed_view_valid=valid, num_steps=2, seed=[23, 24])
    assert not torch.allclose(a.latent[1], right.latent[1])
    assert not torch.allclose(a.tcp_patch[1], right.tcp_patch[1])
    assert torch.equal(a.latent[0], right.latent[0])
    swapped = valid.flip(1)
    d = e.generate(obs, ctx, observed_view_valid=swapped, num_steps=2, seed=[23, 24])
    assert not torch.equal(a.latent, d.latent)
    with pytest.raises(ValueError, match='view'):
        e.generate(obs, ctx, num_steps=1)
    with pytest.raises(ValueError, match='valid'):
        e.generate(obs, ctx, observed_view_valid=torch.zeros_like(valid), num_steps=1)


def test_multiview_prefix_isolation_and_context_padding():
    e = make_expert(observation_mode='multiview').eval()
    obs, ctx, _, valid = multiview_batch(1)
    z = e.encode_observation(obs, observed_view_valid=valid)
    goal = torch.zeros(1, 48, 1, 16, 20)
    tcp = torch.zeros(1, 192); times = torch.tensor([700.])
    a = e.velocity(z, goal, tcp, times, ctx, observed_view_valid=valid, return_observation=True, context_lens=torch.tensor([2]))
    changed_ctx = ctx.clone(); changed_ctx[:, 2:] += 100
    b = e.velocity(z, goal+1, tcp+1, times, changed_ctx, observed_view_valid=valid, return_observation=True, context_lens=torch.tensor([2]))
    assert a[2].shape == (1, 48, 1, 24, 20)
    assert torch.allclose(a[2], b[2], atol=1e-6)
    assert not torch.allclose(a[0], b[0])
    assert not torch.allclose(a[1], b[1])


def test_v1_v2_strict_bundle_roundtrips_explicit_warm_start_and_dependency_identity(tmp_path):
    from openwam.train.goal_patch import save_patch_bundle, load_patch_bundle
    head = make_expert().eval()
    v1 = tmp_path/'head.safetensors'
    save_patch_bundle(head, v1, precomputed_text_fingerprint='a'*64)
    kwargs = dict(vae=TinyVAE(), text_dim=16, precomputed_text_fingerprint='a'*64)
    old = load_patch_bundle(v1, **kwargs)
    assert old.observation_mode == 'head'
    assert old.bundle_metadata['format'] == 'openwam.goal.patch.v1'
    obs, ctx, _ = batch(1)
    assert torch.equal(head.generate(obs, ctx, num_steps=1).latent, old.generate(obs, ctx, num_steps=1).latent)
    with pytest.raises(ValueError, match='mode'):
        load_patch_bundle(v1, observation_mode='multiview', **kwargs)
    warm = load_patch_bundle(v1, observation_mode='multiview', warm_start=True, **kwargs)
    assert warm.observation_mode == 'multiview'
    assert all(torch.equal(v, warm.state_dict()[k]) for k, v in head.state_dict().items())
    with pytest.raises(ValueError, match='text'):
        load_patch_bundle(v1, observation_mode='multiview', warm_start=True, **{**kwargs, 'precomputed_text_fingerprint':'b'*64})
    bad_vae = TinyVAE(); bad_vae.marker.fill_(1)
    with pytest.raises(ValueError, match='VAE'):
        load_patch_bundle(v1, observation_mode='multiview', warm_start=True, **{**kwargs, 'vae':bad_vae})
    v2 = tmp_path/'multi.safetensors'
    metadata = save_patch_bundle(warm, v2, precomputed_text_fingerprint='a'*64)
    assert metadata['format'] == 'openwam.goal.patch.v2' and metadata['observation_mode'] == 'multiview'
    assert metadata['codec']['image_shape'] == (3, 1, 256, 320)
    loaded = load_patch_bundle(v2, **kwargs)
    assert loaded.observation_mode == 'multiview'
    assert all(torch.equal(v, loaded.state_dict()[k]) for k, v in warm.state_dict().items())
    with pytest.raises(ValueError, match='mode'):
        load_patch_bundle(v2, observation_mode='head', **kwargs)
    with pytest.raises(ValueError, match='warm'):
        load_patch_bundle(v2, observation_mode='multiview', warm_start=True, **kwargs)


def test_multiview_gradient_checkpointing_and_training_evaluation_forward_masks(tmp_path):
    from openwam.train.goal_patch import train_patch, make_smoke_data
    e = make_expert(observation_mode='multiview', use_gradient_checkpointing=True)
    obs, ctx, targets, valid = multiview_batch(1)
    _, _, records = make_smoke_data()
    batches = [dict(observed=obs, context=ctx, targets=targets, sources=['synthetic'], observed_view_valid=valid)]
    result = train_patch(e, batches=batches, records=records, output_dir=tmp_path/'run',
                         steps=1, eval_every=1, sampling_steps=1, learning_rate=1e-3,
                         precomputed_text_fingerprint='a'*64)
    assert result['status'] == 'complete'
    assert [v['step'] for v in result['evaluations']] == [0, 1]
    assert e.codec.vae.calls.count((1, 3, 1, 384, 320)) == 3
    metadata = json.loads((tmp_path/'run/patch_goal.json').read_text())
    assert metadata['format'] == 'openwam.goal.patch.v2'


def test_historical_v1_metadata_without_observation_fields_remains_usable(tmp_path):
    from safetensors.torch import save_file
    from openwam.train.goal_patch import save_patch_bundle, load_patch_bundle
    e = make_expert().eval()
    path = tmp_path/'legacy.safetensors'
    meta = save_patch_bundle(e, path, precomputed_text_fingerprint='a'*64)
    meta.pop('observation_mode'); meta.pop('observation')
    save_file({k:v.contiguous() for k,v in e.state_dict().items()}, str(path),
              metadata={'patch_goal_bundle':json.dumps(meta)})
    loaded = load_patch_bundle(path, vae=TinyVAE(), text_dim=16, precomputed_text_fingerprint='a'*64)
    assert loaded.observation_mode == 'head'
    warm = load_patch_bundle(path, vae=TinyVAE(), text_dim=16, precomputed_text_fingerprint='a'*64,
                             observation_mode='multiview', warm_start=True)
    warm_metadata = save_patch_bundle(warm, tmp_path/'warm.safetensors', precomputed_text_fingerprint='a'*64)
    assert warm_metadata['initialization']['trained_multiview'] is False
