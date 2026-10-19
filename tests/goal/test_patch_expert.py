"""Joint image/TCP denoising uses real tiny Wan blocks and frozen codec IO."""
import pytest
import torch

from test_expert import TINY_DIT, TinyVAE, batch
from openwam.model.goal.codec import GoalCodec
from openwam.model.video_backbone.wan.models.dit import WanModel


def make_expert(**kwargs):
    from openwam.model.goal.patch_expert import PatchGoalExpert
    torch.manual_seed(5)
    return PatchGoalExpert(WanModel(**TINY_DIT), GoalCodec(TinyVAE()),
                           dit_config=TINY_DIT, **kwargs)


def test_fixed_repeat_group_mean_codec_preserves_order_continuous_and_unclamped_uv():
    from openwam.model.goal.patch_expert import encode_tcp_patch, decode_tcp_patch
    uv = torch.tensor([[[.125, .25], [.75, 1.125]]], requires_grad=True)
    patch = encode_tcp_patch(uv)
    assert patch.shape == (1, 192)
    assert torch.equal(patch.reshape(1, 4, 48)[0, :, 0], 2 * uv.flatten() - 1)
    assert torch.equal(decode_tcp_patch(patch), uv)
    varied = patch.detach() + torch.arange(48).repeat(4)[None] / 48
    expected = (varied.reshape(1, 4, 48).mean(-1).reshape(1, 2, 2) + 1) / 2
    assert torch.equal(decode_tcp_patch(varied), expected)
    decode_tcp_patch(patch).sum().backward()
    assert torch.equal(uv.grad, torch.ones_like(uv))
    with pytest.raises(ValueError, match='192'):
        decode_tcp_patch(torch.zeros(1, 128))


def test_patch_volume_order_and_shared_input_and_output_projections():
    e = make_expert()
    obs, context, _ = batch(1)
    z = e.codec.encode(obs)
    patch = torch.arange(192, dtype=z.dtype)[None]
    inputs, outputs = [], []
    ih = e.dit.patch_embedding.register_forward_pre_hook(lambda module, args: inputs.append(args[0]))
    oh = e.dit.head.register_forward_hook(lambda module, args, out: outputs.append(out))
    goal_v, tcp_v = e.velocity(z, z, patch, torch.tensor([700.]), context)
    ih.remove(); oh.remove()
    tcp_volume = inputs[-1]
    assert tcp_volume.shape == (1, 48, 1, 2, 2)
    assert torch.equal(tcp_volume, e.dit.unpatchify(patch[:, None], (1, 1, 1)))
    assert torch.equal(tcp_volume[0, :, 0, 0, 1], patch[0, 48:96])
    assert len(outputs) == 1 and outputs[0].shape == (1, 161, 192)
    assert torch.equal(tcp_v, outputs[0][:, -1])
    assert torch.equal(goal_v, e.dit.unpatchify(outputs[0][:, 80:160], (1, 8, 10)))
    assert not any('head' in name or 'embedding' in name for name, _ in e.named_parameters()
                   if not name.startswith('dit.'))


def test_tcp_only_loss_reaches_shared_wan_projection_blocks_and_head():
    e = make_expert()
    obs, ctx, targets = batch()
    out = e(obs, ctx, targets, timestep_indices=torch.tensor([200, 700]))
    assert out['valid_points'] == 4  # Occluded/unknown visibility does not remove geometry.
    assert e.codec.vae.calls == [(2, 3, 1, 256, 320)] * 2
    out['tcp_loss'].backward()
    assert e.dit.patch_embedding.weight.grad.abs().sum() > 0
    assert e.dit.blocks[0].self_attn.q.weight.grad.abs().sum() > 0
    assert e.dit.head.head.weight.grad.abs().sum() > 0
    assert not any('vae' in name for name in e.state_dict())
    assert torch.allclose(out['loss'], .5 * out['fm_loss'] + out['tcp_loss'])


def test_goal_tcp_mutual_coupling_and_observation_prefix_isolation():
    e = make_expert().eval()
    obs, ctx, _ = batch(1)
    z = e.codec.encode(obs)
    times = torch.tensor([700.])
    a = e.velocity(z, torch.zeros_like(z), torch.zeros(1, 192), times, ctx, return_observation=True)
    b = e.velocity(z, torch.ones_like(z), torch.zeros(1, 192), times, ctx, return_observation=True)
    c = e.velocity(z, torch.zeros_like(z), torch.ones(1, 192), times, ctx, return_observation=True)
    assert torch.allclose(a[2], b[2], atol=1e-6)
    assert torch.allclose(a[2], c[2], atol=1e-6)
    assert not torch.allclose(a[1], b[1])  # TCP reads goal.
    assert not torch.allclose(a[0], c[0])  # Goal reads TCP.


def test_full_patch_independent_noise_shared_sigma_and_paper_training_weights():
    from openwam.model.goal.patch_expert import encode_tcp_patch
    e = make_expert()
    obs, ctx, targets = batch()
    idx = torch.tensor([100, 800])
    image_noise = torch.randn(2, 48, 1, 16, 20)
    tcp_noise = torch.randn(2, 192)
    out = e(obs, ctx, targets, timestep_indices=idx, noise=image_noise, tcp_noise=tcp_noise)
    clean_image = e.codec.encode(targets.image)
    clean_tcp = encode_tcp_patch(targets.uv)
    sigma = e.scheduler.sigmas[idx]
    assert torch.equal(out['sigma'], sigma)
    assert torch.allclose(out['noisy_latent'], (1 - sigma[:, None, None, None, None]) * clean_image
                          + sigma[:, None, None, None, None] * image_noise)
    assert torch.allclose(out['noisy_tcp_patch'], (1 - sigma[:, None]) * clean_tcp
                          + sigma[:, None] * tcp_noise)
    assert torch.allclose(out['clean_tcp_patch'], out['noisy_tcp_patch'] - sigma[:, None] * out['tcp_velocity'])
    weights = torch.exp(-2 * (e.scheduler.sigmas - .5).square()) - torch.exp(torch.tensor(-.5))
    weights /= weights.mean()
    image_mse = (out['velocity'] - (image_noise - clean_image)).float().square().flatten(1).mean(1)
    tcp_mse = (out['tcp_velocity'] - (tcp_noise - clean_tcp)).float().square().mean(1)
    assert torch.allclose(out['fm_loss'], (image_mse * weights[idx]).mean())
    assert torch.allclose(out['tcp_loss'], (tcp_mse * weights[idx]).mean())
    auto = e(obs, ctx, targets, timestep_indices=idx)
    # Copies of one coordinate receive independent noise, not repeated 4D noise.
    assert (auto['tcp_noise'].reshape(2, 4, 48).std(-1) > .2).all()
    assert not torch.equal(auto['tcp_noise'][0], auto['tcp_noise'][1])


def test_invalid_nan_tcp_is_removed_before_arithmetic_and_image_loss_survives():
    e = make_expert()
    obs, ctx, targets = batch()
    targets.point_valid[:] = False
    targets.uv[:] = float('nan')
    out = e(obs, ctx, targets)
    assert out['valid_points'] == 0 and out['tcp_loss'].item() == 0
    assert torch.isfinite(out['loss']) and out['fm_loss'] > 0
    assert torch.isfinite(out['noisy_tcp_patch']).all()
    out['tcp_loss'].backward()
    assert e.dit.head.head.weight.grad.abs().sum() == 0
    e.zero_grad()
    out = e(obs, ctx, targets)
    out['loss'].backward()
    assert e.dit.head.head.weight.grad.abs().sum() > 0


def test_masked_coordinate_copies_and_missing_image_have_no_loss_contribution():
    e = make_expert()
    obs, ctx, targets = batch()
    targets.uv[0, 0] = float('nan')
    targets.in_frame[0, 1] = False
    targets.image_valid[1] = False
    targets.image[1] = float('nan')
    out = e(obs, ctx, targets, timestep_indices=torch.tensor([600, 800]))
    assert out['valid_points'] == 0 and out['tcp_loss'].item() == 0
    assert torch.isfinite(out['loss']) and out['fm_loss'].item() > 0
    out['velocity'].retain_grad()
    out['loss'].backward()
    assert out['velocity'].grad[1].abs().sum() == 0


def test_partial_point_mask_supervises_only_its_96_copies():
    e = make_expert()
    obs, ctx, targets = batch(1)
    targets.point_valid[0, 0] = False
    targets.uv[0, 0] = float('nan')
    out = e(obs, ctx, targets, timestep_indices=torch.tensor([700]))
    assert out['valid_points'] == 1
    out['tcp_velocity'].retain_grad()
    out['tcp_loss'].backward()
    assert out['tcp_velocity'].grad[0, :96].abs().sum() == 0
    assert out['tcp_velocity'].grad[0, 96:].abs().sum() > 0


def test_generate_is_target_free_deterministic_continuous_and_preserves_training_grid():
    from openwam.model.goal.patch_expert import decode_tcp_patch
    e = make_expert().eval()
    obs, ctx, _ = batch(1)
    sigmas = e.scheduler.sigmas.clone()
    weights = e.scheduler.linear_timesteps_weights.clone()
    a = e.generate(obs, ctx, seed=91, num_steps=3)
    b = e.generate(obs, ctx, seed=91, num_steps=3)
    c = e.generate(obs, ctx, seed=92, num_steps=3)
    assert a.latent.shape == (1, 48, 1, 16, 20)
    assert a.tcp_patch.shape == (1, 192) and a.uv.shape == (1, 2, 2)
    assert a.token_valid.shape == (1, 81) and a.token_valid.all()
    assert torch.equal(a.latent, b.latent) and torch.equal(a.tcp_patch, b.tcp_patch)
    assert not torch.equal(a.tcp_patch, c.tcp_patch)
    assert torch.equal(a.uv, decode_tcp_patch(a.tcp_patch))
    assert not torch.equal(a.uv * a.uv.new_tensor([319., 255.]),
                           (a.uv * a.uv.new_tensor([319., 255.])).round())
    assert torch.equal(sigmas, e.scheduler.sigmas)
    assert torch.equal(weights, e.scheduler.linear_timesteps_weights)
    assert not a.latent.requires_grad and not a.tcp_patch.requires_grad


def test_context_padding_and_checkpointing_and_frozen_teacher_semantics():
    e = make_expert(use_gradient_checkpointing=True)
    obs, ctx, targets = batch(1)
    lengths = torch.tensor([2])
    a = e(obs, ctx, targets, timestep_indices=torch.tensor([600]),
          noise=torch.zeros(1, 48, 1, 16, 20), tcp_noise=torch.zeros(1, 192), context_lens=lengths)
    changed = ctx.clone(); changed[:, 2:] += 100
    b = e(obs, changed, targets, timestep_indices=torch.tensor([600]),
          noise=torch.zeros(1, 48, 1, 16, 20), tcp_noise=torch.zeros(1, 192), context_lens=lengths)
    assert torch.allclose(a['velocity'], b['velocity'])
    assert torch.allclose(a['tcp_velocity'], b['tcp_velocity'])
    a['loss'].backward()
    assert e.dit.patch_embedding.weight.grad.abs().sum() > 0
    e.freeze().train()
    assert not e.training and not any(p.requires_grad for p in e.parameters())
    e.train_stage_a()
    assert e.training and all(p.requires_grad for p in e.parameters())
    assert not e.codec.vae.training


def test_explicit_bad_shapes_and_parameters():
    e = make_expert()
    obs, ctx, targets = batch(1)
    with pytest.raises(ValueError, match='text'):
        e(obs, ctx[:, :, :8], targets)
    with pytest.raises(ValueError, match='TCP'):
        e(obs, ctx, targets, tcp_noise=torch.zeros(1, 4))
    with pytest.raises(ValueError, match='timestep'):
        e(obs, ctx, targets, timestep_indices=torch.tensor([-1]))
    with pytest.raises(ValueError, match='length'):
        e(obs, ctx, targets, context_lens=torch.tensor([0]))
    with pytest.raises(ValueError):
        make_expert(lambda_tcp=-1)
    with pytest.raises(RuntimeError, match='eval'):
        e.generate(obs, ctx)


def test_bfloat16_wan_keeps_tcp_diffusion_and_group_mean_in_float32():
    from openwam.model.goal.patch_expert import decode_tcp_patch, encode_tcp_patch
    class FloatPoolTinyVAE(TinyVAE):
        # CPU avg_pool3d lacks BF16; GoalCodec casts this frozen IO back to BF16.
        def encode(self, videos, device, **kwargs):
            return super().encode([video.float() for video in videos], device, **kwargs)

    e = make_expert().bfloat16()
    e.codec.vae = FloatPoolTinyVAE()
    obs, ctx, targets = batch(1)
    obs = obs.bfloat16(); ctx = ctx.bfloat16(); targets.image = targets.image.bfloat16()
    targets.uv[0, 0, 0] = .1234567
    out = e(obs, ctx, targets, timestep_indices=torch.tensor([700]))
    assert out['noisy_latent'].dtype == torch.bfloat16
    for key in ('clean_tcp_patch', 'noisy_tcp_patch', 'tcp_noise', 'tcp_velocity', 'uv', 'sigma'):
        assert out[key].dtype == torch.float32, key
    assert encode_tcp_patch(targets.uv)[0, 0].item() == (2 * targets.uv[0, 0, 0] - 1).item()
    varied = torch.arange(192).reshape(1, 192).bfloat16() / 192
    expected = ((varied.float().reshape(1, 4, 48).mean(-1) + 1) / 2).reshape(1, 2, 2)
    assert torch.equal(decode_tcp_patch(varied), expected)
    out['loss'].backward()
    assert torch.isfinite(e.dit.head.head.weight.grad).all()
    pred = e.eval().generate(obs, ctx, num_steps=2)
    assert pred.latent.dtype == torch.bfloat16
    assert pred.tcp_patch.dtype == torch.float32 and pred.uv.dtype == torch.float32


def test_all_missing_targets_produce_differentiable_zero_losses():
    e = make_expert()
    obs, ctx, targets = batch(1)
    targets.image_valid[:] = False
    targets.image[:] = float('nan'); targets.uv[:] = float('nan')
    out = e(obs, ctx, targets)
    assert out['loss'].item() == out['fm_loss'].item() == out['tcp_loss'].item() == 0
    out['loss'].backward()
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in e.parameters())
