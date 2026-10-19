"""Stage2 padding/view masks must reach the real flow loss and its gradients."""
from types import SimpleNamespace
import pytest
import torch
from openwam.model.architectures.base import BaseWAMArchitecture
from openwam.model.architectures.utils.common import downsample_video_mask_to_latent


def test_strict_partial_group_does_not_supervise_padded_rgb():
    padded = torch.tensor([False, False, False, True, True, True, True, True, True])
    assert downsample_video_mask_to_latent(padded).tolist() == [False, True]
    assert downsample_video_mask_to_latent(padded, strict=True).tolist() == [True, True]
    assert downsample_video_mask_to_latent(padded, strict=True, skip_first=False).tolist() == [False, True, True]


def test_view_mask_maps_l_canvas_and_partial_time_groups():
    from openwam.model.architectures.utils.common import stage2_latent_view_mask
    frames = torch.ones(9, dtype=torch.bool)
    views = torch.ones(9, 3, dtype=torch.bool)
    views[:, 2] = False
    views[4, 1] = False
    mask = stage2_latent_view_mask(frames, views, (24, 20), temporal_factor=4, skip_first=True)
    assert mask.shape == (2, 24, 20)
    assert mask[:, :16].all()
    assert not mask[:, 16:, 10:].any()
    assert not mask[0, 16:, :10].any()
    assert mask[1, 16:, :10].all()
    with pytest.raises(ValueError):
        stage2_latent_view_mask(frames, views[:, :2], (24, 20), temporal_factor=4, skip_first=True)


def loss_model():
    return SimpleNamespace(video_backbone=SimpleNamespace(scheduler=SimpleNamespace(
        linear_timesteps_weights=torch.ones(1))))


@pytest.mark.parametrize("missing_target", [100., float("nan")])
def test_view_and_time_invalid_targets_have_zero_loss_gradient(missing_target):
    # One observed latent, two future latents. Only head of first future is valid.
    pred = torch.ones(1, 2, 3, 6, 4, requires_grad=True)
    target = torch.zeros_like(pred)
    target[:, :, 0] = 20
    target[:, :, 1, 4:] = missing_target
    target[:, :, 2] = 200
    valid = torch.zeros(1, 2, 6, 4, dtype=torch.bool)
    valid[:, 0, :4] = True
    inputs = {'first_frame_latents': torch.zeros(1), 'video_is_pad': torch.tensor([[False, True]]),
              'video_spatial_valid': valid}
    loss = BaseWAMArchitecture._compute_video_loss(loss_model(), pred, target, torch.zeros(1, dtype=torch.long), inputs, 'cpu')
    torch.testing.assert_close(loss, torch.tensor(1.))
    loss.backward()
    assert pred.grad[:, :, 1, :4].abs().sum() > 0
    assert not pred.grad[:, :, 0].any() and not pred.grad[:, :, 2].any()
    assert not pred.grad[:, :, 1, 4:].any()


def test_no_valid_view_loss_is_differentiable_zero_and_bad_shape_fails():
    pred = torch.ones(1, 2, 2, 6, 4, requires_grad=True)
    target = torch.zeros_like(pred)
    valid = torch.zeros(1, 2, 6, 4, dtype=torch.bool)
    inputs = {'video_spatial_valid': valid}
    loss = BaseWAMArchitecture._compute_video_loss(loss_model(), pred, target, torch.zeros(1, dtype=torch.long), inputs, 'cpu')
    assert loss.item() == 0
    loss.backward()
    assert not pred.grad.any()
    inputs['video_spatial_valid'] = valid[:, :, :5]
    with pytest.raises(ValueError, match='video_spatial_valid'):
        BaseWAMArchitecture._compute_video_loss(loss_model(), pred, target, torch.zeros(1, dtype=torch.long), inputs, 'cpu')


def test_prepare_inputs_threads_strict_view_mask_into_actual_loss():
    from tests.goal.test_patch_stage2_memory import make_arch
    a = make_arch()
    a.set_training_runtime()
    a._pipeline_transform_instance = SimpleNamespace(apply=lambda sample: sample)
    a.preprocess = lambda **kwargs: {'input_latents': torch.zeros(1, 48, 3, 24, 20),
        'first_frame_latents': torch.zeros(1, 48, 1, 24, 20)}
    views = torch.ones(9, 3, dtype=torch.bool)
    views[:, 2] = False
    sample = {'video': None, 'prompt': 'move', 'video_mask': torch.tensor([True]*6+[False]*3),
              'video_view_mask': views, 'stage2_strict_video_mask': True}
    inputs = a.prepare_inputs([sample])
    assert inputs['video_is_pad'].tolist() == [[False, True]]
    assert inputs['video_spatial_valid'].shape == (1, 2, 24, 20)
    assert not inputs['video_spatial_valid'][0, 1].any()
    assert not inputs['video_spatial_valid'][0, 0, 16:, 10:].any()
    sample.pop('video_view_mask')
    with pytest.raises(ValueError, match='video_view_mask'):
        a.prepare_inputs([sample])


def test_strict_batch_missing_later_time_mask_fails_descriptively():
    from tests.goal.test_patch_stage2_memory import make_arch
    a = make_arch()
    a.set_training_runtime()
    a._pipeline_transform_instance = SimpleNamespace(apply=lambda sample: sample)
    a.preprocess = lambda **kwargs: {'input_latents': torch.zeros(2, 48, 3, 24, 20),
        'first_frame_latents': torch.zeros(2, 48, 1, 24, 20)}
    sample = {'video': None, 'prompt': 'move', 'video_mask': torch.ones(9, dtype=torch.bool),
              'video_view_mask': torch.ones(9, 3, dtype=torch.bool), 'stage2_strict_video_mask': True}
    with pytest.raises(ValueError, match='consistent strict video masks'):
        a.prepare_inputs([sample, {**sample, 'video_mask': None}])
