"""Frozen frame caching must survive shuffled episodes and partial histories."""
import pytest
import torch

from openwam.model.goal.codec import GoalCodec
from openwam.model.goal.history import HeadFeatureCache


class CountingVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.marker = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.single_batches = []
        self.native_batches = []

    def _encode(self, images):
        value = images.float().mean((1, 2, 3, 4)) + self.marker
        return value[:, None, None, None, None].expand(-1, 48, 1, 16, 20).to(images.dtype).clone()

    def encode(self, videos, device):
        self.single_batches.append(len(videos))
        return self._encode(torch.stack(videos).to(device))

    def batch_encode(self, videos, device):
        self.native_batches.append(len(videos))
        return self._encode(videos.to(device))

    @property
    def encoded_frames(self):
        return sum(self.single_batches) + sum(self.native_batches)


def image(value):
    return torch.full((1, 3, 1, 256, 320), float(value))


def history_image(value):
    images = torch.full((1, 10, 3, 1, 256, 320), float('nan'))
    images[:, 0] = image(value)
    valid = torch.zeros(1, 10, dtype=torch.bool)
    valid[:, 0] = True
    return images, valid


def test_known_episodes_reuse_their_own_frames_after_a_switch():
    vae = CountingVAE()
    codec, cache = GoalCodec(vae), HeadFeatureCache(4)
    for episode, value in [(1, .1), (2, .2), (1, .1)]:
        images, valid = history_image(value)
        output = cache.encode(codec, images, valid, episode_ids=torch.tensor([episode]))
        torch.testing.assert_close(output[:, 0], torch.full_like(output[:, 0], value))
        assert not output[:, 1:].any()
    assert vae.encoded_frames == 2


def test_partial_mask_preserves_other_valid_cached_frames():
    vae = CountingVAE()
    codec, cache = GoalCodec(vae), HeadFeatureCache(4)
    images, valid = history_image(.1)
    images[:, 1], valid[:, 1] = image(.2), True
    cache.encode(codec, images, valid, episode_ids=torch.tensor([1]))
    valid[:, 1], images[:, 1] = False, float('nan')
    output = cache.encode(codec, images, valid, episode_ids=torch.tensor([1]))
    assert vae.encoded_frames == 2
    assert torch.isfinite(output).all() and not output[:, 1:].any()


def test_codec_reuses_each_image_across_different_batches():
    vae = CountingVAE()
    codec = GoalCodec(vae)
    codec.enable_cache(max_bytes=1024 * 1024)
    first = codec.encode(torch.cat([image(.1), image(.2)]))
    second = codec.encode(torch.cat([image(.1), image(.3)]))
    assert vae.encoded_frames == 3
    torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)


def test_codec_dispatches_bounded_native_batches():
    vae = CountingVAE()
    codec = GoalCodec(vae)
    codec.max_encode_batch_size = 2
    output = codec.encode(torch.cat([image(i / 10) for i in range(5)]))
    assert vae.native_batches == [2, 2, 1]
    assert not vae.single_batches
    torch.testing.assert_close(output[:, 0, 0, 0, 0], torch.arange(5).float() / 10)


def test_codec_cache_invalidates_when_frozen_weights_change():
    vae = CountingVAE()
    codec = GoalCodec(vae)
    codec.enable_cache(max_bytes=1024 * 1024)
    before = codec.encode(image(.1))
    with torch.no_grad():
        vae.marker.add_(1)
    after = codec.encode(image(.1))
    torch.testing.assert_close(after, before + 1)
    assert vae.encoded_frames == 2


def test_training_preparation_keeps_history_pixels_on_cpu():
    from test_multiview_patch import multiview_batch
    from openwam.train.goal_curated import prepare_batch
    observed, context, targets, views = multiview_batch(1)
    images, valid = history_image(.1)
    batch = dict(observed=observed, context=context, targets=targets,
                 observed_view_valid=views, history_images=images,
                 history_valid=valid, history_ages=torch.arange(10, 0, -1).float()[None])
    prepared = prepare_batch(dict(id='case', family='tiny', instruction='move'), batch, device='meta', dtype=torch.float32)
    assert prepared['history_images'].device.type == 'cpu'
    assert prepared['observed'].device.type == 'meta'
    assert prepared['history_ages'].device.type == 'meta'


def test_unknown_episode_identity_does_not_persist_between_calls():
    vae = CountingVAE()
    codec, cache = GoalCodec(vae), HeadFeatureCache(4)
    images, valid = history_image(.1)
    cache.encode(codec, images, valid)
    cache.encode(codec, images, valid)
    assert vae.encoded_frames == 2


def test_empty_history_never_encodes_invalid_pixels():
    vae = CountingVAE()
    codec, cache = GoalCodec(vae), HeadFeatureCache(4)
    images, valid = history_image(.1)
    output = cache.encode(codec, images.fill_(float('nan')), valid & False,
                          episode_ids=torch.tensor([1]))
    assert not output.any() and vae.encoded_frames == 0
