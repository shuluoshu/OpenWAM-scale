"""Behavioral tests for frozen T5 caching, without downloading any weights."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from openwam.model.video_backbone.wan.encode import encode_text
from openwam.model.video_backbone.wan.text_feature_cache import (
    FrozenTextFeatureCache,
    configure_text_feature_cache,
)
from openwam.model.video_backbone.wan_backbone import Wan22Ti2v


class Tokenizer:
    offset = 0

    def __call__(self, prompts, **kwargs):
        ids = torch.zeros(len(prompts), kwargs["max_length"], dtype=torch.long)
        for i, prompt in enumerate(prompts):
            ids[i, : len(prompt)] = torch.tensor([ord(c) % 32 + self.offset for c in prompt])
        return ids, ids.ne(0).long()


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(64, 4)
        self.dropout = nn.Dropout(0.8)
        self.calls = 0
        self.requires_grad_(False)

    def forward(self, ids, mask):
        self.calls += 1
        return self.dropout(self.embedding(ids))


def run(cache, encoder, prompts=("pick", "dropoff", "pick"), tokenizer=None):
    return encode_text(
        list(prompts), tokenizer=tokenizer or Tokenizer(), text_encoder=encoder, device="cpu", feature_cache=cache
    )


def test_dedup_order_persistence_and_exact_features(tmp_path):
    encoder = Encoder().eval()
    expected, lengths = run(None, encoder)
    encoder.calls = 0
    cache = FrozenTextFeatureCache(tmp_path)
    actual, actual_lengths = run(cache, encoder)
    assert encoder.calls == 2
    assert torch.equal(actual, expected)
    assert torch.equal(actual_lengths, lengths)
    assert not actual.requires_grad
    assert actual.dtype == encoder.embedding.weight.dtype
    for row, length in enumerate(lengths.tolist()):
        assert torch.count_nonzero(actual[row, length:]) == 0
    run(cache, encoder)
    assert encoder.calls == 2
    restored = FrozenTextFeatureCache(tmp_path)
    encoder.train()
    assert torch.equal(run(restored, encoder)[0], expected)
    assert not encoder.training
    assert encoder.calls == 2
    assert restored.stats["disk_hits"] == 2

    # Restart with another encoder instance, rather than just another cache.
    reloaded = Encoder()
    reloaded.load_state_dict(encoder.state_dict())
    assert torch.equal(run(FrozenTextFeatureCache(tmp_path), reloaded)[0], expected)
    assert reloaded.calls == 0


def test_trainable_encoder_rejected_even_on_cache_hit(tmp_path):
    encoder = Encoder()
    cache = FrozenTextFeatureCache(tmp_path)
    run(cache, encoder)
    encoder.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        run(cache, encoder)


def test_weight_and_tokenizer_changes_invalidate(tmp_path):
    encoder = Encoder()
    cache = FrozenTextFeatureCache(tmp_path)
    before = run(cache, encoder)[0]
    with torch.no_grad():
        encoder.embedding.weight.add_(1)
    after = run(cache, encoder)[0]
    assert encoder.calls == 4
    assert not torch.equal(before, after)
    changed = Tokenizer()
    changed.offset = 1
    assert not torch.equal(run(cache, encoder, tokenizer=changed)[0], after)
    assert encoder.calls == 6


def test_lru_eviction_still_reuses_disk(tmp_path):
    encoder = Encoder()
    cache = FrozenTextFeatureCache(tmp_path, max_memory_bytes=0)
    before = run(cache, encoder)[0]
    assert torch.equal(run(cache, encoder)[0], before)
    assert encoder.calls == 2
    assert cache.stats["memory_bytes"] == 0


def test_corrupted_entry_is_rebuilt(tmp_path):
    encoder = Encoder()
    cache = FrozenTextFeatureCache(tmp_path)
    expected = run(cache, encoder, ("pick",))[0]
    next(tmp_path.rglob("*.pt")).write_bytes(b"broken")
    assert torch.equal(run(FrozenTextFeatureCache(tmp_path), encoder, ("pick",))[0], expected)
    assert encoder.calls == 2


def test_cached_context_keeps_projection_and_state_gradients(tmp_path):
    context, _ = run(FrozenTextFeatureCache(tmp_path), Encoder())
    projection = nn.Linear(4, 2)
    state_projection = nn.Linear(3, 2)
    state = torch.ones(3, 3)
    (projection(context).mean() + state_projection(state).mean()).backward()
    assert projection.weight.grad.abs().sum() > 0
    assert state_projection.weight.grad.abs().sum() > 0


def test_backbone_train_and_config(tmp_path):
    holder = SimpleNamespace(text_encoder=Encoder(), dit=None)
    backbone = Wan22Ti2v(holder, text_dim=4)
    arch = SimpleNamespace(video_backbone=backbone)
    configure_text_feature_cache(arch, {"enabled": True, "directory": str(tmp_path), "max_memory_mb": 1})
    backbone.train()
    assert backbone.training
    assert not backbone.text_encoder.training
    assert not any("cache" in key for key in backbone.state_dict())
    configure_text_feature_cache(arch, {"enabled": False})
    backbone.train()
    assert backbone.text_encoder.training


def test_disabled_encode_preserves_training_behavior():
    encoder = Encoder().train()
    run(None, encoder)
    assert encoder.training
    assert encoder.calls == 1


def test_dtype_change_does_not_return_stale_context(tmp_path):
    encoder = Encoder()
    cache = FrozenTextFeatureCache(tmp_path)
    run(cache, encoder)
    encoder.double()
    assert run(cache, encoder)[0].dtype == torch.float64
    assert encoder.calls == 4


def test_training_preprocess_reuses_text_but_reencodes_augmented_rgb(tmp_path, monkeypatch):
    from PIL import Image

    from openwam.model.video_backbone.wan import encode

    holder = SimpleNamespace(
        text_encoder=Encoder(),
        tokenizer=Tokenizer(),
        dit=None,
        height_division_factor=8,
        width_division_factor=8,
        time_division_factor=4,
        time_division_remainder=1,
    )
    backbone = Wan22Ti2v(holder, text_dim=4)
    backbone._device, backbone._dtype = torch.device("cpu"), torch.float32
    configure_text_feature_cache(
        SimpleNamespace(video_backbone=backbone), {"enabled": True, "directory": str(tmp_path)}
    )
    vae_calls = []

    def video_encoder(pixels, **kwargs):
        vae_calls.append(pixels)
        return pixels

    monkeypatch.setattr(encode, "encode_video", video_encoder)
    dark = backbone.preprocess_input_for_train(frames=[[Image.new("RGB", (32, 32), "black")]], text=["pick"])
    backbone.train()
    bright = backbone.preprocess_input_for_train(frames=[[Image.new("RGB", (32, 32), "white")]], text=["pick"])
    assert backbone.text_encoder.calls == 1
    assert len(vae_calls) == 2
    assert torch.equal(dark["context"], bright["context"])
    assert not torch.equal(dark["input_latents"], bright["input_latents"])
