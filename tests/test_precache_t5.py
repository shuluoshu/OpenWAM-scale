"""Offline preparation must produce the exact cache consumed by training."""

import json

import pytest
import torch
from omegaconf import OmegaConf
from safetensors.torch import save_file

from openwam.model.video_backbone.wan.encode import encode_text
from openwam.model.video_backbone.wan.models.text_encoder import WanTextEncoder
from openwam.model.video_backbone.wan.text_feature_cache import FrozenTextFeatureCache
from scripts.cache_t5_text_features import load_prompts, load_text_components, main, precompute


@pytest.mark.parametrize(
    "suffix,content",
    [
        (".txt", "pick\nplace\npick\n\n"),
        (".json", '["pick", "place", "pick"]'),
        (".jsonl", '{"prompt":"pick"}\n{"prompt":"place"}\n{"prompt":"pick"}\n'),
    ],
)
def test_prompt_formats_deduplicate_without_reordering(tmp_path, suffix, content):
    path = tmp_path / ("instructions" + suffix)
    path.write_text(content)
    assert load_prompts(path) == ["pick", "place"]


def test_invalid_prompts_fail_before_loading_model(tmp_path):
    path = tmp_path / "instructions.json"
    path.write_text('["pick", 123]')
    with pytest.raises(ValueError, match="string"):
        load_prompts(path)
    path.write_text("[]")
    with pytest.raises(ValueError, match="empty"):
        load_prompts(path)


class Tokenizer:
    def __call__(self, prompts, **kwargs):
        ids = torch.zeros(len(prompts), kwargs["max_length"], dtype=torch.long)
        ids[:, :2] = torch.tensor([[len(p), 1] for p in prompts])
        return ids, ids.ne(0).long()


def test_offline_precompute_is_reused_by_training_without_encoder_calls(tmp_path):
    encoder = WanTextEncoder(vocab=16, dim=8, dim_attn=8, dim_ffn=16, num_heads=2, num_layers=1).requires_grad_(False)
    tokenizer = Tokenizer()
    report = precompute(
        ["pick", "place", "pick"], tokenizer=tokenizer, text_encoder=encoder, device="cpu", cache_dir=tmp_path
    )
    assert report["unique_prompts"] == 2
    assert report["cache_stats"]["misses"] == 2
    encoder.train()
    calls = []
    hook = encoder.register_forward_hook(lambda *args: calls.append(1))
    context, lengths = encode_text(
        ["place", "pick"],
        tokenizer=tokenizer,
        text_encoder=encoder,
        device="cpu",
        feature_cache=FrozenTextFeatureCache(tmp_path),
    )
    hook.remove()
    assert calls == []
    assert lengths.tolist() == [2, 2]
    assert context.shape == (2, 512, 8)
    assert not encoder.training


def test_loader_loads_only_text_weights_and_checkpoint_tokenizer(tmp_path, monkeypatch):
    kwargs = dict(vocab=16, dim=8, dim_attn=8, dim_ffn=16, num_heads=2, num_layers=1)
    original = WanTextEncoder(**kwargs).eval()
    cfg = {
        "model": {
            "video_backbone": {
                "components": [
                    {"attr": "dit", "model_class": "must.not.load.DiT"},
                    {
                        "attr": "text_encoder",
                        "model_class": "openwam.model.video_backbone.wan.models.text_encoder.WanTextEncoder",
                        "extra_kwargs": kwargs,
                    },
                ],
                "tokenizer": {"class": "test.Tokenizer", "subdir": "tokenizer"},
            }
        }
    }
    OmegaConf.save(OmegaConf.create(cfg), tmp_path / "config.yaml")
    weights = {"video_backbone.text_encoder." + k: v for k, v in original.state_dict().items()}
    weights["video_backbone.dit.ignore_me"] = torch.ones(1)
    save_file(weights, tmp_path / "checkpoint_step_1.safetensors")
    tok = Tokenizer()
    monkeypatch.setattr(
        "openwam.model.video_backbone.wan.pipeline_builder._build_tokenizer", lambda spec, base_dir: tok
    )
    loaded, actual_tok = load_text_components(tmp_path, device="cpu", dtype=torch.float32)
    assert actual_tok is tok
    assert not loaded.training and not any(p.requires_grad for p in loaded.parameters())
    assert all(torch.equal(v, loaded.state_dict()[k]) for k, v in original.state_dict().items())


def test_cli_twice_reuses_cache_with_real_local_tokenizer(tmp_path):
    from tokenizers import Tokenizer as BackendTokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    backend = BackendTokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "pick": 2, "place": 3}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]")
    tokenizer.save_pretrained(tmp_path / "tokenizer")
    kwargs = dict(vocab=16, dim=8, dim_attn=8, dim_ffn=16, num_heads=2, num_layers=1)
    cfg = {
        "model": {
            "video_backbone": {
                "components": [
                    {
                        "attr": "text_encoder",
                        "model_class": "openwam.model.video_backbone.wan.models.text_encoder.WanTextEncoder",
                        "extra_kwargs": kwargs,
                    }
                ],
                "tokenizer": {
                    "class": "openwam.model.video_backbone.wan.models.text_encoder.HuggingfaceTokenizer",
                    "subdir": "tokenizer",
                    "path_kwarg": "name",
                    "kwargs": {"seq_len": 512, "clean": "whitespace"},
                },
            }
        }
    }
    OmegaConf.save(OmegaConf.create(cfg), tmp_path / "config.yaml")
    encoder = WanTextEncoder(**kwargs)
    save_file(
        {"video_backbone.text_encoder." + k: v for k, v in encoder.state_dict().items()},
        tmp_path / "checkpoint_step_1.safetensors",
    )
    prompts = tmp_path / "prompts.json"
    prompts.write_text('["pick", "place", "pick"]')
    report_path = tmp_path / "report.json"
    argv = [
        "--checkpoint",
        str(tmp_path),
        "--prompts-file",
        str(prompts),
        "--cache-dir",
        str(tmp_path / "cache"),
        "--device",
        "cpu",
        "--dtype",
        "float32",
        "--threads",
        "1",
        "--report",
        str(report_path),
    ]
    main(argv)
    assert json.loads(report_path.read_text())["cache_stats"]["misses"] == 2
    main(argv)
    report = json.loads(report_path.read_text())
    assert report["unique_prompts"] == 2
    assert report["cache_stats"]["disk_hits"] == 2
    assert report["cache_stats"]["misses"] == 0
