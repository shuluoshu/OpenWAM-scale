from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

import scripts.open_loop_inference as open_loop


class _Scheduler:
    num_train_timesteps = 1000

    def set_timesteps(self, steps, shift=5.0):
        self.timesteps = torch.linspace(999, 1, steps)


class _Architecture:
    def __init__(self):
        self.video_scheduler = _Scheduler()
        self.action_scheduler = _Scheduler()
        self.action_backbone = type("AB", (), {"shift_action": 5.0})()
        self.video_backbone = type("VB", (), {"shift_video": 5.0})()
        self.kwargs = None

    def generate(self, **kwargs):
        self.kwargs = kwargs
        return {"video": None, "actions": np.zeros((32, 80), dtype=np.float32)}


class _Dataset:
    def __init__(self):
        active = np.r_[np.arange(10), np.arange(34, 44)]
        action_mask = torch.zeros(32, 80, dtype=torch.bool)
        action_mask[:, active] = True
        proprio_mask = torch.zeros(1, 80, dtype=torch.bool)
        proprio_mask[:, active] = True
        self.sample = {
            "video": [Image.new("RGB", (320, 384)) for _ in range(9)],
            "first_frame_image": [Image.new("RGB", (320, 384))],
            "vace_video": None,
            "action": torch.zeros(32, 80),
            "action_mask": action_mask,
            "video_mask": torch.ones(9, dtype=torch.bool),
            "proprio": torch.ones(1, 80),
            "proprio_mask": proprio_mask,
            "prompt": "Stack the blocks.",
            "trace": {
                "dataset_dir": "/data/fake",
                "dataset_id": "fake",
                "episode_index": 0,
                "window_offset": 0,
                "action_target_source": "next_state",
                "action_target_state_key": "observation.states.eef",
                "action_target_offset": 1,
                "action_target_frame_indices": list(range(1, 33)),
                "raw_action_eef20": np.ones((32, 20), dtype=np.float32),
                "normalized_action_eef20": np.zeros((32, 20), dtype=np.float32),
                "stats_path": "/stats/fake.npy",
                "stats_metadata": {"episode_indices": [0, 1]},
            },
        }

    def __getitem__(self, index):
        return self.sample

    def __len__(self):
        return 1

    def gather_unified_eef(self, values):
        values = np.asarray(values)
        return np.concatenate([values[..., :10], values[..., 34:44]], axis=-1)

    def denormalize_eef(self, values, stream="action"):
        return np.asarray(values, dtype=np.float32) + 10.0


def test_run_open_loop_forwards_model_contract_and_saves_outputs(tmp_path: Path, monkeypatch):
    profile = tmp_path / "profile.yaml"
    OmegaConf.save(OmegaConf.create({"type": "canonical_eef_lerobot", "return_trace": True}), profile)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    dataset = _Dataset()
    architecture = _Architecture()
    monkeypatch.setattr(open_loop, "build_dataset", lambda config, split: dataset)

    manifest = open_loop.run_open_loop(
        profile,
        checkpoint,
        tmp_path / "out",
        sample_index=0,
        device="cuda:0",
        denoise_steps=2,
        seed=7,
        model_loader=lambda path, device: (OmegaConf.create({}), architecture),
    )

    kwargs = architecture.kwargs
    assert kwargs["prompt"] == "Stack the blocks."
    assert kwargs["first_frame_image"] == dataset.sample["first_frame_image"]
    assert kwargs["num_frames"] == 9
    assert kwargs["action_num_frames"] == 33
    assert kwargs["num_inference_steps"] == 2
    assert kwargs["decode_video"] is False
    assert kwargs["seed"] == 7
    assert kwargs["proprio"].shape == (1, 80)
    assert kwargs["active_action_mask"].nonzero().flatten().tolist() == list(range(10)) + list(range(34, 44))
    arrays = np.load(tmp_path / "out" / "prediction.npz")
    assert arrays["prediction_normalized80"].shape == (32, 80)
    assert arrays["prediction_normalized_eef20"].shape == (32, 20)
    assert arrays["prediction_physical_eef20"].shape == (32, 20)
    assert arrays["gt_physical_eef20"].shape == (32, 20)
    assert manifest["checkpoint_dir"] == str(checkpoint.resolve())
    assert manifest["action_target_source"] == "next_state"
    assert manifest["action_target_offset"] == 1
    assert manifest["action_target_frame_indices"] == list(range(1, 33))
    assert json.loads((tmp_path / "out" / "manifest.json").read_text())["denoise_steps"] == 2


def test_run_open_loop_rejects_nonfinite_prediction(tmp_path: Path, monkeypatch):
    profile = tmp_path / "profile.yaml"
    OmegaConf.save(OmegaConf.create({"type": "canonical_eef_lerobot", "return_trace": True}), profile)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    dataset = _Dataset()
    architecture = _Architecture()
    monkeypatch.setattr(open_loop, "build_dataset", lambda config, split: dataset)

    def bad_generate(**kwargs):
        result = np.zeros((32, 80), dtype=np.float32)
        result[0, 0] = np.nan
        return {"video": None, "actions": result}

    architecture.generate = bad_generate
    try:
        open_loop.run_open_loop(
            profile,
            checkpoint,
            tmp_path / "out",
            model_loader=lambda path, device: (OmegaConf.create({}), architecture),
        )
    except ValueError as exc:
        assert "non-finite" in str(exc)
    else:
        raise AssertionError("non-finite predictions must fail")


@pytest.mark.parametrize("width", [94, 112])
def test_open_loop_accepts_extended_dimensions_and_preserves_full_predictions(tmp_path, monkeypatch, width):
    profile = tmp_path / "profile.yaml"
    OmegaConf.save(OmegaConf.create({"type": "canonical_eef_lerobot"}), profile)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    ds = _Dataset()
    for key in ("action", "action_mask", "proprio", "proprio_mask"):
        ds.sample[key] = torch.nn.functional.pad(ds.sample[key], (0, width - 80))
    architecture = _Architecture()
    architecture.action_dim = width
    ds.denormalize_head = lambda values, **kwargs: values.copy()
    architecture.generate = lambda **kwargs: {"actions": np.zeros((32, width), np.float32)}
    monkeypatch.setattr(open_loop, "build_dataset", lambda config, split: ds)
    manifest = open_loop.run_open_loop(profile, checkpoint, tmp_path / "out",
                                      model_loader=lambda *a, **k: (None, architecture))
    assert manifest["prediction_shape"] == [32, width]
    assert np.load(tmp_path / "out/prediction.npz")[f"prediction_normalized{width}"].shape == (32, width)
    architecture.action_dim = 80
    with pytest.raises(ValueError, match=f"checkpoint action_dim=80, reader width={width}"):
        open_loop.run_open_loop(profile, checkpoint, tmp_path / "bad",
                               model_loader=lambda *a, **k: (None, architecture))
