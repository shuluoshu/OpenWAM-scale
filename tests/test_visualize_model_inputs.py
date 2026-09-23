from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

import scripts.visualize_model_inputs as visualize


class _Dataset:
    def __len__(self):
        return 1

    def __getitem__(self, index):
        active = np.r_[np.arange(10), np.arange(34, 44)]
        action_mask = torch.zeros(32, 80, dtype=torch.bool)
        action_mask[:, active] = True
        proprio_mask = torch.zeros(1, 80, dtype=torch.bool)
        proprio_mask[:, active] = True
        return {
            "video": [Image.new("RGB", (320, 384), (index * 10, i * 10, 20)) for i in range(9)],
            "action": torch.zeros(32, 80),
            "action_mask": action_mask,
            "proprio": torch.zeros(1, 80),
            "proprio_mask": proprio_mask,
            "video_mask": torch.ones(9, dtype=torch.bool),
            "prompt": "Put the cup on the plate.",
            "trace": {
                "dataset_dir": "/data/example",
                "dataset_id": "example",
                "episode_index": 2,
                "window_offset": 3,
                "raw_frame_indices": list(range(3, 36)),
                "video_frame_indices": list(range(3, 36, 4)),
                "raw_action_eef20": np.ones((32, 20), dtype=np.float32),
                "normalized_action_eef20": np.zeros((32, 20), dtype=np.float32),
                "unified_action80": np.zeros((32, 80), dtype=np.float32),
                "raw_proprio_eef20": np.ones((1, 20), dtype=np.float32),
                "normalized_proprio_eef20": np.zeros((1, 20), dtype=np.float32),
                "unified_proprio80": np.zeros((1, 80), dtype=np.float32),
                "stats_path": "/stats/example.npy",
                "stats_metadata": {"episode_indices": [0, 1]},
            },
        }


def test_render_profile_writes_forward_input_artifacts(tmp_path: Path, monkeypatch):
    profile = tmp_path / "profile.yaml"
    OmegaConf.save(OmegaConf.create({"type": "canonical_eef_lerobot"}), profile)
    monkeypatch.setattr(visualize, "build_dataset", lambda config, split: _Dataset())

    result = visualize.render_profile("demo", profile, tmp_path / "report", sample_index=0)

    image_path = tmp_path / "report" / "demo_frames.png"
    json_path = tmp_path / "report" / "demo.json"
    npz_path = tmp_path / "report" / "demo.npz"
    assert image_path.is_file() and Image.open(image_path).size == (960, 1152)
    assert json_path.is_file() and npz_path.is_file()
    manifest = json.loads(json_path.read_text(encoding="utf-8"))
    assert manifest["prompt"] == "Put the cup on the plate."
    assert manifest["pipeline"] == ["parquet", "absolute EEF20", "source normalization", "scatter OpenWAM80", "model forward"]
    assert manifest["shapes"]["action"] == [32, 80]
    assert manifest["active_action_indices"] == list(range(10)) + list(range(34, 44))
    arrays = np.load(npz_path)
    assert arrays["raw_action_eef20"].shape == (32, 20)
    assert arrays["normalized_action_eef20"].shape == (32, 20)
    assert arrays["unified_action80"].shape == (32, 80)
    assert result["status"] == "ok"


def test_render_index_is_offline_and_links_existing_assets(tmp_path: Path):
    output = tmp_path / "report"
    output.mkdir()
    (output / "demo_frames.png").write_bytes(b"png")
    (output / "demo.json").write_text("{}", encoding="utf-8")
    (output / "demo.npz").write_bytes(b"npz")
    page = visualize.render_index(
        [
            {
                "status": "ok",
                "name": "demo",
                "prompt": "Do the task",
                "image": "demo_frames.png",
                "json": "demo.json",
                "npz": "demo.npz",
                "plot": [[0.0, 1.0], [1.0, 0.0]],
            }
        ],
        output,
    )

    html = page.read_text(encoding="utf-8")
    assert "Do the task" in html
    assert "demo_frames.png" in html
    assert "<svg" in html
    assert "http://" not in html and "https://" not in html
