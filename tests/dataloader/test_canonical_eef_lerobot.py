from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from openwam.dataloader.canonical_eef_lerobot import CanonicalEEFLerobotDataset


CAMS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
STATE_KEY = "observation.states.eef"
ACTION_KEY = "actions.eef"
ROT6D = np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32)


def _eef20(t: float) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray([t, t + 1, t + 2], dtype=np.float32),
            ROT6D,
            np.asarray([t % 2], dtype=np.float32),
            np.asarray([-t, t + 3, t + 4], dtype=np.float32),
            ROT6D,
            np.asarray([(t + 1) % 2], dtype=np.float32),
        ]
    )


def _schema_fingerprint(features: dict, state_key: str = STATE_KEY, action_key: str = ACTION_KEY) -> str:
    selected = {key: features[key] for key in (state_key, action_key, *CAMS)}
    encoded = json.dumps(selected, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _stats_block(offset: float = 0.0) -> dict:
    block = {
        "count": 80,
        "mean": np.full(20, offset, dtype=np.float32),
        "std": np.full(20, 2.0, dtype=np.float32),
        "min": np.full(20, -100.0, dtype=np.float32),
        "max": np.full(20, 100.0, dtype=np.float32),
        "q01": np.full(20, -100.0, dtype=np.float32),
        "q99": np.full(20, 100.0, dtype=np.float32),
    }
    for key, value in (("mean", 0.0), ("std", 1.0), ("min", -1.0), ("max", 1.0), ("q01", -1.0), ("q99", 1.0)):
        block[key][[3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 17, 18]] = value
    return block


def make_bucket(root: Path, *, include_eef: bool = True) -> tuple[Path, Path]:
    bucket = root / "bucket"
    (bucket / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (bucket / "data" / "chunk-000").mkdir(parents=True)

    features = {
        "task_index": {"dtype": "int64", "shape": []},
        **{cam: {"dtype": "video", "shape": [64, 64, 3]} for cam in CAMS},
    }
    if include_eef:
        features[STATE_KEY] = {"dtype": "float32", "shape": [20]}
        features[ACTION_KEY] = {"dtype": "float32", "shape": [20]}
    info = {
        "codebase_version": "v3.0",
        "fps": 30,
        "total_episodes": 3,
        "total_frames": 120,
        "splits": {"train": "0:3", "val": "0:3"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
    }
    (bucket / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    pq.write_table(
        pa.Table.from_pandas(pd.DataFrame({"task_index": [0]}, index=["Move the red block."])),
        bucket / "meta" / "tasks.parquet",
    )

    rows = []
    for episode_index in range(3):
        for frame_index in range(40):
            row = {
                "episode_index": episode_index,
                "frame_index": frame_index,
                "task_index": 0,
            }
            if include_eef:
                row[STATE_KEY] = _eef20(float(frame_index))
                row[ACTION_KEY] = _eef20(float(frame_index + 1))
            rows.append(row)
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(rows)), bucket / "data" / "chunk-000" / "file-000.parquet")

    episodes = []
    for episode_index in range(3):
        ep = {
            "episode_index": episode_index,
            "length": 40,
            "dataset_from_index": episode_index * 40,
            "dataset_to_index": (episode_index + 1) * 40,
            "data/chunk_index": 0,
            "data/file_index": 0,
        }
        for cam in CAMS:
            ep[f"videos/{cam}/chunk_index"] = 0
            ep[f"videos/{cam}/file_index"] = 0
        episodes.append(ep)
    pq.write_table(
        pa.Table.from_pandas(pd.DataFrame(episodes)),
        bucket / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )

    stats_path = root / "stats.npy"
    if include_eef:
        np.save(
            stats_path,
            {
                "eef": _stats_block(0.0),
                "eef_state": _stats_block(1.0),
                "metadata": {
                    "dataset_dir": str(bucket.resolve()),
                    "state_key": STATE_KEY,
                    "action_key": ACTION_KEY,
                    "episode_indices": [0, 1],
                    "state_width": 20,
                    "action_width": 20,
                    "schema_fingerprint": _schema_fingerprint(features),
                },
            },
            allow_pickle=True,
        )
    return bucket, stats_path


@pytest.fixture(autouse=True)
def patch_video_decode(monkeypatch):
    def fake_decode(path, frame_indices, height, width):
        return [Image.new("RGB", (width, height), (10, 20, 30)) for _ in frame_indices]

    monkeypatch.setattr("openwam.dataloader.bases.lerobot_v3_reader._decode_video_frames", fake_decode)


def _dataset(bucket: Path, stats: Path, **kwargs) -> CanonicalEEFLerobotDataset:
    return CanonicalEEFLerobotDataset(
        str(bucket),
        normalization_stats_path=str(stats),
        episode_indices=[0, 1],
        max_episodes=2,
        max_windows=8,
        num_frames=33,
        video_stride=4,
        multiview=True,
        normalize_mode="quantile",
        unify_action=True,
        unify_action_map=["0-9", "34-43"],
        **kwargs,
    )


def test_reader_emits_bounded_openwam80_sample(tmp_path: Path):
    bucket, stats = make_bucket(tmp_path)
    ds = _dataset(bucket, stats)

    assert len(ds) == 8
    sample = ds[0]
    assert len(sample["video"]) == 9
    assert all(frame.size == (320, 384) for frame in sample["video"])
    assert sample["action"].shape == (32, 80)
    assert sample["action_mask"].shape == (32, 80)
    assert sample["proprio"].shape == (1, 80)
    assert sample["proprio_mask"].shape == (1, 80)
    active = np.r_[np.arange(0, 10), np.arange(34, 44)]
    assert sample["action_mask"][0].nonzero().flatten().tolist() == active.tolist()
    assert sample["proprio_mask"][0].nonzero().flatten().tolist() == active.tolist()
    assert sample["prompt"] == "Move the red block."
    assert "trace" not in sample


def test_trace_and_gather_round_trip(tmp_path: Path):
    bucket, stats = make_bucket(tmp_path)
    ds = _dataset(bucket, stats, return_trace=True)
    sample = ds[0]

    trace = sample["trace"]
    assert trace["dataset_dir"] == str(bucket.resolve())
    assert trace["episode_index"] == 0
    assert trace["raw_action_eef20"].shape == (32, 20)
    assert trace["normalized_action_eef20"].shape == (32, 20)
    np.testing.assert_allclose(ds.gather_unified_eef(sample["action"].numpy()), trace["normalized_action_eef20"])
    np.testing.assert_allclose(
        ds.denormalize_eef(trace["normalized_action_eef20"], stream="action"),
        trace["raw_action_eef20"],
        atol=1e-5,
    )


def test_missing_eef_fails_before_sampling(tmp_path: Path):
    bucket, stats = make_bucket(tmp_path, include_eef=False)
    with pytest.raises(ValueError, match=r"observation\.states\.eef.*actions\.eef"):
        _dataset(bucket, stats)


def test_stats_provenance_mismatch_is_fatal(tmp_path: Path):
    bucket, stats = make_bucket(tmp_path)
    payload = np.load(stats, allow_pickle=True).item()
    payload["metadata"]["episode_indices"] = [0, 2]
    np.save(stats, payload, allow_pickle=True)
    with pytest.raises(ValueError, match="episode_indices"):
        _dataset(bucket, stats)


def test_nonfinite_eef_fails_without_retrying_another_window(tmp_path: Path):
    bucket, stats = make_bucket(tmp_path)
    table_path = bucket / "data" / "chunk-000" / "file-000.parquet"
    frame = pq.read_table(table_path).to_pandas()
    bad = np.asarray(frame.at[0, ACTION_KEY]).copy()
    bad[0] = np.nan
    frame.at[0, ACTION_KEY] = bad
    pq.write_table(pa.Table.from_pandas(frame), table_path)
    ds = _dataset(bucket, stats)
    with pytest.raises(ValueError, match="non-finite"):
        ds[0]


def test_separate_video_root_resolves_companion_metadata_layout(tmp_path: Path, monkeypatch):
    bucket, stats = make_bucket(tmp_path)
    video_root = tmp_path / "original_video_bucket"
    seen = []

    def record_decode(path, frame_indices, height, width):
        seen.append(Path(path))
        return [Image.new("RGB", (width, height)) for _ in frame_indices]

    monkeypatch.setattr("openwam.dataloader.bases.lerobot_v3_reader._decode_video_frames", record_decode)
    ds = _dataset(bucket, stats, video_root=str(video_root))
    ds[0]
    assert seen
    assert all(path.is_relative_to(video_root) for path in seen)
