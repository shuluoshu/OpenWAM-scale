import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openwam.dataloader.registry import build_dataset
from scripts.data.prepare_openwam_relative_stats import prepare_relative_stats
from tests.dataloader.test_canonical_eef_lerobot import STATE_KEY, make_bucket, patch_video_decode  # noqa: F401


def make_relative_bucket(tmp_path, **extra):
    bucket, _ = make_bucket(tmp_path)
    info_path = bucket / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["features"]["joints"] = {"shape": [12], "names": [f"{side}_joint_{i}" for side in ("left", "right") for i in range(6)]}
    info_path.write_text(json.dumps(info))
    path = bucket / "data/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    table = table.append_column("joints", pa.array([(np.arange(12) + float(i % 40)).tolist() for i in range(120)]))
    pq.write_table(table, path)
    cfg = dict(type="canonical_eef_lerobot", action_representation="tcp_relative_joint94", dataset_dir=str(bucket),
               normalization_stats_path=str(tmp_path / "relative.npy"), state_key=STATE_KEY,
               joint_state_key="joints", tcp_from_eef={"left": np.eye(4).tolist()},
               num_frames=33, video_stride=4, episode_indices=[0, 1], max_episodes=2,
               max_windows=None, return_trace=True, normalize_mode=None, unify_action=True,
               unify_action_map=["0-9", "34-43", "80-93"])
    cfg.update(extra)
    prepare_relative_stats(cfg)
    return cfg


def test_reader_outputs_94d_and_independent_pose_joint_masks(tmp_path):
    ds = build_dataset(make_relative_bucket(tmp_path))
    sample = ds[7]
    assert sample["action"].shape == (32, 94)
    assert sample["proprio"].shape == (1, 94)
    np.testing.assert_allclose(sample["action"][0, :3], [1, 1, 1])
    np.testing.assert_allclose(sample["action"][1, :3], [2, 2, 2])
    np.testing.assert_allclose(sample["action"][0, 80:86], np.ones(6))
    assert not sample["action_mask"][:, 34:43].any()
    assert not sample["action"][:, 34:43].any()
    assert not sample["action_mask"][:, [86, 93]].any()
    assert sample["action_mask"][:, [9, 43]].all()
    assert len(sample["video"]) == 9
    assert sample["trace"]["action_target_frame_indices"] == list(range(8, 40))


def test_tail_keeps_only_next_state_and_stale_absolute_stats_fail(tmp_path):
    cfg = make_relative_bucket(tmp_path)
    ds = build_dataset(cfg)
    tail = ds[38]
    assert len(ds) == 78
    assert tail["action_mask"][0].any()
    assert not tail["action_mask"][1:].any() and not tail["action"][1:].any()
    path = cfg["normalization_stats_path"]
    payload = np.load(path, allow_pickle=True).item()
    payload["metadata"]["representation"] = "absolute_eef80"
    np.save(path, payload)
    with pytest.raises(ValueError, match="provenance"):
        build_dataset(cfg)


def test_stats_cover_chunk_deltas_and_exclude_masked_dimensions(tmp_path):
    cfg = make_relative_bucket(tmp_path)
    payload = np.load(cfg["normalization_stats_path"], allow_pickle=True).item()
    assert payload["action"]["max"][0] == 32  # not merely adjacent-state deltas
    assert payload["action"]["counts"][10] == 0  # unavailable right TCP
    assert payload["action"]["counts"][26] == 0  # no seventh left joint
    assert payload["action"]["counts"][20] > 0
    assert payload["action"]["q99"][3] == 1
    assert payload["metadata"]["num_frames"] == 33


def test_reader_without_any_robot_labels_keeps_video_supervision(tmp_path):
    cfg = make_relative_bucket(tmp_path, tcp_from_eef={}, joint_state_key="absent", state_key="absent_eef")
    ds = build_dataset(cfg)
    sample = ds[0]
    assert not sample["action_mask"].any() and not sample["proprio_mask"].any()
    assert not sample["action"].any() and sample["video_mask"].all()


def test_stats_follow_reader_split_exclusion_and_hour_selection(tmp_path):
    cfg = make_relative_bucket(tmp_path, episode_indices=None, max_episodes=None)
    info_path = Path(cfg["dataset_dir"]) / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["splits"] = {"train": "0:2", "val": "2:3"}
    info_path.write_text(json.dumps(info))
    excluded = info_path.parent / "excluded_episodes.json"
    excluded.write_text(json.dumps({"episode_indices": [1]}))
    payload = prepare_relative_stats(cfg)
    assert payload["metadata"]["episode_indices"] == [0]
    assert build_dataset(cfg)[0]["action"].shape == (32, 94)
    excluded.write_text(json.dumps({"episode_indices": []}))
    cfg["max_hours"] = 40 / 30 / 3600
    cfg["subsample_seed"] = 123
    payload = prepare_relative_stats(cfg)
    assert len(payload["metadata"]["episode_indices"]) == 1
    assert len(build_dataset(cfg)._eps_df) == 1


def test_quantile_normalization_does_not_leak_into_masked_slots(tmp_path):
    ds = build_dataset(make_relative_bucket(tmp_path, normalize_mode="quantile"))
    sample = ds[7]
    for values, mask in (("action", "action_mask"), ("proprio", "proprio_mask")):
        assert not sample[values][~sample[mask]].any()
        assert np.isfinite(sample[values]).all()
    # Rotation remains a valid unscaled identity rather than per-axis stats.
    np.testing.assert_array_equal(sample["action"][0, 3:9], [1, 0, 0, 0, 1, 0])


def test_dual112_reader_stats_and_missing_head_video(tmp_path):
    cfg = make_relative_bucket(tmp_path, action_representation="head_tcp_relative_joint112",
                               unify_action_map=["0-9", "34-43", "80-93", "94-111"],
                               head_world_to_camera_key="cal", normalize_mode="quantile")
    bucket = Path(cfg["dataset_dir"])
    info_path = bucket / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["features"]["cal"] = {"shape": [16]}
    info_path.write_text(json.dumps(info))
    path = bucket / "data/chunk-000/file-000.parquet"
    table = pq.read_table(path).append_column("cal", pa.array([np.eye(4).ravel().tolist()] * 120))
    pq.write_table(table, path)
    payload = prepare_relative_stats(cfg)
    ds = build_dataset(cfg)
    sample = ds[7]
    assert sample["action"].shape == (32, 112) and sample["proprio"].shape == (1, 112)
    assert sample["action_mask"][:, 94:103].all()
    assert not sample["action_mask"][:, 103:].any()
    assert not sample["action_mask"][:, 34:43].any()
    assert sample["trace"]["raw_action_joint14"].shape == (32, 14)
    np.testing.assert_allclose(sample["trace"]["raw_action_head18"][0, :3], [1, 1, 1])
    assert payload["action"]["max"][34] == 32
    np.testing.assert_array_equal(sample["action"][0, 97:103], [1, 0, 0, 0, 1, 0])
    assert not sample["action"][~sample["action_mask"]].any()
    tail = ds[38]
    assert tail["action_mask"][0, 94:103].all() and not tail["action_mask"][1:].any()
    cfg.update(head_camera=None, primary_camera="observation.images.cam_left_wrist")
    with pytest.raises(ValueError, match="provenance"):
        build_dataset(cfg)
    prepare_relative_stats(cfg)
    sample = build_dataset(cfg)[7]
    assert not sample["action_mask"][:, 94:].any()
    assert not sample["action"][:, 94:].any()
    assert sample["action_mask"][:, :9].all() and len(sample["video"]) == 9
