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
    info["features"]["cmd"] = {"shape": [2]}
    info_path.write_text(json.dumps(info))
    path = bucket / "data/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    table = table.append_column("joints", pa.array([(np.arange(12) + float(i % 40)).tolist() for i in range(120)]))
    table = table.append_column("cmd", pa.array([[.8, .2]] * 120))
    pq.write_table(table, path)
    cfg = dict(type="canonical_eef_lerobot", action_representation="tcp_relative_joint94", dataset_dir=str(bucket),
               normalization_stats_path=str(tmp_path / "relative.npy"), state_key=STATE_KEY,
               joint_state_key="joints", tcp_from_eef={"left": np.eye(4).tolist()},
               num_frames=33, video_stride=4, episode_indices=[0, 1], max_episodes=2,
               max_windows=None, return_trace=True, normalize_mode=None, unify_action=True,
               unify_action_map=["0-9", "34-43", "80-93"])
    cfg.update(gripper_state_filter={"enabled": False},
               gripper_command_sources={"left": {"key": "cmd", "index": 0}, "right": {"key": "cmd", "index": 1}})
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
    cfg = make_relative_bucket(tmp_path, tcp_from_eef={}, joint_state_key="absent", state_key="absent_eef", gripper_command_sources={})
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


@pytest.mark.parametrize("mode", [None, "quantile", "min-max", "z-score"])
@pytest.mark.parametrize("layout", [94, 112])
def test_gripper_is_clipped_openness_in_state_and_action_for_every_mode(tmp_path, mode, layout):
    extra = {}
    if layout == 112:
        extra = dict(action_representation="head_tcp_relative_joint112",
                     unify_action_map=["0-9", "34-43", "80-93", "94-111"])
    cfg = make_relative_bucket(tmp_path, normalize_mode=mode, **extra)
    path = Path(cfg["dataset_dir"]) / "data/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    eef = np.asarray(table[STATE_KEY].to_pylist(), dtype=np.float32)
    eef[:, 9] = .25
    eef[:, 19] = 1  # Constant fully-open gripper must stay 1 under every mode.
    eef[:6, 9] = [.35, -.2, .25, 1.4, np.nan, np.inf]
    eef[0, 19] = 1.2
    table = table.set_column(table.column_names.index(STATE_KEY), STATE_KEY, pa.array(eef.tolist()))
    cmd = np.full((120, 2), [.25, 1], dtype=np.float32)
    cmd[:5, 0] = [-.2, .25, 1.4, np.nan, np.inf]
    table = table.set_column(table.column_names.index("cmd"), "cmd", pa.array(cmd.tolist()))
    pq.write_table(table, path)
    prepare_relative_stats(cfg)
    ds = build_dataset(cfg)
    sample = ds[0]
    np.testing.assert_array_equal(sample["proprio"][0, [9, 43]], np.array([.35, 1], np.float32))
    np.testing.assert_array_equal(sample["action"][:5, 9], [0, .25, 1, 0, 0])
    np.testing.assert_array_equal(sample["action_mask"][:5, 9], [True, True, True, False, False])
    np.testing.assert_array_equal(sample["action"][:, 43], np.ones(32))
    assert sample["action_mask"][:, 43].all()
    np.testing.assert_array_equal(ds[2]["proprio"][0, [9, 43]], sample["action"][1, [9, 43]])
    assert not ds[4]["proprio_mask"][0, 9]
    assert ds[4]["proprio"][0, 9] == 0
    for stream, key in [("action", "action"), ("state", "proprio")]:
        values = ds.gather_unified_eef(sample[key])
        restored = ds.denormalize_eef(values, stream)
        np.testing.assert_array_equal(restored[..., [9, 19]], values[..., [9, 19]])
        prediction = values.copy()
        prediction[..., 9], prediction[..., 19] = -.3, 1.3
        clipped = ds.denormalize_eef(prediction, stream)
        assert (clipped[..., 9] == 0).all() and (clipped[..., 19] == 1).all()
    tail = ds[38]
    np.testing.assert_array_equal(tail["action"][0, [9, 43]], [.25, 1])
    assert not tail["action"][1:].any() and not tail["action_mask"][1:].any()


def test_gripper_does_not_depend_on_clip_specific_stats(tmp_path):
    cfg = make_relative_bucket(tmp_path, normalize_mode="quantile")
    first = build_dataset(cfg)[0]
    path = cfg["normalization_stats_path"]
    payload = np.load(path, allow_pickle=True).item()
    for stream in ("action", "state"):
        payload[stream]["q01"][[9, 19]] = -.5
        payload[stream]["q99"][[9, 19]] = 3
    np.save(path, payload)
    second = build_dataset(cfg)[0]
    for key in ("action", "proprio", "action_mask", "proprio_mask"):
        np.testing.assert_array_equal(first[key], second[key])


def test_spike_filter_remaps_windows_video_and_stats_together(tmp_path):
    cfg = make_relative_bucket(tmp_path)
    path = Path(cfg["dataset_dir"]) / "data/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    eef = np.array(table[STATE_KEY].to_pylist())
    eef[:, [9, 19]] = .5
    eef[20, 9] = 1
    table = table.set_column(table.column_names.index(STATE_KEY), STATE_KEY, pa.array(eef.tolist()))
    pq.write_table(table, path)
    cfg["gripper_state_filter"] = {"enabled": True}
    cfg["max_windows"] = 2  # Cap after filtering; stats still cover all accepted anchors.
    payload = prepare_relative_stats(cfg)
    ds = build_dataset(cfg)
    assert len(ds) == 2
    assert ds.gripper_filter_report["rejected_windows"] == 22
    assert payload["state"]["count"] == 56
    assert payload["metadata"]["gripper_filter_report"] == ds.gripper_filter_report
    seen = []
    def decode(row, ep, offset, indices, idx):
        seen.append((ep, offset, idx, indices.tolist()))
        return [None] * 9
    ds._decode_window_video = decode
    first = ds[0]
    assert first["trace"]["source_sample_index"] == 22
    assert first["trace"]["window_offset"] == 22
    assert seen[0][:3] == (0, 22, 22)
    assert first["trace"]["action_target_frame_indices"][0] == 23
    assert first["trace"]["gripper_command_frame_indices"][0] == 22
    np.testing.assert_allclose(first["action"][0, 80:86], 1)
    assert ds._fail_count == 0
    assert all(not (i <= 20 < min(i+33, 40)) for i in ds._valid_source_indices if i < 39)
    cfg["gripper_state_filter"]["max_step"] = .2
    with pytest.raises(ValueError, match="provenance"):
        build_dataset(cfg)


def test_human_target_echo_keeps_supervision_but_cannot_exempt_state_spike(tmp_path):
    cfg = make_relative_bucket(tmp_path, tcp_from_eef={}, joint_state_key="absent")
    path = Path(cfg["dataset_dir"]) / "data/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    eef = np.array(table[STATE_KEY].to_pylist())
    eef[:, [9, 19]] = .3
    eef[20, 9] = .9
    target = np.full((120, 2), .3)
    target[19, 0] = .9  # Proxy mirrors the future measurement, including its spike.
    table = table.set_column(table.column_names.index(STATE_KEY), STATE_KEY, pa.array(eef.tolist()))
    table = table.set_column(table.column_names.index("cmd"), "cmd", pa.array(target.tolist()))
    pq.write_table(table, path)
    for spec in cfg["gripper_command_sources"].values():
        spec["kind"] = "human_openness_target"
    cfg["gripper_state_filter"] = {"enabled": True}
    prepare_relative_stats(cfg)
    ds = build_dataset(cfg)
    assert ds.gripper_filter_report["rejected_windows"] > 0
    assert all(not (i <= 20 < min(i+33, 40)) for i in ds._valid_source_indices if i < 39)
    sample = ds[0]
    assert sample["action_mask"][:, [9, 43]].any()
    assert not sample["action_mask"][:, list(range(9)) + list(range(34, 43)) + list(range(80, 94))].any()
    tail = ds[len(ds)-1]
    assert tail["action_mask"][0, [9, 43]].all()
    assert not tail["action_mask"][1:].any()


def test_relative_reader_observation_dropout_preserves_targets_and_stats(tmp_path):
    cfg=make_relative_bucket(tmp_path)
    before=build_dataset(cfg)[7]
    cfg['observation_augmentation']={'state_dropout':{'p':1}}
    ds=build_dataset(cfg)  # Observation changes do not require new GT stats.
    after=ds[7]
    assert not after['proprio'].any() and not after['proprio_mask'].any()
    for key in ['action','action_mask','video_mask']:
        np.testing.assert_array_equal(before[key],after[key])
    assert after['_observation_augmentation']['state_dropped']
    ds.observation_augmentation.eval()
    np.testing.assert_array_equal(ds[7]['proprio'],before['proprio'])


def test_shifted_commands_use_episode_context_not_window_boundary(tmp_path):
    cfg = make_relative_bucket(tmp_path)
    path = Path(cfg["dataset_dir"]) / "data/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    cmd = np.column_stack((np.arange(120) / 200, np.arange(120) / 300))
    table = table.set_column(table.column_names.index("cmd"), "cmd", pa.array(cmd.tolist()))
    pq.write_table(table, path)
    for source in cfg["gripper_command_sources"].values():
        source["row_offset"] = -1
    payload = prepare_relative_stats(cfg)
    ds = build_dataset(cfg)
    np.testing.assert_allclose(ds[7]["action"][0, [9, 43]], cmd[6])
    assert ds[7]["action_mask"][0, [9, 43]].all()
    assert not ds[0]["action_mask"][0, [9, 43]].any()
    assert not ds[39]["action_mask"][0, [9, 43]].any()  # No cross-episode carry.
    np.testing.assert_allclose(ds[40]["action"][0, [9, 43]], cmd[40])
    assert payload["action"]["counts"][9] == payload["action"]["count"] - 2


def test_all_spiky_windows_are_explicitly_empty_without_retry(tmp_path):
    cfg = make_relative_bucket(tmp_path)
    path = Path(cfg["dataset_dir"]) / "data/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    eef = np.array(table[STATE_KEY].to_pylist())
    eef[:, 9] = np.arange(120) % 2
    table = table.set_column(table.column_names.index(STATE_KEY), STATE_KEY, pa.array(eef.tolist()))
    pq.write_table(table, path)
    cfg["gripper_state_filter"] = {"enabled": True}
    payload = prepare_relative_stats(cfg)
    ds = build_dataset(cfg)
    assert len(ds) == 0 and len(ds._valid_source_indices) == 0
    assert ds.gripper_filter_report["rejected_windows"] == 78
    assert payload["action"]["count"] == 0
    assert ds._fail_count == 0


def test_hour_subsample_rebuilds_filtered_index_and_command_cache(tmp_path):
    cfg = make_relative_bucket(tmp_path)
    ds = build_dataset(cfg)
    ds._episode_grippers(1)  # Populate cache before indices/episodes change.
    ds._apply_effective_hour_budget(40 / 30 / 3600, seed=123)
    assert len(ds._eps_df) == 1 and len(ds) == 39
    assert len(ds._valid_source_indices) == 39
    assert ds[0]["trace"]["episode_index"] == int(ds._eps_df.iloc[0].episode_index)
    np.testing.assert_allclose(ds[0]["action"][0, [9, 43]], [.8, .2])


def test_nonfinite_measured_gripper_is_masked_not_a_clipped_spike(tmp_path):
    cfg = make_relative_bucket(tmp_path)
    path = Path(cfg["dataset_dir"]) / "data/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    eef = np.array(table[STATE_KEY].to_pylist())
    eef[:, [9, 19]] = .5
    eef[20, 9] = np.inf
    table = table.set_column(table.column_names.index(STATE_KEY), STATE_KEY, pa.array(eef.tolist()))
    pq.write_table(table, path)
    cfg["gripper_state_filter"] = {"enabled": True}
    prepare_relative_stats(cfg)
    ds = build_dataset(cfg)
    assert len(ds) == 78 and ds.gripper_filter_report["rejected_windows"] == 0
    assert not ds[20]["proprio_mask"][0, 9]
    assert ds[20]["action_mask"][0, 9]  # Independent finite command stays supervised.


@pytest.mark.parametrize("with_state_dropout", [False, True])
def test_legacy_jitter_bypasses_prepared_cache_and_keeps_eval_clean(tmp_path, with_state_dropout):
    import random
    import torch
    from PIL import Image
    from openwam.dataloader.transforms.observation import TRACE_KEY
    from openwam.train.utils.observation_cache import prepare_observation

    cfg = make_relative_bucket(tmp_path, color_jitter={"brightness": .8, "contrast": 0, "saturation": 0})
    if with_state_dropout:
        cfg["observation_augmentation"] = {"state_dropout": {"p": 1}}
    ds = build_dataset(cfg)
    def decode(*args, return_missing_masks=False):
        frames = [Image.new("RGB", (24, 24), (120, 120, 120)) for _ in range(9)]
        return (frames, None) if return_missing_masks else frames
    ds._decode_window_video = decode
    calls = []
    def prepare(samples):
        calls.append(1)
        sample = samples[0]
        return {"pixel": np.asarray(sample["video"][0]).copy(),
                "proprio": sample["proprio"].clone(), "proprio_mask": sample["proprio_mask"].clone()}
    cache = {}
    random.seed(17)
    train = ds[7]
    train["_curated_index"] = 7
    assert train.get(TRACE_KEY, {}).get("image_changed"), "legacy jitter must invalidate the RGB cache"
    result = prepare_observation(train, prepare, cache)
    assert not cache
    assert not (result["pixel"] == 120).all()
    ds._color_jitter.eval()
    ds.observation_augmentation.eval()
    clean = ds[7]
    clean["_curated_index"] = 7
    result = prepare_observation(clean, prepare, cache)
    assert (result["pixel"] == 120).all()
    assert torch.equal(result["proprio"], clean["proprio"])
    assert len(calls) == 2 and 7 in cache
