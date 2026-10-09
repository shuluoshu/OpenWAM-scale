from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.data.prepare_openwam_eef_smoke import prepare_stats
from tests.dataloader.test_canonical_eef_lerobot import ACTION_KEY, STATE_KEY, make_bucket


def test_prepare_stats_scans_only_selected_episodes(tmp_path: Path):
    bucket, _ = make_bucket(tmp_path)
    output = tmp_path / "generated" / "stats.npy"

    payload = prepare_stats(
        bucket,
        output,
        episode_indices=[0, 1],
        state_key=STATE_KEY,
        action_key=ACTION_KEY,
        seed=42,
    )

    assert output.is_file()
    assert payload["eef"]["count"] == 78
    assert payload["eef_state"]["count"] == 80
    assert payload["metadata"]["episode_indices"] == [0, 1]
    assert payload["metadata"]["action_rows"] == 78
    assert payload["metadata"]["state_rows"] == 80
    assert payload["metadata"]["seed"] == 42
    assert payload["metadata"]["action_target_source"] == "next_state"
    assert payload["metadata"]["action_target_offset"] == 1
    assert payload["eef"]["mean"][0] == pytest.approx(20.0)
    assert payload["eef"]["min"][0] == 1.0
    assert payload["eef"]["max"][0] == 39.0
    assert len(payload["metadata"]["schema_fingerprint"]) == 64
    for key, expected in (("mean", 0.0), ("std", 1.0), ("min", -1.0), ("max", 1.0), ("q01", -1.0), ("q99", 1.0)):
        np.testing.assert_allclose(np.asarray(payload["eef"][key])[[3, 8, 13, 18]], expected)


def test_prepare_stats_is_deterministic_except_timestamp(tmp_path: Path):
    bucket, _ = make_bucket(tmp_path)
    first = prepare_stats(bucket, tmp_path / "a.npy", episode_indices=[1, 0], state_key=STATE_KEY, action_key=ACTION_KEY)
    second = prepare_stats(bucket, tmp_path / "b.npy", episode_indices=[0, 1], state_key=STATE_KEY, action_key=ACTION_KEY)

    assert first["metadata"]["episode_indices"] == second["metadata"]["episode_indices"] == [0, 1]
    for stream in ("eef", "eef_state"):
        for key in ("mean", "std", "min", "max", "q01", "q99"):
            np.testing.assert_array_equal(first[stream][key], second[stream][key])


def test_prepare_stats_rejects_empty_or_unknown_episode_selection(tmp_path: Path):
    bucket, _ = make_bucket(tmp_path)
    with pytest.raises(ValueError, match="at least one episode"):
        prepare_stats(bucket, tmp_path / "empty.npy", episode_indices=[], state_key=STATE_KEY, action_key=ACTION_KEY)
    with pytest.raises(ValueError, match="not present"):
        prepare_stats(bucket, tmp_path / "missing.npy", episode_indices=[99], state_key=STATE_KEY, action_key=ACTION_KEY)


def test_stats_do_not_read_precomputed_actions(tmp_path: Path):
    bucket, _ = make_bucket(tmp_path)
    data_path = bucket / "data/chunk-000/file-000.parquet"
    pq.write_table(pq.read_table(data_path).drop([ACTION_KEY]), data_path)
    payload = prepare_stats(bucket, tmp_path / "stats.npy", episode_indices=[0])
    assert payload["eef"]["count"] == 39
    assert payload["eef"]["mean"][0] == pytest.approx(20.0)


def test_stats_reject_episode_without_next_state(tmp_path: Path):
    bucket, _ = make_bucket(tmp_path)
    episode_path = bucket / "meta/episodes/chunk-000/file-000.parquet"
    episodes = pq.read_table(episode_path).to_pandas()
    episodes.loc[0, "length"] = 1
    pq.write_table(pa.Table.from_pandas(episodes), episode_path)
    with pytest.raises(ValueError, match="at least two states"):
        prepare_stats(bucket, tmp_path / "stats.npy", episode_indices=[0])
