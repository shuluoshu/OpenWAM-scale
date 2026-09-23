#!/usr/bin/env python3
"""Compute small, provenance-bound EEF20 stats from explicit LeRobot episodes."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from openwam.dataloader.canonical_eef_lerobot import canonical_schema_fingerprint
from openwam.dataloader.utils.lerobotv3 import compute_file_local_offsets, load_episodes_parquet, parse_info_json
from openwam.dataloader.utils.normalization import ROT6D_DIMS_EEF20, pin_rot6d_identity


DEFAULT_STATE_KEY = "observation.states.eef"
DEFAULT_ACTION_KEY = "actions.eef"
DEFAULT_CAMERAS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


def _read_table(path: Path, columns: list[str]):
    try:
        return pq.read_table(path, memory_map=True, columns=columns)
    except pa.ArrowInvalid as exc:
        if "Dot path" not in str(exc):
            raise
        return pq.read_table(path, memory_map=True).select(columns)


def _validate_eef(array: np.ndarray, *, field: str, path: Path) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 20:
        raise ValueError(f"{path}:{field} must be (N, 20), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{path}:{field} contains non-finite values")
    for start in (3, 13):
        if np.any(np.linalg.norm(values[:, start : start + 3], axis=1) < 1e-6):
            raise ValueError(f"{path}:{field} contains a degenerate rot6d first column")
        if np.any(np.linalg.norm(values[:, start + 3 : start + 6], axis=1) < 1e-6):
            raise ValueError(f"{path}:{field} contains a degenerate rot6d second column")
    return values


def _summary(values: np.ndarray) -> dict:
    stats = {
        "count": int(values.shape[0]),
        "mean": values.mean(axis=0, dtype=np.float64).astype(np.float32),
        "std": values.std(axis=0, dtype=np.float64).astype(np.float32),
        "min": values.min(axis=0).astype(np.float32),
        "max": values.max(axis=0).astype(np.float32),
        "q01": np.quantile(values.astype(np.float64), 0.01, axis=0).astype(np.float32),
        "q99": np.quantile(values.astype(np.float64), 0.99, axis=0).astype(np.float32),
    }
    pin_rot6d_identity(stats, ROT6D_DIMS_EEF20)
    return stats


def prepare_stats(
    bucket: Path,
    output: Path,
    *,
    episode_indices: list[int],
    state_key: str = DEFAULT_STATE_KEY,
    action_key: str = DEFAULT_ACTION_KEY,
    camera_keys: tuple[str, str, str] = DEFAULT_CAMERAS,
    seed: int = 42,
) -> dict:
    """Scan only ``episode_indices`` and write reader-compatible stats."""

    bucket = Path(bucket).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if not episode_indices:
        raise ValueError("at least one episode index is required")
    selected_ids = sorted(set(int(value) for value in episode_indices))
    try:
        output.relative_to(bucket)
    except ValueError:
        pass
    else:
        raise ValueError(f"refusing to write generated stats inside source bucket: {output}")

    info = parse_info_json(bucket)
    features = info.get("features", {})
    required = (state_key, action_key, *camera_keys)
    missing_fields = [key for key in required if key not in features]
    if missing_fields:
        raise ValueError(f"missing canonical EEF schema fields in {bucket}/meta/info.json: {missing_fields}")
    for key in (state_key, action_key):
        if list(features[key].get("shape", [])) != [20]:
            raise ValueError(f"{key} must declare shape [20], got {features[key].get('shape')}")

    episodes = load_episodes_parquet(bucket)
    available = set(int(value) for value in episodes["episode_index"].tolist())
    missing_episodes = sorted(set(selected_ids) - available)
    if missing_episodes:
        raise ValueError(f"episode indices not present in {bucket}: {missing_episodes}")
    episodes["_row_offset"] = compute_file_local_offsets(episodes, "data/chunk_index", "data/file_index")
    selected = episodes[episodes["episode_index"].isin(selected_ids)].sort_values("episode_index")

    action_parts: list[np.ndarray] = []
    state_parts: list[np.ndarray] = []
    template = info["data_path"]
    for _, episode in selected.iterrows():
        path = bucket / template.format(
            chunk_index=int(episode["data/chunk_index"]),
            file_index=int(episode["data/file_index"]),
        )
        table = _read_table(path, ["episode_index", state_key, action_key])
        start = int(episode["_row_offset"])
        length = int(episode["length"])
        frame = table.slice(start, length).to_pandas()
        physical = set(int(value) for value in frame["episode_index"].tolist())
        expected_episode = int(episode["episode_index"])
        if physical != {expected_episode}:
            raise ValueError(
                f"{path} row range [{start}:{start + length}] contains episode_index={sorted(physical)}, "
                f"expected only {expected_episode}"
            )
        action_parts.append(_validate_eef(np.stack(frame[action_key].to_list()), field=action_key, path=path))
        state_parts.append(_validate_eef(np.stack(frame[state_key].to_list()), field=state_key, path=path))

    actions = np.concatenate(action_parts, axis=0)
    states = np.concatenate(state_parts, axis=0)
    payload = {
        "eef": _summary(actions),
        "eef_state": _summary(states),
        "metadata": {
            "dataset_dir": str(bucket),
            "state_key": state_key,
            "action_key": action_key,
            "episode_indices": selected_ids,
            "state_width": 20,
            "action_width": 20,
            "state_rows": int(states.shape[0]),
            "action_rows": int(actions.shape[0]),
            "seed": int(seed),
            "schema_fingerprint": canonical_schema_fingerprint(
                features,
                state_key=state_key,
                action_key=action_key,
                camera_keys=camera_keys,
            ),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(f".{output.name}.tmp.npy")
    np.save(temp, payload, allow_pickle=True)
    temp.replace(output)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, action="append", dest="episode_indices", required=True)
    parser.add_argument("--state-key", default=DEFAULT_STATE_KEY)
    parser.add_argument("--action-key", default=DEFAULT_ACTION_KEY)
    parser.add_argument("--head-camera", default=DEFAULT_CAMERAS[0])
    parser.add_argument("--left-wrist-camera", default=DEFAULT_CAMERAS[1])
    parser.add_argument("--right-wrist-camera", default=DEFAULT_CAMERAS[2])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = prepare_stats(
        args.bucket,
        args.output,
        episode_indices=args.episode_indices,
        state_key=args.state_key,
        action_key=args.action_key,
        camera_keys=(args.head_camera, args.left_wrist_camera, args.right_wrist_camera),
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "episodes": payload["metadata"]["episode_indices"],
                "action_rows": payload["metadata"]["action_rows"],
                "state_rows": payload["metadata"]["state_rows"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
