#!/usr/bin/env python3
"""Compute masked statistics for all TCP-relative chunks in selected episodes."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from openwam.dataloader.mora_relative_lerobot import MoraRelativeJointDataset
from openwam.dataloader.utils.lerobotv3 import parse_info_json
from openwam.dataloader.utils.normalization import ROT6D_DIMS_EEF20, pin_rot6d_identity
from scripts.data.prepare_openwam_eef_smoke import _read_table


def _summary(values, masks):
    result = {key: np.full(values.shape[-1], default, np.float32) for key, default in
              (("mean", 0), ("std", 1), ("min", -1), ("max", 1), ("q01", -1), ("q99", 1))}
    result["counts"] = masks.sum(axis=0)
    result["count"] = len(values)
    for i in range(values.shape[-1]):
        selected = values[masks[:, i], i].astype(np.float64)
        if len(selected):
            for key, value in zip(("mean", "std", "min", "max", "q01", "q99"),
                                  (selected.mean(), selected.std(), selected.min(), selected.max(),
                                   np.quantile(selected, .01), np.quantile(selected, .99))):
                result[key][i] = value
    pin_rot6d_identity(result, (*ROT6D_DIMS_EEF20, *range(37, 43), *range(46, 52))
                       if values.shape[-1] == 52 else ROT6D_DIMS_EEF20)
    return result


class _StatsPopulation(MoraRelativeJointDataset):
    """Use the actual reader index without loading the stats being computed."""

    def _load_stats(self, info):
        return {}


def prepare_relative_stats(config, *, split="train"):
    cfg = OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else dict(config)
    bucket = Path(cfg["dataset_dir"]).expanduser().resolve()
    output = Path(cfg["normalization_stats_path"]).expanduser().resolve()
    if output.is_relative_to(bucket):
        raise ValueError("refusing to write stats inside source bucket")
    info = parse_info_json(bucket)
    population = _StatsPopulation.from_config(cfg, split=split)
    rep = population.representation
    episodes = population._eps_df
    horizon, stride = population._num_frames, population._window_stride
    if horizon < 2 or episodes.empty:
        raise ValueError("need nonempty episodes and num_frames >= 2")
    parts = [[], [], [], []]
    for ep_local, (_, ep) in enumerate(episodes.iterrows()):
        path = bucket / info["data_path"].format(chunk_index=int(ep["data/chunk_index"]), file_index=int(ep["data/file_index"]))
        frame = _read_table(path, ["episode_index", *rep.columns]).slice(int(ep["_data_row_offset"]), int(ep.length)).to_pandas()
        if len(frame) != int(ep.length) or set(frame.episode_index) != {int(ep.episode_index)}:
            raise ValueError(f"episode row range mismatch: {path}")
        columns = {k: np.stack(frame[k].tolist()) for k in rep.columns}
        columns["_rows"] = np.arange(len(frame))
        n_starts = int(population._cum_n_starts[ep_local + 1] - population._cum_n_starts[ep_local])
        for i in range(n_starts):
            start = int(population._ep_valid_start[ep_local]) + i * stride
            encoded = rep.encode_window({k: v[start:min(start + horizon, int(population._ep_valid_end[ep_local]))] for k, v in columns.items()})
            for part, arr in zip(parts, encoded):
                part.append(arr)
    if not parts[0]:
        raise ValueError("no valid next-state windows")
    action, action_mask, state, state_mask = [np.concatenate(part) for part in parts]
    payload = {"action": _summary(action, action_mask), "state": _summary(state, state_mask),
               "metadata": {**rep.contract, "dataset_dir": str(bucket),
                            "episode_indices": episodes.episode_index.astype(int).tolist(),
                            "num_frames": horizon, "window_stride": stride, "split": split,
                            "action_rows": len(action), "state_rows": len(state),
                            "population": "all window anchors and their valid future rows; ignores max_windows"}}
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(f".{output.name}.tmp.npy")
    np.save(temp, payload, allow_pickle=True)
    temp.replace(output)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--split", choices=["train", "val"], default="train")
    args = parser.parse_args()
    result = prepare_relative_stats(OmegaConf.load(args.profile), split=args.split)
    print(result["metadata"])
