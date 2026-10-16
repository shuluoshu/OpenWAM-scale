#!/usr/bin/env python3
"""Rebuild curated fixtures with controller/human gripper GT and filtered stats.

Run with python -m scripts.data.prepare_curated_gripper_commands --manifest ...
--output-root ... . The output must be a new directory. Existing training data,
source Parquet, and decoded canvases are never modified.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq
from omegaconf import OmegaConf

from openwam.dataloader.utils.gripper import GripperCommands, FILTER_DEFAULTS
from scripts.data.prepare_openwam_relative_stats import prepare_relative_stats


def prepare(manifest, output_root):
    manifest, output_root = Path(manifest).resolve(), Path(output_root).resolve()
    if output_root.exists():
        raise ValueError(f"output must be new (preserve historical runs): {output_root}")
    output_root.mkdir(parents=True)
    rows = json.loads(manifest.read_text())
    reports = []
    for row in rows:
        old = manifest.parent / "data" / row["id"]
        dest = output_root / "data" / row["id"]
        dest.mkdir(parents=True)
        shutil.copytree(old / "dataset", dest / "dataset", symlinks=True)
        shutil.copyfile(old / "video_offsets.json", dest / "video_offsets.json")
        (dest / "canvases.npy").symlink_to(old / "canvases.npy")
        source_info = json.loads((Path(row["source"]) / "meta/info.json").read_text())
        commands = GripperCommands(source_info)
        # Read context for stored shifted commands before cutting the clip.
        n = row["end"] - row["start"]
        offsets = [s["row_offset"] for s in commands.sources.values()] or [0]
        lo, hi = max(0, row["start"] + min(0, *offsets)), row["end"] + max(0, *offsets)
        expr = ((pads.field("episode_index") == row["episode"]) &
                (pads.field("frame_index") >= lo) & (pads.field("frame_index") < hi))
        frame = pads.dataset(row["source_parquet"], format="parquet").to_table(
            columns=["frame_index", *commands.columns], filter=expr).sort_by([("frame_index", "ascending")])
        frame_ids = frame["frame_index"].to_numpy()
        if len(frame_ids) == 0 or not np.array_equal(frame_ids, np.arange(lo, lo+len(frame_ids))):
            raise ValueError(f"noncontiguous source command context: {row['id']}")
        aligned, valid = commands.align({k: frame[k].to_pylist() for k in commands.columns}, len(frame))
        aligned = aligned[row["start"]-lo:row["end"]-lo].copy()
        valid = valid[row["start"]-lo:row["end"]-lo]
        assert len(aligned) == n
        aligned[~valid] = np.nan
        key = "audit.gripper_command"
        info_path = dest / "dataset/meta/info.json"
        info = json.loads(info_path.read_text())
        info["features"][key] = dict(shape=[2], dtype="float32", names=["left_gripper", "right_gripper"],
                                    gripper_unit="openness_0_closed_1_open")
        info["gripper_command_preparation"] = dict(source=row["source"], episode=row["episode"],
                                                  start=row["start"], end=row["end"],
                                                  sources=commands.sources,
                                                  alignment="robot_same_tick_cmd_or_human_stored_future_target_with_source_context")
        info_path.write_text(json.dumps(info, indent=2))
        path = dest / "dataset/data/chunk-000/file-000.parquet"
        table = pq.read_table(path)
        assert len(table) == n
        table = table.append_column(key, pa.array(aligned.tolist()))
        pq.write_table(table, path)
        cfg = OmegaConf.load(old / "data.yaml")
        cfg.dataset_dir = str(dest / "dataset")
        cfg.normalization_stats_path = str(dest / "stats.npy")
        cfg.gripper_command_sources = {
            side: dict(key=key, index=side_id,
                       provenance=f"source context aligned: {json.dumps(commands.sources[side], sort_keys=True)}",
                       **({"kind": commands.sources[side]["kind"]} if "kind" in commands.sources[side] else {}))
            for side_id, side in enumerate(("left", "right")) if side in commands.sources}
        cfg.gripper_state_filter = FILTER_DEFAULTS
        OmegaConf.save(cfg, dest / "data.yaml")
        stats = prepare_relative_stats(cfg)
        quality = stats["metadata"]["gripper_filter_report"]
        reports.append(dict(id=row["id"], family=row["family"], source_commands=commands.sources,
                            **quality, command_rows=valid.sum(axis=0).tolist(),
                            gripper_gt_cells=stats["action"]["counts"][[9, 19]].tolist()))
        print(row["id"], quality["accepted_windows"], "/", quality["total_windows"],
              "cmd sides", list(commands.sources), flush=True)
    (output_root / "data_manifest.json").write_text(json.dumps(rows, indent=2))
    report = dict(source_manifest=str(manifest), filter=FILTER_DEFAULTS, clips=reports,
                  total_windows=sum(r["total_windows"] for r in reports),
                  accepted_windows=sum(r["accepted_windows"] for r in reports),
                  rejected_windows=sum(r["rejected_windows"] for r in reports))
    (output_root / "gripper_filter_report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    report = prepare(args.manifest, args.output_root)
    print({key: report[key] for key in ("total_windows", "accepted_windows", "rejected_windows")})
