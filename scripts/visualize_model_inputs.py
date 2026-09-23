#!/usr/bin/env python3
"""Render model-forward inputs for canonical EEF dataset profiles."""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

from openwam.dataloader.registry import build_dataset


def _numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    return value


def _contact_sheet(frames: list[Image.Image]) -> Image.Image:
    if len(frames) != 9:
        raise ValueError(f"expected nine processed frames, got {len(frames)}")
    sheet = Image.new("RGB", (960, 1152), "black")
    for index, frame in enumerate(frames):
        resized = frame.convert("RGB").resize((320, 384))
        x = (index % 3) * 320
        y = (index // 3) * 384
        sheet.paste(resized, (x, y))
        ImageDraw.Draw(sheet).text((x + 8, y + 8), f"processed frame {index}", fill="white")
    return sheet


def render_profile(
    name: str,
    profile_path: Path,
    output_dir: Path,
    *,
    sample_index: int = 0,
    prediction_path: Path | None = None,
) -> dict:
    """Render PNG/JSON/NPZ artifacts for one sample immediately before forward."""

    profile_path = Path(profile_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    profile = OmegaConf.load(profile_path)
    OmegaConf.update(profile, "return_trace", True, force_add=True)
    dataset = build_dataset(profile, split="train")
    if not 0 <= sample_index < len(dataset):
        raise IndexError(f"sample_index={sample_index} outside dataset length {len(dataset)}")
    sample = dataset[sample_index]
    trace = sample.get("trace")
    if trace is None:
        raise ValueError(f"profile {profile_path} did not emit trace metadata")

    output_dir.mkdir(parents=True, exist_ok=True)
    image_name = f"{name}_frames.png"
    json_name = f"{name}.json"
    npz_name = f"{name}.npz"
    _contact_sheet(sample["video"]).save(output_dir / image_name)

    arrays = {
        "raw_action_eef20": _numpy(trace["raw_action_eef20"]).astype(np.float32),
        "normalized_action_eef20": _numpy(trace["normalized_action_eef20"]).astype(np.float32),
        "unified_action80": _numpy(trace["unified_action80"]).astype(np.float32),
        "raw_proprio_eef20": _numpy(trace["raw_proprio_eef20"]).astype(np.float32),
        "normalized_proprio_eef20": _numpy(trace["normalized_proprio_eef20"]).astype(np.float32),
        "unified_proprio80": _numpy(trace["unified_proprio80"]).astype(np.float32),
        "action_mask": _numpy(sample["action_mask"]).astype(bool),
        "proprio_mask": _numpy(sample["proprio_mask"]).astype(bool),
        "video_mask": _numpy(sample["video_mask"]).astype(bool),
    }
    if prediction_path is not None:
        prediction_path = Path(prediction_path).expanduser().resolve()
        predicted = np.load(prediction_path)
        for key in (
            "prediction_normalized80",
            "prediction_normalized_eef20",
            "prediction_physical_eef20",
        ):
            if key in predicted:
                arrays[key] = predicted[key]
    np.savez_compressed(output_dir / npz_name, **arrays)

    action_mask = arrays["action_mask"]
    manifest = {
        "name": name,
        "profile_path": str(profile_path),
        "dataset_dir": trace.get("dataset_dir"),
        "dataset_id": trace.get("dataset_id"),
        "episode_index": trace.get("episode_index"),
        "window_offset": trace.get("window_offset"),
        "sample_index": int(sample_index),
        "prompt": sample["prompt"],
        "pipeline": [
            "parquet",
            "absolute EEF20",
            "source normalization",
            "scatter OpenWAM80",
            "model forward",
        ],
        "raw_frame_indices": trace.get("raw_frame_indices"),
        "video_frame_indices": trace.get("video_frame_indices"),
        "shapes": {
            "video": [len(sample["video"]), 384, 320, 3],
            "action": list(_numpy(sample["action"]).shape),
            "action_mask": list(action_mask.shape),
            "proprio": list(_numpy(sample["proprio"]).shape),
            "proprio_mask": list(arrays["proprio_mask"].shape),
        },
        "active_action_indices": np.flatnonzero(action_mask.any(axis=0)).tolist(),
        "stats_path": trace.get("stats_path"),
        "stats_metadata": trace.get("stats_metadata"),
        "summaries": {
            key: {
                "min": float(value.min()),
                "max": float(value.max()),
                "mean": float(value.mean()),
            }
            for key, value in arrays.items()
            if value.dtype.kind == "f"
        },
        "files": {"image": image_name, "npz": npz_name},
    }
    (output_dir / json_name).write_text(json.dumps(_safe(manifest), indent=2, ensure_ascii=False), encoding="utf-8")
    plot = arrays["raw_action_eef20"][:, :2].tolist()
    return {
        "status": "ok",
        "name": name,
        "prompt": sample["prompt"],
        "image": image_name,
        "json": json_name,
        "npz": npz_name,
        "plot": plot,
    }


def _svg_plot(points: list[list[float]], width: int = 520, height: int = 150) -> str:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        return "<svg viewBox='0 0 520 150'></svg>"
    lo, hi = float(values.min()), float(values.max())
    span = max(hi - lo, 1e-9)
    lines = []
    colors = ("#62d4ff", "#ffad66", "#d88cff")
    for dim in range(min(values.shape[1], len(colors))):
        coords = []
        for index, value in enumerate(values[:, dim]):
            x = 8 + index * (width - 16) / max(values.shape[0] - 1, 1)
            y = 8 + (hi - float(value)) / span * (height - 16)
            coords.append(f"{x:.1f},{y:.1f}")
        lines.append(f"<polyline fill='none' stroke='{colors[dim]}' stroke-width='2' points='{' '.join(coords)}'/>")
    return f"<svg viewBox='0 0 {width} {height}' role='img'>{''.join(lines)}</svg>"


def render_index(results: list[dict], output_dir: Path) -> Path:
    output_dir = Path(output_dir).expanduser().resolve()
    cards = []
    for result in results:
        name = html.escape(str(result.get("name", "unknown")))
        if result.get("status") != "ok":
            cards.append(f"<section><h2>{name}</h2><p class='error'>{html.escape(str(result.get('error')))}</p></section>")
            continue
        cards.append(
            "<section>"
            f"<h2>{name}</h2><p><b>Instruction:</b> {html.escape(str(result['prompt']))}</p>"
            f"<img src='{html.escape(result['image'])}' alt='{name} processed frames'>"
            f"{_svg_plot(result.get('plot', []))}"
            f"<p><a href='{html.escape(result['json'])}'>JSON trace</a> · "
            f"<a href='{html.escape(result['npz'])}'>NPZ tensors</a></p></section>"
        )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>OpenWAM-α forward-input report</title>
<style>body{{font:15px system-ui;background:#111827;color:#e5e7eb;margin:2rem auto;max-width:1100px}}section{{background:#1f2937;padding:1rem;margin:1rem 0;border-radius:10px}}img{{width:100%;height:auto}}svg{{background:#0b1020;width:100%;border-radius:6px}}a{{color:#62d4ff}}.error{{color:#ff8080}}</style>
</head><body><h1>OpenWAM-α model-forward inputs</h1>
<p>parquet → absolute EEF20 → source normalization → OpenWAM80 scatter → model forward</p>
{''.join(cards)}</body></html>"""
    page = output_dir / "index.html"
    page.write_text(document, encoding="utf-8")
    return page


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    results = []
    for profile in args.profile:
        name = profile.stem.removeprefix("canonical_eef_").removesuffix("_openloop").removesuffix("_smoke")
        try:
            results.append(render_profile(name, profile, args.output_dir, sample_index=args.sample_index))
        except Exception as exc:
            results.append({"status": "failed", "name": name, "error": f"{type(exc).__name__}: {exc}"})
    page = render_index(results, args.output_dir)
    print(page)
    if any(result["status"] != "ok" for result in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
