#!/usr/bin/env python3
"""Run one auditable offline OpenWAM open-loop prediction from a dataset profile."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from openwam.dataloader.registry import build_dataset


def load_from_checkpoint_dir(ckpt_dir: str, device: str = "cuda"):
    """Import the heavyweight deploy stack only for a real model run."""

    from openwam.deploy.model_loader import load_from_checkpoint_dir as _load

    return _load(ckpt_dir, device=device)


def _make_sync_schedule(architecture, denoise_steps: int, shift_action: float, shift_video: float):
    architecture.video_scheduler.set_timesteps(denoise_steps, shift=shift_video)
    architecture.action_scheduler.set_timesteps(denoise_steps, shift=shift_action)
    video = architecture.video_scheduler.timesteps.tolist()
    action = architecture.action_scheduler.timesteps.tolist()
    return list(zip(video, action)) + [(0.0, 0.0)]


def _numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def run_open_loop(
    profile_path: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    *,
    sample_index: int = 0,
    device: str = "cuda:0",
    denoise_steps: int = 2,
    seed: int = 42,
    model_loader=load_from_checkpoint_dir,
) -> dict:
    """Load one real sample, generate an action chunk, and persist evidence."""

    profile_path = Path(profile_path).expanduser().resolve()
    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not profile_path.is_file():
        raise FileNotFoundError(profile_path)
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(checkpoint_dir)
    if denoise_steps <= 0:
        raise ValueError("denoise_steps must be positive")

    profile = OmegaConf.load(profile_path)
    OmegaConf.update(profile, "return_trace", True, force_add=True)
    dataset = build_dataset(profile, split="train")
    if not 0 <= sample_index < len(dataset):
        raise IndexError(f"sample_index={sample_index} outside dataset length {len(dataset)}")
    sample = dataset[sample_index]
    if "trace" not in sample:
        raise ValueError("open-loop profile must produce trace metadata")

    _checkpoint_cfg, architecture = model_loader(str(checkpoint_dir), device=device)
    width = int(sample["action"].shape[-1])
    model_width = getattr(architecture, "action_dim", width)
    if model_width != width:
        raise ValueError(f"checkpoint action_dim={model_width}, reader width={width}; use a checkpoint trained with this representation")
    shift_action = float(getattr(getattr(architecture, "action_backbone", None), "shift_action", 5.0))
    shift_video = float(getattr(getattr(architecture, "video_backbone", None), "shift_video", shift_action))
    schedule = _make_sync_schedule(architecture, int(denoise_steps), shift_action, shift_video)

    active_action_mask = sample["action_mask"].any(dim=0).to(dtype=torch.bool)
    started = time.perf_counter()
    result = architecture.generate(
        schedule=schedule,
        prompt=sample["prompt"],
        vace_video=sample.get("vace_video"),
        first_frame_image=sample["first_frame_image"],
        num_frames=len(sample["video"]),
        action_num_frames=int(sample["action"].shape[0]) + 1,
        height=int(OmegaConf.select(profile, "height", default=384)),
        width=int(OmegaConf.select(profile, "width", default=320)),
        seed=int(seed),
        tiled=True,
        num_inference_steps=int(denoise_steps),
        shift=shift_action,
        dit_cache=None,
        decode_video=False,
        profile=False,
        vace_cache=None,
        prompt_embed_cache=None,
        proprio=sample["proprio"],
        cfg_scale=1.0,
        cfg_merge=False,
        active_action_mask=active_action_mask,
    )
    elapsed = time.perf_counter() - started

    prediction_unified = _numpy(result["actions"]).astype(np.float32)
    if prediction_unified.shape != tuple(sample["action"].shape):
        raise ValueError(f"model prediction must match reader shape {tuple(sample['action'].shape)}, got {prediction_unified.shape}")
    if not np.isfinite(prediction_unified).all():
        raise ValueError("model prediction contains non-finite values")
    prediction20 = dataset.gather_unified_eef(prediction_unified).astype(np.float32)
    prediction_physical = dataset.denormalize_eef(prediction20, stream="action")

    gt_unified = _numpy(sample["action"]).astype(np.float32)
    gt20 = dataset.gather_unified_eef(gt_unified).astype(np.float32)
    gt_physical = dataset.denormalize_eef(gt20, stream="action")
    arrays = {
        f"prediction_normalized{width}": prediction_unified,
        "prediction_normalized_eef20": prediction20,
        "prediction_physical_eef20": prediction_physical,
        f"gt_normalized{width}": gt_unified,
        "gt_normalized_eef20": gt20,
        "gt_physical_eef20": gt_physical,
        "action_mask": _numpy(sample["action_mask"]).astype(bool),
        f"proprio_normalized{width}": _numpy(sample["proprio"]).astype(np.float32),
        "proprio_mask": _numpy(sample["proprio_mask"]).astype(bool),
    }
    if width == 112:
        head_mask = arrays["action_mask"][:, 94:112]
        for prefix, values in (("prediction", prediction_unified), ("gt", gt_unified)):
            arrays[f"{prefix}_normalized_head18"] = values[:, 94:112].copy()
            physical = dataset.denormalize_head(values[:, 94:112], stream="action")
            physical[~head_mask] = 0
            arrays[f"{prefix}_physical_head18"] = physical
    for name, value in arrays.items():
        if value.dtype.kind == "f" and not np.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values")

    trace = sample["trace"]
    manifest = {
        "profile_path": str(profile_path),
        "checkpoint_dir": str(checkpoint_dir),
        "dataset_dir": trace.get("dataset_dir"),
        "dataset_id": trace.get("dataset_id"),
        "episode_index": trace.get("episode_index"),
        "window_offset": trace.get("window_offset"),
        "action_target_source": trace.get("action_target_source"),
        "action_target_state_key": trace.get("action_target_state_key"),
        "action_target_offset": trace.get("action_target_offset"),
        "action_target_frame_indices": trace.get("action_target_frame_indices"),
        "sample_index": int(sample_index),
        "prompt": sample["prompt"],
        "seed": int(seed),
        "denoise_steps": int(denoise_steps),
        "elapsed_seconds": float(elapsed),
        "prediction_shape": list(prediction_unified.shape),
        "prediction_eef20_shape": list(prediction20.shape),
        "active_action_indices": active_action_mask.nonzero().flatten().tolist(),
        "stats_path": trace.get("stats_path"),
        "stats_metadata": trace.get("stats_metadata"),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    npz_tmp = output_dir / "prediction.tmp.npz"
    np.savez_compressed(npz_tmp, **arrays)
    npz_tmp.replace(output_dir / "prediction.npz")
    json_tmp = output_dir / "manifest.json.tmp"
    json_tmp.write_text(json.dumps(_json_safe(manifest), indent=2, ensure_ascii=False), encoding="utf-8")
    json_tmp.replace(output_dir / "manifest.json")
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--denoise-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = run_open_loop(
        args.profile,
        args.checkpoint_dir,
        args.output_dir,
        sample_index=args.sample_index,
        device=args.device,
        denoise_steps=args.denoise_steps,
        seed=args.seed,
    )
    print(json.dumps(_json_safe(manifest), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
