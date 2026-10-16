"""CPU-only paired image evidence using production functions and local Qwen processor.

This executes isolated image transforms, not the complete Mora YAML pipeline.
All model inputs originate in original video files; JPEG previews are never inputs.
"""

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import platform
import sys
import traceback
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

import av
import numpy as np
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageOps
import torch
import torchvision
import transformers
from transformers import Qwen3VLProcessor

OUT = Path(__file__).resolve().parent
REPORT = OUT.parent
ROOT = REPORT.parents[1]
MORA = (ROOT / "../../pretrain").resolve()
torch.set_num_threads(2)


def load_module(name, path):
    """Load the real leaf module without importing unrelated training packages."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RESIZE_PATH = MORA / "src/data/modules/transforms/utils.py"
MORA_VIDEO_PATH = MORA / "src/data/modules/datasets/video_utils.py"
OPENWAM_VIDEO_PATH = ROOT / "openwam/dataloader/utils/video_io.py"
LAYOUT_PATH = ROOT / "openwam/dataloader/transforms/multiview.py"
resize = load_module("paired_mora_resize", RESIZE_PATH).resize_with_longest_edge
mora_decode = load_module("paired_mora_video", MORA_VIDEO_PATH).decode_video_frames
openwam_decode = load_module("paired_openwam_video", OPENWAM_VIDEO_PATH).decode_video_frames
layout = load_module("paired_openwam_layout", LAYOUT_PATH).assemble_multiview_layout


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")


def tensor_image(tensor):
    """Quantize a CHW float RGB tensor only for display; never feed this back."""
    data = tensor.detach().cpu().permute(1, 2, 0).numpy()
    return Image.fromarray(np.rint(np.clip(data, 0, 1) * 255).astype(np.uint8))


def unpatch(pixels, grid, processor):
    """Invert the actual Qwen merge-group/channel/temporal/spatial patch ordering."""
    grid_t, grid_h, grid_w = [int(x) for x in grid]
    patch = processor.patch_size
    merge = processor.merge_size
    temporal = processor.temporal_patch_size
    values = pixels.reshape(grid_t, grid_h // merge, grid_w // merge, merge, merge, 3, temporal, patch, patch)
    values = values.permute(0, 6, 5, 1, 3, 7, 2, 4, 8)
    return values.reshape(grid_t * temporal, 3, grid_h * patch, grid_w * patch)


def comparison_preview(dest, clip, camera_results):
    """Create a labeled report layout; this layout is never supplied to Mora."""
    rows = len(camera_results)
    canvas = Image.new("RGB", (1000, 65 + rows * 300), "#101820")
    draw = ImageDraw.Draw(canvas)
    draw.text((15, 10), "Report layout only: Mora uses independent images, not a canvas", fill="white")
    for index, label in enumerate(["Same original video frame", "Mora: final RGB before normalization", "OpenWAM: actual slot"]):
        draw.text((index * 333 + 12, 38), label, fill="white")
    for row, (camera, record) in enumerate(camera_results.items()):
        paths = [dest / record["files"]["source"], dest / record["files"]["final"], clip / f"resized_{camera}.png"]
        for col, path in enumerate(paths):
            image = Image.open(path).convert("RGB")
            preview = ImageOps.contain(image, (315, 255))
            canvas.paste(preview, (col * 333 + 10, 65 + row * 300))
            draw.text((col * 333 + 10, 325 + row * 300), f"{camera}: {image.width} x {image.height}", fill="white")
    canvas.save(dest / "image_comparison.jpg", quality=92)


def run_clip(result, image_processor):
    """Decode one anchor, invoke both image paths, and assert numerical alignment."""
    identifier = result["id"]
    clip = REPORT / "clips" / identifier
    dest = OUT / identifier
    dest.mkdir(parents=True, exist_ok=True)
    episode = json.loads((clip / "source_episode.json").read_text())
    offsets = json.loads((clip / "video_offsets.json").read_text())
    info = json.loads((clip / "dataset/meta/info.json").read_text())
    camera_results, slots, missing = {}, {}, []
    for role in ["head", "left", "right"]:
        camera = result[role]
        if camera is None:
            missing.append(role)
            continue
        prefix = f"videos/{camera}/"
        video = Path(result["source"]) / info["video_path"].format(
            video_key=camera, chunk_index=episode[prefix + "chunk_index"], file_index=episode[prefix + "file_index"]
        )
        # Explicit file open preserves ordinary sandbox permission failures.
        with video.open("rb") as stream:
            stream.read(1)
        with av.open(str(video), options={"hwaccel": "none"}) as container:
            native_width = container.streams.video[0].width
            native_height = container.streams.video[0].height
        frame_index = offsets[camera]["clip_base"]
        source = openwam_decode(str(video), [frame_index], native_height, native_width)[0].convert("RGB")
        source_array = np.array(source)
        raw = torch.from_numpy(source_array.copy()).permute(2, 0, 1).float() / 255
        timestamp = float(episode[prefix + "from_timestamp"]) + result["start"] / result["fps"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            mora_raw = mora_decode(str(video), [timestamp], tolerance_s=0.1, backend="pyav")[0]
        decoder_error = float((mora_raw - raw).abs().max())
        # Record any timestamp-vs-index discrepancy explicitly; the paired transforms
        # always receive the same index-anchored source pixels regardless of decoder.
        stage1 = resize(raw, longest_edge=336, mode="bilinear")
        processed = image_processor(stage1, do_rescale=False, device="cpu", return_tensors="pt")
        pixels = processed.pixel_values
        grid = processed.image_grid_thw[0]
        normalized_frames = unpatch(pixels, grid, image_processor)
        assert normalized_frames.shape[0] == image_processor.temporal_patch_size
        temporal_error = float((normalized_frames - normalized_frames[:1]).abs().max())
        assert temporal_error == 0
        normalized = normalized_frames[0]
        # Independent re-patching through the real processor verifies the inverse
        # permutation (including merge-group and temporal ordering), exactly.
        repatched = image_processor(
            normalized, do_resize=False, do_rescale=False, do_normalize=False, device="cpu", return_tensors="pt"
        ).pixel_values
        roundtrip_error = float((pixels - repatched).abs().max())
        assert roundtrip_error == 0
        mean = torch.tensor(image_processor.image_mean).reshape(3, 1, 1)
        std = torch.tensor(image_processor.image_std).reshape(3, 1, 1)
        final = normalized * std + mean
        pre_normalized = image_processor(
            stage1, do_rescale=False, do_normalize=False, device="cpu", return_tensors="pt"
        )
        actual_rgb = unpatch(pre_normalized.pixel_values, grid, image_processor)[0]
        inverse_norm_error = float((final - actual_rgb).abs().max())
        assert inverse_norm_error < 2e-7
        normalized_roundtrip = image_processor(
            final, do_resize=False, do_rescale=False, device="cpu", return_tensors="pt"
        ).pixel_values
        rgb_roundtrip_error = float((pixels - normalized_roundtrip).abs().max())
        assert rgb_roundtrip_error < 5e-7
        height, width = (256, 320) if role == "head" else (128, 160)
        slot = source.resize((width, height), Image.Resampling.LANCZOS)
        saved_slot = Image.open(clip / f"resized_{role}.png").convert("RGB")
        slot_error = int(np.abs(np.asarray(slot).astype(int) - np.asarray(saved_slot).astype(int)).max())
        assert slot_error == 0
        slots[camera] = slot
        filenames = {"source": f"image_source_{role}.png", "stage1": f"mora_{role}_stage1.png", "final": f"mora_{role}_final.png"}
        source.save(dest / filenames["source"])
        tensor_image(stage1).save(dest / filenames["stage1"])
        tensor_image(final).save(dest / filenames["final"])
        camera_results[role] = {
            "camera_key": camera,
            "source_anchor": {"episode_frame": result["start"], "video_frame": frame_index, "timestamp_seconds": timestamp,
                              "video_relative_path": str(video.relative_to(Path(result["source"]))),
                              "rgb_sha256": hashlib.sha256(source_array.tobytes()).hexdigest()},
            "native_size_wh": list(source.size), "stage1_size_wh": list(stage1.shape[-2:][::-1]),
            "final_size_wh": list(final.shape[-2:][::-1]), "openwam_slot_size_wh": [width, height],
            "grid_thw": grid.tolist(), "patch_count": int(pixels.shape[0]), "patch_width": int(pixels.shape[1]),
            "visual_token_count": int(grid.prod() // image_processor.merge_size ** 2),
            "normalized_range": [float(pixels.min()), float(pixels.max())],
            "final_float_rgb_range": [float(final.min()), float(final.max())],
            "display_png_rule": "clip float RGB to [0,1], multiply by 255 and round; never model input",
            "display_clipped_values": int(((final < 0) | (final > 1)).sum()),
            "roundtrip_max_abs_error": roundtrip_error, "rgb_roundtrip_max_abs_error": rgb_roundtrip_error,
            "inverse_normalization_max_abs_error": inverse_norm_error, "temporal_duplicate_max_abs_error": temporal_error,
            "mora_timestamp_decoder_max_abs_error": decoder_error,
            "mora_timestamp_decoder_identical": decoder_error == 0,
            "openwam_slot_max_abs_error_uint8": slot_error, "files": filenames,
        }
    rebuilt = layout(slots, [result[role] or f"__missing_{role}__" for role in ["head", "left", "right"]], 384, 320)
    saved = Image.open(clip / "model_frame_0.png").convert("RGB")
    canvas_error = int(np.abs(np.array(rebuilt).astype(int) - np.array(saved).astype(int)).max())
    assert canvas_error == 0
    metrics = {
        "id": identifier, "status": "ok", "execution_scope": "isolated_transforms",
        "complete_mora_yaml_pipeline_executed": False, "model_weights_loaded": False,
        "source_anchor": {"episode": result["episode"], "episode_frame": result["start"], "clip_local_frame": 0},
        "anchor_selection": "controlled identical original segment start; not a Mora resolver sampling claim",
        "mora_sampler_selects_anchor": result["start"] % 2 == 0,
        "mora_sampler_anchor_rule": "30 Hz source, 15 Hz resolver, even episode frame ordinals",
        "same_source_rgb_used_for_both_pipelines": True,
        "mora_has_model_canvas": False, "report_preview_is_model_input": False,
        "missing_views_omitted_in_mora": missing, "cameras": camera_results,
        "openwam_canvas_size_wh": [320, 384], "openwam_canvas_max_abs_error_uint8": canvas_error,
        "comparison_file": "image_comparison.jpg",
    }
    comparison_preview(dest, clip, camera_results)
    write(dest / "image_metrics.json", metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", nargs="*")
    args = parser.parse_args()
    manifest = json.loads((REPORT / "manifest.json").read_text())
    recipe = OmegaConf.load(manifest["yaml"])
    processor = Qwen3VLProcessor.from_pretrained(recipe.framework.qwenvl.base_vlm, local_files_only=True)
    sources = [RESIZE_PATH, MORA_VIDEO_PATH, OPENWAM_VIDEO_PATH, LAYOUT_PATH, Path(inspect.getfile(type(processor.image_processor)))]
    summary = {
        "execution_scope": "isolated_transforms", "full_yaml_supported_or_executed": False,
        "recipe": str(Path(manifest["yaml"]).relative_to(MORA)), "recipe_sha256": manifest["yaml_sha256"],
        "environment": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                        "torch": torch.__version__, "torchvision": torchvision.__version__, "transformers": transformers.__version__,
                        "pyav": av.__version__, "device": "cpu", "cpu_threads": torch.get_num_threads(), "cuda_visible_devices": ""},
        "processor_class": type(processor).__name__, "image_processor_class": type(processor.image_processor).__name__,
        "processor_config": processor.to_dict(), "image_processor_config": processor.image_processor.to_dict(),
        "processor_call_overrides": {"do_rescale": False, "device": "cpu", "return_tensors": "pt"},
        "stage1": {"function": "src.data.modules.transforms.utils.resize_with_longest_edge", "longest_edge": 336, "mode": "bilinear"},
        "source_files": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in sources],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "notes": ["No model weights loaded; no model forward, training or GPU work.",
                  "Mora outputs are independent camera images; preview layout exists only in the report.",
                  "Original video is decoded at the first selected source frame; lossy JPEGs are never inputs.",
                  "Final PNGs are display-quantized reconstructions of actual normalized pixel_values.",
                  "Absent views are omitted in Mora and remain black in the verified OpenWAM canvas."],
        "results": [],
    }
    selected = [r for r in manifest["results"] if args.ids is None or r["id"] in args.ids]
    for result in selected:
        try:
            metrics = run_clip(result, processor.image_processor)
        except Exception as error:
            metrics = {"id": result["id"], "status": "failed", "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}
            write(OUT / result["id"] / "image_metrics.json", metrics)
        summary["results"].append(metrics)
        print(result["id"], metrics["status"], metrics.get("error", ""), flush=True)
        summary["counts"] = {"selected": len(selected), "completed": len(summary["results"]), "ok": sum(r["status"] == "ok" for r in summary["results"])}
        write(OUT / "image_results.json", summary)
    selected_ok = summary["counts"]["ok"]
    # A selective permission retry retains the other clips' completed evidence.
    # Also reconcile provenance fields if this script was refined during a run.
    combined = []
    for result in manifest["results"]:
        metrics_path = OUT / result["id"] / "image_metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text())
        if metrics["status"] == "ok":
            metrics.update(
                anchor_selection="controlled identical original segment start; not a Mora resolver sampling claim",
                mora_sampler_selects_anchor=result["start"] % 2 == 0,
                mora_sampler_anchor_rule="30 Hz source, 15 Hz resolver, even episode frame ordinals",
            )
            write(metrics_path, metrics)
        combined.append(metrics)
    summary["results"] = combined
    summary["counts"] = {"selected": len(manifest["results"]), "completed": len(combined),
                         "ok": sum(r["status"] == "ok" for r in combined)}
    all_cameras = [camera for result in combined if result["status"] == "ok" for camera in result["cameras"].values()]
    if all_cameras:
        summary["statistics"] = {
            "camera_count": len(all_cameras),
            "timestamp_decoder_identical_count": sum(camera["mora_timestamp_decoder_identical"] for camera in all_cameras),
            "max_patch_roundtrip_error": max(camera["roundtrip_max_abs_error"] for camera in all_cameras),
            "max_rgb_roundtrip_error": max(camera["rgb_roundtrip_max_abs_error"] for camera in all_cameras),
            "max_openwam_slot_error_uint8": max(camera["openwam_slot_max_abs_error_uint8"] for camera in all_cameras),
            "max_openwam_canvas_error_uint8": max(result["openwam_canvas_max_abs_error_uint8"] for result in combined if result["status"] == "ok"),
            "normalized_range": [min(camera["normalized_range"][0] for camera in all_cameras),
                                 max(camera["normalized_range"][1] for camera in all_cameras)],
            "final_sizes_wh": sorted({tuple(camera["final_size_wh"]) for camera in all_cameras}),
            "mora_sampler_omits_controlled_anchor_count": sum(not result["mora_sampler_selects_anchor"] for result in combined if result["status"] == "ok"),
        }
    summary["last_invocation_ids"] = args.ids
    write(OUT / "image_results.json", summary)
    if selected_ok != len(selected):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
