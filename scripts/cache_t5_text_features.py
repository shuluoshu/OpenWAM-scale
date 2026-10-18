"""Precompute frozen T5 instruction features before training, without DiT/VAE."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from openwam.model.video_backbone.wan.encode import encode_text  # noqa: E402
from openwam.model.video_backbone.wan.text_feature_cache import FrozenTextFeatureCache  # noqa: E402


def load_prompts(path):
    """Accept JSON string arrays, JSONL {prompt: string}, or one string per line."""
    path = Path(path)
    if path.suffix == ".json":
        prompts = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(prompts, list):
            raise ValueError("JSON instructions must be an array of strings")
    elif path.suffix == ".jsonl":
        prompts = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                prompts.append(row.get("prompt") if isinstance(row, dict) else row)
    else:
        prompts = path.read_text(encoding="utf-8").splitlines()
    if any(not isinstance(prompt, str) for prompt in prompts):
        raise ValueError("Each instruction must be a string (JSONL objects require a prompt field)")
    prompts = list(dict.fromkeys(prompt for prompt in prompts if prompt.strip()))
    if not prompts:
        raise ValueError("Instruction list is empty")
    return prompts


def load_text_components(checkpoint, *, device, dtype):
    """Read a self-contained OpenWAM checkpoint, materializing only T5 tensors."""
    from omegaconf import OmegaConf
    from safetensors import safe_open

    from openwam.model.video_backbone.wan.loader import build_holder_from_components
    from openwam.model.video_backbone.wan.pipeline_builder import _build_tokenizer
    from openwam.train.utils.checkpointing import find_latest_weights
    from openwam.train.utils.ckpt_model_loader import _resolve_ckpt_source

    directory, explicit = _resolve_ckpt_source(str(Path(checkpoint).resolve()))
    cfg = OmegaConf.load(Path(directory) / "config.yaml")
    spec = OmegaConf.to_container(cfg.model.video_backbone, resolve=True)
    text_specs = [item for item in spec.get("components", []) if item.get("attr") == "text_encoder"]
    if len(text_specs) != 1 or not spec.get("tokenizer"):
        raise ValueError("Checkpoint must contain one Wan text_encoder component and a tokenizer spec")
    holder = build_holder_from_components(text_specs, device="meta")
    prefix = "video_backbone.text_encoder."
    with safe_open(explicit or find_latest_weights(directory), framework="pt", device="cpu") as weights:
        state = {key[len(prefix) :]: weights.get_tensor(key) for key in weights.keys() if key.startswith(prefix)}
    if not state:
        raise ValueError("Checkpoint has no video_backbone.text_encoder weights")
    encoder = holder.text_encoder
    encoder.load_state_dict(state, strict=True, assign=True)
    encoder.to(device=device, dtype=dtype).requires_grad_(False).eval()
    tokenizer = _build_tokenizer(spec["tokenizer"], directory)
    return encoder, tokenizer


@torch.no_grad()
def precompute(prompts, *, tokenizer, text_encoder, device, cache_dir, max_memory_mb=512):
    cache = FrozenTextFeatureCache(cache_dir, max_memory_bytes=int(max_memory_mb * (1 << 20)))
    prompts = list(dict.fromkeys(prompts))
    started = time.perf_counter()
    for index, prompt in enumerate(prompts, 1):
        # Same per-instruction encoding and key namespace as the training path.
        encode_text([prompt], tokenizer=tokenizer, text_encoder=text_encoder, device=device, feature_cache=cache)
        if index % 100 == 0 or index == len(prompts):
            logging.info("Cached %d/%d instructions; %s", index, len(prompts), cache.stats)
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    return dict(
        unique_prompts=len(prompts),
        seconds=time.perf_counter() - started,
        cache_directory=str(cache.directory),
        cache_stats=cache.stats.copy(),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", required=True, type=Path, help="Self-contained checkpoint directory or .safetensors file"
    )
    parser.add_argument(
        "--prompts-file", required=True, type=Path, help="JSON array, JSONL prompt records, or TXT instructions"
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/cache/t5"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--max-memory-mb", type=float, default=512)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--report", type=Path, help="Optional JSON completion report")
    args = parser.parse_args(argv)
    if args.max_memory_mb < 0 or args.threads < 1:
        parser.error("max-memory-mb must be nonnegative and threads must be positive")
    prompts = load_prompts(args.prompts_file)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.set_num_threads(args.threads)
    encoder, tokenizer = load_text_components(args.checkpoint, device=args.device, dtype=getattr(torch, args.dtype))
    report = precompute(
        prompts,
        tokenizer=tokenizer,
        text_encoder=encoder,
        device=args.device,
        cache_dir=args.cache_dir,
        max_memory_mb=args.max_memory_mb,
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
