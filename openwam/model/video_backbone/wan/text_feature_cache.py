"""Persistent, bounded CPU cache of *frozen* T5 outputs, before trainable projections.

Keys use actual encoder weights and tokenized inputs, not checkpoint path names.
Weight hashing is lazy (after resume/device placement), once per weight version.
Each missing instruction is encoded individually in eval mode, matching the
serial smoke preprocessing path. Files are atomically published across ranks.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import os
import tempfile
from collections import OrderedDict
from pathlib import Path

import torch

logger = logging.getLogger(__name__)
_SCHEMA = 2


def _tensor_digest(digest, tensor):
    tensor = tensor.detach().contiguous().view(-1)
    digest.update(str((str(tensor.dtype), tuple(tensor.shape))).encode())
    # Bound temporary host memory even for the XXL vocabulary embedding.
    chunk_elements = max(1, (8 << 20) // tensor.element_size())
    for chunk in tensor.split(chunk_elements):
        raw = chunk.cpu().view(torch.uint8).numpy()
        digest.update(memoryview(raw))


class FrozenTextFeatureCache:
    def __init__(self, directory, *, max_memory_bytes=512 << 20):
        if max_memory_bytes < 0:
            raise ValueError("text feature cache memory limit must be nonnegative")
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_memory_bytes = int(max_memory_bytes)
        self._memory = OrderedDict()
        self._signature = None
        self._namespace = None
        self.stats = dict(memory_hits=0, disk_hits=0, misses=0, memory_bytes=0)

    def _prepare_encoder(self, encoder, device):
        parameters = list(encoder.parameters())
        if any(p.requires_grad for p in parameters):
            raise ValueError("T5 feature caching requires a frozen text encoder")
        encoder.eval()
        tensors = list(encoder.named_parameters()) + list(encoder.named_buffers())
        autocast = torch.is_autocast_enabled(torch.device(device).type)
        runtime = (
            torch.__version__,
            torch.version.cuda,
            str(torch.get_autocast_dtype(torch.device(device).type)) if autocast else None,
            torch.get_float32_matmul_precision(),
        )
        signature = (runtime, tuple((name, id(t), t._version, str(t.dtype), str(t.device)) for name, t in tensors))
        if signature == self._signature:
            return
        if any(t.is_meta for _, t in tensors):
            raise ValueError("Load text encoder weights before using the feature cache")
        digest = hashlib.sha256()
        digest.update(json.dumps((_SCHEMA, runtime, type(encoder).__module__, type(encoder).__qualname__)).encode())
        source = inspect.getsourcefile(type(encoder))
        if source and Path(source).is_file():
            digest.update(Path(source).read_bytes())
        logger.info("Fingerprinting frozen text encoder weights for persistent feature cache (once per weight version)")
        for name, tensor in sorted(encoder.state_dict().items()):
            digest.update(name.encode())
            digest.update(str(tuple(tensor.shape)).encode())
            _tensor_digest(digest, tensor)
        self._namespace = self.directory / digest.hexdigest()
        self._namespace.mkdir(parents=True, exist_ok=True)
        self._memory.clear()
        self.stats["memory_bytes"] = 0
        self._signature = signature
        logger.info("T5 feature cache namespace: %s", self._namespace)

    def _remember(self, key, context):
        size = context.numel() * context.element_size()
        if size > self.max_memory_bytes:
            return
        while self._memory and self.stats["memory_bytes"] + size > self.max_memory_bytes:
            _, old = self._memory.popitem(last=False)
            self.stats["memory_bytes"] -= old.numel() * old.element_size()
        self._memory[key] = context
        self.stats["memory_bytes"] += size

    def _load(self, path, key, length, tokens):
        try:
            item = torch.load(path, map_location="cpu", weights_only=True)
            compact = item["context"]
            if (
                item["key"] != key
                or item["seq_len"] != length
                or item["num_tokens"] != tokens
                or not isinstance(compact, torch.Tensor)
                or compact.ndim != 3
                or compact.shape[:2] != (1, length)
                or not compact.is_floating_point()
                or not torch.isfinite(compact).all()
            ):
                raise ValueError("invalid cached T5 feature metadata or tensor")
            context = compact.new_zeros(1, tokens, compact.shape[-1])
            context[:, :length] = compact
            return context
        except FileNotFoundError:
            return None
        except Exception as exc:
            logger.warning("Rebuilding unreadable T5 cache entry %s: %s", path, exc)
            return None

    @staticmethod
    def _save(path, key, context, length):
        # Same-directory rename is atomic on the shared training filesystem.
        fd, temporary = tempfile.mkstemp(prefix=".t5-", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                # clone avoids serializing the full padded backing storage.
                torch.save(
                    dict(key=key, context=context[:, :length].clone(), seq_len=length, num_tokens=context.shape[1]),
                    stream,
                )
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @torch.no_grad()
    def encode(self, ids, mask, *, text_encoder, device):
        self._prepare_encoder(text_encoder, device)
        ids, mask = ids.detach().cpu(), mask.detach().cpu()
        lengths = mask.gt(0).sum(dim=1).long()
        contexts = []
        for row, length in enumerate(lengths.tolist()):
            digest = hashlib.sha256()
            # Includes padding, masks and cleaning/tokenizer results. Changing
            # tokenizer settings invalidates exactly the affected instructions.
            _tensor_digest(digest, ids[row])
            _tensor_digest(digest, mask[row])
            key = digest.hexdigest()
            context = self._memory.get(key)
            if context is not None:
                self.stats["memory_hits"] += 1
                self._memory.move_to_end(key)
            else:
                path = self._namespace / (key + ".pt")
                context = self._load(path, key, length, ids.shape[1])
                if context is not None:
                    self.stats["disk_hits"] += 1
                else:
                    self.stats["misses"] += 1
                    context = text_encoder(ids[row : row + 1].to(device), mask[row : row + 1].to(device))
                    context[:, length:] = 0
                    context = context.detach().cpu().contiguous()
                    self._save(path, key, context, length)
                self._remember(key, context)
            contexts.append(context.to(device))
        return torch.cat(contexts, dim=0), lengths.to(device)


def configure_text_feature_cache(architecture, config):
    """Configure after freezing; the encoder fingerprint waits until first use."""
    backbone = architecture.video_backbone
    if not config or not config.get("enabled", False):
        backbone._text_feature_cache = None
        return
    if not hasattr(backbone, "_tokenizer") or not hasattr(backbone, "text_encoder"):
        raise ValueError("T5 feature cache requires a Wan text encoder/tokenizer")
    if any(p.requires_grad for p in backbone.text_encoder.parameters()):
        raise ValueError("T5 feature caching requires a frozen text encoder")
    backbone._text_feature_cache = FrozenTextFeatureCache(
        config.get("directory", "outputs/cache/t5"),
        max_memory_bytes=int(float(config.get("max_memory_mb", 512)) * (1 << 20)),
    )
    backbone.text_encoder.eval()
    logger.info(
        "Frozen T5 feature caching enabled: %s (T5 eval; trainable projections remain live)",
        backbone._text_feature_cache.directory,
    )
