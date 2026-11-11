"""Portable RFC-only assets and exact single-process training continuation.

Frozen VAE/text dependencies are external. Runtime caches never enter weights.
The resume payload is a trusted local torch file, distinct from model bundles.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

FORMAT = 'openwam.goal_memory.v1'
STATE_FORMAT = 'openwam.goal_memory.training.v1'


def _json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _atomic(path, write):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    os.close(fd)
    try:
        write(temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def save_memory_bundle(expert, path, *, dit_config, dependencies, training=None):
    """Save shared VideoDiT and new G/horizon/TCP weights once, without VAE/T5."""
    metadata = dict(format=FORMAT, dit_config=dit_config,
                    goal_memory=asdict(expert.config), dependencies=dependencies,
                    training=training or {}, history=dict(fps=1, temporal_vae=False),
                    supervision='instruction-granularity end_exclusive-1')
    serialized = _json(metadata)
    weights = {k: v.detach().cpu().contiguous() for k, v in expert.state_dict().items()}
    _atomic(path, lambda filename: save_file(weights, filename,
                                           metadata={'goal_memory_bundle': serialized}))
    return metadata


def memory_bundle_metadata(path):
    with safe_open(str(path), framework='pt', device='cpu') as file:
        raw = (file.metadata() or {}).get('goal_memory_bundle')
        if raw is None:
            raise ValueError('missing goal_memory bundle metadata; old C* is not compatible')
        metadata = json.loads(raw)
    if metadata.get('format') != FORMAT:
        raise ValueError('unsupported goal_memory bundle format')
    return metadata


def load_memory_bundle(path, *, dependencies, device='cpu', dtype=torch.float32):
    from openwam.model.goal.goal_memory import GoalMemoryConfig, GoalMemoryExpert
    from openwam.model.video_backbone.wan.models.dit import WanModel
    metadata = memory_bundle_metadata(path)
    if _json(metadata['dependencies']) != _json(dependencies):
        raise ValueError('goal_memory VAE/text/preprocessing dependencies differ')
    expert = GoalMemoryExpert(WanModel(**metadata['dit_config']),
                              GoalMemoryConfig(**metadata['goal_memory']), dit_config=metadata['dit_config'])
    with safe_open(str(path), framework='pt', device='cpu') as file:
        expert.load_state_dict({k: file.get_tensor(k) for k in file.keys()}, strict=True)
    expert.to(device=device, dtype=dtype)
    expert.bundle_metadata = metadata
    return expert


def _snapshot(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _snapshot(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_snapshot(v) for v in value)
    return value


def _execution_contract(model):
    """Bind exact continuation to effective tensors, autocast and RNG streams."""
    cuda_initialized = torch.cuda.is_initialized()
    return dict(
        tensors={name: dict(dtype=str(value.dtype), device=str(value.device))
                 for name, value in model.state_dict().items()},
        autocast={backend: dict(enabled=torch.is_autocast_enabled(backend),
                               dtype=str(torch.get_autocast_dtype(backend)))
                  for backend in ('cpu', 'cuda')},
        rng_devices=['cpu'] + ([f'cuda:{index}' for index in range(torch.cuda.device_count())]
                               if cuda_initialized else []),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES') if cuda_initialized else None)


def save_training_state(path, model, optimizer, *, update, contract):
    """Commit model, FP32 masters/moments, data cursor and RNG as one transaction."""
    masters = getattr(optimizer, 'parameter_pairs', None)
    inner = optimizer.optimizer if masters is not None else optimizer
    payload = dict(format=STATE_FORMAT, update=int(update), contract=_json(contract),
                   execution=_execution_contract(model),
                   model=_snapshot(model.state_dict()), optimizer=_snapshot(inner.state_dict()),
                   masters=None if masters is None else [_snapshot(m) for _, m in masters],
                   torch_rng=torch.get_rng_state(), python_rng=random.getstate(),
                   numpy_rng=np.random.get_state(),
                   cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)
    _atomic(path, lambda filename: torch.save(payload, filename))


def restore_training_state(path, model, optimizer, *, contract):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('format') != STATE_FORMAT or payload.get('contract') != _json(contract):
        raise ValueError('resume format/contract differs (data, mixture or schedule)')
    current = model.state_dict()
    if (current.keys() != payload['model'].keys() or
            any(current[k].shape != v.shape for k, v in payload['model'].items())):
        raise ValueError('resume model architecture differs')
    if payload.get('execution') != _execution_contract(model):
        raise ValueError('resume execution differs (model dtype/device, autocast or RNG devices)')
    masters = getattr(optimizer, 'parameter_pairs', None)
    if (masters is None) != (payload['masters'] is None):
        raise ValueError('resume optimizer master contract differs')
    inner = optimizer.optimizer if masters is not None else optimizer
    if (len(inner.param_groups) != len(payload['optimizer']['param_groups']) or
            any(len(group['params']) != len(saved['params']) for group, saved in
                zip(inner.param_groups, payload['optimizer']['param_groups']))):
        raise ValueError('resume optimizer parameter groups differ')
    if masters is not None and (len(masters) != len(payload['masters']) or
            any(m.shape != saved.shape or saved.dtype != torch.float32
                for (_, m), saved in zip(masters, payload['masters']))):
        raise ValueError('resume optimizer master model differs')
    if payload['cuda_rng'] is not None and (not torch.cuda.is_available() or
            len(payload['cuda_rng']) != torch.cuda.device_count()):
        raise ValueError('resume CUDA RNG device count differs')
    model.load_state_dict(payload['model'], strict=True)
    inner.load_state_dict(payload['optimizer'])
    if masters is not None:
        with torch.no_grad():
            for (parameter, master), saved in zip(masters, payload['masters']):
                master.copy_(saved)
                parameter.copy_(master)
    torch.set_rng_state(payload['torch_rng'])
    random.setstate(payload['python_rng'])
    np.random.set_state(payload['numpy_rng'])
    if payload['cuda_rng'] is not None:
        torch.cuda.set_rng_state_all(payload['cuda_rng'])
    return payload['update']
