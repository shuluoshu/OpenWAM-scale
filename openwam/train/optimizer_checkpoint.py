"""Bounded, buffer-based collection for the pinned Torch 2.7 ZeRO optimizer.

Only collection is replaced: Torch still maps local parameter indices to the
global checkpoint and restores the shards. This intentionally depends on its
``_all_state_dicts`` contract; unverified Torch versions use native collection.
No Torch functions are patched, and update state never leaves its GPU.
"""
from __future__ import annotations

import io
import warnings

import torch
import torch.distributed as dist


_TRANSFER_BYTES = 64 * 1024 * 1024


def _cpu_snapshot(value):
    if torch.is_tensor(value):
        # Blocking copies ensure serialization cannot read an unfinished D2H.
        return value.detach().to(device='cpu', non_blocking=False)
    if isinstance(value, dict):
        return {key: _cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_snapshot(item) for item in value)
    return value


def _collect_shard(state, *, source, destination, group, device):
    """All ranks broadcast chunks; only the destination decodes the payload."""
    rank = dist.get_rank()
    stream = io.BytesIO()
    if rank == source:
        torch.save(state, stream)
        # A view over the writable buffer avoids per-byte Python conversion.
        data = torch.frombuffer(stream.getbuffer(), dtype=torch.uint8)
        size = data.numel()
    else:
        data, size = None, 0
    length = torch.tensor([size], dtype=torch.int64, device=device)
    dist.broadcast(length, src=source, group=group)
    size = int(length.item())
    received = torch.empty(size, dtype=torch.uint8) if rank == destination else None
    staging = (torch.empty(min(size, _TRANSFER_BYTES), dtype=torch.uint8, device=device)
               if rank != source else None)
    for offset in range(0, size, _TRANSFER_BYTES):
        count = min(_TRANSFER_BYTES, size - offset)
        chunk = data[offset:offset + count].to(device) if rank == source else staging[:count]
        dist.broadcast(chunk, src=source, group=group)
        if received is not None:
            received[offset:offset + count].copy_(chunk, non_blocking=False)
        del chunk
    if received is not None:
        # Only trusted, locally generated optimizer state crosses this group.
        return torch.load(io.BytesIO(received.numpy()), map_location='cpu', weights_only=False)
    return None


def consolidate_fp32_optimizer(optimizer):
    """Collect to group rank zero while preserving Torch's checkpoint format.

    Verified against Torch 2.7.1's native consolidation in distributed tests.
    Its private list must contain CPU local state dicts in process-group rank
    order, with an empty list on other ranks. Keep this guard when upgrading
    Torch, and extend it only after the parity/resume tests pass there.
    """
    required = ('_all_state_dicts', '_check_overlap_initialized', '_sync_param_groups',
                '_default_device', 'optim', 'process_group', 'rank', 'world_size')
    if (torch.__version__.split('.')[:2] != ['2', '7'] or
            not all(hasattr(optimizer, name) for name in required)):
        warnings.warn('Fast ZeRO checkpoint collection requires the verified Torch 2.7 '
                      'interface; using native consolidation.', RuntimeWarning, stacklevel=2)
        optimizer.consolidate_state_dict(to=0)
        return
    optimizer._check_overlap_initialized()
    optimizer._sync_param_groups(optimizer.param_groups, optimizer.optim.param_groups)
    optimizer._all_state_dicts = []
    group = optimizer.process_group
    destination = dist.get_global_rank(group, 0)
    device = optimizer._default_device if dist.get_backend(group) == 'nccl' else torch.device('cpu')
    for rank in range(optimizer.world_size):
        state = _cpu_snapshot(optimizer.optim.state_dict()) if optimizer.rank == rank else None
        if rank != 0:
            state = _collect_shard(state, source=dist.get_global_rank(group, rank),
                                   destination=destination, group=group, device=device)
        if optimizer.rank == 0:
            optimizer._all_state_dicts.append(state)
