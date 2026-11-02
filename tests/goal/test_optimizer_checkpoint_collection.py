"""Collective checkpoint regression; CUDA variant is intended for DLC only."""
import copy
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype
        torch.testing.assert_close(left.cpu(), right.cpu(), rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_equal(a, b)
    else:
        assert left == right


def _exercise_collection(rank, backend):
    from openwam.train.goal_optimizer import ShardedFP32MasterAdamW

    device = torch.device('cuda', rank) if backend == 'nccl' else torch.device('cpu')

    def make_optimizer():
        params = [torch.nn.Parameter(torch.ones(shape, device=device, dtype=torch.bfloat16))
                  for shape in ((7,), (11,), (), (17,))]
        optimizer = ShardedFP32MasterAdamW([
            dict(params=params[:2], lr=.001, name='goal'),
            dict(params=params[2:], lr=.002, name='history')], betas=(.9, .95))
        return params, optimizer

    def update(params, optimizer, step):
        for index, parameter in enumerate(params):
            parameter.grad = None if index == 1 and step % 2 else torch.full_like(
                parameter, .123 + index * .17 + step * .03)
        optimizer.step()

    params, optimizer = make_optimizer()
    optimizer.optimizer.consolidate_state_dict(to=0)
    initial_expected = copy.deepcopy(optimizer.checkpoint_payload()) if rank == 0 else None
    optimizer.prepare_checkpoint()
    if rank == 0:
        _assert_equal(optimizer.checkpoint_payload(), initial_expected)
        assert all(set(state) == {'master'}
                   for state in optimizer.checkpoint_payload()['adam']['state'].values())
    for step in range(4):
        update(params, optimizer, step)

    # The native path is the compatibility oracle, including global indices.
    optimizer.optimizer.consolidate_state_dict(to=0)
    expected = copy.deepcopy(optimizer.checkpoint_payload()) if rank == 0 else None
    before = copy.deepcopy(optimizer.local_optimizer.state_dict())
    with patch.object(torch, 'ByteTensor', side_effect=AssertionError('slow Python byte conversion')):
        if rank == 0:
            optimizer.prepare_checkpoint()
        else:
            with patch.object(torch, 'load', side_effect=AssertionError('non-destination deserialization')):
                optimizer.prepare_checkpoint()
    _assert_equal(before, optimizer.local_optimizer.state_dict())
    for parameter, state in optimizer.local_optimizer.state.items():
        for key in ('master', 'exp_avg', 'exp_avg_sq'):
            assert state[key].device == parameter.device
            assert state[key].dtype == torch.float32

    if rank == 0:
        payload = optimizer.checkpoint_payload()
        _assert_equal(payload, expected)
        for state in payload['adam']['state'].values():
            assert all(value.device.type == 'cpu' for value in state.values() if torch.is_tensor(value))
            assert all(state[key].dtype == torch.float32 for key in ('master', 'exp_avg', 'exp_avg_sq'))
    else:
        assert optimizer.optimizer._all_state_dicts == []
        with pytest.raises(RuntimeError, match='consolidated'):
            optimizer.checkpoint_payload()
        payload = None
    objects = [payload]
    dist.broadcast_object_list(objects, src=0, device=device)
    restored_params, restored = make_optimizer()
    restored.restore_payload(objects[0])
    _assert_equal(optimizer.local_optimizer.state_dict(), restored.local_optimizer.state_dict())
    for step in range(4, 8):
        update(params, optimizer, step)
        update(restored_params, restored, step)
        _assert_equal(params, restored_params)
        _assert_equal(optimizer.local_optimizer.state_dict(), restored.local_optimizer.state_dict())
    optimizer.prepare_checkpoint()
    restored.prepare_checkpoint()
    if rank == 0:
        _assert_equal(optimizer.checkpoint_payload(), restored.checkpoint_payload())


def _worker(rank, backend, init, world):
    if backend == 'nccl':
        torch.cuda.set_device(rank)
    dist.init_process_group(backend, init_method='file://' + init, rank=rank,
                            world_size=world, timeout=timedelta(seconds=90))
    try:
        # Small chunks exercise transfer boundaries without a huge test model.
        with patch('openwam.train.optimizer_checkpoint._TRANSFER_BYTES', 1024):
            _exercise_collection(rank, backend)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('world', [2, 3])
def test_checkpoint_collection_avoids_python_byte_conversion_and_resumes_exactly(tmp_path, world):
    mp.spawn(_worker, args=('gloo', str(tmp_path / 'init'), world), nprocs=world, join=True)


@pytest.mark.parametrize('version', ['2.6.0', '2.8.0'])
def test_unverified_torch_uses_native_collection(version):
    from openwam.train.optimizer_checkpoint import consolidate_fp32_optimizer
    calls = []
    optimizer = SimpleNamespace(consolidate_state_dict=lambda **kwargs: calls.append(kwargs))
    with patch.object(torch, '__version__', version), pytest.warns(RuntimeWarning, match='native'):
        consolidate_fp32_optimizer(optimizer)
    assert calls == [dict(to=0)]


def test_missing_private_interface_uses_native_collection():
    from openwam.train.optimizer_checkpoint import consolidate_fp32_optimizer
    calls = []
    optimizer = SimpleNamespace(consolidate_state_dict=lambda **kwargs: calls.append(kwargs))
    with patch.object(torch, '__version__', '2.7.1'), pytest.warns(RuntimeWarning, match='native'):
        consolidate_fp32_optimizer(optimizer)
    assert calls == [dict(to=0)]


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='requires two DLC GPUs')
def test_nccl_checkpoint_collection_resumes_exactly(tmp_path):
    mp.spawn(_worker, args=('nccl', str(tmp_path / 'init'), 2), nprocs=2, join=True)
