"""Behavioral coverage and resume tests for the finite global sample order."""
import numpy as np
import pytest


@pytest.mark.parametrize('total,world', [(1, 1), (3, 8), (33, 32), (257, 32), (1031, 8)])
def test_global_shuffle_covers_every_sample_once_with_distributed_tail(total, world):
    from openwam.train.epoch_order import EpochOrder, sampling_contract
    from openwam.train.goal_epoch import rank_sample
    order = EpochOrder(sampling_contract({'seed': 42}, total))
    seen = []
    for cursor in range(0, total, world):
        batch = [rank_sample(cursor, total, rank, world) for rank in range(world)]
        assert sum(weight for _, weight in batch) == pytest.approx(world)
        seen.extend(order[position] for position, weight in batch if weight > 0)
    assert sorted(seen) == list(range(total))
    if total > 3:
        assert seen != list(range(total))


def test_resume_reconstructs_identical_remaining_ids_without_rng_state():
    from openwam.train.epoch_order import EpochOrder, sampling_contract
    contract = sampling_contract({'seed': 42}, 1031)
    uninterrupted = EpochOrder(contract)
    np.random.seed(123)
    np.random.random(200)
    resumed = EpochOrder(contract)
    assert [resumed[i] for i in range(320, 1031)] == [uninterrupted[i] for i in range(320, 1031)]
    assert resumed.sha256 == uninterrupted.sha256
    different = EpochOrder(sampling_contract({'seed': 43}, 1031))
    assert different.sha256 != uninterrupted.sha256


def test_shuffle_mixes_contiguous_sources_and_task_levels_in_first_batch():
    from openwam.train.epoch_order import EpochOrder, sampling_contract
    order = EpochOrder(sampling_contract({'seed': 42}, 32000))
    # Eight consecutive sources, each with a task half then a subtask half.
    batch = [order[i] for i in range(32)]
    assert len({i // 4000 for i in batch}) >= 6
    assert {int(i % 4000 >= 2000) for i in batch} == {0, 1}


def test_contract_rejects_changed_runtime_and_invalid_configuration():
    from openwam.train.epoch_order import EpochOrder, sampling_contract
    contract = sampling_contract({'seed': 42}, 100)
    with pytest.raises(ValueError, match='contract'):
        EpochOrder({**contract, 'numpy_version': 'different'})
    for cfg in ({'sample_order': 'typo'}, {'seed': -1}, {'seed': True}):
        with pytest.raises(ValueError):
            sampling_contract(cfg, 100)
    with pytest.raises(ValueError):
        sampling_contract({}, 0)
    order = EpochOrder(contract)
    for position in (-1, 100, True):
        with pytest.raises((ValueError, IndexError)):
            order[position]
    assert order.nbytes == 100 * 4


def test_sequential_checkpoints_cannot_silently_resume_as_shuffled():
    from openwam.train.epoch_order import EpochOrder, sampling_contract
    shuffled = sampling_contract({'seed': 42}, 100)
    sequential = sampling_contract({'seed': 42, 'sample_order': 'sequential'}, 100)
    assert shuffled != sequential
    assert [EpochOrder(sequential)[i] for i in range(100)] == list(range(100))
