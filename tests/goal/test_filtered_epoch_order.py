import pytest
from openwam.train.epoch_order import EpochOrder, sampling_contract


def config(total=41, ranges=None):
    return dict(seed=42, sample_order='filtered_global_shuffle',
                parent_total_samples=total,
                excluded_sample_ranges=ranges if ranges is not None else [[0, 2], [10, 17], [40, 41]])


def test_retains_parent_order_and_remaps_every_resume_boundary():
    import openwam.train.epoch_order as module
    parent = EpochOrder(sampling_contract({'seed': 42}, 41))
    removed = set(range(2)) | set(range(10, 17)) | {40}
    survivors = [i for i in range(41) if i not in removed]
    compact = {old: new for new, old in enumerate(survivors)}
    parent_ids = [parent[i] for i in range(41)]
    contract = sampling_contract(config(), len(survivors))
    actual = EpochOrder(contract)
    assert [actual[i] for i in range(actual.total)] == [compact[i] for i in parent_ids if i not in removed]
    assert sorted(actual[i] for i in range(actual.total)) == list(range(actual.total))
    for cursor in range(42):
        mapped = module.retained_parent_cursor(contract, cursor)
        assert [actual[i] for i in range(mapped, actual.total)] == [compact[i] for i in parent_ids[cursor:] if i not in removed]
    assert EpochOrder(contract).sha256 == actual.sha256


@pytest.mark.parametrize('ranges', [[[2, 6], [5, 7]], [[5, 7], [2, 4]], [[-1, 2]], [[2, 42]], [[2, 2]], [[True, 3]]])
def test_rejects_ambiguous_or_invalid_exclusions(ranges):
    with pytest.raises(ValueError):
        sampling_contract(config(ranges=ranges), 31)


def test_rejects_wrong_population_runtime_and_cursor():
    import openwam.train.epoch_order as module
    with pytest.raises(ValueError):
        sampling_contract(config(), 32)
    contract = sampling_contract(config(), 31)
    with pytest.raises(ValueError):
        EpochOrder({**contract, 'numpy_version': 'changed'})
    for cursor in [-1, 42, True]:
        with pytest.raises(ValueError):
            module.retained_parent_cursor(contract, cursor)
