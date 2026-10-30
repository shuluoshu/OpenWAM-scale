from types import SimpleNamespace
import pytest
from scripts.compare_z_action_stage2 import training_budget_remaining, evaluation_indices


def test_update_budget_stops_exactly_and_retains_wall_clock_safety():
    args=SimpleNamespace(max_updates=500,smoke=False,smoke_updates=2,seconds_per_arm=3600)
    assert training_budget_remaining(499,10,args)
    assert not training_budget_remaining(500,10,args)
    assert not training_budget_remaining(12,3600,args)
    args.max_updates=None
    assert training_budget_remaining(500,10,args)


def test_evaluation_subset_keeps_original_sample_ids_and_validates_bounds():
    samples=[{'id':i} for i in range(5)]
    assert evaluation_indices(samples,None)==[0,1,2,3,4]
    assert evaluation_indices(samples,[3,1])==[3,1]
    for bad in ([5],[-1],[1,1],[]):
        with pytest.raises(ValueError):evaluation_indices(samples,bad)
