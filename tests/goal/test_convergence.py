"""Deterministic train-population gate, independent of optimizer/model scale."""
import copy
import random
import numpy as np
import pytest
import torch


def policy(**extra):
    return dict(min_updates=50000, eval_every=10000, patience=5, relative_tolerance=.01, **extra)


def metric(value=1., **extra):
    return dict(loss=value, image_loss=value, tcp_loss=value, valid_images=1,
                valid_points=2, lambda_image=.5, lambda_tcp=.5, **extra)


def test_plateau_accepts_small_oscillation_at_minimum_and_replays():
    from openwam.train.convergence import new_state, observe, validate_state
    state = new_state(policy(), 'fixed')
    for step, loss in zip(range(0, 60000, 10000), [1., 1.002, .997, 1.001, .998, 1.]):
        state = observe(state, step, metric(loss), max_updates=154000)
        assert state['converged'] == (step == 50000)
    assert state['best_loss'] == .997 and state['reference_loss'] == 1.
    assert state['streak'] == 5
    assert validate_state(state, policy(), 'fixed', 50000, 154000) == state
    forged = copy.deepcopy(state); forged['streak'] = 100
    with pytest.raises(ValueError, match='state'):
        validate_state(forged, policy(), 'fixed', 50000, 154000)


def test_significant_improvement_and_material_regression_reset_reference():
    from openwam.train.convergence import new_state, observe
    state = new_state(policy(), 'fixed')
    for step, loss in zip(range(0, 60000, 10000), [1., .996, .992, .989, .988, 1.02]):
        state = observe(state, step, metric(loss), max_updates=154000)
    assert state['streak'] == 0 and not state['converged']
    assert state['reference_loss'] is None
    for step in range(60000, 120000, 10000):
        state = observe(state, step, metric(1.02), max_updates=154000)
    assert not state['converged']


@pytest.mark.parametrize('bad', [None, metric(float('nan')), metric(float('inf')),
                                 dict(metric(), valid_images=0), dict(metric(), valid_points=0)])
def test_invalid_evaluation_breaks_consecutive_evidence(bad):
    from openwam.train.convergence import new_state, observe
    state = new_state(policy(), 'fixed')
    for step in range(0, 50000, 10000):
        state = observe(state, step, metric(), max_updates=154000)
    state = observe(state, 50000, bad, max_updates=154000)
    assert state['streak'] == 0 and not state['converged']
    assert state['history'][-1]['metric'] is None
    state = observe(state, 60000, metric(), max_updates=154000)
    assert state['streak'] == 0


def test_final_non_cadence_cannot_count_or_converge():
    from openwam.train.convergence import new_state, observe
    p = dict(policy(), min_updates=54000)
    state = new_state(p, 'fixed')
    for step in range(0, 60000, 10000):
        state = observe(state, step, metric(), max_updates=54000)
    state = observe(state, 54000, metric(), max_updates=54000)
    assert state['streak'] == 5 and not state['converged']


@pytest.mark.parametrize('change', [dict(min_updates=True), dict(eval_every=0), dict(patience=False),
    dict(relative_tolerance=float('nan')), dict(relative_tolerance=1.), dict(extra=1)])
def test_invalid_policy_rejected(change):
    from openwam.train.convergence import convergence_config
    with pytest.raises(ValueError): convergence_config(dict(policy(), **change), 154000)


def test_fixed_flow_metric_is_reproducible_masked_and_rng_neutral():
    from openwam.train.convergence import fixed_flow_metric
    from test_query_goal import make_query
    from test_multiview_patch import multiview_batch
    torch.set_num_threads(1)
    expert = make_query().train_stage_a()
    obs, ctx, targets, views = multiview_batch(1)
    batch = dict(observed=obs, context=ctx, targets=targets, observed_view_valid=views)
    before = torch.get_rng_state().clone(); py = random.getstate(); numpy = np.random.get_state()
    first = fixed_flow_metric(expert, [batch], seed=11)
    assert torch.equal(before, torch.get_rng_state()) and py == random.getstate()
    assert np.array_equal(numpy[1], np.random.get_state()[1]) and expert.training
    torch.randn(17)
    second = fixed_flow_metric(expert, [batch], seed=11)
    assert first == second
    assert first['valid_images'] == 1 and first['valid_points'] == 2
    targets.image_valid[:] = False
    assert fixed_flow_metric(expert, [batch], seed=11)['valid_images'] == 0


def test_input_fingerprint_changes_with_mask_or_history():
    from openwam.train.convergence import evaluation_identity
    from test_multiview_patch import multiview_batch
    obs, ctx, targets, views = multiview_batch(1)
    pairs = [(dict(id='fixed'), dict(observed=obs, context=ctx, targets=targets,
              observed_view_valid=views, history_ages=torch.zeros(1, 2)))]
    initial = evaluation_identity(pairs, seed=42, expert_contract={'weights': [.5, 1.]})
    pairs[0][1]['history_ages'][0, 0] = 1
    assert evaluation_identity(pairs, seed=42, expert_contract={'weights': [.5, 1.]}) != initial


def test_exact_one_percent_improvement_resets_plateau():
    from openwam.train.convergence import new_state, observe
    state = new_state(policy(), 'fixed')
    for step in range(0, 50000, 10000):
        state = observe(state, step, metric(1.5), max_updates=154000)
    state = observe(state, 50000, metric(1.485), max_updates=154000)
    assert state['streak'] == 0 and not state['converged']


def test_metric_counts_effective_masks_and_aggregates_by_denominator():
    from dataclasses import replace
    from openwam.train.convergence import fixed_flow_metric
    from test_query_goal import make_query
    from test_multiview_patch import multiview_batch
    torch.set_num_threads(1)
    expert = make_query()
    obs, ctx, targets, views = multiview_batch(1)
    left = dict(observed=obs, context=ctx, targets=targets, observed_view_valid=views)
    uv = targets.uv.clone(); uv[0, 1] = float('nan')
    right = dict(left, targets=replace(targets, uv=uv))
    first = fixed_flow_metric(expert, [left], seed=12)
    second = fixed_flow_metric(expert, [right], seed=13)
    both = fixed_flow_metric(expert, [left, right], seed=12)
    assert both['valid_images'] == 2 and both['valid_points'] == 3
    assert both['image_loss'] == pytest.approx((first['image_loss'] + second['image_loss']) / 2)
    assert both['tcp_loss'] == pytest.approx((first['tcp_loss'] * 2 + second['tcp_loss']) / 3)


def test_diagnostic_preserves_rng_and_modes_on_lazy_batch_failure():
    from openwam.train.convergence import fixed_flow_metric
    from test_query_goal import make_query
    expert = make_query().train_stage_a()
    before = torch.get_rng_state().clone(); py = random.getstate(); numpy = np.random.get_state()
    def broken():
        random.random(); np.random.rand(); torch.randn(1)
        raise RuntimeError('broken batch')
        yield
    with pytest.raises(RuntimeError, match='broken batch'):
        fixed_flow_metric(expert, broken(), seed=0)
    assert torch.equal(before, torch.get_rng_state()) and py == random.getstate()
    assert np.array_equal(numpy[1], np.random.get_state()[1]) and expert.training


def test_image_improvement_cannot_hide_tcp_regression():
    from openwam.train.convergence import new_state, observe
    state = new_state(policy(), 'fixed')
    state = observe(state, 0, metric(), max_updates=154000)
    for step in range(10000, 70000, 10000):
        value = dict(metric(), image_loss=.97, tcp_loss=1.03)
        state = observe(state, step, value, max_updates=154000)
    assert state['streak'] == 0 and not state['converged']
    assert state['best_image_loss'] == .97 and state['best_tcp_loss'] == 1.


def test_validation_rejects_missing_scheduled_evidence():
    from openwam.train.convergence import new_state, observe, validate_state
    state = observe(new_state(policy(), 'fixed'), 0, metric(), max_updates=154000)
    with pytest.raises(ValueError, match='state'):
        validate_state(state, policy(), 'fixed', 20000, 154000)


def test_rng_preservation_does_not_initialize_other_visible_gpus(monkeypatch):
    from contextlib import contextmanager
    from openwam.train.convergence import preserve_rng
    recorded = []
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 3)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 8)
    @contextmanager
    def fork_rng(*, devices):
        recorded.append(devices)
        yield
    monkeypatch.setattr(torch.random, 'fork_rng', fork_rng)
    with preserve_rng(): pass
    assert recorded == [[3]]
