"""Real tiny optimizer/checkpoint integration for the opt-in Stage1 gate."""
import json
from pathlib import Path
import pytest
import torch


def fixture_run(world=1, max_updates=6):
    from test_query_goal import make_query
    from test_multiview_patch import multiview_batch
    from openwam.model.goal.dependencies import dependency_identity
    from openwam.train.epoch_order import sampling_contract
    torch.set_num_threads(1); torch.manual_seed(19)
    expert = make_query(); obs, ctx, target, views = multiview_batch(1)
    class Data:
        def __len__(self): return 3
        def sample(self, index):
            return dict(id=str(index), family='tiny', instruction='move', granularity='task'), dict(
                observed=obs, context=ctx, targets=target, observed_view_valid=views)
        def evaluation_samples(self): return [self.sample(0)]
        def scope_summary(self): return {'total_samples': 3}
    policy = dict(min_updates=2, eval_every=1, patience=2, relative_tolerance=.01)
    identity = dict(index_sha256='a'*64, source_yaml_sha256='b'*64, total_samples=3,
                    world_size=world, seed=42, config_sha256='c'*64)
    identity['sampling'] = sampling_contract({'seed':42}, 3)
    identity['training'] = dict(mode='bounded_updates', max_updates=max_updates,
                                repeat_population=True, convergence=policy)
    deps = dependency_identity(expert.codec.vae, text_dim=expert.text_dim, precomputed_text_fingerprint='d'*64)
    kwargs = dict(identity=identity, dependencies=deps, max_updates=max_updates, repeat_population=True,
                  convergence=policy, checkpoint_every=3, eval_every=6, sampling_steps=1)
    return expert, Data(), kwargs


def constant_metric(*args, **kwargs):
    return dict(loss=1.5, image_loss=1., tcp_loss=1., valid_images=1, valid_points=2,
                lambda_image=.5, lambda_tcp=1.)


def test_converged_current_checkpoint_and_resume_without_update(tmp_path, monkeypatch):
    import openwam.train.convergence as gate
    from openwam.train.goal_epoch import train_epoch, verify_completion
    monkeypatch.setattr(gate, 'fixed_flow_metric', constant_metric)
    expert, data, kwargs = fixture_run()
    proof = train_epoch(expert, data, None, tmp_path, **kwargs)
    assert proof['updates'] == 2 and proof['completion_kind'] == 'convergence'
    assert proof['convergence']['streak'] == 2
    assert proof == verify_completion(tmp_path, kwargs['identity'])
    before = (tmp_path/'progress.jsonl').read_text()
    clone, _, _ = fixture_run()
    assert train_epoch(clone, data, None, tmp_path, resume=True, **kwargs) == proof
    assert (tmp_path/'progress.jsonl').read_text() == before
    # Crash after the accepted checkpoint but before completion publication.
    (tmp_path/'training_complete.json').unlink()
    resumed, _, _ = fixture_run()
    again = train_epoch(resumed, data, None, tmp_path, resume=True, **kwargs)
    assert again == proof
    for name, value in expert.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[name], atol=0, rtol=0)


def test_budget_exhausted_saves_state_and_pipeline_never_launches_stage2(tmp_path, monkeypatch):
    import openwam.train.convergence as gate
    from openwam.train.goal_epoch import train_epoch
    from openwam.train.stage12 import execute_stages
    count = 0
    def improving(*args, **kw):
        nonlocal count
        result = constant_metric(); result['image_loss'] = result['tcp_loss'] = .8**count
        result['loss'] = 1.5 * .8**count; count += 1
        return result
    monkeypatch.setattr(gate, 'fixed_flow_metric', improving)
    expert, data, kwargs = fixture_run(max_updates=3)
    output = tmp_path/'stage1'
    with pytest.raises(RuntimeError, match='without convergence'):
        train_epoch(expert, data, None, output, **kwargs)
    assert not (output/'training_complete.json').exists()
    latest = Path(json.loads((output/'latest.json').read_text())['directory'])
    assert json.loads((latest/'manifest.json').read_text())['update'] == 3
    clone, _, _ = fixture_run(max_updates=3)
    with pytest.raises(RuntimeError, match='without convergence'):
        train_epoch(clone, data, None, output, resume=True, **kwargs)
    assert count == 4
    calls = []
    with pytest.raises(FileNotFoundError):
        execute_stages(tmp_path, kwargs['identity'], ['stage1'], ['stage2'], runner=lambda c: calls.append(c))
    assert calls == [['stage1']]


def test_checkpoint_history_and_input_identity_are_required(tmp_path, monkeypatch):
    import openwam.train.convergence as gate
    import openwam.train.goal_epoch as epoch
    monkeypatch.setattr(gate, 'fixed_flow_metric', constant_metric)
    expert, data, kwargs = fixture_run()
    original = epoch.save_epoch_checkpoint
    def interrupt(*args, **kw):
        saved = original(*args, **kw)
        if kw['update'] == 1: raise RuntimeError('planned crash')
        return saved
    monkeypatch.setattr(epoch, 'save_epoch_checkpoint', interrupt)
    with pytest.raises(RuntimeError, match='planned crash'):
        epoch.train_epoch(expert, data, None, tmp_path, **kwargs)
    monkeypatch.setattr(epoch, 'save_epoch_checkpoint', original)
    clone, _, _ = fixture_run()
    data.sample(0)[1]['targets'].point_valid[0, 0] = False
    with pytest.raises(RuntimeError, match='convergence state'):
        epoch.train_epoch(clone, data, None, tmp_path, resume=True, **kwargs)


def _distributed_worker(rank, rendezvous, output, mode):
    import torch.distributed as dist
    import openwam.train.convergence as gate
    from openwam.train.goal_epoch import train_epoch
    dist.init_process_group('gloo', init_method='file://'+rendezvous, rank=rank, world_size=2)
    try:
        expert, data, kwargs = fixture_run(world=2, max_updates=3)
        calls = 0
        def diagnostic(*args, **kw):
            nonlocal calls
            calls += 1
            if mode == 'error' and calls == 2:
                raise RuntimeError('planned diagnostic exception')
            result = constant_metric()
            if mode == 'nonfinite': result['loss'] = float('nan')
            return result
        gate.fixed_flow_metric = diagnostic
        error = None
        try:
            proof = train_epoch(expert, data, None, output, **kwargs)
            assert mode == 'converged' and proof['updates'] == 2
        except RuntimeError as exc:
            error = str(exc)
            assert mode != 'converged'
        torch.save(dict(model=expert.state_dict(), error=error), Path(output)/f'result{rank}.pt')
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('mode', ['converged', 'error', 'nonfinite'])
def test_two_ranks_agree_on_gate_stop_and_errors(tmp_path, mode):
    import torch.multiprocessing as mp
    mp.spawn(_distributed_worker, args=(str(tmp_path/'rendezvous'),str(tmp_path/'run'),mode), nprocs=2, join=True)
    left = torch.load(tmp_path/'run/result0.pt', weights_only=True)
    right = torch.load(tmp_path/'run/result1.pt', weights_only=True)
    assert left['error'] == right['error']
    for key, value in left['model'].items():
        torch.testing.assert_close(value, right['model'][key], atol=0, rtol=0)
    assert (tmp_path/'run/training_complete.json').exists() == (mode == 'converged')


def test_resume_replays_checkpoint_history_and_exact_optimizer(tmp_path, monkeypatch):
    import openwam.train.convergence as gate
    import openwam.train.goal_epoch as epoch
    monkeypatch.setattr(gate, 'fixed_flow_metric', constant_metric)
    first, data, kwargs = fixture_run()
    full = epoch.train_epoch(first, data, None, tmp_path/'full', **kwargs)
    original = epoch.save_epoch_checkpoint
    def interrupt(*args, **kw):
        saved = original(*args, **kw)
        if kw['update'] == 1: raise RuntimeError('planned crash')
        return saved
    monkeypatch.setattr(epoch, 'save_epoch_checkpoint', interrupt)
    partial, data, kwargs = fixture_run()
    with pytest.raises(RuntimeError, match='planned crash'):
        epoch.train_epoch(partial, data, None, tmp_path/'resumed', **kwargs)
    monkeypatch.setattr(epoch, 'save_epoch_checkpoint', original)
    resumed, _, _ = fixture_run()
    final = epoch.train_epoch(resumed, data, None, tmp_path/'resumed', resume=True, **kwargs)
    assert full['convergence'] == final['convergence']
    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key], atol=0, rtol=0)
    states = []
    for name in ('full','resumed'):
        directory = Path(json.loads((tmp_path/name/'latest.json').read_text())['directory'])
        states.append(torch.load(directory/'training.pt', weights_only=True)['optimizer'])
    for left, right in zip(states[0]['masters'], states[1]['masters']):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    for key, state in states[0]['adam']['state'].items():
        for field, value in state.items():
            torch.testing.assert_close(value, states[1]['adam']['state'][key][field], atol=0, rtol=0)


def test_convergence_identity_does_not_change_legacy_training_contract():
    from openwam.train.stage12 import epoch_identity
    population = dict(source_yaml_sha256='a'*64,index_sha256='b'*64,total_samples=3)
    legacy = epoch_identity(dict(max_updates=6,repeat_population=True), population, 1)
    assert legacy['training'] == dict(mode='bounded_updates',max_updates=6,repeat_population=True)
    policy = dict(min_updates=2,eval_every=1,patience=2,relative_tolerance=.01)
    gated = epoch_identity(dict(max_updates=6,repeat_population=True,convergence=policy), population, 1)
    assert gated['training']['convergence'] == policy
    assert gated['config_sha256'] != legacy['config_sha256']
    with pytest.raises(ValueError, match='max_updates'):
        epoch_identity(dict(convergence=policy), population, 1)


def test_completed_resume_revalidates_fixed_inputs(tmp_path, monkeypatch):
    import openwam.train.convergence as gate
    from openwam.train.goal_epoch import train_epoch
    monkeypatch.setattr(gate, 'fixed_flow_metric', constant_metric)
    expert, data, kwargs = fixture_run()
    train_epoch(expert, data, None, tmp_path, **kwargs)
    data.sample(0)[1]['targets'].point_valid[0, 0] = False
    clone, _, _ = fixture_run()
    with pytest.raises(RuntimeError, match='convergence state'):
        train_epoch(clone, data, None, tmp_path, resume=True, **kwargs)


def test_diagnostic_failure_checkpoints_invalid_evidence_before_error(tmp_path, monkeypatch):
    import openwam.train.convergence as gate
    from openwam.train.goal_epoch import train_epoch
    count = 0
    def broken(*args, **kwargs):
        nonlocal count
        count += 1
        if count > 1: raise RuntimeError('diagnostic failed')
        return constant_metric()
    monkeypatch.setattr(gate, 'fixed_flow_metric', broken)
    expert, data, kwargs = fixture_run()
    with pytest.raises(RuntimeError, match='diagnostic failed'):
        train_epoch(expert, data, None, tmp_path, **kwargs)
    latest = Path(json.loads((tmp_path/'latest.json').read_text())['directory'])
    assert json.loads((latest/'manifest.json').read_text())['update'] == 1
    state = json.loads((latest/'convergence_state.json').read_text())
    assert state['history'][-1] == dict(update=1, metric=None)
    assert not state['converged']


def test_completed_pipeline_revalidation_never_restarts_stage2(tmp_path, monkeypatch):
    import openwam.train.stage12 as pipeline
    _, _, kwargs = fixture_run()
    identity = kwargs['identity']; calls = []
    proof = dict(checkpoint='/verified/current.safetensors')
    monkeypatch.setattr(pipeline, 'verify_completion', lambda *args: proof)
    (tmp_path/'stage1').mkdir(); (tmp_path/'stage1/training_complete.json').write_text('{}')
    (tmp_path/'pipeline_state.json').write_text(json.dumps(dict(phase='complete',identity=identity)))
    for _ in range(2):
        pipeline.execute_stages(tmp_path, identity, ['stage1'], ['stage2'], resume=True, runner=lambda c: calls.append(c))
    assert calls == [['stage1'], ['stage1']]
    assert json.loads((tmp_path/'pipeline_state.json').read_text())['phase'] == 'complete'


def test_completed_pipeline_revalidation_failure_preserves_stage2_completion(tmp_path, monkeypatch):
    import openwam.train.stage12 as pipeline
    _, _, kwargs = fixture_run(); identity = kwargs['identity']; calls = []
    proof = dict(checkpoint='/verified/current.safetensors')
    monkeypatch.setattr(pipeline, 'verify_completion', lambda *args: proof)
    (tmp_path/'stage1').mkdir(); (tmp_path/'stage1/training_complete.json').write_text('{}')
    (tmp_path/'pipeline_state.json').write_text(json.dumps(dict(phase='complete',identity=identity)))
    def broken(command): raise RuntimeError('validation failed')
    with pytest.raises(RuntimeError, match='validation failed'):
        pipeline.execute_stages(tmp_path, identity, ['stage1'], ['stage2'], resume=True, runner=broken)
    pipeline.execute_stages(tmp_path, identity, ['stage1'], ['stage2'], resume=True, runner=lambda c: calls.append(c))
    assert calls == [['stage1']]


def test_real_visual_history_flow_gate_with_actual_optimizer(tmp_path):
    from openwam.train.goal_epoch import train_epoch, verify_completion
    from openwam.model.goal.dependencies import dependency_identity
    from test_visual_history import make_model, history
    _, data, kwargs = fixture_run()
    expert = make_model(); fixed_history = history()
    original = data.sample
    def sample(index):
        record, batch = original(index)
        return record, dict(batch, **fixed_history)
    data.sample = sample
    kwargs['identity']['representation'] = expert.representation
    kwargs['dependencies'] = dependency_identity(expert.codec.vae,text_dim=expert.context_width,
                                                  precomputed_text_fingerprint='d'*64)
    result = train_epoch(expert,data,None,tmp_path,learning_rate=1e-9,history_learning_rate=1e-9,**kwargs)
    assert result['updates'] == 2 and result['completion_kind'] == 'convergence'
    assert result == verify_completion(tmp_path, kwargs['identity'])
    for row in result['convergence']['history']:
        assert row['metric']['valid_images'] == 1 and row['metric']['valid_points'] == 2
        assert row['metric']['image_loss'] > 0 and row['metric']['tcp_loss'] > 0


def test_final_non_cadence_budget_cannot_authorize_handoff(tmp_path,monkeypatch):
    import openwam.train.convergence as gate
    from openwam.train.goal_epoch import train_epoch
    monkeypatch.setattr(gate,'fixed_flow_metric',constant_metric)
    expert,data,kwargs = fixture_run(max_updates=3)
    kwargs['convergence'].update(min_updates=3,eval_every=2,patience=1)
    with pytest.raises(RuntimeError,match='without convergence'):
        train_epoch(expert,data,None,tmp_path,**kwargs)
    latest = Path(json.loads((tmp_path/'latest.json').read_text())['directory'])
    state = json.loads((latest/'convergence_state.json').read_text())
    assert [row['update'] for row in state['history']] == [0,2,3]
    assert state['streak'] == 1 and not state['converged']
    assert not (tmp_path/'training_complete.json').exists()


def test_missing_global_tcp_evidence_reports_explicit_reason(tmp_path,monkeypatch):
    import openwam.train.convergence as gate
    from openwam.train.goal_epoch import train_epoch
    def no_tcp(*args,**kwargs):
        result = constant_metric(); result.update(valid_points=0,tcp_loss=0.,loss=.5)
        return result
    monkeypatch.setattr(gate,'fixed_flow_metric',no_tcp)
    expert,data,kwargs = fixture_run(max_updates=2)
    with pytest.raises(RuntimeError,match='without convergence'):
        train_epoch(expert,data,None,tmp_path,**kwargs)
    summary = json.loads((tmp_path/'summary.json').read_text())
    assert 'no valid TCP' in summary['convergence_last_invalid_reason']
    events = [json.loads(row) for row in (tmp_path/'progress.jsonl').read_text().splitlines()]
    diagnostic = [row for row in events if row['event'] == 'convergence_evaluation']
    assert all('no valid TCP' in row['invalid_reason'] for row in diagnostic)
