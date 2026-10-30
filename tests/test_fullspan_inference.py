from copy import deepcopy
import json

import pytest
import torch
from PIL import Image

from scripts.experiments.fullspan_inference import (
    conditioning_identity, latest_checkpoint, resume_complete, publish_result,
)


def sample():
    return dict(trace=dict(identity=['repo', 1, 10], source_repo='repo', episode=1, anchor=10),
        prompt='move cup', granularity='task', instruction_start=0, instruction_end_exclusive=50,
        goal_observed_multiview=torch.zeros(3, 1, 4, 4), goal_view_valid=torch.tensor([True, False, False]),
        history_images=torch.zeros(10, 3, 1, 4, 4), history_valid=torch.zeros(10, dtype=torch.bool),
        history_ages=torch.zeros(10), history_episode_id=torch.tensor(1),
        proprio=torch.zeros(112), proprio_mask=torch.ones(112, dtype=torch.bool),
        first_frame_image=[Image.new('RGB', (4, 4))], goal_seed=42, goal_head_valid=True,
        _eval={'active_action_mask': torch.ones(112, dtype=torch.bool)})


def test_condition_identity_changes_with_current_history_state_and_prompt_not_gt():
    s = sample(); identity = conditioning_identity(s)
    for mutate in (lambda x: x['proprio'].__setitem__(0, 1),
                   lambda x: x['history_images'].__setitem__((0, 0, 0, 0, 0), 1),
                   lambda x: x.update(prompt='other'), lambda x: x['trace'].update(anchor=20)):
        other = deepcopy(s); mutate(other)
        assert conditioning_identity(other) != identity
    s['action'] = torch.ones(32, 112); s['report_target'] = {'uv': [100, 200]}
    assert conditioning_identity(s) == identity


def test_latest_checkpoint_numeric_order(tmp_path):
    for name in ['checkpoint_step_9.safetensors', 'checkpoint_step_530.safetensors']:
        (tmp_path / name).touch()
    assert latest_checkpoint(tmp_path).name == 'checkpoint_step_530.safetensors'


def test_atomic_publication_resume_verifies_identity_and_payload(tmp_path):
    folder = tmp_path / 'case'; identity = {'input_sha256': 'abc', 'steps': 20}
    payload = {'normalized_actions': torch.zeros(32, 112)}
    publish_result(folder, identity, payload, {'prediction.png': Image.new('RGB', (4, 4))})
    assert resume_complete(folder, identity)
    with pytest.raises(ValueError, match='identity'):
        resume_complete(folder, dict(identity, steps=10))
    (folder / 'prediction.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='artifact'):
        resume_complete(folder, identity)
    assert not resume_complete(tmp_path / 'absent', identity)


def test_each_anchor_generates_fresh_actions_and_gt_never_enters_model(monkeypatch):
    from types import SimpleNamespace
    import numpy as np
    from scripts.experiments.fullspan_inference import infer_stage2
    from openwam.deploy import denoise_schedule
    monkeypatch.setattr(denoise_schedule, 'make_schedule', lambda *a, **k: 'schedule')
    monkeypatch.setattr(torch.cuda, 'manual_seed_all', lambda seed: None)
    calls = []

    class Model:
        goal_expert = SimpleNamespace(observation_mode='multiview')
        action_backbone = SimpleNamespace(shift_action=5.)
        video_backbone = SimpleNamespace(shift_video=5.)
        video_scheduler = action_scheduler = object()
        dtype = torch.bfloat16
        goal_config = {'num_steps': 1}
        goal_diagnostics = dict(predicted_latent=torch.zeros(1, 120, 8),
            predicted_uv=torch.full((1, 2, 2), float('nan')),
            image_token_valid=torch.ones(1, 120, dtype=torch.bool))

        def generate(self, schedule, prompt, **kwargs):
            calls.append((prompt, kwargs))
            return dict(normalized_actions=np.full((32, 112), len(calls)), video=None)

    s = sample()
    s['goal_observed_multiview'] = torch.zeros(3, 1, 384, 320)
    s['action'] = object(); s['report_target'] = object(); s['video'] = object()
    first, _, _ = infer_stage2(Model(), s, 42, 20, 'none')
    s['first_frame_image'] = [Image.new('RGB', (4, 4), 'red')]
    s['trace']['anchor'] = 20
    second, _, _ = infer_stage2(Model(), s, 43, 20, 'none')
    assert len(calls) == 2
    assert not torch.equal(first['normalized_actions'], second['normalized_actions'])
    for _, kwargs in calls:
        assert not {'action', 'video', 'report_target', 'action_mask'} & kwargs.keys()


def test_checkpoint_hash_cache_reuses_only_exact_stat_identity(tmp_path, monkeypatch):
    from scripts.experiments import fullspan_inference as runner
    path = tmp_path / 'checkpoint'; path.write_bytes(b'first')
    cache = {}; calls = []
    original = runner.sha
    monkeypatch.setattr(runner, 'sha', lambda p: (calls.append(str(p)), original(p))[1])
    first = runner.cached_sha(path, cache)
    assert runner.cached_sha(path, cache) == first
    assert len(calls) == 1
    path.write_bytes(b'other')
    # Some workspace filesystems coalesce successive writes into one timestamp tick.
    import os
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    assert runner.cached_sha(path, cache) != first
    assert len(calls) == 2


def test_streaming_wait_requires_published_sample_identity(tmp_path):
    from scripts.experiments.fullspan_inference import wait_for_sample
    path = tmp_path / 'sample.pt'; manifest = tmp_path / 'manifest.json'
    record = dict(id=1, anchor=20, sample_path=str(path), status='pending')
    manifest.write_text(json.dumps(dict(status='materializing', anchors=[record])))
    with pytest.raises(ValueError, match='not ready'):
        wait_for_sample(record, manifest, wait=False)
    path.write_bytes(b'atomic sample')
    manifest.write_text(json.dumps(dict(status='materializing', anchors=[dict(record, status='ready')])))
    wait_for_sample(record, manifest, wait=False)
    manifest.write_text(json.dumps(dict(status='materializing', anchors=[dict(record, status='ready', anchor=30)])))
    with pytest.raises(ValueError, match='identity changed'):
        wait_for_sample(record, manifest, wait=False)


def test_verified_prompt_context_cache_is_reusable_but_detects_corruption(tmp_path):
    from scripts.experiments.fullspan_inference import load_verified_contexts, sha
    path = tmp_path / 'text_contexts.pt'
    contexts = {'move cup': (torch.ones(1, 4, 8), torch.tensor([3]))}
    torch.save(contexts, path)
    (tmp_path / 'text_contexts_source.json').write_text(json.dumps(dict(
        text_dependency_fingerprint='verified_t5', output_sha256=sha(path))))
    assert 'move cup' in load_verified_contexts(tmp_path, ['move cup'], 'verified_t5')
    assert load_verified_contexts(tmp_path, ['other'], 'verified_t5') is None
    with pytest.raises(ValueError, match='fingerprint'):
        load_verified_contexts(tmp_path, ['move cup'], 'wrong_t5')
    path.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='SHA'):
        load_verified_contexts(tmp_path, ['move cup'], 'verified_t5')


def test_anchor_major_orders_short_sequences_and_keeps_stage_resume_identity():
    from scripts.experiments.fullspan_inference import ordered_work, prediction_identity
    records = [dict(id=0, sequence_id='long', anchor=0), dict(id=1, sequence_id='short', anchor=10)]
    sequences = {'long': {'anchors': [0, 10, 20]}, 'short': {'anchors': [10]}}
    stages = ['stage1', 'joint', 'none']
    major = list(ordered_work(records, stages, sequences, anchor_major=True))
    assert [(r['id'], mode) for r, mode in major] == [(i, mode) for i in (1, 0) for mode in stages]
    original = list(ordered_work(records, stages, sequences, anchor_major=False))
    assert [(r['id'], mode) for r, mode in original] == [(i, mode) for mode in stages for i in (0, 1)]
    hashes = {mode: mode + '_sha' for mode in stages}
    configs = {mode: mode + '_config' for mode in stages}
    a = prediction_identity('stage1', records[0], 'sample_sha', hashes, configs, 100042, 20)
    b = prediction_identity('stage1', major[3][0], 'sample_sha', hashes, configs, 100042, 20)
    assert a == b
    assert a['seed'] == 100042 and a['steps'] == 20


def test_anchor_major_cpu_orchestration_loads_one_sample_once_and_resumes(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from scripts.experiments import fullspan_inference as runner
    run = tmp_path / 'run'; output = tmp_path / 'output'
    (run / 'stage1').mkdir(parents=True)
    (run / 'stage1/visual_history.safetensors').write_bytes(b'frozen')
    (run / 'stage1/protocol.json').write_text('{}')
    for mode in ('joint', 'none'):
        folder = run / 'stage2' / mode / 'checkpoints'; (folder / 'goal').mkdir(parents=True)
        (folder / 'goal/expert.safetensors').write_bytes(b'frozen')
        (folder / 'checkpoint_step_1.safetensors').write_bytes(mode.encode())
        (folder / 'config.yaml').write_text(mode)
    anchors = []
    for i, seq in enumerate(('long', 'short')):
        s = sample(); s['trace']['anchor'] = i * 10; s['_eval']['family'] = seq
        path = tmp_path / f'sample{i}.pt'; torch.save(s, path)
        anchors.append(dict(id=i, sequence_id=seq, anchor=i*10, sample_path=str(path), status='ready'))
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(dict(status='complete', anchors=anchors, sequences=[
        dict(sequence_id='long', anchors=[0, 10, 20], prompt='move cup'),
        dict(sequence_id='short', anchors=[10], prompt='move cup')])))
    calls = []; loads = []; reads = []
    def load_s1(*args):
        loads.append('stage1')
        return object(), {'move cup': (torch.ones(1, 4, 8), torch.tensor([3]))}
    def load_s2(path):
        mode = path.parent.parent.name; loads.append(mode)
        return SimpleNamespace(future_video_mode=mode)
    def infer_s1(model, batch, *, seed, steps):
        calls.append(('stage1', seed)); return Image.new('RGB', (4, 4)), [[.1, .2], [.3, .4]]
    def infer_s2(model, s, seed, steps, mode):
        calls.append((mode, seed))
        return {'normalized_actions': torch.zeros(32,112)}, {}, {}
    native_load = runner.load_anchor
    monkeypatch.setattr(runner, 'load_anchor', lambda r: (reads.append(r['id']), native_load(r))[1])
    monkeypatch.setattr(runner, 'load_stage1', load_s1)
    monkeypatch.setattr(runner, 'load_stage2', load_s2)
    monkeypatch.setattr(runner, 'make_batch', lambda *a, **k: {})
    monkeypatch.setattr(runner, 'infer', infer_s1)
    monkeypatch.setattr(runner, 'infer_stage2', infer_s2)
    for name in ('memory_allocated', 'max_memory_allocated'):
        monkeypatch.setattr(torch.cuda, name, lambda: 0)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(sys, 'argv', ['runner', '--run', str(run), '--manifest', str(manifest),
        '--output', str(output), '--anchor-major'])
    runner.main()
    assert loads == ['stage1', 'joint', 'none']
    assert reads == [1, 0]
    assert calls == [(mode, 100042+i) for i in (1, 0) for mode in ('stage1','joint','none')]
    before = {(mode, i): (output/'inference'/mode/f'{i:06d}'/'metadata.json').read_bytes()
              for mode in ('stage1', 'joint', 'none') for i in (0, 1)}
    runner.main()
    assert reads == [1, 0] and len(calls) == 6
    assert all((output/'inference'/mode/f'{i:06d}'/'metadata.json').read_bytes() == data
               for (mode, i), data in before.items())
    status = json.loads((output/'inference_status.json').read_text())
    assert status['completed'] == {'stage1': 2, 'joint': 2, 'none': 2}
    assert status['existing_results_unverified'] == {'stage1': 0, 'joint': 0, 'none': 0}


def test_native_stage1_configuration_replaces_legacy_smoke_protocol(tmp_path):
    import json,yaml
    from scripts.experiments.fullspan_inference import stage1_dependencies
    folder=tmp_path/'stage1';folder.mkdir()
    cfg={'baseline_checkpoint':'/verified/foundation.safetensors','components':{'vae':{}}}
    deps={'text':{'fingerprint':'verified'},'text_dim':4096}
    (folder/'resolved_config.yaml').write_text(yaml.safe_dump(cfg))
    (folder/'dependencies.json').write_text(json.dumps(deps))
    cache,actual_cfg,actual_deps,official=stage1_dependencies(tmp_path)
    assert cache==folder and actual_cfg==cfg and actual_deps==deps
    assert str(official)==cfg['baseline_checkpoint']


def test_native_stage1_config_identity_verifies_exported_cache(tmp_path):
    import json
    from scripts.experiments import fullspan_inference as runner
    folder=tmp_path/'stage1';folder.mkdir()
    names=['resolved_config.yaml','dependencies.json','text_contexts.pt']
    for name in names:(folder/name).write_bytes(name.encode())
    manifest={'files':{name:runner.sha(folder/name) for name in names},'identity':{'training':{'max_updates':1000}}}
    (folder/'artifacts.json').write_text(json.dumps(manifest))
    original=runner.stage1_config_sha(tmp_path)
    assert len(original)==64 and not (folder/'protocol.json').exists()
    (folder/'text_contexts.pt').write_bytes(b'changed cache')
    with pytest.raises(ValueError,match='artifact'):
        runner.stage1_config_sha(tmp_path)


def test_legacy_stage1_config_identity_preserves_protocol_hash(tmp_path):
    from scripts.experiments import fullspan_inference as runner
    folder=tmp_path/'stage1';folder.mkdir();(folder/'protocol.json').write_text('{}')
    assert runner.stage1_config_sha(tmp_path)==runner.sha(folder/'protocol.json')
