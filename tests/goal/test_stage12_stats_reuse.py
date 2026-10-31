"""Offline statistics must be checked before costly Stage1 population preparation."""
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from tests.dataloader.test_curated_stage2 import make_native_source


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    import openwam.train.stage12 as module
    import openwam.train.goal_curated as goal
    from openwam.dataloader.curated_stage2 import CuratedStage2Dataset
    from openwam.dataloader.native_stage2_stats import prepare_native_stats

    source = make_native_source(tmp_path)
    dataset = CuratedStage2Dataset(source)
    registry = prepare_native_stats(dataset, tmp_path/'statistics',
        max_episodes_per_profile=1, max_windows_per_profile=4)
    stage1 = dict(model=dict(goal_representation='query_latent'), dataset={})
    configs = tmp_path/'configs'
    configs.mkdir()
    (configs/'stage2.yaml').write_text(yaml.safe_dump(dict(
        model=dict(goal_representation='query_latent'), dataloader={})))
    config = dict(source_yaml=str(source), output_dir=str(tmp_path/'output'),
        stage1_config='unused', stage2_config='stage2', stats_registry=registry,
        world_size=1)
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'load_pipeline', lambda path: config)
    monkeypatch.setattr(goal, 'load_config', lambda *a, **kw: stage1)
    calls = []
    return module, config, dataset, calls


def prepare(pipeline):
    module, config, dataset, calls = pipeline
    return module.main(['--config', 'unused', '--prepare'],
        _runner=lambda command, **kw: calls.append(command))


def rewrite_registry(config, mutate):
    payload = json.loads(Path(config['stats_registry']).read_text())
    mutate(payload)
    raw = json.dumps(payload, sort_keys=True).encode()
    path = Path(config['stats_registry']).with_name('registry-'+hashlib.sha256(raw).hexdigest()+'.json')
    path.write_bytes(raw)
    config['stats_registry'] = str(path)


def test_prepare_reuses_immutable_registry_and_audits_before_stage1(pipeline, monkeypatch):
    import openwam.dataloader.native_stage2_stats as stats
    module, config, dataset, calls = pipeline
    registry = Path(config['stats_registry'])
    before = registry.read_bytes()
    original_audit = stats.audit_selection
    audits = []
    def audit(actual):
        assert calls == [], 'metadata must be checked before Stage1 starts'
        audits.append(actual)
        return original_audit(actual)
    monkeypatch.setattr(stats, 'audit_selection', audit)
    monkeypatch.setattr(stats, 'prepare_native_stats', lambda *a, **kw: pytest.fail('must reuse supplied registry'))
    assert prepare(pipeline) == 0
    marker = json.loads((Path(config['output_dir'])/'prepared.json').read_text())
    assert marker['stats_registry'] == str(registry)
    assert len(calls) == 1 and calls[0][-1] == '--prepare'
    assert len(audits) == 1 and audits[0].registry is not None
    assert registry.read_bytes() == before


@pytest.mark.parametrize('case', ['missing', 'checksum', 'selection', 'schema', 'source', 'calibration', 'metadata', 'entry', 'entry_contract'])
def test_invalid_registry_fails_before_stage1(pipeline, case):
    module, config, dataset, calls = pipeline
    if case == 'missing':
        config['stats_registry'] += '.missing'
    elif case == 'checksum':
        with Path(config['stats_registry']).open('a') as stream:
            stream.write(' ')
    elif case in ('selection', 'schema'):
        key = 'selection_sha256' if case == 'selection' else 'source_pose_mask_policy'
        rewrite_registry(config, lambda p: p['contract'].__setitem__(key, 'stale'))
    elif case == 'entry':
        rewrite_registry(config, lambda p: p['entries'].clear())
    elif case == 'entry_contract':
        rewrite_registry(config, lambda p: next(iter(p['entries'].values()))['contract'].clear())
    else:
        key = dict(source='info_sha256', calibration='calibration_sha256', metadata='episode_metadata_audit')[case]
        rewrite_registry(config, lambda p: next(iter(p['sources'].values())).__setitem__(key, 'stale'))
    with pytest.raises((ValueError, FileNotFoundError)):
        prepare(pipeline)
    assert calls == []
    assert not (Path(config['output_dir'])/'prepared.json').exists()


def test_actual_dataloader_override_is_checked_before_stage1(pipeline):
    module, config, dataset, calls = pipeline
    config['stage2_overrides'] = ['++dataloader.gripper_state_filter.enabled=false']
    with pytest.raises(ValueError, match='selection/contract fingerprint_mismatch'):
        prepare(pipeline)
    assert calls == []


@pytest.mark.parametrize('stream', ['action', 'state'])
@pytest.mark.parametrize('case', [
    'missing_stream', 'missing_vector', 'short', 'nested', 'string', 'boolean',
    'nan', 'infinity', 'negative_count', 'fractional_count', 'boolean_count',
    'reversed_extrema', 'reversed_quantiles', 'negative_std', 'missing_joint',
])
def test_malformed_statistics_payload_fails_before_stage1(pipeline, stream, case):
    module, config, dataset, calls = pipeline
    def mutate(payload):
        entry = next(iter(payload['entries'].values()))
        if case == 'missing_stream':
            del entry[stream]
            return
        stats = entry[stream]
        if case == 'missing_vector':
            del stats['q99']
        elif case == 'short':
            stats['mean'].pop()
        elif case == 'nested':
            stats['mean'] = [stats['mean']]
        elif case in ('string', 'boolean', 'nan', 'infinity'):
            stats['mean'][0] = dict(string='0', boolean=True, nan=float('nan'), infinity=float('inf'))[case]
        elif case in ('negative_count', 'fractional_count', 'boolean_count'):
            stats['count'][0] = dict(negative_count=-1, fractional_count=1.5, boolean_count=True)[case]
        elif case == 'reversed_extrema':
            stats['min'][0] = stats['max'][0] + 1
        elif case == 'reversed_quantiles':
            stats['q01'][0] = stats['q99'][0] + 1
        elif case == 'negative_std':
            stats['std'][0] = -1
        else:
            assert entry['contract']['joint_indices']['left']
            stats['count'][20] = 0
    rewrite_registry(config, mutate)
    with pytest.raises(ValueError, match='stats.*(payload|population)'):
        prepare(pipeline)
    assert calls == []
    assert not (Path(config['output_dir'])/'prepared.json').exists()


def test_no_registry_keeps_existing_sampled_preparation(pipeline, monkeypatch):
    import openwam.dataloader.native_stage2_stats as stats
    module, config, dataset, calls = pipeline
    config['stats_registry'] = None
    original = stats.prepare_native_stats
    sampled = []
    def generate(actual, output, **kwargs):
        assert len(calls) == 1 and calls[0][-1] == '--prepare'
        sampled.append(kwargs)
        return original(actual, output, **kwargs)
    monkeypatch.setattr(stats, 'prepare_native_stats', generate)
    config.update(stats_episodes_per_profile=1, stats_windows_per_profile=3)
    assert prepare(pipeline) == 0
    assert len(sampled) == 1
    assert sampled[0]['max_windows_per_profile'] == 3
    assert (Path(config['output_dir'])/'prepared.json').is_file()
