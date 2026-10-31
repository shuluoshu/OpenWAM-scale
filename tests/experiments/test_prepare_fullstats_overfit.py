"""CPU contracts for full-registry normalization and bounded overfit filters."""
import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest


def helpers():
    from scripts.experiments import prepare_fullstats_overfit
    return prepare_fullstats_overfit


def test_filter_is_strict_subset_and_does_not_rebind_full_selection():
    h = helpers()
    repos = [dict(repo='a', family='f', path='/a', weight=1., ranges=[(7, 0, 100), (8, 0, 50)]),
             dict(repo='b', family='g', path='/b', weight=1., ranges=[(0, 0, 20)])]
    ds = SimpleNamespace(selection=dict(source_yaml='/full.yaml', repos=repos), repos=repos,
                         source_yaml_sha256='fullhash', registry={'identity': 'full'})
    before = copy.deepcopy(ds.selection)
    seq = [dict(repo='a', family='f', episode=7, start=10, end_exclusive=90, scope='task')]
    provenance = h.restrict_preparation(ds, seq)
    assert ds.repos[0]['ranges'] == [(7, 10, 90)]
    assert ds.selection == before and ds.selection['repos'] is repos
    assert ds.source_yaml_sha256 == 'fullhash' and ds.registry == {'identity': 'full'}
    assert provenance['normalization_selection_unchanged'] is True
    assert provenance['full_repository_count'] == 2
    with pytest.raises(ValueError, match='outside full selection'):
        h.restrict_preparation(ds, [dict(seq[0], end_exclusive=101)])


def test_subset_cannot_bridge_unselected_gap_or_spoof_family():
    h = helpers()
    repo = dict(repo='a', family='f', path='/a', weight=1., ranges=[(0, 0, 10), (0, 20, 40)])
    ds = SimpleNamespace(selection=dict(repos=[repo]), repos=[repo])
    seq = dict(repo='a', family='f', episode=0, start=5, end_exclusive=25, scope='task')
    with pytest.raises(ValueError, match='outside full selection'):
        h.restrict_preparation(ds, [seq])
    with pytest.raises(ValueError, match='family'):
        h.restrict_preparation(ds, [dict(seq, family='wrong', start=0, end_exclusive=5)])


def test_readiness_rejects_changed_registry_and_incomplete_publication(tmp_path):
    h = helpers()
    source = tmp_path/'source.yaml'; source.write_text('full source')
    registry = tmp_path/'registry.json'; registry.write_text('{"preparation":{"complete":true}}')
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    value = dict(status='complete', stats_registry=str(registry), registry_sha256=sha(registry),
                 source_yaml=str(source), source_sha256=sha(source), horizon=32, dimensions=112, raw_dimensions=52)
    readiness = tmp_path/'readiness.json'; readiness.write_text(json.dumps(value))
    assert h.verify_readiness(readiness)['registry_sha256'] == sha(registry)
    registry.write_text('{"changed":true}')
    with pytest.raises(ValueError, match='registry.*changed'):
        h.verify_readiness(readiness)
    value['status'] = 'running'; readiness.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='complete'):
        h.verify_readiness(readiness)


def test_training_covers_valid_stride_anchors_short_tails_but_not_terminal():
    h = helpers()
    from openwam.dataloader.curated_stage2 import CuratedStage2Dataset
    instruction = dict(start=0, end_exclusive=32, text='complete task')
    m = dict(length=32, tasks=[instruction], subtasks=[], columns={'training.valid': np.ones(32)}, bad=np.zeros(32, bool))
    m['columns']['training.valid'][10] = 0
    repo = dict(repo='a', family='f', ranges=[(0, 0, 32)])
    class Dataset:
        repos = [repo]
        _options = CuratedStage2Dataset._options
        def _episode(self, repo, episode): return m
        def raw_window(self, metadata, option, anchor):
            n = min(32, option['last_source']-anchor)
            mask = np.ones((n, 52), bool)
            if anchor == 20: mask[:] = False
            return None, np.zeros((n, 52)), mask, None, None, None
    seq = dict(repo='a', family='f', episode=0, start=0, end_exclusive=32, scope='task')
    descriptors, coverage = h.select_training_descriptors(Dataset(), [seq])
    assert [d['anchor'] for d in descriptors] == [0, 30]
    assert descriptors[-1]['option']['last_source']-descriptors[-1]['anchor'] == 1
    excluded = {r['anchor']: r['reason'] for r in coverage['report_only']}
    assert excluded == {10: 'native_anchor_filter', 20: 'no_action_supervision', 31: 'terminal'}
    assert coverage['display_anchor_count'] == 5
    assert coverage['trained_anchor_count'] == 2


def test_roundtrip_floor_uses_same_raw_values_and_ignores_padding():
    h = helpers()
    raw = np.zeros((2, 52), np.float32); raw[:, 0] = [2., 999.]; raw[:, 9] = [.3, .7]
    mask = np.zeros_like(raw, bool); mask[0, [0, 9]] = True
    old = dict(q01=[-1.]*52, q99=[1.]*52, count=[1]*52)
    new = dict(old, q01=[-3.]*52, q99=[3.]*52)
    a = h.normalization_floor(raw, mask, old)
    b = h.normalization_floor(raw, mask, new)
    assert a['valid_scalars'] == b['valid_scalars'] == 2
    assert a['clipped_scalars'] == 1 and b['clipped_scalars'] == 0
    assert a['raw_roundtrip_mae'] == pytest.approx(.5)
    assert b['raw_roundtrip_mae'] < 1e-6
    assert a['by_raw_dimension'][0]['clipped_scalars'] == 1
    assert a['by_raw_dimension'][9]['raw_roundtrip_max'] == 0


def test_masked_loss_preflight_detects_wrong_width_and_ignores_invalid_values():
    h = helpers()
    import torch
    sample = dict(action=torch.zeros(32, 112), action_mask=torch.zeros(32, 112, dtype=torch.bool))
    sample['action_mask'][0, 80] = True
    result = h.validate_masked_loss(sample)
    assert result['valid_elements'] == 1 and result['masked_unit_error_loss'] == pytest.approx(1.)
    with pytest.raises(ValueError, match='32.*112'):
        h.validate_masked_loss(dict(action=torch.zeros(32, 52), action_mask=torch.zeros(32, 52, dtype=torch.bool)))


def test_config_only_binds_null_registry_and_preserves_full_source(tmp_path):
    import yaml
    h = helpers()
    path = tmp_path/'comparison_config.yaml'
    config = dict(dataloader=dict(source_yaml='/full.yaml', normalization_registry=None,
                                 history=dict(enabled=True)), goal=dict(checkpoint='/stage1'))
    path.write_text(yaml.safe_dump(config))
    ready = dict(source_yaml='/full.yaml', stats_registry='/registry-new.json')
    bound = h.bind_config(path, ready)
    expected = copy.deepcopy(config); expected['dataloader']['normalization_registry'] = '/registry-new.json'
    assert bound == expected and yaml.safe_load(path.read_text()) == expected
    with pytest.raises(ValueError, match='previously bound'):
        h.bind_config(path, dict(ready, stats_registry='/registry-other.json'))
    with pytest.raises(ValueError, match='full normalization source'):
        h.bind_config(path, dict(ready, source_yaml='/three-episode.yaml'))


def test_complete_readiness_cannot_publish_incomplete_registry(tmp_path):
    h = helpers()
    source = tmp_path/'source.yaml'; source.write_text('full')
    registry = tmp_path/'registry.json'; registry.write_text(json.dumps({'preparation': {'complete': False}}))
    ready = tmp_path/'readiness.json'
    ready.write_text(json.dumps(dict(status='complete', source_yaml=str(source), stats_registry=str(registry),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        registry_sha256=hashlib.sha256(registry.read_bytes()).hexdigest(), horizon=32, dimensions=112, raw_dimensions=52)))
    with pytest.raises(ValueError, match='registry preparation is not complete'):
        h.verify_readiness(ready)
