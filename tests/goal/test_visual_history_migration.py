"""Explicit architecture continuation must preserve names, masters and coverage."""
import copy
import importlib.util
import json

import pytest
import torch

from test_expert import TINY_DIT, TinyVAE
from openwam.model.goal.codec import GoalCodec
from openwam.model.goal.visual_history import VisualHistoryGoalExpert
from openwam.model.video_backbone.wan.models.dit import WanModel
from openwam.train.goal_curated import goal_parameter_groups


def test_migration_module_is_available():
    assert importlib.util.find_spec('openwam.train.visual_history_migration') is not None


def expert(language=False, causal=False):
    return VisualHistoryGoalExpert(WanModel(**TINY_DIT), GoalCodec(TinyVAE()),
        dit_config=TINY_DIT, num_history_tokens=20, pool_dim=32, pool_heads=4,
        language_conditioning=language, causal_history=causal).to(torch.bfloat16)


def named_groups(model):
    names = {id(p): n for n, p in model.named_parameters()}
    return [dict(name=g['name'], params=[names[id(p)] for p in g['params']])
            for g in goal_parameter_groups(model, 1e-5, 1e-4)]


def payload(model, groups):
    weights = model.state_dict()
    state, serialized, index = {}, [], 0
    for g in groups:
        ids = []
        for name in g['params']:
            master = weights[name].float().clone()
            # Preserve precision beyond the model's BF16 rounding.
            master = torch.nextafter(master, torch.full_like(master, float('inf')))
            state[index] = dict(master=master, step=torch.tensor(7.),
                                exp_avg=torch.full_like(master, .123),
                                exp_avg_sq=torch.full_like(master, .456))
            ids.append(index)
            index += 1
        serialized.append(dict(name=g['name'], params=ids,
                               lr=1e-5 if g['name'] == 'goal' else 1e-4,
                               betas=(.9, .95), eps=1e-8, weight_decay=.01))
    return dict(format='openwam.zero1_fp32.v1', adam=dict(state=state, param_groups=serialized))


def migration_inputs(language=False):
    old, new = expert(language), expert(True, True)
    old.history_gate.data.fill_(.25)
    groups = named_groups(old)
    return old, new, groups, payload(old, groups)


@pytest.mark.parametrize('language', [False, True], ids=['v1_to_v3', 'v2_to_v3'])
def test_named_migration_preserves_existing_state_and_initializes_only_new_parameters(language, tmp_path):
    from openwam.train.visual_history_migration import migrate_visual_history_state
    old, new, old_groups, saved = migration_inputs(language)
    new_groups = named_groups(new)
    # Force every preserved parameter to move, including the scalar gate.
    for group in new_groups:
        group['params'].reverse()
    new_groups.reverse()
    weights, migrated, report = migrate_visual_history_state(old.state_dict(), saved,
        old_groups, new_groups, target_shapes={n: p.shape for n, p in new.state_dict().items()})
    new.load_state_dict(weights, strict=True, assign=True)
    from openwam.train.goal_visual_history import save_visual_history_bundle
    metadata = save_visual_history_bundle(new, tmp_path / 'migrated.safetensors',
                                         precomputed_text_fingerprint='a' * 64)
    assert metadata['format'] == 'openwam.visual_history.v3'
    source = {name: saved['adam']['state'][pid]
              for names, group in zip(old_groups, saved['adam']['param_groups'])
              for name, pid in zip(names['params'], group['params'])}
    target = {name: migrated['adam']['state'][pid]
              for names, group in zip(new_groups, migrated['adam']['param_groups'])
              for name, pid in zip(names['params'], group['params'])}
    assert report['preserved_parameters'] == len(source)
    assert len(report['initialized_parameters']) == (0 if language else 16)
    for name, entry in source.items():
        assert target[name].keys() == entry.keys()
        for key, tensor in entry.items():
            assert torch.equal(target[name][key], tensor), (name, key)
        assert torch.equal(weights[name], old.state_dict()[name])
    for name, donor in report['initialized_parameters'].items():
        assert torch.equal(weights[name], old.state_dict()[donor])
        assert weights[name].untyped_storage().data_ptr() != weights[donor].untyped_storage().data_ptr()
        assert torch.equal(target[name]['master'], source[donor]['master'])
        assert target[name]['master'].untyped_storage().data_ptr() != source[donor]['master'].untyped_storage().data_ptr()
        assert target[name]['step'].item() == 0
        assert not target[name]['exp_avg'].any() and not target[name]['exp_avg_sq'].any()
    assert new.history_gate.item() == .25
    assert list(migrated['adam']['state']) == list(range(len(target)))
    for expected, actual in zip(new_groups, migrated['adam']['param_groups']):
        parent = next(g for g in saved['adam']['param_groups'] if g['name'] == expected['name'])
        assert {k: v for k, v in actual.items() if k != 'params'} == {k: v for k, v in parent.items() if k != 'params'}


@pytest.mark.parametrize('damage,match', [
    ('shape', 'shape'), ('unknown_new', 'initializ'), ('dropped', 'removed'),
    ('duplicate', 'duplicate'), ('bf16_master', 'FP32'),
    ('nan_moment', 'finite'), ('inf_moment', 'finite'),
    ('negative_second_moment', 'negative'), ('fractional_step', 'step'),
    ('wrong_name_order', 'master.*weight|weight.*master'), ('optimizer_format', 'format')])
def test_mapping_rejects_incompatible_or_ambiguous_states(damage, match):
    from openwam.train.visual_history_migration import migrate_visual_history_state
    old, new, old_groups, saved = migration_inputs()
    shapes = {n: p.shape for n, p in new.state_dict().items()}
    if damage == 'shape':
        shapes['history_gate'] = (2,)
    elif damage == 'unknown_new':
        shapes['unrecognized.weight'] = (2,)
    elif damage == 'dropped':
        del shapes['history_gate']
    elif damage == 'duplicate':
        old_groups[0]['params'][1] = old_groups[0]['params'][0]
    elif damage == 'bf16_master':
        saved['adam']['state'][0]['master'] = saved['adam']['state'][0]['master'].bfloat16()
    elif damage == 'nan_moment':
        saved['adam']['state'][0]['exp_avg'].fill_(float('nan'))
    elif damage == 'inf_moment':
        saved['adam']['state'][0]['exp_avg_sq'].fill_(float('inf'))
    elif damage == 'negative_second_moment':
        saved['adam']['state'][0]['exp_avg_sq'].fill_(-1.)
    elif damage == 'fractional_step':
        saved['adam']['state'][0]['step'].fill_(7.5)
    elif damage == 'wrong_name_order':
        # Same-shaped layers expose silent positional state reuse.
        names = old_groups[0]['params']
        i, j = names.index('dit.blocks.0.self_attn.q.weight'), names.index('dit.blocks.0.self_attn.k.weight')
        names[i], names[j] = names[j], names[i]
    elif damage == 'optimizer_format':
        saved['format'] = 'unknown'
    with pytest.raises(ValueError, match=match):
        migrate_visual_history_state(old.state_dict(), saved, old_groups, named_groups(new), target_shapes=shapes)


def identity():
    return dict(code_sha256='a' * 64, config_sha256='b' * 64,
        source_yaml_sha256='c' * 64, index_sha256='d' * 64, total_samples=100,
        world_size=32, seed=42, representation='visual_history_masked_sa',
        sampling=dict(mode='filtered_global_shuffle', excluded_sample_ranges=[[1, 2]]),
        data_loading=dict(batch_size=192, dataset_num_workers=20, prefetch_factor=10))


def test_continuation_identity_allows_only_explicit_architecture_change():
    from openwam.train.visual_history_migration import validate_continuation_identity
    old = identity()
    new = {**old, 'code_sha256': 'e' * 64, 'config_sha256': 'f' * 64}
    validate_continuation_identity(old, new)
    with pytest.raises(ValueError, match='new identity'):
        validate_continuation_identity(old, old)
    for key in ('source_yaml_sha256', 'index_sha256', 'total_samples', 'world_size', 'seed', 'sampling', 'data_loading'):
        with pytest.raises(ValueError, match='contract|identity'):
            validate_continuation_identity(old, {**new, key: None})


def test_verified_source_rejects_tampering_and_dependency_mismatch(tmp_path):
    from openwam.train.goal_epoch import atomic_json, sha256
    from openwam.train.goal_visual_history import save_visual_history_bundle
    from openwam.train.visual_history_migration import load_migration_source
    old = expert()
    meta = save_visual_history_bundle(old, tmp_path / 'visual_history.safetensors',
                                     precomputed_text_fingerprint='a' * 64)
    state = dict(identity=identity(), cursor=71, update=7, optimizer=payload(old, named_groups(old)))
    torch.save(state, tmp_path / 'training.pt')
    manifest = dict(identity=identity(), cursor=71, update=7,
        files={n: sha256(tmp_path / n) for n in ('training.pt', 'visual_history.safetensors', 'visual_history.json')})
    atomic_json(tmp_path / 'manifest.json', manifest)
    weights, loaded, metadata = load_migration_source(tmp_path, expected_dependencies=meta['dependencies'])
    assert loaded['cursor'] == 71 and loaded['update'] == 7
    assert metadata == json.loads(json.dumps(meta)) and set(weights) == set(old.state_dict())
    with pytest.raises(ValueError, match='dependenc'):
        load_migration_source(tmp_path, expected_dependencies={})
    with (tmp_path / 'training.pt').open('ab') as out:
        out.write(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        load_migration_source(tmp_path, expected_dependencies=meta['dependencies'])


def test_migrated_payload_restores_real_zero1_and_matches_next_fp32_adam_update(tmp_path):
    import torch.distributed as dist
    from openwam.train.goal_optimizer import ShardedFP32MasterAdamW
    from openwam.train.visual_history_migration import migrate_visual_history_state
    old, new, old_groups, saved = migration_inputs()
    groups = named_groups(new)
    weights, migrated, _ = migrate_visual_history_state(old.state_dict(), saved,
        old_groups, groups, target_shapes={n: p.shape for n, p in new.state_dict().items()})
    new.load_state_dict(weights, strict=True)
    reference = []
    ref_groups = []
    ref_state = {}
    for group in migrated['adam']['param_groups']:
        params = []
        for index in group['params']:
            state = migrated['adam']['state'][index]
            p = torch.nn.Parameter(state['master'].clone())
            params.append(p)
            reference.append(p)
            ref_state[p] = {k: v.clone() for k, v in state.items() if k != 'master'}
        ref_groups.append({**group, 'params': params})
    ref = torch.optim.AdamW(ref_groups)
    ref.state.update(ref_state)
    dist.init_process_group('gloo', init_method='file://' + str(tmp_path / 'init'), rank=0, world_size=1)
    try:
        optimizer = ShardedFP32MasterAdamW(goal_parameter_groups(new, 1e-5, 1e-4))
        optimizer.restore_payload(migrated)
        flat = [p for group in goal_parameter_groups(new, 1e-5, 1e-4) for p in group['params']]
        for p, q in zip(flat, reference):
            p.grad = torch.full_like(p, .125)
            q.grad = p.grad.float()
        optimizer.step()
        ref.step()
        for p, q in zip(flat, reference):
            torch.testing.assert_close(p, q.bfloat16(), rtol=0, atol=0)
            actual = optimizer.local_optimizer.state[p]
            torch.testing.assert_close(actual['master'], q, rtol=0, atol=0)
            for key in ('step', 'exp_avg', 'exp_avg_sq'):
                torch.testing.assert_close(actual[key], ref.state[q][key], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()
