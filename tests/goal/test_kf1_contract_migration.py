import copy

import pytest

from openwam.train.goal_memory_kf1_migration import validate_kf1_contract_change


def contracts(arch='G'):
    old = dict(arch=arch, model_config=dict(memory=dict(history_dropout=.15)),
               source=dict(pinned=dict(directory='/old', python_tree_sha256='old'), reviewed_helpers={'a': 'b'}),
               runner={'trainer.py': 'a', 'validation16.py': 'b'},
               data={'fingerprint': 'data'}, execution={'global_batch_size': 2560}, noise='noise')
    new = copy.deepcopy(old)
    new['source']['pinned'] = dict(directory='/new', python_tree_sha256='new')
    new['runner']['validation16.py'] = 'c'
    if arch == 'G':
        new['model_config']['memory']['goal_horizon_mode'] = 'kf1_endpoint'
    return old, new


@pytest.mark.parametrize('arch', ['G', 'Cstar'])
def test_only_kf1_condition_and_evaluation_may_change(arch):
    old, new = contracts(arch)
    validate_kf1_contract_change(old, new)


@pytest.mark.parametrize('change', ['data', 'batch', 'noise', 'helper', 'trainer', 'dropout', 'mode'])
def test_migration_rejects_any_unapproved_semantic_change(change):
    old, new = contracts()
    if change == 'data': new['data']['fingerprint'] = 'different'
    if change == 'batch': new['execution']['global_batch_size'] = 1280
    if change == 'noise': new['noise'] = 'different'
    if change == 'helper': new['source']['reviewed_helpers']['a'] = 'different'
    if change == 'trainer': new['runner']['trainer.py'] = 'different'
    if change == 'dropout': new['model_config']['memory']['history_dropout'] = 0
    if change == 'mode': new['model_config']['memory']['goal_horizon_mode'] = 'seconds'
    with pytest.raises(ValueError):
        validate_kf1_contract_change(old, new)


def test_native_rebind_preserves_model_optimizer_and_rng(tmp_path):
    import json
    import random
    import numpy as np
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from openwam.train.goal_memory_kf1_migration import rebind_native_checkpoint, canonical, sha256, assert_preserved
    old, new = contracts()
    source = tmp_path / 'source'
    source.mkdir()
    weight = torch.arange(6).reshape(2, 3).to(torch.bfloat16)
    weights = source / 'goal_memory.safetensors'
    save_file({'weight': weight}, str(weights), metadata={'goal_memory_bundle': json.dumps({'goal_memory': {}, 'training': {}})})
    payload = dict(format='openwam.stage1_longtrain.zero1.named_shards.v1', update=12,
                   global_cursor=30720, rank=0, world=1, contract=canonical(old),
                   states={'weight': dict(master=weight.float(), step=torch.tensor(12.),
                                           exp_avg=torch.ones(2, 3), exp_avg_sq=torch.ones(2, 3) * 2)},
                   torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
                   python_rng=random.getstate(), cuda_rng=None)
    shard = source / 'optimizer_rank_00.pt'
    torch.save(payload, shard)
    manifest = dict(format=payload['format'], update=12, global_cursor=30720, world=1,
                    arch='G', contract=old,
                    weights=dict(file=weights.name, sha256=sha256(weights), bytes=weights.stat().st_size),
                    shards=[dict(rank=0, file=shard.name, names=['weight'], sha256=sha256(shard), bytes=shard.stat().st_size)])
    path = source / 'manifest.json'
    path.write_text(json.dumps(manifest))
    rebound = rebind_native_checkpoint(path, tmp_path / 'new', new)
    assert rebound['update'] == 12 and rebound['global_cursor'] == 30720
    result = torch.load(tmp_path / 'new' / shard.name, weights_only=False)
    assert result['contract'] == canonical(old)
    assert rebound['optimizer_contract'] == old
    assert __import__('os').path.samefile(tmp_path / 'new' / shard.name, shard)
    assert_preserved(payload, result)
    with safe_open(str(tmp_path / 'new' / weights.name), framework='pt') as restored:
        assert torch.equal(restored.get_tensor('weight'), weight)
        assert json.loads(restored.metadata()['goal_memory_bundle'])['goal_memory']['goal_horizon_mode'] == 'kf1_endpoint'
    assert sha256(shard) == manifest['shards'][0]['sha256']


def test_optimizer_source_contract_is_only_accepted_for_verified_kf1_bridge():
    from openwam.train.goal_memory_kf1_migration import optimizer_contract_for_restore, canonical
    old, new = contracts()
    manifest = dict(contract=new, optimizer_contract=old,
                    lineage=dict(optimizer_state_policy='source-shards-immutable-v1'))
    assert optimizer_contract_for_restore(manifest, new) == canonical(old)
    manifest['optimizer_contract']['execution']['global_batch_size'] = 8
    with pytest.raises(ValueError):
        optimizer_contract_for_restore(manifest, new)
    with pytest.raises(ValueError):
        optimizer_contract_for_restore(dict(contract=new, optimizer_contract=old), new)
