"""Strict contract rebind for the KF1-only native Stage1 continuation."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_kf1_contract_change(source, target):
    """Allow the pinned source, evaluation helper and G's KF1 mode to change."""
    old, new = copy.deepcopy(source), copy.deepcopy(target)
    if old.get('arch') not in ('G', 'Cstar') or new.get('arch') != old['arch']:
        raise ValueError('KF1 continuation requires unchanged G or Cstar architecture')
    if old['source']['reviewed_helpers'] != new['source']['reviewed_helpers']:
        raise ValueError('reviewed helper semantics differ')
    old_pin, new_pin = old['source']['pinned'], new['source']['pinned']
    for field in ('directory', 'python_tree_sha256'):
        if not old_pin.get(field) or not new_pin.get(field):
            raise ValueError('missing pinned source identity')
        old_pin[field] = new_pin[field]
    if set(old['runner']) != set(new['runner']):
        raise ValueError('runner file set differs')
    old['runner']['validation16.py'] = new['runner']['validation16.py']
    if old['arch'] == 'G':
        old_memory = old['model_config'].setdefault('memory', {})
        new_memory = new['model_config'].get('memory', {})
        if (old_memory.get('goal_horizon_mode', 'seconds') != 'seconds'
                or new_memory.get('goal_horizon_mode') != 'kf1_endpoint'):
            raise ValueError('expected explicit seconds to kf1_endpoint migration')
        old_memory['goal_horizon_mode'] = 'kf1_endpoint'
    if canonical(old) != canonical(new):
        raise ValueError('KF1 migration would change data, model, optimizer or execution semantics')


def assert_preserved(source, target):
    """Compare every tensor, dtype, RNG value and nested scalar exactly."""
    if type(source) is not type(target):
        raise ValueError('preserved value type differs')
    if torch.is_tensor(source):
        if source.dtype != target.dtype or not torch.equal(source, target):
            raise ValueError('preserved tensor differs')
    elif isinstance(source, np.ndarray):
        if source.dtype != target.dtype or not np.array_equal(source, target):
            raise ValueError('preserved numpy RNG differs')
    elif isinstance(source, dict):
        if source.keys() != target.keys():
            raise ValueError('preserved keys differ')
        for key in source:
            assert_preserved(source[key], target[key])
    elif isinstance(source, (list, tuple)):
        if len(source) != len(target):
            raise ValueError('preserved sequence differs')
        for first, second in zip(source, target):
            assert_preserved(first, second)
    elif source != target:
        raise ValueError('preserved scalar differs')


def rebind_native_checkpoint(source_manifest, destination, target_contract):
    """Publish a new immutable manifest after reopening and comparing all state.

    The caller protects the source generation and switches the run pointer only
    after this returns. No existing output or source file is overwritten.
    """
    source_manifest, destination = Path(source_manifest), Path(destination)
    parent = json.loads(source_manifest.read_text())
    validate_kf1_contract_change(parent['contract'], target_contract)
    if parent['format'] != 'openwam.stage1_longtrain.zero1.named_shards.v1':
        raise ValueError('unsupported native checkpoint format')
    if (parent['arch'] != parent['contract']['arch']
            or sorted(r['rank'] for r in parent['shards']) != list(range(parent['world']))):
        raise ValueError('checkpoint architecture or rank coverage differs')
    destination.mkdir(parents=True, exist_ok=False)
    manifest = copy.deepcopy(parent)
    manifest['contract'] = target_contract
    source_weights = source_manifest.parent / parent['weights']['file']
    if source_weights.stat().st_size != parent['weights']['bytes'] or sha256(source_weights) != parent['weights']['sha256']:
        raise ValueError('source weights checksum differs')
    weights = destination / source_weights.name
    with safe_open(str(source_weights), framework='pt', device='cpu') as original:
        metadata = original.metadata()
        bundle_key = 'goal_memory_bundle' if parent['arch'] == 'G' else 'visual_history'
        bundle = json.loads(metadata[bundle_key])
        if parent['arch'] == 'G':
            bundle['goal_memory']['goal_horizon_mode'] = 'kf1_endpoint'
        bundle.setdefault('training', {})['contract'] = target_contract
        metadata[bundle_key] = canonical(bundle)
        tensors = {name: original.get_tensor(name) for name in original.keys()}
        save_file(tensors, str(weights), metadata=metadata)
        with safe_open(str(weights), framework='pt', device='cpu') as reopened:
            if set(original.keys()) != set(reopened.keys()):
                raise ValueError('model tensor keys differ')
            for name in original.keys():
                assert_preserved(original.get_tensor(name), reopened.get_tensor(name))
        if parent['arch'] == 'Cstar':
            weights.with_suffix('.json').write_text(canonical(bundle) + '\n')
        all_names = set()
        for receipt in manifest['shards']:
            old_receipt = next(r for r in parent['shards'] if r['rank'] == receipt['rank'])
            shard = source_manifest.parent / receipt['file']
            if shard.stat().st_size != receipt['bytes'] or sha256(shard) != receipt['sha256']:
                raise ValueError('source optimizer shard checksum differs')
            payload = torch.load(shard, map_location='cpu', mmap=True, weights_only=False)
            expected = dict(format=parent['format'], update=parent['update'], global_cursor=parent['global_cursor'],
                            rank=receipt['rank'], world=parent['world'], contract=canonical(parent['contract']))
            if any(payload[key] != value for key, value in expected.items()):
                raise ValueError('source optimizer identity differs')
            if sorted(payload['states']) != old_receipt['names'] or all_names.intersection(payload['states']):
                raise ValueError('optimizer parameter ownership differs')
            all_names.update(payload['states'])
            for name, state in payload['states'].items():
                if set(state) not in ({'master'}, {'master', 'step', 'exp_avg', 'exp_avg_sq'}):
                    raise ValueError('optimizer state keys differ')
                model_tensor = original.get_tensor(name)
                if not torch.equal(state['master'].to(model_tensor.dtype), model_tensor):
                    raise ValueError('source FP32 master/model mismatch')
                for key, value in state.items():
                    if not torch.is_tensor(value) or not torch.isfinite(value).all():
                        raise ValueError('invalid optimizer tensor')
                    if key == 'step':
                        if value.numel() != 1 or not 0 <= float(value) <= parent['update'] or float(value) != int(value):
                            raise ValueError('invalid Adam step')
                    elif value.dtype != torch.float32 or value.shape != model_tensor.shape:
                        raise ValueError('invalid FP32 optimizer shape/dtype')
                if 'exp_avg_sq' in state and (state['exp_avg_sq'] < 0).any():
                    raise ValueError('negative Adam second moment')
            payload['contract'] = canonical(target_contract)
            target = destination / receipt['file']
            torch.save(payload, target)
            reopened = torch.load(target, map_location='cpu', mmap=True, weights_only=False)
            assert_preserved(payload, reopened)
            # Verify against the source again with only the authorized contract edit.
            reopened['contract'] = canonical(parent['contract'])
            assert_preserved(torch.load(shard, map_location='cpu', mmap=True, weights_only=False), reopened)
            receipt.update(bytes=target.stat().st_size, sha256=sha256(target))
            del payload, reopened
    manifest['weights'].update(bytes=weights.stat().st_size, sha256=sha256(weights))
    manifest['lineage'] = dict(reason='KF1 endpoint fixed0 and copy/region evaluation',
                               source_manifest=str(source_manifest), source_manifest_sha256=sha256(source_manifest),
                               preserved_model_tensors=len(tensors), preserved_optimizer_names=len(all_names),
                               update_cursor_rng_preserved=True)
    del tensors
    for path in destination.iterdir():
        with path.open('rb') as file:
            os.fsync(file.fileno())
    temporary = destination / 'manifest.pending.json'
    temporary.write_text(canonical(manifest) + '\n')
    with temporary.open('rb') as file:
        os.fsync(file.fileno())
    temporary.replace(destination / 'manifest.json')
    descriptor = os.open(destination, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return manifest
