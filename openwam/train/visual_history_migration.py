"""Explicit, name-checked v1/v2 to causal language visual-history continuation.

This is deliberately separate from dataset-subset migration. The caller supplies
named parameter groups reconstructed with the source and target architectures,
and publishes the result into an independent run with a new preparation proof.
Common tensors are shared read-only with the input to bound CPU RAM when loading
large checkpoints with ``torch.load(..., mmap=True)``. This module never writes a
checkpoint, changes its source, creates an optimizer, or starts training.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import torch


def validate_continuation_identity(source, target):
    """Permit code/config changes while preserving the entire finite data contract."""
    required = {'code_sha256', 'config_sha256', 'source_yaml_sha256', 'index_sha256',
                'total_samples', 'world_size', 'seed', 'sampling', 'data_loading', 'representation'}
    if not required.issubset(source) or not required.issubset(target):
        raise ValueError('incomplete continuation identity')
    if source.get('representation') != 'visual_history_masked_sa':
        raise ValueError('continuation identity requires visual history')
    if source.get('training') is not None or target.get('training') is not None:
        raise ValueError('only finite non-convergence continuation is supported')
    mutable = {'code_sha256', 'config_sha256'}
    if {k: v for k, v in source.items() if k not in mutable} != {k: v for k, v in target.items() if k not in mutable}:
        raise ValueError('continuation changes the population/sampling/data contract identity')
    if source == target:
        raise ValueError('architecture continuation requires an explicit new identity')


def _group_names(groups):
    by_group, seen = {}, set()
    for group in groups:
        name, names = group.get('name'), group.get('params')
        if not isinstance(name, str) or name in by_group or not isinstance(names, (list, tuple)):
            raise ValueError('invalid or duplicate named parameter group')
        for parameter in names:
            if not isinstance(parameter, str) or parameter in seen:
                raise ValueError('invalid or duplicate parameter name')
            seen.add(parameter)
        by_group[name] = list(names)
    return by_group, seen


def _donor(name, adapter_block):
    if name.startswith('condition_text_embedding.'):
        return 'dit.text_embedding.' + name.removeprefix('condition_text_embedding.')
    if name.startswith(('visual_adapter.cross_attn.', 'visual_adapter.norm3.')):
        return f'dit.blocks.{adapter_block}.' + name.removeprefix('visual_adapter.')
    raise ValueError('cannot initialize unrecognized new parameter: ' + name)


def _validate_entry(name, entry, weight):
    if not isinstance(entry, dict) or 'master' not in entry:
        raise ValueError('missing FP32 master state: ' + name)
    for key in ('master', 'exp_avg', 'exp_avg_sq'):
        if key not in entry:
            continue
        tensor = entry[key]
        if not torch.is_tensor(tensor) or tensor.dtype != torch.float32:
            raise ValueError('invalid FP32 optimizer tensor: ' + name + ':' + key)
        if tensor.shape != weight.shape:
            raise ValueError('optimizer tensor shape mismatch: ' + name + ':' + key)
        if not torch.isfinite(tensor).all():
            raise ValueError('optimizer tensors must be finite: ' + name + ':' + key)
        if key == 'exp_avg_sq' and (tensor < 0).any():
            raise ValueError('negative optimizer second moment: ' + name)
    if ('exp_avg' in entry) != ('exp_avg_sq' in entry) or ('step' in entry) != ('exp_avg' in entry):
        raise ValueError('incomplete Adam moments/step: ' + name)
    if 'step' in entry:
        step = entry['step']
        if (not torch.is_tensor(step) or step.numel() != 1 or not torch.isfinite(step).all()
                or step.item() < 0 or step.item() != int(step.item())):
            raise ValueError('invalid Adam step: ' + name)
    # Repeated transformer layers have identical shapes. Confirm the supplied
    # name ordering also binds each FP32 master to the actual saved model weight.
    if not torch.equal(entry['master'].to(device=weight.device, dtype=weight.dtype), weight):
        raise ValueError('optimizer master disagrees with named model weight: ' + name)


def migrate_visual_history_state(source_state, source_optimizer, source_groups, target_groups,
                                 *, target_shapes, adapter_block=0):
    """Return ``(weights, zero1_payload, report)`` using exact parameter names.

    ``source_groups`` and ``target_groups`` are lists of ``{name, params}`` with
    parameter *names* in each real optimizer group's serialization order. The
    source list must come from the legacy topology, never from the new topology.
    Only text projection and adapter text cross-attention/norm parameters may be
    added. They copy the trained main DiT's weights/FP32 masters and receive fresh
    zero Adam moments and step zero. Existing history-pool/gate state is retained.
    Target group order and parameter order may change; group settings may not.
    """
    if type(adapter_block) is not int or adapter_block < 0:
        raise ValueError('invalid source adapter block')
    if source_optimizer.get('format') != 'openwam.zero1_fp32.v1':
        raise ValueError('unsupported optimizer checkpoint format')
    if set(source_state) - set(target_shapes):
        raise ValueError('existing model parameters would be removed')
    weights, initialized = {}, {}
    for name, shape in target_shapes.items():
        donor = name if name in source_state else _donor(name, adapter_block)
        if donor not in source_state:
            raise ValueError('new parameter initialization donor is absent: ' + donor)
        value = source_state[donor]
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(shape):
            raise ValueError('model parameter shape mismatch: ' + name)
        # Added modules are independent trainable copies, never tied aliases of
        # the main DiT. This also permits assign=True + safetensors publication.
        weights[name] = value if donor == name else value.detach().clone()
        if donor != name:
            initialized[name] = donor
    source_names, source_all = _group_names(source_groups)
    target_names, target_all = _group_names(target_groups)
    if source_all != set(source_state) or target_all != set(target_shapes):
        raise ValueError('named optimizer groups do not cover the complete model state')
    if source_names.keys() != target_names.keys():
        raise ValueError('optimizer group names differ')
    # Existing parameters retain their learning-rate group as well as moments.
    for group, names in source_names.items():
        if not set(names).issubset(target_names[group]):
            raise ValueError('existing optimizer parameter changed group: ' + group)
    adam = source_optimizer.get('adam', {})
    serialized = adam.get('param_groups', [])
    if len(serialized) != len(source_groups) or not isinstance(adam.get('state'), dict):
        raise ValueError('invalid consolidated optimizer payload')
    entries, settings, used = {}, {}, set()
    for names, group in zip(source_groups, serialized):
        if group.get('name') != names['name'] or len(group.get('params', [])) != len(names['params']):
            raise ValueError('source named group disagrees with serialized optimizer group')
        settings[group['name']] = {k: copy.deepcopy(v) for k, v in group.items() if k != 'params'}
        for name, index in zip(names['params'], group['params']):
            if type(index) is not int or index in used or index not in adam['state']:
                raise ValueError('invalid or duplicate optimizer parameter index')
            used.add(index)
            _validate_entry(name, adam['state'][index], source_state[name])
            entries[name] = adam['state'][index]
    if used != set(adam['state']):
        raise ValueError('unbound optimizer states in consolidated checkpoint')
    state, groups, next_index = {}, [], 0
    for group in target_groups:
        ids = []
        for name in group['params']:
            if name in entries:
                entry = entries[name]
            else:
                master = entries[initialized[name]]['master'].detach().cpu().clone()
                entry = dict(master=master, step=torch.tensor(0., dtype=torch.float32),
                             exp_avg=torch.zeros_like(master), exp_avg_sq=torch.zeros_like(master))
            state[next_index] = entry
            ids.append(next_index)
            next_index += 1
        groups.append({**settings[group['name']], 'params': ids})
    return weights, dict(format=source_optimizer['format'], adam=dict(state=state, param_groups=groups)), dict(
        format='openwam.visual_history_architecture_continuation.v1',
        preserved_parameters=len(entries), initialized_parameters=initialized,
        optimizer='existing named FP32 masters/moments retained; only new parameters start at step zero')


def load_migration_source(directory, *, expected_dependencies, expected_identity=None):
    """Verify all manifest files and frozen identities before returning CPU views.

    Returns ``(weights, training_state, bundle_metadata)``. The manifest/training
    identity is authoritative: bundle training metadata can contain older subset
    provenance. A caller must still validate the new identity and publish a new
    v3 bundle with the actual target topology and new training metadata.
    """
    from safetensors import safe_open
    from openwam.train.goal_epoch import sha256

    directory = Path(directory).resolve()
    manifest = json.loads((directory / 'manifest.json').read_text())
    files = manifest.get('files', {})
    required = {'visual_history.safetensors', 'visual_history.json', 'training.pt'}
    if not required.issubset(files):
        raise ValueError('source checkpoint manifest is incomplete')
    for name, digest in files.items():
        if Path(name).name != name or not isinstance(digest, str):
            raise ValueError('invalid source manifest filename/digest')
        if sha256(directory / name) != digest:
            raise ValueError('source checkpoint checksum mismatch: ' + name)
    if expected_identity is not None and manifest['identity'] != expected_identity:
        raise ValueError('source checkpoint identity mismatch')
    saved = torch.load(directory / 'training.pt', map_location='cpu', weights_only=True, mmap=True)
    if any(saved.get(key) != manifest.get(key) for key in ('identity', 'cursor', 'update')):
        raise ValueError('source training state identity/cursor/update mismatch')
    if (type(saved['cursor']) is not int or not 0 <= saved['cursor'] <= saved['identity']['total_samples']
            or type(saved['update']) is not int or saved['update'] < 0):
        raise ValueError('invalid source finite cursor/update')
    with safe_open(str(directory / 'visual_history.safetensors'), framework='pt', device='cpu') as bundle:
        metadata = json.loads((bundle.metadata() or {}).get('visual_history', '{}'))
        language = metadata.get('history', {}).get('language_conditioning', False)
        expected_format = 'openwam.visual_history.v2' if language else 'openwam.visual_history.v1'
        if (type(language) is not bool or metadata.get('format') != expected_format
                or metadata.get('history', {}).get('causal_history', False)):
            raise ValueError('source requires a verified v1 or v2 visual history format')
        if metadata.get('dependencies') != expected_dependencies:
            raise ValueError('source frozen dependency identities differ')
        if json.loads((directory / 'visual_history.json').read_text()) != metadata:
            raise ValueError('source bundle sidecar metadata differs')
        weights = {name: bundle.get_tensor(name) for name in bundle.keys()}
    return weights, saved, metadata
