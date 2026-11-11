"""Opt-in RFC goal-memory recipe and single-device optimizer loop.

No existing C* entry point is changed. Data, frozen codecs and text encoding
are bound by the CLI; the optimizer sees only the shared trainable policy.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch

ROOT = Path(__file__).resolve().parents[2]


def load_config(path):
    from omegaconf import OmegaConf
    def read(filename, seen):
        filename = Path(filename).resolve()
        if filename in seen:
            raise ValueError('cyclic base_config')
        value = OmegaConf.load(filename)
        base = value.pop('base_config', None)
        return OmegaConf.merge(read(filename.parent / base, seen | {filename}), value) if base else value
    config = OmegaConf.to_container(read(path, set()), resolve=True)
    if config.get('representation') != 'goal_memory' or config.get('stage') not in (1, 2):
        raise ValueError('goal_memory representation with stage 1 or 2 required')
    history = config.get('history', {})
    if history.get('fps', 1) != 1 or history.get('recent_frames', 2) not in (2, 3, 4):
        raise ValueError('this implementation requires 1Hz sink + 2..4 recent frames')
    probability = config.get('dataset', {}).get('subtask_probability', .7)
    if isinstance(probability, bool) or not isinstance(probability, (int, float)) or not 0 <= probability <= 1:
        raise ValueError('invalid subtask_probability')
    for key, default in (('max_updates', 50000), ('checkpoint_every', 1000)):
        value = config.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f'{key} must be positive integer')
        config[key] = value
    for key, default in (('learning_rate', 1e-5), ('new_learning_rate', 1e-4)):
        value = config.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'invalid {key}')
        config[key] = value
    from openwam.model.goal.goal_memory import GoalMemoryConfig
    memory = GoalMemoryConfig(**config.get('memory', {}))
    if memory.recent_frames != history.get('recent_frames', 2):
        raise ValueError('model/data recent_frames differ')
    betas = config.get('adam_betas', (.9, .999))
    if (not isinstance(betas, (list, tuple)) or len(betas) != 2 or
            any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<1 for v in betas)):
        raise ValueError('invalid adam_betas')
    for key,default,positive in (('weight_decay',.01,False),('max_grad_norm',1.,True),
                                  ('projection_learning_rate',config['new_learning_rate'],True)):
        value = config.get(key,default)
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or (value<=0 if positive else value<0):
            raise ValueError(f'invalid {key}')
    if type(config.get('memory_cache',{}).get('require_cache',True)) is not bool:
        raise ValueError('require_cache must be boolean')
    if config.get('optimizer_device', 'model') not in ('model', 'cpu'):
        raise ValueError('optimizer_device must be model or cpu')
    for section, keys in ((config, ('baseline_checkpoint', 'stage1_checkpoint')),
                          (config.get('dataset', {}), ('source_yaml', 'cache_dir', 'data_root', 'index_path', 'path_aliases')),
                          (config.get('action_data', {}), ('normalization_registry',)),
                          (config.get('memory_cache', {}), ('directory',))):
        for key in keys:
            if section.get(key) and isinstance(section[key], (str, Path)):
                value = Path(os.path.expandvars(os.path.expanduser(str(section[key]))))
                section[key] = str(value if value.is_absolute() else ROOT / value)
    return config


def parameter_groups(model, *, learning_rate, new_learning_rate, projection_learning_rate=None):
    """Shared native video backbone gets its pretrained LR; new modules their LR."""
    video = getattr(model, 'video', model)
    dit = getattr(video, 'dit', None)
    if dit is None:
        raise ValueError('model must expose shared VideoDiT through .dit or .video.dit')
    backbone_ids = {id(p) for p in dit.parameters()}
    projection_ids = {id(p) for p in getattr(model, 'proprio_encoder', torch.nn.Identity()).parameters()}
    backbone, new, projection = [], [], []
    for parameter in model.parameters():
        if parameter.requires_grad:
            (backbone if id(parameter) in backbone_ids else projection if id(parameter) in projection_ids else new).append(parameter)
    return [dict(params=p, lr=lr) for p, lr in ((backbone, learning_rate), (new, new_learning_rate),
                    (projection, projection_learning_rate or new_learning_rate)) if p]


def _publish_training(model, optimizer, output, *, update, contract, config,
                      dependencies, dit_config):
    """Publish one current generation for either stage, including zero-step recovery."""
    from openwam.train.goal_memory_checkpoint import save_training_state, save_memory_bundle, _execution_contract
    save_training_state(output / 'resume.pt', model, optimizer, update=update, contract=contract)
    training = dict(update=update, config=config, execution=_execution_contract(model))
    if config['stage'] == 1:
        save_memory_bundle(model, output / 'goal_memory.safetensors', dit_config=dit_config,
                           dependencies=dependencies, training=training)
    else:
        from openwam.train.goal_memory_stage2 import save_policy_bundle
        save_policy_bundle(model, output / 'policy.safetensors', dependencies=dependencies,
                           training=training)


def train_updates(model, batch_provider, *, output_dir, config, dependencies,
                  dit_config, population_identity, resume=None, stop_after=None):
    """Single process; global update is the deterministic mixture/data cursor.

    stop_after is for bounded interruption validation; it never changes the
    configured total budget or keyframe text-warmup schedule.
    """
    from openwam.train.goal_overfit import FP32MasterAdamW
    from openwam.train.goal_memory_checkpoint import restore_training_state, _execution_contract
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('goal_memory entry currently requires one process')
    output = Path(output_dir)
    if not resume and output.exists() and any(output.iterdir()):
        raise ValueError('nonempty output_dir; use explicit --resume or a new directory')
    output.mkdir(parents=True, exist_ok=True)
    contract = dict(config=config, dependencies=dependencies, population=population_identity,
                    dit_config=dit_config)
    groups = parameter_groups(model, learning_rate=config['learning_rate'],
                               new_learning_rate=config['new_learning_rate'],
                               projection_learning_rate=config.get('projection_learning_rate'))
    for group in groups:
        group['betas'] = tuple(config.get('adam_betas', (.9, .999)))
    optimizer = FP32MasterAdamW(groups, weight_decay=config.get('weight_decay', .01), master_device='cpu' if config.get('optimizer_device') == 'cpu' else None)
    update = restore_training_state(resume, model, optimizer, contract=contract) if resume else 0
    from openwam.train.goal_memory_checkpoint import _atomic
    _atomic(output/'dependencies.json',lambda name:Path(name).write_text(
        json.dumps(dependencies,indent=2,sort_keys=True,allow_nan=False)+'\n'))
    budget = config['max_updates']
    end = min(budget, update + stop_after) if stop_after is not None else budget
    if stop_after is not None and stop_after <= 0:
        raise ValueError('stop_after must be positive')
    if update > budget:
        raise ValueError('checkpoint update exceeds configured budget')
    last = {}
    initial_update = update
    model.train()
    while update < end:
        started = time.perf_counter()
        batch = dict(batch_provider(update))
        metadata = batch.pop('_metadata', {})
        optimizer.zero_grad()
        result = model.training_loss(**batch, update=update, total_updates=budget)
        if not torch.isfinite(result['loss']):
            raise FloatingPointError('nonfinite goal_memory loss')
        result['loss'].backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('max_grad_norm', 1.),
                                              error_if_nonfinite=True, foreach=False)
        optimizer.step()
        device = next(model.parameters()).device
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        update += 1
        last = dict(**metadata, update=update, seconds=time.perf_counter() - started,
                    gradient_norm=float(norm),
                    **{key: float(value.detach()) for key, value in result.items()
                       if torch.is_tensor(value) and value.numel() == 1})
        with (output / 'train.jsonl').open('a') as stream:
            stream.write(json.dumps(last, allow_nan=False) + '\n')
        print(json.dumps(last, allow_nan=False), flush=True)
        del result, batch
        if update % config['checkpoint_every'] == 0 or update == end:
            _publish_training(model, optimizer, output, update=update, contract=contract,
                              config=config, dependencies=dependencies, dit_config=dit_config)
    if update == initial_update:
        _publish_training(model, optimizer, output, update=update, contract=contract,
                          config=config, dependencies=dependencies, dit_config=dit_config)
    summary = dict(update=update, budget=budget, last=last, execution=_execution_contract(model),
                   status='complete' if update == budget else 'interrupted',
                   resume=str(output/'resume.pt' if (output/'resume.pt').is_file() else resume))
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary
