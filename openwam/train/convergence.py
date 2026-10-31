"""Opt-in convergence evidence on fixed training examples, never held-out accuracy."""
from __future__ import annotations
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import fields, is_dataclass
import hashlib
import json
import math
import random


DIAGNOSTIC_VERSION = 'masked-fixed-flow-v1'


def convergence_config(value, max_updates):
    if value is None:
        return None
    keys = {'min_updates', 'eval_every', 'patience', 'relative_tolerance'}
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError('convergence requires exactly ' + ', '.join(sorted(keys)))
    if type(max_updates) is not int or max_updates < 1:
        raise ValueError('convergence requires positive max_updates')
    if any(type(value[k]) is not int or value[k] < 1 for k in keys - {'relative_tolerance'}):
        raise ValueError('convergence intervals must be positive integers')
    tolerance = value['relative_tolerance']
    if type(tolerance) not in (int, float) or not math.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError('invalid convergence relative_tolerance')
    if value['min_updates'] > max_updates:
        raise ValueError('convergence min_updates exceeds max_updates')
    return dict(value)


def new_state(policy, evaluation_identity):
    return dict(version=DIAGNOSTIC_VERSION, policy=dict(policy), evaluation_identity=evaluation_identity,
                history=[], best_loss=None, best_image_loss=None, best_tcp_loss=None,
                reference_loss=None, streak=0, converged=False)


def _valid_metric(metric):
    if not isinstance(metric, dict):
        return False
    for key in ('loss', 'image_loss', 'tcp_loss', 'lambda_image', 'lambda_tcp'):
        value = metric.get(key)
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            return False
    for key, weight in (('valid_images', 'lambda_image'), ('valid_points', 'lambda_tcp')):
        count = metric.get(key)
        if type(count) is not int or count < 0 or (metric[weight] > 0 and count == 0):
            return False
    expected = metric['lambda_image'] * metric['image_loss'] + metric['lambda_tcp'] * metric['tcp_loss']
    return (metric['lambda_image'] + metric['lambda_tcp'] > 0
            and math.isclose(metric['loss'], expected, rel_tol=1e-7, abs_tol=1e-12))


def observe(state, update, metric, *, max_updates):
    """Return a new JSON-safe state; only uninterrupted scheduled evidence counts."""
    policy = convergence_config(state['policy'], max_updates)
    if type(update) is not int or not 0 <= update <= max_updates or state['converged']:
        raise ValueError('invalid convergence update or already converged')
    history = state['history']
    expected = 0 if not history else history[-1]['update'] + policy['eval_every']
    # The only off-cadence evidence is the terminal budget diagnostic.
    if update != expected and not (history and update == max_updates and expected > update > history[-1]['update']):
        raise ValueError('convergence evaluation cadence mismatch')
    result = deepcopy(state)
    valid = _valid_metric(metric)
    previous_valid = next((row['metric'] for row in history if row['metric'] is not None), None)
    if valid and previous_valid is not None:
        valid = all(metric[k] == previous_valid[k] for k in
                    ('valid_images', 'valid_points', 'lambda_image', 'lambda_tcp'))
    metric = deepcopy(metric) if valid else None
    result['history'].append(dict(update=update, metric=metric))
    if update % policy['eval_every']:
        return result
    if not valid:
        result.update(streak=0, reference_loss=None)
        return result
    loss = metric['loss']
    best = loss if result['best_loss'] is None else min(loss, result['best_loss'])
    result['best_loss'] = best
    tolerance = policy['relative_tolerance']
    degraded = loss > best + tolerance * max(abs(best), 1e-12)
    # The joint scalar must never hide a regressing supervised modality.
    for component, weight in (('image_loss', 'lambda_image'), ('tcp_loss', 'lambda_tcp')):
        if metric[weight] > 0:
            key = 'best_' + component
            previous_best = result[key]
            result[key] = metric[component] if previous_best is None else min(previous_best, metric[component])
            degraded |= metric[component] > result[key] + tolerance * max(abs(result[key]), 1e-12)
    if degraded:
        result.update(streak=0, reference_loss=None)
        return result
    reference = result['reference_loss']
    if reference is None or loss <= reference - tolerance * max(abs(reference), 1e-12):
        result.update(streak=0, reference_loss=loss)
    else:
        result['streak'] += 1
    result['converged'] = update >= policy['min_updates'] and result['streak'] >= policy['patience']
    return result


def validate_state(state, policy, evaluation_identity, checkpoint_update, max_updates):
    """Replay canonical evidence; mutable summaries are never resume authority."""
    try:
        replay = new_state(convergence_config(policy, max_updates), evaluation_identity)
        if type(checkpoint_update) is not int or not 0 <= checkpoint_update <= max_updates:
            raise ValueError('checkpoint update outside budget')
        for row in state['history']:
            if row['update'] > checkpoint_update:
                raise ValueError('evidence after checkpoint')
            replay = observe(replay, row['update'], row['metric'], max_updates=max_updates)
        expected = (checkpoint_update if checkpoint_update == max_updates
                    else checkpoint_update // policy['eval_every'] * policy['eval_every'])
        if not replay['history'] or replay != state:
            raise ValueError('derived fields differ')
        if replay['history'][-1]['update'] != expected:
            raise ValueError('missing scheduled evaluation')
        if replay['converged'] and replay['history'][-1]['update'] != checkpoint_update:
            raise ValueError('accepted evidence does not match checkpoint')
        return replay
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('invalid convergence state: ' + str(exc)) from exc


@contextmanager
def preserve_rng():
    """Isolate even lazy batch preparation and exception paths from training RNG."""
    import numpy as np
    import torch
    py, numpy = random.getstate(), np.random.get_state()
    # One local model device per rank; touching all visible devices creates
    # unwanted CUDA contexts on peers' GPUs during rank-zero evaluation.
    devices = [torch.cuda.current_device()] if torch.cuda.is_initialized() else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(py)
        np.random.set_state(numpy)


def evaluation_identity(pairs, *, seed, expert_contract):
    """Hash ordered actual model inputs, masks and stable case metadata, streaming."""
    import torch
    digest = hashlib.sha256()
    def add(value):
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            add(dict(tensor_shape=list(tensor.shape), dtype=str(tensor.dtype)))
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        elif is_dataclass(value):
            add({f.name: getattr(value, f.name) for f in fields(value)})
        elif isinstance(value, dict):
            digest.update(b'{')
            for key in sorted(value):
                add(key); add(value[key])
            digest.update(b'}')
        elif isinstance(value, (tuple, list)):
            digest.update(b'[')
            for item in value: add(item)
            digest.update(b']')
        else:
            digest.update(json.dumps(value, sort_keys=True, allow_nan=False).encode() + b'\0')
    add(dict(version=DIAGNOSTIC_VERSION, seed=seed, expert=expert_contract))
    keys = {'observed', 'context', 'context_lens', 'targets', 'observed_view_valid',
            'history_images', 'history_latents', 'history_ages', 'history_valid', 'history_episode_id'}
    for record, batch in pairs:
        add({k: record.get(k) for k in ('id', 'instruction', 'family', 'granularity')})
        add({k: v for k, v in batch.items() if k in keys})
    return digest.hexdigest()


def flow_contract(expert):
    decoder = getattr(expert, 'decoder', expert)
    return dict(lambda_image=expert.lambda_image, lambda_tcp=expert.lambda_tcp,
                sigmas=decoder.scheduler.sigmas.tolist(),
                weights=decoder.scheduler.linear_timesteps_weights.tolist(),
                latent_shape=list(expert.codec.spec.latent_shape))


def fixed_flow_metric(expert, batches, *, seed):
    """One fixed noise/timestep draw per case, aggregated by actual valid counts."""
    import torch
    from openwam.model.goal.history import history_kwargs
    from openwam.model.goal.patch_expert import TCP_PATCH_WIDTH
    decoder = getattr(expert, 'decoder', expert)
    eligible = (decoder.scheduler.linear_timesteps_weights.detach().cpu() > 0).nonzero().flatten()
    if not len(eligible):
        raise ValueError('fixed flow diagnostic has no positive-weight timesteps')
    modes = [(module, module.training) for module in expert.modules()]
    image_sum = tcp_sum = 0.
    images = points = 0
    try:
        with preserve_rng(), torch.no_grad():
            expert.eval()
            for index, batch in enumerate(batches):
                observed = batch['observed']; size = len(observed)
                generator = torch.Generator(device='cpu').manual_seed(seed + index)
                indices = eligible[torch.randint(len(eligible), (size,), generator=generator)]
                noise = torch.randn((size, *expert.codec.spec.latent_shape), generator=generator)
                tcp_noise = torch.randn((size, TCP_PATCH_WIDTH), generator=generator)
                out = expert(observed, batch['context'], batch['targets'],
                    timestep_indices=indices.to(observed.device), noise=noise.to(observed),
                    tcp_noise=tcp_noise.to(observed.device), context_lens=batch.get('context_lens'),
                    observed_view_valid=batch.get('observed_view_valid'), **history_kwargs(batch))
                n_image = int(batch['targets'].image_valid.sum().item())
                n_point = int(out['valid_points'])
                image_sum += float(out['fm_loss']) * n_image
                tcp_sum += float(out['tcp_loss']) * n_point
                images += n_image; points += n_point
                del out
    finally:
        for module, mode in modes:
            module.training = mode
    image_loss = image_sum / images if images else 0.
    tcp_loss = tcp_sum / points if points else 0.
    return dict(loss=expert.lambda_image * image_loss + expert.lambda_tcp * tcp_loss,
                image_loss=image_loss, tcp_loss=tcp_loss, valid_images=images, valid_points=points,
                lambda_image=expert.lambda_image, lambda_tcp=expert.lambda_tcp)
