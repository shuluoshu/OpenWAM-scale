"""Exact, request-scoped first-frame/C* KV cache for Wan action inference.

Adapted from the reference OpenWAM first-frame prefill / cached action loop.
Unlike a velocity cache this never approximates a denoising step. C* remains
read-only, excludes clean-video queries, and is appended AFTER action keys,
matching the native joint driver. Nothing is stored on the model between calls.
"""
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


@dataclass(frozen=True)
class ActionKVCache:
    video_kv: tuple[tuple[Tensor, Tensor], ...]
    context: Optional[Tensor]
    context_mask: Optional[Tensor]
    goal_keys: Optional[Tensor]
    goal_values: Optional[Tensor]
    goal_token_valid: Optional[Tensor]


@torch.no_grad()
def prefill(architecture, inputs) -> ActionKVCache:
    if architecture.training:
        raise ValueError('action KV cache requires eval mode')
    if not architecture.can_cache_action_only(decode_video=False, cfg_scale=1.):
        raise ValueError('action KV cache requires isolated first-frame dependencies')
    vb = architecture.video_backbone
    driver = architecture.mot_driver
    values = dict(inputs)
    goal = values.pop('goal_latent', None)
    valid = values.pop('goal_token_valid', None)
    goal_keys = goal_values = None
    if goal is not None:
        goal_keys, goal_values = architecture.goal_adapter(goal, vb)
        if valid is None or valid.dtype != torch.bool or valid.shape != goal.shape[:2]:
            raise ValueError('action KV cache requires matching C* token validity')
    elif getattr(architecture, 'goal_expert', None) is not None:
        raise ValueError('goal-enabled cache requires prepared predicted goal tensors')
    proprio = values.pop('proprio', None)
    values = architecture._append_proprio_context_token(values, proprio)
    context = values.get('context')
    context_mask = values.get('context_mask')
    if context is not None and context_mask is None and values.get('seq_lens') is not None:
        lengths = values['seq_lens'].to(context.device)
        context_mask = torch.arange(context.shape[1], device=context.device)[None] < lengths[:, None]
    current = values.get('first_frame_latents')
    if current is None or current.ndim != 5 or current.shape[2] != 1:
        raise ValueError('action KV cache requires one clean first_frame_latents frame')
    values.update(latents=current, num_frames=1,
                  timestep=torch.zeros(current.shape[0], device=current.device, dtype=current.dtype),
                  force_per_token_t_mod=True, zero_clean_prefix_t_mod=True)
    state = vb.prepare(use_gradient_checkpointing=False,
                       use_gradient_checkpointing_offload=False, **values)
    count = driver._video_tokens_per_frame(state)
    mask = vb.build_video_to_video_mask(video_seq_len=count,
        video_tokens_per_frame=count, device=current.device)
    if getattr(state, 'prefix_kv_len', 0):
        raise ValueError('Wan first-frame cache does not support extra prefix keys')
    kv = []
    for layer in range(driver.num_layers):
        q, k, v, post = vb.pre_attn_at_layer(layer, state)
        attended = driver._mixed_attention(q, k, v, mask)
        state = vb.post_attn_at_layer(layer, state, attended.contiguous(), post)
        kv.append((k.detach(), v.detach()))
    return ActionKVCache(tuple(kv), context, context_mask, goal_keys, goal_values, valid)


@torch.no_grad()
def action_step(architecture, actions, timestep, cache: ActionKVCache):
    if architecture.training:
        raise ValueError('action KV cache requires eval mode')
    ab, driver = architecture.action_backbone, architecture.mot_driver
    if len(cache.video_kv) != driver.num_layers:
        raise ValueError('action KV cache layer count mismatch')
    state = ab.prepare_state(actions, timestep, context=cache.context,
        context_mask=cache.context_mask, use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False)
    batch, steps = actions.shape[:2]
    video_count = cache.video_kv[0][0].shape[1]
    mask = torch.ones(batch, 1, steps, video_count + steps,
                      device=actions.device, dtype=torch.bool)
    if cache.goal_keys is not None:
        goal_mask = cache.goal_token_valid[:, None, None, :].expand(batch, 1, steps, -1)
        mask = torch.cat((mask, goal_mask), dim=-1)
    for layer, (kv, vv) in enumerate(cache.video_kv):
        qa, ka, va, post = ab.pre_attn_at_layer(layer, state)
        keys, values = [kv, ka], [vv, va]
        if cache.goal_keys is not None:
            keys.append(cache.goal_keys[layer]); values.append(cache.goal_values[layer])
        attended = driver._mixed_attention(qa, torch.cat(keys, dim=1), torch.cat(values, dim=1), mask)
        state = ab.post_attn_at_layer(layer, state, attended.contiguous(), post)
    return ab.extract_prediction(state)
