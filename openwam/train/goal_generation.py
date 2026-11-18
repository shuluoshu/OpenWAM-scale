"""G inference depends on current observations, never future GT availability."""
import torch
from openwam.model.goal.goal_memory_masks import view_token_mask, group_attention_mask

GENERATION_POLICY = 'current-view-kf1-fixed0-no-gt-v1'


def goal_condition_inputs(batch):
    fields = ('current_latents', 'memory_latents', 'text', 'text_mask', 'memory_valid',
              'observed_view_valid', 'memory_view_valid', 'memory_ages', 'memory_frame_ids')
    inputs = {name: batch[name] for name in fields if name in batch}
    observed = batch['observed_view_valid']
    inputs.update(target_view_valid=observed, target_image_valid=observed.any(-1),
                  goal_horizon_s=torch.zeros(len(observed), device=observed.device))
    return inputs


@torch.inference_mode()
def generate_goal_latents(model, batch, *, common_noise, seed, index, steps=20, arm='default'):
    if type(steps) is not int or steps < 1:
        raise ValueError('positive integer generation steps required')
    if model.config.goal_horizon_mode != 'kf1_endpoint':
        raise ValueError('this generation policy requires the KF1 fixed0 model')
    if arm not in ('default', 'no_history', 'no_g'):
        raise ValueError('unknown ablation')
    inputs = goal_condition_inputs(batch)
    current = inputs['current_latents']
    template = current.new_empty((len(current), *model.config.latent_shape))
    _, noise = common_noise(template, seed=seed, cursor=index + 1000000, validation=True)
    h, w = template.shape[-2:]
    tokens = view_token_mask(inputs['target_view_valid'], h // 2, w // 2)
    pixels = tokens.reshape(len(current), h // 2, w // 2)
    pixels = pixels.repeat_interleave(2, -2).repeat_interleave(2, -1)[:, None, None]
    latent = torch.where(pixels, noise, 0.)
    for step in range(steps):
        timestep = torch.full((len(current),), (1. - step / steps) * 1000, device=current.device)
        state = model.prepare_state(**inputs, noisy_target_latents=latent,
                                    timestep=timestep, warmup_y_text=False)
        if arm == 'no_g': state.key_valid[:, state.goal_slice] = False
        if arm == 'no_history': state.key_valid[:, :state.current_slice.start] = False
        if arm != 'default':
            state.attention_mask = group_attention_mask(state.memory_slices, state.current_slice,
                state.goal_slice, state.target_slice, state.key_valid)
        outputs = model.outputs(model.run_state(state))
        latent = torch.where(pixels, latent - outputs['velocity'] / steps, 0.)
    return dict(latents=latent, uv=outputs['uv'], steps=steps,
                noise_seed_index=index + 1000000, generation_policy=GENERATION_POLICY)
