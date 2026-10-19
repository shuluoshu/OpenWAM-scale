"""Stage A joint terminal-image / ordered TCP flow matching in Wan patch space.

TCP uses a fixed repeat/group-mean codec in the native 192-dimensional Wan
head order (patch frame, patch row, patch column, channel). Unlike the released
Awomo-0.5I low-dimensional action-noise path, this follows the paper's independent
Gaussian noise in every encoded patch dimension. No TCP-specific learned layer
or heatmap head is present.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from openwam.model.video_backbone.wan.models.dit import sinusoidal_embedding_1d
from openwam.model.video_backbone.wan.shared.diffusion.flow_match import FlowMatchScheduler


TCP_PATCH_WIDTH = 192
TCP_COORD_COPIES = 48


def encode_tcp_patch(uv):
    """[B,2,2] or [B,4] normalized ordered UV -> [B,192], without clamping."""
    if uv.ndim == 3 and tuple(uv.shape[1:]) == (2, 2):
        uv = uv.flatten(1)
    if uv.ndim != 2 or uv.shape[1] != 4 or not uv.is_floating_point():
        raise ValueError('TCP UV must be floating [B,2,2] or [B,4]')
    return (2 * uv.float() - 1).repeat_interleave(TCP_COORD_COPIES, dim=-1)


def decode_tcp_patch(patch):
    """Group mean -> continuous normalized [B,2,2] UV, without clamping."""
    if patch.ndim != 2 or patch.shape[1] != TCP_PATCH_WIDTH or not patch.is_floating_point():
        raise ValueError('TCP patch must be floating [B,192]')
    coordinates = patch.float().reshape(-1, 4, TCP_COORD_COPIES).mean(-1)
    return ((coordinates + 1) / 2).reshape(-1, 2, 2)


@dataclass
class PatchGoalPrediction:
    latent: torch.Tensor
    uv: torch.Tensor  # Normalized continuous [B,2,2], ordered left/right, unclamped.
    tcp_patch: torch.Tensor
    token_valid: torch.Tensor  # 80 generated image tokens followed by one TCP token.


class PatchGoalExpert(nn.Module):
    """One Wan transformer jointly predicts image and TCP patch velocities.

    The observation is a clean attention prefix. Rotary positions (frame 0 for
    observation, 1 for goal, 2 for TCP) distinguish the modalities without any
    additional learned embeddings. Goal and TCP tokens attend each other.
    """

    def __init__(self, dit, codec, *, dit_config, shift=5., lambda_image=.5,
                 lambda_tcp=1., use_gradient_checkpointing=False):
        super().__init__()
        if (dit.in_dim != 48 or dit.head.head.out_features != TCP_PATCH_WIDTH
                or tuple(dit.patch_size) != (1, 2, 2) or dit.has_image_input
                or dit.require_clip_embedding or dit.require_vae_embedding
                or not dit.fuse_vae_embedding_in_latents):
            raise ValueError('patch expert requires Wan2.2 48-channel patch(1,2,2) configuration')
        if (not all(math.isfinite(float(v)) for v in (shift, lambda_image, lambda_tcp))
                or shift <= 0 or lambda_image < 0 or lambda_tcp < 0):
            raise ValueError('invalid patch goal shift/loss weights')
        self.dit = dit
        self.codec = codec  # Plain codec borrows a frozen VAE outside the state dict.
        self.dit_config = dict(dit_config)
        self.shift = float(shift)
        self.lambda_image = float(lambda_image)
        self.lambda_tcp = float(lambda_tcp)
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.scheduler = FlowMatchScheduler(template='Wan')
        self.scheduler.set_timesteps(1000, training=True, shift=self.shift)
        # Explicit paper formula; normalize across the actual shifted training grid.
        weights = torch.exp(-2 * (self.scheduler.sigmas - .5).square()) - math.exp(-.5)
        self.scheduler.linear_timesteps_weights = weights / weights.mean()

    def velocity(self, observed_latent, noisy_goal, noisy_tcp, timestep, context, *,
                 context_lens=None, return_observation=False):
        """Return (goal velocity, TCP velocity[, observation velocity]).

        TCP patches use Wan's head flattening order, whereas Conv3d's kernel is
        channel-first. Unpatchifying to a tiny latent volume before the shared
        patchify operation enforces exactly the same order as image patches.
        """
        self.codec.validate_latent(observed_latent)
        self.codec.validate_latent(noisy_goal)
        d = self.dit
        b = observed_latent.shape[0]
        if noisy_goal.shape != observed_latent.shape:
            raise ValueError('goal and observation latent batch mismatch')
        if noisy_tcp.shape != (b, TCP_PATCH_WIDTH) or not torch.isfinite(noisy_tcp).all():
            raise ValueError('noisy TCP patch must be finite [B,192]')
        if timestep.shape != (b,):
            raise ValueError('patch timestep must be [B]')
        if (context.ndim != 3 or context.shape[0] != b
                or context.shape[-1] != d.text_embedding[0].in_features):
            raise ValueError('incompatible patch goal text context [B,L,text_dim]')

        image_patches = d.patchify(torch.cat((observed_latent, noisy_goal), dim=2))
        _, _, frames, h, w = image_patches.shape
        n = h * w
        tcp_volume = d.unpatchify(noisy_tcp[:, None].to(observed_latent), (1, 1, 1))
        tcp_tokens = d.patchify(tcp_volume).flatten(2).transpose(1, 2)
        tokens = torch.cat((image_patches.flatten(2).transpose(1, 2), tcp_tokens), dim=1)
        length = 2 * n + 1

        token_t = torch.cat((torch.zeros(b, n, device=tokens.device, dtype=timestep.dtype),
                             timestep[:, None].expand(-1, n + 1)), dim=1)
        time = d.time_embedding(sinusoidal_embedding_1d(d.freq_dim, token_t.flatten())
                                .to(tokens.dtype)).reshape(b, length, d.dim)
        modulation = d.time_projection(time).unflatten(2, (6, d.dim))
        image_frequencies = torch.cat((
            d.freqs[0][:frames, None, None, :].expand(frames, h, w, -1),
            d.freqs[1][None, :h, None, :].expand(frames, h, w, -1),
            d.freqs[2][None, None, :w, :].expand(frames, h, w, -1)), dim=-1).reshape(2 * n, 1, -1)
        tcp_frequencies = torch.cat((d.freqs[0][2], d.freqs[1][0], d.freqs[2][0]), dim=-1)
        frequencies = torch.cat((image_frequencies, tcp_frequencies[None, None]), dim=0).to(tokens.device)
        attention_mask = torch.ones(length, length, device=tokens.device, dtype=torch.bool)
        attention_mask[:n, n:] = False

        context_mask = None
        if context_lens is not None:
            context_lens = torch.as_tensor(context_lens, device=tokens.device)
            if (context_lens.shape != (b,) or (context_lens <= 0).any()
                    or (context_lens > context.shape[1]).any()):
                raise ValueError('invalid patch context lengths')
            context_mask = (torch.arange(context.shape[1], device=tokens.device)[None]
                            < context_lens[:, None])[:, None, None]
        embedded_context = d.text_embedding(context)
        for block in d.blocks:
            if self.training and self.use_gradient_checkpointing:
                tokens = checkpoint(block, tokens, embedded_context, modulation, frequencies,
                                    context_mask, attention_mask, use_reentrant=False)
            else:
                tokens = block(tokens, embedded_context, modulation, frequencies,
                               context_mask=context_mask, self_attn_mask=attention_mask)
        output = d.head(tokens, time)
        image_output = d.unpatchify(output[:, :2 * n], (frames, h, w))
        result = (image_output[:, :, 1:], output[:, -1].float())
        return (*result, image_output[:, :, :1]) if return_observation else result

    def forward(self, observed, context, targets, *, timestep_indices=None, noise=None,
                tcp_noise=None, context_lens=None):
        observed_latent = self.codec.encode(observed)
        image_valid = targets.image_valid.bool()
        safe_image = torch.where(image_valid[:, None, None, None, None], targets.image,
                                 torch.zeros_like(targets.image))
        clean_image = self.codec.encode(safe_image)
        b = clean_image.shape[0]
        if targets.uv.shape != (b, 2, 2):
            raise ValueError('target TCP UV must be [B,2,2]')
        point_valid = (image_valid[:, None] & targets.point_valid.bool()
                       & targets.in_frame.bool() & torch.isfinite(targets.uv).all(-1))
        # Remove invalid NaNs before normalization, repetition, interpolation or loss.
        safe_uv = torch.where(point_valid[:, :, None], targets.uv, torch.zeros_like(targets.uv))
        clean_tcp = encode_tcp_patch(safe_uv).to(device=clean_image.device)
        copy_valid = point_valid.repeat_interleave(2, dim=-1).repeat_interleave(TCP_COORD_COPIES, dim=-1)
        idx = (torch.randint(0, len(self.scheduler.timesteps), (b,), device=clean_image.device)
               if timestep_indices is None else timestep_indices.to(device=clean_image.device, dtype=torch.long))
        if idx.shape != (b,) or (idx < 0).any() or (idx >= len(self.scheduler.timesteps)).any():
            raise ValueError('invalid patch goal timestep indices')
        sigma = self.scheduler.sigmas.to(clean_image.device)[idx]
        timesteps = self.scheduler.timesteps.to(clean_image.device)[idx].to(clean_image.dtype)
        noise = torch.randn_like(clean_image) if noise is None else noise.to(clean_image)
        tcp_noise = torch.randn_like(clean_tcp) if tcp_noise is None else tcp_noise.to(clean_tcp)
        if noise.shape != clean_image.shape or not torch.isfinite(noise).all():
            raise ValueError('goal noise must match latent shape and be finite')
        if tcp_noise.shape != clean_tcp.shape or not torch.isfinite(tcp_noise).all():
            raise ValueError('TCP noise must be finite [B,192]')
        s_image = sigma[:, None, None, None, None].to(clean_image.dtype)
        s_tcp = sigma[:, None]
        noisy_image = (1 - s_image) * clean_image + s_image * noise
        noisy_tcp = (1 - s_tcp) * clean_tcp + s_tcp * tcp_noise
        image_velocity, tcp_velocity = self.velocity(observed_latent, noisy_image, noisy_tcp,
            timesteps, context, context_lens=context_lens)
        weights = self.scheduler.linear_timesteps_weights.to(clean_image.device)[idx]
        image_error = torch.where(image_valid[:, None, None, None, None],
                                  image_velocity.float() - (noise.float() - clean_image.float()),
                                  torch.zeros_like(image_velocity, dtype=torch.float32))
        image_mse = image_error.square().flatten(1).mean(1)
        fm_loss = (image_mse * weights).sum() / image_valid.sum().clamp_min(1)
        tcp_error = torch.where(copy_valid, tcp_velocity.float() - (tcp_noise - clean_tcp).float(),
                               torch.zeros_like(tcp_velocity, dtype=torch.float32))
        tcp_loss = (tcp_error.square() * weights[:, None]).sum() / copy_valid.sum().clamp_min(1)
        estimated_image = noisy_image - s_image * image_velocity
        estimated_tcp = noisy_tcp - s_tcp * tcp_velocity
        return dict(loss=self.lambda_image * fm_loss + self.lambda_tcp * tcp_loss,
                    fm_loss=fm_loss, tcp_loss=tcp_loss, valid_points=int(point_valid.sum().item()),
                    clean_latent=estimated_image, observed_latent=observed_latent,
                    noisy_latent=noisy_image, velocity=image_velocity, sigma=sigma,
                    noisy_tcp_patch=noisy_tcp, clean_tcp_patch=estimated_tcp,
                    tcp_velocity=tcp_velocity, tcp_noise=tcp_noise, tcp_valid=point_valid,
                    uv=decode_tcp_patch(estimated_tcp))

    @torch.no_grad()
    def generate(self, observed, context, *, num_steps=20, seed=0, context_lens=None):
        """Joint Euler integration from independent image and 192D TCP noise."""
        if self.training:
            raise RuntimeError('patch goal generation requires eval()')
        if not isinstance(num_steps, int) or num_steps < 1:
            raise ValueError('patch goal num_steps must be a positive integer')
        observed_latent = self.codec.encode(observed)
        b = observed_latent.shape[0]
        seeds = [int(seed)] * b if isinstance(seed, int) else list(seed)
        if len(seeds) != b:
            raise ValueError('one patch goal seed per sample required')
        image_noise, tcp_noise = [], []
        for sample_seed in seeds:
            generator = torch.Generator(device=observed_latent.device).manual_seed(int(sample_seed))
            image_noise.append(torch.randn((1, *observed_latent.shape[1:]),
                device=observed_latent.device, dtype=observed_latent.dtype, generator=generator))
            tcp_noise.append(torch.randn((1, TCP_PATCH_WIDTH), device=observed_latent.device,
                dtype=torch.float32, generator=generator))
        image = torch.cat(image_noise, dim=0)
        tcp = torch.cat(tcp_noise, dim=0)
        scheduler = FlowMatchScheduler(template='Wan')
        scheduler.set_timesteps(num_steps, shift=self.shift)
        for timestep in scheduler.timesteps:
            times = timestep.to(device=image.device, dtype=image.dtype).expand(b)
            image_v, tcp_v = self.velocity(observed_latent, image, tcp, times, context,
                                          context_lens=context_lens)
            image = scheduler.step(image_v, timestep, image)
            tcp = scheduler.step(tcp_v, timestep, tcp)
        image_tokens = math.prod(observed_latent.shape[2:]) // math.prod(self.dit.patch_size)
        token_valid = torch.ones(b, image_tokens + 1, device=image.device, dtype=torch.bool)
        return PatchGoalPrediction(image, decode_tcp_patch(tcp), tcp, token_valid)

    def train(self, mode=True):
        return super().train(False if getattr(self, '_frozen', False) else mode)

    def train_stage_a(self):
        """Thaw this separate Stage A expert; the borrowed VAE stays frozen."""
        self._frozen = False
        self.requires_grad_(True)
        if isinstance(self.codec.vae, nn.Module):
            self.codec.vae.eval().requires_grad_(False)
        return self.train()

    def freeze(self):
        self._frozen = True
        self.eval().requires_grad_(False)
        return self
