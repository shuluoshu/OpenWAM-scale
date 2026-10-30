"""Wan query-condition interfaces; prefix mode borrows PatchWAM's attention mask.

This is a Wan ablation, not FLUX PatchWAM: the query encoder and fixed TCP patch
codec stay identical. Prefix mode replaces blank observation slots with Z and
bypasses Wan cross-attention; goal and TCP still share patch input/output weights.
"""
import torch
from torch.utils.checkpoint import checkpoint
from .patch_expert import PatchGoalExpert, TCP_PATCH_WIDTH
from openwam.model.video_backbone.wan.models.dit import sinusoidal_embedding_1d

CONDITIONING_MODES = ('legacy_text', 'normalized_cross_attention', 'prefix_self_attention')


class QueryAuxiliaryDecoder(PatchGoalExpert):
    def __init__(self, *args, conditioning_mode='legacy_text', **kwargs):
        if conditioning_mode not in CONDITIONING_MODES:
            raise ValueError(f'unknown query conditioning mode: {conditioning_mode}')
        super().__init__(*args, **kwargs)
        self.conditioning_mode = conditioning_mode

    def encode_observation(self, observed, *, observed_view_valid=None):
        return observed.new_zeros((len(observed), *self.codec.spec.latent_shape))

    @property
    def context_width(self):
        return super().context_width if self.conditioning_mode == 'legacy_text' else self.dit.dim

    def embed_context(self, context):
        return super().embed_context(context) if self.conditioning_mode == 'legacy_text' else context

    def velocity(self, observed_latent, noisy_goal, noisy_tcp, timestep, context, *,
                 context_lens=None, return_observation=False, observed_view_valid=None):
        if self.conditioning_mode != 'prefix_self_attention':
            return super().velocity(observed_latent, noisy_goal, noisy_tcp, timestep, context,
                context_lens=context_lens, return_observation=return_observation,
                observed_view_valid=observed_view_valid)
        if return_observation or observed_view_valid is not None:
            raise ValueError('query prefix has no observation reconstruction or view mask')
        d = self.dit
        self.codec.validate_latent(noisy_goal)
        b = noisy_goal.shape[0]
        if context.ndim != 3 or context.shape[0] != b or context.shape[-1] != d.dim or context.shape[1] < 1:
            raise ValueError('query prefix must be [B,L,Wan_dim]')
        if not torch.isfinite(context).all():
            raise ValueError('query prefix must be finite')
        if noisy_tcp.shape != (b, TCP_PATCH_WIDTH) or not torch.isfinite(noisy_tcp).all():
            raise ValueError('noisy TCP patch must be finite [B,192]')
        if timestep.shape != (b,):
            raise ValueError('patch timestep must be [B]')
        n_prefix = context.shape[1]
        if n_prefix > len(d.freqs[2]):
            raise ValueError('query prefix exceeds rotary position capacity')
        goal = d.patchify(noisy_goal)
        h, w = goal.shape[-2:]; n_goal = h * w
        tcp_volume = d.unpatchify(noisy_tcp[:, None].to(noisy_goal), (1, 1, 1))
        tcp = d.patchify(tcp_volume).flatten(2).transpose(1, 2)
        tokens = torch.cat((context, goal.flatten(2).transpose(1, 2), tcp), dim=1)
        length = n_prefix + n_goal + 1
        token_t = torch.cat((timestep.new_zeros(b, n_prefix), timestep[:, None].expand(-1, n_goal+1)), dim=1)
        time = d.time_embedding(sinusoidal_embedding_1d(d.freq_dim, token_t.flatten()).to(tokens.dtype)).reshape(b,length,d.dim)
        modulation = d.time_projection(time).unflatten(2, (6,d.dim))
        prefix_freq = torch.cat((d.freqs[0][0].expand(n_prefix,-1),
            d.freqs[1][0].expand(n_prefix,-1), d.freqs[2][:n_prefix]), dim=-1)[:,None]
        goal_freq = torch.cat((d.freqs[0][1].expand(h,w,-1),
            d.freqs[1][:h,None].expand(h,w,-1),d.freqs[2][None,:w].expand(h,w,-1)),dim=-1).reshape(n_goal,1,-1)
        tcp_freq = torch.cat((d.freqs[0][2], d.freqs[1][0], d.freqs[2][0]),dim=-1)[None,None]
        freqs = torch.cat((prefix_freq,goal_freq,tcp_freq)).to(tokens.device)
        mask = torch.ones(length,length,device=tokens.device,dtype=torch.bool)
        mask[:n_prefix,n_prefix:] = False
        if context_lens is not None:
            lens = torch.as_tensor(context_lens, device=tokens.device)
            if lens.shape != (b,) or (lens <= 0).any() or (lens > n_prefix).any():
                raise ValueError('invalid query prefix lengths')
            valid = torch.cat((torch.arange(n_prefix,device=tokens.device)[None] < lens[:,None],
                               torch.ones(b,n_goal+1,device=tokens.device,dtype=torch.bool)),dim=1)
            mask = mask[None,None] & valid[:,None,None,:]
        for block in d.blocks:
            if self.training and self.use_gradient_checkpointing:
                tokens = checkpoint(block,tokens,None,modulation,freqs,None,mask,use_reentrant=False)
            else:
                tokens = block(tokens,None,modulation,freqs,self_attn_mask=mask)
        output = d.head(tokens[:,n_prefix:],time[:,n_prefix:])
        return d.unpatchify(output[:,:n_goal],(1,h,w)), output[:,-1].float()
