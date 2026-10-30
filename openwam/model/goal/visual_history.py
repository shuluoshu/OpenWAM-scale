"""History-only pooling and visual fusion, followed by native Wan conditioning.

No text or future target enters the history pool or visual adapter. Current
spatial tokens retain their native positions. Native text CA remains active in
the main decoder; the inherited mask prevents noisy targets updating current.
"""
import copy
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .history import HeadFeatureCache, validate_history
from .observation import observation_token_mask, validate_view_valid
from .patch_expert import PatchGoalExpert
from openwam.model.video_backbone.wan.models.dit import sinusoidal_embedding_1d


class HistoryTokenPool(nn.Module):
    """A fixed total token budget over all valid past frames, with real ages."""

    def __init__(self, width, num_tokens=32, hidden_dim=256, num_heads=8):
        super().__init__()
        if (any(type(x) is not int or x < 1 for x in (num_tokens, hidden_dim, num_heads))
                or hidden_dim % num_heads):
            raise ValueError('invalid history pool dimensions')
        self.input = nn.Linear(width, hidden_dim)
        self.position = nn.Parameter(torch.randn(1, 1, 80, hidden_dim) * .02)
        self.queries = nn.Parameter(torch.randn(1, num_tokens, hidden_dim) * .02)
        self.time = nn.Sequential(nn.Linear(33, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=0.)
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, width))

    def forward(self, patches, ages, valid):
        b, frames, spatial, _ = patches.shape
        if spatial != 80 or ages.shape != (b,frames) or valid.shape != (b,frames):
            raise ValueError('history pool expects [B,H,80,D] with ages and validity')
        patches = torch.where(valid[:, :, None, None], patches, 0.)
        age = torch.where(valid, ages, 0.).float()
        frequencies = torch.exp(torch.linspace(math.log(.1), math.log(10.), 16, device=age.device))
        phase = age[..., None] * frequencies
        time = torch.cat((torch.log1p(age)[..., None], phase.sin(), phase.cos()), dim=-1)
        memory = self.input(patches) + self.position + self.time(time.to(patches.dtype))[:, :, None]
        memory = self.norm(memory).flatten(1,2)
        key_valid = valid.repeat_interleave(spatial, dim=1).clone()
        # Empty samples get one safe dummy key and are explicitly zeroed below.
        empty = ~valid.any(1)
        key_valid[empty, 0] = True
        memory = torch.where(valid.any(1)[:, None, None], memory, 0.)
        q = self.queries.expand(b,-1,-1)
        pooled = q + self.attention(q, memory, memory, key_padding_mask=~key_valid, need_weights=False)[0]
        out = self.output(pooled)
        return torch.where(empty[:, None, None], 0., out)


class VisualHistoryGoalExpert(PatchGoalExpert):
    representation = 'visual_history_masked_sa'

    def __init__(self, dit, codec, *, dit_config, num_history_tokens=32,
                 pool_dim=256, pool_heads=8, adapter_block=0, **kwargs):
        if kwargs.pop('observation_mode', 'multiview') != 'multiview':
            raise ValueError('visual history expert requires multiview observation')
        super().__init__(dit, codec, dit_config=dit_config, observation_mode='multiview', **kwargs)
        self.history_options = dict(num_history_tokens=num_history_tokens, pool_dim=pool_dim,
                                    pool_heads=pool_heads, adapter_block=adapter_block)
        self.history_pool = HistoryTokenPool(dit.dim, num_history_tokens, pool_dim, pool_heads)
        self.visual_adapter = copy.deepcopy(dit.blocks[adapter_block])
        # This copy is visual-only; keep native text modules in the main Wan.
        del self.visual_adapter.cross_attn
        del self.visual_adapter.norm3
        self.history_gate = nn.Parameter(torch.zeros(()))
        self._history_cache = HeadFeatureCache(256)

    def clear_history_cache(self):
        self._history_cache.clear()

    def _apply(self, fn, recurse=True):
        self.clear_history_cache()
        return super()._apply(fn, recurse=recurse)

    def condition_tokens(self, observed, *, observed_view_valid, history_images=None,
                         history_latents=None, history_ages=None, history_valid=None,
                         history_episode_id=None):
        latent = self.encode_observation(observed, observed_view_valid=observed_view_valid)
        current_volume = self.dit.patchify(latent)
        current = current_volume.flatten(2).transpose(1,2)
        b, m, width = current.shape
        supplied = (history_images, history_latents, history_ages, history_valid, history_episode_id)
        if all(x is None for x in supplied):
            return current
        validate_history(history_ages, history_valid, batch_size=b)
        if (history_images is None) == (history_latents is None):
            raise ValueError('history requires exactly one of images or latents')
        if history_images is not None:
            history_latents = self._history_cache.encode(self.codec, history_images, history_valid,
                episode_ids=history_episode_id)
        else:
            self.clear_history_cache()
        if history_latents.shape != (b,10,48,1,16,20):
            raise ValueError('history latents must be [B,10,48,1,16,20]')
        safe = torch.where(history_valid[:, :, None, None, None, None], history_latents.detach(), 0.)
        if not torch.isfinite(safe).all():
            raise ValueError('valid history latents must be finite')
        patches = self.dit.patchify(safe.flatten(0,1).to(latent)).flatten(2).transpose(1,2)
        pooled = self.history_pool(patches.reshape(b,10,80,width), history_ages, history_valid)
        n = pooled.shape[1]
        x = torch.cat((pooled.to(current), current), dim=1)
        rows, cols = current_volume.shape[-2:]
        freqs = self.dit.freqs
        current_freq = torch.cat((freqs[0][0,None,None,:].expand(rows,cols,-1),
            freqs[1][:rows,None,:].expand(rows,cols,-1),
            freqs[2][None,:cols,:].expand(rows,cols,-1)), dim=-1).reshape(m,1,-1).to(x.device)
        # Abstract slots are not a fictitious 2-D image grid. Identity RoPE;
        # their learned query vectors carry slot identity and pooled age content.
        history_freq = torch.ones(n,1,current_freq.shape[-1],device=x.device,dtype=current_freq.dtype)
        position = torch.cat((history_freq,current_freq),dim=0)
        views = validate_view_valid(observed_view_valid,batch_size=b,device=x.device)
        keys = torch.cat((history_valid.any(1)[:,None].expand(-1,n),observation_token_mask(views)),dim=1)
        mask = keys[:,None,None,:].expand(-1,1,n+m,-1)
        times = torch.zeros(b,n+m,device=x.device,dtype=x.dtype)
        time = self.dit.time_embedding(sinusoidal_embedding_1d(self.dit.freq_dim,times.flatten()).to(x.dtype))
        modulation = self.dit.time_projection(time).reshape(b,n+m,6,width)
        if self.training and self.use_gradient_checkpointing:
            adapted = checkpoint(self.visual_adapter,x,None,modulation,position,None,mask,use_reentrant=False)
        else:
            adapted = self.visual_adapter(x,None,modulation,position,self_attn_mask=mask)
        delta = adapted[:,n:] - current
        fused = current + self.history_gate.tanh().to(current.dtype) * delta
        return torch.where(history_valid.any(1)[:,None,None],fused,current)

    def forward(self, observed, context, targets, *, observed_view_valid=None,
                history_images=None, history_latents=None, history_ages=None,
                history_valid=None, history_episode_id=None, **kwargs):
        tokens = self.condition_tokens(observed, observed_view_valid=observed_view_valid,
            history_images=history_images,history_latents=history_latents,history_ages=history_ages,
            history_valid=history_valid,history_episode_id=history_episode_id)
        result = super().forward(observed,context,targets,observed_view_valid=observed_view_valid,
                                 observation_tokens=tokens,**kwargs)
        result['condition_tokens'] = tokens
        return result

    @torch.no_grad()
    def generate(self, observed, context, *, observed_view_valid=None,
                 history_images=None, history_latents=None, history_ages=None,
                 history_valid=None, history_episode_id=None, **kwargs):
        tokens = self.condition_tokens(observed,observed_view_valid=observed_view_valid,
            history_images=history_images,history_latents=history_latents,history_ages=history_ages,
            history_valid=history_valid,history_episode_id=history_episode_id)
        return super().generate(observed,context,observed_view_valid=observed_view_valid,
                                observation_tokens=tokens,**kwargs)
