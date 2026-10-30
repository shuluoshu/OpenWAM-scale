"""Input-dependent learned queries with training-only goal reconstruction heads.

The encoder never accepts a target, noisy future token, diffusion time or seed.
Stage2 owns QueryGoalEncoder only; QueryGoalExpert adds removable Stage1 heads.
"""
from dataclasses import asdict, dataclass

from einops import rearrange
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .codec import GoalCodec, GoalCodecSpec
from .observation import (MULTIVIEW_IMAGE_SHAPE, MULTIVIEW_LATENT_SHAPE,
                          observation_token_mask, sanitize_canvas)
from .query_conditioning import QueryAuxiliaryDecoder, CONDITIONING_MODES
from .history import HistoryConfig, HistoryCompressor, HeadFeatureCache, validate_history, history_episode_ids

QUERY_FORMAT = 'openwam.goal.query.v1'
DEFAULT_QUERY = dict(num_image_tokens=64, num_tcp_tokens=2, dim=512,
                     num_heads=8, num_layers=4, ffn_dim=2048)


@dataclass(frozen=True)
class QueryConfig:
    num_image_tokens: int = 64
    num_tcp_tokens: int = 2
    dim: int = 512
    num_heads: int = 8
    num_layers: int = 4
    ffn_dim: int = 2048

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in asdict(self).values()):
            raise ValueError('query dimensions/counts must be positive integers')
        if self.dim % self.num_heads:
            raise ValueError('query dim must be divisible by num_heads')

    @property
    def num_tokens(self):
        return self.num_image_tokens + self.num_tcp_tokens


class GoalQueryTransformer(nn.Module):
    def __init__(self, config, *, text_dim, observation_tokens, history_config=None):
        super().__init__()
        self.config = config
        d = config.dim
        self.image_queries = nn.Parameter(torch.randn(1, config.num_image_tokens, d) * .02)
        self.tcp_queries = nn.Parameter(torch.randn(1, config.num_tcp_tokens, d) * .02)
        self.observation_position = nn.Parameter(torch.randn(1, observation_tokens, d) * .02)
        self.modality = nn.Parameter(torch.randn(2, d) * .02)
        self.observation_projection = nn.Linear(192, d)
        self.text_projection = nn.Linear(text_dim, d)
        self.blocks = nn.ModuleList(nn.TransformerDecoderLayer(
            d, config.num_heads, config.ffn_dim, dropout=0., activation='gelu',
            batch_first=True, norm_first=True) for _ in range(config.num_layers))
        self.norm = nn.LayerNorm(d)
        self.use_gradient_checkpointing = False
        if history_config is not None and history_config.enabled:
            self.history = HistoryCompressor(history_config, d)

    def forward(self, observation, context, *, observation_valid, context_lens=None,
                history_latents=None, history_ages=None, history_valid=None):
        patches = rearrange(observation, 'b c 1 (h p) (w q) -> b (h w) (p q c)', p=2, q=2)
        b = patches.shape[0]
        if patches.shape[1] != self.observation_position.shape[1]:
            raise ValueError('query observation patch shape mismatch')
        if (context.ndim != 3 or context.shape[0] != b or
                context.shape[2] != self.text_projection.in_features):
            raise ValueError('query context must be [B,L,text_dim]')
        lens = torch.full((b,), context.shape[1], device=context.device, dtype=torch.long)
        if context_lens is not None:
            lens = torch.as_tensor(context_lens, device=context.device)
        if lens.shape != (b,) or (lens <= 0).any() or (lens > context.shape[1]).any():
            raise ValueError('invalid query context_lens')
        text_valid = torch.arange(context.shape[1], device=context.device)[None] < lens[:, None]
        # Mask before projections: NaN padding cannot contaminate attention.
        context = torch.where(text_valid[..., None], context, 0.)
        patches = torch.where(observation_valid[..., None], patches, 0.)
        if not torch.isfinite(context).all() or not torch.isfinite(patches).all():
            raise ValueError('valid query inputs must be finite')
        memory = torch.cat((self.observation_projection(patches) + self.observation_position + self.modality[0],
                            self.text_projection(context) + self.modality[1]), dim=1)
        padding = ~torch.cat((observation_valid, text_valid), dim=1)
        if hasattr(self, 'history'):
            ht, hv = self.history(history_latents, history_ages, history_valid)
            memory = torch.cat((memory, ht.to(memory.dtype)), dim=1)
            padding = torch.cat((padding, ~hv), dim=1)
        queries = torch.cat((self.image_queries, self.tcp_queries), dim=1).expand(b, -1, -1)
        for block in self.blocks:
            if self.training and self.use_gradient_checkpointing:
                queries = checkpoint(block, queries, memory, memory_key_padding_mask=padding,
                                     use_reentrant=False)
            else:
                queries = block(queries, memory, memory_key_padding_mask=padding)
        return self.norm(queries)


class QueryGoalEncoder(nn.Module):
    """Deployable standalone encoder; deliberately has no auxiliary decoder."""
    representation = 'query_latent'
    shift = 1.  # Deterministic runtime cache contract; no diffusion sampling.

    def __init__(self, codec, *, text_dim, query_config=None, observation_mode='multiview',
                 history_config=None):
        super().__init__()
        if observation_mode not in ('head', 'multiview'):
            raise ValueError('invalid query observation_mode')
        self.codec = codec
        self.text_dim = int(text_dim)
        self.query_config = QueryConfig(**(query_config or {}))
        self.observation_mode = observation_mode
        self.history_config = HistoryConfig(**(history_config or {}))
        self.encoder = GoalQueryTransformer(self.query_config, text_dim=text_dim,
            observation_tokens=120 if observation_mode == 'multiview' else 80,
            history_config=self.history_config)
        self._history_cache = HeadFeatureCache(self.history_config.cache_size)

    def clear_history_cache(self):
        self._history_cache.clear()

    def _apply(self, fn, recurse=True):
        self.clear_history_cache()
        return super()._apply(fn, recurse=recurse)

    @property
    def observation_codec(self):
        if self.observation_mode == 'head':
            return self.codec
        return GoalCodec(self.codec.vae, GoalCodecSpec(
            name='wan22_vae48_multiview_single_frame', image_shape=MULTIVIEW_IMAGE_SHAPE,
            latent_shape=MULTIVIEW_LATENT_SHAPE))

    def extract_tokens(self, observed, context, *, context_lens=None, observed_view_valid=None,
                       history_images=None, history_latents=None, history_ages=None, history_valid=None, history_episode_id=None):
        history_args = {}
        provided = any(v is not None for v in (history_images,history_latents,history_ages,history_valid,history_episode_id))
        if not self.history_config.enabled and provided:
            raise ValueError('history inputs supplied to a history-disabled encoder')
        if self.history_config.enabled:
            validate_history(history_ages,history_valid,batch_size=len(observed))
            if (history_images is None) == (history_latents is None):
                raise ValueError('history requires exactly one of images or cached latents')
            if history_latents is None:
                history_latents=self._history_cache.encode(self.codec,history_images,history_valid,episode_ids=history_episode_id)
            else:
                for episode_id in history_episode_ids(history_episode_id,len(observed)):
                    self._history_cache.begin_episode(episode_id)
            if history_images is None and not bool(history_valid.all()):
                # Switching to precomputed features at an episode start must
                # also discard any pixel-feature cache left by earlier inputs.
                self.clear_history_cache()
            history_args=dict(history_latents=history_latents,history_ages=history_ages,history_valid=history_valid)
        if self.observation_mode == 'multiview':
            observed = sanitize_canvas(observed, observed_view_valid)
            valid = observation_token_mask(observed_view_valid).to(observed.device)
        else:
            if observed_view_valid is not None:
                raise ValueError('head query encoder does not accept a view mask')
            valid = torch.ones(len(observed), 80, device=observed.device, dtype=torch.bool)
        observation = self.observation_codec.encode(observed)
        return self.encoder(observation, context, observation_valid=valid, context_lens=context_lens,**history_args)

    def train(self, mode=True):
        return super().train(False if getattr(self, '_frozen', False) else mode)

    def freeze(self):
        self._frozen = True
        self.requires_grad_(False)
        self.codec.vae.eval().requires_grad_(False)
        return self.eval()


class QueryGoalExpert(QueryGoalEncoder):
    """Stage1 encoder + joint-flow decoder. Decoder conditions only on Z."""
    def __init__(self, dit, codec, *, dit_config, query_config=None,
                 observation_mode='multiview', shift=5., lambda_image=.5,
                 lambda_tcp=1.,
                 use_gradient_checkpointing=False, history_config=None, conditioning_mode='legacy_text'):
        if conditioning_mode not in CONDITIONING_MODES:
            raise ValueError(f'unknown query conditioning mode: {conditioning_mode}')
        self.conditioning_mode = conditioning_mode
        super().__init__(codec, text_dim=dit.text_embedding[0].in_features,
                         query_config=query_config, observation_mode=observation_mode, history_config=history_config)
        self.decoder = QueryAuxiliaryDecoder(dit, codec, dit_config=dit_config, shift=shift,
            lambda_image=lambda_image, lambda_tcp=lambda_tcp,
            use_gradient_checkpointing=use_gradient_checkpointing, observation_mode='head',
            conditioning_mode=conditioning_mode)
        self.encoder.use_gradient_checkpointing = use_gradient_checkpointing
        self.condition_projection = (nn.Linear(self.query_config.dim, self.text_dim)
            if conditioning_mode == 'legacy_text' else
            nn.Sequential(nn.LayerNorm(self.query_config.dim), nn.Linear(self.query_config.dim, dit.dim)))
        self.dit_config = dict(dit_config)
        self.shift, self.lambda_image, self.lambda_tcp = shift, lambda_image, lambda_tcp

    @property
    def dit(self):
        return self.decoder.dit

    def _condition(self, observed, context, **kwargs):
        z = self.extract_tokens(observed, context, **kwargs)
        # Constant zero prefix for reuse of the legacy joint flow decoder.
        # No original observation/text reaches this branch except through Z.
        blank = observed.new_zeros((len(observed), 3, 1, 256, 320))
        return z, blank, self.condition_projection(z)

    def forward(self, observed, context, targets, *, context_lens=None,
                observed_view_valid=None, history_images=None, history_latents=None,
                history_ages=None, history_valid=None, history_episode_id=None, **flow_kwargs):
        z, blank, condition = self._condition(observed, context, context_lens=context_lens,
            observed_view_valid=observed_view_valid,history_images=history_images,
            history_latents=history_latents,history_ages=history_ages,history_valid=history_valid,history_episode_id=history_episode_id)
        result = self.decoder(blank, condition, targets, **flow_kwargs)
        result['query_tokens'] = z
        return result

    @torch.no_grad()
    def generate(self, observed, context, *, context_lens=None, observed_view_valid=None,
                 num_steps=20, seed=0, history_images=None, history_latents=None,
                 history_ages=None, history_valid=None, history_episode_id=None):
        if self.training:
            raise RuntimeError('query goal diagnostics require eval()')
        _, blank, condition = self._condition(observed, context, context_lens=context_lens,
            observed_view_valid=observed_view_valid,history_images=history_images,
            history_latents=history_latents,history_ages=history_ages,history_valid=history_valid,history_episode_id=history_episode_id)
        return self.decoder.generate(blank, condition, num_steps=num_steps, seed=seed)

    def train_stage_a(self):
        self._frozen = False
        self.requires_grad_(True)
        self.decoder.train_stage_a()
        if self.conditioning_mode != 'legacy_text':
            self.dit.text_embedding.requires_grad_(False)
        if self.conditioning_mode == 'prefix_self_attention':
            for block in self.dit.blocks:
                block.cross_attn.requires_grad_(False)
                block.norm3.requires_grad_(False)
        self.codec.vae.eval().requires_grad_(False)
        return self.train()
