"""Frozen Stage A patch expert inference for Stage2 predicted conditioning.

Only clean current observation RGB and instruction context enter this runtime. Goal
labels and future frames are deliberately absent from the inference/cache API.
"""
from collections import OrderedDict
import hashlib
import json
import torch
from .patch_expert import PatchGoalExpert, PatchGoalPrediction
from .types import GoalCondition
from .history import HistoryConfig, history_kwargs, move_history, validate_history, history_episode_ids

PREPROCESSING_VERSION = 'openwam.goal.clean_head.rgb320x256.v1'
MULTIVIEW_PREPROCESSING_VERSION = 'openwam.goal.clean_multiview.rgb320x384.v2'


def goal_representation(config):
    representation = config.get('representation', 'patch_denoise')
    if representation not in ('patch_denoise', 'query_latent', 'visual_history_masked_sa'):
        raise ValueError('unsupported goal representation')
    is_query = config.get('format') == 'openwam.goal.query.v1'
    if is_query != (representation == 'query_latent'):
        raise ValueError('goal representation/format mismatch')
    if (config.get('format') == 'openwam.visual_history.v1') != (representation == 'visual_history_masked_sa'):
        raise ValueError('goal representation/format mismatch')
    return representation


def visual_history_config(config):
    """Native current-token memory dimensions, independent of history pool size."""
    options = dict(config.get('visual_history', {}))
    if set(options) - {'dim', 'num_tokens'}:
        raise ValueError('unsupported visual history memory configuration')
    result = {'dim':3072, 'num_tokens':120, **options}
    if (type(result['dim']) is not int or result['dim'] < 1
            or type(result['num_tokens']) is not int or result['num_tokens'] != 120):
        raise ValueError('visual history memory requires 120 positive-width tokens')
    return result


def goal_observation_mode(config):
    """Validate paired policy/version; absent fields retain historical v1 defaults."""
    representation = goal_representation(config)
    pair = (config.get('format', 'openwam.goal.patch.v1'), config.get('input_policy', 'clean_head'))
    contracts = {('openwam.goal.patch.v1', 'clean_head'): 'head',
                 ('openwam.goal.patch.v2', 'clean_multiview'): 'multiview',
                 ('openwam.goal.query.v1', 'clean_head'): 'head',
                 ('openwam.goal.query.v1', 'clean_multiview'): 'multiview',
                 ('openwam.visual_history.v1', 'clean_multiview'): 'multiview'}
    if pair not in contracts:
        raise ValueError('goal format/input_policy contract mismatch')
    mode = contracts[pair]
    if config.get('observation_mode', mode) != mode:
        raise ValueError('goal observation_mode disagrees with format/input_policy contract')
    return mode


def validate_goal_model_config(config, model):
    """Also used by public loaders that do not execute the training preflight."""
    representation = goal_representation(config)
    history = HistoryConfig(**config.get('history', {}))
    if history.enabled and representation not in ('query_latent', 'visual_history_masked_sa'):
        raise ValueError('history requires query_latent or visual_history_masked_sa representation')
    if representation == 'visual_history_masked_sa':
        visual_history_config(config)
        if not history.enabled:
            raise ValueError('visual history requires enabled ten-frame history sampling')
    if HistoryConfig(**model.get('query_history', {})) != history:
        raise ValueError('model.query_history disagrees with goal.history')
    if model.get('goal_representation', representation) != representation:
        raise ValueError('model.goal_representation disagrees with goal representation')
    if representation == 'query_latent' and 'query_goal' in model:
        from .query_expert import QueryConfig
        if QueryConfig(**dict(model['query_goal'])) != QueryConfig(**config.get('query', {})):
            raise ValueError('model.query_goal disagrees with goal.query')


def validate_goal_compatibility(requested, saved):
    """No topology changes when loading an existing Stage2 checkpoint."""
    if HistoryConfig(**requested.get('history', {})) != HistoryConfig(**saved.get('history', {})):
        raise ValueError('Stage2 history differs from saved checkpoint')
    if goal_representation(requested) != goal_representation(saved):
        raise ValueError('Stage2 goal representation differs from saved checkpoint; use original baseline for a new mode')
    if goal_observation_mode(requested) != goal_observation_mode(saved):
        raise ValueError('Stage2 configured goal policy/mode disagrees with saved checkpoint')
    if goal_representation(requested) == 'query_latent':
        from .query_expert import QueryConfig
        if QueryConfig(**requested.get('query', {})) != QueryConfig(**saved.get('query', {})):
            raise ValueError('Stage2 query dimensions differ from saved checkpoint')
    if goal_representation(requested) == 'visual_history_masked_sa':
        if visual_history_config(requested) != visual_history_config(saved):
            raise ValueError('Stage2 visual history dimensions differ from saved checkpoint')


def goal_prediction_policy(architecture):
    """Keep old head resume serialization while binding v2 to its actual policy."""
    config = architecture.goal_config
    mode = goal_observation_mode(config)
    if getattr(getattr(architecture, 'goal_expert', None), 'observation_mode', mode) != mode:
        raise ValueError('goal configured mode disagrees with loaded expert')
    result = dict(num_steps=int(config.get('num_steps',20)),
                  input_policy='clean_multiview' if mode == 'multiview' else 'clean_head', mode='predicted')
    if mode == 'multiview':
        result.update(format='openwam.goal.patch.v2', observation_mode=mode,
                      preprocessing_version=MULTIVIEW_PREPROCESSING_VERSION)
    if goal_representation(config) == 'query_latent':
        from dataclasses import asdict
        from .query_expert import QueryConfig
        result.update(format='openwam.goal.query.v1', representation='query_latent',
                      query=asdict(QueryConfig(**config.get('query', {}))), num_steps=1)
    if goal_representation(config) == 'visual_history_masked_sa':
        result.update(format='openwam.visual_history.v1', representation='visual_history_masked_sa',
                      visual_history=visual_history_config(config), num_steps=1)
    history = HistoryConfig(**config.get('history', {}))
    if history.enabled:
        from dataclasses import asdict
        result['history'] = asdict(history)
    return result



def goal_asset_sha256(path):
    """Hash multi-gigabyte frozen assets without retaining a file-sized buffer."""
    from pathlib import Path
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_stage2_goal(path, *, vae, text_encoder, tokenizer, text_dim, device,
                     dtype, precomputed_text_fingerprint=None) -> torch.nn.Module:
    """Load the frozen conditioner, removing token-expert target decoders."""
    from safetensors import safe_open
    with safe_open(str(path), framework='pt', device='cpu') as file:
        metadata = file.metadata() or {}
        is_query = 'query_goal_bundle' in metadata
        is_visual_history = 'visual_history' in metadata
    if is_visual_history:
        from openwam.train.goal_visual_history import load_visual_history_bundle
        if text_encoder is not None:
            from .dependencies import text_identity
            if getattr(text_encoder, 'dim', None) != text_dim:
                raise ValueError('incompatible text encoder dimensions')
            actual = text_identity(text_encoder, tokenizer)['fingerprint']
            if precomputed_text_fingerprint is not None and precomputed_text_fingerprint != actual:
                raise ValueError('explicit text fingerprint disagrees with live text dependency')
            precomputed_text_fingerprint = actual
        return load_visual_history_bundle(path, vae=vae, text_dim=text_dim,
            precomputed_text_fingerprint=precomputed_text_fingerprint,
            device=device, dtype=dtype, encoder_only=True)
    if is_query:
        from openwam.train.goal_query import load_query_bundle
        return load_query_bundle(path, vae=vae, text_encoder=text_encoder,
            tokenizer=tokenizer, text_dim=text_dim, device=device, dtype=dtype,
            precomputed_text_fingerprint=precomputed_text_fingerprint, encoder_only=True)
    from openwam.train.goal_patch import load_patch_bundle
    return load_patch_bundle(path, vae=vae, text_encoder=text_encoder,
        tokenizer=tokenizer, text_dim=text_dim, device=device, dtype=dtype,
        precomputed_text_fingerprint=precomputed_text_fingerprint).freeze()


def to_goal_condition(pred: PatchGoalPrediction) -> GoalCondition:
    b = pred.latent.shape[0]
    if tuple(pred.latent.shape) != (b,48,1,16,20):
        raise ValueError('goal latent must be [B,48,1,16,20]')
    if pred.token_valid.shape != (b,81) or pred.token_valid.dtype != torch.bool:
        raise ValueError('patch token mask must be bool [B,81]')
    if pred.uv.shape != (b,2,2) or not pred.uv.is_floating_point():
        raise ValueError('goal UV must be floating [B,2,2]')
    return GoalCondition(pred.latent.detach(), pred.token_valid[:,:80].detach(),
                         pred.uv.detach().float())


def _copy_condition(value, device='cpu'):
    return GoalCondition(*(getattr(value,name).detach().to(device=device).clone()
        for name in ('latent','image_token_valid','uv')))


class GoalPredictionCache:
    """LRU bounded by both entry count and owned CPU tensor storage."""
    def __init__(self, maxsize=128, max_bytes=256*1024*1024):
        self.maxsize=max(0,int(maxsize)); self.max_bytes=max(0,int(max_bytes))
        self.entries=OrderedDict(); self.bytes_used=0

    @staticmethod
    def _size(value):
        return sum(t.numel()*t.element_size() for t in
                   (value.latent,value.image_token_valid,value.uv))

    def get(self,key):
        value=self.entries.get(key)
        if value is not None:
            self.entries.move_to_end(key)
            return _copy_condition(value)
        return None

    def put(self,key,value):
        previous=self.entries.pop(key,None)
        if previous is not None: self.bytes_used-=self._size(previous)
        size=self._size(value)
        if not self.maxsize or size>self.max_bytes: return
        self.entries[key]=_copy_condition(value); self.bytes_used+=size
        while len(self.entries)>self.maxsize or self.bytes_used>self.max_bytes:
            _,old=self.entries.popitem(last=False); self.bytes_used-=self._size(old)


def _tensor_hash(tensor):
    value=tensor.detach().cpu().contiguous()
    digest=hashlib.sha256(str((tuple(value.shape),value.dtype)).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def goal_cache_key(*, observed, context, context_len, prompt, seed, num_steps,
                   asset_identity, shift, preprocessing_version=PREPROCESSING_VERSION,
                   observation_mode='head', observed_view_valid=None,
                   history_images=None, history_latents=None, history_ages=None, history_valid=None, history_episode_id=None):
    value=dict(version=preprocessing_version,image=_tensor_hash(observed),
        context=_tensor_hash(context),context_len=None if context_len is None else int(context_len),
        prompt=prompt,seed=int(seed),steps=int(num_steps),asset=asset_identity,shift=float(shift))
    if observation_mode != 'head':
        value.update(observation_mode=observation_mode, view_valid=_tensor_hash(observed_view_valid))
    for key, tensor in history_kwargs(locals()).items():
        if tensor is not None: value[key] = _tensor_hash(tensor)
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


@torch.no_grad()
def predict_goals(*, expert, observed, context, context_lens, seeds, num_steps,
                  cache, asset_identity, prompts, augmented, microbatch_size=1,
                  observed_view_valid=None, history_images=None, history_latents=None,
                  history_ages=None, history_valid=None, history_episode_id=None):
    """Generate detached goals with per-row seeds; augmented inputs bypass cache."""
    if not isinstance(microbatch_size,int) or microbatch_size<1:
        raise ValueError('goal microbatch_size must be positive')
    count=observed.shape[0]
    if count<1 or len(seeds)!=count or len(prompts)!=count or context.shape[0]!=count:
        raise ValueError('one context, seed and original prompt required per goal sample')
    if context_lens is not None:
        context_lens=torch.as_tensor(context_lens,device=context.device)
        if context_lens.shape!=(count,): raise ValueError('context_lens must be [B]')
    mode = getattr(expert, 'observation_mode', 'head')
    if mode == 'multiview':
        from .observation import sanitize_canvas, validate_view_valid
        observed_view_valid = validate_view_valid(observed_view_valid, batch_size=count, device=observed.device)
        observed = sanitize_canvas(observed, observed_view_valid)
    elif observed_view_valid is not None:
        raise ValueError('head goal expert does not accept a view mask')
    history = _validate_runtime_history(expert, count, history_images, history_latents, history_ages, history_valid, history_episode_id)
    _reset_history_caches(expert, cache, history, count)
    expert.freeze()
    results=[None]*count; keys=[None]*count; pending=[]
    for i in range(count):
        if not augmented and cache is not None:
            keys[i]=goal_cache_key(observed=observed[i],context=context[i],
                context_len=None if context_lens is None else int(context_lens[i]),
                prompt=prompts[i],seed=seeds[i],num_steps=num_steps,
                asset_identity=asset_identity,shift=expert.shift,observation_mode=mode,
                **{k:v[i] for k,v in history.items()},
                observed_view_valid=None if observed_view_valid is None else observed_view_valid[i],
                preprocessing_version=MULTIVIEW_PREPROCESSING_VERSION if mode=='multiview' else PREPROCESSING_VERSION)
            hit=cache.get(keys[i])
            if hit is not None:
                results[i]=GoalCondition(hit.latent.to(device=observed.device),
                    hit.image_token_valid.to(device=observed.device),hit.uv.to(device=observed.device))
        if results[i] is None: pending.append(i)
    for start in range(0,len(pending),microbatch_size):
        ids=pending[start:start+microbatch_size]
        kwargs = dict(context_lens=None if context_lens is None else context_lens[ids],
            **({'observed_view_valid': observed_view_valid[ids]} if mode=='multiview' else {}))
        representation = getattr(expert, 'representation', 'patch_denoise')
        if representation == 'visual_history_masked_sa':
            from .observation import observation_token_mask
            z = expert.condition_tokens(observed[ids], observed_view_valid=observed_view_valid[ids],
                **{k:v[ids] for k,v in history.items()}).detach()
            if z.shape != (len(ids), 120, expert.dit.dim):
                raise ValueError('visual history conditioner must return [B,120,DiT_dim]')
            pred = GoalCondition(z, observation_token_mask(observed_view_valid[ids]),
                                 torch.full((len(ids),2,2), float('nan'), device=z.device))
        elif representation == 'query_latent':
            z = expert.extract_tokens(observed[ids], context[ids], **kwargs, **{k:v[ids] for k,v in history.items()}).detach()
            pred = GoalCondition(z, torch.ones(z.shape[:2], device=z.device, dtype=torch.bool),
                                 torch.full((len(ids),2,2), float('nan'), device=z.device))
        else:
            pred=to_goal_condition(expert.generate(observed[ids],context[ids],
                seed=[seeds[i] for i in ids],num_steps=num_steps, **kwargs))
        for row,i in enumerate(ids):
            results[i]=GoalCondition(*(getattr(pred,name)[row:row+1] for name in
                                      ('latent','image_token_valid','uv')))
            if keys[i] is not None: cache.put(keys[i],results[i])
    return GoalCondition(*(torch.cat([getattr(r,name) for r in results]) for name in
        ('latent','image_token_valid','uv'))), {'generated_samples':len(pending)}

def configure_goal(architecture, cfg, *, checkpoint_dir=None):
    """Attach an independent frozen patch teacher after VAE/T5 restoration."""
    from pathlib import Path
    from omegaconf import OmegaConf
    config = OmegaConf.select(cfg, 'goal', default={}) if OmegaConf.is_config(cfg) else cfg.get('goal', {})
    config = OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else dict(config)
    architecture._frozen_dependencies_eval = bool(config.get('frozen_dependencies_eval', False))
    if not config.get('enabled', False):
        architecture.train(architecture.training)
        return
    model = OmegaConf.select(cfg, 'model', default={}) if OmegaConf.is_config(cfg) else cfg.get('model', {})
    model = OmegaConf.to_container(model, resolve=True) if OmegaConf.is_config(model) else model
    validate_goal_model_config(config, model)
    future_mode = architecture.future_video_mode
    if future_mode == 'none':
        from openwam.model.architectures.dual_system.joint_self_attn import DualSystemSelfAttnArchitecture
        if not isinstance(architecture, DualSystemSelfAttnArchitecture):
            raise ValueError("future_video_mode='none' requires dual-system joint_self_attn")
        if goal_representation(config) not in ('query_latent', 'visual_history_masked_sa'):
            raise ValueError("future_video_mode='none' requires predicted token conditioning")
    if config.get('mode', 'predicted') != 'predicted':
        raise ValueError('Stage2 requires predicted goals')
    mode = goal_observation_mode(config)
    vb = architecture.video_backbone
    driver = architecture.mot_driver
    if driver.attention_mask_mode != 'mutual' or vb.video_attention_mask_mode != 'first_frame_causal':
        raise ValueError('goal memory requires mutual + first_frame_causal')
    asset = resolve_goal_asset(config, checkpoint_dir)
    identity = goal_asset_sha256(asset)
    expected = config.get("asset_sha256")
    if checkpoint_dir and not expected:
        raise ValueError("Self-contained goal asset requires SHA256 identity")
    if expected and identity != expected:
        raise ValueError("Goal asset SHA256 mismatch")
    if goal_representation(config) == 'query_latent':
        from safetensors import safe_open
        with safe_open(str(asset), framework='pt', device='cpu') as file:
            metadata = json.loads((file.metadata() or {}).get('query_goal_bundle', '{}'))
        if HistoryConfig(**metadata.get('history_config', {})) != HistoryConfig(**config.get('history', {})):
            raise ValueError('goal history config disagrees with checkpoint')
    placement = next(vb.dit.parameters())
    expert = load_stage2_goal(asset, vae=vb.vae, text_encoder=getattr(vb,'text_encoder',None),
        tokenizer=getattr(vb,'_tokenizer',None), text_dim=vb.text_dim,
        device=placement.device, dtype=placement.dtype,
        precomputed_text_fingerprint=config.get('precomputed_text_fingerprint'))
    if getattr(expert, 'observation_mode', 'head') != mode:
        raise ValueError('goal configured mode disagrees with loaded expert mode')
    representation = goal_representation(config)
    if getattr(expert, 'representation', 'patch_denoise') != representation:
        raise ValueError('goal representation disagrees with loaded expert')
    if representation == 'query_latent':
        from .query_expert import QueryConfig
        if getattr(expert, 'history_config', HistoryConfig()) != HistoryConfig(**config.get('history', {})):
            raise ValueError('goal history config disagrees with checkpoint')
        if expert.query_config != QueryConfig(**config.get('query', {})):
            raise ValueError('goal query config disagrees with checkpoint')
    if representation == 'visual_history_masked_sa':
        if (expert.token_dim, expert.num_tokens) != tuple(visual_history_config(config)[k] for k in ('dim', 'num_tokens')):
            raise ValueError('goal visual history dimensions disagree with checkpoint')
        if expert.history_config != HistoryConfig(**config.get('history', {})):
            raise ValueError('goal history config disagrees with visual history sampling contract')
    if {id(p) for p in expert.parameters()} & {id(p) for p in vb.dit.parameters()}:
        raise ValueError('goal and video DiT Parameters must be independent')
    architecture.goal_expert = expert
    configure_goal_adapter(architecture, config)
    architecture.goal_config = config
    architecture.goal_asset_identity = identity
    architecture.goal_asset_source = str(asset.resolve())
    architecture.goal_cache = GoalPredictionCache(config.get('cache_size',128),config.get('cache_max_bytes',256*1024*1024))
    architecture.goal_diagnostics = {}
    architecture.train(architecture.training)


@torch.no_grad()
def prepare_goal_inputs(architecture, goal_observed_head, prompts, context, *,
                        head_valid, seeds, context_lens=None,
                        goal_observed_multiview=None, goal_view_valid=None,
                        history_images=None, history_latents=None, history_ages=None, history_valid=None, history_episode_id=None):
    """Condition only on explicit clean same-anchor RGB, never a canvas crop.

    Missing observations are masked without invoking the teacher on black images.
    Visual-history memory also supports rows with only valid wrist cameras.
    UV is detached and stored separately for diagnostics, never in model inputs.
    """
    count = context.shape[0]
    history = _validate_runtime_history(architecture.goal_expert, count, history_images, history_latents, history_ages, history_valid, history_episode_id)
    _reset_history_caches(architecture.goal_expert, architecture.goal_cache, history, count)
    valid = torch.as_tensor(head_valid,device=context.device)
    if valid.dtype != torch.bool or valid.shape != (count,):
        raise ValueError('head_valid must be bool [B]')
    if len(seeds)!=count or len(prompts)!=count:
        raise ValueError('explicit per-example seeds and prompts required')
    mode = getattr(architecture.goal_expert, 'observation_mode', 'head')
    observed = goal_observed_head
    view_valid = None
    if mode == 'multiview':
        if goal_view_valid is None:
            raise ValueError('multiview goal requires explicit goal_view_valid mask')
        view_valid = torch.as_tensor(goal_view_valid, device=context.device)
        if view_valid.dtype != torch.bool or view_valid.shape != (count,3):
            raise ValueError('goal_view_valid must be bool [B,3]')
        if not torch.equal(view_valid[:,0],valid):
            raise ValueError('goal view mask head must agree with head_valid')
        observed = goal_observed_multiview
        shape = (count,3,1,384,320)
        name = 'goal_observed_multiview'
    else:
        shape = (count,3,1,256,320)
        name = 'goal_observed_head'
    visual = getattr(architecture.goal_expert, 'representation', None) == 'visual_history_masked_sa'
    active = view_valid.any(1) if visual else valid
    if observed is None:
        if active.any(): raise ValueError(f'valid observations require explicit clean {name} RGB')
    elif tuple(observed.shape) != shape:
        raise ValueError(f'clean {name} must be {list(shape)}')
    representation = getattr(architecture.goal_expert, 'representation', 'patch_denoise')
    query = representation == 'query_latent'
    visual = representation == 'visual_history_masked_sa'
    spec = getattr(architecture.goal_expert, 'query_config', None)
    latent_shape = ((architecture.goal_expert.num_tokens, architecture.goal_expert.token_dim) if visual
                    else (spec.num_tokens, spec.dim) if query else (48,1,16,20))
    latent = context.new_zeros((count,*latent_shape))
    tokens = torch.zeros(count,latent_shape[0] if query or visual else 80,dtype=torch.bool,device=context.device)
    uv = torch.full((count,2,2),float('nan'),device=context.device)
    ids = active.nonzero(as_tuple=True)[0]
    if ids.numel():
        config = architecture.goal_config
        pred,_ = predict_goals(expert=architecture.goal_expert,
            observed=observed[ids],context=context[ids], **{k:v[ids] for k,v in history.items()},
            context_lens=None if context_lens is None else torch.as_tensor(context_lens,device=context.device)[ids],
            seeds=[seeds[i] for i in ids.tolist()],prompts=[prompts[i] for i in ids.tolist()],
            num_steps=int(config.get('num_steps',20)),cache=architecture.goal_cache,
            asset_identity=architecture.goal_asset_identity,augmented=False,
            microbatch_size=int(config.get('microbatch_size',1)),
            **({'observed_view_valid':view_valid[ids]} if mode=='multiview' else {}))
        latent[ids]=pred.latent.to(latent); tokens[ids]=pred.image_token_valid; uv[ids]=pred.uv
    architecture.goal_diagnostics = {'predicted_uv':uv.detach(),
        'predicted_latent':latent.detach(), 'image_token_valid':tokens.detach(), 'observation_mode':mode,
        'representation':representation, 'goal_decode_available':not (query or visual),
        'observed_view_valid':None if view_valid is None else view_valid.detach()}
    return {'goal_latent':latent.detach(),'goal_token_valid':tokens.detach()}


def resolve_goal_asset(config, checkpoint_dir=None):
    from pathlib import Path
    asset = Path(config.get('checkpoint') or '')
    if checkpoint_dir is not None:
        # Saved checkpoints always use a relative independent asset.
        if asset.is_absolute() or '..' in asset.parts:
            raise ValueError('Saved goal checkpoint must be relative to checkpoint directory')
        asset = Path(checkpoint_dir) / asset
    if not asset.is_file():
        raise FileNotFoundError(f'Goal patch asset missing: {asset}')
    return asset


def configure_goal_adapter(architecture, config=None):
    """Build adapter before strict unified load; teacher follows restored VAE/T5."""
    from .conditioning import GoalMemoryAdapter, QueryGoalMemoryAdapter, VisualHistoryMemoryAdapter
    config = config or {}
    representation = goal_representation(config)
    expected_type = {'query_latent':QueryGoalMemoryAdapter,
                     'visual_history_masked_sa':VisualHistoryMemoryAdapter,
                     'patch_denoise':GoalMemoryAdapter}[representation]
    if getattr(architecture, 'goal_adapter', None) is None:
        placement = next(architecture.video_backbone.dit.parameters())
        if representation == 'query_latent':
            adapter = QueryGoalMemoryAdapter(architecture.video_backbone, dict(config.get('query', {})))
        elif representation == 'visual_history_masked_sa':
            adapter = VisualHistoryMemoryAdapter(architecture.video_backbone, **visual_history_config(config))
        else:
            adapter = GoalMemoryAdapter(architecture.video_backbone)
        architecture.goal_adapter = adapter.to(device=placement.device, dtype=placement.dtype)
    elif not isinstance(architecture.goal_adapter, expected_type):
        raise ValueError('existing goal adapter representation mismatch')
    elif representation == 'query_latent':
        from .query_expert import QueryConfig
        if architecture.goal_adapter.config != QueryConfig(**config.get('query', {})):
            raise ValueError('existing query adapter dimensions mismatch')
    elif representation == 'visual_history_masked_sa':
        adapter = architecture.goal_adapter
        if dict(dim=adapter.dim, num_tokens=adapter.num_tokens) != visual_history_config(config):
            raise ValueError('existing visual history adapter dimensions mismatch')


def save_goal_assets(architecture, output_dir, cfg):
    """Copy frozen bundle once with bounded I/O and pin the portable identity."""
    import os
    import shutil
    from pathlib import Path
    from omegaconf import OmegaConf, open_dict
    if getattr(architecture, 'goal_expert', None) is None:
        return
    goal_prediction_policy(architecture)
    source = Path(architecture.goal_asset_source)
    target = Path(output_dir) / 'goal/expert.safetensors'
    target.parent.mkdir(parents=True, exist_ok=True)
    identity = architecture.goal_asset_identity
    if target.exists():
        if goal_asset_sha256(target) != identity:
            raise ValueError('Existing goal asset SHA256 mismatch')
    else:
        if goal_asset_sha256(source) != identity:
            raise ValueError('Source goal asset SHA256 mismatch')
        temporary = target.with_suffix('.tmp')
        shutil.copyfile(source, temporary)
        if goal_asset_sha256(temporary) != identity:
            temporary.unlink()
            raise ValueError('Copied goal asset SHA256 mismatch')
        os.replace(temporary, target)
    if OmegaConf.is_config(cfg):
        with open_dict(cfg):
            OmegaConf.update(cfg, 'goal.checkpoint', 'goal/expert.safetensors')
            OmegaConf.update(cfg, 'goal.asset_sha256', identity, force_add=True)
    else:
        cfg['goal'].update(checkpoint='goal/expert.safetensors', asset_sha256=identity)


def clean_goal_head(image):
    """Shared raw PIL RGB -> clean [3,1,256,320] input, before augmentation."""
    import numpy as np
    from PIL import Image
    if not isinstance(image, Image.Image):
        raise TypeError('clean goal head preprocessing requires a PIL image')
    image = image.convert('RGB').resize((320,256), Image.Resampling.BILINEAR)
    return torch.from_numpy(np.array(image, copy=True)).permute(2,0,1).unsqueeze(1).float().div(127.5).sub(1)


@torch.no_grad()
def prepare_inference_goal_inputs(architecture, goal_observed_head, prompts, context, *,
                                  head_valid, seeds, context_lens=None,
                                  goal_observed_multiview=None, goal_view_valid=None,
                                  history_images=None, history_latents=None, history_ages=None, history_valid=None, history_episode_id=None):
    """Inference uses the exact training condition builder and explicit row seeds."""
    from PIL import Image
    history = move_history({k:v for k,v in history_kwargs(locals()).items() if v is not None}, context.device, architecture.dtype)
    mode = getattr(architecture.goal_expert, 'observation_mode', 'head')
    if mode == 'multiview':
        if goal_observed_multiview is not None:
            if not isinstance(goal_observed_multiview, torch.Tensor):
                raise TypeError('goal_observed_multiview must be a normalized tensor')
            goal_observed_multiview = goal_observed_multiview.to(device=context.device,dtype=architecture.dtype)
        return prepare_goal_inputs(architecture, None, prompts, context,
            head_valid=head_valid,seeds=seeds,context_lens=context_lens,
            goal_observed_multiview=goal_observed_multiview,goal_view_valid=goal_view_valid, **history)
    if isinstance(goal_observed_head, Image.Image):
        goal_observed_head = clean_goal_head(goal_observed_head).unsqueeze(0)
    if goal_observed_head is not None:
        goal_observed_head = goal_observed_head.to(device=context.device, dtype=architecture.dtype)
        if not torch.isfinite(goal_observed_head).all() or goal_observed_head.abs().max() > 1.00001:
            raise ValueError('clean goal head must be finite RGB normalized to [-1,1]')
    return prepare_goal_inputs(architecture, goal_observed_head, prompts, context,
        head_valid=head_valid, seeds=seeds, context_lens=context_lens, **history)


def _validate_runtime_history(expert, count, images, latents, ages, valid, episode_id=None):
    values = dict(history_images=images, history_latents=latents, history_ages=ages, history_valid=valid, history_episode_id=episode_id)
    values = {k:v for k,v in values.items() if v is not None}
    history_config = getattr(expert, 'history_config', HistoryConfig(
        enabled=getattr(expert, 'representation', None) == 'visual_history_masked_sa'))
    if not history_config.enabled:
        if values: raise ValueError('history supplied to disabled goal expert')
        return {}
    history_episode_ids(episode_id, count)
    validate_history(ages, valid, batch_size=count)
    if (images is None) == (latents is None):
        raise ValueError('enabled history requires exactly one history images or latents tensor')
    tensor, shape = (images, (count,10,3,1,256,320)) if images is not None else (latents, (count,10,48,1,16,20))
    if tensor.shape != shape: raise ValueError('history input shape mismatch')
    if any(v.device != tensor.device for v in (ages, valid)):
        raise ValueError('history tensors must share a device')
    selected = tensor[valid]
    if not selected.is_floating_point() or not torch.isfinite(selected).all():
        raise ValueError('valid history inputs must be finite floating tensors')
    if images is not None and selected.numel() and selected.abs().max() > 1.00001:
        raise ValueError('history RGB must be normalized [-1,1]')
    return values


def _reset_history_caches(expert, cache, history, count):
    if not history:
        return
    ids=tuple(history_episode_ids(history.get('history_episode_id'),count))
    # Missing identity is safe for legacy callers, but disables persistent reuse.
    owner=cache if cache is not None else expert
    if (None in ids or ids != getattr(owner, '_history_episode_ids', None)
            or not bool(history['history_valid'].all())):
        if cache is not None:
            cache.entries.clear();cache.bytes_used=0
        expert.clear_history_cache()
    owner._history_episode_ids=ids
