"""Versioned query bundles; Stage2 selectively loads the encoder tensors only."""
from dataclasses import asdict
import json
import warnings
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from openwam.model.goal.codec import GoalCodec
from openwam.model.goal.dependencies import dependency_identity, module_identity, text_identity, precomputed_identity
from openwam.model.goal.observation import observation_metadata
from openwam.model.goal.query_expert import QUERY_FORMAT, QueryGoalEncoder, QueryGoalExpert, QueryConfig
from openwam.model.video_backbone.wan.models.dit import WanModel


def save_query_bundle(expert, checkpoint, *, training_metadata=None, dependency_metadata=None,
                      dependency_identities=None, precomputed_text_fingerprint=None):
    identities = dependency_identities or dependency_identity(expert.codec.vae,
        text_dim=expert.text_dim, precomputed_text_fingerprint=precomputed_text_fingerprint)
    meta = dict(format=QUERY_FORMAT, representation='query_latent',
        query_config=asdict(expert.query_config), history_config=asdict(expert.history_config),
        observation_mode=expert.observation_mode,
        observation=observation_metadata(expert.observation_mode), codec=expert.codec.metadata(),
        dit_config=expert.dit_config, shift=expert.shift,
        lambda_image=expert.lambda_image, lambda_tcp=expert.lambda_tcp,
        auxiliary_readout=False, conditioning_mode=expert.conditioning_mode,
        dependencies={**identities, 'reconstruction': dependency_metadata or {}},
        training=training_metadata or {},
        runtime='single encoder forward; encoder.* tensors only; auxiliary decoder removed')
    path = Path(checkpoint); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pending.safetensors')
    save_file({k: v.detach().cpu().contiguous() for k, v in expert.state_dict().items()},
              str(temporary), metadata={'query_goal_bundle': json.dumps(meta, sort_keys=True)})
    temporary.replace(path)
    path.with_suffix('.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return meta


def load_query_bundle(checkpoint, *, vae, text_encoder=None, tokenizer=None, text_dim=None,
                      precomputed_text_fingerprint=None, device='cpu', dtype=torch.float32,
                      encoder_only=True):
    with safe_open(str(checkpoint), framework='pt', device='cpu') as file:
        meta = json.loads((file.metadata() or {}).get('query_goal_bundle', '{}'))
        if meta.get('format') != QUERY_FORMAT or meta.get('representation') != 'query_latent':
            raise ValueError('unsupported query goal bundle format')
        config = QueryConfig(**meta['query_config'])
        mode = meta['observation_mode']
        if meta.get('observation') != json.loads(json.dumps(observation_metadata(mode))):
            raise ValueError('incompatible query observation contract')
        dep = meta['dependencies']
        if getattr(text_encoder, 'dim', text_dim) != dep['text_dim']:
            raise ValueError('incompatible text dimension')
        actual = text_identity(text_encoder, tokenizer) if text_encoder is not None else precomputed_identity(precomputed_text_fingerprint)
        if actual['fingerprint'] != dep['text']['fingerprint']:
            raise ValueError('incompatible text dependency')
        if module_identity(vae) != dep['vae']:
            raise ValueError('incompatible VAE dependency')
        if json.loads(json.dumps(GoalCodec(vae).metadata())) != meta['codec']:
            raise ValueError('incompatible query codec')
        if encoder_only:
            expert = QueryGoalEncoder(GoalCodec(vae), text_dim=dep['text_dim'],
                query_config=asdict(config), observation_mode=mode, history_config=meta.get('history_config'))
            state = {k: file.get_tensor(k) for k in file.keys() if k.startswith('encoder.')}
        else:
            expert = QueryGoalExpert(WanModel(**meta['dit_config']), GoalCodec(vae),
                dit_config=meta['dit_config'], query_config=asdict(config), observation_mode=mode,
                history_config=meta.get('history_config'),
                conditioning_mode=meta.get('conditioning_mode', 'legacy_text'),
                **{k: meta[k] for k in ('shift', 'lambda_image', 'lambda_tcp')})
            retired = [k for k in file.keys() if k.startswith('readout.')]
            if retired:
                warnings.warn('Loading legacy query bundle: removed readout weights are discarded. '
                              'Model weights only; start a new optimizer/run.', UserWarning, stacklevel=2)
            state = {k: file.get_tensor(k) for k in file.keys() if not k.startswith('readout.')}
        expert.load_state_dict(state, strict=True)
    expert.to(device=device, dtype=dtype).freeze()
    expert.bundle_metadata = meta
    for module in (vae, text_encoder):
        if isinstance(module, torch.nn.Module):
            for t in list(module.parameters()) + list(module.buffers()):
                if t.device != next(expert.parameters()).device or (t.is_floating_point() and t.dtype != dtype):
                    raise ValueError('place shared dependencies on requested device/dtype before load')
            module.eval().requires_grad_(False)
    return expert
