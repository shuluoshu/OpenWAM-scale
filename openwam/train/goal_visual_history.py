"""Portable visual-history Stage1 checkpoints with explicit frozen dependencies."""
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from openwam.model.goal.codec import GoalCodec
from openwam.model.goal.dependencies import dependency_identity, module_identity, precomputed_identity
from openwam.model.goal.visual_history import VisualHistoryGoalExpert
from openwam.model.video_backbone.wan.models.dit import WanModel
from openwam.train.goal_overfit import _write_json

FORMAT = 'openwam.visual_history.v1'


def save_visual_history_bundle(expert, path, *, precomputed_text_fingerprint=None,
                               dependency_identities=None, training_metadata=None, dependency_metadata=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    deps=dependency_identities or dependency_identity(expert.codec.vae,
        text_dim=expert.context_width,precomputed_text_fingerprint=precomputed_text_fingerprint)
    meta=dict(format=FORMAT,dit_config=expert.dit_config,history=expert.history_options,
        shift=expert.shift,lambda_image=expert.lambda_image,lambda_tcp=expert.lambda_tcp,
        codec=expert.codec.metadata(),dependencies=deps,training=training_metadata or {},
        representation=expert.representation,observation_mode=expert.observation_mode,
        reconstruction=dependency_metadata or {})
    temporary=path.with_suffix('.pending.safetensors')
    save_file({k:v.detach().cpu().contiguous() for k,v in expert.state_dict().items()},str(temporary),
              metadata={'visual_history':json.dumps(meta)})
    temporary.replace(path);_write_json(path.with_suffix('.json'),meta)
    return meta


def load_visual_history_bundle(path, *, vae, text_dim, precomputed_text_fingerprint,
                               device='cpu', dtype=torch.float32, encoder_only=False):
    """Caller supplies the frozen VAE and fingerprint of the text encoder/cache.

    No historical training path is opened; all trainable weights and structure
    are reconstructed from this single bundle. Returned expert is frozen.
    encoder_only removes target blocks/head/text embedding on CPU before device
    placement, retaining the modules used by condition_tokens exactly.
    """
    with safe_open(str(path),framework='pt',device='cpu') as f:
        raw=(f.metadata() or {}).get('visual_history')
        if raw is None:raise ValueError('missing visual history metadata')
        meta=json.loads(raw)
        if meta.get('format')!=FORMAT:raise ValueError('unsupported visual history format')
        dep=meta['dependencies']
        if text_dim!=dep['text_dim']:raise ValueError('incompatible text dimension')
        if precomputed_identity(precomputed_text_fingerprint)['fingerprint']!=dep['text']['fingerprint']:
            raise ValueError('incompatible text dependency')
        if module_identity(vae)!=dep['vae']:raise ValueError('incompatible VAE dependency')
        codec=GoalCodec(vae)
        if 'codec' in meta and json.loads(json.dumps(codec.metadata()))!=meta['codec']:
            raise ValueError('incompatible codec')
        expert=VisualHistoryGoalExpert(WanModel(**meta['dit_config']),codec,dit_config=meta['dit_config'],
            shift=meta['shift'],lambda_image=meta['lambda_image'],lambda_tcp=meta['lambda_tcp'],**meta['history'])
        expert.load_state_dict({k:f.get_tensor(k) for k in f.keys()},strict=True)
    if encoder_only:
        from openwam.model.goal.visual_history_encoder import FrozenVisualHistoryEncoder
        expert = FrozenVisualHistoryEncoder(expert)
    expert.to(device=device,dtype=dtype).freeze();expert.bundle_metadata=meta
    resolved_device=next(expert.parameters()).device
    for tensor in list(vae.parameters())+list(vae.buffers()):
        if tensor.device!=resolved_device or (tensor.is_floating_point() and tensor.dtype!=dtype):
            raise ValueError('place shared VAE on requested device/dtype before load')
    vae.eval().requires_grad_(False)
    return expert
