"""Portable identities for the frozen representations used by a goal teacher.

Hash in-memory modules/tokenizers at bootstrap or asset load, never during goal
forward/sampling. File locations are deliberately absent from these identities.
"""
import hashlib
import inspect
import json
from pathlib import Path
import torch
from torch import nn


def _digest_json(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def _class_descriptor(value):
    cls=type(value)
    try:code=inspect.getsource(cls)
    except (OSError,TypeError):code=None
    return dict(model_class=cls.__module__+'.'+cls.__qualname__,
                implementation_sha256=hashlib.sha256(code.encode()).hexdigest() if code else None)


def _tensor_digest(value):
    value=value.detach().cpu().contiguous()
    digest=hashlib.sha256()
    digest.update(json.dumps([str(value.dtype),list(value.shape)]).encode())
    raw=memoryview(value.reshape(-1).view(torch.uint8).numpy())
    for offset in range(0,len(raw),8*1024*1024):digest.update(raw[offset:offset+8*1024*1024])
    return digest.hexdigest()


def _portable(value):
    if value is None or isinstance(value,(bool,int,float,str)):return value
    if isinstance(value,torch.Tensor):return dict(tensor_sha256=_tensor_digest(value))
    if isinstance(value,(tuple,list)):return [_portable(v) for v in value]
    if isinstance(value,dict):return {str(k):_portable(v) for k,v in value.items()}
    raise TypeError(type(value).__name__)


def _module_config(module):
    """Record module topology and immutable public numeric/config attributes.

    Registered tensors are hashed separately; nonregistered normalization scale
    tensors are included here (Wan VAE mean/std/scale). Runtime lists such as
    causal caches/call traces are excluded unless constructor configuration.
    """
    result={}
    for name,child in module.named_modules():
        try:constructor=set(inspect.signature(type(child).__init__).parameters)
        except (TypeError,ValueError):constructor=set()
        attrs={}
        for key,value in vars(child).items():
            if key.startswith('_') or key=='training' or isinstance(value,(nn.Module,nn.Parameter)):continue
            if isinstance(value,list) and key not in constructor and not any(isinstance(v,torch.Tensor) for v in value):continue
            if isinstance(value,(torch.Tensor,tuple,list,dict,str,bool,int,float)) or value is None:
                try:attrs[key]=_portable(value)
                except TypeError:continue
        result[name]={**_class_descriptor(child),'config':attrs}
    return result


def module_identity(module):
    if not isinstance(module,nn.Module):raise ValueError('dependency must be a torch module')
    digest=hashlib.sha256()
    for key,value in sorted(module.state_dict().items()):
        if not isinstance(value,torch.Tensor):raise ValueError('unsupported nontensor dependency state')
        digest.update(key.encode());digest.update(_tensor_digest(value).encode())
    descriptor=dict(**_class_descriptor(module),configuration=_module_config(module),weights_sha256=digest.hexdigest())
    return {**descriptor,'fingerprint':_digest_json(descriptor)}


def tokenizer_identity(tokenizer):
    if tokenizer is None:raise ValueError('goal text dependency requires tokenizer')
    underlying=getattr(tokenizer,'tokenizer',tokenizer)
    if not hasattr(underlying,'get_vocab'):raise ValueError('tokenizer dependency must expose vocabulary')
    config={}
    for key in ('clean','seq_len','model_max_length','padding_side','truncation_side','vocab_size','special_tokens_map',
                'split_special_tokens','add_prefix_space','legacy','add_bos_token','add_eos_token','is_fast'):
        for prefix,instance in (('wrapper',tokenizer),('tokenizer',underlying)):
            if hasattr(instance,key):
                try:config[f'{prefix}.{key}']=_portable(getattr(instance,key))
                except TypeError:config[f'{prefix}.{key}']=str(getattr(instance,key))
    backend=getattr(underlying,'backend_tokenizer',None)
    if backend is not None:
        behavior=json.loads(backend.to_str())
        # These are mutable call-time settings; encode_text explicitly supplies
        # padding/truncation/max_length512. Keep model, normalization, added
        # tokens, pre/post processors and decoder behavior in the identity.
        behavior.pop('truncation',None);behavior.pop('padding',None)
    elif hasattr(underlying,'sp_model'):
        behavior={'sentencepiece_sha256':hashlib.sha256(underlying.sp_model.serialized_model_proto()).hexdigest(),
                  'legacy':getattr(underlying,'legacy',None),'add_prefix_space':getattr(underlying,'add_prefix_space',None)}
    else:
        # Custom tokenizers still have a class implementation identity, vocab,
        # and explicit cleaning/sequence settings; no opaque callable accepted.
        behavior={'callable':_class_descriptor(underlying)}
    descriptor=dict(wrapper=_class_descriptor(tokenizer),tokenizer=_class_descriptor(underlying),config=config,
                    vocabulary_sha256=_digest_json(underlying.get_vocab()),behavior_sha256=_digest_json(behavior),
                    context_call={'add_special_tokens':True,'max_length':512,'padding':'max_length','truncation':True})
    return {**descriptor,'fingerprint':_digest_json(descriptor)}


def text_identity(text_encoder,tokenizer):
    descriptor=dict(mode='encoded',encoder=module_identity(text_encoder),tokenizer=tokenizer_identity(tokenizer))
    return {**descriptor,'fingerprint':_digest_json(descriptor)}


def precomputed_identity(fingerprint):
    if not isinstance(fingerprint,str) or len(fingerprint)!=64 or any(c not in '0123456789abcdef' for c in fingerprint):
        raise ValueError('precomputed text requires an explicit compatible SHA256 fingerprint')
    return dict(mode='precomputed',fingerprint=fingerprint)


def dependency_identity(vae, *, text_dim,text_encoder=None,tokenizer=None,precomputed_text_fingerprint=None):
    if text_encoder is not None:
        if getattr(text_encoder,'dim',None)!=text_dim:raise ValueError('incompatible text encoder dimensions')
        text=text_identity(text_encoder,tokenizer)
    else:text=precomputed_identity(precomputed_text_fingerprint)
    return dict(vae=module_identity(vae),text_dim=text_dim,text=text)


def resolve_tokenizer_config(config):
    """Resolve relocated checkpoint-relative tokenizer defaults; explicit wins."""
    import copy
    token_cfg=copy.deepcopy(config['tokenizer'])
    if not token_cfg['kwargs'].get('name'):
        subdir=token_cfg.get('subdir','tokenizer/google/umt5-xxl')
        path=Path(config['baseline_checkpoint']).resolve().parent/subdir
        if not path.is_dir():raise ValueError(f'baseline checkpoint tokenizer directory is missing: {path}')
        token_cfg['kwargs']['name']=str(path)
    return token_cfg
