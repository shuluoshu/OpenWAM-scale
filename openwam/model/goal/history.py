"""Head-only history: detached feature cache, learned compression and physical time.

Each frame is independent of every other frame, current views and language.
Only the output goal queries fuse memory, so history cannot acquire future context.
"""
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import math

import torch
from torch import nn
from einops import rearrange


@dataclass(frozen=True)
class HistoryConfig:
    enabled: bool = False
    num_frames: int = 10
    tokens_per_frame: int = 16
    compressor_dim: int = 128
    cache_size: int = 256

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError('history.enabled must be bool')
        if self.num_frames != 10 or type(self.num_frames) is not int:
            raise ValueError('history requires 10 past slots')
        if self.tokens_per_frame != 16 or type(self.tokens_per_frame) is not int:
            raise ValueError('history requires 16 tokens per frame')
        if type(self.compressor_dim) is not int or self.compressor_dim < 8 or self.compressor_dim % 4:
            raise ValueError('history compressor_dim must be positive and divisible by 4')
        if type(self.cache_size) is not int or self.cache_size < 0:
            raise ValueError('history cache_size must be nonnegative')


def validate_history(ages, valid, *, batch_size, frames=10):
    if ages is None or valid is None:
        raise ValueError('enabled history requires explicit history_ages and history_valid')
    if valid.dtype != torch.bool or valid.shape != (batch_size,frames) or ages.shape != valid.shape:
        raise ValueError('history ages/mask must be [B,10], mask bool')
    if not ages.is_floating_point():
        raise ValueError('history ages must be floating-point seconds')
    for row, mask in zip(ages, valid):
        t=row[mask]
        if not torch.isfinite(t).all() or (t<=0).any() or (t[1:]>=t[:-1]).any():
            raise ValueError('history ages must be finite positive, strictly oldest-to-newest')


class HistoryCompressor(nn.Module):
    """Sixteen learned spatial queries; FP32 even in a BF16 parent model."""
    def __init__(self, config, output_dim):
        super().__init__()
        self.config=config
        d=config.compressor_dim
        self.input=nn.Linear(192,d)
        self.position=nn.Parameter(torch.randn(1,80,d)*.02)
        self.queries=nn.Parameter(torch.randn(1,16,d)*.02)
        self.norm=nn.LayerNorm(d)
        self.attention=nn.MultiheadAttention(d,4,batch_first=True,dropout=0.)
        self.output=nn.Linear(d,output_dim)
        self.time=nn.Sequential(nn.Linear(33,d),nn.SiLU(),nn.Linear(d,output_dim))
        # Nonzero initialization lets attention and input projection learn on step 1.
        nn.init.normal_(self.output.weight,std=.02)
        nn.init.zeros_(self.output.bias)

    def _apply(self, fn, recurse=True):
        def preserve_float32(tensor):
            transformed = fn(tensor)
            if tensor.is_floating_point():
                # Use the original values: casting BF16 back to FP32 loses bits.
                return tensor.to(device=transformed.device, dtype=torch.float32)
            return transformed
        return super()._apply(preserve_float32, recurse=recurse)

    def forward(self, latents, ages, valid):
        b,h=valid.shape
        if latents.shape != (b,h,48,1,16,20):
            raise ValueError('history latents must be [B,10,48,1,16,20]')
        # Detach even caller-provided features, matching the cached VAE route.
        x=torch.where(valid[:,:,None,None,None,None],latents.detach(),0.)
        if not torch.isfinite(x).all():
            raise ValueError('valid history features must be finite')
        with torch.autocast(device_type=x.device.type,enabled=False):
            x=rearrange(x.float(),'b t c 1 (h p) (w q) -> (b t) (h w) (p q c)',p=2,q=2)
            kv=self.norm(self.input(x)+self.position)
            q=self.queries.expand(b*h,-1,-1)
            compressed=q+self.attention(q,kv,kv,need_weights=False)[0]
            tokens=self.output(compressed).reshape(b,h,16,-1)
            age=torch.where(valid,ages,0.).float()
            frequencies=torch.exp(torch.linspace(math.log(.1),math.log(10.),16,device=x.device))
            phase=age[...,None]*frequencies
            time=torch.cat((torch.log1p(age)[...,None],phase.sin(),phase.cos()),dim=-1)
            tokens=tokens+self.time(time)[:,:,None,:]
            tokens=torch.where(valid[:,:,None,None],tokens,0.)
        return tokens.flatten(1,2),valid.repeat_interleave(16,dim=1)


class HeadFeatureCache:
    """Bounded CPU cache of frozen pre-compressor features, never trainable tokens.

    Pixels are keyed after model dtype conversion within one episode only.
    Episode changes evict all entries, even when pixels happen to be identical.
    Incomplete histories explicitly evict earlier entries before encoding their
    valid frames; missing slots always remain zero, never filled from the cache.
    """
    def __init__(self, size):
        self.size=size
        self.entries=OrderedDict()
        self.identity=None
        self.episode_id=None
        self.hits=self.misses=0

    def __len__(self): return len(self.entries)

    def clear(self):
        self.entries.clear(); self.identity=None; self.episode_id=None
        self.hits=self.misses=0

    def begin_episode(self, episode_id):
        # Legacy callers without identity cannot safely reuse cross-call state.
        if episode_id is None or episode_id != self.episode_id:
            self.entries.clear(); self.hits=self.misses=0
            self.episode_id=episode_id

    @torch.no_grad()
    def encode(self, codec, images, valid, *, episode_ids=None):
        b,h=valid.shape
        if images.shape != (b,h,3,1,256,320):
            raise ValueError('history images must be [B,10,3,1,256,320]')
        identity=(id(codec.vae),str(images.device),str(images.dtype),
            tuple((id(p),p._version) for p in codec.vae.parameters()))
        ids=history_episode_ids(episode_ids,b)
        cpu_valid=valid.detach().cpu()
        if identity!=self.identity or not bool(cpu_valid.all()):
            self.clear();self.identity=identity
        # One device-to-host transfer and one host-to-device transfer per batch.
        # Per-slot GPU validation/copy introduces a synchronization for every frame
        # and costs substantially more than the small learned compressor.
        cpu_images=images.detach().cpu().contiguous()
        result=torch.zeros((b,h,48,1,16,20),dtype=images.dtype)
        for row in range(b):
            self.begin_episode(ids[row])
            for slot in cpu_valid[row].nonzero().flatten().tolist():
                image=cpu_images[row,slot]
                if not torch.isfinite(image).all() or image.abs().max()>1.00001:
                    raise ValueError('history RGB must be finite normalized [-1,1]')
                raw=image.view(torch.uint8).numpy().tobytes()
                key=hashlib.sha256(raw).digest()
                if key in self.entries:
                    self.hits+=1;self.entries.move_to_end(key)
                    latent=self.entries[key]
                else:
                    self.misses+=1;latent=codec.encode(images[row,slot][None])[0].detach().cpu()
                    if self.size:
                        self.entries[key]=latent.clone()
                        while len(self.entries)>self.size:self.entries.popitem(last=False)
                result[row,slot]=latent
        return result.to(device=images.device)


def history_episode_ids(episode_ids, batch_size):
    if episode_ids is None:
        return [None]*batch_size
    if episode_ids.dtype != torch.int64 or episode_ids.shape != (batch_size,):
        raise ValueError('history_episode_id must be int64 [B]')
    return episode_ids.detach().cpu().tolist()


HISTORY_FIELDS=('history_images','history_latents','history_ages','history_valid','history_episode_id')


def history_kwargs(batch):
    return {key:batch[key] for key in HISTORY_FIELDS if key in batch}


def move_history(batch, device, dtype):
    return {key:value.to(device=device,dtype=(torch.int64 if key=='history_episode_id' else torch.bool if key=='history_valid' else
            torch.float32 if key=='history_ages' else dtype))
            for key,value in history_kwargs(batch).items()}
