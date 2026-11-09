"""Single-frame Wan2.2 codec, borrowing the caller-owned frozen VAE once."""
from dataclasses import asdict, dataclass
from collections import OrderedDict
import hashlib
import torch


def vae_cache_identity(vae):
    """Track frozen dependency changes without hashing weight bytes per batch."""
    if not isinstance(vae, torch.nn.Module):
        return (id(vae),)
    return (id(vae), tuple((id(t), t._version, str(t.device), str(t.dtype))
                          for t in (*vae.parameters(), *vae.buffers())))


@dataclass(frozen=True)
class GoalCodecSpec:
    name: str = 'wan22_vae48_single_frame'
    image_shape: tuple = (3, 1, 256, 320)
    latent_shape: tuple = (48, 1, 16, 20)
    pixel_range: tuple = (-1., 1.)


class LatentCache:
    """Bounded CPU LRU for a single run's immutable frozen VAE outputs."""
    def __init__(self,max_bytes):
        if type(max_bytes) is not int or max_bytes<0:raise ValueError('codec cache bytes must be nonnegative')
        self.max_bytes=max_bytes;self.nbytes=0;self.entries=OrderedDict()
        self.dependency=None

    def clear(self):
        self.entries.clear();self.nbytes=0;self.dependency=None

    def get(self,key):
        if key not in self.entries:return None
        value=self.entries.pop(key);self.entries[key]=value
        return value

    def put(self,key,value):
        size=value.numel()*value.element_size()
        if size>self.max_bytes:return
        if key in self.entries:self.nbytes-=self.entries.pop(key).numel()*value.element_size()
        while self.entries and self.nbytes+size>self.max_bytes:
            old=self.entries.popitem(last=False)[1];self.nbytes-=old.numel()*old.element_size()
        self.entries[key]=value.detach().cpu().clone();self.nbytes+=size


class GoalCodec:
    """Plain object deliberately: the shared VAE is never a child of GoalExpert.

    Its dependency must already be on the right device and frozen. Independent
    encode calls reset causal VAE state; observation and terminal image are
    never represented as a temporal clip.
    """
    def __init__(self, vae, spec=None, *, cache=None, max_encode_batch_size=4):
        if type(max_encode_batch_size) is not int or max_encode_batch_size < 1:
            raise ValueError('VAE encode batch size must be a positive integer')
        self.vae = vae
        self.spec = spec or GoalCodecSpec()
        self.cache=cache
        self.max_encode_batch_size=max_encode_batch_size

    def enable_cache(self,*,max_bytes):
        self.cache=LatentCache(max_bytes)

    def clear_cache(self):
        if self.cache is not None:self.cache.clear()

    @torch.no_grad()
    def encode(self, image, *, device=None):
        if image.ndim != 5 or not len(image) or tuple(image.shape[1:]) != self.spec.image_shape:
            raise ValueError(f'goal image must be [B,{self.spec.image_shape}], got {tuple(image.shape)}')
        if type(self.max_encode_batch_size) is not int or self.max_encode_batch_size < 1:
            raise ValueError('VAE encode batch size must be a positive integer')
        if not torch.isfinite(image).all():
            raise ValueError('goal codec image contains nonfinite pixels')
        device=torch.device(image.device if device is None else device)
        if isinstance(self.vae, torch.nn.Module):
            self.vae.eval().requires_grad_(False)
        cache=self.cache if self.cache is not None and self.cache.max_bytes else None
        keys=None
        if cache is not None:
            identity=vae_cache_identity(self.vae)
            if cache.dependency!=identity:
                cache.clear();cache.dependency=identity
            # Per-image keys let AB -> AC reuse A. History keys stay on CPU.
            cpu=image.detach().contiguous().cpu()
            prefix=(self.spec.name,tuple(self.spec.image_shape),tuple(self.spec.latent_shape),
                    str(image.dtype),str(device))
            keys=[prefix+(hashlib.sha256(frame.view(torch.uint8).numpy().tobytes()).digest(),)
                  for frame in cpu]
        output=torch.empty((len(image),*self.spec.latent_shape),device=device,dtype=image.dtype)
        pending=OrderedDict();hit_indices=[];hit_values=[]
        for i in range(len(image)):
            key=i if keys is None else keys[i]
            cached=None if cache is None else cache.get(key)
            if cached is not None:
                hit_indices.append(i);hit_values.append(cached)
            elif key in pending:
                pending[key].append(i)
            else:
                pending[key]=[i]
        if hit_indices:
            output[torch.tensor(hit_indices,device=device)]=torch.stack(hit_values).to(device=device,dtype=image.dtype)
        missing=list(pending.items())
        native=getattr(self.vae,'batch_encode',None)
        for start in range(0,len(missing),self.max_encode_batch_size):
            chunk=missing[start:start+self.max_encode_batch_size]
            pixels=image[[positions[0] for _,positions in chunk]]
            # Keep T=1: batching must not turn sparse history into a video.
            latent=(native(pixels,device=device) if callable(native)
                    else self.vae.encode(list(pixels.unbind(0)),device=device))
            self.validate_latent(latent)
            latent=latent.to(device=device,dtype=image.dtype)
            cpu_latent=latent.detach().cpu() if cache is not None else None
            for j,(key,positions) in enumerate(chunk):
                output[positions]=latent[j]
                if cache is not None:cache.put(key,cpu_latent[j])
        return output

    def validate_latent(self, latent):
        if tuple(latent.shape[1:]) != self.spec.latent_shape:
            raise ValueError(f'incompatible VAE latent: expected {self.spec.latent_shape}, got {tuple(latent.shape)}')
        if not torch.isfinite(latent).all():
            raise ValueError('goal codec latent contains nonfinite values')

    @torch.no_grad()
    def decode(self, latent):
        self.validate_latent(latent)
        if isinstance(self.vae, torch.nn.Module): self.vae.eval()
        image = self.vae.decode(latent, device=latent.device, tiled=False)
        if tuple(image.shape[1:]) != self.spec.image_shape:
            raise ValueError(f'incompatible decoded goal image: {tuple(image.shape)}')
        return image

    def metadata(self):
        return asdict(self.spec)
