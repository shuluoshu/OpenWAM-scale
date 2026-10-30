"""Single-frame Wan2.2 codec, borrowing the caller-owned frozen VAE once."""
from dataclasses import asdict, dataclass
from collections import OrderedDict
import hashlib
import torch


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

    def clear(self):
        self.entries.clear();self.nbytes=0

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
    def __init__(self, vae, spec=None, *, cache=None):
        self.vae = vae
        self.spec = spec or GoalCodecSpec()
        self.cache=cache

    def enable_cache(self,*,max_bytes):
        self.cache=LatentCache(max_bytes)

    def clear_cache(self):
        if self.cache is not None:self.cache.clear()

    @torch.no_grad()
    def encode(self, image):
        if tuple(image.shape[1:]) != self.spec.image_shape:
            raise ValueError(f'goal image must be [B,{self.spec.image_shape}], got {tuple(image.shape)}')
        if not torch.isfinite(image).all():
            raise ValueError('goal codec image contains nonfinite pixels')
        if isinstance(self.vae, torch.nn.Module):
            self.vae.eval().requires_grad_(False)
        key=None
        if self.cache is not None and self.cache.max_bytes:
            # Hash actual normalized input bits, geometry and placement/dtype.
            # This cache is run-local, initialized after frozen dependency checks.
            first=next(iter(self.vae.parameters()),None) if isinstance(self.vae,torch.nn.Module) else None
            placement=(str(first.device),str(first.dtype)) if first is not None else None
            key=(self.spec.name,tuple(self.spec.image_shape),tuple(self.spec.latent_shape),str(image.dtype),placement,tuple(image.shape),hashlib.sha256(
                image.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).digest())
            cached=self.cache.get(key)
            if cached is not None:return cached.to(device=image.device,dtype=image.dtype).clone()
        latent = self.vae.encode(list(image.unbind(0)), device=image.device)
        self.validate_latent(latent)
        latent=latent.to(device=image.device, dtype=image.dtype)
        if key is not None:self.cache.put(key,latent)
        return latent

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
