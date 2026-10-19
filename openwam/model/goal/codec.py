"""Single-frame Wan2.2 codec, borrowing the caller-owned frozen VAE once."""
from dataclasses import asdict, dataclass
import torch


@dataclass(frozen=True)
class GoalCodecSpec:
    name: str = 'wan22_vae48_single_frame'
    image_shape: tuple = (3, 1, 256, 320)
    latent_shape: tuple = (48, 1, 16, 20)
    pixel_range: tuple = (-1., 1.)


class GoalCodec:
    """Plain object deliberately: the shared VAE is never a child of GoalExpert.

    Its dependency must already be on the right device and frozen. Independent
    encode calls reset causal VAE state; observation and terminal image are
    never represented as a temporal clip.
    """
    def __init__(self, vae, spec=None):
        self.vae = vae
        self.spec = spec or GoalCodecSpec()

    @torch.no_grad()
    def encode(self, image):
        if tuple(image.shape[1:]) != self.spec.image_shape:
            raise ValueError(f'goal image must be [B,{self.spec.image_shape}], got {tuple(image.shape)}')
        if not torch.isfinite(image).all():
            raise ValueError('goal codec image contains nonfinite pixels')
        if isinstance(self.vae, torch.nn.Module):
            self.vae.eval().requires_grad_(False)
        latent = self.vae.encode(list(image.unbind(0)), device=image.device)
        self.validate_latent(latent)
        return latent.to(device=image.device, dtype=image.dtype)

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
