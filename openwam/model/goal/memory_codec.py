"""Independent T=1 multiview VAE encoding and provenance-addressed disk storage."""
import hashlib
import torch
from openwam.model.goal.codec import GoalCodec, GoalCodecSpec, vae_cache_identity
from openwam.model.goal.goal_memory_cache import PersistentFrameLatentCache, tensor_fingerprint
from openwam.model.goal.observation import sanitize_canvas

PREPROCESS_IDENTITY = 'rgb-bilinear-head256x320-wrists128x160-canvas384x320-t1-v1'

class GoalMemoryCodec:
    latent_shape = (48, 1, 24, 20)

    def __init__(self, vae, cache_dir, *, vae_identity, preprocess_identity=PREPROCESS_IDENTITY,
                 device=None, dtype=None, max_encode_batch_size=4):
        self.vae=vae; self.device=device; self.dtype=dtype
        self.preprocess_identity=preprocess_identity
        self._signature=vae_cache_identity(vae)
        # Include actual weights, not just a caller-provided checkpoint filename.
        digest=hashlib.sha256(str(vae_identity).encode())
        if isinstance(vae,torch.nn.Module):
            for name,value in vae.state_dict().items():
                digest.update((name+str(tuple(value.shape))).encode());digest.update(tensor_fingerprint(value.reshape(-1)).encode())
        self.vae_identity=digest.hexdigest()
        self.cache=PersistentFrameLatentCache(cache_dir,vae_identity=self.vae_identity,
            preprocess_identity=preprocess_identity,latent_shape=self.latent_shape)
        self.codec=GoalCodec(vae,GoalCodecSpec(name='wan22_multiview_t1',
            image_shape=(3,1,384,320),latent_shape=self.latent_shape),max_encode_batch_size=max_encode_batch_size)

    def _check(self):
        if vae_cache_identity(self.vae)!=self._signature:
            raise ValueError('frozen VAE dependency changed; create a new GoalMemoryCodec')

    def get_cached(self, *, source, episode, frame, view='head+left_wrist+right_wrist'):
        self._check()
        return self.cache.get(source=source,episode=episode,frame=frame,view=view)

    def put_cached(self, *, source, episode, frame, latent, view='head+left_wrist+right_wrist'):
        self._check()
        self.cache.put(source=source,episode=episode,frame=frame,view=view,latent=latent)

    def encode_images(self, canvases, view_valid=None):
        self._check()
        if view_valid is not None:canvases=sanitize_canvas(canvases,view_valid)
        if self.dtype is not None:canvases=canvases.to(dtype=self.dtype)
        return self.codec.encode(canvases,device=self.device)

    def encode_image(self, canvas):
        return self.encode_images(canvas.unsqueeze(0))[0]
