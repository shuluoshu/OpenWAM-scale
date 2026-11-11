"""Atomic frozen-VAE frame storage and bounded episode-local pre-RoPE KV cache.

Frames are independently encoded at T=1 (1 Hz), never stride-4/streaming VAE.
Persistent keys carry source, episode, frame, view, preprocessing and VAE identity.
Runtime entries contain normalized K before rotary embedding, so changing window
positions never reuses stale rotary phases. Runtime caching is inference-only.
"""
from collections import OrderedDict
from pathlib import Path
import hashlib
import json
import os
import tempfile
import torch


def tensor_fingerprint(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(str((tuple(value.shape), value.dtype)).encode() + value.view(torch.uint8).numpy().tobytes()).hexdigest()


class PersistentFrameLatentCache:
    def __init__(self, root, *, vae_identity, preprocess_identity, latent_shape=(48,1,24,20)):
        if not vae_identity or not preprocess_identity:
            raise ValueError('explicit VAE and preprocessing identities are required')
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.dependencies = dict(format='openwam.goal_memory.frame.v1', vae=str(vae_identity),
                                 preprocess=str(preprocess_identity), latent_shape=list(latent_shape))

    def _identity(self, source, episode, frame, view):
        if not str(source) or not str(episode) or not str(view):
            raise ValueError('source, episode and view identities are required')
        return self.dependencies | dict(source=str(source), episode=str(episode), frame=int(frame), view=str(view))

    def _path(self, identity):
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        return self.root / (digest + '.pt')

    def _validate(self, latent):
        if tuple(latent.shape) != tuple(self.dependencies['latent_shape']) or not latent.is_floating_point() or not torch.isfinite(latent).all():
            raise ValueError('cache latent must be a finite independent single-frame tensor of the declared shape')

    def get(self, *, source, episode, frame, view):
        identity = self._identity(source, episode, frame, view)
        path = self._path(identity)
        if not path.exists():
            return None
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if payload['identity'] != identity:
            raise ValueError('frame cache dependency mismatch')
        self._validate(payload['latent'])
        return payload['latent']

    def put(self, *, source, episode, frame, view, latent):
        self._validate(latent)
        identity = self._identity(source, episode, frame, view)
        fd, temporary = tempfile.mkstemp(prefix='.frame-', suffix='.tmp', dir=self.root)
        try:
            with os.fdopen(fd, 'wb') as stream:
                torch.save(dict(identity=identity, latent=latent.detach().cpu()), stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self._path(identity))
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


class GoalMemoryRuntimeCache:
    def __init__(self, *, model_identity, max_entries=32):
        if not model_identity or max_entries < 1:
            raise ValueError('model identity and positive runtime cache capacity required')
        self.model_identity = str(model_identity)
        self.max_entries = int(max_entries)
        self.episode_id = None
        self.entries = OrderedDict()
        self.hits = self.misses = 0
        self._model_signature = None

    def set_episode(self, episode_id):
        if episode_id is None:
            raise ValueError('runtime cache requires an episode identity')
        if self.episode_id != episode_id:
            self.entries.clear()
            self.episode_id = episode_id

    def bind_model(self, model):
        parameters = tuple(model.parameters())
        # Parameter dtype alone does not describe mixed-precision execution.
        # Bind before lookup so approximate autocast entries never leak into FP32.
        precision = tuple(sorted(
            (device_type, torch.is_autocast_enabled(device_type),
             torch.get_autocast_dtype(device_type) if torch.is_autocast_enabled(device_type) else None)
            for device_type in {p.device.type for p in parameters}))
        signature = (tuple((id(p), p._version, p.device, p.dtype) for p in parameters), precision)
        if self._model_signature != signature:
            self.entries.clear()
            self._model_signature = signature

    def get(self, key):
        if key not in self.entries:
            self.misses += 1
            return None
        self.hits += 1
        self.entries.move_to_end(key)
        return self.entries[key]

    def put(self, key, value):
        self.entries[key] = value
        self.entries.move_to_end(key)
        while len(self.entries) > self.max_entries:
            self.entries.popitem(last=False)

    def __len__(self):
        return len(self.entries)


def select_memory_indices(frame_ids, *, recent_frames=2):
    """Keep the episode sink plus up to 2 (configurable 0..4) recent frames."""
    if not 0 <= recent_frames <= 4:
        raise ValueError('stage A supports zero through four recent memory frames')
    ids = list(frame_ids)
    if not ids:
        return []
    ordered = sorted(range(len(ids)), key=lambda i: ids[i])
    return sorted(set(ordered[:1] + (ordered[-recent_frames:] if recent_frames else [])), key=lambda i: ids[i])
