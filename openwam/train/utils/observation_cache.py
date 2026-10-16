"""Reuse clean prepared inputs without freezing stochastic RGB/state changes."""
import torch

from openwam.dataloader.transforms.observation import CLEAN_STATE_KEY, TRACE_KEY


def prepare_observation(sample, prepare, cache):
    """prepare accepts a list of samples; cache contains only clean observations.

    Changed RGB requires fresh encoder work. State-only dropout reuses the clean
    image encoding but substitutes new zero tensors, never mutating the cache.
    """
    augmentation = sample.get(TRACE_KEY, {})
    key = sample.get("_curated_index")
    if augmentation.get("image_changed") or key is None:
        return prepare([sample])
    if key not in cache:
        clean = sample
        if augmentation.get("state_dropped"):
            clean = dict(sample)
            clean["proprio"], mask = sample[CLEAN_STATE_KEY]
            if mask is None:
                clean.pop("proprio_mask",None)
            else:
                clean["proprio_mask"] = mask
        cache[key] = prepare([clean])
    result = cache[key]
    if augmentation.get("state_dropped"):
        result = dict(result)
        result["proprio"] = torch.zeros_like(result["proprio"])
        result["proprio_mask"] = torch.zeros_like(result["proprio_mask"],dtype=torch.bool)
    return result
