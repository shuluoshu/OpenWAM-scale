"""Image-only Stage1 augmentation, isolated from labels and history."""
import random

from openwam.dataloader.transforms.observation import (
    ObservationAugmentation, TRACE_KEY, missing_view_mask,
)


class GoalObservationAugmentation:
    def __init__(self, config=None, *, observation_mode):
        config = {} if config is None else dict(config)
        if 'state_dropout' in config:
            raise ValueError('Stage1 observation augmentation cannot perturb state')
        self.transform = ObservationAugmentation(config)
        if self.transform.has_image_operations and observation_mode != 'multiview':
            raise ValueError('Stage1 image augmentation requires multiview layout')

    def __call__(self, image, view_valid, *, seed, training):
        if not training or not self.transform.enabled:
            return image, None
        # Existing PIL transforms use Python random only. Restore the caller's
        # stream so augmentation cannot change history sampling or flow noise.
        state = random.getstate()
        try:
            random.seed(seed)
            missing = missing_view_mask(*image.size, view_valid)
            output = self.transform.apply({'video': [image]}, [missing])
        finally:
            random.setstate(state)
        return output['video'][0], dict(output[TRACE_KEY], seed=seed)
