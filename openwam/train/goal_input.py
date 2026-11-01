"""CPU Stage1 inputs built directly from decoded, augmented RGB images.

Keep the conversion identical to prepare_overfit_images: resize in PIL before
float32 normalization, then cast image tensors only. Labels retain their dtypes.
"""
import numpy as np
from PIL import Image
import torch

from openwam.model.goal.types import GoalTargets


def _image_tensor(image, dtype, *, size=(320,256)):
    array=np.array(image.convert('RGB').resize(size,Image.Resampling.BILINEAR),copy=True)
    return (torch.from_numpy(array).permute(2,0,1)[None,:,None].float()/127.5-1).to(dtype=dtype)


def prepare_goal_images(record, observed, goal, projection, *, dtype):
    """Build one singleton batch without an intermediate PNG or NPZ asset."""
    targets=GoalTargets(
        _image_tensor(goal,dtype),
        torch.tensor([record.get('goal_image_valid',True)],dtype=torch.bool),
        torch.as_tensor(projection['uv'].copy(),dtype=torch.float32)[None],
        torch.as_tensor(projection['projection_valid'].copy(),dtype=torch.bool)[None],
        torch.as_tensor(projection['in_frame'].copy(),dtype=torch.bool)[None],
        torch.as_tensor(projection['visibility'].copy(),dtype=torch.int8)[None])
    batch=dict(targets=targets,sources=[record['family']])
    mode=record.get('observation_mode','head')
    if mode=='multiview':
        from openwam.model.goal.observation import sanitize_canvas,validate_view_valid
        if observed.size!=(320,384):raise ValueError('multiview observation asset must be W320 H384')
        valid=validate_view_valid(torch.as_tensor(record['view_valid'])[None],batch_size=1)
        batch.update(observed=sanitize_canvas(_image_tensor(observed,dtype,size=(320,384)),valid),observed_view_valid=valid)
    elif mode=='head':batch['observed']=_image_tensor(observed,dtype)
    else:raise ValueError(f'unsupported observation_mode: {mode}')
    return batch
