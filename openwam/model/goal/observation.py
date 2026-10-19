"""Clean three-camera observation layout; goal images remain head-only.

View order is head, left wrist, right wrist. A missing view is black in RGB
(-1 after normalization) before the frozen VAE runs, and its latent patches
are excluded as transformer keys. Masks are explicit booleans, never inferred
from pixel values.
"""
import numpy as np
import torch
from PIL import Image

OBSERVATION_MODES = ('head', 'multiview')
MULTIVIEW_IMAGE_SHAPE = (3, 1, 384, 320)
MULTIVIEW_LATENT_SHAPE = (48, 1, 24, 20)
MULTIVIEW_PATCH_GRID = (1, 12, 10)
VIEW_NAMES = ('head', 'left_wrist', 'right_wrist')
# (row start, row stop, column start, column stop), in normalized RGB canvas.
VIEW_REGIONS = ((0, 256, 0, 320), (256, 384, 0, 160), (256, 384, 160, 320))


def validate_view_valid(view_valid, *, batch_size=None, device=None):
    """Return bool [B,3] on device; reject implicit casts or empty observations."""
    if view_valid is None:
        raise ValueError('multiview observation requires explicit observed_view_valid')
    valid = torch.as_tensor(view_valid, device=device)
    if valid.dtype != torch.bool:
        raise ValueError('observed_view_valid must have bool dtype')
    if valid.ndim != 2 or valid.shape[1] != 3 or (batch_size is not None and valid.shape[0] != batch_size):
        raise ValueError('observed_view_valid must be [B,3] matching observation batch')
    if valid.shape[0] == 0 or not valid.any(dim=1).all():
        raise ValueError('each observation must contain at least one valid view')
    return valid


def sanitize_canvas(canvas, view_valid):
    """[B,3,1,384,320] normalized RGB -> copy with missing views black.

    Select before finite/range validation, so arbitrary values (including NaN)
    in unavailable camera slots never reach VAE convolutions.
    """
    if canvas.ndim != 5 or tuple(canvas.shape[1:]) != MULTIVIEW_IMAGE_SHAPE or not canvas.is_floating_point():
        raise ValueError('multiview canvas must be floating [B,3,1,384,320]')
    valid = validate_view_valid(view_valid, batch_size=canvas.shape[0], device=canvas.device)
    pixel_valid = torch.empty((canvas.shape[0], 1, 1, 384, 320), device=canvas.device, dtype=torch.bool)
    for i, (y0, y1, x0, x1) in enumerate(VIEW_REGIONS):
        pixel_valid[..., y0:y1, x0:x1] = valid[:, i, None, None, None, None]
    result = torch.where(pixel_valid, canvas, -torch.ones((), device=canvas.device, dtype=canvas.dtype))
    if not torch.isfinite(result).all():
        raise ValueError('valid multiview pixels must be finite')
    if result.abs().max() > 1.00001:
        raise ValueError('multiview pixels must be normalized RGB in [-1,1]')
    return result


def observation_token_mask(view_valid):
    """Return bool [B,120] keys, flattened row-major on the 12x10 grid."""
    valid = validate_view_valid(view_valid)
    grid = torch.empty((len(valid), 12, 10), device=valid.device, dtype=torch.bool)
    grid[:, :8] = valid[:, 0, None, None]
    grid[:, 8:, :5] = valid[:, 1, None, None]
    grid[:, 8:, 5:] = valid[:, 2, None, None]
    return grid.flatten(1)


def clean_multiview(frames, view_valid):
    """Ordered PIL RGB frames/arrays -> clean float32 [3,1,384,320].

    Missing slots may be None. All supplied views must already refer to the
    same native anchor instant; this helper only performs BILINEAR resizing.
    """
    if len(frames) != 3:
        raise ValueError('multiview frames must be ordered head/left_wrist/right_wrist')
    valid = torch.as_tensor(view_valid)
    if valid.shape != (3,):
        raise ValueError('clean_multiview view_valid must be bool [3]')
    valid = validate_view_valid(valid[None])[0]
    canvas = torch.full(MULTIVIEW_IMAGE_SHAPE, -1., dtype=torch.float32)
    for i, (y0, y1, x0, x1) in enumerate(VIEW_REGIONS):
        if not valid[i]:
            continue
        frame = frames[i]
        if isinstance(frame, np.ndarray):
            frame = Image.fromarray(frame)
        if not isinstance(frame, Image.Image):
            raise TypeError('valid multiview frames must be PIL images or RGB arrays')
        frame = frame.convert('RGB').resize((x1-x0, y1-y0), Image.Resampling.BILINEAR)
        pixels = torch.from_numpy(np.array(frame, copy=True)).permute(2, 0, 1).unsqueeze(1).float().div(127.5).sub(1)
        canvas[:, :, y0:y1, x0:x1] = pixels
    return canvas


def observation_metadata(mode):
    """Portable observation contract stored independently of head goal codec."""
    if mode not in OBSERVATION_MODES:
        raise ValueError(f'unsupported observation_mode: {mode}')
    if mode == 'head':
        return dict(mode='head', image_shape=(3, 1, 256, 320), latent_shape=(48, 1, 16, 20), tokens=80)
    return dict(mode='multiview', image_shape=MULTIVIEW_IMAGE_SHAPE,
                latent_shape=MULTIVIEW_LATENT_SHAPE, tokens=120,
                views=VIEW_NAMES, regions=VIEW_REGIONS, resize='BILINEAR',
                missing_pixel=-1., view_valid='bool[B,3]', token_mask='valid observation keys',
                rotary_frames=dict(observation=0, goal=1, tcp=2))
