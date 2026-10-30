"""Canonical RGB at the encoded-video boundary, scoped to audited releases.

RoboMind AgileX stores JPEG buffers with the recorder's RGB/BGR convention.
The V0 release converter decoded them with PIL then wrote rgb24 MP4 without
undoing that convention. Standard video decoders faithfully return swapped
colors. This is not a global OpenCV/PyAV conversion and excludes Franka/UR.
"""
from pathlib import Path

from PIL import Image

RGB = 'native_rgb_v1'
AGILEX_RGB = 'robomind_agilex_release_rb_fix_v1'
RELEASES = {'pretrain_datasetsV0_1',
            'pretrain_datasetsV0_2', 'pretrain_datasetsV0_3', 'pretrain_datasetsV0_4'}


def video_color_policy(path):
    parts = Path(path).parts
    for i, part in enumerate(parts[:-1]):
        if part not in RELEASES:
            continue
        tail = parts[i+1:]
        if tail[0] == 'robomind_v1' and 'h5_agilex_3rgb' in tail:
            return AGILEX_RGB
        if tail[0] == 'robomind_v2' and 'agilex__source_native_joint' in tail:
            return AGILEX_RGB
    return RGB


def canonical_video_rgb(image, video_path):
    """Convert once, before resize, multiview composition or normalization."""
    policy = video_color_policy(video_path)
    prior = image.info.get('openwam_color_policy')
    if prior is not None:
        if prior != policy:
            raise ValueError('image has a different video color policy')
        return image
    image = image.convert('RGB')
    if policy == AGILEX_RGB:
        r, g, b = image.split()
        image = Image.merge('RGB', (b, g, r))
    image.info['openwam_color_policy'] = policy
    return image


def _sample_image_hashes(batch):
    import hashlib
    import torch
    fields = {'observed': batch['observed'], 'target_image': batch['targets'].image}
    if 'history_images' in batch:
        fields['history_images'] = batch['history_images']
    return {key: hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            for key, value in fields.items()}


def stamp_sample_color(record, batch):
    """Seal actual normalized inputs after all views/history/targets are built."""
    policy = video_color_policy(record.get('source_repo', ''))
    record['image_color_policy'] = policy
    batch['image_color_policy'] = policy
    if policy != RGB:
        if 'history_latents' in batch:
            raise ValueError('rebuild RGB history before caching VAE latents')
        record['color_input_sha256'] = _sample_image_hashes(batch)


def validate_sample_color(record, batch):
    """Reject historical .pt caches instead of silently training swapped RGB."""
    policy = video_color_policy(record.get('source_repo', ''))
    if policy == RGB:
        return
    if record.get('image_color_policy') != policy or batch.get('image_color_policy') != policy:
        raise ValueError('stale RGB cache: rebuild current/history/target from source with '+policy)
    if 'history_latents' in batch:
        raise ValueError('stale RGB cache: cached history latents need regeneration')
    actual = _sample_image_hashes(batch)
    expected = record.get('color_input_sha256', {})
    if set(actual) != set(expected):
        raise ValueError('stale RGB cache: image fields changed')
    for key, value in actual.items():
        if expected.get(key) != value:
            raise ValueError('stale RGB cache: inconsistent '+key)


def _stage2_image_hashes(sample):
    import hashlib
    import torch
    def digest(value):
        if isinstance(value, Image.Image):return hashlib.sha256(value.convert('RGB').tobytes()).hexdigest()
        if isinstance(value, torch.Tensor):return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        if isinstance(value, (tuple, list)):return [digest(x) for x in value]
        if value is None:return None
        raise TypeError('unsupported cached Stage2 image type: '+type(value).__name__)
    keys=('video','vace_video','first_frame_image','goal_observed_head','goal_observed_multiview','history_images')
    return {key:digest(sample[key]) for key in keys if key in sample}


def stamp_stage2_color(sample):
    policy=video_color_policy(sample.get('trace',{}).get('source_repo',''))
    sample['image_color_policy']=policy
    if policy != RGB:sample['color_input_sha256']=_stage2_image_hashes(sample)


def validate_stage2_color(sample):
    policy=video_color_policy(sample.get('trace',{}).get('source_repo',''))
    if policy == RGB:return
    if sample.get('image_color_policy') != policy:
        raise ValueError('stale RGB cache: regenerate Stage2 samples from source')
    actual=_stage2_image_hashes(sample);expected=sample.get('color_input_sha256',{})
    if set(actual)!=set(expected):raise ValueError('stale RGB cache: Stage2 image fields changed')
    for key,value in actual.items():
        if value!=expected[key]:raise ValueError('stale RGB cache: Stage2 '+key)
