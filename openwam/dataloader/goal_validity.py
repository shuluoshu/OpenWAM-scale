"""Stage1 endpoint observations, separately from Stage2 next-action masks.

Only the explicitly recorded Genrobot conversion contract is relaxed. Missing
or unreviewed semantics keep the legacy rule. Decoding must still succeed; TCP
geometry continues to use its own endpoint/calibration/per-hand masks.
"""
import numpy as np

GOAL_LABEL_POLICY = 'stage1-goal-observation-v2'
LEGACY_POLICY = 'legacy-source-frame-validity-v1'
NEXT_ACTION_POLICY = 'genrobot-observation-without-next-action-v1'
GENROBOT_CAMERA_FLAGS = {
    'observation.images.cam_high': 'raw.robot0.sensor.camera2.video_alignment_valid',
    'observation.images.cam_left_wrist': 'raw.robot1.sensor.camera0.video_alignment_valid',
    'observation.images.cam_right_wrist': 'raw.robot2.sensor.camera0.video_alignment_valid',
}


def stage1_goal_validity(columns, n, *, family, info):
    from openwam.dataloader.native_stage2_io import source_frame_validity, _valid_flags
    metadata = info.get('starvla_metadata', {})
    action = metadata.get('action', {}) if isinstance(metadata, dict) else {}
    action = action if isinstance(action, dict) else {}
    verified = (family == 'Genrobot' and isinstance(metadata, dict)
                and metadata.get('source') == 'genrobot'
                and action.get('semantic') == 'absolute_next_step'
                and type(action.get('frame_offset')) is int and action['frame_offset'] == 1
                and action.get('last_frame_policy') == 'copy_last_available_masked'
                and 'canonical.frame_valid' in columns)
    if not verified:
        return source_frame_validity(columns, n), LEGACY_POLICY
    valid = np.ones(n, dtype=bool)
    for name in ('canonical.frame_valid', 'source.frame_validity'):
        if name in columns:
            flags = np.asarray(columns[name]).reshape(n, -1)
            valid &= _valid_flags(flags).all(-1)
    return valid, NEXT_ACTION_POLICY


def goal_camera_validity(row, cameras, *, observation_valid, policy):
    """Per-target camera alignment, in addition to successful exact decode."""
    from openwam.dataloader.native_stage2_io import _valid_flags
    result = []
    for camera in cameras:
        valid = bool(observation_valid and camera is not None)
        key = GENROBOT_CAMERA_FLAGS.get(camera) if policy == NEXT_ACTION_POLICY else None
        if valid and key in row:
            flags = np.asarray(row[key]).reshape(-1)
            valid = flags.size == 1 and bool(_valid_flags(flags).all())
        result.append(valid)
    return result
