"""Evidence-based endpoints for reports only; never changes training masks.

A named URDF gripper_center is not necessarily a contact TCP. A scalar tool
length does not establish task-frame orientation. Scope: the three audited
example profiles; these policies must not be generalized across embodiments.
"""
from pathlib import Path

import numpy as np

AGIBOT_VISUAL_TCP_X_M = .200
AGIBOT_VISUAL_TASKS = {'task_362', 'task_365', 'task_410', 'task_512'}


def apply_endpoint_policy(name, arrays, metadata):
    """Update report-local arrays/metadata in place, idempotently from source EEF."""
    if name not in ('agibotworld', 'robodojo', 'robotwin'):
        return
    eef = arrays['eef_world']
    metadata['endpoint_audit_version'] = '2026-10-08'
    metadata['origin_diagnostics'] = []
    if name in ('robodojo', 'robotwin'):
        transform = np.eye(4)
        transform[0, 3] = .12
        arrays['tcp_world'] = eef @ transform
        arrays['tcp_valid'] = arrays['eef_valid'].copy()
        arrays['tcp_world'][~arrays['tcp_valid']] = np.nan
        metadata.update(tcp_status='user_defined_tcp_preview', tcp_orientation_verified=False,
                        tcp_orientation_source='User-defined same wrist/flange axes; identity rotation',
                        acceptance_status='user_visual_accepted',
                        tcp_fixed_transform=transform.tolist(),
                        endpoint='User review definition: TCP = wrist/flange pose @ TransX(+0.12m), same orientation; left and right arms use their own current local X axis.',
                        limitations='User visually confirmed this +0.12m local-X, same-orientation definition. Both source EEF and joint/URDF FK use the same transform. Visual acceptance is not independent physical calibration; training reader configuration unchanged.')
    else:
        arrays['tcp_world'] = np.full_like(eef, np.nan)
        arrays['tcp_valid'] = np.zeros_like(arrays['eef_valid'], dtype=bool)
        metadata['tcp_orientation_verified'] = False
        if name == 'agibotworld':
            audit = metadata.get('source_episode_audit', {})
            quality = audit.get('head_extrinsic_audit_classification',
                                metadata.get('head_calibration_quality', 'unknown'))
            base_quality = audit.get('head_extrinsic_base_fk_classification', 'unknown')
            repair = audit.get('head_extrinsic_repair_status', 'unknown')
            metadata.update(tcp_status='unresolved_contact_tcp',
                            endpoint='Source EEF = wrist flange. Prior +100mm audit-URDF gripper_center is NOT a verified contact TCP and is retired from TCP overlays.',
                            limitations=f'Selected episode head audit: {quality}; base-FK audit: {base_quality}; repair: {repair}. Source audit classification is not independent camera calibration. Audit URDF omits 28mm camera mount; visual URDF also changes right tool orientation. No verified contact TCP recovered. Named centers appear only in origin diagnostics.',
                            head_calibration_quality=quality,
                            origin_diagnostics=[
                                {'label': 'Audit URDF named center (NOT TCP)', 'offset_local_m': [.1, 0., 0.], 'color_bgr': [60, 80, 245]},
                                {'label': 'Visual URDF center + mount (NOT TCP)', 'offset_local_m': [.128, 0., 0.], 'color_bgr': [60, 225, 245]},
                            ])
            dataset = Path(metadata.get('dataset_dir', ''))
            if (dataset.name == 'a2d_gripper__dual_gripper__base__unmodified'
                    and dataset.parent.name in AGIBOT_VISUAL_TASKS):
                transform = np.eye(4)
                transform[0, 3] = AGIBOT_VISUAL_TCP_X_M
                arrays['tcp_world'] = eef @ transform
                arrays['tcp_valid'] = arrays['eef_valid'].copy()
                arrays['tcp_world'][~arrays['tcp_valid']] = np.nan
                metadata.update(
                    tcp_status='visual_tcp_preview',
                    acceptance_status='pending_user_visual_review',
                    tcp_fixed_transform=transform.tolist(),
                    tcp_orientation_source='Visualization convention: retain canonical wrist axes; no independent tool-axis calibration',
                    tcp_training_supervision_enabled=False,
                    endpoint='Report-only multi-view preview: canonical wrist @ TransX(+0.200m), same rotation in all views. Previous 245mm overshoots wrist images. 200mm reduces this error; left/right image-based optimal lengths disagree, so this is not a calibrated physical TCP or a complete camera correction.',
                    limitations=f'User authorized manual visual adjustment. This is a fixed virtual point, not the official gripper_center link or independently calibrated physical TCP. Actual pad center can depend on opening; camera residuals and occlusion remain. Head audit: {quality}/{base_quality}, repair: {repair}. Training reader and pose supervision unchanged.')
