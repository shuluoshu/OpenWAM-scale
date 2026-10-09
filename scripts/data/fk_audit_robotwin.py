"""Reproduce RobotWin canonical EEF from native next-state joints and declared URDF."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from scripts.data.urdf_fk_audit import (
    URDFModel,
    eef20_to_matrix,
    load_episode,
    pose7_to_matrix,
    pose_errors,
    write_result,
)

PROFILE = Path('configs/dataloader/canonical_eef_robotwin_openloop.yaml')
URDF = Path('/mnt/data/cpfs/b5big/pretrain_urdfs/robotwin/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf')


def run(output_dir):
    cfg, info, ep, frame = load_episode(PROFILE, columns=['observation.state', 'observation.states.eef'])
    model = URDFModel(URDF)
    conversion = info['robotwin_postrain_eef6d_conversion']
    world_from_base = pose7_to_matrix(np.array([*conversion['root_position_xyz'], *conversion['root_quaternion_wxyz']]))
    native = np.stack(frame['observation.state'])
    labels = eef20_to_matrix(np.stack(frame['observation.states.eef']))
    joint_values = {f'{prefix}_joint{i+1}': native[:, offset+i]
                    for prefix, offset in [('fl', 0), ('fr', 7)] for i in range(6)}
    raw_fk = np.stack([model.fk(joint_values, conversion['fk_target_links'][side]) for side in ('left','right')], axis=1)
    fk = world_from_base @ raw_fk
    joint14 = np.zeros((len(frame), 14))
    joint14[:, :6], joint14[:, 7:13] = native[:, :6], native[:, 7:13]
    pos, angle = pose_errors(labels[1:], fk[1:])
    mapping = {f'{prefix}_joint{i+1}': f'observation.state[{offset+i}]'
               for prefix, offset in [('fl', 0), ('fr', 7)] for i in range(6)}
    metadata = {
        'dataset': 'RobotWin Aloha', 'dataset_key': 'robotwin', 'profile': str(PROFILE),
        'dataset_dir': str(cfg.dataset_dir), 'episode_index': int(ep.episode_index), 'num_states': len(frame),
        'num_targets': len(frame)-1, 'target_semantics': 'state[t+1]; absolute native joints before normalization',
        'urdf': str(URDF), 'urdf_sha256': model.sha256, 'joint_map': mapping,
        'coordinate_frame': 'world', 'target_links': conversion['fk_target_links'],
        'labels': 'observation.states.eef: world-frame left/right link6 flange',
        'mapping_evidence': {'conversion_metadata': str(Path(cfg.dataset_dir)/'meta/info.json'),
                             'metadata_key': 'robotwin_postrain_eef6d_conversion',
                             'world_from_base': world_from_base.tolist(),
                             'embodiment_config': str(URDF.parent.parent/'config.yml')},
        'limitations': ['EEF labels were themselves generated using this URDF/FK. Agreement checks reconstruction and mappings, not independent physical calibration.',
                        'Targets are flange link6, not an independently calibrated tool TCP; each arm has six revolute joints.'],
        'status': 'pass' if pos.max() < 1 and angle.max() < 1 else 'fail',
        'position_max_mm': float(pos.max()), 'rotation_max_deg': float(angle.max()),
        'independent_calibration': False,
    }
    return write_result(output_dir, metadata, frame_indices=frame.frame_index.to_numpy()[1:],
                        time=frame.timestamp.to_numpy()[1:], label_pose=labels[1:], fk_pose=fk[1:],
                        raw_fk_pose=raw_fk[1:], joint=joint14[1:],
                        reference_label_pose=labels[0], reference_fk_pose=fk[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/fk_consistency/robotwin'))
    args = parser.parse_args()
    result = run(args.output_dir)
    print(result['status'], result['position_max_mm'], result['rotation_max_deg'])
