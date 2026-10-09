"""Read-only AgiBot episode audit; missing torso state is never inferred from EEF.

The absolute FK shown is explicitly a zero-torso diagnostic. The inter-arm
relative-pose comparison cancels the unobserved common torso transform.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from .urdf_fk_audit import URDFModel
except ImportError:
    from urdf_fk_audit import URDFModel

DATASET = Path('/mnt/data/cpfs/b5big/pretrain_data_release/pretrain_datasetsV0_1/agibotworld/task_512/a2d_gripper__dual_gripper__base__unmodified')
URDF_DIR = Path('/mnt/data/cpfs/b5big/pretrain_urdfs/agibotworld')
AUDIT_DIR = Path('/mnt/data/cpfs/b5/pretrain_data/audits/agibotworld_wrist_eef_fk_v1')
AXIS_REMAP = np.array([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]])


def poses_from_eef(eef: np.ndarray) -> np.ndarray:
    poses = np.broadcast_to(np.eye(4), (len(eef), 2, 4, 4)).copy()
    for arm, offset in enumerate((0, 10)):
        a = eef[:, offset + 3:offset + 6].copy()
        b = eef[:, offset + 6:offset + 9].copy()
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        b -= (a * b).sum(axis=1, keepdims=True) * a
        b /= np.linalg.norm(b, axis=1, keepdims=True)
        poses[:, arm, :3, :3] = np.stack([a, b, np.cross(a, b)], axis=-1)
        poses[:, arm, :3, 3] = eef[:, offset:offset + 3]
    return poses


def errors(a: np.ndarray, b: np.ndarray) -> dict:
    p = np.linalg.norm(a[..., :3, 3] - b[..., :3, 3], axis=-1) * 1000
    r = np.swapaxes(a[..., :3, :3], -1, -2) @ b[..., :3, :3]
    angle = np.degrees(np.arccos(np.clip((np.trace(r, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))
    def summarize(v):
        return {'median': float(np.median(v)), 'p95': float(np.percentile(v, 95)), 'max': float(np.max(v))}
    return {'translation_mm': summarize(p), 'rotation_deg': summarize(angle)}


def run(output_dir: Path, *, dataset=DATASET, episode_index=0) -> dict:
    dataset = Path(dataset)
    info = json.loads((dataset / 'meta/info.json').read_text())
    expected_axes = {'canonical_x': 'source_z', 'canonical_y': '-source_x', 'canonical_z': '-source_y'}
    if info['features']['observation.states.eef'].get('axis_remap') != expected_axes:
        raise ValueError('AgiBot FK audit requires the declared canonical wrist-axis remap')
    ep = pq.read_table(dataset / 'meta/episodes', filters=[('episode_index', '=', episode_index)]).to_pandas().iloc[0]
    parquet = dataset / info['data_path'].format(chunk_index=int(ep['data/chunk_index']), file_index=int(ep['data/file_index']))
    frame = pq.read_table(parquet, filters=[('episode_index', '=', episode_index)], columns=[
        'frame_index', 'timestamp', 'observation.states.joint', 'observation.states.eef',
    ]).to_pandas().sort_values('frame_index')
    assert len(frame) == int(ep['length'])
    state = np.stack(frame['observation.states.joint']).astype(np.float64)
    joint = state[:, [*range(7), *range(8, 15)]]
    label = poses_from_eef(np.stack(frame['observation.states.eef']).astype(np.float64))
    names = [f'idx{20+i}_arm_l_joint{i}' for i in range(1, 8)] + [f'idx{60+i}_arm_r_joint{i}' for i in range(1, 8)]
    values = {name: joint[:, i] for i, name in enumerate(names)}
    # Missing body joints are explicit assumptions, not a silent FK default.
    values.update(idx01_body_joint1=np.zeros(len(joint)), idx02_body_joint2=np.zeros(len(joint)))
    target_links = ['arm_l_end_link', 'arm_r_end_link']
    audit_path = URDF_DIR / 'g1_120s_agibot_beta_audit.urdf'
    original_path = URDF_DIR / 'G1_120s.urdf'
    raw_fk = np.stack([URDFModel(audit_path).fk(values, link, base_link='base_link') for link in target_links], axis=1)
    fk = raw_fk.copy()
    fk[..., :3, :3] = raw_fk[..., :3, :3] @ AXIS_REMAP
    original_fk = np.stack([URDFModel(original_path).fk(values, link, base_link='base_link') for link in target_links], axis=1)
    original_fk[..., :3, :3] = original_fk[..., :3, :3] @ AXIS_REMAP
    reference_label, reference_fk = label[0].copy(), fk[0].copy()
    # A row at index t+1 supplies both GT EEF and matched joint state.
    frame = frame.iloc[1:]
    label, fk, joint, raw_fk, original_fk = (x[1:] for x in (label, fk, joint, raw_fk, original_fk))
    label_relative = np.linalg.inv(label[:, 0]) @ label[:, 1]
    fk_relative = np.linalg.inv(fk[:, 0]) @ fk[:, 1]
    relative_metrics = errors(label_relative, fk_relative)
    metrics = {arm: errors(label[:, i], fk[:, i]) for i, arm in enumerate(('left', 'right'))}
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / 'arrays.npz', frame_indices=frame['frame_index'].to_numpy(),
                        time=frame['timestamp'].to_numpy(), joint=joint, label_pose=label, fk_pose=fk,
                        raw_axis_fk_pose=raw_fk, original_urdf_fk_pose=original_fk,
                        cross_arm_label_pose=label_relative, cross_arm_fk_pose=fk_relative,
                        reference_label_pose=reference_label, reference_fk_pose=reference_fk)
    metadata = {
        'dataset': 'agibotworld', 'dataset_dir': str(dataset), 'episode_index': episode_index,
        'task': str(ep['tasks']), 'frame_count': len(frame), 'source_episode_rows': int(ep['length']),
        'frame_indices': [int(frame['frame_index'].iloc[0]), int(frame['frame_index'].iloc[-1])],
        'urdf_path': str(audit_path), 'urdf_sha256': hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        'original_urdf_path': str(original_path), 'original_urdf_sha256': hashlib.sha256(original_path.read_bytes()).hexdigest(),
        'joint_map': [{'canonical_index': i if i < 7 else i + 1, 'joint14_index': i, 'urdf_joint': name, 'sign': 1, 'offset': 0, 'unit': 'radian'} for i, name in enumerate(names)],
        'coordinate_frame': 'agibotworld_robot_base / URDF base_link', 'target_links': target_links,
        'labels': {'source': 'observation.states.eef rows 1: (state at t+1)', 'pose_reference': 'wrist_flange',
                   'rotation_6d': 'first two rotation columns, column-major',
                   'axis_remap': info['features']['observation.states.eef']['axis_remap'],
                   'axis_remap_matrix': AXIS_REMAP.tolist()},
        'missing_chain_joints': ['idx01_body_joint1', 'idx02_body_joint2'],
        'missing_joint_baseline': {'idx01_body_joint1': 0., 'idx02_body_joint2': 0.},
        'status': 'absolute_fk_unverified_missing_torso_state',
        'fk_pose_description': 'Audit URDF base-root FK with documented wrist-axis remap and explicitly assumed zero torso joints. Diagnostic baseline only.',
        'metrics': metrics, 'interarm_metrics': relative_metrics,
        'raw_axis_metrics': {arm: errors(label[:, i], raw_fk[:, i]) for i, arm in enumerate(('left', 'right'))},
        'original_urdf_metrics': {arm: errors(label[:, i], original_fk[:, i]) for i, arm in enumerate(('left', 'right'))},
        'mapping_evidence': [
            'meta/info.json declares arm indices 0:7 and 8:15, radians and identity joint conversion.',
            'URDF arm joints 1..7 map directly to canonical arm_0..6; both signs +1; independently checked by torso-cancelling inter-arm relative pose.',
            'meta/info.json declares flange origin and x=source_z,y=-source_x,z=-source_y; FK rotation is right-multiplied by that fixed remap.',
            'URDF manifest and parse report designate g1_120s_agibot_beta_audit.urdf as audit model; original G1_120s.urdf is visualization model and raises torso origin by 0.300 m.',
        ],
        'limitations': [
            'Canonical statejoint16 omits two torso joints required for absolute base-root FK; no root or body state exists in release parquet.',
            'Declared Stage2 sources are unavailable at their metadata paths and checked current mounts. No missing joint was estimated from EEF labels.',
            'Absolute diagnostic errors cannot be attributed to arm FK correctness until torso state is recovered.',
            'Inter-arm pose cancels a common torso transform, validating both chains together but not absolute base registration or each arm independently.',
            'No TCP translation is applied; target remains wrist flange. Gripper scalars do not affect flange FK.',
        ],
        'data_path': str(parquet), 'reproduction': f'python scripts/data/fk_audit_agibot.py --dataset {dataset} --episode-index {episode_index} --output-dir {output_dir}',
    }
    # The old raw-ID audit is only established for the original sample. Never
    # borrow its result for another release task or episode.
    if dataset == DATASET and episode_index == 0:
        previous = json.loads((AUDIT_DIR / 'task_results/base/task_512.json').read_text())['episodes'][0]
        metadata['historical_audit'] = {
            'path': str(AUDIT_DIR / 'task_results/base/task_512.json'), 'episode_index': 0,
            'sampled_frames': previous['sampled_frames'],
            'left': previous['arms']['left']['state_base_xyzw'], 'right': previous['arms']['right']['state_base_xyzw']}
    (output_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/fk_consistency/agibotworld'))
    parser.add_argument('--dataset', type=Path, default=DATASET)
    parser.add_argument('--episode-index', type=int, default=0)
    args = parser.parse_args()
    result = run(args.output_dir, dataset=args.dataset, episode_index=args.episode_index)
    print(json.dumps({k: result[k] for k in ('status', 'frame_count', 'metrics', 'interarm_metrics')}, indent=2))
