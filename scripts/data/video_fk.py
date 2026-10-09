"""Frame-exact, read-only FK tracks for videos of previously audited episodes.

World FK availability is independent of GT availability. All invalid entries are
NaN and carry a boolean mask; missing frames are never interpolated. AgiBot's
unobserved torso prevents world projection, so its useful comparison is the
right wrist expressed in the instantaneous left-wrist frame, computed separately
for GT and FK. That cancels a common torso transform without fitting GT to FK.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _indices(values, label):
    values = np.asarray(values)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f'{label}: expected a one-dimensional integer frame_indices array')
    if len(np.unique(values)) != len(values):
        raise ValueError(f'{label}: duplicate frame_indices')
    return values.astype(np.int64)


def _pose_valid(value):
    finite = np.isfinite(value).all(axis=(-1, -2))
    rotation = np.where(finite[..., None, None], value[..., :3, :3], np.eye(3))
    return (finite
            & np.isclose(value[..., 3, :], [0, 0, 0, 1], atol=1e-5).all(-1)
            & np.isclose(np.swapaxes(rotation, -1, -2) @ rotation,
                         np.eye(3), atol=1e-4).all(axis=(-1, -2))
            & np.isclose(np.linalg.det(rotation), 1, atol=1e-4))


def _world_from_robot(indices, sample_arrays, sample_metadata, source):
    """Read declared per-row robot_to_env, never solve a transform from EEF GT."""
    for container in (sample_arrays, sample_metadata):
        for key in ('world_from_robot', 'robot_to_env', 'states.robot_to_env_pose'):
            if key in container:
                values = np.asarray(container[key], dtype=float)
                break
        else:
            continue
        if values.shape == (len(indices), 7):
            values = _pose7(values)
        if values.shape != (len(indices), 4, 4):
            raise ValueError(f'{key} must contain one robot-to-world transform per requested frame')
        return values, f'sample {key}'
    directory = source.get('dataset_dir')
    if not directory or not (Path(directory) / 'data').exists():
        return None, 'robot_to_env unavailable; FK TCP world projection masked'
    import pyarrow.dataset as ds

    dataset = ds.dataset(Path(directory) / 'data', format='parquet')
    key = 'states.robot_to_env_pose'
    if key not in dataset.schema.names:
        return None, 'source robot_to_env column unavailable; FK TCP world projection masked'
    table = dataset.to_table(columns=['frame_index', key], filter=(
        (ds.field('episode_index') == int(source.get('episode_index', 0)))
        & ds.field('frame_index').isin(indices.tolist())))
    rows = table.to_pylist()
    source_indices = _indices(np.asarray([row['frame_index'] for row in rows], dtype=np.int64),
                              'source robot_to_env')
    mapping = {int(frame): row[key] for frame, row in zip(source_indices, rows)}
    values = np.asarray([mapping.get(int(i), [np.nan] * 7) for i in indices], dtype=float)
    return _pose7(values), f'{directory}/data: {key}, exact episode/frame join'


def _pose7(values):
    from scipy.spatial.transform import Rotation

    result = np.full((len(values), 4, 4), np.nan)
    valid = np.isfinite(values).all(-1) & (np.linalg.norm(values[:, 3:], axis=-1) > 1e-10)
    result[valid] = np.eye(4)
    result[valid, :3, 3] = values[valid, :3]
    if valid.any():
        result[valid, :3, :3] = Rotation.from_quat(values[valid][:, [4, 5, 6, 3]]).as_matrix()
    return result


def load_fk_tracks(name: str, frame_indices: np.ndarray, sample_arrays: dict,
                   sample_metadata: dict, root=Path('outputs/fk_consistency')) -> tuple[dict, dict]:
    """Return masked FK, audited joints, local GT/FK, and provenance metadata.

    Pose arrays have shape (N,2,4,4), pose masks (N,2), and fk_joint/
    fk_joint_valid (N,14). Six-DOF arms leave slots 6 and 13 invalid.
    Inputs with mismatched episode, dataset or frame identity fail explicitly.
    A missing audit or individual frame yields invalid tracks with a reason.
    """
    indices = _indices(frame_indices, 'requested')
    for container in (sample_arrays, sample_metadata):
        if 'frame_indices' in container and not np.array_equal(container['frame_indices'], indices):
            raise ValueError('sample frame_indices do not match requested frame_indices')
    n = len(indices)
    arrays = {'frame_indices': indices.copy(),
              'fk_joint': np.full((n, 14), np.nan),
              'fk_joint_valid': np.zeros((n, 14), bool)}
    for pose, mask in (('fk_eef_world', 'fk_eef_valid'), ('fk_tcp_world', 'fk_tcp_valid'),
                       ('fk_local_pose', 'fk_local_valid'), ('label_local_pose', 'label_local_valid')):
        arrays[pose] = np.full((n, 2, 4, 4), np.nan)
        arrays[mask] = np.zeros((n, 2), bool)
    meta = dict(dataset=name, status='unavailable', local_frame='unavailable',
                world_frame_status='unavailable', independent_world_calibration=False,
                missing_frame_indices=indices.tolist(), limitations=[])
    bundle = Path(root)/'fk_tracks'/name
    if (bundle/'arrays.npz').exists():
        evidence = json.loads((bundle/'metadata.json').read_text())
        source = evidence['source_metadata']
        if int(source['episode_index']) != int(sample_metadata.get('episode_index', sample_metadata.get('episode', 0))):
            raise ValueError('FK bundle episode differs from sampled episode')
        if Path(source['dataset_dir']).resolve() != Path(sample_metadata['dataset_dir']).resolve():
            raise ValueError('FK bundle dataset differs from sampled dataset')
        with np.load(bundle/'arrays.npz', allow_pickle=False) as stored:
            values = dict(stored)
        if not np.array_equal(_indices(values['frame_indices'], 'FK bundle'), indices):
            raise ValueError('FK bundle frame indices must exactly match sample order')
        for key, template in arrays.items():
            if key not in values or values[key].shape != template.shape:
                raise ValueError(f'FK bundle {key}: expected shape {template.shape}')
            arrays[key] = values[key].astype(template.dtype, copy=True)
        for pose, mask in [('fk_eef_world', 'fk_eef_valid'), ('fk_tcp_world', 'fk_tcp_valid'),
                           ('fk_local_pose', 'fk_local_valid'), ('label_local_pose', 'label_local_valid')]:
            arrays[mask] &= _pose_valid(arrays[pose])
            arrays[pose][~arrays[mask]] = np.nan
        arrays['fk_joint_valid'] &= np.isfinite(arrays['fk_joint'])
        arrays['fk_joint'][~arrays['fk_joint_valid']] = np.nan
        if 'fk_tcp_candidate_local' in values:
            pose, mask = values['fk_tcp_candidate_local'], values['fk_tcp_candidate_local_valid']
            if pose.shape != (n, 2, 4, 4) or mask.shape != (n, 2):
                raise ValueError('FK local TCP candidate shape differs from sample')
            mask = mask.astype(bool) & _pose_valid(pose)
            arrays['fk_tcp_candidate_local'] = np.where(mask[..., None, None], pose, np.nan)
            arrays['fk_tcp_candidate_local_valid'] = mask
        meta.update(evidence)
        meta.update(missing_frame_indices=[], audit_arrays=str(bundle/'arrays.npz'),
                    audit_metadata=str(bundle/'metadata.json'))
        return arrays, meta
    if name in ('genrobot', 'lightwheel', 'mecka'):
        meta['limitations'].append('No matched robot joint labels and URDF for this handheld/human source; robot FK unavailable.')
        return arrays, meta
    if name not in ('interndata', 'agibotworld', 'robodojo', 'robotwin'):
        raise ValueError(f'Unknown FK dataset: {name}')
    folder = Path(root) / name
    if not (folder / 'arrays.npz').exists():
        meta['limitations'].append(f'Audited FK arrays unavailable: {folder / "arrays.npz"}')
        return arrays, meta
    metadata_path = folder / 'metadata.json'
    if metadata_path.exists():
        source = json.loads(metadata_path.read_text())
    elif (folder / 'report_metrics.json').exists():
        report = json.loads((folder / 'report_metrics.json').read_text())
        source = report.get('source_metadata', report)
        metadata_path = folder / 'report_metrics.json'
    else:
        meta['limitations'].append('FK provenance metadata unavailable; tracks masked.')
        return arrays, meta
    episode = int(sample_metadata.get('episode', sample_metadata.get('episode_index', 0)))
    if episode != int(source.get('episode_index', 0)):
        raise ValueError(f'{name}: sample episode does not match audited episode')
    if sample_metadata.get('dataset_dir') and source.get('dataset_dir'):
        if Path(sample_metadata['dataset_dir']).resolve() != Path(source['dataset_dir']).resolve():
            raise ValueError(f'{name}: sample dataset_dir does not match audited dataset')
    with np.load(folder / 'arrays.npz', allow_pickle=False) as stored:
        audit = {key: stored[key] for key in stored.files}
    audit_indices = _indices(audit['frame_indices'], 'audit')
    lookup = {int(frame): i for i, frame in enumerate(audit_indices)}
    present = np.array([int(i) in lookup for i in indices], dtype=bool)
    rows = np.array([lookup.get(int(i), 0) for i in indices], dtype=int)
    meta.update(status='available' if present.all() else 'partial',
                missing_frame_indices=indices[~present].tolist(),
                audit_arrays=str(folder / 'arrays.npz'), audit_metadata=str(metadata_path),
                episode_index=episode, source_metadata=source,
                limitations=list(source.get('limitations', [])),
                joint_units='radian', joint_layout='left 7 slots, right 7 slots; invalid slots masked')

    def matched(key, shape):
        result = np.full((n, *shape), np.nan)
        if key in audit:
            value = np.asarray(audit[key])
            if value.shape != (len(audit_indices), *shape):
                raise ValueError(f'{key}: shape does not match audit frame_indices')
            result[present] = value[rows[present]]
        return result

    def put(pose, mask, value, valid=None):
        validity = _pose_valid(value) & present[:, None]
        if valid is not None:
            validity &= valid
        value = value.copy()
        value[~validity] = np.nan
        arrays[pose], arrays[mask] = value, validity

    joints = matched('joint', (14,))
    arrays['fk_joint'] = joints
    arrays['fk_joint_valid'] = np.isfinite(joints)
    if name in ('robodojo', 'robotwin'):
        arrays['fk_joint_valid'][:, [6, 13]] = False
        arrays['fk_joint'][:, [6, 13]] = np.nan
    fk = matched('fk_pose', (2, 4, 4))
    label = matched('label_pose', (2, 4, 4))
    explicit = matched('fk_valid', (2,)) == 1 if 'fk_valid' in audit else None
    if name == 'agibotworld':
        for value, pose, mask in ((fk, 'fk_local_pose', 'fk_local_valid'),
                                  (label, 'label_local_pose', 'label_local_valid')):
            valid = _pose_valid(value).all(-1) & present
            if pose.startswith('fk') and explicit is not None:
                valid &= explicit.all(-1)
            relative = np.full_like(value, np.nan)
            relative[valid, 0] = np.eye(4)
            relative[valid, 1] = np.linalg.inv(value[valid, 0]) @ value[valid, 1]
            put(pose, mask, relative)
        meta.update(local_frame='current left wrist; right wrist inter-arm pose (left is identity)',
                    world_frame_status='masked: absolute FK assumes zero missing torso joints; not calibrated world',
                    tcp_status='unavailable: no calibrated world FK or verified TCP',
                    local_semantics='GT and FK each use inv(left) @ right; cancels common torso; no GT-to-FK alignment')
        return arrays, meta
    put('fk_local_pose', 'fk_local_valid', fk, explicit)
    put('label_local_pose', 'label_local_valid', label)
    if name == 'interndata':
        put('fk_eef_world', 'fk_eef_valid', matched('canonical_diagnostic_fk_pose', (2, 4, 4)), explicit)
        world, provenance = _world_from_robot(indices, sample_arrays, sample_metadata, source)
        if world is not None:
            put('fk_tcp_world', 'fk_tcp_valid', world[:, None] @ matched('tcp_diagnostic_fk_pose', (2, 4, 4)), explicit)
        else:
            meta['limitations'].append(provenance)
        meta.update(local_frame='arm_base_link; independent joint-to-URDF arm-local FK',
                    world_frame_status='source row 0 diagnostic base/tool transforms; not independent calibration',
                    tcp_status='source row 0 diagnostic TCP; native tool axes, robot_to_env per row',
                    robot_to_env_provenance=provenance)
    else:
        put('fk_eef_world', 'fk_eef_valid', fk, explicit)
        meta.update(local_frame='world (same audited FK and GT poses)',
                    world_frame_status='declared source world/base transform applied to URDF FK',
                    tcp_status='unavailable: audited wrist/flange only; no verified tool transform')
        tcp_status = sample_metadata.get('tcp_status')
        if (name in ('robotwin', 'robodojo') and tcp_status == 'user_defined_tcp_preview'
                or name == 'robotwin' and tcp_status == 'source_config_derived_tcp'):
            transform = np.asarray(sample_metadata['tcp_fixed_transform'], dtype=float)
            if transform.shape != (4, 4) or not _pose_valid(transform):
                raise ValueError('Derived TCP transform must be a rigid 4x4 matrix')
            put('fk_tcp_world', 'fk_tcp_valid', fk @ transform, explicit)
            meta['tcp_status'] = ('user-defined acceptance preview: wrist/flange FK @ TransX(+0.12m), same orientation'
                                  if tcp_status == 'user_defined_tcp_preview' else
                                  'source-config simulator TCP: link6 @ TransX(0.12m); not independent tool calibration')
            meta['limitations'] = [x for x in meta['limitations'] if 'not an independently calibrated tool TCP' not in x]
            meta['limitations'].append('Current TCP overlay uses the explicitly declared report transform; user-defined preview is awaiting acceptance, not an independent calibration result.')
    return arrays, meta
