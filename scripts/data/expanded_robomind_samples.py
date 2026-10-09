"""Bounded, read-only RoboMIND examples with explicit missing-calibration masks.

Run with the report environment: python3 -m scripts.data.expanded_robomind_samples
Only the first episode and requested consecutive window are read. No source
dataset, training reader, or shared report renderer is modified.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from scripts.data.urdf_fk_audit import URDFModel, eef20_to_matrix
from scripts.data.video_projection_samples import decode_sequence

ROOT = Path(__file__).resolve().parents[2]
RELEASE = Path('/mnt/data/cpfs/b5big/pretrain_data_release/pretrain_datasetsV0_2')
URDFS = Path('/mnt/data/cpfs/b5big/pretrain_urdfs')
FACTORY = Path('/mnt/workspace/mark/opensource_data_factory')
ARX5 = URDFS / 'robotwin/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf'
INTRINSIC_SOURCE = FACTORY / 'sources/robomind_v1/visualize_robomind_v1_eef_projection.py'
SPECS = {
    'robomind_v1_agilex': dict(
        name='RoboMIND v1 · AgileX', group='robomind_v1', domain='h5_agilex_3rgb',
        repo='robomind_v1/benchmark1_0_release/h5_agilex_3rgb/10_packplate',
        cameras=('cam_front', 'cam_left_wrist', 'cam_right_wrist'), dof=6,
        state_starts=(0, 7)),
    'robomind_v1_fr3': dict(
        name='RoboMIND v1 · dual Franka FR3', group='robomind_v1', domain='h5_franka_fr3_dual',
        repo='robomind_v1/benchmark1_1_release/h5_franka_fr3_dual/both_pour_water',
        cameras=('cam_front', 'cam_left', 'cam_right'), dof=7,
        state_starts=(0, 8)),
    'robomind_v2_agilex': dict(
        name='RoboMIND v2 · AgileX', group='robomind_v2', domain=None,
        repo='robomind_v2/arrange_blocks_and_place_orange_in_center_with_arms/agilex__source_native_joint',
        cameras=('camera_front', 'camera_left', 'camera_right'), dof=6,
        state_starts=(0, 6)),
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def read_static_intrinsics():
    """Read the literal calibration table without executing factory code."""
    tree = ast.parse(INTRINSIC_SOURCE.read_text())
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == 'INTRINSICS':
            return ast.literal_eval(node.value)
    raise ValueError('Factory static INTRINSICS table missing')


def load_first_window(dataset, anchor, count):
    info = json.loads((dataset / 'meta/info.json').read_text())
    manifest_path = sorted((dataset / 'meta/episodes').rglob('*.parquet'))[0]
    episodes = pq.read_table(manifest_path).to_pandas()
    episode = episodes.loc[episodes.episode_index == 0].iloc[0]
    if int(episode.dataset_from_index) != 0 or int(episode.length) < anchor + count:
        raise ValueError('Expected first episode with enough consecutive rows')
    data_path = dataset / info['data_path'].format(
        chunk_index=int(episode['data/chunk_index']), file_index=int(episode['data/file_index']))
    columns = ['frame_index', 'episode_index', 'timestamp', 'observation.state']
    columns += [k for k in info['features'] if k == 'observation.states.eef' or
                k.startswith('raw.puppet.end_effector_') and k.endswith('_pose_align')]
    # Reading one bounded batch avoids loading all trajectories in a shared shard.
    batch = next(pq.ParquetFile(data_path).iter_batches(batch_size=anchor + count, columns=columns))
    window = batch.to_pandas().iloc[anchor:anchor + count]
    indices = window.frame_index.to_numpy(dtype=np.int64)
    np.testing.assert_array_equal(indices, np.arange(anchor, anchor + count))
    if set(window.episode_index) != {0}:
        raise ValueError('Physical row/episode alignment failed')
    np.testing.assert_allclose(np.diff(window.timestamp), 1 / info['fps'], atol=2e-5)
    return info, episode, window, data_path


def pose_arrays(n):
    return np.full((n, 2, 4, 4), np.nan), np.zeros((n, 2), bool)


def mesh_tip_candidate(model):
    """Geometric hypothesis from actual distal finger surfaces, not a claimed TCP.

    The two prismatic axes oppose along Y. Their midpoint is aperture-invariant
    only under symmetric travel, so no unsupported source-gripper-to-meters
    conversion is needed. Orientation is inherited from the flange by definition.
    """
    surfaces = []
    for number in (7, 8):
        joint = model.joints[f'fl_joint{number}']
        link = next(x for x in model.xml.findall('link') if x.get('name') == f'fl_link{number}')
        collision = link.find('collision')
        mesh = collision.find('geometry/mesh')
        path = model.path.parent / mesh.get('filename')
        data = path.read_bytes()
        triangles = struct.unpack_from('<I', data, 80)[0]
        if len(data) != 84 + 50 * triangles:
            raise ValueError(f'Expected binary STL: {path}')
        vertices = np.array([struct.unpack_from('<12f', data, 84 + i * 50)[3:]
                             for i in range(triangles)]).reshape(-1, 3)
        # This mesh has identity URDF collision/joint rotation and opposing Y axes.
        # Check that geometry before assigning the explicit +X distal-face rule.
        for element in (collision.find('origin'), joint.find('origin')):
            if not np.allclose(np.fromstring(element.get('rpy', '0 0 0'), sep=' '), 0):
                raise ValueError('Tip candidate requires identity local mesh/joint rotation')
        scale = np.fromstring(mesh.get('scale', '1 1 1'), sep=' ')
        vertices *= scale
        vertices += np.fromstring(collision.find('origin').get('xyz', '0 0 0'), sep=' ')
        tip = vertices[vertices[:, 0] >= vertices[:, 0].max() - 1e-5]
        center = (tip.min(axis=0) + tip.max(axis=0)) / 2
        center += np.fromstring(joint.find('origin').get('xyz'), sep=' ')
        surfaces.append(dict(link=f'fl_link{number}', mesh=str(path),
            mesh_sha256=hashlib.sha256(data).hexdigest(), mesh_bounds_m=[vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
            distal_face_center_in_flange_m=center.tolist(),
            axis=np.fromstring(joint.find('axis').get('xyz'), sep=' ').tolist()))
    np.testing.assert_allclose(np.sum([s['axis'] for s in surfaces], axis=0), 0)
    offset = np.mean([s['distal_face_center_in_flange_m'] for s in surfaces], axis=0)
    transform = np.eye(4)
    transform[:3, 3] = offset
    return transform, dict(status='mesh_geometric_candidate_not_calibrated_tcp',
        definition='Midpoint of both collision-mesh distal (+X maximum, tolerance 10 micrometers) face bounding-box centers, transformed through finger joint origins; flange orientation inherited.',
        offset_in_flange_m=offset.tolist(), flange_to_candidate=transform.tolist(), meshes=surfaces,
        assumptions=['The recorded gripper matches the ARX5 model.', 'Both fingers travel symmetrically along opposing Y axes.',
                     'Distal mesh face midpoint is a geometric visualization convention, not a manufacturer or measured physical TCP.'],
        physical_tcp_verified=False, independent_ground_truth=False)


def fk_tracks(spec, window, dataset):
    n = len(window)
    state = np.stack(window['observation.state']).astype(float)
    arrays = dict(frame_indices=window.frame_index.to_numpy(dtype=np.int64),
                  fk_joint=np.full((n, 14), np.nan), fk_joint_valid=np.zeros((n, 14), bool))
    for key, mask in [('fk_eef_world', 'fk_eef_valid'), ('fk_tcp_world', 'fk_tcp_valid'),
                      ('fk_local_pose', 'fk_local_valid'), ('label_local_pose', 'label_local_valid')]:
        arrays[key], arrays[mask] = pose_arrays(n)
    for side, start in enumerate(spec['state_starts']):
        q = state[:, start:start + spec['dof']]
        valid = np.isfinite(q) & (np.abs(q) <= 4 * np.pi)
        arrays['fk_joint'][:, side * 7:side * 7 + spec['dof']] = np.where(valid, q, np.nan)
        arrays['fk_joint_valid'][:, side * 7:side * 7 + spec['dof']] = valid
    limitations = [
        'No independently established world-from-arm-base transform; all world FK poses remain NaN/masked.',
        'No independent Cartesian ground truth established; label_local_valid is false. Joint-derived FK is never compared to itself.',
        'No calibrated flange-to-physical-TCP definition for this recording; TCP remains NaN/masked.',
    ]
    meta = dict(name=spec['name'], dataset=spec['group'], status='unavailable',
                local_frame='unavailable', world_frame_status='unavailable_unknown_arm_installation',
                independent_world_calibration=False, eef_is_fk_derived=True, tcp_is_fk_derived=False,
                source_metadata=dict(dataset_dir=str(dataset), episode_index=0),
                joint_units='radian', joint_layout='left 7 arm slots; right 7 arm slots; sixth-axis arms leave slots 6 and 13 invalid',
                missing_frame_indices=[], limitations=limitations, tcp_status='unavailable_unverified_physical_endpoint',
                fk_error_status='unavailable: no independent GT and no established shared Cartesian frame',
                tcp_physical_definition_source=None)
    if spec['dof'] == 6:
        model = URDFModel(ARX5)
        candidate_transform, candidate_meta = mesh_tip_candidate(model)
        arrays['fk_tcp_candidate_local'], arrays['fk_tcp_candidate_local_valid'] = pose_arrays(n)
        chains = {}
        for side, prefix in enumerate(('fl', 'fr')):
            valid = arrays['fk_joint_valid'][:, side * 7:side * 7 + 6].all(axis=1)
            values = {f'{prefix}_joint{j + 1}': arrays['fk_joint'][valid, side * 7 + j] for j in range(6)}
            if valid.any():
                arrays['fk_local_pose'][valid, side] = model.fk(values, f'{prefix}_link6', f'{prefix}_base_link')
                arrays['fk_local_valid'][valid, side] = True
                arrays['fk_tcp_candidate_local'][valid, side] = arrays['fk_local_pose'][valid, side] @ candidate_transform
                arrays['fk_tcp_candidate_local_valid'][valid, side] = True
            chains[prefix] = {
                'base_link': f'{prefix}_base_link', 'tip_link': f'{prefix}_link6',
                'joints': [j.get('name') for j in model.chain(f'{prefix}_link6', f'{prefix}_base_link')],
                'tip_children': [dict(name=j.get('name'), type=j.get('type'), child=j.find('child').get('link'),
                                     origin=j.find('origin').attrib)
                                 for j in model.joints.values() if j.find('parent').get('link') == f'{prefix}_link6'],
            }
        meta.update(status='local_fk_only', local_frame='per-arm fl_base_link / fr_base_link (different origins; no shared world)',
                    urdf_path=str(ARX5), urdf_sha256=model.sha256, urdf_read_success=True,
                    urdf_chains=chains, endpoint='URDF fl_link6 / fr_link6 flange origins, not physical tool tips',
                    tcp_fixed_chain='No fixed descendant from link6 to a TCP. Only joint7 and joint8 prismatic finger links are present.',
                    tcp_candidate=candidate_meta,
                    urdf_association_evidence=str(FACTORY / 'sources/robomind_v2/convert_robomind_v2_stage3_canonical.py'))
        if spec['group'] == 'robomind_v1':
            limitations.append('ARX5 is a candidate model by AgileX embodiment and v2 factory association; v1 model identity and joint zero offsets are not independently calibrated. Local FK is diagnostic only.')
        limitations.append('URDF footprint mounting geometry is not established for these recordings and is deliberately not applied.')
    else:
        # Actually parse the available Franka model, but reject a Panda as a FR3 substitute.
        panda = URDFS / 'robotwin/embodiments/franka-panda/panda.urdf'
        candidate = URDFModel(panda)
        matched_fr3 = sorted(str(p) for p in URDFS.rglob('*.urdf') if 'fr3' in str(p).lower())
        meta.update(urdf_read_success=True, attempted_urdf_path=str(panda),
                    attempted_urdf_sha256=candidate.sha256,
                    attempted_urdf_robot_name=candidate.xml.get('name'),
                    candidate_rejected_reason='Available model is Franka Panda, not the recorded Franka FR3; cannot claim exact FR3 FK.',
                    matching_fr3_urdfs=matched_fr3,
                    urdf_search_roots=[str(URDFS), str(FACTORY / 'sources/robomind_v1'), str(FACTORY / 'data_curation/fk_pipeline')])
        limitations.append('No matching FR3 URDF found in supplied URDF collection or RoboMIND factory profile. Panda model parsed and rejected; FR3 joint trajectory is preserved, FK masked.')
    # Keep native measurements inspectable, without mislabeling their frame as world/GT.
    if 'observation.states.eef' in window:
        arrays['source_native_eef_pose'] = eef20_to_matrix(np.stack(window['observation.states.eef']))
        meta['source_eef_status'] = 'Present in release but source_native_unknown coordinate frame and unverified endpoint; not independent GT.'
    else:
        for side in ('left', 'right'):
            key = f'raw.puppet.end_effector_{side}_pose_align'
            if key in window:
                arrays[f'source_native_{side}_raw_pose'] = np.stack(window[key])
        meta['source_eef_status'] = 'Raw source-native pose columns retained without interpreting unvalidated units/orientation/world frame.'
    return arrays, meta


def extract_one(sample_id, output, anchor=30, count=180):
    spec = SPECS[sample_id]
    dataset = RELEASE / spec['repo']
    info, episode, window, data_path = load_first_window(dataset, anchor, count)
    n = len(window)
    indices = window.frame_index.to_numpy(dtype=np.int64)
    root = output / sample_id
    fk, fk_meta = fk_tracks(spec, window, dataset)
    fk_dir = root / 'fk_tracks' / sample_id
    fk_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(fk_dir / 'arrays.npz', **fk)
    write_json(fk_dir / 'metadata.json', fk_meta)
    static_k = read_static_intrinsics().get(spec['domain'], {})
    views = {}
    for view, camera in zip(('head', 'left', 'right'), spec['cameras']):
        key = f'observation.images.{camera}'
        shape = info['features'][key]['shape']
        native_h, native_w = shape[:2]
        path = dataset / info['video_path'].format(video_key=key,
            chunk_index=int(episode[f'videos/{key}/chunk_index']), file_index=int(episode[f'videos/{key}/file_index']))
        target = root / 'video_samples' / sample_id if view == 'head' else root / 'wrist_samples' / sample_id / view
        target.mkdir(parents=True, exist_ok=True)
        raw_camera = camera.replace('cam_', 'camera_')
        k = np.asarray(static_k[raw_camera], dtype=float) if raw_camera in static_k else np.full((3, 3), np.nan)
        arrays = dict(frame_indices=indices, K=np.broadcast_to(k, (n, 3, 3)).copy(),
                      world_to_camera=np.full((n, 4, 4), np.nan), camera_valid=np.zeros(n, bool),
                      distortion=np.empty((n, 0)))
        meta = dict(name=spec['name'], dataset_family=sample_id, dataset_dir=str(dataset),
                    episode=0, episode_index=0, episode_length=int(episode.length),
                    frame_indices=indices.tolist(), anchor=anchor, horizon=n - 1,
                    camera_key=key, model='pinhole', image_width=native_w, image_height=native_h,
                    nativewh=[native_w, native_h], video_path=str(path),
                    video_episode_from_timestamp=float(episode[f'videos/{key}/from_timestamp']),
                    intrinsic_source=str(INTRINSIC_SOURCE) if np.isfinite(k).all() else None,
                    intrinsic_status='static_factory_table_not_per_recording_verified' if np.isfinite(k).all() else 'unavailable',
                    extrinsic_status='unavailable', video_available=True, projection_available=False,
                    camera_missing_reason=('World-to-camera extrinsics are unavailable; world/arm-base alignment is unverified.'
                                           if np.isfinite(k).all() else
                                           'Camera intrinsics and world-to-camera extrinsics are unavailable; world/arm-base alignment is unverified.'),
                    status='video_only', reason='Real source video available; world-to-camera extrinsics unavailable, so projection is masked.',
                    tcp_status=fk_meta['tcp_status'], endpoint=fk_meta.get('endpoint', 'Unverified native EEF endpoint'),
                    eef_is_fk_derived=False, tcp_is_fk_derived=False,
                    pose_source='No world pose exported: native EEF frame unknown; FK remains arm-local diagnostic.',
                    origin_diagnostics=[], clip_label=f'{spec["name"]} · episode 0 · frames {anchor}–{anchor + count - 1}',
                    limitations=' '.join(fk_meta['limitations']), requested_frame_count=count,
                    data_path=str(data_path), source_info=str(dataset / 'meta/info.json'),
                    source_metadata=str(dataset / 'meta/source_stage2.json'),
                    task=str(episode['tasks'].tolist()),
                    selection_policy='First listed physical repository, episode 0, fixed consecutive window; no search for favorable FK error.',
                    independent_world_calibration=False)
        if view == 'head':
            arrays['eef_world'], arrays['eef_valid'] = pose_arrays(n)
            arrays['tcp_world'], arrays['tcp_valid'] = pose_arrays(n)
        else:
            meta['side'] = view
            meta['camera_role'] = f'{view} source view; wrist mounting unverified'
            meta['camera_mount_status'] = 'wrist_mounting_unverified'
            meta['camera_original_key'] = raw_camera
            meta['camera_role_evidence'] = (
                'Released meta/source_stage2.json maps the original HDF5 RGB camera key to this video key; '
                'a wrist token in the source name is not independent evidence of physical wrist mounting.'
                if spec['group'] == 'robomind_v1' else
                'Released meta/source_field_map.json preserves camera_left/camera_right from HDF5 '
                'camera_observations/color_images; no mounting transform or verified wrist role is provided.')
        meta.update(decode_sequence(meta, target))
        np.savez_compressed(target / 'arrays.npz', **arrays)
        write_json(target / 'metadata.json', meta)
        if view != 'head':
            views[view] = meta
    write_json(root / 'wrist_samples' / sample_id / 'summary.json', dict(
        name=spec['name'], dataset_dir=str(dataset), episode_index=0, views=views))
    write_json(root / 'entry.json', dict(id=sample_id, title=spec['name'], family=sample_id,
        root=f'expanded_sources/{sample_id}', source_group=spec['group'],
        reason='Real-robot embodiment from source list; actual continuous videos plus explicit calibration/TCP/FK availability.'))
    print(json.dumps(dict(id=sample_id, frames=n, videos=3, fk_status=fk_meta['status'], tcp_status=fk_meta['tcp_status']), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'outputs/fk_consistency/expanded_sources')
    parser.add_argument('--names', nargs='+', choices=list(SPECS), default=list(SPECS))
    parser.add_argument('--anchor', type=int, default=30)
    parser.add_argument('--count', type=int, default=180)
    args = parser.parse_args()
    for name in args.names:
        extract_one(name, args.output_root, args.anchor, args.count)
