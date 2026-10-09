"""Bounded, read-only A1 embodiment extraction with independent local joint FK.

Run python -m scripts.data.expanded_a1_samples. Released source trees are never
modified. World installation and TCP tool transforms inferred from source labels
are explicitly diagnostic; local joint/URDF FK uses no fitted pose corrections.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from scripts.data.urdf_fk_audit import URDFModel, eef20_to_matrix, load_episode, pose7_to_matrix, pose_errors
from scripts.data.video_projection_samples import decode_sequence
from scripts.data.wrist_projection_samples import camera_validity

ROOT = Path(__file__).resolve().parents[2]
DATA = Path('/mnt/data/cpfs/b5big/pretrain_data_release/pretrain_datasetsV0_2/interndata_a1/basic_tasks')
URDFS = Path('/mnt/data/cpfs/b5big/pretrain_urdfs/interndata_a1')
SPECS = {
    'a1_genie1': dict(name='A1 / Genie1', leaf='scan_qrcode_part2/genie1__parallel_gripper', episode=0,
        urdf='G1_120s/G1_120s.urdf', robot='genie1', dof=7),
    'a1_arx_lift2': dict(name='A1 / ARX Lift2', leaf='arrange_the_tableware/lift2__parallel_gripper', episode=0,
        urdf='ARX_Lift2/urdf/lift.urdf', robot='lift2', dof=6),
    'a1_split_aloha': dict(name='A1 / Split Aloha', leaf='collect_the_shoes_language/split_aloha__parallel_gripper', episode=0,
        urdf='AgileX_Split_Aloha_piper100/split_aloha_mid_360_with_piper.urdf', robot='split_aloha', dof=6),
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def error_summary(label, predicted):
    distance, angle = pose_errors(label, predicted)
    return dict(position_mm_median=float(np.median(distance)), position_mm_max=float(distance.max()),
                rotation_deg_median=float(np.median(angle)), rotation_deg_max=float(angle.max()))


def extract_one(identifier, output, anchor=120, count=180, skip_decode=False):
    """Export one source window and explicit provenance; reject incomplete windows."""
    spec = SPECS[identifier]
    dataset = DATA / spec['leaf']
    root = output / identifier
    profile = root / 'source.yaml'
    root.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(dict(dataset_dir=str(dataset))), profile)
    source_meta = json.loads((dataset / 'meta/source_stage2.json').read_text())
    columns = ['states.robot_to_env_pose', 'observation.states.eef']
    for side in ('left', 'right'):
        columns.extend(f'states.{side}_{key}' for key in (
            'joint.position', 'ee_to_robot_pose', f'ee_to_{side}_armbase_pose',
            'tcp_to_robot_pose', f'tcp_to_{side}_armbase_pose'))
    for camera in ('cam_high', 'cam_left_wrist', 'cam_right_wrist'):
        columns.extend(f'observation.states.cameras.{camera}.{field}' for field in ('intrinsic', 'world_to_camera'))
    columns.extend(f'{camera}_camera_to_robot_extrinsics' for camera in ('head', 'hand_left', 'hand_right'))
    _, info, episode, frame = load_episode(profile, episode_index=spec['episode'], columns=columns)
    window = frame.set_index('frame_index').loc[np.arange(anchor, anchor + count)]
    indices = window.index.to_numpy(dtype=int)
    n = len(indices)
    eef = eef20_to_matrix(np.stack(window['observation.states.eef']))
    world_robot = pose7_to_matrix(np.stack(window['states.robot_to_env_pose']))
    tcp = np.full_like(eef, np.nan)
    fk_world = np.full_like(eef, np.nan)
    fk_tcp = np.full_like(eef, np.nan)
    fk_local = np.full_like(eef, np.nan)
    label_local = np.full_like(eef, np.nan)
    joint = np.full((n, 14), np.nan)
    joint_valid = np.zeros((n, 14), dtype=bool)
    model = URDFModel(URDFS / spec['urdf'])
    diagnostics = {}
    chain_evidence = {}
    source_correction = source_meta['starvla_metadata']['eef_axis_correction']['per_arm']
    for side_id, (side, letter) in enumerate((('left', 'l'), ('right', 'r'))):
        raw_ee = pose7_to_matrix(np.stack(window[f'states.{side}_ee_to_robot_pose']))
        raw_tcp = pose7_to_matrix(np.stack(window[f'states.{side}_tcp_to_robot_pose']))
        label = pose7_to_matrix(np.stack(window[f'states.{side}_ee_to_{side}_armbase_pose']))
        local_tcp = pose7_to_matrix(np.stack(window[f'states.{side}_tcp_to_{side}_armbase_pose']))
        tcp[:, side_id] = world_robot @ raw_tcp
        q = np.stack(window[f'states.{side}_joint.position'])
        assert q.shape == (n, spec['dof'])
        joint[:, side_id * 7:side_id * 7 + spec['dof']] = q
        joint_valid[:, side_id * 7:side_id * 7 + spec['dof']] = np.isfinite(q)
        if spec['robot'] == 'genie1':
            base, target = 'arm_base_link', f'arm_{letter}_end_link'
            joints = [f'idx{2 if side_id == 0 else 6}{i}_arm_{letter}_joint{i}' for i in range(1, 8)]
        elif spec['robot'] == 'split_aloha':
            base, target = f'{side}/arm_base', f'{side}/link6'
            joints = [f'{side}/joint{i}' for i in range(1, 7)]
        else:
            base = 'body4'
            target = f'{"L" if side_id == 0 else "R"}_arm_link{16 if side_id == 0 else 26}'
            joints = [f'joint{1 if side_id == 0 else 2}{i}' for i in range(1, 7)]
        local = model.fk(dict(zip(joints, q.T)), target, base)
        virtual_base = np.eye(4)
        if spec['robot'] == 'lift2':
            # Source arm-base is at joint11/21's pre-motion frame; strip only its
            # declared URDF fixed mounting translation, retaining all link lengths.
            origin = model.joints[joints[0]].find('origin')
            virtual_base[:3, 3] = np.fromstring(origin.get('xyz'), sep=' ')
            local = np.linalg.inv(virtual_base) @ local
        install = raw_ee @ np.linalg.inv(label)
        tool = np.linalg.inv(raw_ee) @ raw_tcp
        fixed_install = install[0]
        fixed_tool = tool[0]
        correction = np.eye(4)
        correction[:3, :3] = np.asarray(source_correction[side])
        canonical_from_raw = world_robot @ raw_ee @ correction
        np.testing.assert_allclose(canonical_from_raw, eef[:, side_id], atol=3e-6, rtol=0)
        np.testing.assert_allclose(install @ local_tcp, raw_tcp, atol=3e-6, rtol=0)
        fk_local[:, side_id] = local
        label_local[:, side_id] = label
        fk_world[:, side_id] = world_robot @ fixed_install @ local @ correction
        fk_tcp[:, side_id] = world_robot @ fixed_install @ local @ fixed_tool
        diagnostics[side] = dict(local_fk_error=error_summary(label, local),
            world_eef_error=error_summary(eef[:, side_id], fk_world[:, side_id]),
            world_tcp_error=error_summary(tcp[:, side_id], fk_tcp[:, side_id]),
            raw_tool_transform=fixed_tool.tolist(), raw_tool_constancy=error_summary(np.broadcast_to(fixed_tool, tool.shape), tool),
            robot_from_armbase=fixed_install.tolist(), installation_constancy=error_summary(np.broadcast_to(fixed_install, install.shape), install),
            canonical_axis_correction=correction.tolist(),
            world_installation_evidence='First selected source EE_to_robot @ inverse(EE_to_armbase); derived from GT, not independent calibration.',
            tool_evidence='First selected inverse(raw EE_to_robot) @ raw TCP_to_robot; constant verified across window, source label derived, not independent physical calibration.')
        chain_evidence[side] = dict(base_link=base, target_link=target, joint_names=joints,
            source_joint_key=f'states.{side}_joint.position', virtual_armbase_from_urdf_base=virtual_base.tolist(),
            joints=[dict(name=j.get('name'), type=j.get('type'), parent=j.find('parent').get('link'),
                         child=j.find('child').get('link'), origin=j.find('origin').attrib if j.find('origin') is not None else {})
                    for j in model.chain(target, base)])
        if spec['robot'] == 'genie1':
            center = model.fk({}, f'gripper_{letter}_center_link', target)[0]
            diagnostics[side]['urdf_flange_to_named_center'] = center.tolist()
            diagnostics[side]['center_policy'] = 'Named center is not the raw TCP; retain raw +0.220 m tool, do not substitute center.'
    limitations = [
        'World FK uses source-label-derived arm installation; local joint FK is independent of source EE/TCP labels.',
        'FK TCP tool is verified constant in source labels but is not an independently calibrated physical contact frame.',
        'Source lineage is sim_updated_lerobotv30; source video modality is preserved without claiming physical camera acquisition.',
        'Camera calibration is source supplied pinhole; no extra distortion or independently measured calibration is invented.',
        'Missing lift/body joints prevent independent whole-body world FK.',
    ]
    if spec['robot'] == 'lift2':
        limitations.append('Released ARX URDF disagrees with source arm-local EE position by roughly 56 mm; preserve the mismatch without fitting link lengths or tool corrections.')
    endpoint = 'Declared source TCP xyz+wxyz transformed by states.robot_to_env_pose; exactly the same TCP is used for head and both wrists.'
    base_meta = dict(name=spec['name'], id=identifier, dataset_family=identifier, source_group='a1',
        dataset_dir=str(dataset), episode_index=int(episode.episode_index), episode=int(episode.episode_index),
        episode_length=int(episode.length), frame_indices=indices.tolist(), anchor=anchor, horizon=n-1,
        tcp_status='declared_tcp', endpoint=endpoint, limitations=' '.join(limitations),
        origin_diagnostics=[], clip_label=f'{spec["name"]} · episode {int(episode.episode_index)} · frames {anchor}–{anchor+n-1}',
        task=str(episode.get('tasks', '')), source_variant=source_meta.get('source_variant'),
        selection_policy='Fixed 180-row contiguous window; Genie1 scan_qrcode_part2 selected after bounded camera inspection because part1 head view was heavily occluded. No pose/calibration fitting used for clip selection.',
        pose_frame='world', pose_units='meter', quaternion_order='wxyz',
        pose_source='states.left_tcp_to_robot_pose / states.right_tcp_to_robot_pose',
        quaternion_evidence='Source converter pose_wxyz_to_matrix; raw EE composed with robot_to_env and declared canonical rotation correction matches canonical EEF within 3e-6.',
        source_info=str(dataset/'meta/info.json'))
    wrist_summary = dict(name=spec['name'], dataset_dir=str(dataset), episode_index=int(episode.episode_index), views={})
    for camera, side in [('cam_high', None), ('cam_left_wrist', 'left'), ('cam_right_wrist', 'right')]:
        camera_key = 'observation.images.' + camera
        ikey, ekey = [f'observation.states.cameras.{camera}.{field}' for field in ('intrinsic', 'world_to_camera')]
        k = np.stack(window[ikey]).reshape(n, 3, 3)
        e = np.stack(window[ekey]).reshape(n, 4, 4)
        raw_camera_key = ('head' if side is None else f'hand_{side}') + '_camera_to_robot_extrinsics'
        expected_extrinsic = np.linalg.inv(world_robot @ pose7_to_matrix(np.stack(window[raw_camera_key])))
        np.testing.assert_allclose(e, expected_extrinsic, atol=3e-6, rtol=0)
        valid = camera_validity(k, e)
        height, width = info['features'][camera_key]['shape'][:2]
        tcp_homogeneous = np.concatenate((tcp[..., :3, 3], np.ones((n, 2, 1))), axis=-1)
        camera_tcp = np.einsum('nij,naj->nai', e, tcp_homogeneous)[..., :3]
        projected = np.einsum('nij,naj->nai', k, camera_tcp)
        pixels = projected[..., :2] / projected[..., 2:]
        in_frame = ((camera_tcp[..., 2] > 0) & (pixels[..., 0] >= 0) & (pixels[..., 0] < width)
                    & (pixels[..., 1] >= 0) & (pixels[..., 1] < height))
        video = dataset / info['video_path'].format(video_key=camera_key,
            chunk_index=int(episode[f'videos/{camera_key}/chunk_index']), file_index=int(episode[f'videos/{camera_key}/file_index']))
        target_dir = root / ('video_samples' if side is None else 'wrist_samples') / identifier
        if side is not None:
            target_dir /= side
        target_dir.mkdir(parents=True, exist_ok=True)
        meta = dict(base_meta, model='pinhole', image_width=int(width), image_height=int(height),
            nativewh=[int(width), int(height)], camera_key=camera_key, intrinsic_key=ikey, extrinsic_key=ekey,
            video_path=str(video), video_episode_from_timestamp=float(episode[f'videos/{camera_key}/from_timestamp']),
            camera_model_evidence='Released per-camera OpenCV K and world_to_camera; pinhole without additional distortion.',
            camera_transform_evidence=f'world_to_camera matches inverse(robot_to_env @ {raw_camera_key}) within 3e-6; source xyz+wxyz.',
            camera_transform_max_abs_error=float(np.max(np.abs(e - expected_extrinsic))),
            tcp_projection_in_frame_counts_left_right=in_frame.sum(axis=0).tolist(),
            tcp_projection_behind_camera_counts_left_right=(camera_tcp[..., 2] <= 0).sum(axis=0).tolist(),
            video_available=True, projection_available=bool(valid.any()), status='calibrated_source',
            reason='This camera uses its own source K and frame-dependent world_to_camera with the same world TCP trajectory.',
            valid_camera_frames=int(valid.sum()))
        arrays = dict(frame_indices=indices, K=k, world_to_camera=e, camera_valid=valid, distortion=np.empty((n, 0)))
        if side is None:
            arrays.update(eef_world=eef, tcp_world=tcp, eef_valid=np.isfinite(eef).all(axis=(-1,-2)),
                          tcp_valid=np.isfinite(tcp).all(axis=(-1,-2)), world_from_robot=world_robot)
        else:
            meta['side'] = side
        np.savez_compressed(target_dir/'arrays.npz', **arrays)
        if skip_decode and (target_dir/'metadata.json').exists():
            previous = json.loads((target_dir/'metadata.json').read_text())
            for key in ('dataset_dir', 'episode_index', 'frame_indices', 'video_path', 'video_episode_from_timestamp'):
                if previous[key] != meta[key]:
                    raise ValueError(f'Cannot reuse decoded frames with changed {key}')
            for key in ('image_filenames','fps','video_fps','display_width','display_height','displaywh',
                        'video_frame_offset','video_ordinal_start','video_frame_ordinals','video_pts_seconds',
                        'frame_count','duration_seconds','pts_span_seconds','video_pts_spacing_max_error_seconds',
                        'video_ordinal_contiguity_checked','video_pts_spacing_checked','video_alignment'):
                if key in previous:
                    meta[key] = previous[key]
        else:
            print(f'{identifier}: decoding {camera}', flush=True)
            meta.update(decode_sequence(meta, target_dir))
        write_json(target_dir/'metadata.json', meta)
        if side is not None:
            wrist_summary['views'][side] = meta
    write_json(root/'wrist_samples'/identifier/'summary.json', wrist_summary)
    fk_dir = root/'fk_tracks'/identifier
    fk_dir.mkdir(parents=True, exist_ok=True)
    arrays = dict(frame_indices=indices, fk_joint=joint, fk_joint_valid=joint_valid)
    for key, value, mask_key in [('fk_eef_world',fk_world,'fk_eef_valid'), ('fk_tcp_world',fk_tcp,'fk_tcp_valid'),
                                ('fk_local_pose',fk_local,'fk_local_valid'), ('label_local_pose',label_local,'label_local_valid')]:
        arrays[key] = value
        arrays[mask_key] = np.isfinite(value).all(axis=(-1,-2))
    np.savez_compressed(fk_dir/'arrays.npz', **arrays)
    fk_meta = dict(name=spec['name'], status='local_fk_mismatch' if spec['robot']=='lift2' else 'independent_local_fk_verified',
        local_frame='arm_base_link (common left/right)' if spec['robot']=='genie1' else 'each arm pre-joint1 base',
        world_frame_status='source_gt_derived_installation_diagnostic', independent_world_calibration=False,
        tcp_status='declared_source_tcp_with_label_derived_fixed_tool', limitations=limitations,
        source_metadata=dict(dataset_dir=str(dataset), episode_index=int(episode.episode_index)),
        urdf_path=str(model.path), urdf_sha256=model.sha256, dof_per_arm=spec['dof'],
        joint_layout='left 7 slots then right 7; six-axis arms have masked NaN slot 7, no gripper-as-joint',
        diagnostics=diagnostics, chains=chain_evidence, source_variant=source_meta.get('source_variant'))
    write_json(fk_dir/'metadata.json', fk_meta)
    write_json(root/'entry.json', dict(id=identifier,title=spec['name'],family=identifier,
        root=f'expanded_sources/{identifier}',source_group='a1', reason='Distinct A1 embodiment with released head and bilateral wrist videos, declared raw TCP, and independent local joint/URDF FK.'))
    print(identifier, json.dumps({s:d['local_fk_error'] for s,d in diagnostics.items()}), flush=True)
    return fk_meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=ROOT/'outputs/fk_consistency/expanded_sources')
    parser.add_argument('--names', nargs='+', choices=list(SPECS), default=list(SPECS))
    parser.add_argument('--anchor', type=int, default=120)
    parser.add_argument('--count', type=int, default=180)
    parser.add_argument('--skip-decode', action='store_true')
    args = parser.parse_args()
    for identifier in args.names:
        extract_one(identifier, args.output_root, args.anchor, args.count, args.skip_decode)
