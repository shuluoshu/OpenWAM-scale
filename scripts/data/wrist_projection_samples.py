"""Extract real wrist-camera frames with their own calibration and provenance."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from scripts.data.genrobot_wrist_geometry import (
    FINGER_BASE_TO_CAMERA,
    FINGER_URDF_SHA256,
    FINGER_URDF_SOURCE,
    genrobot_wrist_world_to_camera,
)
from scripts.data.urdf_fk_audit import eef20_to_matrix, load_episode
from scripts.data.video_projection_samples import decode_sequence


def wrist_keys(side, features):
    camera = f'cam_{side}_wrist'
    k = f'observation.states.cameras.{camera}.intrinsic'
    if k not in features:
        k = f'observation.states.camera.calibration.{side}_camera.intrinsic_cv'
    e = f'observation.states.cameras.{camera}.world_to_camera'
    return (k if k in features else None, e if e in features else None)


def camera_validity(k, e):
    finite = np.isfinite(k).all(axis=(-1, -2)) & np.isfinite(e).all(axis=(-1, -2))
    r = np.where(finite[:, None, None], e[:, :3, :3], np.eye(3))
    return (finite & (k[:, 0, 0] > 0) & (k[:, 1, 1] > 0)
            & np.isclose(k[:, 2], [0, 0, 1], atol=1e-5).all(-1)
            & np.isclose(e[:, 3], [0, 0, 0, 1], atol=1e-5).all(-1)
            & np.isclose(np.swapaxes(r, -1, -2) @ r, np.eye(3), atol=1e-3).all(axis=(-1, -2))
            & np.isclose(np.linalg.det(r), 1., atol=1e-3))


def extract_wrist_sample(name, root):
    folder = root/'video_samples'/name
    source = json.loads((folder/'metadata.json').read_text())
    dataset = Path(source['dataset_dir'])
    info = json.loads((dataset/'meta/info.json').read_text())
    features = info['features']
    with np.load(folder/'arrays.npz') as f:
        source_arrays = dict(f)
    indices = source_arrays['frame_indices']
    n = len(indices)
    output = root/'wrist_samples'/name
    output.mkdir(parents=True, exist_ok=True)
    summary = dict(name=name, dataset_dir=str(dataset), episode_index=source['episode_index'], views={})
    keys = {}
    columns = ['observation.states.eef']
    for side in ['left', 'right']:
        video_key = f'observation.images.cam_{side}_wrist'
        if video_key not in features:
            summary['views'][side] = dict(status='no_wrist_video', video_available=False,
                projection_available=False, reason='当前来源没有该侧腕部相机视频；不能用头相机或TCP-local坐标图代替。')
            continue
        k, e = wrist_keys(side, features)
        keys[side] = dict(k=k, e=e, video_key=video_key)
        columns += [key for key in [k, e] if key]
        if name == 'genrobot':
            raw = f'raw.robot{1 if side == "left" else 2}.sensor.camera0'
            keys[side]['raw'] = raw
            columns += [raw+'.camera_info.'+field for field in ['K', 'D', 'T_b_c']]
            columns += [raw+'.video_alignment_valid']
            columns += [f'raw.robot{1 if side == "left" else 2}.finger.eef_pose.alignment_valid',
                        'canonical.frame_valid']
    if keys:
        profile = output/'source.yaml'
        OmegaConf.save(OmegaConf.create(dict(dataset_dir=str(dataset))), profile)
        _, _, episode, frame = load_episode(profile, episode_index=source['episode_index'], columns=list(dict.fromkeys(columns)))
        window = frame.set_index('frame_index').loc[indices]
        np.testing.assert_array_equal(window.episode_index, source['episode_index'])
        # Prove that this camera window addresses the same state trajectory.
        np.testing.assert_allclose(eef20_to_matrix(np.stack(window['observation.states.eef'])),
                                   source_arrays['eef_world'], atol=2e-5, rtol=0)
        head_key = source['camera_key']
        head_relative = Path(info['video_path'].format(video_key=head_key,
            chunk_index=int(episode[f'videos/{head_key}/chunk_index']),
            file_index=int(episode[f'videos/{head_key}/file_index'])))
        head_path = Path(source['video_path'])
        if tuple(head_path.parts[-len(head_relative.parts):]) != head_relative.parts:
            raise ValueError('Head video path does not match source episode video layout')
        video_root = head_path.parents[len(head_relative.parts)-1]
    for side, fields in keys.items():
        target = output/side
        target.mkdir(parents=True, exist_ok=True)
        camera_key = fields['video_key']
        height, width = features[camera_key]['shape'][:2]
        camera_meta = dict(name=name, side=side, dataset_dir=str(dataset),
            episode_index=source['episode_index'], frame_indices=indices.tolist(),
            image_width=width, image_height=height, camera_key=camera_key,
            model='pinhole', intrinsic_key=fields['k'], extrinsic_key=fields['e'],
            video_path=str(video_root/info['video_path'].format(video_key=camera_key,
                chunk_index=int(episode[f'videos/{camera_key}/chunk_index']),
                file_index=int(episode[f'videos/{camera_key}/file_index']))),
            video_episode_from_timestamp=float(episode[f'videos/{camera_key}/from_timestamp']),
            calibration_fields={key: features[key] for key in [fields['k'], fields['e']] if key})
        k = np.stack(window[fields['k']]).reshape(n, 3, 3) if fields['k'] else np.full((n, 3, 3), np.nan)
        e = np.stack(window[fields['e']]).reshape(n, 4, 4) if fields['e'] else np.full((n, 4, 4), np.nan)
        distortion = np.empty((n, 0))
        reason = '使用该腕部相机自己的逐帧K及world/base→camera矩阵；随动相机与背景逐帧同步。'
        if 'raw' in fields:
            raw = fields['raw']
            k = np.stack(window[raw+'.camera_info.K']).reshape(n, 3, 3)
            d = np.stack(window[raw+'.camera_info.D'])
            np.testing.assert_allclose(d[:, :4], k[:, [0, 1, 0, 1], [0, 1, 2, 2]], atol=1e-4)
            distortion = d[:, 4:6]
            camera_meta.update(model='double_sphere', intrinsic_key=raw+'.camera_info.K',
                               distortion_key=raw+'.camera_info.D',
                               raw_T_b_c_first=np.asarray(window[raw+'.camera_info.T_b_c'].iloc[0]).tolist(),
                               raw_video_alignment_valid_count=int(window[raw+'.video_alignment_valid'].sum()))
            alignment_keys = ['canonical.frame_valid', raw+'.video_alignment_valid',
                              f'raw.robot{1 if side == "left" else 2}.finger.eef_pose.alignment_valid']
            source_valid = np.logical_and.reduce([
                np.isfinite(window[key].to_numpy(dtype=float)) & (window[key].to_numpy(dtype=float) > .5)
                for key in alignment_keys])
            e, derived_valid = genrobot_wrist_world_to_camera(
                source_arrays['eef_world'][:, 0 if side == 'left' else 1],
                np.stack(window[raw+'.camera_info.T_b_c']), source_valid)
            camera_meta.update(calibration_status='cad_derived_preview',
                camera_chain_source=FINGER_URDF_SOURCE, camera_chain_sha256=FINGER_URDF_SHA256,
                base_from_optical=FINGER_BASE_TO_CAMERA.tolist(), source_validity_keys=alignment_keys,
                derived_valid_frames=int(derived_valid.sum()),
                formula='T_camera_world = inverse(T_world_finger_base @ T_base_cam_Link); identity raw T_b_c only',
                limitations='CAD optical frame interpretation validated by video projections; acquisition serial/CAD revision and independent calibration are unverified. TCP remains a fixed-offset candidate.')
            reason = '已补Finger URDF的base_link→cam_Link固定旋转，结合本侧逐帧EEF和double-sphere K/D投影；仅接受单位T_b_c及有效对齐帧。CAD推导可视化候选，设备/CAD版本未独立确认，非实物标定。'
        valid = camera_validity(k, e)
        arrays = dict(frame_indices=indices, K=k, world_to_camera=e,
                      camera_valid=valid, distortion=distortion)
        camera_meta.update(decode_sequence(camera_meta, target))
        if abs(camera_meta['fps']-source['fps']) > 1e-6:
            raise ValueError('Wrist/source FPS mismatch requires explicit resampling, not ordinal reuse')
        camera_meta.update(status='calibrated_source' if valid.any() else 'projection_masked',
                           video_available=True, projection_available=bool(valid.any()),
                           valid_camera_frames=int(valid.sum()), reason=reason,
                           lens_note='Only supplied model is used; no invented distortion coefficients.')
        if 'raw' in fields and valid.any():
            camera_meta['status'] = 'cad_derived_preview'
        np.savez_compressed(target/'arrays.npz', **arrays)
        (target/'metadata.json').write_text(json.dumps(camera_meta, indent=2, ensure_ascii=False)+'\n')
        summary['views'][side] = camera_meta
    (output/'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n')
    return summary
