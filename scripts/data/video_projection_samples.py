"""Extract consecutive source frames and calibrated state poses for video reports.

Source datasets and the existing snapshot report remain read-only. The output
contains one JPEG per real decoded video frame, with source PTS and ordinals.
Run ``python -m scripts.data.video_projection_samples`` from the repository root.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import cv2
import numpy as np
from omegaconf import OmegaConf

from scripts.data.projection_v3_samples import SOURCES, extract_one
from scripts.data.tcp_endpoint_policy import apply_endpoint_policy
from scripts.data.urdf_fk_audit import eef20_to_matrix, load_episode, pose7_to_matrix
from scripts.visualize_fk_consistency import NAMES, PROFILES

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / 'outputs/fk_consistency/video_samples'


def decode_sequence(metadata: dict, target: Path) -> dict:
    """Decode each requested ordinal and check actual presentation-time spacing."""
    indices = np.asarray(metadata['frame_indices'], dtype=int)
    if len(indices) < 2 or not np.all(np.diff(indices) == 1):
        raise ValueError('Video samples must be consecutive state frames')
    image_dir = target / 'images'
    image_dir.mkdir(parents=True, exist_ok=True)
    offset_s = float(metadata.get('video_episode_from_timestamp', 0))
    filenames, pts, ordinals = [], [], []
    with av.open(metadata['video_path']) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        offset = round(offset_s * fps)
        if abs(offset_s * fps - offset) > 1e-3:
            raise ValueError(f'Non-integral episode video offset: {offset_s}s at {fps}fps')
        first, last = offset + int(indices[0]), offset + int(indices[-1])
        for ordinal, frame in enumerate(container.decode(stream)):
            if ordinal < first:
                continue
            if ordinal > last:
                break
            image = frame.to_ndarray(format='bgr24')
            h, w = image.shape[:2]
            if (w, h) != (metadata['image_width'], metadata['image_height']):
                raise ValueError(f'Calibration/video dimensions mismatch: {(w, h)}')
            width, height = 640, round(h * 640 / w)
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            filename = f'images/frame_{len(filenames):05d}.jpg'
            if not cv2.imwrite(str(target / filename), image, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                raise IOError(f'Unable to write {filename}')
            if frame.pts is None:
                raise ValueError('Source video frame lacks PTS')
            filenames.append(filename)
            pts.append(float(frame.pts * stream.time_base))
            ordinals.append(ordinal)
            if ordinal == last:
                break
    if len(filenames) != len(indices):
        raise ValueError(f'Video ended after {len(filenames)}/{len(indices)} requested frames')
    np.testing.assert_array_equal(ordinals, offset + indices)
    np.testing.assert_allclose(np.diff(pts), 1 / fps, atol=0.1 / fps, rtol=0)
    timestamp_error = float(np.max(np.abs(np.diff(pts) - 1 / fps)))
    return dict(image_filenames=filenames, fps=fps, video_fps=fps,
                display_width=width, display_height=height, displaywh=[width, height],
                video_frame_offset=offset, video_ordinal_start=first,
                video_frame_ordinals=ordinals, video_pts_seconds=pts,
                video_episode_from_timestamp=offset_s, frame_count=len(indices),
                duration_seconds=len(indices) / fps,
                pts_span_seconds=pts[-1] - pts[0],
                video_pts_spacing_max_error_seconds=timestamp_error,
                video_ordinal_contiguity_checked=True, video_pts_spacing_checked=True,
                video_alignment='Actual consecutive decoded source frames; ordinal = round(episode from_timestamp * source fps) + frame_index. PTS spacing checked against source fps.')


def extract_robot(name: str, output_root: Path, anchor: int, count: int, *,
                  profile_path=None, episode_index=0) -> dict:
    profile = profile_path or ROOT / PROFILES[name]
    cfg = OmegaConf.load(profile)
    info = json.loads((Path(cfg.dataset_dir) / 'meta/info.json').read_text())
    intrinsic = 'observation.states.camera.intrinsic'
    if intrinsic not in info['features']:
        intrinsic = 'observation.states.cameras.cam_high.intrinsic'
    extrinsic = 'observation.states.camera.world_to_camera'
    keys = ['observation.states.eef', intrinsic, extrinsic]
    if name == 'interndata':
        keys += ['states.left_tcp_to_robot_pose', 'states.right_tcp_to_robot_pose',
                 'states.robot_to_env_pose']
    _, _, episode, frame = load_episode(profile, episode_index=episode_index, columns=keys)
    window = frame[(frame.frame_index >= anchor) & (frame.frame_index < anchor + count)]
    indices = window.frame_index.to_numpy(dtype=int)
    np.testing.assert_array_equal(indices, np.arange(anchor, anchor + len(indices)))
    if len(indices) < 33:
        raise ValueError(f'{name}: insufficient rows after requested anchor')
    n = len(indices)
    eef = eef20_to_matrix(np.stack(window['observation.states.eef']))
    tcp = np.broadcast_to(np.eye(4), eef.shape).copy()
    eef_valid = np.isfinite(eef).all(axis=(-1, -2))
    tcp_valid = np.zeros((n, 2), bool)
    status = 'unavailable'
    endpoint = 'Only source wrist/flange EEF; no verified TCP in this profile.'
    extra = {}
    if name == 'interndata':
        reference = pose7_to_matrix(np.stack(window['states.robot_to_env_pose']))
        extra['world_from_robot'] = reference
        for side_id, side in enumerate(('left', 'right')):
            tcp[:, side_id] = reference @ pose7_to_matrix(np.stack(window[f'states.{side}_tcp_to_robot_pose']))
        tcp_valid[:] = np.isfinite(tcp).all(axis=(-1, -2))
        status = 'declared_tcp'
        endpoint = 'Native xyz+wxyz TCP, robot_to_env applied per row; EEF is a different endpoint.'
    camera = np.stack(window[extrinsic]).reshape(-1, 4, 4)
    k = np.stack(window[intrinsic]).reshape(-1, 3, 3)
    camera_valid = np.isfinite(camera).all(axis=(-1, -2)) & np.isfinite(k).all(axis=(-1, -2))
    camera_key = cfg.head_camera
    native_h, native_w = info['features'][camera_key]['shape'][:2]
    video_root = Path(cfg.get('video_root') or cfg.dataset_dir)
    video = video_root / info['video_path'].format(video_key=camera_key,
        chunk_index=int(episode[f'videos/{camera_key}/chunk_index']),
        file_index=int(episode[f'videos/{camera_key}/file_index']))
    target = output_root / name
    target.mkdir(parents=True, exist_ok=True)
    arrays = dict(eef_world=eef, tcp_world=tcp,
        eef_valid=eef_valid, tcp_valid=tcp_valid, world_to_camera=camera, K=k,
        distortion=np.empty((n, 0)), camera_valid=camera_valid, frame_indices=indices, **extra)
    metadata = dict(name=NAMES[name], dataset_dir=str(cfg.dataset_dir),
        episode=int(episode.episode_index), episode_index=int(episode.episode_index),
        episode_length=int(episode.length), anchor=anchor, horizon=n - 1,
        frame_indices=indices.tolist(), model='pinhole', distortion_order=[],
        image_width=native_w, image_height=native_h, nativewh=[native_w, native_h],
        camera_key=camera_key, tcp_status=status, endpoint=endpoint,
        video_path=str(video), video_episode_from_timestamp=float(episode.get(f'videos/{camera_key}/from_timestamp', 0)),
        intrinsic_key=intrinsic, extrinsic_key=extrinsic, pose_frame='world', pose_units='meter',
        pose_source='observation.states.eef', source_info=str(Path(cfg.dataset_dir) / 'meta/info.json'),
        camera_model_evidence='Released head video and row-major K; pinhole projection with no extra distortion applied. No claim of independently verified reprojection calibration.',
        limitations='Native endpoint semantics and calibration are source-specific; see per-source endpoint audit. Image plausibility alone is not independent camera/tool calibration.')
    metadata['dataset_family'] = name
    metadata['task'] = str(episode.get('task_instruction', episode.get('tasks', '')))
    metadata['source_episode_audit'] = {
        key: str(episode[key]) for key in (
            'head_extrinsic_audit_classification', 'head_extrinsic_base_fk_classification',
            'head_extrinsic_repair_status', 'wrist_eef_fk_classification') if key in episode}
    apply_endpoint_policy(name, arrays, metadata)
    np.savez_compressed(target / 'arrays.npz', **arrays)
    return metadata


def extract_all(output_root: Path = DEFAULT_OUTPUT, *, names=None, anchor=30, count=180):
    result = {}
    for name in names or [*PROFILES, *SOURCES]:
        if name in PROFILES:
            metadata = extract_robot(name, output_root, anchor, count)
        else:
            metadata = extract_one(name, output_root, anchor_frame=anchor,
                                   horizon=count - 1, require_all_valid=False)
        target = output_root / name
        metadata.update(decode_sequence(metadata, target))
        for key in ('snapshot_indices', 'snapshot_frame_indices'):
            metadata.pop(key, None)
        metadata['requested_frame_count'] = count
        metadata['selection_policy'] = 'Consecutive actual states starting at requested anchor, bounded by episode end; retain source validity masks without searching for a fully valid window.'
        (target / 'metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + '\n')
        result[name] = metadata
        print(f'{name}: {metadata["frame_count"]} frames, {metadata["fps"]:g} fps, {metadata["duration_seconds"]:.3f}s', flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--names', nargs='+', choices=[*PROFILES, *SOURCES])
    parser.add_argument('--anchor', type=int, default=30)
    parser.add_argument('--count', type=int, default=180)
    args = parser.parse_args()
    extract_all(args.output_root, names=args.names, anchor=args.anchor, count=args.count)
