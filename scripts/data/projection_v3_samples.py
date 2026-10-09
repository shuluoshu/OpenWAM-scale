"""Bounded, read-only V0_3 extraction for the trajectory projection report.

Run from the repository root: python -m scripts.data.projection_v3_samples
No training reader or source dataset is modified. Arrays include anchor state t
and states t+1:t+33; action columns and horizon-50 sidecar masks are not labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import cv2
import numpy as np
import pyarrow.parquet as pq

RELEASE = Path('/mnt/data/cpfs/b5big/pretrain_data_release/pretrain_datasetsV0_3')
FACTORY = Path('/mnt/workspace/mark/opensource_data_factory')
SOURCES = {
    'genrobot': ('Genrobot/shard_0148', 'cam_high'),
    'lightwheel': ('LightWheel/EgoStandard/shard_0148', 'head_left'),
    'mecka': ('mecka/shard_0148', 'egocentric'),
}


def _read_episode(root: Path, info: dict, limit: int = 512):
    """Read episode metadata without its huge stats columns, then bounded rows."""
    meta = pq.ParquetFile(sorted((root / 'meta/episodes').rglob('*.parquet'))[0])
    columns = [c for c in meta.schema_arrow.names if not c.startswith('stats/')]
    episode = next(meta.iter_batches(batch_size=1, columns=columns)).to_pylist()[0]
    path = root / info['data_path'].format(
        chunk_index=episode['data/chunk_index'], file_index=episode['data/file_index'])
    parquet = pq.ParquetFile(path)
    columns = [c for c in parquet.schema_arrow.names if (
        c in ('frame_index', 'episode_index', 'timestamp', 'source_frame_index',
              'observation.states.eef', 'observation.states.eef_tcp',
              'observation.states.eef.mask', 'observation.states.hand.position',
              'observation.states.hand.valid', 'observation.states.hand.camera.valid',
              'observation.states.camera.valid', 'observation.states.camera.intrinsic',
              'observation.states.camera.world_to_camera', 'canonical.frame_valid',
              'source.frame_validity', 'source.camera.world_to_camera_valid')
        or c == 'observation.states.cameras.head_left.intrinsic'
        or c.startswith('raw.robot0.sensor.camera2.camera_info.')
        or c == 'raw.robot0.sensor.camera2.video_alignment_valid')]
    records = []
    for batch in parquet.iter_batches(batch_size=128, columns=columns):
        for row in batch.to_pylist():
            if row['episode_index'] == episode['episode_index']:
                records.append(row)
            if len(records) >= min(limit, episode['length']):
                break
        if len(records) >= min(limit, episode['length']):
            break
    if len(records) < 33:
        raise ValueError(f'{path}: fewer than 33 rows in first episode')
    data = {k: np.asarray([r[k] for r in records]) for k in columns}
    return episode, path, data


def _poses(values: np.ndarray):
    blocks = values.reshape(-1, 2, 10).astype(np.float64)
    a, b = blocks[..., 3:6], blocks[..., 6:9]
    c = np.cross(a, b)
    rotation = np.stack((a, b, c), axis=-1)
    valid = (np.isfinite(blocks[..., :9]).all(-1)
             & (np.abs(np.linalg.norm(a, axis=-1) - 1) < 1e-3)
             & (np.abs(np.linalg.norm(b, axis=-1) - 1) < 1e-3)
             & (np.abs(np.sum(a*b, axis=-1)) < 1e-3)
             & (np.linalg.det(rotation) > 0.999))
    pose = np.broadcast_to(np.eye(4), (*blocks.shape[:2], 4, 4)).copy()
    pose[..., :3, :3] = rotation
    pose[..., :3, 3] = blocks[..., :3]
    return pose, valid


def _valid_geometry(data, eef, tcp):
    """Conservative diagnostics supplement (rather than invent) source masks."""
    hands = data['observation.states.hand.position'].reshape(-1, 2, 21, 3)
    pinch = (hands[..., 4, :] + hands[..., 8, :]) / 2
    y = hands[..., 5, :] - hands[..., 2, :]
    palm = (hands[..., 5, :] + hands[..., 17, :]) / 2 - hands[..., 0, :]
    geometry = (np.isfinite(hands).all(axis=(-1, -2))
                & (np.linalg.norm(y, axis=-1) > 1e-7)
                & (np.linalg.norm(np.cross(palm, y), axis=-1) > 1e-9))
    origin_error = np.linalg.norm(tcp[..., :3, 3] - pinch, axis=-1)
    # Metadata explicitly says invalid virtual frames copy the wrist block.
    fallback = np.all(np.isclose(tcp, eef, atol=1e-7, rtol=0), axis=(-1, -2))
    return geometry & (origin_error < 1e-5) & ~fallback, fallback, origin_error


def _images(path, episode, video_key, data, selection, target: Path, native_wh):
    """Decode by actual encoded frame ordinal, recording presentation times.

    Merged-video from_timestamp is applied once. Raw source PTS is not an extra
    offset: episode metadata already addresses the released video timeline.
    """
    prefix = f'videos/{video_key}/'
    offset = float(episode[prefix + 'from_timestamp'])
    indices = data['frame_index'][selection]
    snapshots = [0, 16, 32]
    pts = []
    filenames = [f'image{i}.jpg' for i in snapshots]
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        offset_frames = int(round(offset * fps))
        if abs(offset * fps - offset_frames) > 1e-3:
            raise ValueError(f'Nonintegral merged video frame offset: {offset}')
        ordinals = [offset_frames + int(indices[i]) for i in snapshots]
        wanted = dict(zip(ordinals, filenames))
        sizes = []
        for ordinal, frame in enumerate(container.decode(stream)):
            if ordinal in wanted:
                image = frame.to_ndarray(format='bgr24')
                h, w = image.shape[:2]
                if [w, h] != list(native_wh):
                    raise ValueError(f'Video/calibration resolution mismatch: {(w,h)} != {native_wh}')
                scale = min(1., 800. / w)
                width, height = round(w * scale), round(h * scale)
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
                if not cv2.imwrite(str(target / wanted[ordinal]), image):
                    raise IOError('Cannot write snapshot')
                sizes.append([width, height])
                pts.append(float(frame.pts * stream.time_base))
            if ordinal >= max(ordinals):
                break
    if len(pts) != 3:
        raise ValueError('Video ended before all requested snapshots')
    # A fixed initial PTS is allowed, but selected frame spacing must be 30 Hz.
    np.testing.assert_allclose(np.diff(pts), np.diff(ordinals) / fps, atol=1. / fps * .1)
    return dict(image_filenames=filenames, snapshot_indices=snapshots,
                snapshot_frame_indices=[int(indices[i]) for i in snapshots],
                displaywh=sizes[0], display_width=sizes[0][0], display_height=sizes[0][1],
                video_frame_ordinals=ordinals, video_pts_seconds=pts, video_fps=fps,
                video_episode_from_timestamp=offset,
                video_alignment='Released-video frame ordinal = round(episode from_timestamp * video fps) + episode frame_index; decoded PTS spacing checked. Raw source first PTS is recorded, not added twice.')


def extract_one(name: str, output_root: Path, *, anchor_frame: int = 30,
                horizon: int = 32, require_all_valid: bool = True) -> dict:
    leaf, camera = SOURCES[name]
    root = RELEASE / leaf
    info = json.loads((root / 'meta/info.json').read_text())
    episode, parquet_path, data = _read_episode(root, info, limit=max(512, anchor_frame + horizon + 1))
    n = len(data['frame_index'])
    eef, eef_valid = _poses(data['observation.states.eef'])
    mask_keys = []
    for key in ('canonical.frame_valid', 'source.frame_validity'):
        if key in data:
            eef_valid &= data[key].astype(bool)[:, None]
            mask_keys.append(key)
    for key in ('observation.states.eef.mask', 'observation.states.hand.valid'):
        if key in data:
            mask = data[key].reshape(n, 2, -1)
            eef_valid &= np.isfinite(mask).all(-1) & (mask > .5).all(-1)
            mask_keys.append(key)
    camera_valid = np.ones(n, bool)
    camera_mask_keys = []
    for key in ('source.camera.world_to_camera_valid', 'observation.states.camera.valid',
                'observation.states.hand.camera.valid', 'raw.robot0.sensor.camera2.video_alignment_valid'):
        if key in data:
            camera_valid &= np.isfinite(data[key]) & (data[key] > .5)
            camera_mask_keys.append(key)
    w2c = data['observation.states.camera.world_to_camera'].reshape(-1, 4, 4)
    camera_valid &= (np.isfinite(w2c).all(axis=(-1, -2))
                     & (np.abs(np.linalg.det(w2c[:, :3, :3]) - 1) < 1e-3)
                     & np.all(np.isclose(w2c[:, 3], [0, 0, 0, 1], atol=1e-5), axis=-1))
    video_key = f'observation.images.{camera}'
    height, width = info['features'][video_key]['shape'][:2]
    fallback = np.zeros((n, 2), bool)
    origin_error = np.zeros((n, 2))
    limitations = ['Representative state window only; overlay plausibility is not calibration ground truth.',
                   'No robot joint supervision or robot FK claim for these sources.']
    if name == 'genrobot':
        stamp_path = RELEASE / 'traj2d_valid/Genrobot/stamp.json'
        spec = json.loads(stamp_path.read_text())['spec']
        offset = np.asarray(spec['tcp_offset_m'])
        tcp = eef.copy()
        tcp[..., :3, 3] += np.einsum('nsij,j->nsi', eef[..., :3, :3], offset)
        tcp_valid = eef_valid.copy()
        k_key = 'raw.robot0.sensor.camera2.camera_info.K'
        d_key = 'raw.robot0.sensor.camera2.camera_info.D'
        K = data[k_key].reshape(n, 3, 3)
        distortion = data[d_key]
        np.testing.assert_allclose(distortion[:, :4], K[:, [0, 1, 0, 1], [0, 1, 2, 2]], atol=1e-4)
        distortion = distortion[:, 4:6]
        model, order = 'double_sphere', ['xi', 'alpha']
        tcp_status = 'candidate_fixed_offset_orientation_assumed'
        tcp_label = 'TCP candidate (offset only; orientation assumed)'
        eef_label = 'Finger device base_link (source EEF)'
        limitations += ['TCP offset comes from existing projection sidecar, not independently verified physical TCP calibration.',
                        'Assume offset is expressed in each EEF local frame for both hands; preserve EEF rotation. No confirmed TCP orientation.',
                        'Use full-resolution camera_info K/D, not camera_info_resize calibration. Raw D order is fx,fy,cx,cy,xi,alpha; output distortion retains xi,alpha with fx,fy,cx,cy in K.']
        calibration_sources = [str(stamp_path), str(FACTORY / 'data_factory/data_pipeline/genrobot/lerobot_writer.py')]
        tcp_provenance = {'offset_m': offset.tolist(), 'offset_frame': 'EEF local (assumed)',
                          'rotation': 'identity EEF-to-TCP rotation assumed', 'source': str(stamp_path)}
    else:
        tcp, tcp_valid = _poses(data['observation.states.eef_tcp'])
        geometry, fallback, origin_error = _valid_geometry(data, eef, tcp)
        tcp_valid &= eef_valid & geometry
        tcp_status, tcp_label, eef_label = 'human_virtual_tcp', 'Human virtual pinch TCP', 'Wrist (source EEF)'
        tcp_provenance = info['starvla_metadata']['supplementary_fields']['observation.states.eef_tcp']
        limitations += ['Virtual pinch frame is articulated human geometry, not a fixed wrist offset or robot grasp TCP.',
                        'Finite-pose checks, landmark nondegeneracy, pinch-midpoint agreement and TCP==wrist fallback detection supplement available source masks.']
        if name == 'lightwheel':
            k_key = 'observation.states.cameras.head_left.intrinsic'
            d_key = None
            K = data[k_key].reshape(n, 3, 3)
            distortion = np.empty((n, 0))
            model, order = 'pinhole', []
            calibration_sources = [str(RELEASE / 'traj2d_valid/LightWheel/stamp.json'), str(FACTORY / 'sources/lightwheel_egopro/convert_lerobot.py')]
            limitations += ['Released head_left calibration and existing projection policy specify pinhole K only; no distortion parameters or raw lens model available in this shard.',
                            'No source hand/camera validity or confidence masks exist in this LightWheel schema; geometry checks cannot recover unknown tracking failures.']
        else:
            k_key = d_key = 'observation.states.camera.intrinsic'
            raw = data[k_key]
            if raw.shape != (n, 8):
                raise ValueError('Selected mono Mecka camera must use 8-value intrinsics')
            K = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
            K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2] = raw[:, :4].T
            distortion = np.column_stack((raw[:, 4:8], np.zeros(n)))
            model, order = 'brown_conrady', ['k1', 'k2', 'p1', 'p2', 'k3']
            calibration_sources = [str(root / 'meta/info.json'), str(root / 'meta/source_stage2.json')]
            limitations += ['Per-frame mono Brown-Conrady values override representative info.json summary; k3=0 because the source supplies only k1,k2,p1,p2.',
                            'Existing mecka horizon-50 sidecar uses pinhole projection; it is not evidence of native distortion-correct projection.']
    camera_valid &= (np.isfinite(K).all(axis=(-1, -2)) & (K[:, 0, 0] > 0) & (K[:, 1, 1] > 0)
                     & np.isfinite(distortion).all(-1))
    all_valid = eef_valid.all(1) & tcp_valid.all(1) & camera_valid
    if horizon < 32:
        raise ValueError('Snapshot extraction requires horizon >= 32')
    if require_all_valid:
        candidates = [i for i in range(n - horizon) if all_valid[i:i+horizon+1].all()
                      and np.all(np.diff(data['frame_index'][i:i+horizon+1]) == 1)]
        if not candidates:
            raise ValueError(f'{name}: no contiguous fully valid {horizon+1}-row window in bounded sample')
        anchor_row = min(candidates, key=lambda i: abs(int(data['frame_index'][i]) - anchor_frame))
    else:
        candidates = np.flatnonzero(data['frame_index'] == anchor_frame)
        if len(candidates) != 1:
            raise ValueError(f'{name}: anchor frame {anchor_frame} not unique/present')
        anchor_row = int(candidates[0])
    selection = slice(anchor_row, min(n, anchor_row + horizon + 1))
    selected_indices = data['frame_index'][selection]
    if len(selected_indices) < 33 or not np.all(np.diff(selected_indices) == 1):
        raise ValueError(f'{name}: selected window must contain at least 33 consecutive rows')
    target = output_root / name
    target.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target / 'arrays.npz', eef_world=eef[selection], tcp_world=tcp[selection],
                        eef_valid=eef_valid[selection], tcp_valid=tcp_valid[selection],
                        world_to_camera=w2c[selection], K=K[selection], distortion=distortion[selection],
                        camera_valid=camera_valid[selection], frame_indices=data['frame_index'][selection],
                        tcp_fallback=fallback[selection], pinch_origin_error_m=origin_error[selection])
    prefix = f'videos/{video_key}/'
    video_path = root / episode.get(prefix + 'path', info['video_path'].format(
        video_key=video_key, chunk_index=episode[prefix+'chunk_index'], file_index=episode[prefix+'file_index']))
    images = _images(video_path, episode, video_key, data, selection, target, (width, height))
    metadata = dict(name=name, dataset_dir=str(root), episode=int(episode['episode_index']),
                    episode_index=int(episode['episode_index']), anchor=int(data['frame_index'][anchor_row]),
                    horizon=len(selected_indices)-1, frame_indices=selected_indices.tolist(),
                    model=model, distortion_order=order, image_width=width, image_height=height,
                    nativewh=[width, height], camera_key=video_key, video_path=str(video_path),
                    parquet_path=str(parquet_path), pose_frame='world', pose_units='meter',
                    pose_source='observation.states.eef', tcp_status=tcp_status,
                    endpoint_labels={'eef': eef_label, 'tcp': tcp_label}, eef_label=eef_label, tcp_label=tcp_label,
                    axis_convention=info['starvla_metadata']['coordinate_system'],
                    rotation_convention='first two matrix columns; no transpose; third column = cross(first,second)',
                    tcp_provenance=tcp_provenance, source_calibration_keys=[k for k in (k_key, d_key, 'observation.states.camera.world_to_camera') if k],
                    calibration_sources=calibration_sources, source_info=str(root / 'meta/info.json'),
                    source_stage2=str(root / 'meta/source_stage2.json'), source_mask_keys=mask_keys,
                    camera_mask_keys=camera_mask_keys, limitations=limitations,
                    future_label_semantics=f'arrays[0] = state t; arrays[1:] = future states t+1 through t+{len(selected_indices)-1}, within one episode.',
                    fallback_count=int(fallback[selection].sum()),
                    pinch_origin_error_max_m=float(origin_error[selection].max()),
                    inspected_rows=n, selected_valid_hand_states=int(tcp_valid[selection].sum()),
                    source_video_first_pts_s=episode.get('source_video_first_pts_s'), **images)
    (target / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata


def extract_all(output_root=Path('outputs/fk_consistency/projection_samples')):
    return {name: extract_one(name, Path(output_root)) for name in SOURCES}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=Path('outputs/fk_consistency/projection_samples'))
    args = parser.parse_args()
    result = extract_all(args.output_root)
    print(json.dumps({k: {f: v[f] for f in ('anchor', 'model', 'nativewh', 'displaywh', 'fallback_count')}
                      for k, v in result.items()}, indent=2))
