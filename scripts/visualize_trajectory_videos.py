#!/usr/bin/env python3
"""Continuous source-FPS GT/relative/FK videos in the portable coverage report.

Run after ``python -m scripts.data.video_projection_samples``. All pose streams
are source states, never model predictions. Relative representations are explicit
roundtrip diagnostics on matched targets, not a claim that Mora actions.eef has
the same temporal alignment. Source datasets and training readers are untouched.
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import re
from fractions import Fraction
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import av
import cv2
import numpy as np

from scripts.data.tcp_endpoint_policy import apply_endpoint_policy
from scripts.data.trajectory_projection import project_points, transform_points
from scripts.data.video_fk import load_fk_tracks
from scripts.data.video_trajectory_geometry import decode_relative, encode_relative, future_indices

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'outputs/fk_consistency'
NAMES = ('interndata', 'agibotworld', 'robodojo', 'robotwin', 'genrobot', 'lightwheel', 'mecka')
GT_COLORS = ((235, 218, 30), (35, 156, 255))  # BGR: cyan/orange, left/right.
FK_COLORS = ((95, 235, 95), (235, 90, 235))
WIDTH, HEIGHT = 640, 600


def text(image, message, xy, color=(225, 232, 240), size=.48):
    cv2.putText(image, str(message), xy, cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)


def panel(title, subtitle=''):
    image = np.full((HEIGHT, WIDTH, 3), (28, 23, 19), dtype=np.uint8)
    text(image, title, (14, 25), size=.61)
    text(image, subtitle, (14, 48), size=.42)
    return image


def image_panel(image, title, subtitle):
    canvas = panel(title, subtitle)
    scale = min(WIDTH / image.shape[1], 460 / image.shape[0])
    width, height = round(image.shape[1]*scale), round(image.shape[0]*scale)
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    x, y = (WIDTH-width)//2, 58 + (460-height)//2
    canvas[y:y+height, x:x+width] = resized
    return canvas, (x, y, width, height)


def pixels(poses, valid, arrays, metadata, camera_index, rectangle, model=None):
    """Project one matched world track into one displayed camera snapshot."""
    shape = poses.shape[:-2]
    if not arrays['camera_valid'][camera_index]:
        return np.full((*shape, 2), np.nan), np.zeros(shape, bool)
    model = model or metadata['model']
    points = transform_points(arrays['world_to_camera'][camera_index], poses[..., :3, 3])
    distortion = None if model == 'pinhole' else arrays['distortion'][camera_index]
    uv, good = project_points(points, arrays['K'][camera_index], model, distortion)
    native_w, native_h = metadata['image_width'], metadata['image_height']
    good &= valid & (uv[..., 0] >= 0) & (uv[..., 0] < native_w) & (uv[..., 1] >= 0) & (uv[..., 1] < native_h)
    x, y, width, height = rectangle
    uv = uv * [width/native_w, height/native_h] + [x, y]
    uv[~good] = np.nan
    return uv, good


def lines(canvas, points, valid, colors=GT_COLORS, dashed=False, markers=False, thickness=2):
    """Invalid rows break tracks; never join across a missing observation."""
    for side, color in enumerate(colors):
        for i in range(1, len(points)):
            if valid[i-1, side] and valid[i, side] and (not dashed or i % 4 < 2):
                cv2.line(canvas, tuple(np.rint(points[i-1, side]).astype(int)),
                         tuple(np.rint(points[i, side]).astype(int)), color, thickness, cv2.LINE_AA)
        if markers:
            for i in range(0, len(points), 4):
                if valid[i, side]:
                    cv2.circle(canvas, tuple(np.rint(points[i, side]).astype(int)), 3, color, 1, cv2.LINE_AA)
        if valid[0, side]:
            cv2.drawMarker(canvas, tuple(np.rint(points[0, side]).astype(int)), color,
                           cv2.MARKER_SQUARE, 9, 2, cv2.LINE_AA)
        if valid[-1, side]:
            cv2.drawMarker(canvas, tuple(np.rint(points[-1, side]).astype(int)), color,
                           cv2.MARKER_STAR, 11, 1, cv2.LINE_AA)


def draw_track(canvas, rectangle, poses, valid, arrays, metadata, camera, **kwargs):
    model = kwargs.pop('model', None)
    uv, good = pixels(poses, valid, arrays, metadata, camera, rectangle, model)
    lines(canvas, uv, good, **kwargs)
    return good.sum(0).tolist()


def draw_axes(canvas, rectangle, poses, valid, arrays, metadata, camera):
    """Current TCP orientation triads, 3cm physical axes, with image masks."""
    ends = np.repeat(poses[None], 4, axis=0)
    for axis in range(3):
        ends[axis+1, :, :3, 3] += poses[:, :3, axis] * .03
    uv, good = pixels(ends, np.broadcast_to(valid, (4, 2)), arrays, metadata, camera, rectangle)
    for side in range(2):
        for axis, color in enumerate(((70, 70, 255), (80, 230, 80), (255, 130, 70))):
            if good[0, side] and good[axis+1, side]:
                cv2.line(canvas, tuple(np.rint(uv[0, side]).astype(int)),
                         tuple(np.rint(uv[axis+1, side]).astype(int)), color, 1, cv2.LINE_AA)


def bounds(*tracks):
    points = np.concatenate([np.asarray(a).reshape(-1, 3) for a in tracks])
    finite = points[np.isfinite(points).all(-1)]
    if not len(finite):
        return np.zeros(3), .1
    center = (finite.min(0) + finite.max(0))/2
    radius = max(float(np.ptp(finite, axis=0).max())/2, .015)
    return center, radius*1.18


def plot3d(title, subtitle, tracks, center, radius, footer):
    """Metric orthographic 3D animation; this is NOT a camera pixel projection."""
    canvas = panel(title, subtitle)
    azimuth, elevation = np.deg2rad([35, 25])
    basis = np.array([[np.cos(azimuth), -np.sin(azimuth), 0],
                      [-np.sin(elevation)*np.sin(azimuth), -np.sin(elevation)*np.cos(azimuth), np.cos(elevation)]])
    def project(points):
        uv = (points-center) @ basis.T * (180/radius)
        return uv * [1, -1] + [320, 300]
    origin = project(center)
    for axis, color in enumerate(((80, 80, 240), (80, 210, 80), (240, 140, 80))):
        endpoint = project(center + np.eye(3)[axis]*radius*.65)
        cv2.arrowedLine(canvas, tuple(origin.astype(int)), tuple(endpoint.astype(int)), color, 1, cv2.LINE_AA, tipLength=.08)
        text(canvas, 'XYZ'[axis], tuple(endpoint.astype(int)+[3, 0]), color)
    for points, valid, colors, dashed in tracks:
        points = np.asarray(points).copy()
        points[~valid] = np.nan
        lines(canvas, project(points), valid, colors=colors, dashed=dashed)
    text(canvas, f'Equal metric scale; axis arrow = {radius*.65:.3f} m', (14, 538))
    text(canvas, f'Plot center XYZ = {np.round(center, 3).tolist()} m', (14, 558), size=.43)
    text(canvas, footer, (14, 583), size=.42)
    return canvas


class VideoWriter:
    """H.264/yuv420p + faststart, using every source frame at its source rate."""
    def __init__(self, path, fps):
        self.path = path
        self.container = av.open(str(path), 'w', options={'movflags': '+faststart'})
        self.stream = self.container.add_stream('libx264', rate=Fraction(str(fps)).limit_denominator(100000))
        self.stream.width, self.stream.height = WIDTH*2, HEIGHT
        self.stream.pix_fmt = 'yuv420p'
        self.stream.options = {'crf': '23', 'preset': 'fast', 'threads': '2'}
        self.count = 0

    def write(self, frame):
        video_frame = av.VideoFrame.from_ndarray(frame, format='bgr24')
        video_frame.pts = self.count
        self.count += 1
        for packet in self.stream.encode(video_frame):
            self.container.mux(packet)

    def close(self):
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()


def joint_history_panel(joints, valid, side, current, fps):
    """Plot actual joint angles when no matching URDF can supply a pose."""
    canvas = panel(('LEFT' if side == 0 else 'RIGHT')+' JOINT TRAJECTORY',
                   f'{current/fps:.2f}s | radians | no FK or TCP inferred')
    values = joints[:, side*7:(side+1)*7].copy()
    mask = valid[:, side*7:(side+1)*7]
    values[~mask] = np.nan
    finite = values[np.isfinite(values)]
    low, high = (float(finite.min())-.1, float(finite.max())+.1) if len(finite) else (-1, 1)
    colors = [(255, 180, 70), (80, 200, 255), (100, 240, 100), (230, 100, 230),
              (100, 100, 255), (220, 220, 80), (230, 230, 230)]
    cv2.rectangle(canvas, (50, 140), (620, 490), (100, 100, 100), 1)
    for joint, color in enumerate(colors):
        text(canvas, f'q{joint+1}: {values[current, joint]:.3f}', (14+(joint%4)*153, 80+(joint//4)*24), color, size=.4)
        for row in range(1, current+1):
            if mask[row-1:row+1, joint].all():
                points = [(round(50+570*t/max(1, len(values)-1)),
                           round(490-350*(values[t, joint]-low)/(high-low))) for t in [row-1, row]]
                cv2.line(canvas, *points, color, 1, cv2.LINE_AA)
    text(canvas, f'Y: {low:.2f}..{high:.2f} rad; X: 0..{(len(values)-1)/fps:.2f}s', (14, 540), size=.45)
    text(canvas, 'Recorded arm joints only. Missing robot model is not substituted.', (14, 573), size=.36)
    return canvas


def render_sample(name, output=OUTPUT):
    folder = output/'video_samples'/name
    metadata = json.loads((folder/'metadata.json').read_text())
    with np.load(folder/'arrays.npz') as data:
        arrays = dict(data)
    apply_endpoint_policy(name, arrays, metadata)
    count = len(arrays['frame_indices'])
    if len(metadata['image_filenames']) != count or not np.all(np.diff(arrays['frame_indices']) == 1):
        raise ValueError(f'{name}: images and contiguous pose rows must match')
    fk, fk_meta = load_fk_tracks(name, arrays['frame_indices'], arrays, metadata, root=output)
    has_fk = bool(fk['fk_local_valid'].any())
    out = output/'trajectory_videos'/name
    out.mkdir(parents=True, exist_ok=True)
    modes = ['head', 'projection', 'relative'] + (['fk'] if has_fk else [])
    if not has_fk and fk['fk_joint_valid'].any():
        modes.append('joint')
    if metadata.get('origin_diagnostics'):
        modes.append('origin')
    writers = {mode: VideoWriter(out/f'{mode}.mp4', metadata['fps']) for mode in modes}
    anchor_image = cv2.imread(str(folder/metadata['image_filenames'][0]))
    # Each relative branch uses the same endpoint; invalid TCP never becomes EEF.
    stream = 'tcp' if arrays['tcp_valid'].any() else 'eef'
    endpoint = 'TCP' if stream == 'tcp' else 'EEF (NO TCP; diagnostic)'
    if metadata['tcp_status'].startswith('candidate'):
        endpoint += ' CANDIDATE'
    if metadata['tcp_status'] == 'visual_tcp_preview':
        endpoint += ' (CAD + manual visual preview)'
    relative_bounds = {}
    encoded_all = {mode: [] for mode in ('head', 'local')}
    for i in range(count):
        ids = future_indices(i, count)
        valid = arrays[f'{stream}_valid'][ids] & arrays[f'{stream}_valid'][i]
        for mode in encoded_all:
            encoded = encode_relative(arrays[f'{stream}_world'][ids], arrays[f'{stream}_world'][i], arrays['world_to_camera'][i], mode)
            encoded[~valid] = np.nan
            if mode == 'head' and not arrays['camera_valid'][i]:
                encoded[:] = np.nan
            encoded_all[mode].append(encoded)
    finite_deltas = np.concatenate([x[..., :3, 3].reshape(-1, 3)
                                    for values in encoded_all.values() for x in values])
    finite_deltas = finite_deltas[np.isfinite(finite_deltas).all(-1)]
    common_radius = max(float(np.linalg.norm(finite_deltas, axis=-1).max()) if len(finite_deltas) else 0., .015)*1.1
    for mode in encoded_all:
        relative_bounds[mode] = np.zeros(3), common_radius
    if has_fk:
        local_label = fk['label_local_pose'][..., :3, 3].copy()
        local_fk = fk['fk_local_pose'][..., :3, 3].copy()
        local_label[~fk['label_local_valid']] = np.nan
        local_fk[~fk['fk_local_valid']] = np.nan
        fk_bounds = bounds(local_label, local_fk, fk.get('fk_tcp_candidate_local', np.full((1, 2, 4, 4), np.nan))[..., :3, 3])
    max_errors = {'head_position_m': 0., 'local_position_m': 0., 'head_rotation_element': 0., 'local_rotation_element': 0.}
    seen = {'head': 0, 'local': 0}
    in_frame = []
    try:
        for i in range(count):
            image = cv2.imread(str(folder/metadata['image_filenames'][i]))
            if image is None:
                raise ValueError(f'{name}: missing frame image {i}')
            ids = future_indices(i, count)
            target_range = f'+1..+{len(ids)-1}' if len(ids) > 1 else 'no future target'
            clip_label = metadata.get('clip_label', name)
            timing = f'{clip_label} | state {arrays["frame_indices"][i]} | {target_range} | {i/metadata["fps"]:.2f}s'
            # Absolute source endpoints vs the same endpoint reconstructed from deltas.
            left, rect = image_panel(image, 'HEAD CAMERA: wrist/EEF vs TCP', timing)
            counts = {}
            for key, dashed in [('eef', True), ('tcp', False)]:
                counts[key] = draw_track(left, rect, arrays[f'{key}_world'][ids], arrays[f'{key}_valid'][ids], arrays, metadata, i, dashed=dashed)
            draw_axes(left, rect, arrays['tcp_world'][i], arrays['tcp_valid'][i], arrays, metadata, i)
            text(left, 'L cyan / R orange | EEF dashed | TCP solid', (14, 539))
            text(left, f'In image L/R: EEF {counts["eef"]} / TCP {counts["tcp"]}', (14, 560), size=.43)
            text(left, f'TCP: {metadata["tcp_status"]}', (14, 582), size=.43)
            if not arrays['camera_valid'][i]:
                text(left, 'CAMERA PROJECTION MASKED: missing/invalid calibration', (14, 80), (80, 180, 255), size=.42)
            right, rect2 = image_panel(image, 'ABSOLUTE vs RELATIVE -> WORLD', timing)
            source = arrays[f'{stream}_world'][ids]
            source_valid = arrays[f'{stream}_valid'][ids]
            draw_track(right, rect2, source, source_valid, arrays, metadata, i)
            for mode, color in [('head', (255, 255, 255)), ('local', (225, 100, 225))]:
                encoded = encoded_all[mode][i]
                decoded = decode_relative(encoded, arrays[f'{stream}_world'][i], arrays['world_to_camera'][i], mode)
                valid = source_valid & arrays[f'{stream}_valid'][i]
                if mode == 'head':
                    valid &= bool(arrays['camera_valid'][i])
                if valid.any():
                    max_errors[mode+'_position_m'] = max(max_errors[mode+'_position_m'], float(np.linalg.norm(decoded[..., :3, 3]-source[..., :3, 3], axis=-1)[valid].max()))
                    max_errors[mode+'_rotation_element'] = max(max_errors[mode+'_rotation_element'], float(np.max(np.abs(decoded[..., :3, :3]-source[..., :3, :3])[valid])))
                    seen[mode] += int(valid.sum())
                uv, good = pixels(decoded, valid, arrays, metadata, i, rect2)
                for arm in range(2):
                    for j in range(0 if mode == 'head' else 2, len(ids), 4):
                        if good[j, arm]:
                            cv2.drawMarker(right, tuple(np.rint(uv[j, arm]).astype(int)), color,
                                           cv2.MARKER_CROSS if mode == 'local' else cv2.MARKER_DIAMOND, 7, 1, cv2.LINE_AA)
            text(right, 'Color: absolute; white: head restore; purple: local restore', (14, 539), size=.39)
            text(right, f'SAME endpoint: {endpoint}', (14, 560), size=.42)
            text(right, 'Overlap is expected. GT conversion, NOT model predictions.', (14, 582), size=.41)
            writers['head'].write(np.hstack((left, right)))
            in_frame.append(counts)
            # Fixed initial camera stays paired with its own background image.
            fixed, rect0 = image_panel(anchor_image, 'FIXED initial head camera + image', timing)
            moving, rectm = image_panel(image, 'CURRENT head: native lens vs pinhole', timing)
            poses, valid = arrays[f'{stream}_world'][ids], arrays[f'{stream}_valid'][ids]
            draw_track(fixed, rect0, poses, valid, arrays, metadata, 0)
            draw_track(moving, rectm, poses, valid, arrays, metadata, i)
            if metadata['model'] != 'pinhole':
                draw_track(moving, rectm, poses, valid, arrays, metadata, i, model='pinhole', colors=((210, 210, 210),)*2, dashed=True)
            text(fixed, f'Same rolling future world track: {endpoint}', (14, 539), size=.41)
            text(fixed, 'Static background is intentional: one fixed camera reference.', (14, 560), size=.4)
            text(fixed, 'Moving square = current state. Star = last real target.', (14, 582), size=.41)
            text(moving, f'Color: source model {metadata["model"]}', (14, 539), size=.45)
            text(moving, 'Gray dashed: ignoring distortion (diagnostic)' if metadata['model'] != 'pinhole' else 'Pinhole preview; distortion treatment documented in report.', (14, 560), size=.41)
            text(moving, 'Same endpoint / world trajectory; matched camera and image.', (14, 582), size=.4)
            writers['projection'].write(np.hstack((fixed, moving)))
            # Units in meters, equal metric axes; both representations at same t.
            panes = []
            for mode, label in [('head', 'HEAD-reference relative XYZ'), ('local', 'CURRENT endpoint-local relative XYZ')]:
                values = encoded_all[mode][i][..., :3, 3]
                valid = np.isfinite(values).all(-1)
                panes.append(plot3d(label, timing, [(values, valid, GT_COLORS, False)], *relative_bounds[mode],
                                    f'{endpoint} | units m | square = origin t | not pixels'))
            writers['relative'].write(np.hstack(panes))
            if 'joint' in writers:
                writers['joint'].write(np.hstack([joint_history_panel(
                    fk['fk_joint'], fk['fk_joint_valid'], side, i, metadata['fps']) for side in [0, 1]]))
            if 'origin' in writers:
                original, recto = image_panel(image, 'ORIGIN AUDIT: source wrist/flange', timing)
                origins, rectd = image_panel(image, 'ORIGIN CANDIDATES: NOT full TCP labels', timing)
                draw_track(original, recto, arrays['eef_world'][ids], arrays['eef_valid'][ids], arrays, metadata, i, dashed=True)
                for number, diagnostic in enumerate(metadata['origin_diagnostics']):
                    candidate = arrays['eef_world'][ids].copy()
                    candidate[..., :3, 3] += np.einsum('nsij,j->nsi', candidate[..., :3, :3], diagnostic['offset_local_m'])
                    color = tuple(diagnostic['color_bgr'])
                    draw_track(origins, rectd, candidate, arrays['eef_valid'][ids], arrays, metadata, i, colors=(color, color), markers=True)
                    text(origins, diagnostic['label'], (14, 539+21*number), color, size=.39)
                text(original, 'L cyan / R orange | dashed: source EEF', (14, 539), size=.44)
                if name == 'agibotworld':
                    quality = metadata.get('head_calibration_quality', 'unknown')
                    text(original, f'Head calibration source audit: {quality}', (14, 560), (90, 180, 255), size=.41)
                    text(original, '+100mm named center lies near housing, not contact TCP.', (14, 582), size=.4)
                    text(origins, '+28mm mount alone does not establish contact TCP.', (14, 582), size=.4)
                else:
                    text(original, 'Source TCP fields omitted during Stage3 conversion.', (14, 560), size=.41)
                    text(original, 'Tool position candidate shown separately; rotation unproven.', (14, 582), size=.39)
                    text(origins, 'Point can be visualized; full TCP pose remains masked.', (14, 582), size=.41)
                if metadata['tcp_status'] == 'visual_tcp_preview':
                    # Compare the old named centers to the requested virtual
                    # grasp point, highlighting t separately from its future.
                    original, recto = image_panel(image, 'OLD named centers: +100 / +128 mm', timing)
                    for diagnostic in metadata['origin_diagnostics']:
                        candidate = arrays['eef_world'][ids].copy()
                        candidate[..., :3, 3] += np.einsum('nsij,j->nsi', candidate[..., :3, :3], diagnostic['offset_local_m'])
                        color = tuple(diagnostic['color_bgr'])
                        draw_track(original, recto, candidate, arrays['eef_valid'][ids], arrays, metadata, i, colors=(color, color), markers=True)
                    origins, rectd = image_panel(image, 'NEW virtual TCP: wrist local +X 200 mm', timing)
                    draw_track(origins, rectd, arrays['tcp_world'][ids], arrays['tcp_valid'][ids], arrays, metadata, i)
                    uv, valid = pixels(arrays['tcp_world'][i:i+1], arrays['tcp_valid'][i:i+1], arrays, metadata, i, rectd)
                    for arm, label in enumerate(('L TCP t', 'R TCP t')):
                        if valid[0, arm]:
                            x, y = np.rint(uv[0, arm]).astype(int)
                            cv2.circle(origins, (x, y), 9, GT_COLORS[arm], 2, cv2.LINE_AA)
                            cv2.drawMarker(origins, (x, y), GT_COLORS[arm], cv2.MARKER_CROSS, 13, 1)
                            text(origins, label, (max(10, min(x+12, 550)), max(75, y-12)), GT_COLORS[arm], size=.4)
                    text(original, 'Red: 100mm / yellow: 128mm (NOT fingertip centers)', (14, 539), size=.4)
                    text(original, 'Named URDF frames were near the gripper housing.', (14, 560), size=.42)
                    text(original, f'Head source audit: {metadata.get("head_calibration_quality", "unknown")}', (14, 582), size=.42)
                    text(origins, 'L cyan / R orange: ring = current TCP t; line = future', (14, 539), size=.4)
                    text(origins, 'CAD reference + manual visual adjustment; axes unchanged', (14, 560), size=.38)
                    text(origins, 'PREVIEW, not factory calibration. Training unchanged.', (14, 582), size=.4)
                writers['origin'].write(np.hstack((original, origins)))
            if has_fk:
                fk_panel, rectf = image_panel(image, 'HEAD CAMERA: source GT vs joint -> FK', timing)
                for key, dashed in [('eef', True), ('tcp', False)]:
                    draw_track(fk_panel, rectf, arrays[f'{key}_world'][ids], arrays[f'{key}_valid'][ids], arrays, metadata, i, dashed=dashed, thickness=1)
                    draw_track(fk_panel, rectf, fk[f'fk_{key}_world'][ids], fk[f'fk_{key}_valid'][ids], arrays, metadata, i,
                               colors=FK_COLORS, dashed=dashed, markers=True)
                text(fk_panel, 'GT L/R cyan/orange; FK L/R green/pink', (14, 539), size=.45)
                text(fk_panel, 'EEF dashed / TCP solid; no verified transform => no FK TCP', (14, 560), size=.38)
                if not fk['fk_eef_valid'][i].any():
                    text(fk_panel, 'WORLD FK MASKED: see declared local frame and missing chain', (14, 582), (100, 180, 255), size=.38)
                elif name == 'interndata':
                    text(fk_panel, 'Base/tool transforms derived from source row0: diagnostic', (14, 582), (100, 180, 255), size=.39)
                elif name in ('robotwin', 'robodojo'):
                    text(fk_panel, 'USER PREVIEW: wrist local +X 0.12m; same rotation.', (14, 582), size=.41)
                else:
                    text(fk_panel, str(fk_meta.get('overlay_note', 'See report for FK endpoint and base/tool provenance.'))[:89], (14, 582), size=.37)
                history = slice(0, i+1)
                tracks = [(local_label[history], fk['label_local_valid'][history], GT_COLORS, False),
                          (local_fk[history], fk['fk_local_valid'][history], FK_COLORS, True)]
                if 'fk_tcp_candidate_local' in fk:
                    tracks.append((fk['fk_tcp_candidate_local'][history, :, :3, 3],
                                   fk['fk_tcp_candidate_local_valid'][history], ((0, 220, 255),)*2, False))
                fk3d = plot3d('3D: GT vs URDF FK (growing path)', timing, tracks, *fk_bounds,
                              'Yellow: mesh endpoint candidate, NOT verified TCP.' if 'fk_tcp_candidate_local' in fk else
                              'Left/right colors as image; common scale in meters.')
                # Show actual joint values to connect this animation to the FK input.
                text(fk3d, 'qL(rad): '+np.array2string(fk['fk_joint'][i, :7], precision=2, suppress_small=True), (14, 78), size=.38)
                text(fk3d, 'qR(rad): '+np.array2string(fk['fk_joint'][i, 7:], precision=2, suppress_small=True), (14, 98), size=.38)
                text(fk3d, str(fk_meta['local_frame'])[:86], (14, 120), (100, 180, 255), size=.35)
                writers['fk'].write(np.hstack((fk_panel, fk3d)))
            if i in (0, count//2, count-1):
                cv2.imwrite(str(out/f'preview_{i}.jpg'), np.hstack((left, right)))
            if i % 60 == 0:
                print(f'{name}: rendered {i+1}/{count}', flush=True)
    finally:
        for writer in writers.values():
            writer.close()
    fk_errors = {}
    for label, expected, actual, valid in (
        ('local_or_interarm_eef', fk['label_local_pose'], fk['fk_local_pose'], fk['label_local_valid'] & fk['fk_local_valid']),
        ('world_eef', arrays['eef_world'], fk['fk_eef_world'], arrays['eef_valid'] & fk['fk_eef_valid']),
        ('world_tcp', arrays['tcp_world'], fk['fk_tcp_world'], arrays['tcp_valid'] & fk['fk_tcp_valid']),
    ):
        if (label == 'world_eef' and metadata.get('eef_is_fk_derived')
                or label == 'world_tcp' and metadata.get('tcp_is_fk_derived')):
            valid = np.zeros_like(valid)
        distance = np.linalg.norm(expected[..., :3, 3] - actual[..., :3, 3], axis=-1)*1000
        fk_errors[label] = {'valid_count_left_right': valid.sum(0).tolist(),
                            'max_position_mm_left_right': [float(distance[:, side][valid[:, side]].max())
                                                           if valid[:, side].any() else None for side in range(2)]}
    result = {'name': name, 'frames': count, 'fps': metadata['fps'], 'duration_s': count/metadata['fps'],
              'source': metadata, 'fk': fk_meta, 'relative_endpoint': endpoint,
              'fk_clip_errors': fk_errors,
              'roundtrip_max_errors': max_errors, 'roundtrip_valid_pose_count': seen,
              'per_frame_in_image_counts': in_frame,
              'videos': {mode: str((out/f'{mode}.mp4').relative_to(output)) for mode in modes},
              'target_semantics': 'source state t+1..t+32, truncated at sampled segment end; not actions.eef or model prediction'}
    (out/'metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f'{name}: finished {len(modes)} continuous videos', flush=True)
    return result


def video_data(path):
    return 'data:video/mp4;base64,' + base64.b64encode(path.read_bytes()).decode()


def build_user_review(output):
    """Small standalone page for the explicitly requested two-robot acceptance."""
    cards = []
    for name in ('robodojo', 'robotwin'):
        metrics = json.loads((output/'trajectory_videos'/name/'metrics.json').read_text())
        if metrics['source']['tcp_status'] != 'user_defined_tcp_preview':
            return
        players = []
        for mode, label in [('head', '腕部与 TCP；同端点 relative 还原对照'),
                            ('fk', 'joint → URDF → FK → TCP；右侧为 FK/EEF 三维轨迹')]:
            players.append('<h3>'+label+'</h3><video controls playsinline loop preload="none" src="'+
                           video_data(output/metrics['videos'][mode])+'"></video>')
        cards.append(f'<section id="{name}"><h2>{name}</h2><p>{metrics["frames"]} 连续帧，{metrics["fps"]:g} FPS，{metrics["duration_s"]:.1f} 秒。用户已视觉确认。</p>'+''.join(players)+'</section>')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RoboDojo / RobotWin +120mm TCP 验收</title>
<style>body{font:16px/1.7 system-ui,sans-serif;max-width:1280px;margin:auto;padding:24px;background:#edf3f8;color:#182d40}section{background:white;padding:20px;margin:25px 0;border-radius:12px}video{width:100%;background:#111820}a{color:#006e9c}code{background:#dde8f0;padding:5px}h3{font-size:18px}</style>
<h1>RoboDojo / RobotWin：腕部局部 +X 120mm TCP 验收版</h1>
<p>两组均按你指定的定义绘制：<code>T_world_TCP = T_world_wrist × TransX(+0.12m)</code>，旋转不变。源 EEF 与 joint→URDF/FK 使用相同偏移。此页仅用于视频验收，训练 reader 配置未变。</p>
<p>青色/橙色＝左/右 GT；腕部/法兰为虚线，TCP 为实线。FK 视频中绿色/粉色＝左/右 FK。方框＝当前帧，星号＝该窗口最后一个真实目标；轨迹为未来最多32步。</p>
<p><a href="#robodojo">RoboDojo</a> · <a href="#robotwin">RobotWin</a> · <a href="tcp_coverage_v2_v3.html">完整报告</a>。视频可暂停、拖动进度或全屏；下载本 HTML 后也能离线播放。</p>'''+''.join(cards)+'</html>'
    (output/'tcp_120mm_review.html').write_text(page)


def endpoint_audit_html(output):
    """Separate missing columns from a known full transform or position candidate."""
    path = output/'tcp_origin_audit/recoverability.json'
    if not path.exists():
        return ''
    audit = json.loads(path.read_text())
    status_labels = {
        'native_source_tcp': '原生 TCP 字段',
        'explicit_full_transform_contact_center': '完整变换可恢复接触中心',
        'explicit_full_transform_historical_source': '可恢复历史 source TCP 定义',
        'explicit_full_transform_source_gripper_center': '源 helper 定义的夹持中心',
        'origin_only_candidate': '仅候选原点；朝向/来源待核实',
        'known_flange_task_tcp_unresolved': '法兰已知；任务 TCP 未确定',
        'known_wrist_task_tcp_unresolved': '腕部已知；工具 TCP 变换缺失',
        'unknown_mount_link_world_frame': '安装/末端 link/参考系链不完整',
        'human_virtual_tcp_native': '已有虚拟手部 TCP',
        'human_virtual_tcp_derivable': '可由手部关键点派生虚拟 TCP',
        'uninspectable_alias': '别名不可读，未确认',
        'sidecar_not_source': '辅助文件，并非轨迹数据集',
        'recoverable_source_config': '由匹配源码/配置恢复完整 TCP',
    }
    rows = []
    for row in audit['rows']:
        family = row['family'] + (' / '+str(row['variant']) if row.get('variant') else '')
        detail = html.escape(json.dumps(row, ensure_ascii=False, indent=2))
        rows.append('<tr><td>'+html.escape(str(row['version']))+'</td><td>'+html.escape(family)+'</td><td>'+
                    html.escape(status_labels.get(row['recovery_status'], row['recovery_status']))+'</td><td><details><summary>字段、变换及缺失证据</summary><pre>'+detail+'</pre></details></td></tr>')
    source_reports = ''.join('<details><summary>'+html.escape(report.stem)+'</summary><pre>'+html.escape(report.read_text())+'</pre></details>'
                             for report in sorted((output/'tcp_origin_audit').glob('*audit.txt')))
    geometry = output/'tcp_origin_audit/agibotworld/visual_tcp_geometry.json'
    if geometry.exists():
        source_reports = '<details><summary>AgiWorld：URDF公式验证、接触垫几何及200mm视觉预览定义</summary><pre>'+html.escape(geometry.read_text())+'</pre></details>'+source_reports
    return '''<section id="tcp-origin-audit"><h2>2026-10-08 端点排查：缺 TCP 字段 ≠ 不能恢复 TCP</h2>
<p><b>AgiWorld 最新视觉预览：</b>用户授权按CAD与画面调整后，四个A2D样例统一使用腕部canonical局部 +X <b>0.200m</b>、旋转不变。完整G1视觉URDF的零关节接触垫包围盒中心平均约为237.55mm；初版240/245mm主要依据头部画面，腕部复核发现明显过长，本版先缩短至200mm候选。左右腕部图像支持的偏移不同，单一X长度不能消除全部视角误差。100/128mm是旧具名点，保留作前后对照；200mm用于减少腕部过冲，仍保留右腕和头部残差；不能称为已校准的夹持中心。此为CAD参考＋人工目视调整，不是官方标定；接触垫随开合变化、相机误差仍存在。原task512相机审计borderline，新增task362/365/410均normal/acceptable。<a href="#agibot-examples">查看新视频</a>。训练监督未修改。</p>
<p><b>InternData：</b>直接使用原生 TCP pose，并逐帧经过 robot_to_env；不依赖 AgiBot 的具名 URDF center。它叠加正确，不能证明 AgiBot 使用的端点和相机标定也正确。</p>
<p><b>用户已视觉确认：RoboDojo 与 RobotWin</b>，从腕部/法兰沿各自局部 +X 偏移 <b>0.12m</b>，旋转不变：T_world_TCP = T_world_wrist × TransX(+0.12m)。源 EEF 和 joint→URDF/FK 均应用同一变换。<a href="#video-robodojo">查看 RoboDojo</a> · <a href="#video-robotwin">查看 RobotWin</a>。</p>
<p>历史来源审计单独保留：RobotWin 匹配源码/config 支持 +120mm；RoboDojo 曾查到 config 的145mm候选，但本次验收视频已改用用户指定的120mm。以下可恢复性表与原始审计记录描述此前证据，当前显示定义以本段及视频卡片为准。</p>
<p>表格现在区分「原生字段」「源码/CAD完整变换」「仅原点候选」「缺安装或坐标链」「人手关键点派生」。CAD/URDF可给出几何和安装关系，但工具版本、接触点选择、坐标轴、开合依赖仍需明确。此轮只更正报告和样例缓存；训练 reader 的监督配置没有自动扩展。</p>
<details><summary>全部 V0_2 / V0_3 来源及当前 posttrain 的可恢复性（27 项）</summary><table><thead><tr><th>版本</th><th>来源/型号</th><th>可恢复性</th><th>详细证据</th></tr></thead><tbody>'''+''.join(rows)+'''</tbody></table></details>'''+source_reports+'</section>'


def wrist_coverage_html(output):
    path = output/'wrist_samples/coverage.json'
    if not path.exists():
        return ''
    rows = []
    for sample in json.loads(path.read_text()):
        statuses = []
        for side in ['left', 'right']:
            view = sample['views'][side]
            status = ('有相机链；缺TCP定义' if view['projection_available'] and not sample.get('tcp_available', True) else
                      'CAD链候选投影' if view['status'] in ('cad_derived_preview', 'derived_candidate') else 'TCP轨迹可投影' if view['projection_available'] else
                      '有视频；投影mask' if view['video_available'] else '无腕部相机视频')
            statuses.append('<td>'+status+'<br><small>'+html.escape(view['reason'])+'</small></td>')
        target = 'video-'+sample['sample']
        link = f'<a href="#{target}" onclick="openWristView(\'{target}\')">{html.escape(sample["sample"])}</a>'
        rows.append('<tr><td>'+link+'</td>'+''.join(statuses)+'</tr>')
    audits = ''.join('<details><summary>'+title+'</summary><pre>'+html.escape(path.read_text())+'</pre></details>'
                    for title, path in [
                        ('Agi多视角偏移量诊断（人工像素参考，不是标定）', output/'tcp_origin_audit/agibotworld/multiview_offset_audit.json'),
                        ('Genrobot腕部坐标链恢复证据', output/'genrobot_wrist_audit/evidence.txt')]
                    if path.exists())
    return '''<section id="wrist-views"><h2>腕部与来源左右相机中的TCP轨迹</h2>
<p>覆盖原有样例及新增a1、RoboMind v1/v2、RoboChallenge v2、ABC130k各embodiment。点击样例切换左右相机视图；RoboMind的腕部安装未确认，视频内单独标注。青色/橙色分别为左/右TCP。有完整坐标链时，使用各自的K、光学模型与逐帧外参，把同一world/base TCP轨迹投影到来源画面；候选标定与近似光学模型见各条证据。随动视图每帧同步更新相机和背景；固定视图冻结首帧相机及背景。未来最多32步，画面外、相机后方或无效点mask，不检查遮挡。</p>
<p>这与“TCP-local相对坐标”是不同的视图。Agi沿用200mm视觉TCP、RoboDojo/RobotWin沿用120mm定义、InternData使用原生TCP。Genrobot已接入Finger URDF的base_link→cam_Link固定旋转，以CAD推导候选链投影；LightWheel/Mecka当前样例没有腕部相机。</p>
<p>腕部投影采用源数据声明的标定，不代表已独立校准。RobotWin部分时刻没有可见轨迹：例如state 120左腕当前TCP投影约为(321,323)，超出320×240画面，已按视野mask；固定首帧左腕视图在该窗口内也可能全空。Agi的200mm仍是待验收的统一人工偏移，头部与腕部画面的偏差不能单凭延长TCP来消除。</p>
<p><b>本轮复核：</b>Agi task362/frame216的人工末端中点参考下，245mm左右腕像素偏差约50/67px，200mm约10/36px。缩短有改善，但右腕仍不齐；左右图像单独拟合约192/167mm，说明不能宣称一个长度已经修复所有视角。Genrobot上一版缺少base_link到cam_Link的固定轴旋转，因此整段mask；已按Finger URDF恢复并保留源对齐有效性。单位T_b_c本身不代表EEF坐标轴等于光学坐标轴。</p>
<details><summary>全部样例的腕部视频与投影覆盖情况</summary><table><thead><tr><th>样例（点击查看）</th><th>左腕</th><th>右腕</th></tr></thead><tbody>'''+''.join(rows)+'</tbody></table></details>'+audits+'</section>'


def expanded_coverage_html(output, entries):
    """Summarize real endpoint/model/camera gaps separately for every embodiment."""
    rows = []
    for entry in entries:
        root, name = output/entry['root'], entry['family']
        metrics = json.loads((root/'trajectory_videos'/name/'metrics.json').read_text())
        source, fk = metrics['source'], metrics['fk']
        with np.load(root/'video_samples'/name/'arrays.npz') as arrays:
            tcp = arrays['tcp_valid'].sum(0).tolist()
            camera = int(arrays['camera_valid'].sum())
        wrist = metrics['wrist']['views']
        coverage = f'头部有效标定 {camera}/{metrics["frames"]}；腕部链 L/R：'+str([bool(wrist[side]['projection_available']) for side in ['left', 'right']])
        limitation = '; '.join(fk.get('limitations', []))
        evidence = json.dumps(dict(endpoint=source.get('endpoint'), tcp_status=source['tcp_status'],
            fk=fk, lens_note=source.get('lens_note'), camera_missing_reason=source.get('camera_missing_reason'),
            source_variant=source.get('source_variant')), ensure_ascii=False, indent=2)
        rows.append(f'<tr><td><a href="#video-{entry["id"]}">{html.escape(entry["title"])}</a></td>'
                    f'<td>{html.escape(source["tcp_status"])}<br>有效TCP帧 L/R {tcp}</td>'
                    f'<td>{html.escape(coverage)}</td><td>{html.escape(str(fk["local_frame"]))}<br>{html.escape(limitation)}</td>'
                    '<td><details><summary>URDF/模型、固定变换、掩码与来源</summary><pre>'+html.escape(evidence)+'</pre></details></td></tr>')
    return '''<section id="expanded-datasets"><h2>本次补充：5类来源、9种 embodiment</h2>
<p>前一版只选了部分代表样例，遗漏不等于数据缺TCP。此次按指定Mora审计中的embodiment逐一取样，检查当前release实际字段、URDF/MJCF及转换历史；同名数据源的不同版本不能直接套用旧EEF定义。每例180连续帧，头部/左右相机视频均来自该样例，未以单张图片代替。</p>
<p><b>a1：</b>三种均有原生TCP；世界FK安装/工具链部分由源标签派生，只作诊断，独立检查在arm-base局部系。ARX约55mm局部差异保留。其sim_updated_lerobotv30来源分支已记录，不能据目录名断言视频是实拍或仿真。</p>
<p><b>RoboChallenge v2 / ABC：</b>DOS-W1与YAM通过release转换记录恢复TCP，并核对URDF/MJCF。ALOHA已实际检查指框与STL，但仍无确认的接触TCP；135.03mm指框原点不冒充TCP。Challenge右臂缺独立episode基座标定，world FK mask，局部FK保留。ABC原D405 inverse-Brown系数保留，当前为pinhole近似预览，不宣称像素级畸变校准。</p>
<p><b>RoboMind：</b>三个样例有视频但缺相机外参，不能投影。v1 FR3与v2原字段仅声明左右相机；v1 AgileX虽有wrist命名，也缺安装标定，因此均标注“来源左右相机，腕部安装未确认”。AgileX读取URDF和夹指STL得到约155.57mm的几何末端候选，FK三维图用黄色标注，非已校准TCP；FR3缺匹配模型，不以Panda替代，提供实际关节角动画。缺独立GT时不报告FK与自身的零误差。</p>
<table><thead><tr><th>样例</th><th>TCP</th><th>相机</th><th>FK及限制</th><th>证据</th></tr></thead><tbody>'''+''.join(rows)+'</tbody></table></section>'


def build_report(output=OUTPUT):
    page = output/'tcp_coverage_v2_v3.html'
    current = re.sub(r'<!-- VIDEOS BEGIN -->.*?<!-- VIDEOS END -->', '', page.read_text(), flags=re.S)
    # The user's saved #previous-fk URL now opens the current videos, not a
    # hidden historical iframe. Migrate the old ID once when rebuilding.
    current = current.replace('id="previous-fk"', 'id="archived-fk"')
    current = current.replace('href="#previous-fk"', 'href="#archived-fk"')
    # Keep earlier snapshot material available without making it the main report.
    current = current.replace('<details id="previous-static"><summary>上一版静态投影与完整 FK 报告</summary>', '')
    current = current.replace('<!-- STATIC WRAPPER END --></details>', '')
    current = current.replace('<!-- PROJECTIONS BEGIN -->', '<details id="previous-static"><summary>上一版静态投影与完整 FK 报告</summary><!-- PROJECTIONS BEGIN -->')
    current = current.replace('<!-- PROJECTIONS END -->', '<!-- PROJECTIONS END --><!-- STATIC WRAPPER END --></details>')
    descriptions = {
        'head': ('头相机：EEF/TCP 与 absolute/relative 还原对照', '左：源 EEF/腕部（虚线）与 TCP（实线）。右：同一端点的绝对轨迹，与 head-relative / local-relative 还原后的轨迹应重合。'),
        'projection': ('投影方式：固定头相机、随动头相机、相机模型', '左固定首次头相机与背景，轨迹窗口随时间推进；右为连续真实视频及当帧标定。灰虚线仅表示忽略畸变的诊断。'),
        'relative': ('相对坐标轨迹：head-reference 与 TCP/EEF-local', '两侧是同一物理端点的米制 3D 位移，均相对各窗口当前 t。不是图像像素投影；缺 TCP 时 EEF-local 仅为诊断。'),
        'fk': ('joint → URDF → FK：头相机投影与 3D 轨迹', '左为真实头相机视频叠加 GT 和 FK；右按时间绘制源 EEF 与 FK 的 3D 路径，并显示输入关节角。具体验证坐标系和限制见本卡说明。'),
        'joint': ('原生关节轨迹：缺匹配URDF时的直接数据展示', '左右分别绘制真实关节角随时间变化，单位rad。没有用其他机器人模型代替，也不把关节图当作TCP/FK轨迹。'),
        'origin': ('端点排查：法兰、CAD/config 候选原点（不是已确认 TCP）', '真实视频中比较法兰与具名 URDF 点/配置偏移。候选原点不代表已确认完整 TCP pose；AgiBot 的旧 +100mm 点已撤下 TCP 标注。'),
        'wrist': ('腕部相机：左右真实视频＋TCP未来轨迹（随动视角）', '左/右画面分别是左/右腕相机。每帧使用各自相机t的内外参和真实背景，投影world TCP[t:t+32]。青色左TCP，橙色右TCP；方框为当前点。不是TCP-local XYZ图。缺坐标链时仅播放原视频并明确mask。'),
        'wrist_fixed': ('腕部投影对照：固定首帧相机与背景', '左右腕相机及背景都固定在本段第一帧，仅滚动显示未来世界TCP轨迹。切回随动视图可在同一时刻比较，固定背景不是视频卡住。'),
        'tcp_multiview': ('Agi多视角校验：160 / 190 / 200 / 245 mm', '四列各使用同一物理候选点，三行依次为头部、左腕、右腕；只画当前t点。200mm为本版暂选，245mm保留旧版对照。左右腕最佳长度不同，未使用按相机修改的TCP或像素偏移。'),
        'tcp_length': ('Agi微调对比：190 / 200 / 210 mm（左右夹爪放大）', '三列使用完全相同的当前帧裁剪：上排左夹爪、下排右夹爪，只画当前t的点。中列200mm为本版候选；另有头部/左腕/右腕三视角同步对比。单一长度仍无法消除全部视角误差。'),
    }
    entries = [dict(id=name, title=name, root=output, family=name) for name in NAMES]
    manifest_path = output/'agibot_examples/manifest.json'
    extras = json.loads(manifest_path.read_text())['samples'] if manifest_path.exists() else []
    # Show the new examples before the old whiteboard clip, retain its URL.
    entries[1:1] = [dict(id=s['id'], title=s['title'], root=output/s['root'],
                         family='agibotworld', selection=s['reason']) for s in extras]
    expanded_path = output/'expanded_sources/manifest.json'
    expanded = json.loads(expanded_path.read_text())['samples'] if expanded_path.exists() else []
    entries.extend(dict(id=s['id'], title=s['title'], root=output/s['root'],
                        family=s['family'], selection=s['reason']) for s in expanded)
    cards = []
    for entry in entries:
        name, family, video_root = entry['id'], entry['family'], entry['root']
        metrics = json.loads((video_root/'trajectory_videos'/family/'metrics.json').read_text())
        source, fk = metrics['source'], metrics['fk']
        options, players = [], []
        default_mode = ('tcp_multiview' if 'tcp_multiview' in metrics['videos'] else 'origin') if family == 'agibotworld' else 'head'
        for index, (mode, relative_path) in enumerate(metrics['videos'].items()):
            title, detail = descriptions[mode]
            if mode == 'origin' and source['tcp_status'] == 'visual_tcp_preview':
                title = 'TCP修正预览：旧100/128mm点 vs 新200mm虚拟中心'
                detail = '左：旧URDF具名点；右：局部+X 200mm视觉TCP预览。青/橙分别为左右臂，圆圈十字为当前t的点，线为未来轨迹。轴方向沿用canonical腕部；非官方标定。'
            selected = mode == default_mode
            options.append(f'<option value="{index}" data-kind="{mode}" {"selected" if selected else ""}>{html.escape(title)}</option>')
            players.append(f'<div class="video-mode" data-mode="{index}" {"" if selected else "hidden"}><p>{html.escape(detail)}</p><video controls playsinline loop preload="none" data-mode="{index}" src="{video_data(video_root/relative_path)}"></video></div>')
        limitation = html.escape(str(fk.get('limitations', '')))
        if 'fk' not in metrics['videos']:
            limitation = '本来源没有可用机器人 joint + URDF；不生成伪 FK。' + limitation
        numeric = html.escape(json.dumps(metrics['roundtrip_max_errors'], ensure_ascii=False))
        fk_rows = ''.join(f'<tr><td>{key}</td><td>{value["valid_count_left_right"]}</td><td>{value["max_position_mm_left_right"]}</td></tr>'
                          for key, value in metrics['fk_clip_errors'].items()) if 'fk' in metrics['videos'] else ''
        fk_table = '<table><thead><tr><th>本段 FK 对照</th><th>有效点数 [左,右]</th><th>最大位置误差 mm [左,右]</th></tr></thead><tbody>'+fk_rows+'</tbody></table>' if fk_rows else ''
        acceptance = '<p><b>已视觉确认：腕部局部 +X 120mm，旋转不变。</b></p>' if source.get('acceptance_status') == 'user_visual_accepted' else ''
        if extras and name == extras[0]['id']:
            cards.append('<h2 id="agibot-examples">AgiWorld 新版：200mm虚拟TCP视觉预览</h2><p>每个样例10秒，默认显示160/190/200/245mm在头部、左腕、右腕的同步对比。可切换头相机投影、固定/随动视角、相对坐标和joint→URDF→FK；缺躯干时FK只作跨臂比较。原擦白板样例保留在这三组后面。</p>')
        if expanded and name == expanded[0]['id']:
            cards.append(expanded_coverage_html(output, expanded))
        sample_detail = ''
        if 'selection' in entry:
            audit = source.get('source_episode_audit', {})
            sample_detail = f'<p>样例：{html.escape(name)} · episode {source["episode_index"]} · {html.escape(entry["selection"])}</p><p>源相机审计：{html.escape(str(audit))}（源记录，并非本次独立标定）。</p>'
        if 'wrist' in metrics:
            sample_detail += '<p><b>腕部视角：</b>'+html.escape('；'.join(side+': '+view['reason'] for side, view in metrics['wrist']['views'].items()))+'</p>'
        cards.append(f'''<section class="trajectory-video-card" id="video-{name}"><h3>{html.escape(entry['title'])}</h3>
{sample_detail}
{acceptance}
<p>{metrics['frames']} 连续帧 · {metrics['fps']:g} FPS · {metrics['duration_s']:.2f} 秒 · 源帧 {source['frame_indices'][0]}–{source['frame_indices'][-1]} · TCP：<b>{html.escape(source['tcp_status'])}</b></p>
<p>端点定义：{html.escape(str(source.get('endpoint_labels', source.get('endpoint', ''))))}</p>
<p class="video-note">FK 3D 坐标系：{html.escape(str(fk.get('local_frame', 'unavailable')))}。{limitation}</p>
<label>视频内容 <select class="video-selector" onchange="selectTrajectoryVideo(this)">{''.join(options)}</select></label>
<label>播放速度 <select onchange="setTrajectorySpeed(this)"><option value="1">1× 原始帧率</option><option value="0.5">0.5×</option><option value="0.25">0.25×</option></select></label>
{''.join(players)}
<p>同端点 relative → absolute 还原最大误差（position：米，rotation_element：旋转矩阵元素）：<code>{numeric}</code></p>{fk_table}
<details><summary>还原误差、FK 证据、相机模型、逐帧 mask 和源视频时间</summary><pre>{html.escape(json.dumps(metrics, ensure_ascii=False, indent=2))}</pre></details></section>''')
    semantics = '''<table><thead><tr><th>路径</th><th>位置/旋转定义</th><th>物理端点</th></tr></thead><tbody>
<tr><td>OpenWAM 原始/absolute EEF 路径</td><td>源参考系下的绝对 p、R；绝对不等于所有数据都是 world</td><td>数据源声明的 EEF，可能是法兰或手腕</td></tr>
<tr><td>MoraVLA 当前 dual_eef_local_96d</td><td>head-reference delta + 当前源 EEF-local delta；both 模式</td><td>默认 observation.states.eef / actions.eef，不自动读取 eef_tcp</td></tr>
<tr><td>当前 OpenWAM 112D 适配</td><td>head-reference delta + 当前 TCP-local delta，并有 joint delta</td><td>两套必须是同一显式 TCP；缺 TCP 两套 pose mask；缺头相机仅 head mask</td></tr>
</tbody></table>
<p>Mora 的 wrist_eef 在当前 local recipe 表示当前 EEF 局部坐标系，并非腕相机。random 头/腕参考系选择器属于 Mora 的可选配置；当前 both recipe 不执行随机切换。</p>
<pre>local: Δp = R_tᵀ(p_k − p_t), ΔR = R_tᵀ R_k
restore local: p_k = p_t + R_t Δp, R_k = R_t ΔR
head: Δp = R_Ct(p_k − p_t), ΔR = R_Ct R_k R_tᵀ R_Ctᵀ
restore head: p_k = p_t + R_Ctᵀ Δp, R_k = R_Ctᵀ ΔR R_Ct R_t
image at t: pixel = π_model(K_t, D_t, T_Ct_world · p_world(k))</pre>
<p>本视频把目标时刻统一为 state[t+1..t+32] 做表示对照；没有调用模型，也没有把 Mora 的 actions.eef 偷换成已验证的同一时序。正确还原的 absolute/relative 同端点轨迹应重合。手腕与 TCP 的差异另画，不能归因于 absolute/relative。</p>'''
    audit = output/'trajectory_videos/semantics_audit.txt'
    evidence = f'<details><summary>两个仓库的代码证据与行号</summary><pre>{html.escape(audit.read_text())}</pre></details>' if audit.exists() else ''
    section = '''<!-- VIDEOS BEGIN --><style>.trajectory-video-card{padding:20px;border:1px solid #acc5d5;border-radius:12px;margin:24px 0;background:#f4f8fb}.trajectory-video-card video{width:100%;max-height:80vh;background:#171c23}.trajectory-video-card select{padding:9px;max-width:100%;margin:8px}.trajectory-video-card pre{max-height:600px;overflow:auto;white-space:pre-wrap}.video-note{color:#8b531c}.video-mode[hidden]{display:none}.video-nav{display:flex;gap:16px;flex-wrap:wrap;margin:16px 0}</style>
<section id="continuous-videos"><h2 id="previous-fk">新版可视化汇总：头相机 TCP、表示还原、URDF/FK 与投影方式</h2>
<p><b>RoboDojo / RobotWin：</b>已视觉确认的腕部局部 +X 120mm 版本；<b>AgiWorld：</b>最新200mm虚拟TCP视觉预览，默认显示头部和左右腕部的同步长度对比；CAD参考＋人工目视调整，并非官方标定。InternData及另外三组来源的最新版视频也在本页。</p>
<p>这里是逐帧解码的连续真实视频，所有 pose、joint、相机和图像按同一 frame_index 对齐；每帧滚动显示未来最多 32 步，末端不足则截断。方框是当前 t，星号是最后一个真实目标。图像外或无效点断开；不检查遮挡。</p>
<p>每组下拉切换视频内容，可暂停、拖动进度、慢放或全屏。视频和证据均内嵌，离线 HTML 可直接播放。GT/转换检查与模型推理结果严格区分。Agi新增190/200/210mm放大对比，腕部相机视角见下表和每组下拉菜单。</p>'''+wrist_coverage_html(output)+'''<details><summary>坐标定义、端点排查与完整来源证据</summary>'''+endpoint_audit_html(output)+semantics+evidence+'</details><nav class="video-nav">'+''.join(f'<a href="#video-{entry["id"]}">{html.escape(entry["title"])}</a>' for entry in entries)+'</nav>'+''.join(cards)+'''</section>
<script>
function selectTrajectoryVideo(select){const card=select.closest('.trajectory-video-card');let current=0;card.querySelectorAll('.video-mode').forEach(block=>{const v=block.querySelector('video');if(!block.hidden){current=v.currentTime;v.pause();}});card.querySelectorAll('.video-mode').forEach(block=>{block.hidden=block.dataset.mode!==select.value;if(!block.hidden){const v=block.querySelector('video');const seek=()=>{if(Number.isFinite(v.duration))v.currentTime=Math.min(current,Math.max(0,v.duration-.01));v.playbackRate=v.defaultPlaybackRate;};if(v.readyState>=1)seek();else{v.addEventListener('loadedmetadata',seek,{once:true});v.load();}}});}
function setTrajectorySpeed(select){select.closest('.trajectory-video-card').querySelectorAll('video').forEach(v=>{v.defaultPlaybackRate=Number(select.value);v.playbackRate=Number(select.value);});}
function openWristView(id){const card=document.getElementById(id);if(!card)return;const select=card.querySelector('select.video-selector');if(!select)return;const option=select.querySelector('option[data-kind="wrist"]');if(option){select.value=option.value;selectTrajectoryVideo(select);}}
document.querySelectorAll('.trajectory-video-card video').forEach(video=>video.addEventListener('play',()=>{document.querySelectorAll('.trajectory-video-card video').forEach(other=>{if(other!==video)other.pause();});}));
function revealReportAnchor(){const id=decodeURIComponent(location.hash.slice(1));if(!id)return;const target=document.getElementById(id);if(!target)return;let node=target.parentElement;while(node){if(node.tagName==='DETAILS')node.open=true;node=node.parentElement;}target.scrollIntoView();}
window.addEventListener('hashchange',revealReportAnchor);window.addEventListener('load',revealReportAnchor);
</script><!-- VIDEOS END -->'''
    heading = re.search(r'<h1>.*?</h1>', current, re.S)
    if heading is None:
        raise ValueError('coverage HTML must have an h1')
    merged = current[:heading.end()]+section+current[heading.end():]
    page.write_text(merged)
    (ROOT/'outputs/action_alignment_audit/tcp_coverage_v2_v3.html').write_text(merged)
    with ZipFile(page.with_suffix('.zip'), 'w', ZIP_DEFLATED) as archive:
        archive.writestr(page.name, merged)
    build_user_review(output)
    print('Updated continuous-video HTML:', len(merged.encode()), 'bytes', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', nargs='+', choices=NAMES, default=list(NAMES))
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--no-report', action='store_true')
    args = parser.parse_args()
    if not args.report_only:
        for sample in args.samples:
            render_sample(sample)
    if not args.no_report:
        build_report()
