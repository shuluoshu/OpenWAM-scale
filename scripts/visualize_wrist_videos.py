"""Add physical wrist-camera views and Agi TCP-length comparisons to HTML."""
from __future__ import annotations

import argparse
import json

import cv2
import numpy as np

from scripts.data.tcp_endpoint_policy import apply_endpoint_policy
from scripts.data.video_trajectory_geometry import future_indices
from scripts.data.wrist_projection_samples import extract_wrist_sample
from scripts.visualize_trajectory_videos import (
    GT_COLORS,
    NAMES,
    OUTPUT,
    VideoWriter,
    build_report,
    draw_track,
    image_panel,
    panel,
    pixels,
    text,
)


def length_comparison(image, arrays, source, i):
    """Identical enlarged head crops for 190/200/210, current points only."""
    h, w = image.shape[:2]
    reference = arrays['eef_world'][i:i+1].copy()
    reference[..., :3, 3] += reference[..., :3, 0] * .200
    valid = arrays['eef_valid'][i:i+1]
    centers, good = pixels(reference, valid, arrays, source, i, (0, 0, w, h))
    canvas = np.full((600, 1280, 3), (28, 23, 19), np.uint8)
    for col, length in enumerate([.190, .200, .210]):
        x0 = col*426
        text(canvas, f'Wrist local +X {length*1000:g} mm'+(' SELECTED' if col == 1 else ''), (x0+8, 25), size=.56)
        text(canvas, f'{source.get("clip_label", "agibot")} state {arrays["frame_indices"][i]}', (x0+8, 47), size=.4)
        candidate = arrays['eef_world'][i:i+1].copy()
        candidate[..., :3, 3] += candidate[..., :3, 0]*length
        uv, candidate_valid = pixels(candidate, valid, arrays, source, i, (0, 0, w, h))
        for side in range(2):
            y0 = 65+side*240
            if not good[0, side] or not candidate_valid[0, side]:
                text(canvas, f'{"LEFT" if side == 0 else "RIGHT"}: outside head image', (x0+8, y0+40), size=.43)
                continue
            cx, cy = centers[0, side]
            cw, ch = min(w, 160), min(h, 90)
            left = int(np.clip(round(cx-cw/2), 0, w-cw))
            top = int(np.clip(round(cy-ch/2), 0, h-ch))
            crop = cv2.resize(image[top:top+ch, left:left+cw], (410, 220))
            point = np.rint((uv[0, side]-[left, top])*[410/cw, 220/ch]).astype(int)
            cv2.circle(crop, tuple(point), 9, GT_COLORS[side], 1, cv2.LINE_AA)
            cv2.drawMarker(crop, tuple(point), GT_COLORS[side], cv2.MARKER_CROSS, 15, 1)
            canvas[y0:y0+220, x0+8:x0+418] = crop
        text(canvas, 'Top: left / bottom: right; current t only', (x0+8, 550), size=.4)
        text(canvas, 'Same crop at t; visual preview, not calibration', (x0+8, 574), size=.37)
    return canvas


def multiview_comparison(head_image, arrays, source, cameras, i):
    """The same physical candidate in head, left wrist and right wrist images."""
    canvas = np.full((600, 1280, 3), (28, 23, 19), np.uint8)
    views = [('HEAD', head_image, arrays, source)]
    for side in ['left', 'right']:
        folder, meta, calibration = cameras[side]
        views.append((side.upper()+' WRIST', cv2.imread(str(folder/meta['image_filenames'][i])), calibration, meta))
    for row, (label, frame, calibration, meta) in enumerate(views):
        for col, length in enumerate([.160, .190, .200, .245]):
            x, y = col*320, row*200
            scale = min(320/frame.shape[1], 170/frame.shape[0])
            w, h = round(frame.shape[1]*scale), round(frame.shape[0]*scale)
            rectangle = (x+(320-w)//2, y+27, w, h)
            left, top, _, _ = rectangle
            canvas[top:top+h, left:left+w] = cv2.resize(frame, (w, h))
            poses = arrays['eef_world'][i:i+1].copy()
            poses[..., :3, 3] += poses[..., :3, 0]*length
            uv, good = pixels(poses, arrays['eef_valid'][i:i+1], calibration, meta, i, rectangle)
            for side in range(2):
                if good[0, side]:
                    cv2.drawMarker(canvas, tuple(np.rint(uv[0, side]).astype(int)), GT_COLORS[side], cv2.MARKER_CROSS, 11, 1)
            text(canvas, f'{label} {length*1000:g}mm'+(' PREVIEW' if col == 2 else ''), (x+5, y+18), size=.4)
    return canvas


def render_wrist(name, root, *, reuse_samples=False):
    folder = root/'video_samples'/name
    source = json.loads((folder/'metadata.json').read_text())
    with np.load(folder/'arrays.npz') as f:
        arrays = dict(f)
    apply_endpoint_policy(name, arrays, source)
    summary_path = root/'wrist_samples'/name/'summary.json'
    summary = (json.loads(summary_path.read_text()) if reuse_samples and summary_path.exists()
               else extract_wrist_sample(name, root))
    cameras = {}
    for side, meta in summary['views'].items():
        if not meta['video_available']:
            continue
        side_folder = summary_path.parent/side
        with np.load(side_folder/'arrays.npz') as f:
            calibration = dict(f)
        np.testing.assert_array_equal(calibration['frame_indices'], arrays['frame_indices'])
        cameras[side] = (side_folder, meta, calibration)
    modes = ['wrist'] if cameras else []
    if any(m['projection_available'] for m in summary['views'].values()):
        modes.append('wrist_fixed')
    if name == 'agibotworld':
        modes.extend(['tcp_length', 'tcp_multiview'])
    out = root/'trajectory_videos'/name
    writers = {mode: VideoWriter(out/f'{mode}.mp4', source['fps']) for mode in modes}
    counts = {mode: [] for mode in modes if mode.startswith('wrist')}
    count = len(arrays['frame_indices'])
    try:
        for i in range(count):
            ids = future_indices(i, count)
            timing = f'{source.get("clip_label", name)} | state {arrays["frame_indices"][i]} | t..t+{len(ids)-1} | {i/source["fps"]:.2f}s'
            for mode in counts:
                camera_index = i if mode == 'wrist' else 0
                panels, frame_counts = [], {}
                for side in ['left', 'right']:
                    if side not in cameras:
                        canvas = panel(f'{side.upper()} WRIST: unavailable', timing)
                        text(canvas, 'Source has no video for this wrist camera.', (20, 290))
                        frame_counts[side] = [0, 0]
                    else:
                        side_folder, meta, calibration = cameras[side]
                        image = cv2.imread(str(side_folder/meta['image_filenames'][camera_index]))
                        if image is None:
                            raise ValueError(f'Missing wrist image {side_folder} frame {camera_index}')
                        unverified_mount = meta.get('camera_mount_status') == 'wrist_mounting_unverified'
                        title = (side.upper()+' SOURCE CAMERA (MOUNT UNVERIFIED)' if unverified_mount else
                                 ('MOVING ' if mode == 'wrist' else 'FIXED FIRST ')+side.upper()+' WRIST CAMERA')
                        canvas, rectangle = image_panel(image, title, timing)
                        c = draw_track(canvas, rectangle, arrays['tcp_world'][ids], arrays['tcp_valid'][ids],
                                       calibration, meta, camera_index)
                        frame_counts[side] = c
                        text(canvas, f'TCP L cyan / R orange | in image L/R: {c}', (14, 539), size=.43)
                        if meta['projection_available']:
                            text(canvas, f'Own calibration: {meta["model"]}; square=t, star=last target', (14, 560), size=.41)
                            note = ('Derived camera chain; see endpoint/calibration limitations.'
                                    if meta.get('calibration_status') in ('cad_derived_preview', 'derived_candidate') else
                                    'Camera and background update together at t.' if mode == 'wrist' else
                                    'First camera AND first image held fixed; future world path rolls.')
                            text(canvas, note, (14, 582), size=.38)
                        else:
                            text(canvas, 'TCP PROJECTION MASKED: missing/invalid camera chain', (14, 560), (80, 180, 255), size=.37)
                            text(canvas, str(meta.get('camera_missing_reason', 'Source wrist video; missing chain explained in report.'))[:88], (14, 582), size=.36)
                        if not arrays['tcp_valid'][i].any():
                            text(canvas, 'TCP UNAVAILABLE: no verified endpoint transform', (14, 80), (80, 180, 255), size=.4)
                    panels.append(canvas)
                writers[mode].write(np.hstack(panels))
                counts[mode].append(frame_counts)
            if 'tcp_length' in writers:
                image = cv2.imread(str(folder/source['image_filenames'][i]))
                writers['tcp_length'].write(length_comparison(image, arrays, source, i))
                writers['tcp_multiview'].write(multiview_comparison(image, arrays, source, cameras, i))
            if modes and i % 90 == 0:
                print(f'{source.get("clip_label", name)}: wrist/compare {i+1}/{count}', flush=True)
    finally:
        for writer in writers.values():
            writer.close()
    metrics_path = out/'metrics.json'
    metrics = json.loads(metrics_path.read_text())
    # Replace only these supplementary modes, leaving existing head/FK modes.
    for mode in ['wrist', 'wrist_fixed', 'tcp_length', 'tcp_multiview']:
        metrics['videos'].pop(mode, None)
    metrics['videos'].update({mode: str((out/f'{mode}.mp4').relative_to(root)) for mode in modes})
    metrics['wrist'] = summary
    metrics['wrist_in_image_counts'] = counts
    metrics['wrist_semantics'] = 'Project world TCP[t:t+32] with this wrist camera T_camera(t)_world, K(t), lens model and image(t). Fixed view uses camera(0) and image(0). Not TCP-local XYZ projection; never uses head K/extrinsic.'
    if name == 'agibotworld':
        metrics['tcp_length_comparison_m'] = [.190, .200, .210]
        metrics['tcp_multiview_comparison_m'] = [.160, .190, .200, .245]
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False)+'\n')
    return summary


def main(reuse_samples=False):
    entries = [(name, name, OUTPUT) for name in NAMES]
    for entry in json.loads((OUTPUT/'agibot_examples/manifest.json').read_text())['samples']:
        entries.append((entry['id'], 'agibotworld', OUTPUT/entry['root']))
    coverage = []
    for sample, name, root in entries:
        summary = render_wrist(name, root, reuse_samples=reuse_samples)
        coverage.append(dict(sample=sample, family=name, root=str(root.relative_to(OUTPUT)),
            views={side: {k: m[k] for k in ['status', 'video_available', 'projection_available', 'reason']}
                   for side, m in summary['views'].items()}))
    coverage_path = OUTPUT/'wrist_samples/coverage.json'
    if coverage_path.exists():
        rendered = {entry['sample'] for entry in coverage}
        coverage.extend(entry for entry in json.loads(coverage_path.read_text()) if entry['sample'] not in rendered)
    coverage_path.write_text(json.dumps(coverage, indent=2, ensure_ascii=False)+'\n')
    build_report()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reuse-samples', action='store_true')
    args = parser.parse_args()
    main(reuse_samples=args.reuse_samples)
