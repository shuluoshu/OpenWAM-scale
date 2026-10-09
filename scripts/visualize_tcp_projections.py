#!/usr/bin/env python3
"""Add calibrated trajectory projections and prior FK plots to the coverage HTML.

All render assets are embedded. Source datasets and training profiles are read
only. Execute as ``python -m scripts.visualize_tcp_projections`` from repo root.
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import re
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from PIL import Image

from openwam.dataloader.utils.video_io import decode_video_frames
from scripts.data.tcp_endpoint_policy import apply_endpoint_policy
from scripts.data.trajectory_projection import project_points, transform_points
from scripts.data.urdf_fk_audit import eef20_to_matrix, load_episode, pose7_to_matrix
from scripts.visualize_fk_consistency import NAMES, PROFILES

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'outputs/fk_consistency'
COLORS = ('#08becb', '#ff9c30')


def embedded_image(image, *, photo=False):
    buffer = io.BytesIO()
    if photo:
        image.convert('RGB').save(buffer, format='JPEG', quality=90, subsampling=0)
    else:
        image.save(buffer, format='PNG')
    mime = 'jpeg' if photo else 'png'
    return f'data:image/{mime};base64,' + base64.b64encode(buffer.getvalue()).decode()


def extract_existing(output_root):
    """Extract 33 contiguous state rows and correctly timed source video frames."""
    for name, profile in PROFILES.items():
        cfg = OmegaConf.load(ROOT / profile)
        info = json.loads((Path(cfg.dataset_dir) / 'meta/info.json').read_text())
        features = info['features']
        intrinsic = 'observation.states.camera.intrinsic'
        if intrinsic not in features:
            intrinsic = 'observation.states.cameras.cam_high.intrinsic'
        extrinsic = 'observation.states.camera.world_to_camera'
        keys = ['observation.states.eef', intrinsic, extrinsic]
        if name == 'interndata':
            keys += ['states.left_tcp_to_robot_pose', 'states.right_tcp_to_robot_pose', 'states.robot_to_env_pose']
        _, _, episode, frame = load_episode(ROOT / profile, columns=keys)
        anchor = min(30, len(frame) - 33)
        if anchor < 0:
            raise ValueError(f'{name}: episode shorter than the 33-row visualization window')
        window = frame.iloc[anchor:anchor + 33]
        frame_indices = window.frame_index.to_numpy(dtype=int)
        if not np.array_equal(frame_indices, np.arange(anchor, anchor + 33)):
            raise ValueError(f'{name}: window is not contiguous')
        eef = eef20_to_matrix(np.stack(window['observation.states.eef']))
        tcp = np.broadcast_to(np.eye(4), eef.shape).copy()
        eef_valid = np.isfinite(eef).all(axis=(-1, -2))
        tcp_valid = np.zeros((33, 2), bool)
        status = 'unavailable'
        endpoint = 'Only source wrist/flange EEF; no verified TCP in this profile.'
        if name == 'interndata':
            reference = pose7_to_matrix(np.stack(window['states.robot_to_env_pose']))
            for side_id, side in enumerate(('left', 'right')):
                tcp[:, side_id] = reference @ pose7_to_matrix(np.stack(window[f'states.{side}_tcp_to_robot_pose']))
            tcp_valid[:] = np.isfinite(tcp).all(axis=(-1, -2))
            status, endpoint = 'declared_tcp', 'Native xyz+wxyz TCP, robot_to_env applied per row; EEF is a different endpoint.'
        camera = np.stack(window[extrinsic]).reshape(-1, 4, 4)
        k = np.stack(window[intrinsic]).reshape(-1, 3, 3)
        cv = np.isfinite(camera).all(axis=(-1, -2)) & np.isfinite(k).all(axis=(-1, -2))
        native_h, native_w = features[cfg.head_camera]['shape'][:2]
        width = min(800, native_w)
        height = round(native_h * width / native_w)
        camera_key = cfg.head_camera
        video_root = Path(cfg.get('video_root') or cfg.dataset_dir)
        video = video_root / info['video_path'].format(video_key=camera_key,
            chunk_index=int(episode[f'videos/{camera_key}/chunk_index']),
            file_index=int(episode[f'videos/{camera_key}/file_index']))
        offset = round(float(episode.get(f'videos/{camera_key}/from_timestamp', 0)) * float(info['fps']))
        folder = output_root / name
        folder.mkdir(parents=True, exist_ok=True)
        images = []
        for index in (0, 16, 32):
            decoded = decode_video_frames(str(video), [offset + int(frame_indices[index])], height=height, width=width)
            if len(decoded) != 1:
                raise ValueError(f'{name}: failed to decode snapshot {index}')
            filename = f'image{index}.jpg'
            decoded[0].save(folder / filename, quality=92)
            images.append(filename)
        arrays = dict(eef_world=eef, tcp_world=tcp,
            eef_valid=eef_valid, tcp_valid=tcp_valid, world_to_camera=camera,
            K=k, distortion=np.empty((33, 0)), camera_valid=cv, frame_indices=frame_indices)
        metadata = dict(name=NAMES[name], dataset_dir=str(cfg.dataset_dir), episode=0,
            anchor=anchor, horizon=32, frame_indices=frame_indices.tolist(), model='pinhole',
            distortion_order=[], image_width=native_w, image_height=native_h,
            display_width=width, display_height=height, image_filenames=images,
            snapshot_indices=[0, 16, 32], tcp_status=status, endpoint=endpoint,
            video_path=str(video), video_frame_offset=offset,
            intrinsic_key=intrinsic, extrinsic_key=extrinsic,
            camera_model_evidence='Released head video and row-major K; pinhole projection with no extra distortion applied. No claim of independently verified reprojection calibration.',
            limitations='Native endpoint semantics and calibration are source-specific; see per-source endpoint audit. Image plausibility alone is not independent camera/tool calibration.')
        apply_endpoint_policy(name, arrays, metadata)
        np.savez_compressed(folder / 'arrays.npz', **arrays)
        (folder / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
        print('Extracted', name, flush=True)


def projected(arrays, poses, valid, view_index, model):
    """Project one whole trajectory into ONE camera, never mixed camera axes."""
    if not bool(arrays['camera_valid'][view_index]):
        return np.full((*poses.shape[:-2], 2), np.nan), np.zeros(poses.shape[:-2], bool)
    camera_points = transform_points(arrays['world_to_camera'][view_index], poses[..., :3, 3])
    distortion = None if model == 'pinhole' else arrays['distortion'][view_index]
    pixels, visible = project_points(camera_points, arrays['K'][view_index], model=model, distortion=distortion)
    visible &= valid & bool(arrays['camera_valid'][view_index])
    pixels[~visible] = np.nan
    return pixels, visible


def overlay(folder, metadata, arrays, index, model, path):
    snapshot = metadata['snapshot_indices'].index(index)
    image = Image.open(folder / metadata['image_filenames'][snapshot]).convert('RGB')
    native_w, native_h = metadata['image_width'], metadata['image_height']
    sx, sy = image.width / native_w, image.height / native_h
    fig, ax = plt.subplots(figsize=(10, 10 * image.height / image.width))
    ax.imshow(image)
    metrics = {}
    for stream, style, alpha in (('eef', '--', .80), ('tcp', '-', 1.0)):
        uv, valid = projected(arrays, arrays[f'{stream}_world'], arrays[f'{stream}_valid'], index, model)
        in_frame = valid & (uv[..., 0] >= 0) & (uv[..., 0] < native_w) & (uv[..., 1] >= 0) & (uv[..., 1] < native_h)
        # A missing/out-of-frame point creates a gap; never bridge invalid spans.
        scaled = uv * [sx, sy]
        scaled[~in_frame] = np.nan
        metrics[stream] = {'valid_domain_by_arm': valid.sum(0).tolist(), 'in_frame_by_arm': in_frame.sum(0).tolist()}
        for side, color in enumerate(COLORS):
            label = ('L' if side == 0 else 'R') + ' ' + ('EEF/wrist' if stream == 'eef' else 'TCP')
            if stream == 'tcp' and metadata['tcp_status'].startswith('candidate'):
                label += ' candidate'
            if stream == 'tcp' and metadata['tcp_status'] == 'visual_tcp_preview':
                label += ' visual preview'
            if not in_frame[:, side].any():
                continue
            ax.plot(scaled[:, side, 0], scaled[:, side, 1], color=color, linestyle=style,
                    linewidth=2.2, alpha=alpha, label=label)
            ax.scatter(*scaled[0, side], color=color, marker='s', s=45, edgecolors='black', linewidths=.5)
            ax.scatter(*scaled[-1, side], color=color, marker='*', s=90, edgecolors='black', linewidths=.5)
    ax.set_xlim(0, image.width - 1)
    ax.set_ylim(image.height - 1, 0)
    ax.set_axis_off()
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc='upper left', fontsize=8, facecolor='white', framealpha=.85)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(path, dpi=110, bbox_inches=None)
    plt.close(fig)
    return embedded_image(Image.open(path), photo=True), metrics


def frame_comparison(folder, metadata, arrays):
    """Compare head-axis deltas and local-frame deltas in meters, not pixels."""
    use_tcp = bool(arrays['tcp_valid'].any())
    stream = 'tcp' if use_tcp else 'eef'
    poses = arrays[f'{stream}_world']
    valid = arrays[f'{stream}_valid'] & arrays[f'{stream}_valid'][0]
    rc = arrays['world_to_camera'][0, :3, :3]
    delta = poses[..., :3, 3] - poses[0, :, :3, 3]
    head = np.einsum('ij,nsj->nsi', rc, delta)
    local = np.einsum('sji,nsj->nsi', poses[0, :, :3, :3], delta)
    head[~valid] = np.nan
    if not arrays['camera_valid'][0]:
        head[:] = np.nan
    local[~valid] = np.nan
    fig = plt.figure(figsize=(12, 4.6))
    for plot_id, (values, title) in enumerate(((head, 'Current head axes: relative XYZ (m)'),
                                              (local, f'Current {stream.upper()} local axes: relative XYZ (m)'))):
        ax = fig.add_subplot(1, 2, plot_id + 1, projection='3d')
        for side, color in enumerate(COLORS):
            ax.plot(*values[:, side].T, color=color, label='Left' if side == 0 else 'Right')
            ax.scatter(0, 0, 0, color=color, marker='s', s=25)
        finite = values[np.isfinite(values).all(-1)]
        if len(finite):
            center = (finite.min(0) + finite.max(0)) / 2
            radius = max(float(np.ptp(finite, axis=0).max()) / 2, .003)
            for setter, middle in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
                setter(middle - radius, middle + radius)
        ax.set(xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)', title=title)
        ax.set_box_aspect((1, 1, 1))
        ax.legend()
    if not use_tcp or metadata['tcp_status'].startswith('candidate') or metadata['tcp_status'] == 'visual_tcp_preview':
        fig.suptitle('DIAGNOSTIC endpoint; not an enabled TCP training label', color='#9b5521')
    path = folder / 'relative_frames.png'
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return embedded_image(Image.open(path))


def render_sample(folder):
    metadata = json.loads((folder / 'metadata.json').read_text())
    with np.load(folder / 'arrays.npz') as data:
        arrays = {key: data[key] for key in data.files}
    apply_endpoint_policy(folder.name, arrays, metadata)
    n = len(arrays['frame_indices'])
    if n != 33 or not np.all(np.diff(arrays['frame_indices']) == 1):
        raise ValueError(f'{folder.name}: expected 33 contiguous state rows')
    model = metadata['model']
    pictures, metrics = [], {}
    for index in metadata['snapshot_indices']:
        image, result = overlay(folder, metadata, arrays, index, model, folder / f'projection_{index}.png')
        pictures.append(image)
        metrics[str(index)] = result
    # Compare projection models with identical image, camera, endpoint and points.
    uv, native_valid = projected(arrays, arrays['tcp_world'], arrays['tcp_valid'], 0, model)
    simple, simple_valid = projected(arrays, arrays['tcp_world'], arrays['tcp_valid'], 0, 'pinhole')
    common = native_valid & simple_valid
    error = np.linalg.norm(uv[common] - simple[common], axis=-1)
    model_delta = {'compared_points': len(error), 'median_native_pixels': float(np.median(error)) if len(error) else None,
                   'max_native_pixels': float(error.max()) if len(error) else None}
    if model != 'pinhole':
        naive, _ = overlay(folder, metadata, arrays, 0, 'pinhole', folder / 'pinhole_diagnostic.png')
        diagnostic = f'<details><summary>相机模型对照：{html.escape(model)} vs 忽略畸变的 pinhole（诊断）</summary><div class="projection-grid"><figure><figcaption>按源模型 {html.escape(model)}</figcaption><img src="{pictures[0]}" alt="native calibrated projection"></figure><figure><figcaption>仅 K·[X/Z,Y/Z,1]，忽略源畸变；不是另一套 GT</figcaption><img src="{naive}" alt="pinhole-only diagnostic"></figure></div><p>相同 TCP 点与当前相机，像素差（原始视频分辨率）：{html.escape(str(model_delta))}</p></details>'
    else:
        diagnostic = '<p>本样本按已发布 K 使用 pinhole；没有额外畸变参数可比较，不虚构第二套相机模型。</p>'
    sample_id = re.sub('[^a-z0-9_]', '_', folder.name)
    options = ''.join(f'<option value="{i}">相机/图像帧 {int(arrays["frame_indices"][index])}（t+{index}）</option>' for i, index in enumerate(metadata['snapshot_indices']))
    moving = ''.join(f'<img data-snapshot="{i}" style="display:{"block" if i == 0 else "none"}" src="{image}" alt="trajectory reprojected in matched camera snapshot {i}">' for i, image in enumerate(pictures))
    notes = html.escape(str(metadata.get('endpoint', metadata.get('endpoint_labels', ''))))
    limitations = html.escape(str(metadata.get('limitations', '')))
    title = html.escape(metadata.get('name', folder.name))
    count_rows = ''
    for index in metadata['snapshot_indices']:
        counts = metrics[str(index)]
        cells = []
        for stream in ('eef', 'tcp'):
            for side in range(2):
                total = counts[stream]['valid_domain_by_arm'][side]
                inside = counts[stream]['in_frame_by_arm'][side]
                cells.append('无有效投影' if total == 0 else f'{inside}/33' + ('（全部画面外）' if inside == 0 else ''))
        count_rows += f'<tr><td>t+{index}</td>' + ''.join(f'<td>{cell}</td>' for cell in cells) + '</tr>'
    count_table = '<p>源相机模型下，画面内点数（含 t，共 33 点；不检查遮挡）：</p><table><thead><tr><th>相机/图像</th><th>左 EEF</th><th>右 EEF</th><th>左 TCP</th><th>右 TCP</th></tr></thead><tbody>' + count_rows + '</tbody></table>'
    block = f'''<section class="projection-card" id="projection-{sample_id}"><h3>{title}</h3>
<p>episode {metadata['episode']} · anchor {int(arrays['frame_indices'][0])} · state targets {int(arrays['frame_indices'][1])}–{int(arrays['frame_indices'][-1])} · 模型 <b>{html.escape(model)}</b> · TCP 状态 <b>{html.escape(metadata['tcp_status'])}</b></p>
<p>{notes}</p><p class="projection-warning">{limitations}</p>
<p>青色：左侧；橙色：右侧。虚线：源 EEF/wrist；实线：TCP（候选会额外标注）。方块为 t，星形为 t+32。图像外/投影无效点断开，不跨缺失段连线；整段画面外的端点不列入图例。</p>{count_table}
<div class="projection-grid"><figure><figcaption>A · 固定当前 t 的相机与图像，投影整段未来轨迹</figcaption><img src="{pictures[0]}" alt="fixed anchor-camera trajectory"></figure>
<figure><figcaption>B · 切换相机时刻，同时切换对应图像，再将同一世界轨迹重投影</figcaption><select onchange="showProjectionSnapshot(this)">{options}</select><div>{moving}</div></figure></div>
{diagnostic}<details><summary>相对坐标对照：head-reference 与当前 TCP/EEF-local（单位 m，不是像素投影）</summary><img src="{frame_comparison(folder, metadata, arrays)}" alt="head reference versus local frame 3D trajectory"></details>
<details><summary>有效点数、来源、标定和限制</summary><pre>{html.escape(json.dumps({'metadata':metadata,'projection_counts':metrics,'model_difference':model_delta}, ensure_ascii=False, indent=2))}</pre></details></section>'''
    (folder / 'projection_metrics.json').write_text(json.dumps({'projection_counts': metrics, 'model_difference': model_delta}, indent=2))
    return block


def build_report(output=OUTPUT):
    page = output / 'tcp_coverage_v2_v3.html'
    current = page.read_text()
    current = re.sub(r'<!-- PROJECTIONS BEGIN -->.*?<!-- PROJECTIONS END -->', '', current, flags=re.S)
    samples = output / 'projection_samples'
    order = ('interndata', 'agibotworld', 'robodojo', 'robotwin', 'genrobot', 'lightwheel', 'mecka')
    missing = [name for name in order if not (samples / name / 'metadata.json').exists()]
    if missing:
        raise ValueError(f'missing sample artifacts: {missing}')
    cards = ''.join(render_sample(samples / name) for name in order)
    fk_source = (output / 'index.html').read_text()
    # Keep the original arrays/metrics downloadable in the standalone HTML too.
    def embed_download(match):
        relative = match.group(1)
        path = output / relative
        mime = 'application/json' if path.suffix == '.json' else 'application/octet-stream'
        data = base64.b64encode(path.read_bytes()).decode()
        return f'href="data:{mime};base64,{data}" download="{path.parent.name}_{path.name}"'
    fk_source = re.sub(r'href="([^"<>]+\.(?:npz|json))"\s+download', embed_download, fk_source)
    fk = html.escape(fk_source, quote=True)
    section = '''<!-- PROJECTIONS BEGIN --><style>.projection-card{background:#f4f8fb;border:1px solid #cbdce8;padding:20px;margin:22px 0;border-radius:12px}.projection-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.projection-grid figure{margin:0}.projection-card img{width:100%;height:auto}.projection-warning{color:#8a4e16}.projection-card select{padding:7px;margin:8px 0}figcaption{font-weight:600;margin:8px 0}iframe.fk-embedded{width:100%;height:1100px;border:1px solid #ccd5df}nav.projection-nav{display:flex;flex-wrap:wrap;gap:15px;margin:15px 0}nav.projection-nav a{color:#075f83}@media(max-width:800px){.projection-grid{grid-template-columns:1fr}}</style>
<section id="trajectory-projections"><h2>轨迹可视化：端点、参考相机、成像模型分开对比</h2>
<p>同一组 state[t+1:t+33]。A/B 使用同一个世界轨迹，区别只有参考相机及匹配图像；模型对照保持图像/相机/端点不变。图像重投影可用于发现标定问题，叠加看起来合理不等于标定已独立通过。</p>
<pre>图像投影：u = π_model(T_camera(t)_world · p_world(t+k))
head-relative XYZ = R_camera(t)_world · (p_world(t+k) − p_world(t))
TCP-local XYZ = R_world_TCP(t)^T · (p_world(t+k) − p_world(t))
把 relative XYZ 投影到图像前，必须加回当前端点原点；不能直接对 delta XYZ 做透视除法。
B 每次用所选时刻 s 的 T_camera(s)_world 和图像 I(s)，而不是混合各时刻像素再叠到 I(t)。</pre>
<nav class="projection-nav">'''+''.join(f'<a href="#projection-{name}">{name}</a>' for name in order)+'''<a href="#archived-fk">原四源 FK 轨迹图</a></nav>'''+cards+'''</section>
<section id="archived-fk"><h2>前面的四数据源 joint → URDF → FK 轨迹可视化</h2><p>原报告完整内嵌，包含 3D 轨迹、误差、交互旋转及原有验证范围。它是运动学一致性检查，与上面的像素投影检查分开。</p><iframe class="fk-embedded" title="原四数据源FK轨迹报告" srcdoc="'''+fk+'''"></iframe></section>
<script>function showProjectionSnapshot(select){const images=select.nextElementSibling.querySelectorAll('[data-snapshot]');images.forEach(img=>img.style.display=img.dataset.snapshot===select.value?'block':'none');}</script><!-- PROJECTIONS END -->'''
    # Put visualization before the coverage table; preserve the original audit.
    heading = re.search(r'<h1>.*?</h1>', current, re.S)
    if not heading:
        raise ValueError('coverage report has no h1')
    merged = current[:heading.end()] + section + current[heading.end():]
    page.write_text(merged)
    with ZipFile(page.with_suffix('.zip'), 'w', ZIP_DEFLATED) as archive:
        archive.writestr(page.name, merged)
    mirror = ROOT / 'outputs/action_alignment_audit/tcp_coverage_v2_v3.html'
    mirror.write_text(merged)
    print('Updated', page, len(merged.encode()), 'bytes', flush=True)
    return page


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reuse-samples', action='store_true')
    args = parser.parse_args()
    if not args.reuse_samples:
        extract_existing(OUTPUT / 'projection_samples')
        from scripts.data.projection_v3_samples import extract_all
        extract_all(OUTPUT / 'projection_samples')
    build_report()
