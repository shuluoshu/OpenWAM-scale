#!/usr/bin/env python3
"""Audit one episode per Mora source and build a standalone offline HTML report.

Run from repository root: python -m scripts.visualize_fk_consistency
"""
from __future__ import annotations

import argparse
import base64
import html
import importlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

from openwam.dataloader.registry import build_dataset
from openwam.dataloader.utils.video_io import decode_video_frames
from scripts.data.urdf_fk_audit import load_episode, pose_errors

PROFILES = {
    'interndata': 'configs/dataloader/canonical_eef_interndata_smoke.yaml',
    'agibotworld': 'configs/dataloader/canonical_eef_agibotworld_smoke.yaml',
    'robodojo': 'configs/dataloader/canonical_eef_robodojo_openloop.yaml',
    'robotwin': 'configs/dataloader/canonical_eef_robotwin_openloop.yaml',
}
NAMES = {'interndata': 'InternData · Genie-1', 'agibotworld': 'AgiBotWorld · A2D',
         'robodojo': 'RoboDojo · ARX-X5', 'robotwin': 'RobotWin · Aloha'}
COPY = {
    'interndata': {
        'badge': '机械臂局部 EEF 通过', 'color': 'pass',
        'scope': '相同 arm_base_link 下，原始 EE 标签 vs URDF arm_l/r_end_link。FK 不使用 pose 标签。',
        'finding': '源数据的 left/right_armbase 均对应公共 arm_base_link。此前按两个独立臂基座比较造成的大偏差已解释。',
        'limit': '全机器人基座位姿缺少躯干 joint；当前 reader 的 TCP 还包含 EE→TCP 偏移。下方 TCP 复现使用首帧 pose 推得的固定变换，单独列为诊断。左臂几乎静止，检查覆盖有限。',
    },
    'agibotworld': {
        'badge': '双臂相对检查通过', 'color': 'partial',
        'scope': '绝对比较仅为躯干 joint=0 的显式假设基线，不能作为绝对 FK 结论。有效检查为左末端坐标下的右末端位姿。',
        'finding': 'T_left⁻¹ T_right 消去两臂共享的躯干变换，可独立检验两条机械臂链的一致性；使用 metadata 声明的末端轴转换。',
        'limit': '当前发布 parquet 不含 idx01_body_joint1 / idx02_body_joint2，无法完整恢复 robot_base 下绝对 EEF。双臂相对通过不能认证全身坐标标定。',
    },
    'robodojo': {
        'badge': '源配置绝对 EEF 通过', 'color': 'pass',
        'scope': '使用实际 RoboDojo dual_x5.yml 的左右安装位姿，将 URDF link6 FK 转到仿真环境坐标，与原始 endpose 比较。',
        'finding': '已找到实际 RoboDojo 源码、X5A URDF 与 dual_x5 场景配置。joint 和左右基座变换直接来自数据／源配置；绝对检查无需标签拟合。',
        'limit': '使用目前可访问的源代码与配置；数据未记录生成时的代码提交。配置四元数 [0.707,0,0,0.707] 需单位化，残留约 0.017° 的固定偏差；未用拟合消除。末端为 link6，不是另一个工具 TCP。',
    },
    'robotwin': {
        'badge': '绝对 EEF 重建通过', 'color': 'pass',
        'scope': '由 metadata 声明的 world→base 位姿与 URDF fl_link6 / fr_link6 重建世界系 flange pose。',
        'finding': 'joint 顺序、URDF 末端 link 和基座位姿均有转换 metadata，可直接复现当前 EEF 标签。',
        'limit': '这些 EEF 标签本身由同一 URDF/FK 生成，因此这是转换与映射一致性验证，不是独立真实机器人标定验证；末端仍是 flange，不是额外工具 TCP。',
    },
}


def summary_stats(values):
    return {'median': float(np.median(values)), 'p95': float(np.percentile(values, 95)),
            'max': float(np.max(values)), 'rms': float(np.sqrt(np.mean(np.square(values))))}


def compare(label, fk):
    pos, rot = pose_errors(label, fk)
    if pos.ndim == 1:
        pos, rot = pos[:, None], rot[:, None]
    return {'position_mm': [summary_stats(v) for v in pos.T],
            'rotation_deg': [summary_stats(v) for v in rot.T],
            'all_below_1mm_1deg': bool(pos.max() < 1 and rot.max() < 1)}


def relative_poses(poses, reference):
    return np.linalg.inv(reference)[None] @ poses


def image_data(image, fmt='PNG'):
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return f'data:image/{fmt.lower()};base64,'+base64.b64encode(buffer.getvalue()).decode()


def head_frames(key, frame_indices, output_dir):
    cfg, info, episode, _ = load_episode(PROFILES[key], columns=[])
    camera = cfg.head_camera
    root = Path(cfg.get('video_root') or cfg.dataset_dir)
    video_path = root/info['video_path'].format(video_key=camera,
            chunk_index=int(episode[f'videos/{camera}/chunk_index']), file_index=int(episode[f'videos/{camera}/file_index']))
    # All adapters select episode 0: still assert the episode's media starts at 0.
    timestamp_key = f'videos/{camera}/from_timestamp'
    base = round(float(episode.get(timestamp_key, 0))*float(info['fps']))
    picks = [int(frame_indices[i]) for i in (0, len(frame_indices)//2, len(frame_indices)-1)]
    frames = []
    for index in picks:  # seek each frame; do not decode the entire long clip
        frames.extend(decode_video_frames(str(video_path), [base+index], height=240, width=320))
    if len(frames) != 3:
        raise ValueError(f'{key}: expected three source frames, got {len(frames)}')
    sheet = Image.new('RGB', (960, 268), '#17293b')
    draw = ImageDraw.Draw(sheet)
    for i, (frame, index) in enumerate(zip(frames, picks)):
        sheet.paste(frame, (320*i, 28))
        draw.text((320*i+10, 7), f'episode 0 | frame {index} | {index / info["fps"]:.2f} s', fill='white')
    sheet.save(output_dir/'source_frames.png')
    return image_data(sheet), {'video_path': str(video_path), 'frame_indices': picks, 'file_frame_offset': base}


def plot_comparison(label, fk, time, title, path):
    if label.ndim == 3:
        label, fk = label[:, None], fk[:, None]
    arms = label.shape[1]
    pos, angle = pose_errors(label, fk)
    fig = plt.figure(figsize=(13, 9), facecolor='white')
    fig.subplots_adjust(left=.08, right=.98, bottom=.07, top=.92, wspace=.24, hspace=.34)
    arm_names = ['Left arm', 'Right arm'] if arms == 2 else ['Right EEF in left EEF frame']
    for side in range(2):
        if side >= arms:
            ax = fig.add_subplot(2, 2, side+1)
            for dim, color in enumerate(('#087f8c', '#525cbd', '#bd5e23')):
                ax.plot(time, label[:, 0, dim, 3], color=color, label='XYZ'[dim]+' label')
                ax.plot(time, fk[:, 0, dim, 3], color=color, linestyle='--', linewidth=1)
            ax.set(xlabel='Episode time (s)', ylabel='Position (m)', title='Coordinates: label solid / FK dashed')
            ax.grid(alpha=.22)
            ax.legend(fontsize=8)
            continue
        ax = fig.add_subplot(2, 2, side+1, projection='3d')
        gt, pred = label[:, side, :3, 3], fk[:, side, :3, 3]
        ax.plot(*gt.T, color='#087f8c', linewidth=2.7, label='Label')
        ax.plot(*pred.T, color='#ec7a23', linewidth=1.3, linestyle='--', label='URDF FK')
        ax.scatter(*gt[0], color='#087f8c', s=24, marker='o')
        ax.scatter(*pred[0], color='#ec7a23', s=24, marker='x')
        xyz = np.concatenate((gt, pred))
        center = (xyz.min(0)+xyz.max(0))/2
        radius = max(float(np.ptp(xyz, axis=0).max())/2, .002)
        for setter, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
            setter(c-radius, c+radius)
        ax.set_box_aspect((1, 1, 1))
        ax.set(xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)', title=arm_names[side])
        ax.legend(loc='upper left', fontsize=8)
    for i, (error, label_axis) in enumerate(((pos, 'Translation error (mm)'), (angle, 'Rotation error (deg)'))):
        ax = fig.add_subplot(2, 2, i+3)
        for side in range(arms):
            ax.plot(time, error[:, side], label=arm_names[side], linewidth=1)
        ax.set(xlabel='Episode time (s)', ylabel=label_axis)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 4))
        ax.grid(alpha=.22)
        ax.legend(fontsize=8)
    fig.suptitle(title, fontsize=13)
    fig.savefig(path, dpi=145)
    plt.close(fig)
    return image_data(Image.open(path))


def interactive_plot(label, fk):
    if label.ndim == 3:
        label, fk = label[:, None], fk[:, None]
    arms = label.shape[1]
    fig = make_subplots(rows=1, cols=arms, specs=[[{'type': 'scene'}]*arms],
                        subplot_titles=['Left arm', 'Right arm'] if arms == 2 else ['Right EEF in left EEF frame'])
    # Reduce only plotted vertices; all statistics use every original frame.
    indices = np.unique(np.linspace(0, len(label)-1, min(800, len(label))).astype(int))
    for side in range(arms):
        for name, arr, color, dash in [('Label', label, '#087f8c', 'solid'), ('URDF FK', fk, '#ec7a23', 'dash')]:
            points = arr[indices, side, :3, 3]
            fig.add_trace(go.Scatter3d(x=points[:,0], y=points[:,1], z=points[:,2], mode='lines',
                name=f'{name} {side+1}', line={'color': color, 'width': 5, 'dash': dash},
                customdata=indices, hovertemplate='target index=%{customdata}<br>x=%{x:.6f}<br>y=%{y:.6f}<br>z=%{z:.6f}<extra>%{fullData.name}</extra>'), row=1, col=side+1)
    fig.update_scenes(aspectmode='data', xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)')
    fig.update_layout(height=420, margin={'l':0,'r':0,'t':40,'b':0}, paper_bgcolor='white', showlegend=True)
    return fig.to_html(full_html=False, include_plotlyjs=False, config={'displaylogo':False,'responsive':True})


def metric_table(stats):
    rows = []
    for i, (pos, rot) in enumerate(zip(stats['position_mm'], stats['rotation_deg'])):
        name = ['左臂','右臂'][i] if len(stats['position_mm']) == 2 else '双臂相对'
        rows.append('<tr><th>'+name+'</th>'+''.join(
            f'<td>{values[key]:.6g}</td>' for values in (pos, rot) for key in ('median','p95','max','rms'))+'</tr>')
    return '<div class="scroll"><table><thead><tr><th rowspan="2">对象</th><th colspan="4">位置误差 mm</th><th colspan="4">旋转误差 °</th></tr><tr>'+''.join('<th>'+x+'</th>' for x in ['中位数','P95','最大','RMS']*2)+'</tr></thead><tbody>'+''.join(rows)+'</tbody></table></div>'


def reader_joint_check(key, arrays):
    """Check actual reader supervision reconstructs the audited absolute joints."""
    cfg = OmegaConf.load(PROFILES[key])
    cfg.return_trace = True
    ds = build_dataset(cfg)
    # Bypass video decoding solely for this numerical reader check.
    ds._decode_window_video = lambda *args, **kwargs: []
    trace = ds[0]['trace']
    reconstructed = trace['raw_action_joint14'] + trace['raw_proprio_joint14']
    n = min(len(reconstructed), len(arrays['joint']))
    delta = np.abs(reconstructed[:n]-arrays['joint'][:n])
    error = float(delta.max())
    if error > 1e-6:
        raise ValueError(f'{key}: reader reconstructed joints differ from audited labels: {error}')
    return {'checked_targets': n, 'max_abs_difference_rad': error,
            'formula': 'raw_proprio_joint14[t] + raw_action_joint14[k] == absolute_joint[t+k+1]',
            'normalized_network_targets_used_directly_as_angles': False}


def render_report(output_root):
    output_root = Path(output_root)
    cards, rows, results = [], [], {}
    for key in PROFILES:
        folder = output_root/key
        metadata = json.loads((folder/'metadata.json').read_text())
        with np.load(folder/'arrays.npz') as data:
            arrays = {name: data[name] for name in data.files}
        label, fk, times = arrays['label_pose'], arrays['fk_pose'], arrays['time']
        if label.shape != fk.shape or label.shape[1:] != (2,4,4):
            raise ValueError(f'{key}: invalid comparison shape {label.shape}/{fk.shape}')
        if not all(np.isfinite(v).all() for v in (label, fk, times)):
            raise ValueError(f'{key}: nonfinite poses')
        checks = reader_joint_check(key, arrays)
        primary = compare(label, fk)
        ref_label, ref_fk = arrays['reference_label_pose'], arrays['reference_fk_pose']
        rel_label, rel_fk = relative_poses(label, ref_label), relative_poses(fk, ref_fk)
        relative = compare(rel_label, rel_fk)
        contexts, media = head_frames(key, arrays['frame_indices'], folder)
        primary_png = plot_comparison(label, fk, times, NAMES[key]+' | direct comparison (scope below)', folder/'direct_comparison.png')
        main_label, main_fk = label, fk
        if key == 'agibotworld':
            main_label, main_fk = arrays['cross_arm_label_pose'], arrays['cross_arm_fk_pose']
            main_title = '有效检查：左末端坐标下的右末端（消去未知躯干变换）'
        else:
            main_title = '有效检查：'+COPY[key]['scope']
        validated = compare(main_label, main_fk)
        if not validated['all_below_1mm_1deg']:
            raise ValueError(f'{key}: expected scoped check failed; report wording must be revised')
        active_png = plot_comparison(main_label, main_fk, times, NAMES[key]+' | scoped kinematic check', folder/'scoped_comparison.png')
        extra = ''
        if key == 'interndata':
            tcp_stats = compare(arrays['tcp_label_pose'], arrays['tcp_diagnostic_fk_pose'])
            tcp_png = plot_comparison(arrays['tcp_label_pose'], arrays['tcp_diagnostic_fk_pose'], times,
                'InternData | robot-frame TCP, reference-derived transforms (diagnostic only)', folder/'tcp_diagnostic.png')
            extra = '<details><summary>TCP 诊断：使用首帧推得的基座／工具变换，不计作独立通过</summary><p>URDF EE 与源 TCP 相差沿 EE 局部 +Z 的 0.22 m。这里固定使用第 0 帧推得的 base 与 tool 变换，检查后续整条轨迹；不逐帧拟合。</p>'+metric_table(tcp_stats)+f'<img alt="TCP diagnostic" src="{tcp_png}"></details>'
        ranges = np.ptp(arrays['joint'], axis=0)
        pmax = max(v['max'] for v in validated['position_mm'])
        rmax = max(v['max'] for v in validated['rotation_deg'])
        rows.append(f'<tr><td><a href="#{key}">{NAMES[key]}</a></td><td>{len(times):,}</td><td>{COPY[key]["badge"]}</td><td>{pmax:.6g}</td><td>{rmax:.6g}</td></tr>')
        full = {'source_metadata': metadata, 'media': media, 'reader_joint_check':checks,
                'direct_comparison':primary, 'temporal_relative_comparison':relative,
                'scoped_comparison':validated, 'joint_peak_to_peak_rad':ranges.tolist(),
                'primary_scope': COPY[key]['scope'], 'scoped_check_title':main_title}
        results[key] = full
        (folder/'report_metrics.json').write_text(json.dumps(full,indent=2,ensure_ascii=False))
        cards.append(f'''<section id="{key}"><div class="section-title"><h2>{NAMES[key]}</h2><span class="badge {COPY[key]['color']}">{COPY[key]['badge']}</span></div>
<p class="sub">episode 0 · {len(times):,} 个 t+1 标签 · 帧 {int(arrays['frame_indices'][0])}–{int(arrays['frame_indices'][-1])} · {float(times[-1]):.2f} s</p>
<div class="note">{COPY[key]['finding']}</div><p>{COPY[key]['limit']}</p>
<img class="context" alt="{key} source head camera frames" src="{contexts}"><p class="caption">同一轨迹起始、中间、末尾头相机画面，仅作为任务上下文，不作重投影标定证据。</p>
<h3>{main_title}</h3>{metric_table(validated)}<img alt="{key} scoped FK comparison" src="{active_png}">
<details><summary>交互式 3D 轨迹：拖动旋转，滚轮缩放</summary>{interactive_plot(main_label, main_fk)}</details>
<details><summary>直接绝对／局部坐标比较与误差（含未确认基座的诊断基线）</summary><p>{COPY[key]['scope']}</p>{metric_table(primary)}<img alt="{key} direct comparison" src="{primary_png}"></details>
{extra}<details><summary>当前 reader 的 joint 监督、URDF、坐标定义与完整证据</summary>
<p>已将 reader 的前 {checks['checked_targets']} 个原始 joint delta 加回当前 joint，重建绝对 q；与本审计输入最大差 {checks['max_abs_difference_rad']:.3g} rad。未把归一化数值直接当关节角。</p>
<p>各 joint 在本轨迹中的范围（rad）：<code>{html.escape(np.array2string(ranges,precision=4))}</code></p>
<pre>{html.escape(json.dumps(full,ensure_ascii=False,indent=2))}</pre></details>
<p class="downloads"><a href="{key}/arrays.npz" download>下载逐帧 pose / joints NPZ</a> · <a href="{key}/report_metrics.json" download>下载指标与来源 JSON</a></p></section>''')
    generated = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    doc = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Mora joint → URDF FK 一致性检查</title>
<style>:root{color-scheme:light;--ink:#142b40;--muted:#5b7182;--line:#dce6ed}*{box-sizing:border-box}body{margin:0;background:#edf3f7;color:var(--ink);font:15px/1.7 system-ui,-apple-system,"Noto Sans CJK SC",sans-serif}main{max-width:1240px;margin:auto;padding:28px 24px 70px}header{background:linear-gradient(125deg,#143e5e,#167a87);color:white;padding:36px;border-radius:18px}header p{max-width:930px;color:#e0f1f7}h1{font-size:31px;line-height:1.3;margin:0 0 18px}h2{font-size:23px;margin:0}h3{font-size:17px;margin-top:26px}nav{display:flex;gap:14px;flex-wrap:wrap;margin:18px 0}nav a{background:white;padding:8px 15px;border-radius:8px}a{color:#066f91;text-decoration:none}a:hover{text-decoration:underline}section,.overview{padding:26px;background:white;margin-top:22px;border:1px solid var(--line);border-radius:14px;box-shadow:0 5px 22px #142b4005}.section-title{display:flex;gap:15px;justify-content:space-between;align-items:center;flex-wrap:wrap}.badge{padding:5px 13px;border-radius:30px;font-size:13px;font-weight:650}.pass{background:#dcf4ed;color:#10644b}.partial{background:#fff0cf;color:#825b04}.sub,.caption{color:var(--muted);font-size:13px}.note{background:#eaf4f7;padding:14px 18px;border-left:4px solid #128595;border-radius:4px}img{width:100%;height:auto;display:block;margin:12px 0}.context{border-radius:8px}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-size:13px;margin:14px 0}th,td{text-align:left;padding:10px 11px;border-bottom:1px solid var(--line);white-space:nowrap}th{background:#f3f7fa}details{border:1px solid var(--line);border-radius:8px;padding:12px 15px;margin:14px 0}summary{cursor:pointer;color:#215771;font-weight:600}pre{font:12px/1.6 ui-monospace,monospace;background:#f3f7fa;padding:18px;overflow:auto;max-height:480px}code{font-size:12px}.downloads{font-size:13px}.formula{font-family:ui-monospace,monospace;background:#f3f7fa;border-radius:7px;padding:14px;overflow:auto}footer{color:var(--muted);font-size:13px;padding:24px 0}@media(max-width:650px){main{padding:12px}header,section,.overview{padding:18px}h1{font-size:25px}}
</style><script>'''+get_plotlyjs()+'''</script></head><body><main><header><h1>joint 标签 → URDF → FK<br>四数据源逐轨迹一致性检查</h1><p>固定选择每个当前 profile 的 episode 0，使用全部有效 state[t+1] 标签。分别核对关节顺序、机械臂链、坐标参考系与末端定义。只有独立坐标映射支持的检查才记为通过；缺失基座状态与首帧推得的变换均单独注明。</p><small>'''+generated+''' · CPU / 只读数据审计 · 无模型训练</small></header>
<nav>'''+''.join(f'<a href="#{k}">{v}</a>' for k,v in NAMES.items())+'''</nav><div class="overview"><h2>结果总览</h2><p>四个样本均通过各自明确限定范围的运动学检查。<b>不等于四个数据源的绝对 TCP 均已验证</b>：InternData 为臂基座下的 EE，AgiBot 为双臂相对关系，RoboDojo 为源场景配置下的绝对 EEF，RobotWin 为绝对 flange 重建。</p>
<div class="scroll"><table><thead><tr><th>数据源</th><th>检查目标数</th><th>已验证范围</th><th>最大位置误差 mm</th><th>最大旋转误差 °</th></tr></thead><tbody>'''+''.join(rows)+'''</tbody></table></div>
<p>表中均为有效检查的最大误差。1 mm / 1° 只是本次审计的展示阈值，不是数据集承诺；所有统计使用完整轨迹，3D 交互显示最多 800 个点。每源仅一条轨迹，不能推断全数据集通过。</p>
<div class="formula">q_target[k] = q_current + unnormalized_joint_delta[k] = q[t+k+1]<br>T_FK = T_reference_base · FK_URDF(q_target) · T_end_tool<br>position_error = ‖p_label − p_FK‖₂ · 1000 mm<br>rotation_error = angle(R_labelᵀ R_FK) in degrees</div>
<p>相对运动检查：T(0)⁻¹T(t)，消去固定安装变换。双臂相对检查：T_left(t)⁻¹T_right(t)，消去两臂共享的躯干变换。标签与 FK 分别计算这些量，无逐帧配准。</p></div>'''+''.join(cards)+'''<footer>复现：<code>python -m scripts.visualize_fk_consistency --output-dir outputs/fk_consistency</code><br>此 HTML 内嵌所有图像和交互库，可离线打开；下载链接指向同目录下的 JSON / NPZ 原始审计产物。</footer></main><script>document.querySelectorAll("details").forEach(d=>d.addEventListener("toggle",()=>{if(d.open)d.querySelectorAll(".plotly-graph-div").forEach(g=>Plotly.Plots.resize(g));}));</script></body></html>'''
    path = output_root/'index.html'
    path.write_text(doc)
    (output_root/'summary.json').write_text(json.dumps(results,indent=2,ensure_ascii=False))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/fk_consistency'))
    parser.add_argument('--reuse-results',action='store_true')
    args = parser.parse_args()
    if not args.reuse_results:
        for key, module in [('interndata','interndata'),('agibotworld','agibot'),('robodojo','robodojo'),('robotwin','robotwin')]:
            importlib.import_module('scripts.data.fk_audit_'+module).run(args.output_dir/key)
    print(render_report(args.output_dir))


if __name__ == '__main__':
    main()
