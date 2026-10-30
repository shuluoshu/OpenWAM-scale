"""Offline live report for the timed, frozen masked-SA Stage1 comparison."""
import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import shutil
import statistics
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

CODE=Path(__file__).resolve().parents[2];sys.path.insert(0,str(CODE))
from openwam.train.z_action_report import _chart, atomic_json
from openwam.train.stage2_eval import _json_safe
from scripts.experiments.stage1_tcp_report import tcp_overlay_data, draw_tcp_overlay, coverage_rows
from scripts.experiments.stage12_video import render_tcp_video_frame

MODES=['joint','none'];E=html.escape
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'

def read(path,default=None):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return {} if default is None else default

def lines(path):
    if not path.exists():return []
    result=[]
    for line in path.read_text().splitlines():
        try:result.append(json.loads(line))
        except ValueError:pass  # A live writer can be between writes.
    return result

def fmt(x):return f'{x:.5g}' if isinstance(x,(float,int)) and np.isfinite(x) else 'N/A'

def metric(v):
    a=np.asarray(v,float).reshape(-1);a=a[np.isfinite(a)]
    return dict(value=float(a.mean()) if len(a) else None,valid_count=len(a))

def save_json(path,value):atomic_json(path,_json_safe(value))

def video(path,frames,rate=4):
    import av
    temporary=path.with_name('.'+path.stem+'.pending.mp4')
    with av.open(str(temporary),'w') as c:
        s=c.add_stream('libx264',rate=rate);s.width=frames[0].width;s.height=frames[0].height;s.pix_fmt='yuv420p'
        for im in frames:
            for packet in s.encode(av.VideoFrame.from_image(im)):c.mux(packet)
        for packet in s.encode():c.mux(packet)
    temporary.replace(path)

def projection_panel(image, data, title, path):
    """Fixed current camera/background. Out-of-frame predictions are not clipped."""
    font=ImageFont.truetype(FONT,12);canvas=Image.new('RGB',(640,330),'white')
    for side,color in enumerate(['#d93636','#267acc']):
        tile=image.crop((0,0,320,256)).copy();draw=ImageDraw.Draw(tile)
        gt=np.asarray(data['gt_uv'])[:,side];pred=np.asarray(data['pred_uv'])[:,side]
        valid=np.asarray(data['metric_valid'])[:,side].astype(bool)&np.asarray(data['gt_in_frame'])[:,side].astype(bool)
        projectable=np.asarray(data['pred_projectable'])[:,side].astype(bool)
        inside=np.asarray(data['pred_in_frame'])[:,side].astype(bool)
        for uv,ok,width in [(gt,valid,2),(pred,valid&projectable&inside,1)]:
            for k in range(1,len(uv)):
                if ok[k-1] and ok[k]:draw.line([tuple(uv[k-1]),tuple(uv[k])],fill=color,width=width)
        for k in range(len(gt)):
            if valid[k]:
                x,y=gt[k];draw.rectangle((x-2,y-2,x+2,y+2),outline=color,width=1)
            if valid[k] and projectable[k] and inside[k]:
                x,y=pred[k];draw.ellipse((x-3,y-3,x+3,y+3),outline=color,width=1)
        canvas.paste(tile,(side*320,74))
    d=ImageDraw.Draw(canvas);d.text((6,4),title,font=font,fill='black')
    d.text((6,23),'Fixed current camera / background | Left red, right blue | GT squares, pred circles',font=font,fill='black')
    d.text((6,42),'Out-of-frame predictions are not clipped. Invalid supervision is excluded.',font=font,fill='black')
    canvas.save(path)

def stage1_sample_path(run,record):
    """Use the run's actual prepared RGB cache; legacy runs retain their records."""
    protocol=read(run/'stage1/protocol.json')
    smoke=protocol.get('smoke_data') or read(run/'plan.json').get('smoke_data')
    root=Path(smoke) if smoke else Path(record['observed_image_path']).parents[2]
    return root/'samples'/f'{record["source_index"]:02d}_{record["anchor_index"]}.pt'


def stage1_section(run,out):
    status=read(run/'stage1/status.json');vals=read(run/'stage1/report/metrics.json')
    parts=['<h2>Stage1：masked SA + 视觉历史融合</h2>',f'<p>状态 {E(str(status.get("status","待启动")))}；更新 {status.get("step",0)}；耗时 {fmt(status.get("elapsed_seconds"))} 秒。</p>',
        '<p>继续训练最新 masked SA checkpoint；实际每例指令走原生 text CA。92 个训练锚点、12 家族；历史 10 槽经 learned pooling 压缩为 32 tokens，与当前视觉融合。单帧/历史对比使用同一个模型和相同采样噪声。</p>']
    if (out/'stage1_preview/index.html').exists():parts.append('<p><a href="stage1_preview/index.html">Stage1 中途 checkpoint 的真实图像 / TCP 预测预览</a></p>')
    if (run/'stage1/rgb_prediction_review.png').exists():
        for name in ['rgb_prediction_review.png','rgb_prediction_review.json']:
            shutil.copy2(run/'stage1'/name,out/name)
        review=read(run/'stage1/rgb_prediction_review.json',[])
        parts.append('<h3>RGB 修正前后的原始预测</h3><p>面包预测恢复米黄/棕色，杯呈粉色；面包案例的整体 MAE 变差，笔筒案例改善。实际图中仍有绿/粉色笔，不能认定指令“红黑笔”语义已经匹配。下表把旧、新预测都与同一个修正后 GT 比较，不对输出做换色。</p><img src="rgb_prediction_review.png"><table><tr><th>案例</th><th>旧预测对修正 GT MAE</th><th>新预测对修正 GT MAE</th></tr>')
        for row in review:
            parts.append(f'<tr><td>{E(row["family"])}</td><td>{fmt(row["old_history_saved_png_mae_against_corrected_gt"])}</td><td>{fmt(row["new_history_saved_png_mae_against_corrected_gt"])}</td></tr>')
        parts.append('</table><p><a href="rgb_prediction_review.json">原始指标及同 GT 对照数值</a></p>')
    if (out/'rgb_audit/index.html').exists():parts.append('<p class="notice"><a href="rgb_audit/index.html">RGB 专项核验：官方 / PyAV / OpenCV / 缓存与红蓝互换诊断</a>。核验的 4 帧在三种解码路径中逐像素相同，缓存也与源缩放像素一致；这只证明解码路径一致，不证明源视频颜色正确。原始 MP4 显示紫笔紫杯，红蓝交换后为红笔粉杯，疑似上游错色；尚无原录制证明，本报告不改训练 RGB。</p>')
    if not vals:return ''.join(parts)
    link=out/'stage1';target=run/'stage1/report'
    if link.is_symlink():link.unlink()
    link.mkdir(exist_ok=True)
    for source in target.rglob('*'):
        if source.is_file() and source.suffix in ['.png','.json','.html']:
            dest=link/source.relative_to(target);dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists() or dest.stat().st_size!=source.stat().st_size or dest.stat().st_mtime<source.stat().st_mtime:
                shutil.copy2(source,dest)
    prepared=[]
    for x in vals['cases']:
        record=x['record']
        source=stage1_sample_path(run,record)
        original_record,original=torch.load(source,weights_only=False,map_location='cpu')
        if original_record['id']!=record['id']:raise ValueError('Stage1 report sample identity mismatch')
        tcp={mode:tcp_overlay_data(record,original['targets'],x['values'][mode]) for mode in ['single','history']}
        prepared.append((x,original,tcp))
    coverage=coverage_rows([dict(record=x['record'],split=x['split'],tcp=tcp['history']) for x,_,tcp in prepared])
    save_json(out/'stage1_tcp_coverage_v2.json',dict(version=2,scope=f'{len(prepared)} saved evaluation cases; not full training-set coverage',families=coverage))
    parts.append('<h3>Stage1 TCP 显示 v2：GT 与未监督预测分开显示</h3><p class="notice">绿色方框是有效 GT；有有效 GT 的预测用红圈。没有有效 GT 的预测仅用橙圈显示，属于该样例/手臂无监督、不可信的二维输出，不能视为标定投影或成功抓取点。GT N/A 时不计算误差，不把 (0,0) 占位符画成坐标。有限但画外的预测保留数值，不截断到图内。</p>')
    parts.append(f'<p>坐标按 320×256 目标图：x=u×319，y=v×255。以下覆盖统计仅针对当前已保存的 {len(prepared)} 个评估样例（完整评估为 12 家族各 fit / alternate_anchor，共 24 例），不是全部训练数据覆盖率。</p><table><tr><th>家族</th><th>有效 GT 点 / 可能点</th><th>fit / alternate 点</th><th>缺失原因</th></tr>')
    for row in coverage:
        parts.append(f'<tr><td>{E(row["family"])}</td><td>{row["valid_gt_points"]} / {row["possible_gt_points"]}</td><td>{row["fit_points"]} / {row["alternate_points"]}</td><td>{E("; ".join(row["reasons"]) or "有效 GT 可比较")}</td></tr>')
    parts.append('</table><p><a href="stage1_tcp_coverage_v2.json">TCP 覆盖与缺失原因 JSON</a></p>')
    for x,original,tcp in prepared:
        i=x['case'];a=x['values']['single'];b=x['values']['history'];record=x['record']
        overlay=out/'stage1_tcp_v2'/f'case_{i:03d}';overlay.mkdir(parents=True,exist_ok=True)
        gt_image=Image.open(link/f'inputs/case_{i:04d}_target.png').convert('RGB')
        draw_tcp_overlay(gt_image,tcp['history'],overlay/'gt.png',f'Case {i} | GT target',show_predictions=False)
        for mode in ['single','history']:
            draw_tcp_overlay(gt_image,tcp[mode],overlay/f'{mode}_on_gt.png',f'{mode} | GT background')
            draw_tcp_overlay(Image.open(link/f'case_{i:03d}/{mode}/prediction.png'),tcp[mode],overlay/f'{mode}_on_prediction.png',f'{mode} | generated background')
        save_json(overlay/'coordinates.json',dict(case=i,source_index=record['source_index'],anchor_index=record['anchor_index'],
            anchor_frame=record['anchor_source_frame'],goal_frame=record['goal_source_frame'],modes=tcp))
        parts.append(f'<details><summary>{E(record["family"])} · {E(x["split"])} · 图像 MAE 单帧 {fmt(a["mae"])} / 历史 {fmt(b["mae"])}, TCP px {fmt(a["tcp_error_px"])} / {fmt(b["tcp_error_px"])}</summary>')
        parts.append(f'<p>{E(record["instruction"])}；当前 f{record["anchor_source_frame"]} → 目标 f{record["goal_source_frame"]}；有效 GT TCP {tcp["history"]["valid_gt_count"]}/2。alternate_anchor 是同 episode 另一锚点，并非未见 episode；其中部分曾用于之前诊断。</p>')
        if tcp['history']['reason']:parts.append('<p class="notice">GT 不可用原因：'+E(tcp['history']['reason'])+'。橙色输出仅作未监督预测检查。</p>')
        prefix=f'stage1_tcp_v2/case_{i:03d}'
        gt_label='GT 目标；GT TCP N/A（没有有效坐标）' if not tcp['history']['valid_gt_count'] else 'GT 目标；有效 GT 为绿色方框'
        parts.append('<div class="row">')
        for name,label in [(f'stage1/inputs/case_{i:04d}_observed.png','实际当前输入'),(f'{prefix}/gt.png',gt_label),
                (f'{prefix}/single_on_prediction.png','单帧生成图 + 二维 TCP 输出'),(f'{prefix}/history_on_prediction.png','历史生成图 + 二维 TCP 输出'),
                (f'{prefix}/single_on_gt.png','GT 图上显示单帧输出（无 GT 时为橙色不可信预测）'),(f'{prefix}/history_on_gt.png','GT 图上显示历史输出（无 GT 时为橙色不可信预测）')]:
            parts.append(f'<figure><img loading="lazy" src="{name}"><figcaption>{label}</figcaption></figure>')
        parts.append('</div><table><tr><th>模式 / 手臂</th><th>GT 像素</th><th>预测归一化 UV</th><th>预测像素 x,y</th><th>mask / 状态</th></tr>')
        def pair(value):return 'N/A' if value is None else ', '.join(f'{v:.4f}' for v in value)
        for mode in ['single','history']:
            for arm in tcp[mode]['arms']:
                state='有有效 GT 可比较' if arm['gt_valid'] else '无有效 GT；未监督、不可信预测'
                if not arm['pred_drawable']:state+='；预测非有限或画外，不绘制'
                parts.append(f'<tr><td>{mode} / {arm["side"]}</td><td>{pair(arm["gt_px"])}</td><td>{pair(arm["pred_uv"])}</td><td>{pair(arm["pred_px"])}</td><td>point_valid={arm["point_valid"]}, in_frame={arm["in_frame"]}；{state}</td></tr>')
        parts.append(f'</table><p><a href="{prefix}/coordinates.json">坐标、mask、端点与相机来源 JSON</a></p><p>历史头图输入（旧 → 新）；灰块不参与 attention：</p><div class="row">')
        for j in range(10):
            target=out/'assets'/f'stage1_{i:03d}_history_{j:02d}.png';valid=bool(original['history_valid'][0,j]);age=float(original['history_ages'][0,j])
            if not target.exists() or target.stat().st_mtime<stage1_sample_path(run,record).stat().st_mtime:
                if valid:
                    from openwam.train.goal_overfit import _display_image
                    _display_image(original['history_images'][:,j].to(torch.bfloat16)).save(target)
                else:
                    image=Image.new('RGB',(320,256),'#e5e9ef');ImageDraw.Draw(image).text((120,120),'MASKED',fill='gray');image.save(target)
            parts.append(f'<figure style="width:144px"><img loading="lazy" src="assets/{target.name}"><figcaption>slot{j} · '+(f'{age:g}s 前' if valid else 'MASKED')+'</figcaption></figure>')
        parts.append('</div></details>')
    return ''.join(parts)

def stage2_sources(run):
    """Recovery is a distinct run; never mix its updates or assets with original none."""
    roots={m:run/'stage2'/m for m in MODES}
    recovery=run/'none_recovery'
    active=(recovery/'none/status.json').is_file()
    if active:roots['none']=recovery/'none'
    return roots,active,read(recovery/'recovery.json')


def render(run,samples_path,*,geometry=True):
    out=run/'report';out.mkdir(exist_ok=True);assets=out/'assets';assets.mkdir(exist_ok=True)
    roots,recovering,recovery=stage2_sources(run)
    plan=read(run/'plan.json');supervisor=read(run/'supervisor.json');updates={m:lines(roots[m]/'updates.jsonl') for m in MODES}
    status={m:read(roots[m]/'status.json') for m in MODES}
    if recovering:
        archive=out/'index_before_none_recovery.html'
        if (out/'index.html').exists() and not archive.exists():shutil.copy2(out/'index.html',archive)
        evidence=out/'evidence_original_none';evidence.mkdir(exist_ok=True)
        for name in ['status.json','updates.jsonl']:
            if (run/'stage2/none'/name).exists():shutil.copy2(run/'stage2/none'/name,evidence/name)
        old_snapshots=sorted((run/'stage2/none').glob('eval_*/metrics.json'))
        if old_snapshots:shutil.copy2(old_snapshots[-1],evidence/'latest_metrics.json')
        save_json(out/'none_recovery.json',recovery)
    parts=['<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Masked SA Stage1 → 两种 Stage2</title>',
        '<style>body{font:15px system-ui;margin:24px auto;max-width:1500px;padding:16px;color:#17263c;background:#f6f8fb}h1,h2{color:#153451}section,details{background:white;padding:16px;margin:14px 0;border-radius:10px}table{border-collapse:collapse;width:100%;background:white}td,th{padding:9px;border:1px solid #d7dee8;text-align:left}.row{display:flex;gap:12px;flex-wrap:wrap}figure{margin:6px;width:320px}figure img,video{width:100%;height:auto}img{max-width:100%}.notice{background:#fff0cd;padding:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere}svg{max-width:100%;height:auto}summary{cursor:pointer;font-weight:600}</style>',
        '<h1>最新 masked SA Stage1 → joint / none Stage2</h1>',
        f'<p>页面更新：{datetime.now(timezone.utc).isoformat(timespec="seconds")}；总流程：{E(str(supervisor.get("status","Stage1 训练中，适配验证进行中")))}。</p>',
        '<p>两组共享同一个冻结 Stage1。Stage2 memory 是融合历史后的当前视觉 C*（120×3072），保留视图 mask；不是旧版 64+2 Z。joint 联合预测未来视频与动作；none 不生成未来视频，训练动作与有效 FK 目标。</p>',
        '<p class="notice">这是 smoke 小数据集拟合实验，不是泛化或任务成功率评估。直接 TCP 预测、关节→FK 与 Stage1 二维 TCP 是不同指标。缺监督、匹配 URDF 或可信相机标定的项保持 N/A。</p>']
    if (out/'stage12_comparison.html').exists():
        parts.append('<section id="stage12-contract-entry" class="notice"><h2><a href="stage12_comparison.html">同输入 Stage1 / Stage2 完整视频对比</a></h2><p>逐案例展示实际三视图和指令、同锚点 Stage1 图像与 TCP、真实逐帧背景上的直接 action/FK 投影，以及缺失项说明。支持共享暂停、拖动和逐采样帧切换。</p></section>')
    if plan.get('deadline_unix'):parts.append(f'<p>目标截止：{datetime.fromtimestamp(plan["deadline_unix"],timezone.utc).isoformat()}（北京时间 +8 小时）。</p>')
    if plan.get('smoke_data'):
        prepared=read(Path(plan['smoke_data'])/'prepared.json')
        parts.append('<section class="notice"><h2>RGB v2 输入修复后的短程验证</h2><p>训练与历史图都来自 '+E(plan['smoke_data'])+'。'
            +f'缓存共 {prepared.get("samples","N/A")} 例，重建修正 {prepared.get("corrected","N/A")} 例，其余 {prepared.get("unaffected","N/A")} 例保持原样。'
            +f'Stage1 预算 {plan.get("stage1_seconds","N/A")} 秒；两种 Stage2 各 {plan.get("stage2_seconds_per_arm","N/A")} 秒，从同一个原始 Stage2 checkpoint 和全新优化器开始，共享本次新冻结的 Stage1。'
            +'实际更新数、墙钟及保存/评估完成状态以下方记录为准。生成图为模型原始解码输出，没有对预测结果做红蓝互换。</p></section>')
    if recovering:
        parts.append('<section class="notice"><h2>none 独立恢复运行</h2><p>原 none 因 CUDA OOM 失败，未完成原定训练。当前 none 从已验证的 checkpoint_step_250 恢复模型权重，使用全新 Adam 优化器状态、warmup=0。下方 none 曲线、步数和墙钟仅属于本次较短恢复：步数是恢复后的新增更新，不是原实验总步数；不得描述为完成 4 小时训练。原 joint 保留原运行结果，两组当前初始化历史与训练预算不同。</p>')
        parts.append('<p>恢复训练最迟停止：2026-10-29 03:20 UTC / 北京时间 11:20；以实时状态中的实际耗时和完成状态为准，保存/评估可能随后完成。</p><p><a href="none_recovery.json">恢复来源、PID、预算记录</a> · <a href="evidence_original_none/status.json">原 none 失败状态与 traceback</a>')
        if (out/'evidence_original_none/updates.jsonl').exists():parts.append(' · <a href="evidence_original_none/updates.jsonl">原 none 训练记录</a>')
        if (out/'evidence_original_none/latest_metrics.json').exists():parts.append(' · <a href="evidence_original_none/latest_metrics.json">原 none 最后评估</a>')
        if (out/'index_before_none_recovery.html').exists():parts.append(' · <a href="index_before_none_recovery.html">恢复前报告存档</a>')
        parts.append('</p></section>')
    parts.append('<table><tr><th>Stage2</th><th>状态 / 更新</th><th>墙钟小时</th><th>训练步中位秒</th><th>峰值已分配 GiB</th></tr>')
    for m in MODES:
        rows=updates[m];st=status[m];sec=[x['step_seconds'] for x in rows];mem=[x['cuda_peak_allocated_bytes']/2**30 for x in rows]
        label='none（恢复，新增更新）' if m=='none' and recovering else m
        parts.append(f'<tr><td>{label}</td><td>{E(str(st.get("status","等待 Stage1 冻结")))} / {st.get("step",0)}</td><td>{fmt(st.get("elapsed_seconds",0)/3600)}</td><td>{fmt(statistics.median(sec) if sec else None)}</td><td>{fmt(max(mem) if mem else None)}</td></tr>')
    parts.append('</table><h2>拟合曲线与训练效率</h2><p>图例：橙色 joint；紫色 none。动作 loss 单独比较；joint total 含视频项，不能直接和 none total 比较。横轴分别为更新数和实际墙钟秒，时间包含评估停顿。训练噪声采用相同步级随机种子，但两种结构消耗 RNG 顺序不同。</p>')
    for key,label in [('action','Action loss'),('fk_position','FK position loss'),('fk_rotation','FK rotation loss')]:
        for axis in ['step','elapsed_seconds']:
            series={m:[(r[axis],r['losses'].get(key)) for r in updates[m] if r['losses'].get(key) is not None] for m in MODES}
            parts.append(_chart(series,label,axis))
    parts.append(stage1_section(run,out))
    snapshots={m:[(p,read(p)) for p in sorted(roots[m].glob('eval_*/metrics.json'))] for m in MODES}
    parts.append('<h2>Stage2 预测误差（有效监督）</h2>')
    for key in ['action_mae','joint_radian_mae','eef_delta_translation_m_mae','fk_position_m','fk_rotation_rad']:
        parts.append(_chart({m:[(v['step'],v.get('aggregate',{}).get(key,{}).get('value')) for p,v in snapshots[m]] for m in MODES},key,'Optimizer updates'))
    if all(snapshots.values()):
        parts.append('<table><tr><th>有效指标</th><th>joint 最新快照</th><th>none 最新快照</th></tr>')
        for key in ['action_mae','joint_radian_mae','eef_delta_translation_m_mae','fk_position_m','fk_rotation_rad']:
            parts.append('<tr><td>'+key+'</td>'+''.join('<td>'+fmt(snapshots[m][-1][1]['aggregate'].get(key,{}).get('value'))+'</td>' for m in MODES)+'</tr>')
        parts.append('</table><p>上表比较最新已保存预测，更新数可能不同；详细步数与耗时见上方。它衡量固定训练窗口的拟合，不代表未见数据泛化。</p>')
        med={m:statistics.median([x['step_seconds'] for x in updates[m]]) if updates[m] else None for m in MODES}
        if all(med.values()):parts.append(f'<p>纯训练步耗时比 joint/none = {med["joint"]/med["none"]:.3f}；>1 表示 none 每步更快。排除评估与保存耗时。</p>')
    samples=torch.load(samples_path,weights_only=False,map_location='cpu') if any(snapshots.values()) else []
    geoms=None;video_geoms=None
    if samples and geometry:
        from stage12_projection import prepare_geometry, trajectories, prepare_video_geometry, video_trajectories
        data_hash=hashlib.sha256()
        with samples_path.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024*1024),b''):data_hash.update(block)
        renderer_sha=hashlib.sha256(Path(__file__).read_bytes()+(CODE/'scripts/experiments/stage12_projection.py').read_bytes()+(CODE/'scripts/experiments/stage12_video.py').read_bytes()+data_hash.digest()).hexdigest()
        gp=out/'geometry.pt'
        cached=torch.load(gp,weights_only=False,map_location='cpu') if gp.exists() else None
        if isinstance(cached,dict) and cached.get('renderer_sha')==renderer_sha and cached.get('video_geometries') is not None:
            geoms=cached['geometries'];video_geoms=cached.get('video_geometries')
        else:
            geoms=prepare_geometry(samples);video_geoms=[prepare_video_geometry(s) for s in samples]
            torch.save(dict(renderer_sha=renderer_sha,geometries=geoms,video_geometries=video_geoms),gp)
        save_json(out/'geometry_coverage.json',geoms)
    selected={m:([snapshots[m][0]]+([snapshots[m][-1]] if len(snapshots[m])>1 else [])) if snapshots[m] else [] for m in MODES}
    # Only the earliest and latest immutable snapshots are copied/rendered here.
    analysis=[]
    for i,sample in enumerate(samples):
        parts.append(f'<details id="case-{i:03d}"><summary>case {i:03d} · {E(sample["_eval"]["family"])} · {E(sample["granularity"])}</summary><p>{E(sample["prompt"])}</p>')
        for m in MODES:
            for mp,v in selected[m]:
                source=mp.parent/f'case_{i:03d}'
                if not (source/'prediction.pt').exists():continue
                mode_tag='none_recovery' if m=='none' and recovering else m
                stem=f'{mode_tag}_{v["step"]:08d}_{i:03d}';dest=assets/stem;dest.mkdir(exist_ok=True)
                for name in ['current.png','ground_truth_future.png','action_curves.svg']:
                    if (source/name).exists() and not (dest/name).exists():shutil.copy2(source/name,dest/name)
                row=next(c for c in v['cases'] if c['name']==f'case_{i:03d}')
                parts.append(f'<section><h3>{mode_tag} · '+('恢复新增 ' if m=='none' and recovering else '')+f'step {v["step"]}</h3><p>'+ '；'.join(f'{E(k)}={fmt(x.get("value"))} (n={x.get("valid_count",0)})' for k,x in row['metrics'].items())+'</p>')
                parts.append(f'<div class="row"><figure><img loading="lazy" src="assets/{stem}/current.png"><figcaption>当前输入拼图</figcaption></figure><figure><img loading="lazy" src="assets/{stem}/ground_truth_future.png"><figcaption>GT horizon 末帧</figcaption></figure></div>')
                if (dest/'action_curves.svg').exists():parts.append(f'<details><summary>112 维 action 轨迹（红预测 / 青 GT，有效 mask）</summary><img loading="lazy" src="assets/{stem}/action_curves.svg"></details>')
                dynamic=None
                if geoms is not None:
                    gp=dest/'projection.json';proj=read(gp) if gp.exists() else {}
                    if proj.get('_renderer_sha')!=renderer_sha:
                        predicted=torch.load(source/'prediction.pt',weights_only=False,map_location='cpu')['normalized_actions'].numpy()
                        proj=trajectories(predicted,sample,geoms[i]);proj['_renderer_sha']=renderer_sha;save_json(gp,proj)
                        for kind in ['direct','fk']:
                            for filename in [kind+'.png','video_'+kind+'_tcp.mp4']:(dest/filename).unlink(missing_ok=True)
                        (dest/'video_projection.json').unlink(missing_ok=True)
                    dynamic=video_trajectories(proj,sample,video_geoms[i])
                    endpoint=sample.get('report_target',{}).get('endpoint',{})
                    dynamic['endpoint_definition']={k:endpoint[k] for k in ['definition_id','used_fields','transforms'] if k in endpoint}
                    save_json(dest/'video_projection.json',dynamic)
                    save_json(dest/'video_camera_geometry.json',video_geoms[i])
                shown_direct=False
                if dynamic is not None:
                    for kind in ['direct','fk']:
                        data=dynamic[kind]
                        if not np.asarray(data['metric_valid']).any():continue
                        vp=dest/f'video_{kind}_tcp.mp4'
                        if not vp.exists() or not (dest/f'video_{kind}_tcp_frame4.png').exists():
                            payload={**data,'source_frames':dynamic['source_frames'],'action_indices':dynamic['action_indices'],'frame_reasons':dynamic['frame_reasons']}
                            frames=[]
                            for j,gt in enumerate(sample['video']):
                                predicted_path=source/f'video_{j:02d}.png'
                                if m=='joint' and not predicted_path.exists():raise FileNotFoundError(predicted_path)
                                generated=Image.open(predicted_path) if m=='joint' else None
                                frames.append(render_tcp_video_frame(gt,generated,payload,j,f'{mode_tag} step{v["step"]} | {kind.upper()} TCP'))
                            video(vp,frames);frames[4].save(dest/f'video_{kind}_tcp_frame4.png')
                        shown_direct |= kind=='direct'
                        label='直接预测 TCP' if kind=='direct' else '关节预测经 FK 得到 TCP'
                        extra='左：GT 视频；右：joint 生成视频；两侧均叠加同一源相机坐标。' if m=='joint' else 'GT 视频背景上的 none 动作预测（none 不生成视频）。'
                        counts=data['counts'];count_text=f'未来采样GT {counts["eligible_gt"]}点，计分 {counts["scored"]}点，预测出画 {counts["outside_pred"]}点（保留误差、不截断）。'
                        parts.append(f'<figure style="width:640px"><video controls muted loop preload="none" poster="assets/{stem}/video_{kind}_tcp_frame4.png" src="assets/{stem}/video_{kind}_tcp.mp4"></video><figcaption><b>{label}逐帧投影：GT 绿方框 / 预测红圆圈，L/R 标侧。</b>{extra}每帧使用对应源相机 K/E/畸变；首帧仅观测 GT，后续对应未来第4、8…32步；4fps预览。{count_text}腕图未叠加未验证投影。</figcaption></figure>')
                    parts.append(f'<a href="assets/{stem}/video_projection.json">逐帧 TCP 坐标、动作索引、有效 mask 与出画状态</a> · <a href="assets/{stem}/video_camera_geometry.json">逐帧相机与解码来源</a>')
                if m=='joint':
                    vp=dest/'video.mp4'
                    if not vp.exists():
                        frames=[]
                        for j,gt in enumerate(sample['video']):
                            pred=source/f'video_{j:02d}.png'
                            if pred.exists():
                                im=Image.new('RGB',(640,384));im.paste(gt,(0,0));im.paste(Image.open(pred),(320,0));frames.append(im)
                        if frames:video(vp,frames)
                    if vp.exists():
                        if shown_direct:parts.append('<details><summary>不叠加投影的原始视频</summary>')
                        reason='' if shown_direct else '；直接TCP投影不可用：'+E(str((dynamic or {}).get('reason') or (proj.get('reason') if geoms is not None else '未启用几何')))
                        parts.append(f'<figure style="width:640px"><video controls muted loop preload="none" src="assets/{stem}/video.mp4"></video><figcaption>GT 左 / joint 预测右；9 个 horizon 采样帧，预览4fps（非源视频速度）{reason}</figcaption></figure>')
                        if shown_direct:parts.append('</details>')
                else:parts.append('<p>none 无未来视频输出。</p>')
                if geoms is not None:
                    gp=dest/'projection.json'
                    proj=read(gp) if gp.exists() else {}
                    if proj.get('_renderer_sha')!=renderer_sha:
                        pred=torch.load(source/'prediction.pt',weights_only=False,map_location='cpu')['normalized_actions'].numpy()
                        proj=trajectories(pred,sample,geoms[i]);proj['_renderer_sha']=renderer_sha;save_json(gp,proj)
                        for kind in ['direct','fk']:(dest/(kind+'.png')).unlink(missing_ok=True)
                    for kind in ['direct','fk']:
                        data=proj.get(kind)
                        if data is None or not np.asarray(data['metric_valid']).any():
                            reason=geoms[i]['fk_reason'] if kind=='fk' else proj.get('reason','没有有效监督')
                            parts.append(f'<p>{kind} 投影：N/A；{E(str(reason))}</p>');continue
                        image=dest/(kind+'.png')
                        if not image.exists():projection_panel(sample['video'][0],data,f'{m} step{v["step"]} | {kind}',image)
                        pm=metric(data['pixel_error'])
                        parts.append(f'<figure style="width:640px"><img loading="lazy" src="assets/{stem}/{kind}.png"><figcaption>{E(str(data.get("label",kind)))}；像素误差 {fmt(pm["value"])}，有效数 {pm["valid_count"]}；点数 {E(str(data.get("counts",{})))}</figcaption></figure>')
                    analysis.append(dict(mode=m,step=v['step'],case=i,projection=proj))
                    parts.append(f'<a href="assets/{stem}/projection.json">投影坐标、mask、出画和几何来源</a>')
                parts.append('</section>')
        parts.append('</details>')
    save_json(out/'projection_analysis.json',analysis)
    s1_origin=E(plan['initial_stage1_checkpoint']) if plan.get('initial_stage1_checkpoint') else '最新 masked SA run_v2'
    parts.append('<h2>运行配置与来源</h2><p>Stage1 从 '+s1_origin+' 继续训练，其 Wan/VAE 源头为官方 OpenWAM checkpoint_step_154000。Stage2 原始两组共同从此前已验证的 112 维动作 checkpoint_step_1000 初始化（curated_gripper_human_20261014 / gripper_human_aug_lr0.0001_b320 / 2026-10-15_21-16-09）。</p><p>Stage1 LR：Wan 2e-5 / history 2e-4；原始 Stage2 参数组 LR ×1.5，100 更新预热。'+('当前 none 恢复组改为自身 checkpoint_step_250 + fresh optimizer + warmup=0；见上方独立恢复记录。' if recovering else '')+'冻结资产与基础模型路径、SHA256 见运行记录。原流程状态以 supervisor 为准；恢复组状态单独展示。</p>')
    for name in ['plan.json','supervisor.json','comparison_config.yaml','stage1_identity.json','code_identity.json','native_cpu_preflight.json','paired_initialization_check.json','data_identity.json','completion.json']:
        if (run/name).exists():shutil.copy2(run/name,out/name);parts.append(f'<p><a href="{name}">{name}</a></p>')
    parts.append('<pre>'+E(json.dumps(supervisor,ensure_ascii=False,indent=2))+'</pre></html>')
    temporary=out/'index.pending.html';temporary.write_text(''.join(parts));temporary.replace(out/'index.html')
    temporary=out/'index_stage1_tcp_v2.pending.html';temporary.write_text(''.join(parts));temporary.replace(out/'index_stage1_tcp_v2.html')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--samples',type=Path,required=True)
    p.add_argument('--no-geometry',action='store_true');a=p.parse_args();render(a.run,a.samples,geometry=not a.no_geometry)
