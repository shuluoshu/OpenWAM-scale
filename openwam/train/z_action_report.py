"""Append-only experiment artifacts, atomically refreshed Chinese comparison page."""
from pathlib import Path
import html
import json
import math
import os

COLORS={'joint':'#ee8b28','none':'#8754c8'}
LABELS={'joint':'联合视频 + 动作','none':'当前观测 + Z，仅动作'}


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False,default=str)+'\n')
    os.replace(temporary,path)


def _read(path, default=None):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return default


def _number(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def _fmt(value):return f'{value:.5g}' if _number(value) else 'N/A'


def _metric(snapshot):
    """Native _aggregate is metric -> {sum, valid_count, value}; zero is valid."""
    metric=snapshot.get('aggregate',{}).get('action_mae',{})
    value=metric.get('value')
    return value if _number(value) and metric.get('valid_count',0)>0 else None


def _updates(path):
    rows=[]
    try:
        for line in Path(path).read_text().splitlines():
            try:row=json.loads(line)
            except ValueError:continue
            if isinstance(row,dict):rows.append(row)
    except OSError:pass
    return rows


def _chart(series,title,xlabel):
    points=[(x,y) for values in series.values() for x,y in values if _number(x) and _number(y)]
    if not points:return '<p class="muted">'+html.escape(title)+'：等待有效数据。</p>'
    xmax=max(max(x for x,y in points),1);ymin=min(y for x,y in points);ymax=max(y for x,y in points)
    span=max(ymax-ymin,1e-8);lo=min(0,ymin);span=max(ymax-lo,span)
    svg=[f'<svg viewBox="0 0 900 300" role="img" aria-label="{html.escape(title)}"><rect width="900" height="300" fill="white"/><text x="65" y="24">{html.escape(title)}</text><path d="M65 42V250H870" fill="none" stroke="#a8b0bd"/>']
    for j in range(5):
        y=250-j*48;v=lo+span*j/4
        svg.append(f'<path d="M65 {y}H870" stroke="#e8edf3"/><text x="4" y="{y+4}" font-size="12">{v:.3g}</text>')
    for mode,values in series.items():
        clean=[(x,y) for x,y in values if _number(x) and _number(y)]
        coords=' '.join(f'{65+805*x/xmax:.2f},{250-192*(y-lo)/span:.2f}' for x,y in clean)
        svg.append(f'<polyline points="{coords}" fill="none" stroke="{COLORS[mode]}" stroke-width="2.5"/>')
        for x,y in clean:svg.append(f'<circle cx="{65+805*x/xmax:.2f}" cy="{250-192*(y-lo)/span:.2f}" r="2" fill="{COLORS[mode]}"/>')
    svg.append(f'<text x="65" y="275" font-size="12">0</text><text x="830" y="275" font-size="12">{xmax:.4g}</text><text x="410" y="295" font-size="13">{html.escape(xlabel)}</text></svg>')
    return ''.join(svg)


def render(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    snapshots={mode:[] for mode in COLORS};issues=[];statuses={};updates={}
    for mode in COLORS:
        statuses[mode]=_read(root/mode/'status.json',{}) or {}
        updates[mode]=_updates(root/mode/'updates.jsonl')
        for path in sorted((root/mode).glob('eval_*/metrics.json')):
            item=_read(path)
            if not isinstance(item,dict) or not _number(item.get('step')) or not _number(item.get('elapsed_seconds')) or not isinstance(item.get('cases'),list):
                issues.append(str(Path(os.path.relpath(path,root))));continue
            item=dict(item,_path=path);snapshots[mode].append(item)
    supervisor=_read(root/'supervisor.json',{}) or {}
    parts=['<h1>全尺寸 Stage2 对照实验</h1><p>同一冻结 Stage1 query 编码器、固定真实窗口、归一化统计、优化器配置与采样种子；比较联合未来视频 + 动作与当前观测 + Z 仅动作。当前结果仅反映固定训练样本的过拟合能力，不能推断泛化或真机效果。</p>','<div class="legend"><span style="color:#ee8b28">● 联合视频 + 动作（joint）</span>　<span style="color:#8754c8">● 当前观测 + Z，仅动作（none）</span></div>']
    for asset,label in [('input_audit/index.html','固定样例输入 / GT 审计'),('../input_audit/index.html','固定样例输入 / GT 审计'),('operation.md','操作手册'),('../operation.md','操作手册')]:
        if (root/asset).is_file():parts.append(f'<p><a href="{asset}">{label}</a></p>')
    child_runs=[p for p in root.iterdir() if p.is_dir() and ((p/'supervisor.json').exists() or (p/'joint/status.json').exists() or (p/'none/status.json').exists())]
    if child_runs:
        parts.append('<h2>运行入口</h2><ul>'+''.join(f'<li><a href="{html.escape(p.name)}/index.html">{html.escape(p.name)} 实时状态 / 对照报告</a></li>' for p in sorted(child_runs))+'</ul>')
    parts.append('<div class="cards">')
    for mode in COLORS:
        status=statuses[mode];u=updates[mode];times=[v.get('step_seconds') for v in u if _number(v.get('step_seconds'))];memory=[v.get('cuda_peak_allocated_bytes') for v in u if _number(v.get('cuda_peak_allocated_bytes'))]
        if _number(status.get('cuda_peak_allocated_bytes')):memory.append(status['cuda_peak_allocated_bytes'])
        parts.append(f'<section style="border-top:4px solid {COLORS[mode]}"><h2>{LABELS[mode]}</h2><p>状态：{html.escape(str(status.get("status","等待启动")))}</p><p>更新：{html.escape(str(status.get("step",0)))} · 已用 {_fmt(status.get("elapsed_seconds"))} 秒</p><p>平均每步 {_fmt(sum(times)/len(times) if times else None)} 秒 · 峰值已分配显存 {_fmt(max(memory)/2**30 if memory else None)} GiB</p></section>')
    parts.append('</div><h2>动作损失与资源</h2><p>主图使用 action loss；joint 的 total 还包含视频损失，不能直接与仅动作 total 比较。逐步随机种子一致；joint 先采样视频噪声，动作噪声抽样不保证逐元素相同。</p>')
    parts.append(_chart({mode:[(v.get('step'),v.get('losses',{}).get('action')) for v in updates[mode]] for mode in COLORS},'训练动作损失（action loss）','优化器更新数'))
    parts.append(_chart({mode:[(v.get('elapsed_seconds'),v.get('step_seconds')) for v in updates[mode]] for mode in COLORS},'每步耗时（秒）','训练 / 评估累计秒数'))
    parts.append('<h2>归一化动作 MAE 对比</h2><p>总体曲线按有效动作元素数加权；不同样本的有效维度数不同，不能将总体值视为每家族等权平均。判断架构差异优先查看同一案例、相同更新数的对照。</p>')
    parts.append(_chart({mode:[(v['step'],_metric(v)) for v in snapshots[mode]] for mode in COLORS},'有效动作维度 MAE，越低越好','优化器更新数'))
    comparison=[]
    for a in snapshots['joint']:
        b=next((b for b in snapshots['none'] if b['step']==a['step']),None)
        if b is not None:comparison.append(('相同更新数',a,b))
    if snapshots['joint'] and snapshots['none']:
        limit=min(snapshots['joint'][-1]['elapsed_seconds'],snapshots['none'][-1]['elapsed_seconds'])
        a=min(snapshots['joint'],key=lambda v:abs(v['elapsed_seconds']-limit));b=min(snapshots['none'],key=lambda v:abs(v['elapsed_seconds']-limit))
        comparison.append(('最接近共同时间预算的可用快照（非精确等时）',a,b))
    parts.append('<table><tr><th>比较口径</th><th>joint 更新 / 秒 / MAE</th><th>none 更新 / 秒 / MAE</th></tr>')
    for label,a,b in comparison:parts.append('<tr><td>'+label+'</td>'+''.join(f'<td>{v["step"]} / {_fmt(v["elapsed_seconds"])} / {_fmt(_metric(v))}</td>' for v in (a,b))+'</tr>')
    parts.append('</table>')
    parts.append('<details open><summary>同一案例 / 相同更新数：主要可比结果</summary><table><tr><th>更新</th><th>家族 / 案例</th><th>joint MAE</th><th>none MAE</th><th>none − joint</th><th>有效元素（joint / none）</th></tr>')
    for label,a,b in comparison:
        if label!='相同更新数':continue
        right={c.get('name'):c for c in b['cases']}
        for left in a['cases']:
            other=right.get(left.get('name'))
            if other is None:continue
            lm=left.get('metrics',{}).get('action_mae',{});rm=other.get('metrics',{}).get('action_mae',{})
            lv=lm.get('value');rv=rm.get('value');delta=rv-lv if _number(lv) and _number(rv) else None
            parts.append(f'<tr><td>{a["step"]}</td><td>{html.escape(left.get("family","")+" / "+left.get("name",""))}</td><td>{_fmt(lv)}</td><td>{_fmt(rv)}</td><td>{_fmt(delta)}</td><td>{lm.get("valid_count",0)} / {rm.get("valid_count",0)}</td></tr>')
    parts.append('</table></details>')
    if all(statuses[m].get('status')=='complete' for m in COLORS):
        parts.append('<h2>实测结论</h2>')
        valid=[(label,a,b) for label,a,b in comparison if _metric(a) is not None and _metric(b) is not None and a['step']>0 and b['step']>0]
        for label,a,b in valid[-2:]:
            delta=_metric(b)-_metric(a)
            parts.append(f'<p>{label}：仅动作 MAE {_fmt(_metric(b))}，联合基线 {_fmt(_metric(a))}；差值（none − joint）{_fmt(delta)}。'+('仅动作误差较低。' if delta<0 else '联合基线误差较低。' if delta>0 else '两者误差相同。')+'</p>')
        if not valid:parts.append('<p>两臂已结束，但缺少可比的有效非零更新快照，不能形成误差优劣结论。</p>')
    else:parts.append('<p class="notice">实验尚未全部完成；当前只展示已测数据，不生成最终优劣结论。</p>')
    parts.append('<h2>数据覆盖与来源</h2>')
    manifests=list(root.glob('*/checkpoints/fixed_cases_manifest.json'))
    if not manifests:
        manifests=list(root.glob('../data_v4/manifest.json')) or list(root.glob('data_v4/manifest.json'))
    manifest=_read(manifests[0],{}) if manifests else {}
    if manifest:
        cases=manifest.get('cases',[]);families={c.get('family') for c in cases};platforms={c.get('robot_type') for c in cases if c.get('robot_type')}
        parts.append(f'<p>准备清单：{len(families)} 个数据家族，{len(platforms)} 个机器人平台，{len(cases)} 个固定窗口。动作 horizon 32；9 帧视频，stride 4。有效监督以每例 mask 为准。</p>')
        parts.append('<p class="notice">监督范围限制：多个 human 数据家族在当前原生 Stage2 映射中仅有夹爪监督，缺少可信 TCP 动作映射；不能据此声称所有家族都进行了完整位姿学习。Genrobot 的已审计 TCP 映射已修复。下表逐例列出实际有效维度与屏蔽原因；物理指标只统计有效且可反归一化的目标。</p>')
        parts.append('<details><summary>逐例实际监督范围（清单 / mask）</summary><table><tr><th>案例 / 家族 / 平台</th><th>粒度</th><th>有效动作元素</th><th>有效动作维度（112维空间）</th><th>映射限制 / 屏蔽原因</th></tr>')
        for c in cases:
            name='case_'+str(c.get('index','?')).zfill(3)
            parts.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in (name+' / '+c.get('family','')+' / '+str(c.get('robot_type','')),c.get('granularity',''),c.get('valid_action_elements','N/A'),c.get('valid_action_dims',[]),c.get('mask_reasons',[])))+'</tr>')
        parts.append('</table></details>')
    else:parts.append('<p>等待运行保存数据覆盖清单；目标为 12 家族、15 平台、21 窗口，实际数量以清单为准。</p>')
    parts.append('<p>冻结 Stage1 使用广覆盖训练 20,000 次更新的 query checkpoint；精确资产路径与 SHA256 请见下方配置和来源文件。</p>')
    parts.append('<details><summary>配置、SHA256 与来源清单</summary>')
    for path in [root/'supervisor.json',root/'stage1_asset.json']+sorted(root.glob('comparison_config*.yaml'))+sorted(root.glob('*/checkpoints/config.yaml'))+sorted(root.glob('*/checkpoints/*provenance.json'))+manifests:
        if path.exists():parts.append(f'<p><a href="{Path(os.path.relpath(path,root))}">{html.escape(str(Path(os.path.relpath(path,root))))}</a></p>')
    parts.append('<pre>'+html.escape(json.dumps(supervisor,ensure_ascii=False,indent=2))+'</pre></details>')
    parts.append('<h2>逐数据家族评估轨迹</h2><p>每个点来自不可变评估快照；同家族多个样本按有效元素数加权。点击快照展开真实图像与动作曲线。</p>')
    family_names=sorted({case.get('family','unknown') for values in snapshots.values() for v in values for case in v['cases']})
    for family in family_names:
        series={mode:[] for mode in COLORS}
        for mode in COLORS:
            for v in snapshots[mode]:
                metrics=[c.get('metrics',{}).get('action_mae',{}) for c in v['cases'] if c.get('family')==family]
                count=sum(m.get('valid_count',0) for m in metrics);total=sum(m.get('sum',m.get('value',0)*m.get('valid_count',0) if _number(m.get('value')) else 0) for m in metrics)
                if count:series[mode].append((v['step'],total/count))
        parts.append('<details><summary>'+html.escape(family)+'</summary>'+_chart(series,family+'：有效动作 MAE','优化器更新数')+'</details>')
    parts.append('<h2>全部历史快照 / 真实预测</h2>')
    for mode in COLORS:
        for v in snapshots[mode]:
            rel=v['_path'].parent.relative_to(root)
            parts.append(f'<details><summary>{LABELS[mode]} — update {v["step"]}, {v["elapsed_seconds"]:.1f}s，动作 MAE {_fmt(_metric(v))}</summary>')
            for case in v['cases']:
                folder=root/rel/case['name'];base=rel/case['name']
                parts.append('<details><summary>'+html.escape(case['name']+' '+case.get('family','')+' / '+str(case.get('granularity','')))+'</summary><div class="images">')
                for filename,label in [('current.png','当前观测'),('ground_truth_future.png','未来真实标签')]:
                    if (folder/filename).exists():parts.append(f'<figure><img loading="lazy" src="{base/filename}"><figcaption>{label}</figcaption></figure>')
                if mode=='none':parts.append('<p>Stage2 预测视频：不适用 / not applicable (action-only)。</p>')
                else:
                    video=sorted(folder.glob('video_*.png'))
                    if video:
                        for image in video:parts.append(f'<figure><img loading="lazy" src="{base/image.name}"><figcaption>联合基线预测 {html.escape(image.stem)}</figcaption></figure>')
                    else:parts.append('<p>联合基线视频图像尚未发布。</p>')
                parts.append('</div><table><tr><th>指标</th><th>数值</th><th>有效元素</th></tr>')
                for name,m in case.get('metrics',{}).items():parts.append(f'<tr><td>{html.escape(name)}</td><td>{_fmt(m.get("value"))}</td><td>{m.get("valid_count",0)}</td></tr>')
                parts.append(f'</table><p><a href="{base}/prediction.pt">归一化预测、GT 与有效 mask</a></p><p>动作曲线：GT 青色，预测红色；无效 GT 不连线。</p><img loading="lazy" style="width:100%" src="{base}/action_curves.svg"></details>')
            parts.append('</details>')
    if issues:parts.append('<details><summary>未完成 / 无效快照（未纳入比较）</summary><pre>'+html.escape('\n'.join(issues))+'</pre></details>')
    body='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Stage2 仅动作对照实验</title><style>body{font:15px system-ui;color:#263248;background:#f4f6fa;max-width:1400px;margin:30px auto;padding:0 24px}h1{font-size:32px}h2{margin-top:28px}p{line-height:1.7}section,details,svg,table,.notice{background:white;border-radius:8px}section,details{padding:14px;margin:12px 0}summary{cursor:pointer;font-weight:600}.cards{display:grid;grid-template-columns:1fr 1fr;gap:18px}svg{width:100%;max-height:380px}table{width:100%;border-collapse:collapse;margin:12px 0}td,th{padding:8px;border-bottom:1px solid #e2e7ef;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere}.muted{color:#718096}.notice{padding:16px;border-left:4px solid #789}.images{display:flex;gap:10px;overflow:auto}figure{margin:0}figure img{width:220px}figcaption{font-size:12px}.legend{padding:12px}a{color:#3a65ac}@media(max-width:700px){.cards{grid-template-columns:1fr}}</style>'''+''.join(parts)+'</html>'
    tmp=root/'index.html.tmp';tmp.write_text(body);os.replace(tmp,root/'index.html')
