"""Append-only single-subtask ablation results, with directly comparable units."""
from pathlib import Path
import html,json,shutil

NAMES={'legacy_text':'A · 原 Wan 文本映射','normalized_cross_attention':'B · 独立 LN + Linear / cross-attention',
       'prefix_self_attention':'C · Wan + PatchWAM 式 prefix'}

def _chart(series,title,key):
    colors=['#d17c26','#157c80','#8557a5'];w,h=900,210
    values=[float(p[key]) for s in series for p in s if p.get(key) is not None]
    if not values:return ''
    hi=max(values) or 1; xmax=max(p['step'] for s in series for p in s) or 1
    lines=[]
    for points,color in zip(series,colors):
        xy=' '.join(f'{50+810*p["step"]/xmax:.1f},{180-155*p[key]/hi:.1f}' for p in points if p.get(key) is not None)
        lines.append(f'<polyline points="{xy}" stroke="{color}" stroke-width="2" fill="none"/>')
    return f'<h3>{title}</h3><svg viewBox="0 0 {w} {h}" role="img"><path d="M50 20V180H870" fill="none" stroke="#aaa"/><text x="2" y="25">{hi:.2f}</text><text x="20" y="185">0</text><text x="820" y="202">{xmax} 步</text>'+''.join(lines)+'</svg>'

def render(output):
    output=Path(output);report=output/'report';report.mkdir(exist_ok=True)
    record=json.loads((output/'record.json').read_text());audit=json.loads((output/'input_audit.json').read_text())
    results={}
    for mode in NAMES:
        p=output/mode/'summary.json'
        results[mode]=json.loads(p.read_text()) if p.exists() else {'status':'pending','evaluations':[]}
    # A small movable report package; checkpoints and optimizer data stay outside it.
    for name in ('protocol.json','input_audit.json','record.json'):
        if (output/name).exists():shutil.copy2(output/name,report/name)
    for mode,summary in results.items():
        dirs=[f'eval_{v["step"]:06d}' for v in summary['evaluations']]
        dirs += [f'final_seed_{v["seed"]}' for v in summary.get('extra_eval_seeds',[])]
        for directory in dirs:
            source=output/mode/directory;destination=report/'assets'/mode/directory
            destination.mkdir(parents=True,exist_ok=True)
            for filename in ('0000_00_generated.png','gt_vs_pred.png','metrics.json'):
                a,b=source/filename,destination/filename
                if a.exists() and (not b.exists() or a.stat().st_mtime_ns!=b.stat().st_mtime_ns):shutil.copy2(a,b)
    rows=[]
    for mode,name in NAMES.items():
        s=results[mode];v=s['evaluations'][-1] if s['evaluations'] else {}
        rows.append('<tr>'+''.join(f'<td>{html.escape(str(x))}</td>' for x in [name,s.get('status'),s.get('completed_steps',0),
            round(v.get('image_mae_uint8',0),3) if v else '—',round(v.get('tcp_mean_px',0),3) if v else '—',
            round(s.get('median_train_seconds',0),3) if s.get('median_train_seconds') else '—',
            round(s.get('peak_train_allocated_gib',0),2) if s.get('peak_train_allocated_gib') else '—'])+'</tr>')
    series=[results[m]['evaluations'] for m in NAMES]
    charts=_chart(series,'图像平均像素绝对误差（0–255，越低越好）','image_mae_uint8')+_chart(series,'TCP 平均欧氏像素误差（左右有效点平均）','tcp_mean_px')+_chart(series,'固定 5 组时间步与噪声的联合 flow loss','fixed_flow_loss')
    steps=sorted({x['step'] for a in series for x in a});cards=''
    for step in steps:
        cards+=f'<h3>第 {step} 步</h3><div class="grid">'
        for mode,name in NAMES.items():
            v=next((v for v in results[mode]['evaluations'] if v['step']==step),None)
            if v:
                rel=f'assets/{mode}/eval_{step:06d}'
                cards+=f'<article><strong>{name}</strong><img src="{rel}/0000_00_generated.png"><p>图像 MAE {v["image_mae_uint8"]:.3f} / 255<br>TCP 平均 {v["tcp_mean_px"]:.3f} px · 左 {v["tcp_left_px"]:.3f} / 右 {v["tcp_right_px"]:.3f} px</p><details><summary>在真实目标图像上对照预测点</summary><img src="{rel}/gt_vs_pred.png"></details><a href="{rel}/metrics.json">原始指标</a></article>'
            else:cards+=f'<article>{name}<p>尚未完成此步评估</p></article>'
        cards+='</div>'
    robustness = '<h2>末步多噪声种子复测</h2><p>除主时间线的 seed100042，再独立测试两个采样种子。以下为三次生成的算术均值，不是训练重复实验。</p><table><tr><th>组别</th><th>图像 MAE 均值 /255</th><th>TCP 误差均值 px</th></tr>'
    for mode,name in NAMES.items():
        summary=results[mode];extra=summary.get('extra_eval_seeds',[])
        if extra and summary['evaluations']:
            values=[summary['evaluations'][-1]]+extra
            robustness+=f'<tr><td>{name}</td><td>{sum(v["image_mae_uint8"] for v in values)/len(values):.3f}</td><td>{sum(v["tcp_mean_px"] for v in values)/len(values):.3f}</td></tr>'
    robustness+='</table>'
    for seed in (100043,100044):
        robustness+=f'<h3>末步 · seed {seed}</h3><div class="grid">'
        for mode,name in NAMES.items():
            value=next((v for v in results[mode].get('extra_eval_seeds',[]) if v['seed']==seed),None)
            if value:
                robustness+=f'<article>{name}<img src="assets/{mode}/final_seed_{seed}/0000_00_generated.png"><p>图像 MAE {value["image_mae_uint8"]:.3f} /255 · TCP {value["tcp_mean_px"]:.3f} px</p></article>'
            else:robustness+=f'<article>{name} · 待完成</article>'
        robustness+='</div>'
    baseline_file=output/'image_baselines.json'
    baseline_note=''
    if baseline_file.exists():
        baselines=json.loads(baseline_file.read_text())
        baseline_note=f'<p>参考：直接复制当前头部图像的 MAE={baselines["copy_current_head_mae_uint8"]:.3f}/255；冻结 VAE 对目标图像的重构 MAE={baselines["vae_reconstruction_mae_uint8"]:.3f}/255。后者不是严格误差下界。</p>'
    diagnostics=''.join('<h3>'+NAMES[m]+' · 条件干预诊断</h3><pre>'+html.escape(json.dumps(results[m].get('sensitivity',{}),ensure_ascii=False,indent=2))+'</pre>' for m in NAMES)
    document='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Stage1 条件融合 · 单 subtask 对照</title><style>body{font:16px/1.6 system-ui;margin:32px;background:#f4f7f6;color:#193b35}main{max-width:1320px;margin:auto}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}article{padding:16px;background:white;border:1px solid #cfddd8;border-radius:10px}img{max-width:100%;display:block;margin:12px 0}table{border-collapse:collapse;width:100%;background:white}td,th{border:1px solid #ccd9d4;padding:8px;text-align:left}svg{width:100%;max-height:240px;background:white}pre{white-space:pre-wrap;font-size:12px}.note{background:#e7efe8;padding:16px}a{color:#146d73}</style><main>'''
    document+=f'<h1>Stage1 条件融合 · 单 subtask 过拟合</h1><p><strong>EgoDex · {html.escape(record["instruction"])}</strong> · episode {record["source_episode"]} · 当前帧 {record["anchor_source_frame"]} → subtask 末帧 {record["goal_source_frame"]}</p>'
    document+='<p class="note">三组均基于 02edd62（已移除 readout），使用同一个 Wan 初始化、同一 query encoder 初始化、训练样本与噪声序列。C 为 Wan 内的 PatchWAM 式融合，不是完整 FLUX PatchWAM。目标为子任务区间末帧，未声称成功标注。单样本过拟合不证明语言理解或泛化。图像/TCP 都来自联合去噪；本页保留所有评估步，不覆盖。</p>'
    document+=f'<p>时间差 {audit["goal_gap_seconds"]:.2f} 秒；历史有效 {sum(audit["history"]["valid"])}/10，真实 age 编码，训练逐帧 dropout=0.2；缺失腕相机已 mask。<a href="protocol.json">完整实验配置</a> · <a href="input_audit.json">输入来源核查</a></p>'
    document+='<div class="grid"><article>送给模型的当前图像（三视角布局）<img src="inputs/case_0000_observed.png"></article><article>送给模型的目标图像 + TCP GT（方框）<img src="inputs/case_0000_target_points.png"></article><article>VAE 重构参考（不是模型预测）<img src="inputs/vae_reconstruction.png"><p>用于区分 VAE 重构误差与去噪器误差。颜色：左红、右蓝；预测为圆圈。</p></article></div>'
    document+='<table><tr><th>组别</th><th>状态</th><th>更新步数</th><th>图像 MAE /255</th><th>TCP px</th><th>训练秒/步</th><th>训练峰值显存 GiB</th></tr>'+''.join(rows)+'</table><p>A 橙色 · B 青色 · C 紫色。速度统计排除前10步及评估/保存时间；同一 DSW 卡顺序运行。B 期间观察到其他 GPU 进程，速度受到共享负载干扰，不能作为独占吞吐对比。冻结 VAE 特征缓存后计时，不包含首次编码与 T5 预计算。</p>'+baseline_note+charts+cards+robustness+diagnostics+'</main></html>'
    tmp=report/'index.pending.html';tmp.write_text(document);tmp.replace(report/'index.html')
