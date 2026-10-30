"""Static experiment summary with verified counts, native metrics and video links."""
from __future__ import annotations
import argparse
import html
import json
from pathlib import Path
import numpy as np


def read(path,default=None):
    try:return json.loads(Path(path).read_text())
    except FileNotFoundError:return {} if default is None else default


def events(path):
    if not path.exists():return []
    rows=[]
    for line in path.read_text().splitlines():
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:pass  # producer can be between write/flush
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);a=p.parse_args()
    root=a.run.resolve();out=root/'report';out.mkdir(exist_ok=True)
    stages={name:events(root/path) for name,path in [('stage1','stage1/progress.jsonl'),('joint','stage2/joint/updates.jsonl'),('none','stage2/none/updates.jsonl')]}
    stages['stage1']=[r for r in stages['stage1'] if r.get('event')=='update']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,3.3))
    for ax,(name,rows) in zip(axes,stages.items()):
        keys=['fm_loss','tcp_loss'] if name=='stage1' else ['total']
        for key in keys:
            pairs=[(r['step'],r.get('losses',r).get(key)) for r in rows]
            pairs=[(x,y) for x,y in pairs if y is not None and np.isfinite(y)]
            if not pairs:continue
            x,y=map(np.asarray,zip(*pairs));window=min(25,len(y))
            ax.plot(x[window-1:],np.convolve(y,np.ones(window)/window,mode='valid'),label=key)
        ax.set_title(name);ax.set_xlabel('optimizer updates');ax.set_ylabel('loss (rolling mean <=25)');ax.grid(alpha=.25)
        if ax.lines:ax.legend()
    fig.tight_layout();fig.savefig(out/'losses.svg');plt.close(fig)
    state=read(root/'supervisor.json');proof=read(root/'completion.json')
    budgets=dict(stage1=state.get('stage1_updates',1000),joint=state.get('stage2_updates',500),none=state.get('stage2_updates',500))
    stage1_evaluations=read(root/'stage1/summary.json').get('evaluations',[])
    counts={k:max([r['step'] for r in v],default=0) for k,v in stages.items()}
    video=read(root/'fullspan/report/coverage.json');manifest=read(root/'fullspan/data/manifest.json')
    validation=read(root/'fullspan/report/validation.json');publication=read(root/'code_publication.json')
    metrics=[]
    for arm in ['joint','none']:
        for folder in sorted((root/'stage2'/arm).glob('eval_*')):
            value=read(folder/'metrics.json')
            for case in value.get('cases',[]):
                m=case['metrics'];metrics.append(dict(arm=arm,step=value['step'],family=case['family'],case=case['name'],metrics=m))
    result=dict(status=state.get('status','preparing'),completed_updates=counts,requested_updates=budgets,stage1_evaluations=stage1_evaluations,
        training_complete=proof.get('status')=='complete',completed_videos=video.get('completed_videos',0),expected_videos=6,
        verified_videos=validation.get('verified_videos',0),video_validation_status=validation.get('status','pending'),
        publication=publication,metrics=metrics,teacher=read(root/'stage1_identity.json'),selection=[{k:s[k] for k in ('family','episode','start','end_exclusive','scope','prompt')} for s in manifest.get('sequences',[])])
    (out/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    esc=lambda value:html.escape(str(value))
    complete=result['training_complete'] and result['completed_videos']==6 and result['verified_videos']==6 and result['video_validation_status']=='complete'
    status='训练及六段视频校验完成' if complete else (f'训练完成；视频生成 {result["completed_videos"]}/6，校验通过 {result["verified_videos"]}/6' if result['training_complete'] else state.get('status','preparing'))
    parts=['<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>Stage1 / Stage2 三条 episode 过拟合</title>',
        '<style>body{font:16px system-ui;max-width:1180px;margin:32px auto;padding:0 20px;line-height:1.6;color:#17263b}table{border-collapse:collapse;width:100%}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}img{width:100%}a{color:#145ac1}.cards{display:flex;gap:20px;flex-wrap:wrap}.card{padding:16px 28px;border:1px solid #ccd7e4;border-radius:12px}small{color:#567}</style>',
        '<h1>Stage1 / Stage2 三条 episode 过拟合</h1>',f'<p>状态：{esc(status)}</p>',
        '<div class="cards">'+''.join(f'<div class="card">{esc(k)}<br><strong>{counts[k]} / {n}</strong> 步</div>' for k,n in budgets.items())+'</div>']
    if not complete:parts.append('<meta http-equiv="refresh" content="60">')
    if publication.get('url'):parts.append(f'<p><a href="{esc(publication["url"])}">远程代码分支</a> · commit <code>{esc(publication.get("remote_commit",""))[:12]}</code> · <a href="../code_publication.json">代码树一致性记录</a></p>')
    if result['training_complete']:
        regressions=[]
        for arm in ['joint','none']:
            values={(m['step'],m['family']):(m['metrics'].get('eef_delta_translation_m_mae') or {}).get('value') for m in metrics if m['arm']==arm}
            worse=[family for step,family in values if step==budgets[arm] and values[(step,family)] is not None and values.get((0,family)) is not None and values[(step,family)]>values[(0,family)]]
            if worse:regressions.append(f'{arm} 的 '+ '、'.join(worse)+' 在最终固定锚点评估中比训练前误差更高')
        if regressions:parts.append('<p><strong>结果提示：</strong>'+esc('；'.join(regressions))+'。实验完成不等于过拟合成功；请结合下方真实生成误差和完整轨迹视频判断。</p>')
    if (root/'fullspan/report/index.html').exists():parts.append('<h2><a href="../fullspan/report/index.html">播放 GT / 预测对照视频（joint 和 none）</a></h2>')
    parts+=['<p>每隔 10 个源帧重新推理，并补最后一帧。上排 GT、下排预测；前三列是同一当前时刻的头部和双腕视图，第四列为 Stage1 目标关键帧与二维 TCP。</p>',
        '<table><tr><th>数据集</th><th>Episode</th><th>展示源帧范围（含末帧）</th><th>完整指令</th></tr>']
    for s in result['selection']:parts.append(f'<tr><td>{esc(s["family"])}</td><td>{s["episode"]}</td><td>{s["start"]}–{s["end_exclusive"]-1}</td><td>{esc(s["prompt"])}</td></tr>')
    parts+=['</table><p>Stage1 使用这三条 episode 的 1 FPS task/subtask 样本；Stage2 使用其中 60 个固定窗口并按来源均衡抽样。中间评估是三个训练锚点，不能代表未见数据泛化。</p>',
        '<p>RoboChallenge v2 的双腕外参仍为人工候选：保留原图，但不画未经验证的投影。三个样例均未配置匹配的 FK 监督；本报告的直接 TCP 投影不等于 URDF/FK 验证。缺头图会屏蔽对应的头图与 TCP 监督。Interndata 的终止帧双手均已出画，Stage1 二维 TCP 不计分；其有效中途轨迹仍参与 Stage2 监督与评估。</p>',
        '<h2>训练曲线</h2><img src="losses.svg" alt="训练损失曲线">',
        '<h2>Stage2 原生动作评估</h2><p>EEF 相对平移 MAE 来自真实反归一化坐标，按有效通道计分。生成使用固定种子与 20 步去噪；相同训练种子不保证两种配置的 action noise 完全相同。</p>',
        '<table><tr><th>配置</th><th>更新步</th><th>数据集</th><th>EEF 平移 MAE（m）</th><th>有效元素</th><th>计分状态</th></tr>']
    for m in metrics:
        metric=m['metrics'].get('eef_delta_translation_m_mae') or {}
        value=metric.get('value');value='—' if value is None else f'{value:.5f}'
        parts.append('<tr>'+''.join(f'<td>{esc(x)}</td>' for x in [m['arm'],m['step'],m['family'],value,metric.get('valid_count',0),metric.get('status','unavailable')])+'</tr>')
    parts.append('</table><h2>Stage1 关键帧生成评估</h2><p>固定训练诊断锚点；二维 TCP 只统计图内有效 GT。图像误差按 0–255 像素值计算，不能替代轨迹误差。</p><table><tr><th>更新步</th><th>来源 / 指令范围</th><th>图像 MAE</th><th>TCP 误差（px）</th><th>有效 TCP 点</th></tr>')
    for evaluation in stage1_evaluations:
        for group,value in evaluation.get('group_metrics',{}).items():
            cells=[evaluation['step'],group,f"{value['image_mae_uint8']:.3f}" if value['image_mae_uint8'] is not None else '—',f"{value['tcp_pixel_error']:.3f}" if value['tcp_pixel_error'] is not None else '不计分',value['valid_points']]
            parts.append('<tr>'+''.join(f'<td>{esc(x)}</td>' for x in cells)+'</tr>')
    parts.append('</table>')
    for path,label in [('stage1/index.html','Stage1 关键帧与 TCP 训练诊断'),('fullspan/report/projection_coverage.html','逐相机实际投影覆盖'),('fullspan/report/validation.json','完整视频校验'),('source.yaml','三条 episode 数据选择'),('stage1_config.yaml','Stage1 配置'),('stage2_config.yaml','Stage2 配置'),('completion.json','训练完成记录')]:
        if (root/path).exists():parts.append(f'<p><a href="../{path}">{label}</a></p>')
    parts.append('<p><a href="summary.json">机器可读评估与 teacher 身份</a></p></html>')
    temp=out/'index.pending.html';temp.write_text('\n'.join(parts));temp.replace(out/'index.html')
    print(json.dumps(dict(counts=counts,videos=result['completed_videos'],status=result['status'])))


if __name__=='__main__':main()
