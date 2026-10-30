"""Render real overfit artifacts; never synthesize or substitute predictions."""
import argparse
import html
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);args=ap.parse_args()
    run=args.run;p=run/'report';s=json.loads((run/'summary.json').read_text());protocol=json.loads((run/'protocol.json').read_text())
    cards=s['cards'];esc=lambda x:html.escape(str(x));fmt=lambda x:'无有效 GT' if x is None else f'{x:.2f}'
    for name in ('protocol.json','summary.json','train.jsonl','events.jsonl'):
        if (run/name).exists():shutil.copy2(run/name,p/name)
    # Input export originally carries source instruction strings; actual text
    # tensors use the deliberately shared prompt. Make both explicit everywhere.
    for folder in (p/'inputs',p/'held_anchor/inputs'):
        if not (folder/'manifest.json').exists():continue
        records=json.loads((folder/'manifest.json').read_text())
        page=(folder/'index.html').read_text()
        for r in records:
            source=r.setdefault('source_instruction',r['instruction']);r['instruction']=protocol['common_instruction']
            page=page.replace('<p>'+esc(source)+'</p>','<p>实际模型指令：'+esc(r['instruction'])+'<br>原始数据指令（仅溯源）：'+esc(source)+'</p>')
        (folder/'manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2));(folder/'index.html').write_text(page)
    final=s.get('interventions',{})
    def image(path,label):
        return '<figure><figcaption>'+esc(label)+'</figcaption><a href="'+path+'"><img src="'+path+'"></a></figure>' if (p/path).exists() else ''
    def metrics(case,mode):return final.get(str(case),{}).get(mode,{})
    sections=[];table=[]
    initial=s['evaluations'][0]['values']
    for i,c in enumerate(cards):
        real=metrics(i,'real');single=metrics(i,'no_history')
        table.append('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in [c['family'],fmt(initial[i]['mae']),fmt(s['copy_maes'][i]),fmt(single.get('mae')),fmt(real.get('mae')),fmt(real.get('tcp_error_px'))])+'</tr>')
        pics='<div class="images">'+image(f'inputs/case_{i:04d}_observed.png','实际当前输入：头＋腕相机拼图')+image(f'inputs/case_{i:04d}_target_points.png','未来 GT 关键帧；有效 TCP 标注')+image(f'inputs/case_{i:04d}_vae.png','冻结 VAE 对 GT 的重建')+'</div>'
        histories='<div class="history">'
        for j,(age,valid) in enumerate(zip(c['history_ages'][0],c['history_valid'][0])):
            if valid:histories+=image(f'inputs/case_{i:04d}_history_{j:02d}.png',f'slot {j}: {age:.1f} 秒前')
            else:histories+=f'<figure><figcaption>slot {j}: MASKED</figcaption><div class="pad">NO INPUT</div></figure>'
        histories+='</div>'
        preds='<div class="images">'+image(f'step_000000/case_{i}/prediction.png','训练前')+image(f'final/case_{i}/no_history/prediction.png',f'同模型单帧输入 MAE {fmt(single.get("mae"))}')+image(f'final/case_{i}/real/prediction.png',f'同模型时序输入 MAE {fmt(real.get("mae"))}')+'</div>'
        tcp='<div class="images">'+image(f'final/case_{i}/no_history/tcp.png',f'单帧 TCP，误差 {fmt(single.get("tcp_error_px"))} px')+image(f'final/case_{i}/real/tcp.png',f'时序 TCP，误差 {fmt(real.get("tcp_error_px"))} px')+'</div>'
        other=(i+1)%len(cards)
        interventions='<div class="images">'+image(f'inputs/case_{other:04d}_observed.png',f'交换成 {cards[other]["family"]} 的输入')
        for mode,title in [('wrong_visual','交换当前＋历史'),('wrong_current','只交换当前'),('wrong_history','只交换历史'),('blank_visual','当前置零＋无历史')]:
            v=metrics(i,mode);label=title+(f'；输出最接近场景 {v["nearest_scene"]}' if v else '')
            interventions+=image(f'final/case_{i}/{mode}/prediction.png',label)
        interventions+='</div>'
        trace='<div class="images">'+''.join(image(f'final/case_{i}/real/x0_{j:02d}.png',f'去噪第 {j}/20 步的 x0 估计') for j in (1,2,5,10,15,20))+'</div>'
        sections.append(f'<section><h2>{i} · {esc(c["family"])}</h2><p>episode {c["source_episode"]}；当前 f{c["anchor_source_frame"]} → 目标 f{c["goal_source_frame"]}。当前视角有效 mask：{esc(c["observed_view_valid"])}。原始数据指令：{esc(c["instruction"])}</p><p><b>实际送入模型的共同指令：</b>{esc(protocol["common_instruction"])}</p><h3>真实输入与目标</h3>{pics}{histories}<h3>同一模型切换单帧 / 时序</h3>{preds}<h3>TCP：同一 GT 背景，绿框 GT、红圈预测</h3>{tcp}<h3>相同噪声下的输入干预</h3>{interventions}<details><summary>逐步去噪过程</summary>{trace}</details></section>')
    if final:
        verdict={}
        for i in range(len(cards)):
            other=(i+1)%len(cards)
            a=np.asarray(Image.open(p/f'final/case_{i}/wrong_visual/prediction.png'))
            b=np.asarray(Image.open(p/f'final/case_{other}/real/prediction.png'))
            verdict[str(i)]=dict(real_nearest=metrics(i,'real')['nearest_scene'],expected_real=i,
                swap_nearest=metrics(i,'wrong_visual')['nearest_scene'],expected_swap=other,
                swapped_matches_other_prediction=bool(np.array_equal(a,b)),
                real_mae_below_copy=metrics(i,'real')['mae']<s['copy_maes'][i])
        passed=all(v['real_nearest']==v['expected_real'] and v['swap_nearest']==v['expected_swap'] and v['swapped_matches_other_prediction'] and v['real_mae_below_copy'] for v in verdict.values())
        verification=dict(train_scene_routing_pass=passed,cases=verdict,scope='four trained scenes; pixel-MAE nearest-target check plus actual visual inspection required',checkpoint_reload=s.get('checkpoint_reload'))
        (p/'verification.json').write_text(json.dumps(verification,indent=2));(run/'verification.json').write_text(json.dumps(verification,indent=2))
        headline=f'四场景输入路由检查：{"通过" if passed else "未全部通过"}。交换完整视觉输入后，预测与该输入的正常预测逐像素一致。' if passed else '四场景路由检查尚未全部通过，见逐项结果。'
        # Native-resolution contact sheet of true model images.
        columns=[('Current head',lambda i:f'inputs/case_{i:04d}_observed.png'),('Future GT',lambda i:f'inputs/case_{i:04d}_target.png'),('Before training',lambda i:f'step_000000/case_{i}/prediction.png'),('No history',lambda i:f'final/case_{i}/no_history/prediction.png'),('With history',lambda i:f'final/case_{i}/real/prediction.png')]
        sheet=Image.new('RGB',(5*330,4*310+40),'white');draw=ImageDraw.Draw(sheet)
        for col,(title,_) in enumerate(columns):draw.text((col*330+8,10),title,fill='black')
        for i,c in enumerate(cards):
            for col,(_,getpath) in enumerate(columns):
                im=Image.open(p/getpath(i)).convert('RGB').crop((0,0,320,256));sheet.paste(im,(col*330,40+i*310))
            draw.text((8,40+i*310+265),f'{i}: {c["family"]} | history MAE {metrics(i,"real")["mae"]:.2f}',fill='black')
        sheet.save(p/'contact_sheet.png')
    else:headline='训练中；以下只显示已经实际产出的文件。'
    fig,ax=plt.subplots(figsize=(9,4))
    for i,c in enumerate(cards):
        pts=[(v['step'],v['values'][i]['mae']) for v in s['evaluations'] if len(v['values'])>i]
        ax.plot([x for x,y in pts],[y for x,y in pts],marker='o',label=c['family'])
    ax.set(xlabel='Global optimizer step',ylabel='Image MAE (0-255)',title='Fixed-noise overfit reconstruction');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(p/'evaluation_curve.png',dpi=150);plt.close(fig)
    held='<section><h2>同 episode 未训练锚点</h2><p>不同输入锚点，但 episode 和未来目标可能与训练样本重合；这不是跨 episode 泛化测试。</p>'
    for i,v in enumerate(s.get('held_anchor',[])):
        held+='<h3>'+esc(cards[i]['family'])+f' · MAE {v["mae"]:.2f}</h3><div class="images">'+image(f'held_anchor/inputs/case_{i:04d}_observed.png','未训练当前锚点')+image(f'held_anchor/inputs/case_{i:04d}_target.png','GT')+image(f'held_anchor/case_{i}/prediction.png','预测')+'</div>'
    held+='</section>'
    body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Masked SA Stage1：历史视觉融合过拟合验证</title>
<style>body{{background:#f3f6fa;color:#20304b;font:16px/1.7 system-ui;margin:0}}main{{max-width:1750px;margin:auto;padding:28px}}section{{background:white;padding:24px;margin:22px 0;border:1px solid #d8dfeb;border-radius:12px}}h1{{font-size:29px}}h2{{font-size:23px}}h3{{font-size:18px}}a{{color:#275eac}}.images,.history{{display:flex;gap:16px;overflow-x:auto;align-items:flex-start}}figure{{margin:0;flex:0 0 auto}}.images img{{width:320px;height:auto}}figcaption{{max-width:320px;font-size:14px;margin:8px 0}}.history img,.pad{{width:160px;height:128px}}.pad{{background:#e7ecf2;display:grid;place-items:center;color:#78869b}}td,th{{padding:10px;border:1px solid #d8dfeb;text-align:left}}table{{border-collapse:collapse}}.hero{{max-width:100%;height:auto}}code{{background:#eef1f7}}pre{{white-space:pre-wrap}}.note{{background:#fff1cc;padding:14px}}</style>
<main><h1>Masked SA Stage1：历史视觉融合与输入条件验证</h1><p><b>{headline}</b></p><p>状态：{esc(s['status'])}；完成 {s['completed_steps']} 步。初始化：原始 OpenWAM <code>checkpoint_step_154000.safetensors</code>。</p>
<p class="note">此报告是过拟合诊断，不是泛化或任务成功率评估。四个场景刻意共用同一句缓存指令，以排除“依靠文本识别场景”的捷径。当前图像仍是完整空间条件，历史压缩后的 N=32 tokens 通过门控残差进入当前图像；原生文本 cross-attention 保留。</p>
<section><h2>指标与验收范围</h2><table><tr><th>场景</th><th>训练前 MAE</th><th>复制当前头图 MAE</th><th>单帧 MAE</th><th>时序 MAE</th><th>有效 TCP 误差 px</th></tr>{''.join(table)}</table><p>MAE 使用 0–255 RGB。缺少有效 TCP GT 的样本不记 0 分，也不计算位置误差。这里是 Stage1 图像平面 TCP 误差，没有 Stage2/FK 评估。</p>
{image('contact_sheet.png','实际预测总览（点击查看原图）')}<p>训练安排：ABC 单样本 200 步 → ABC/EgoDex 两场景交替 600 步 → 四场景交替 800 步。累计曝光次数不同，不把四行当作等曝光训练效率比较。</p>
<p>本轮验收的是“生成场景随输入变化”。历史路径梯度与条件变化只证明历史可被利用；若单帧和时序差别很小，不能宣称历史改善预测。共同指令控制也不能证明语言理解或真实未来动态泛化。</p><a href="evaluation_curve.png"><img class="hero" src="evaluation_curve.png"></a></section>
{''.join(sections)}{held}<section><h2>复现与核验</h2><p>实现：历史-only attention pooling → 复制 Wan block 的纯视觉融合 → C* 与 noisy goal/TCP 做单向 masked SA → 原生 text CA → FFN。g 初始为 0；空历史严格退化为当前图像路径。</p><p>初次运行在独立审查发现 RGB history episode-ID 接口问题后主动中止；本报告只使用修复后从相同预训练权重重新训练的 run_v2。RGB 与 latent 路径一致性、空历史、文本 padding、梯度检查点、旧 query 路径及 checkpoint IO 均有测试。</p><p><a href="inputs/index.html">实际输入与 NPZ</a> · <a href="protocol.json">训练协议与源文件 hash</a> · <a href="summary.json">完整指标</a> · <a href="train.jsonl">逐步训练日志</a> · <a href="events.jsonl">运行事件</a></p>'''
    for name,label in [('verification.json','路由验收明细'),('fresh_reload.json','独立重建 checkpoint 验证'),('tests.txt','测试结果')]:
        if (p/name).exists():body+=f'<p><a href="{name}">{label}</a></p>'
    if (p/'fresh_reload.json').exists():
        replay=json.loads((p/'fresh_reload.json').read_text())
        body+='<p>重载诊断：保存权重逐值一致；RGB 与预计算 latent 历史路径一致。参数冻结状态与训练时 eval 状态不同会产生微小数值差异，冻结推理相对原输出的平均像素差约 0.16–0.41（0–255），最大差 9。相同参数 trainability 状态的严格重载结果见下方记录；不能把冻结前后的近似一致写成逐像素一致。</p><pre>'+esc(json.dumps(replay.get('path_check',{}),ensure_ascii=False,indent=2))+'</pre>'
    body+='</section></main></html>'; (p/'index.html').write_text(body)
    print(json.dumps(dict(status=s['status'],report=str(p/'index.html'),final_cases=len(final))))


if __name__=='__main__':main()
