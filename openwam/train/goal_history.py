"""Per-case, append-only visual comparison of every published goal evaluation."""
from __future__ import annotations

import html
import copy
import tempfile
import json
from pathlib import Path


def render_goal_history(output_dir, summary, cards):
    """Static snapshot plus same-origin live appends; never replace old panels."""
    output_dir = Path(output_dir)
    summary = copy.deepcopy(summary)
    for item in summary['evaluations']:
        # Per-case MAE lives in immutable evaluation reports, not the global mean.
        report = output_dir / item['directory'] / 'metrics.json'
        if report.is_file():
            by_source = {}
            for card in json.loads(report.read_text()).get('review', []):
                value = card.get('image_mae_uint8')
                if value is not None:
                    by_source.setdefault(card['source'], []).append(value)
            item['image_metrics'] = {source: sum(values)/len(values) for source, values in by_source.items()}
    evaluations = sorted(summary['evaluations'], key=lambda item: item['step'])
    state = dict(cards=[{k: c[k] for k in ('id', 'family', 'valid_points', 'metric_key') if k in c} for c in cards],
                 summary=summary)
    rows = []
    for index, card in enumerate(cards):
        supported = all(card['valid_points'])
        suffix = 'generated.png' if supported else 'generated_clean.png'
        note = 'GT 方框，预测圆圈；左红、右蓝。' if supported else 'TCP GT 不完整：展示生成图像，原始预测点可通过诊断链接查看。'
        panels = []
        for item in evaluations:
            directory = html.escape(item['directory'], quote=True)
            metrics = item.get('point_metrics', {}).get('source:'+card.get('metric_key',card['family']), {})
            errors = []
            for side in ('left', 'right'):
                value = metrics.get(side, {}).get('pixel_error')
                errors.append('—' if value is None else f'{value:.2f}')
            image_error = item.get('image_metrics', {}).get(card.get('metric_key',card['family']))
            image_text = '—' if image_error is None else f'{image_error:.2f} / 255'
            sides = [metrics.get(side, {}) for side in ('left', 'right')]
            eligible = [side for side in sides if side.get('pixel_error') is not None and side.get('count', 0)>0]
            count = sum(side['count'] for side in eligible)
            mean_text = (f"{sum(side['pixel_error']*side['count'] for side in eligible)/count:.2f} px"
                         if count else '—（无有效 GT）')
            diagnostic = '' if supported else f' · <a href="{directory}/{index:04d}_00_generated.png">原始点诊断</a>'
            panels.append(f'<figure class="prediction" data-step="{item["step"]}">'
                f'<figcaption>第 {item["step"]} 步</figcaption>'
                f'<img loading="lazy" width="320" height="256" src="{directory}/{index:04d}_00_{suffix}" alt="第 {item["step"]} 步预测">'
                f'<p>图像 MAE：{image_text}</p><p>TCP 平均误差：{mean_text}</p>'
                f'<p>TCP 误差 L / R：{errors[0]} / {errors[1]} px</p>'
                f'<a href="{directory}/index.html">本轮详情</a>{diagnostic}</figure>')
        rows.append(f'<article data-family="{html.escape(card["family"], quote=True)}" data-index="{index}">'
            f'<h2>{html.escape(card["family"])} · {html.escape(card.get("granularity","task"))}</h2><p>{html.escape(card.get("instruction",""))}</p><p class="note">{note}</p><div class="comparison">'
            f'<figure class="reference"><figcaption>固定目标 GT</figcaption><img width="320" height="256" '
            f'src="inputs/{html.escape(card["id"], quote=True)}_target_points.png" alt="实际模型目标与有效 TCP GT">'
            f'<p>同一目标，所有步数共用</p></figure><div class="strip" tabindex="0" aria-label="按步数排列的预测">'
            f'{"".join(panels)}</div></div></article>')
    embedded = json.dumps(state, ensure_ascii=False).replace('<', '\\u003c').replace('&', '\\u0026')
    page = '''<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="goal-history-metrics" content="per-dataset-v1"><meta name="viewport" content="width=device-width,initial-scale=1"><title>G1 · 每500步连续预测对比</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f4f6f3;color:#213930;font:16px/1.6 system-ui,sans-serif}
header{position:sticky;top:0;z-index:2;background:#f4f6f3f5;border-bottom:1px solid #ccd9d0;padding:16px 24px}
header h1{margin:0;font-size:23px}header p{margin:6px 0}a{color:#17694f}main{padding:0 24px 40px}
article{padding:22px 0 28px;border-bottom:1px solid #c9d6ce}h2{margin:0;font-size:21px}.note{font-size:13px;color:#546b60;margin:5px 0 14px}
.comparison{display:flex;gap:16px;align-items:flex-start}.reference{flex:0 0 320px;background:#e1eee4;border:2px solid #7ea58b}
.strip{display:flex;flex:1;min-width:0;gap:14px;overflow-x:auto;padding:0 0 18px;scrollbar-color:#7ba78e #dce6de;scrollbar-width:auto}
figure{margin:0;border:1px solid #cedbd2;border-radius:10px;background:white;overflow:hidden}.prediction{flex:0 0 322px}
figcaption{padding:9px 12px;font-weight:750;background:#eaf0e9}figure img{display:block;width:320px;height:256px;object-fit:contain}
figure p,figure>a{font-size:12px;margin:8px 12px;display:block}figure p{color:#506858}button{cursor:pointer;font:inherit;padding:4px 12px;border:1px solid #a6bfb0;background:white;border-radius:7px}
.toolbar{display:flex;gap:20px;align-items:center;flex-wrap:wrap;font-size:13px}#live-status{font-weight:650}.help{color:#52685b;font-size:13px}
@media(max-width:700px){header,main{padding-left:12px;padding-right:12px}.comparison{flex-direction:column}.strip{width:100%}.reference{flex:auto}header h1{font-size:19px}}
@media print{header{position:static}.comparison{display:block}.strip{overflow:visible;flex-wrap:wrap}.prediction{break-inside:avoid}.toolbar{display:none}}
</style></head><body><header><h1>每个样例一行：数据集 / 粒度 / 指令 → GT → 各步预测</h1>
<p id="live-status"></p><div class="toolbar"><label><input id="sync" type="checkbox" checked>同步各行横向滚动</label>
<button id="check" type="button">检查新结果</button><a href="../index.html">实验总览</a><a href="../g1_concept_and_plan.html">方案与计划</a></div>
<p class="help">预测按步数从左到右连续排列，向右滚动比较；每500步追加一列，历史图片保留。页面每15秒检查新结果，不整页刷新。图像 MAE 为无标记 RGB 图与目标的平均绝对差（0–255）；TCP 为有效点的二维欧氏距离均值（像素），缺失 GT 不计入。</p></header><main>'''
    page += ''.join(rows) + '</main><script id="initial-state" type="application/json">' + embedded + '</script>'
    page += r'''<script>
const initial=JSON.parse(document.getElementById('initial-state').textContent);
const cards=initial.cards;
const published=new Set(initial.summary.evaluations.map(e=>e.step));
const status=document.getElementById('live-status');
function showStatus(s){const steps=[...published].sort((a,b)=>a-b);status.textContent=`训练 ${s.completed_steps??'—'} / ${s.steps??'—'} 步 · ${s.status??''} · 已展示：${steps.join(' → ')} 步`;}
showStatus(initial.summary);
function panel(item,card,index){
 const figure=document.createElement('figure');figure.className='prediction';figure.dataset.step=String(item.step);
 const title=document.createElement('figcaption');title.textContent=`第 ${item.step} 步`;figure.appendChild(title);
 const valid=card.valid_points.every(Boolean),stem=String(index).padStart(4,'0')+'_00_';
 const img=document.createElement('img');img.src=`${item.directory}/${stem}${valid?'generated.png':'generated_clean.png'}`;
 img.width=320;img.height=256;img.loading='lazy';img.alt=`第 ${item.step} 步预测`;figure.appendChild(img);
 const metric=item.point_metrics?.['source:'+(card.metric_key??card.family)]??{};
 const format=side=>metric[side]?.pixel_error==null?'—':metric[side].pixel_error.toFixed(2);
 const imageP=document.createElement('p'),imageError=item.image_metrics?.[card.metric_key??card.family];
 imageP.textContent=`图像 MAE：${imageError==null?'—':imageError.toFixed(2)+' / 255'}`;figure.appendChild(imageP);
 const eligible=['left','right'].map(side=>metric[side]).filter(side=>side?.pixel_error!=null&&side.count>0);
 const count=eligible.reduce((sum,side)=>sum+side.count,0);
 const mean=count?eligible.reduce((sum,side)=>sum+side.pixel_error*side.count,0)/count:null;
 const meanP=document.createElement('p');meanP.textContent=`TCP 平均误差：${mean==null?'—（无有效 GT）':mean.toFixed(2)+' px'}`;figure.appendChild(meanP);
 const p=document.createElement('p');p.textContent=`TCP 误差 L / R：${format('left')} / ${format('right')} px`;figure.appendChild(p);
 const a=document.createElement('a');a.href=`${item.directory}/index.html`;a.textContent='本轮详情';figure.appendChild(a);
 if(!valid){const d=document.createElement('a');d.href=`${item.directory}/${stem}generated.png`;d.textContent='原始点诊断';figure.appendChild(d);}
 return figure;
}
let polling=false;
async function update(){
 if(polling||location.protocol==='file:')return;polling=true;
 try{
  const response=await fetch('summary.json',{cache:'no-store'});if(!response.ok)throw new Error('HTTP '+response.status);
  const summary=await response.json();
  for(const item of [...summary.evaluations].sort((a,b)=>a.step-b.step)){
   if(published.has(item.step))continue;
   if(!/^eval_step_[0-9]+$/.test(item.directory))throw new Error('invalid report path');
   if(!item.image_metrics){
    const response=await fetch(`${item.directory}/metrics.json`,{cache:'no-store'});
    if(!response.ok)throw new Error('metrics HTTP '+response.status);
    const report=await response.json(),groups={};
    for(const row of report.review??[]){if(row.image_mae_uint8!=null)(groups[row.source]??=[]).push(row.image_mae_uint8);}
    item.image_metrics=Object.fromEntries(Object.entries(groups).map(([source,values])=>[source,values.reduce((a,b)=>a+b,0)/values.length]));
   }
   // Append only. Never clear, replace or fold previous predictions.
   cards.forEach((card,index)=>document.querySelector(`article[data-index="${index}"] .strip`).appendChild(panel(item,card,index)));
   published.add(item.step);
  }
  showStatus(summary);
 }catch(error){status.textContent='暂时无法更新，已展示的历史结果保持不变。'+error.message;}
 finally{polling=false;}
}
const strips=[...document.querySelectorAll('.strip')];let synchronizing=false;
strips.forEach(strip=>strip.addEventListener('scroll',()=>{
 if(synchronizing||!document.getElementById('sync').checked)return;synchronizing=true;
 strips.forEach(other=>{if(other!==strip)other.scrollLeft=strip.scrollLeft;});
 requestAnimationFrame(()=>{synchronizing=false;});
},{passive:true}));
document.getElementById('check').addEventListener('click',update);setInterval(update,15000);
</script></body></html>'''
    with tempfile.NamedTemporaryFile(mode='w', dir=output_dir, prefix='.timeline-', suffix='.tmp', delete=False) as stream:
        stream.write(page)
        temporary = Path(stream.name)
    temporary.replace(output_dir/'timeline.html')
