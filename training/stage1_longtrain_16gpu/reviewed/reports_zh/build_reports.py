"""Chinese presentation only: preserve frozen execution code and source metrics."""
from __future__ import annotations
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import statistics

ARMS=('default','no_history','no_g','causal_cstar')
LABELS={'default':'默认 G','no_history':'无历史','no_g':'屏蔽 G→图像路径','causal_cstar':'旧因果 C*'}
COLORS={'default':'#1762c4','no_history':'#c2630b','no_g':'#15825d','causal_cstar':'#8741bd'}
FAMILIES={'morphirobot0930':'Morphi 机器人','posttrain_native':'后训练原生数据','abc_130k':'ABC 130K','interndata_a1':'Interndata A1'}
CSS='''
:root{font-family:Inter,"Noto Sans CJK SC","Microsoft YaHei",sans-serif;color:#17243b;background:#f3f6fa;font-size:15px}*{box-sizing:border-box}body{max-width:1280px;margin:0 auto;padding:34px 24px 70px}a{color:#175eb1;text-underline-offset:3px}h1{font-size:30px;margin:8px 0 14px;letter-spacing:-.5px}h2{font-size:22px;margin:0 0 14px}h3{font-size:17px}p,li{line-height:1.8}nav{display:flex;flex-wrap:wrap;gap:18px;margin:18px 0 26px}.eyebrow{font-size:12px;color:#526c8c;letter-spacing:1px}.muted,.note{color:#56677c}.card{background:white;border:1px solid #dce4ef;border-radius:14px;padding:24px;margin:20px 0;box-shadow:0 3px 14px #132e5710}.callout{background:#edf4ff;border-left:4px solid #1762c4;padding:14px 20px;border-radius:6px}.conclusion{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.finding{padding:17px;background:#f6f9fd;border:1px solid #e2eaf3;border-radius:10px}.finding b{display:block;margin-bottom:8px}.finding p{margin:0;font-size:14px}.table-wrap{overflow:auto}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:14px}td,th{text-align:left;padding:11px 14px;border-bottom:1px solid #e4eaf2;white-space:nowrap}th{background:#f3f7fb;color:#405976;font-weight:600}.swatch{display:inline-block;width:26px;height:4px;vertical-align:middle;margin-right:9px;border-radius:4px}select,input{font:inherit;max-width:100%}select,input[type=search]{border:1px solid #bdcadb;border-radius:7px;padding:8px 12px;background:white;color:#223b5a}.controls{display:flex;flex-wrap:wrap;gap:16px;align-items:center;margin:12px 0 16px}.controls label{display:flex;align-items:center;gap:8px}.legend{display:flex;flex-wrap:wrap;gap:10px;margin:15px 0 8px}.legend-item{font:inherit;font-weight:600;color:#24364e;border:1px solid #bac9dd;border-radius:8px;background:white;padding:10px 14px;cursor:pointer}.legend-item[aria-pressed=false]{opacity:.42;text-decoration:line-through}.legend-item:focus-visible{outline:3px solid #8ab5ed}.legend-item .swatch{width:34px}.chart-wrap{position:relative}.chart-panel canvas{display:block;width:100%;height:350px;touch-action:pan-y}.chart-tooltip{position:absolute;right:10px;top:14px;background:#fffffff5;border:1px solid #afc0d7;border-radius:9px;padding:13px;box-shadow:0 6px 22px #1c345c26;pointer-events:none;font-size:12px;z-index:2;max-width:95%}.chart-tooltip table{font-size:12px}.chart-tooltip td,.chart-tooltip th{padding:5px 7px}.chart-tooltip i{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px}.badges{display:flex;flex-wrap:wrap;gap:10px}.badge{background:#edf2f8;border-radius:20px;padding:7px 13px;font-size:13px}.case-card{border:1px solid #dbe4ef;border-radius:10px;margin:12px 0;background:white}.case-card summary{cursor:pointer;padding:17px;font-weight:600}.case-body{padding:0 18px 18px}.instruction{background:#f4f7fb;padding:12px;border-radius:7px;word-break:break-word}figure{margin:18px 0;border:1px solid #dce5ef;border-radius:9px;overflow:hidden}figcaption{padding:10px 14px;background:#f5f8fc;line-height:1.8;font-size:14px}.image-labels{display:grid;grid-template-columns:repeat(3,1fr);text-align:center;background:#eaf0f8;padding:9px 0;font-size:13px}.triptych-frame{position:relative;aspect-ratio:960/256;overflow:hidden;background:#eee}.triptych-frame img{position:absolute;bottom:0;left:0;width:100%;display:block}details.group{margin:12px 0}details.group>summary{cursor:pointer;padding:10px 0;font-weight:600}.small{font-size:13px}code{font:12px ui-monospace,monospace;overflow-wrap:anywhere}footer{font-size:13px;color:#64738a;margin-top:30px}.count{font-size:13px;color:#526c8c}.filters{position:sticky;top:0;background:#fffffff2;padding:12px;z-index:3;border:1px solid #e0e7ef;border-radius:8px}.hidden{display:none!important}@media(max-width:650px){body{padding:20px 12px}h1{font-size:25px}.card{padding:16px}.conclusion{grid-template-columns:1fr}.legend-item{font-size:13px;padding:9px}.chart-tooltip{left:8px;right:auto}.chart-tooltip table{font-size:10px}.chart-tooltip td,.chart-tooltip th{padding:4px}.controls{gap:12px}.image-labels{font-size:11px}figcaption{font-size:12px}}
'''

def esc(v):return html.escape(str(v))
def num(v,places=4):return '不可评估' if v is None else f'{v:.{places}f}'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def relative(path,directory):return os.path.relpath(path,directory)
def jsdata(value):return json.dumps(value,ensure_ascii=False,allow_nan=False).replace('</',r'<\/')
def family(value):return FAMILIES.get(value,value)
def grain(value):return {'task':'任务级','subtask':'子任务级'}.get(value,value)

def layout(title,body,links,scripts=''):
    return '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+esc(title)+'</title><style>'+CSS+'</style></head><body><div class="eyebrow">OPENWAM · 四组对照 · 更新至 1000 步</div><h1>'+esc(title)+'</h1><nav>'+''.join('<a href="'+esc(u)+'">'+esc(t)+'</a>' for t,u in links)+'</nav>'+body+'<footer>界面与说明已中文化。原始指令、数据来源名称、样本编号和机器可读指标保留原文；训练权重、原始统计与预测图片未改动。</footer>'+scripts+'</body></html>'

def conclusions(full):
    m={a:full['arms'][a]['metrics'] for a in ARMS};g=m['default'];old=m['causal_cstar'];nh=m['no_history'];ng=m['no_g']
    tcp=(g['tcp_px']-old['tcp_px'])/g['tcp_px']*100
    hist=(nh['head_image_mae']/g['head_image_mae']-1)*100
    img=(ng['head_image_mae']/g['head_image_mae']-1)*100
    return '<section class="card" id="conclusions"><h2>这轮实验的结论</h2><div class="conclusion">'+''.join('<div class="finding"><b>'+t+'</b><p>'+p+'</p></div>' for t,p in [
      ('旧因果 C* 的 TCP 误差最低',f'旧 C* 为 {old["tcp_px"]:.2f} 像素，默认 G 为 {g["tcp_px"]:.2f} 像素；本轮相对降低 {tcp:.1f}%。旧 C* 与默认 G 的头图误差仅相差 {abs(old["head_image_mae"]-g["head_image_mae"]):.6f}，图像结果接近。'),
      ('历史输入有助于图像预测',f'去掉历史后，头图 MAE 从 {g["head_image_mae"]:.5f} 增至 {nh["head_image_mae"]:.5f}，增加 {hist:.1f}%。但无历史组的 TCP 为 {nh["tcp_px"]:.2f} 像素，反而低于默认 G；不能概括为历史同时改善了所有指标。'),
      ('G→图像通路尚未显示明确增益',f'屏蔽该通路后，图像 MAE 仅增加 {img:.2f}%，TCP 为 {ng["tcp_px"]:.2f} 像素。此消融仍保留 G 的 UV 辅助监督，不等于删除 G 模块，也不能据此否定 G 的全部作用。'),
      ('结论适用范围','本轮只有一个随机种子、256 条固定训练样本和 64 条验证记录。旧 C* 保留 10 帧头图历史，新 G 使用多视图历史；损失与扩散调度也不同。这是系统基线对照，尚不能把差异单独归因于 G 结构。')])+'</div></section>'

def summary_table(full):
    header='<tr><th>实验</th><th>头图 MAE ↓（0–1）</th><th>TCP 误差 ↓（像素）</th><th>验证记录</th><th>有效头图</th><th>有效 TCP 点</th></tr>'
    rows=''
    for a in ARMS:
      v=full['arms'][a];c=v['counts'];m=v['metrics'];rows+='<tr><td><span class="swatch" style="background:'+COLORS[a]+'"></span>'+LABELS[a]+'</td><td>'+num(m['head_image_mae'],5)+'</td><td>'+num(m['tcp_px'],2)+'</td><td>'+str(c['records'])+'</td><td>'+str(c['valid_head_images'])+'</td><td>'+str(c['valid_points'])+'</td></tr>'
    return '<div class="table-wrap"><table>'+header+rows+'</table></div>'

def chart_panel(panel_id,names,title,note):
    legends=''.join('<button type="button" class="legend-item" data-arm="'+a+'" aria-pressed="true"><span class="swatch" style="background:'+COLORS[a]+'"></span>'+LABELS[a]+'</button>' for a in names)
    line_note='蓝色实线、橙色虚线、绿色点线分别对应上方图例。' if len(names)>1 else '紫色实线表示旧因果 C*。'
    return '<section class="card chart-panel" data-arms="'+','.join(names)+'"><h2>'+title+'</h2><p class="small muted">'+note+'</p><div class="legend" aria-label="'+title+'图例">'+legends+'</div><p class="small muted">点击图例可单独显示或隐藏；将鼠标移到曲线上可读取数值。'+line_note+'</p><div class="chart-wrap"><canvas id="'+panel_id+'" role="img" aria-label="'+title+'，图例在图上方"></canvas><div class="chart-tooltip" hidden></div></div></section>'

def loss_page(exp,loss,full):
    directory=exp/'runs/comparison_report'
    links=[('完整效果与中文结论',relative(exp/'full_generation/report/index.html',directory)),('原始曲线指标（JSON）','metrics.json')]
    body='<div class="callout">三组 G 可以比较相同定义的 loss；旧 C* 的总损失、FM 和辅助项定义不同，因此单独绘图。当前 G 曲线展示 8 卡续训的第 501–1000 步；旧 C* 展示第 1–1000 步。</div>'
    body+='<section class="card"><h2>曲线显示设置</h2><div class="controls"><label>横轴<select id="axis"><option value="update">训练更新步数</option><option value="samples">累计训练样本数</option><option value="wall_seconds">当前执行阶段耗时</option></select></label><label>指标<select id="term"><option value="loss">总损失</option><option value="fm_loss">图像流匹配损失（FM）</option><option value="auxiliary">辅助损失</option></select></label><label>平滑窗口<select id="smooth"><option value="1">原始曲线</option><option value="10">10 步均值</option><option value="25" selected>25 步均值</option><option value="50">50 步均值</option></select></label><label><input type="checkbox" id="raw-lines">叠加淡色原始曲线</label></div><p id="term-note" class="note">G：图像 FM＋UV 回归；旧 C*：0.5×头图 FM＋TCP patch FM。两类模型的绝对 loss 不直接比较。</p><p class="small muted">均值为向后看的滑动平均；开头不足窗口长度时使用已有点。平滑只改变显示，不改变数据。耗时轴包含当前循环内评估和保存，不含模型启动。</p></section>'
    body+=chart_panel('g',['default','no_history','no_g'],'G 系列：三组消融的损失曲线','同一输入、相同全局 batch、样本顺序和噪声。屏蔽 G→图像路径仍保留 G 的 UV 监督。')
    body+=chart_panel('cstar',['causal_cstar'],'旧因果 C*：原生损失曲线','头图流匹配权重为 0.5，辅助项为 192 维 TCP patch 流匹配；与 G 使用不同的纵轴。')
    body+='<section class="card"><h2>G 三组在共同窗口的下降情况</h2><p>比较 501–600 步与 901–1000 步的原始总 loss 均值；不能用下降幅度代替最终验证效果。</p><div class="table-wrap"><table><tr><th>实验</th><th>501–600 步均值</th><th>901–1000 步均值</th><th>相对下降</th></tr>'
    for a in ARMS[:3]:
      r=loss['arms'][a]['train'];before=statistics.mean(x['loss'] for x in r if 501<=x['update']<=600);after=statistics.mean(x['loss'] for x in r if 901<=x['update']<=1000)
      body+='<tr><td>'+LABELS[a]+'</td><td>'+num(before)+'</td><td>'+num(after)+'</td><td>'+f'{(1-after/before)*100:.1f}%'+'</td></tr>'
    body+='</table></div></section>'+conclusions(full)
    data={a:[{k:r[k] for k in ('update','samples','wall_seconds','loss','fm_loss',('tcp_loss' if a=='causal_cstar' else 'uv_loss'))} for r in loss['arms'][a]['train']] for a in ARMS}
    scripts='<script type="application/json" id="chart-data">'+jsdata(data)+'</script><script>'+Path(__file__).with_name('charts.js').read_text()+'</script>'
    return layout('训练损失曲线：中文图例与交互读数',body,links,scripts)

def group_title(value):
    kind,name=value.split(':',1)
    if kind=='grain':return '指令粒度：'+grain(name)
    if kind=='family':return '数据来源：'+family(name)
    f,g=name.split('/',1);return '数据来源 × 粒度：'+family(f)+' / '+grain(g)

def effect_page(exp,full):
    directory=exp/'full_generation/report'
    links=[('损失曲线与中文图例',relative(exp/'runs/comparison_report/index.html',directory)),('原始完整指标（JSON）','metrics.json'),('逐样本预测','#cases')]
    body='<div class="badges"><span class="badge">4 组均完成 1000 步</span><span class="badge">同一批 64 条验证记录</span><span class="badge">57 张有效头图 · 91 个有效 TCP 点</span><span class="badge">图像均为 20 步生成</span></div>'+conclusions(full)
    body+='<section class="card"><h2>完整验证结果</h2>'+summary_table(full)+'<p class="small muted">MAE 为原始目标头图与预测图像的 RGB 平均绝对误差，范围 0–1，越低越好。TCP 为有效端点的平均欧氏像素误差，以 320×256 头图和 [319,255] 缩放计算，不包含无标签点。</p><p class="small muted">四组样本顺序、图像/TCP 标签与有效性掩码的指纹完全一致。G 的 TCP 来自干净 G 读出；旧 C* 的 TCP 来自原生联合 20 步生成，推理方式不同但物理误差单位相同。</p></section>'
    body+='<section class="card"><h2>实验范围与解读边界</h2><ul><li>训练固定 256 条样本、8000 次抽样，任务级与子任务级按 3:7 混训；验证为 40 条任务级、24 条子任务级，验证比例与训练比例不同。</li><li>验证覆盖 10 类数据来源，其中 Genrobot 的 7 条没有有效头图/TCP 标签，因此显示“不可评估”；有可计量监督的是另外 9 类。</li><li>这是单个随机种子的小样本对照，不能据此声称统计显著、完整数据泛化或机器人任务成功率。</li><li>三组 G 从共同第 500 步检查点保留权重与 Adam 状态转为 8 卡继续训练；旧 C* 从相同基础模型重新训练至 1000 步，原 update_8133 没有用于本轮重新训练。</li></ul></section>'
    body+='<section class="card"><h2>按数据来源与指令粒度查看</h2><p class="small muted">“不可评估”表示有效分母为零，不能记成零误差。以下均使用各组相同的样本与有效标注。</p>'
    for group in sorted(full['arms']['default']['groups'],key=lambda s:(0 if s.startswith('grain:') else 1 if s.startswith('family:') else 2,s)):
      view={'arms':{a:full['arms'][a]['groups'][group] for a in ARMS}}
      body+='<details class="group"'+(' open' if group.startswith('grain:') else '')+'><summary>'+esc(group_title(group))+'</summary>'+summary_table(view)+'</details>'
    body+='</section><section class="card"><h2>计算与来源记录</h2><p>每组用 8 个 GPU 进程，各处理 8 条验证记录。下表只统计生成阶段，模型加载与依赖校验另计。</p><div class="table-wrap"><table><tr><th>实验</th><th>生成用时（秒）</th><th>GPU 进程数</th><th>每卡峰值显存（GiB）</th></tr>'
    for a in ARMS:
      m=full['arms'][a];peaks=[r['peak_allocated_bytes']/2**30 for r in m['ranks']]
      body+='<tr><td>'+LABELS[a]+'</td><td>'+num(m['generation_wall_seconds'],2)+'</td><td>8</td><td>'+f'{min(peaks):.2f}–{max(peaks):.2f}'+'</td></tr>'
    body+='</table></div><details class="group"><summary>查看权重校验值</summary>'+''.join('<p>'+LABELS[a]+'<br><code>'+full['arms'][a]['weights']['sha256']+'</code></p>' for a in ARMS)+'</details></section>'
    families=sorted({r['family'] for r in full['arms']['default']['cases']})
    body+='<section class="card" id="cases"><h2>全部 64 条样本：当前头图／真实目标／预测结果</h2><p>每条样本可展开四组对照。青色圆点表示左端点，洋红色表示右端点；只绘制有有效标签的点。图像上方的英文页眉已在页面中收起，由中文列名替代，原始图片保留不变。</p><div class="controls filters"><label>来源<select id="case-family"><option value="">全部来源</option>'+''.join('<option value="'+esc(f)+'">'+esc(family(f))+'</option>' for f in families)+'</select></label><label>粒度<select id="case-grain"><option value="">全部粒度</option><option value="task">任务级</option><option value="subtask">子任务级</option></select></label><input id="case-search" type="search" placeholder="搜索原始指令或样本编号" aria-label="搜索原始指令或样本编号"><span id="case-count" class="count">64 / 64 条</span></div>'
    for index,case in enumerate(full['arms']['default']['cases']):
      body+='<details class="case-card" data-family="'+esc(case['family'])+'" data-grain="'+case['grain']+'"'+(' open' if index==0 else '')+'><summary>'+f'{index+1:02d} · '+esc(family(case['family']))+' · '+grain(case['grain'])+' · '+esc(case['id'])+'</summary><div class="case-body"><p class="instruction">原始训练指令（保留原文）：'+esc(case['instruction'])+'</p>'
      for a in ARMS:
        r=full['arms'][a]['cases'][index];image=exp/'full_generation'/a/'step_001000'/r['image']
        body+='<figure><figcaption><span class="swatch" style="background:'+COLORS[a]+'"></span><b>'+LABELS[a]+'</b>　头图 MAE：'+num(r['metrics']['head_image_mae'],5)+'　TCP：'+num(r['metrics']['tcp_px'],2)+' 像素　有效端点：'+str(r['counts']['valid_points'])+'</figcaption><div class="image-labels"><span>当前头图</span><span>真实目标头图</span><span>模型预测</span></div><div class="triptych-frame"><img loading="lazy" src="'+esc(relative(image,directory))+'" alt="'+esc(LABELS[a]+'：当前头图、真实目标、模型预测')+'"></div></figure>'
      body+='</div></details>'
    body+='</section>'
    script='''<script>const cards=[...document.querySelectorAll('.case-card')];function filterCases(){const f=document.getElementById('case-family').value,g=document.getElementById('case-grain').value,q=document.getElementById('case-search').value.toLowerCase();let n=0;cards.forEach(c=>{const show=(!f||c.dataset.family===f)&&(!g||c.dataset.grain===g)&&(!q||c.textContent.toLowerCase().includes(q));c.hidden=!show;if(show)n++;});document.getElementById('case-count').textContent=n+' / 64 条';}['case-family','case-grain','case-search'].forEach(id=>document.getElementById(id).addEventListener('input',filterCases));</script>'''
    return layout('四组模型完整效果对比与中文结论',body,links,script)

def write_page(path,text):
    backup=path.with_name('index.en.20261111.html')
    if path.exists() and not backup.exists():backup.write_bytes(path.read_bytes())
    temp=path.with_name('index.zh.tmp');temp.write_text(text);temp.replace(path)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);args=p.parse_args()
    exp=args.experiment.resolve();loss_path=exp/'runs/comparison_report/metrics.json';full_path=exp/'full_generation/report/metrics.json'
    before={str(p):sha(p) for p in (loss_path,full_path)}
    loss=json.loads(loss_path.read_text());full=json.loads(full_path.read_text())
    if not full['matching']['matched'] or not full['matching']['all_four_present']:raise ValueError('Requires four matched complete evaluations')
    pages={exp/'runs/comparison_report/index.html':loss_page(exp,loss,full),exp/'full_generation/report/index.html':effect_page(exp,full)}
    for path,text in pages.items():write_page(path,text)
    if before!={str(p):sha(p) for p in (loss_path,full_path)}:raise ValueError('Source metrics changed')
    args.receipt.parent.mkdir(parents=True,exist_ok=True);args.receipt.write_text(json.dumps(dict(source_metric_hashes=before,pages={str(p):sha(p) for p in pages},renderer=str(Path(__file__).resolve()),language='zh-CN'),indent=2)+'\n')
    print(json.dumps(dict(pages=list(map(str,pages)),source_metrics_unchanged=True),ensure_ascii=False))

if __name__=='__main__':main()
