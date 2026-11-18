"""Four-arm report with separate native objectives and matched generated metrics."""
from __future__ import annotations
import argparse
import html
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
import rfc_distributed as shared
base=shared.base
ARMS=('default','no_history','no_g','causal_cstar')


def report(runs_root,output):
    root=Path(runs_root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    arms={};galleries=[]
    for arm in ARMS:
        directory=root/arm
        if not directory.is_dir():continue
        train=base.read_jsonl(directory/'train.jsonl');evaluations=base.read_jsonl(directory/'eval.jsonl')
        visuals=base.read_jsonl(directory/'visual.jsonl')
        arms[arm]=dict(train=train,evaluations=evaluations,visuals=visuals,
            latest_train_update=train[-1]['update'] if train else None,
            latest_eval_update=evaluations[-1]['update'] if evaluations else None,
            latest_visual=visuals[-1] if visuals else None,
            train_slopes={metric:{axis:base.ols(train,axis,metric) for axis in ('update','samples','wall_seconds')}
                          for metric in (('loss','fm_loss','tcp_loss') if arm=='causal_cstar' else ('loss','fm_loss','uv_loss'))})
        for path in sorted((directory/'visuals').glob('step_*/index.html')):
            galleries.append((arm,os.path.relpath(path,output)))
    complete=set(arms)==set(ARMS)
    common=set.intersection(*[{r['update'] for r in arms[a]['visuals']} for a in ARMS]) if complete else set()
    common_step=None;common_rows={};rejected=[]
    for step in sorted(common,reverse=True):
        selected={arm:next(r for r in reversed(arms[arm]['visuals']) if r['update']==step) for arm in ARMS}
        ids={arm:[r['id'] for r in row['records']] for arm,row in selected.items()}
        counts={arm:(row['counts'].get('valid_points'),row['counts'].get('head_rgb_values')) for arm,row in selected.items()}
        if len({tuple(v) for v in ids.values()})==1 and len(set(counts.values()))==1:
            common_step=step;common_rows=selected;break
        rejected.append(dict(update=step,reason='generated source IDs or validity denominators differ',ids=ids,counts=counts))
    payload=dict(format='openwam.rfc_stage1.causal_cstar_comparison.report.v1',generated_unix=time.time(),
        expected_arms=ARMS,arms=arms,objectives_comparable=False,
        objective_protocols=dict(G='multiview continuous-sigma FM + clean G UV regression',
            causal_cstar='native head image FM weighted shifted1000-grid (lambda .5) +192D TCP-patch FM (lambda1)'),
        common_generation=dict(update=common_step,all_four_present=complete,rows=common_rows,rejected=rejected,
            protocol='Same original head RGB/valid TCP ground truth and same source IDs; native20-step generation. G UV is clean G readout, C* UV jointly generated TCP patch. Representations and objectives differ.'),
        scope='Fresh causal C* seed42 vs three G arms continued strictly from500; same1000 total-update/8000draw budget. Only generated original-headRGB MAE and valid TCPpx share physical units. Wall-time origin resets by execution phase.')
    base.atomic_json(output/'metrics.json',payload)
    def number(value):return 'n/a' if value is None else f'{value:.6g}'
    def row(arm,visual):
        visual=visual or {};metrics=visual.get('metrics',{});counts=visual.get('counts',{})
        values=(arm,arms[arm]['latest_train_update'],arms[arm]['latest_eval_update'],visual.get('update'),
                number(metrics.get('head_image_mae')),number(metrics.get('tcp_px')),counts.get('valid_points',0),counts.get('head_rgb_values',0))
        return '<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in values)+'</tr>'
    header='<tr><th>Arm</th><th>Train update</th><th>Eval update</th><th>Generation update</th><th>Original head RGB MAE[0,1]</th><th>TCPpx</th><th>Valid TCP points</th><th>Head RGB values</th></tr>'
    matched=''.join(row(a,v) for a,v in common_rows.items())
    latest=''.join(row(a,v['latest_visual']) for a,v in arms.items())
    links=''.join(f'<li><a href="{html.escape(url)}">{html.escape(arm+" "+url)}</a></li>' for arm,url in galleries)
    chart=json.dumps({a:v['train'] for a,v in arms.items()},allow_nan=False).replace('</',r'<\/')
    page='''<!doctype html><meta charset="utf-8"><title>G and native causal C* comparison</title>
<style>body{font:15px system-ui;max-width:1250px;margin:30px auto;color:#17212b;background:#fafafa}td,th{padding:8px;border:1px solid #ccc}table{border-collapse:collapse}canvas{width:100%;height:320px;background:white;border:1px solid #ccc}select{margin:10px}p{line-height:1.5}</style>
<h1>Three G arms and fresh native causal C*</h1>
<p>Same foundation seed42, source anchors/instructions/grain-terminal labels, finite8000draw stream and1000total updates. C* retains its original causal10-frame1Hz head history, language C* adapter and joint image/TCP flow objective. G continuation begins at checkpoint500; the new C* arm begins at zero. Training objective scales and diffusion schedules differ.</p>
<p><a href="metrics.json">Machine metrics, full curves and slope counts</a>. Only original-headRGB MAE and valid TCP pixel errors share physical units across architectures. G TCP is its clean G readout; C* TCP comes from native joint20-step generation. Native one-step FM validation stays separate.</p>
<h2>Matched common generated cases</h2><p>Latest common generation update: COMMON. All four present: PRESENT. Matching requires identical ordered source IDs and validity denominators.</p><table>HEADERMATCHED</table>
<h2>Latest per-arm generation (updates may differ)</h2><table>HEADERLATEST</table>
<label>Horizontal axis<select id="axis"><option>update</option><option>samples</option><option>wall_seconds</option></select></label>
<label>Objective term<select id="term"><option>loss</option><option>fm_loss</option><option>auxiliary</option></select></label>
<h2>G objective</h2><p>Default blue; no_history orange; no_g green. FM supervises the multiview target, auxiliary is G UV regression.</p><canvas id="g" width="1200" height="320"></canvas>
<h2>Native causal C* objective</h2><p>Purple. Head image FM uses native shifted-grid weights and lambda .5; auxiliary is192D TCP-patch FM with lambda1. Its y-axis is separate from G.</p><canvas id="cstar" width="1200" height="320"></canvas>
<h2>Fixed original head RGB / current-weight generation galleries</h2><ul>LINKS</ul>
<p>Missing points remain n/a. TCP uses normalized UV×[319,255]. The three G early-phase artifacts remain in their original execution directory. Current-phase wall seconds exclude model/data bootstrap and include evaluation/visualization/checkpoints.</p>
<script>const data=DATA;function panel(id,names){const c=document.getElementById(id),g=c.getContext('2d'),x=document.getElementById('axis').value,t=document.getElementById('term').value;const metric=a=>t==='auxiliary'?(a==='causal_cstar'?'tcp_loss':'uv_loss'):t;const rows=names.flatMap(a=>(data[a]||[]).map(r=>({x:r[x],y:r[metric(a)]}))).filter(r=>Number.isFinite(r.x)&&Number.isFinite(r.y));g.clearRect(0,0,c.width,c.height);if(!rows.length)return;const xa=Math.min(...rows.map(r=>r.x)),xb=Math.max(...rows.map(r=>r.x)),ya=Math.min(...rows.map(r=>r.y)),yb=Math.max(...rows.map(r=>r.y));g.strokeStyle='#ddd';g.strokeRect(65,15,1100,260);g.fillStyle='#333';g.fillText(xa,65,300);g.fillText(xb,1100,300);g.fillText(ya.toPrecision(4),2,270);g.fillText(yb.toPrecision(4),2,20);names.forEach(a=>{g.strokeStyle={default:'#1677bb',no_history:'#e67e22',no_g:'#249647',causal_cstar:'#8e44ad'}[a];g.beginPath();let started=false;(data[a]||[]).forEach(r=>{const y=r[metric(a)];if(!Number.isFinite(r[x])||!Number.isFinite(y))return;const u=65+(r[x]-xa)/(xb-xa||1)*1100,v=275-(y-ya)/(yb-ya||1)*260;started?g.lineTo(u,v):g.moveTo(u,v);started=true});g.stroke()})}function draw(){panel('g',['default','no_history','no_g']);panel('cstar',['causal_cstar'])}document.getElementById('axis').onchange=draw;document.getElementById('term').onchange=draw;draw();</script>'''
    page=page.replace('COMMON',str(common_step) if common_step is not None else 'none yet').replace('PRESENT',str(complete))
    page=page.replace('HEADER',header).replace('MATCHED',matched).replace('LATEST',latest).replace('LINKS',links).replace('DATA',chart)
    base.atomic_text(output/'index.html',page);return payload


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--runs-root',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=report(args.runs_root,args.output)
    print(json.dumps(dict(output=args.output,arms=list(result['arms']),common_generation=result['common_generation']['update']),indent=2))
