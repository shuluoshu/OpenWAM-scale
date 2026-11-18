#!/usr/bin/env python3
"""Chinese, offline before/after report from verified seven-case DSW artifacts."""
from __future__ import annotations
import argparse
import base64
import html
import io
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from openwam.train.goal_image_metrics import goal_image_metrics


def number(value, digits=4): return '不可计分' if value is None else f'{value:.{digits}f}'
def esc(value): return html.escape(str(value))


def png(value):
    if isinstance(value, torch.Tensor):
        value = value[:, 0].permute(1, 2, 0).numpy()
        if value.dtype != np.uint8: value = np.rint(np.clip(value, 0, 1) * 255).astype(np.uint8)
        value = Image.fromarray(value)
    output = io.BytesIO(); value.save(output, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(output.getvalue()).decode()


def tile(image, title, note='', *, truth=None, valid=None, prediction=None):
    layers = []
    for hand, name in enumerate(('左', '右')):
        if truth is not None and valid[hand]:
            x, y = np.asarray(truth[hand]) * [319, 255]
            layers.append(f'<g class="truth"><circle cx="{x}" cy="{y}" r="5" fill="none" stroke="#00ffc6" stroke-width="2.2"/><text x="{x+6}" y="{y-6}" fill="#00ffc6">{name}GT</text></g>')
        if prediction is not None:
            x, y = np.asarray(prediction[hand]) * [319, 255]
            if np.isfinite([x, y]).all():
                layers.append(f'<g class="pred"><path d="M{x-5},{y-5}l10,10m-10,0l10,-10" stroke="#ffb020" stroke-width="2.2"/><text x="{x+6}" y="{y+14}" fill="#ffb020">{name}预测</text></g>')
    return (f'<figure><figcaption>{esc(title)}</figcaption><svg viewBox="0 0 320 256" role="img" aria-label="{esc(title)}">'
            f'<image href="{png(image)}" width="320" height="256"/>{"".join(layers)}</svg><p>{esc(note)}</p></figure>')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runs-root', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    args = parser.parse_args(); out = args.output
    labels = json.loads((out / 'labels.json').read_text()); inference = json.loads((out / 'inference.json').read_text())
    assert inference['complete'] and len(labels['cases']) == 7
    evidence = []; cards = []
    originals = json.loads((args.prepared / 'manifest.json').read_text())
    versioned = json.loads((out / 'prepared/G/manifest.json').read_text())
    for case in labels['cases']:
        index = case['source_index']; valid = case['new_image_valid']
        entry = next(e for e in versioned['records']['val'] if e['source_index'] == index)
        row = torch.load(out / 'prepared/G' / entry['file'], weights_only=False)
        oldrow = torch.load(args.prepared / originals['records']['val'][index]['file'], weights_only=False)
        truth = row['target_uv'].tolist()
        point_valid = (row['target_point_valid'] & row['target_in_frame'] & row['target_image_valid']).tolist()
        current = row['raw_current_rgb'][:, :, :256]; target = row['raw_target_rgb'][:, :, :256]
        source = Image.open(row['record']['goal_image_path']).convert('RGB')
        inputs = tile(current, f'当前帧 · {case["anchor"]}', '两套模型输入沿用旧缓存，文本与历史不变。')
        inputs += tile(oldrow['raw_target_rgb'][:, :, :256], '修复前 · 被mask的目标', '旧协议图像无效；黑图是占位，不是源视频丢帧。')
        inputs += tile(target if valid else source, f'修复后 · {"目标关键帧" if valid else "原始候选帧（仍无效）"} · {case["goal"]}',
                       '目标源帧及源图hash不变；绿圈为有效GT TCP。' if valid else 'canonical无效。原图仅供定位，不能作为GT或计分。',
                       truth=truth, valid=point_valid)
        generated = []; metrics_rows = []; overlay = []; measured = {}
        for arch, step, name in [('G', 5515, '新G'), ('Cstar', 3321, '旧C*')]:
            olddir = args.runs_root / arch / 'validation' / f'step_{step:09d}'
            historical = json.loads((olddir / 'records' / f'{index:04d}.json').read_text())
            before = json.loads((out / 'before_dsw' / arch / f'{index:04d}.json').read_text())
            after = json.loads((out / 'after' / arch / f'{index:04d}.json').read_text())
            payload = torch.load(out / 'after' / arch / f'{index:04d}.pt', weights_only=True)
            assert before['id'] == after['id'] == case['id']
            assert after['counts']['valid_head_images'] == int(valid)
            if not valid: assert all(v is None for v in after['metrics'].values())
            baseline = torch.load(out / 'before_dsw' / arch / f'{index:04d}.pt', weights_only=True)
            old_tensor = baseline['head']; old_prediction = old_tensor
            a = payload['head']; pred = payload['uv'].tolist()
            note = '旧G被目标mask强制置零latent后的解码占位，不能称为有效生成。' if arch == 'G' else '原生20步真实预测；旧GT无效，因此原报告不计分。'
            generated.append(tile(old_prediction, f'{name} · 修复前 · {step}步', note, prediction=before['generated_uv']))
            generated.append(tile(a, f'{name} · 修复后 · {step}步', '同权重、同输入、同seed的20步真实生成；绿圈GT、橙叉预测。',
                                  truth=truth, valid=point_valid, prediction=pred))
            overlay.append(tile(target if valid else source, f'{name} · GT背景TCP对照',
                '用相同GT背景直接看TCP位置误差。' if valid else '仅显示橙色模型输出；无有效GT不画绿圈。',
                truth=truth, valid=point_valid, prediction=pred))
            region = after['region_errors']['regions']
            metrics_rows.append('<tr>' + ''.join(f'<td>{cell}</td>' for cell in [esc(name),
                '不可计分', number(after['metrics']['head_image_mae']), number(after['metrics']['tcp_px'], 2),
                number(region['full']['copy_mae']), number(region['changed']['model_mae']),
                number(region['changed']['gain_over_copy']), number(region['tcp']['model_mae']),
                number(region['tcp']['gain_over_copy'])]) + '</tr>')
            quantized_same = bool(torch.equal((old_tensor * 255).round().byte(), (a * 255).round().byte()))
            delta = float((old_tensor - a).abs().mean())
            measured[arch] = dict(old_native_metrics=before['metrics'], historical_DLC_metrics=historical['metrics'],
                new_metrics=after['metrics'],
                prediction_quantized_equal=quantized_same, prediction_change_mae=delta,
                new_region_errors=after['region_errors'],
                old_output_rescored_new_labels=goal_image_metrics(old_tensor, current, target,
                    image_valid=valid, target_uv=row['target_uv'], point_valid=torch.tensor(point_valid)),
                tcp_uv_delta_max=float((torch.tensor(before['generated_uv']) - payload['uv']).abs().max()))
        status = '恢复RGB与双手TCP' if valid else '维持无效，不计分'
        masktext = f'canonical={case["source_masks"]["canonical.frame_valid"]}；training.valid={case["source_masks"]["training.valid"]}；Stage2 source_valid={case["old_source_valid"]}；RGB：{case["old_image_valid"]} → {valid}；TCP：{case["old_point_valid"]} → {case["new_point_valid"]}'
        table = '<div class="table-wrap"><table><thead><tr><th>模型</th><th>旧协议MAE</th><th>新协议MAE↓</th><th>TCP像素↓</th><th>复制全图MAE</th><th>变化区域MAE↓</th><th>变化区域改进↑</th><th>TCP邻域MAE↓</th><th>TCP邻域改进↑</th></tr></thead><tbody>' + ''.join(metrics_rows) + '</tbody></table></div>'
        cards.append(f'<article class="case" data-valid="{str(valid).lower()}"><h2>样本{index} · {status}</h2><p class="instruction">{esc(case["instruction"])}</p><p class="meta">task粒度 · episode{case["episode"]} · 当前{case["anchor"]} → 目标{case["goal"]} · {case["id"]}</p><p class="masks">{esc(masktext)}</p><div class="inputs">{inputs}</div><div class="predictions">{"".join(generated)}</div><details><summary>在相同目标背景查看两套TCP预测</summary><div class="overlays">{"".join(overlay)}</div></details>{table}</article>')
        evidence.append(dict(**case, models=measured))
    def summary_rows():
        rows = []
        for arch, name in [('G', '新G · 5515步'), ('Cstar', '旧C* · 3321步')]:
            model = inference['models'][arch]; region = model['region_errors']
            rows.append('<tr>' + ''.join('<td>' + s + '</td>' for s in [name, '5 / 7', '10 / 14',
                number(model['metrics']['head_image_mae']), number(model['metrics']['tcp_px'], 2),
                number(region['full']['copy_mae']), number(region['changed']['model_mae']),
                number(region['changed']['gain_over_copy']), number(region['tcp']['model_mae']),
                number(region['tcp']['gain_over_copy'])]) + '</tr>')
        return ''.join(rows)
    gains = {arch: inference['models'][arch]['region_errors']['changed']['gain_over_copy'] for arch in ('G', 'Cstar')}
    unchanged_cstar = sum(c['models']['Cstar']['prediction_change_mae'] == 0
                         and c['models']['Cstar']['tcp_uv_delta_max'] == 0 for c in evidence)
    unchanged_g_uv = sum(c['models']['G']['tcp_uv_delta_max'] == 0 for c in evidence)
    quality = '；'.join(f'{name}变化区域相对复制改进为{number(gains[arch])}' for arch, name in [('G', '新G'), ('Cstar', '旧C*')])
    conclusion = f'本次修复解决标签漏监督和G零latent占位，不会补回历史训练中缺失的监督。{quality}。正值代表优于复制当前帧，负值代表仍不及复制；不能把标签恢复等同于模型已经学会任务。'
    css = '''*{box-sizing:border-box}body{margin:0;background:#f2f5f9;color:#17283d;font:15px/1.65 system-ui,"Microsoft Yahei",sans-serif}main{max-width:1480px;margin:auto;padding:24px}header,section,article{background:white;border:1px solid #dbe3ee;border-radius:14px;padding:24px;margin:0 0 20px}h1{font-size:30px;line-height:1.35}h2{font-size:22px;margin:0 0 10px}.lead{font-size:18px}.muted,.meta{color:#617187}.badge{display:inline-block;padding:4px 12px;background:#d9f7e8;border-radius:20px}.warning{border-left:4px solid #dc9600;padding:12px;background:#fff8df}.inputs,.predictions,.overlays{display:grid;gap:12px}.inputs{grid-template-columns:repeat(3,minmax(0,1fr))}.predictions{grid-template-columns:repeat(4,minmax(0,1fr));margin-top:16px}.overlays{grid-template-columns:repeat(2,minmax(0,1fr))}figure{margin:0;border:1px solid #dbe3ee;border-radius:10px;overflow:hidden;background:#f8fafc;min-width:0}figcaption{font-weight:650;padding:9px 10px}svg{display:block;width:100%;height:auto;background:#111}svg text{font:11px system-ui,sans-serif;paint-order:stroke;stroke:#17202d;stroke-width:2px}figure p{font-size:12px;padding:0 10px;min-height:42px;color:#546479}.instruction{font-size:16px;overflow-wrap:anywhere}.meta,.masks{font-size:13px;overflow-wrap:anywhere}.masks{background:#edf4ff;padding:10px;border-radius:7px}.table-wrap{overflow:auto;max-width:100%}table{width:100%;border-collapse:collapse;white-space:nowrap;font-variant-numeric:tabular-nums}th,td{padding:10px 12px;border-bottom:1px solid #e2e8f0;text-align:left;font-size:13px}th{background:#f3f6fa}details{padding:12px 0}summary{cursor:pointer;color:#1661a2}button{padding:8px 12px;border:1px solid #adc2dc;background:#edf4ff;border-radius:6px;cursor:pointer}.controls{display:flex;flex-wrap:wrap;gap:16px;align-items:center}.hide-truth .truth,.hide-pred .pred{display:none}a{color:#1364ab}code{overflow-wrap:anywhere}.flows{display:grid;grid-template-columns:1fr 1fr;gap:14px}.flow{background:#edf4ff;padding:16px;border-radius:10px}footer{font-size:12px;color:#617187}@media(max-width:900px){.predictions{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:580px){main{padding:10px}header,section,article{padding:15px;border-radius:10px}h1{font-size:25px}.inputs,.predictions,.overlays,.flows{grid-template-columns:1fr}.lead{font-size:16px}}'''
    document = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Stage1目标有效性修复 · DSW七样本前后对比</title><style>{css}</style></head><body><main>
<header><span class="badge">DSW实测 · 7条样本 · 未训练 · 未启动DLC</span><h1>Stage1目标有效性修复<br>G5515 / C*3321 同样本前后对比</h1><p class="lead">5条Wipe恢复RGB与10个有效TCP点；另外2条canonical无效样本继续不计分。目标帧、权重、当前帧、语言和因果历史均保持不变。</p><p class="warning">这7条均为task粒度：5条Wipe来自同一episode、同一目标3726帧、5个不同当前帧；另2条来自episode3、目标4814帧。本页是有界拟合诊断，不代表独立测试集或subtask整体效果。</p><p>{esc(conclusion)}</p></header>
<section><h2>改动边界</h2><div class="flows"><div class="flow"><b>修复前</b><p>canonical ∧ training.valid → Stage1目标mask<br>末帧没有下一步action → RGB与TCP被整体屏蔽<br>G生成复用GT mask → 全程零latent → 解码占位</p></div><div class="flow"><b>修复后</b><p>已确认Genrobot语义：canonical ∧ 来源帧标志 ∧ 目标相机有效 → RGB监督<br>EEF、标定、投影、画内检查 → 独立TCP监督<br>当前视图 → G生成mask；GT mask仅参与loss与计分</p></div></div><p>Stage2仍使用原source_frame_validity与action.mask；没有下一步action的末帧仍无效。未知来源语义仍沿用旧严格规则。success_verified仍为False；可读终点图不证明指令执行成功。</p></section>
<section><h2>新协议计分结果（仅5条有效样本）</h2><p>旧协议7条均无有效GT，原始MAE/TCP都不可计分，因此不存在可靠的“旧分数→新分数”提升率。两套模型训练步数不同；原生训练loss不能跨架构直接比较。</p><p><b>同一DSW控制对照：</b>前后均使用同权重、同输入、同随机种子。C*有{unchanged_cstar}/7条图像浮点输出及TCP完全一致；G有{unchanged_g_uv}/7条TCP完全一致，图像则从旧零latent占位变为真实生成。这证明本次没有给旧C*或G的TCP制造学习收益。历史DLC输出与评分仍单独保留。</p><div class="table-wrap"><table><thead><tr><th>检查点</th><th>有效图像</th><th>有效TCP点</th><th>全图MAE↓</th><th>TCP像素↓</th><th>复制全图MAE</th><th>变化区域MAE↓</th><th>变化区域改进↑</th><th>TCP邻域MAE↓</th><th>TCP邻域改进↑</th></tr></thead><tbody>{summary_rows()}</tbody></table></div><p class="muted">RGB范围0–1。TCP误差在训练头图320×256上计算：‖[319Δu,255Δv]‖₂。变化区域为当前帧与GT平均通道差&gt;0.1；TCP邻域为有效GT点半径24像素圆并集。“改进”=复制MAE−预测MAE，为绝对差值。G拼图为320×384，表内只计头图320×256。</p></section>
<section class="controls"><b>叠加图例</b><label><input id="truth" type="checkbox" checked> <span style="color:#078367">○ 绿色：GT TCP</span></label><label><input id="pred" type="checkbox" checked> <span style="color:#ac6500">× 橙色：模型TCP</span></label><button data-filter="all">全部7条</button><button data-filter="true">恢复有效的5条</button><button data-filter="false">仍无效的2条</button><span id="count">7条样本</span></section>
{''.join(cards)}
<section><h2>可复核证据与后续</h2><p>新标签与prepared目录单独版本化；G多视图和C*头图各自独立经过原VAE编码，未裁剪G latent充当C*目标。历史缓存与原prepared文件均保留。旧输出按新GT补算的区域误差单独存入comparison.json，不覆盖原评分。</p><p>本次未加载优化器、未写入检查点、未启动DLC。后续若继续训练，需要明确的新contract与标签迁移记录，保留原权重、优化器、步数及历史评分；不能将本页新标签直接塞回旧run绕过严格校验。</p><p><a href="labels.json">逐样本源mask与帧号</a> · <a href="inference.json">检查点SHA、代码身份与推理结果</a> · <a href="comparison.json">前后输出与补算误差</a> · <a href="https://github.com/shuluoshu/OpenWAM-scale/tree/codex/stage1-goal-mask-fix-20261118">实现分支</a></p></section><footer>新G：5515步；旧C*：3321步。20步采样；G固定KF1时间条件0。橙色TCP始终表示模型二维输出，不当作物理标定投影。图外预测不会被悄悄移动到边界。</footer></main>
<script>document.getElementById('truth').onchange=e=>document.body.classList.toggle('hide-truth',!e.target.checked);document.getElementById('pred').onchange=e=>document.body.classList.toggle('hide-pred',!e.target.checked);document.querySelectorAll('[data-filter]').forEach(b=>b.onclick=()=>{{let n=0;document.querySelectorAll('.case').forEach(c=>{{c.hidden=b.dataset.filter!=='all'&&b.dataset.filter!==c.dataset.valid;if(!c.hidden)n++;}});document.getElementById('count').textContent=n+'条样本';}});</script></body></html>'''
    (out / 'index.html').write_text(document)
    (out / 'comparison.json').write_text(json.dumps(dict(cases=evidence), ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(page=str(out / 'index.html'), cases=len(cards), recovered=sum(c['new_image_valid'] for c in labels['cases']),
                         conclusion=conclusion), ensure_ascii=False))


if __name__ == '__main__': main()
