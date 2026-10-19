"""Self-contained data-only subtask review page, no generated model predictions."""
import html
import json
from pathlib import Path


def render_report(out,summary,episodes,records):
    esc=lambda x:html.escape(str(x),quote=True)
    groups={e['id']:[] for e in episodes}
    for row in records:groups[row['episode_id']].append(row)
    families=summary['families'];articles=[]
    for ep in episodes:
        panels=[]
        for r in groups[ep['id']]:
            valid=r['valid_points'];p=r['provenance'];a=r['instruction_start'];b=r['instruction_end_exclusive'];fps=r['fps']
            note='TCP mask 左/右：'+str(valid)+'；可见性：未知，投影有效不等于可见。'
            reasons=p['missing_reasons']
            if not all(valid):note+=' 缺失/无效项：'+'; '.join(reasons or ['点位于画面外或跟踪无效'])
            def fig(title,path):return f'<figure><figcaption>{esc(title)}</figcaption><img loading="lazy" width="320" height="256" src="{esc(path)}"></figure>'
            figures=fig(f'输入：subtask 起始帧 {a}',r['observed_preview'])+fig(f'目标候选：区间末帧 {b-1} + TCP GT',r['goal_overlay'])+fig(f'对照：episode 末帧 {ep["length"]-1}',ep['terminal'])
            detail=dict(source_repo=r['source_repo'],source_episode=r['source_episode'],subtask_index=r['subtask_index'],
                contiguous_occurrence=r['segment_number'],interval=[a,b],condition=r['instruction'],parent_task=r['parent_task'],
                parent_task_used_as_condition=False,goal_source_frame=r['goal_source_frame'],success_verified=False,
                valid_points=valid,target_uv_px=r['target_uv_px'],provenance=p)
            panels.append(f'<section class="sample" data-search="{esc(r["instruction"].lower())}"><h3>连续片段 {r["segment_number"]} · subtask index {r["subtask_index"]} · [{a}, {b}) · {(b-a)/fps:.2f}s</h3>'
                f'<p class="instruction">{esc(r["instruction"])}</p><p class="mask">{esc(note)}</p><div class="panels">{figures}</div>'
                f'<div class="buttons"><button data-video="v-{esc(ep["id"])}" data-start="{a/fps}" data-end="{b/fps}">播放这个 subtask</button>'
                f'<button data-video="v-{esc(ep["id"])}" data-start="{max(a,b-int(fps))/fps}" data-end="{min(ep["length"],b+int(fps))/fps}">查看边界前后 1 秒</button>'
                f'<a href="{esc(r["goal_preview"])}">无标记目标</a><a href="{esc(r["model_input_npz"])}">实际送模张量 NPZ</a></div>'
                f'<details><summary>源字段、坐标变换、帧对齐与 mask 证据</summary><pre>{esc(json.dumps(detail,ensure_ascii=False,indent=2))}</pre></details></section>')
        excluded=json.dumps(ep['excluded_segments'],ensure_ascii=False,indent=2)
        qa=ep.get('calibration_eligibility');qanote=''
        if qa is not None:qanote=f'<p class="warning">源相机 QA：{esc(qa.get("broadcast_qa_status"))}。被拒绝的标定保持 TCP 投影 mask，不把数值可计算当成可用 GT。</p>'
        articles.append(f'<article data-family="{esc(ep["family"])}" data-episode="{esc(ep["id"])}"><h2>{esc(ep["family"])} · {esc(ep["id"])} · episode {ep["episode"]}</h2>'
            f'<p>父任务（仅对照，不送入模型）：{esc(" | ".join(ep["task_texts"]))}</p>{qanote}'
            f'<video id="v-{esc(ep["id"])}" controls preload="none" poster="{esc(ep["initial"])}" src="{esc(ep["playback"])}"></video>'
            '<p class="muted">完整 episode 回放；点击片段按钮会定位区间。视频不叠加动态 TCP；下方目标图显示同一目标帧的投影。</p>'
            f'<details><summary>排除的空标注／过短区间：{len(ep["excluded_segments"])} 个</summary><pre>{esc(excluded)}</pre></details>'+''.join(panels)+'</article>')
    missing=''.join(f'<li>{esc(x["family"])}：当前提供的标注范围未覆盖；未用 task 文本冒充 subtask。检查路径：<code>{esc(x["checked_file"])}</code></li>' for x in summary['excluded_families'])
    page='''<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Subtask · 送模图像与 TCP GT</title><style>
*{box-sizing:border-box}body{background:#f3f5f1;color:#233b32;font:16px/1.6 system-ui;margin:0}header{padding:30px max(24px,calc((100vw - 1100px)/2));background:#143d32;color:white}h1{font-size:30px;margin:0}header a{color:#d0ecaa}.toolbar{position:sticky;top:0;background:#e8eee4;padding:12px 24px;z-index:3;display:flex;gap:12px;flex-wrap:wrap}input,select,button{font:inherit;padding:7px 12px;border:1px solid #a4b9aa;border-radius:6px;background:white}main{max-width:1120px;margin:auto;padding:24px}article{margin:20px 0 50px}video{width:640px;max-width:100%;max-height:480px;background:#111}.sample{background:white;border:1px solid #ced9cb;border-radius:12px;padding:18px;margin:20px 0}.instruction{font-size:20px;font-weight:650;color:#245b41}.panels{display:flex;gap:12px;overflow:auto}figure{margin:0;flex:0 0 320px}figcaption{font-size:13px;padding:8px 0}img{display:block;width:320px;height:256px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}details{margin:14px 0}.mask,.muted{font-size:13px;color:#536957}.warning{background:#fff1cc;padding:12px}.buttons{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin-top:12px}a{color:#236b4f}code{overflow-wrap:anywhere}li{margin:10px 0}[hidden]{display:none!important}
</style></head><body>'''
    page+=f'<header><h1>Subtask 指令 → 局部目标图像 + TCP</h1><p>{len(families)} 类 · {len(episodes)} 个 episode · {len(records)} 个连续子任务样例</p><p>仅数据适配与可视化。没有新训练，也没有模型预测图。目标是 subtask 区间末帧候选，仍需审核动作完成、回撤与边界是否准确。</p><a href="manifest.jsonl">数据清单 JSONL</a> · <a href="config.yaml">适配配置 YAML</a> · <a href="validation.json">送模张量检查</a></header>'
    options=''.join(f'<option value="{esc(x)}">{esc(x)}</option>' for x in families)
    page+=f'<div class="toolbar"><label>数据集 <select id="family"><option value="">全部</option>{options}</select></label><label>搜索子任务 <input id="search" placeholder="例如 stamp / place / close"></label><span id="count"></span></div>'
    page+='<main><p>图像为实际 BF16 pre-VAE 张量的显示结果（320×256，无裁剪/增强）；TCP 为同一目标帧投影，左红、右蓝方框。父任务只用于对照。重复的 subtask index 按连续出现区间分别保留，未标注区间排除。</p><p>保留参考对话的 12 个 episode，另加入当前实验中相机 QA 通过的 ABC episode 及具有 subtask 标注的 Morphi episode；未自动按清晰度挑选或改动目标帧。</p>'+''.join(articles)+f'<details><summary>未纳入本轮 subtask 样例的数据集</summary><ul>{missing}</ul></details></main>'
    page+='''<script>
const family=document.getElementById('family'),search=document.getElementById('search');
function filter(){let visible=0;document.querySelectorAll('article').forEach(article=>{const match=!family.value||article.dataset.family===family.value;let n=0;article.querySelectorAll('.sample').forEach(sample=>{const keep=match&&sample.dataset.search.includes(search.value.trim().toLowerCase());sample.hidden=!keep;if(keep)n++;});article.hidden=!n;visible+=n;});document.getElementById('count').textContent=`显示 ${visible} 个子任务`;}
family.addEventListener('change',filter);search.addEventListener('input',filter);filter();
document.querySelectorAll('button[data-video]').forEach(button=>button.addEventListener('click',()=>{document.querySelectorAll('video').forEach(v=>v.pause());const v=document.getElementById(button.dataset.video);v.dataset.stop=button.dataset.end;v.currentTime=Number(button.dataset.start);v.play().catch(()=>{});v.scrollIntoView({behavior:'smooth',block:'center'});}));
document.querySelectorAll('video').forEach(v=>v.addEventListener('timeupdate',()=>{if(v.dataset.stop&&v.currentTime>=Number(v.dataset.stop)){v.pause();delete v.dataset.stop;}}));
</script></body></html>'''
    (out/'index.html').write_text(page)
