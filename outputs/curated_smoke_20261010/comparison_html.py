"""Build file:// compatible paired evidence cards from completed artifacts."""
from pathlib import Path
import json, html

OUT = Path(__file__).resolve().parent
def read(p): return json.loads(p.read_text())
def esc(v): return html.escape(str(v), quote=True)
def link(p, label): return f'<a href="{esc(p)}">{esc(label)}</a>'
def dims(v): return '×'.join(map(str, v))
def table(headers, rows):
    return '<div class="scroll"><table><tr>'+''.join(f'<th>{esc(x)}</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join(f'<td>{x}</td>' for x in row)+'</tr>' for row in rows)+'</table></div>'

def build_comparison_html():
    base = OUT/'comparison'
    manifest = read(OUT/'manifest.json')['results']
    gt = {x['id']:x for x in read(base/'gt_results.json')['results']}
    rendered = {x['id']:x for x in read(base/'render_results.json')}
    images = {r['id']:read(base/r['id']/'image_metrics.json') for r in manifest}
    forward = {x['id']:x for x in read(base/'forward_input_results.json')}
    assert len(forward)==46 and all(x['display_paste_verified_without_resize'] for x in forward.values())
    assert len(gt)==len(rendered)==len(images)==46
    assert all(x['status']=='ok' for x in images.values())
    cameras = [c for x in images.values() for c in x['cameras'].values()]
    assert all(c['mora_timestamp_decoder_identical'] and c['openwam_slot_max_abs_error_uint8']==0 and c['roundtrip_max_abs_error']==0 for c in cameras)
    assert sum(r['plot'] for r in rendered.values())==11
    assert sum(r['video'] for r in rendered.values())==7
    odd = sum(not r['mora_anchor_sampling']['natural_global_stride2_eligible'] for r in gt.values())
    maxclip = max(r['ow']['clipped_xyz_max_m'] for r in gt.values())
    normerr = max(max(v or 0 for v in r['mora']['external_stats']['inverse_roundtrip_max_abs'].values()) for r in gt.values() if r['mora']['status']=='executed')
    parts = [
      '<section class="panel" id="paired-visuals"><h2>⑦ 同一源帧：两边处理后的图像与轨迹</h2>',
      f'<p><strong>46 组同帧图像 / {len(cameras)} 路相机；11 组 relative 轨迹对比；7 段头相机投影对比动画。</strong>保留页面下方原有 46 段滑动锚点视频。这里固定同一个 source anchor，分别执行两边现有处理路径。</p>',
      '<p>图像按“原始源帧 → Mora 实际 processor 输出 → OpenWAM 实际槽位”排列。每张 Mora 图独立进入 patch/token；横向拼图只是报告展示，不是 Mora 模型的输入 canvas。</p>',
      f'<p class="warn">原 Mora YAML 与当前 checkout 有参数兼容问题，详见上方分析。这里实际执行的是隔离图像 transforms / processor，以及 EEF-only 96D composer；没有执行完整 Mora loader、joint 混合监督或训练。{odd} 个奇数 anchor 为控制同帧而强制指定，不是其 15 Hz sampler 自然选出的起点。</p>',
      '<p>轨迹色标：<strong style="color:#69dbe9">青色 Mora 原始 canonical EEF</strong>；<strong style="color:#ffbf63">橙色 OpenWAM 明确 TCP</strong>；<strong style="color:#ff657d">粉色 OpenWAM quantile 裁剪后反归一化目标</strong>。实线加深区间按目标源帧编号匹配，不按 action 数组下标强配。坐标轴红 X / 绿 Y / 蓝 Z。</p>',
      f'<p>所有相机均验证两边解码 RGB 一致；OpenWAM 槽位逐像素一致，Mora normalized patch 重组误差为 0。Mora 发布 stats 经真实 composer 再逆变换，最大数值误差 {normerr:.3g}；OpenWAM 首窗口裁剪导致的最大 XYZ 分量变化为 {maxclip*1000:.3f} mm。</p>',
      '<p class="muted">Mora bicubic 可能产生超出 [0,1] 的浮点 RGB。PNG 展示执行 clip + uint8 舍入，实际张量范围保存在每条证据中；不可用 PNG 代替原浮点模型输入。轨迹展示则使用物理单位，无截图估算。</p>',
      '<div id="forward-inputs"><h3>新增：forward 前最终图像组织对比</h3><p><strong>OpenWAM 先空间拼接；MoraVLA 不空间拼接。</strong>每条样例顶部新增最终输入对比图：左边是处理后完整 canvas，右边是 Mora 各相机实际 processor 输出的像素还原。Mora 最后执行 <code>torch.cat(pixel_values)</code>，拼的是 patch 序列，保留每张图各自的 <code>image_grid_thw</code>。</p><p>这里展示可解释的图像输入边界：OpenWAM 为 bf16 归一化后的 pre-VAE 像素还原，Mora 为 normalized patch 的逆变换。OpenWAM 去噪网络 forward 还涉及 VAE latent 和加噪，本报告没有运行这些步骤；图像也不是模型预测。右侧相机之间的间隔是报告排版，不是训练 padding。</p></div>',
      '<label for="pairSelect">选择同帧样例：</label><select id="pairSelect">'+''.join(f'<option value="pair_{esc(r["id"])}"'+(' selected' if r['id']=='20_interndata_a1' else '')+f'>{esc(r["id"])} · episode {r["episode"]} · anchor {r["start"]} · '+('Head + local' if rendered[r['id']]['video'] else 'local / head masked' if rendered[r['id']]['plot'] else 'pose masked')+'</option>' for r in manifest)+'</select>',
      '<p><small>无 TCP 的条目仍提供同帧图像及两边 GT/mask 证据，不补造可对比的 TCP 曲线。</small></p>'
    ]
    for r in manifest:
        ident=r['id']; im=images[ident]; e=gt[ident]; m=rendered[ident]; p=f'comparison/{ident}/'; mora=e['mora']; ow=e['ow']
        parts += [f'<article class="pair-card" id="pair_{esc(ident)}"'+('' if ident=='20_interndata_a1' else ' hidden')+f'><h3>{esc(ident)} · episode {r["episode"]} · source anchor {r["start"]}</h3>',
          f'<p class="muted">{esc(r["repo"])} · Mora 自然 stride=2 起点：'+('是' if e['mora_anchor_sampling']['natural_global_stride2_eligible'] else '否，本对照强制固定同帧')+'</p>']
        f=forward[ident]
        parts += [f'<h4>最终输入：OpenWAM 完整拼接图 ↔ Mora 独立相机图像</h4><a href="{p}forward_input_comparison.png"><img style="width:100%;height:auto" loading="lazy" src="{p}forward_input_comparison.png" alt="{esc(ident)} forward 前完整 canvas 与 Mora 独立图像对比"></a>',
          '<p>左侧：head 在上、左右 wrist 在下，缺失腕相机保留黑槽；图像从实际 bf16 像素预处理输出还原。右侧：当前时刻各有效视角分别处理，缺视角省略；并排排列仅用于展示。</p>',
          table(['最终组织','OpenWAM','MoraVLA'],[
            ['空间图像','一张 320×384 canvas / 时刻','独立相机图像，不做 RGB 大图拼接'],
            ['视觉张量','[1,3,9,384,320]，bf16，B/C/T/H/W',esc(f['mora']['concatenated_patch_shape'])+' pixel_values；'+esc(f['mora']['image_grid_thw_shape'])+' image_grid_thw（单样本）'],
            ['时序','图中为 t+0；训练视频还有 t+4…t+32，当前帧作条件','当前帧；processor 的 temporal patch 复制不是历史帧'],
            ['展示量化',f'bf16 逆变换再 uint8 舍入，相对 reader PNG 最大差 {f["openwam"]["inverse_bf16_display_max_difference_uint8"]} / 255','实际 normalized patch 逆变换；RGB clip 后保存 PNG']]),
          '<p>'+' · '.join([link(p+'forward_input_comparison.png','打开完整对比 PNG'),link(p+'forward_openwam_canvas.png','单独查看 OpenWAM 最终 canvas'),link(p+'forward_input_metrics.json','最终输入形状与验证'),link('comparison/evidence/mora__qwen_processor_transform.py','Mora patch 串接源码')])+'</p>',
          '<details><summary>展开 OpenWAM 完整 9 帧训练视频输入（无轨迹 overlay）</summary><div class="gallery">'+''.join(f'<figure><a href="clips/{ident}/model_frame_{k}.png"><img loading="lazy" src="clips/{ident}/model_frame_{k}.png"></a><figcaption>t+{k*4} · 源帧 {r["start"]+k*4}<br>reader canvas，归一化前 uint8</figcaption></figure>' for k in range(9))+'</div><p>t+0 为当前条件，其余为训练未来视频目标；不能把这 9 帧称为推理历史输入。</p></details>',
          '<h4>处理过程对照：源帧 → Mora 最终图像 → OpenWAM 各槽位</h4>',
          f'<a href="{p}image_comparison.jpg"><img style="width:100%;height:auto" loading="lazy" src="{p}image_comparison.jpg" alt="{esc(ident)} source Mora OpenWAM 同帧图像对照"></a>',
          '<details><summary>展开各相机尺寸、实际张量范围与无损 PNG</summary>']
        rows=[]
        for side,c in im['cameras'].items():
            files=c['files']; rows.append([esc(side),str(c['source_anchor']['episode_frame']),dims(c['native_size_wh']),dims(c['stage1_size_wh'])+' → '+dims(c['final_size_wh']),dims(c['openwam_slot_size_wh']),esc(c['grid_thw'])+f' / {c["visual_token_count"]}',esc([round(v,5) for v in c['normalized_range']]),' · '.join(link(p+files[k],l) for k,l in [('source','源 PNG'),('stage1','Mora resize'),('final','Mora final')])+ ' · '+link(f'clips/{ident}/resized_{side}.png','Open 槽位')])
        parts += [table(['相机','源帧','原始 W×H','Mora stage1 → final','Open W×H','grid THW / tokens','Mora normalized 范围','查看原文件'],rows),
          '<p>缺失 Mora views：'+esc(im['missing_views_omitted_in_mora'])+'。OpenWAM 实际首帧 canvas：'+link(f'clips/{ident}/model_frame_0.png','320×384 PNG')+'。所有尺寸均为宽×高，图表 grid 为 T/H/W。</p></details>']
        target=f'{mora.get("target_first")}…{mora.get("target_last")}' if mora.get('target_first') is not None else '无可核验 target frame 字段，不推定 +3'
        rows=[['GT 端点',esc(mora.get('endpoint','无源 EEF')),esc(ow['endpoint'])],
          ['读取行 / 目标源帧',f'action 行 {r["start"]}…{r["start"]+49}；目标 {target}',f'state {r["start"]+1}…{r["start"]+32}'],
          ['local / head 有效 pose 数',f'{mora.get("local_valid_pose_steps",0)} / {mora.get("head_valid_pose_steps",0)}（最多100=50步×2臂）',f'{ow["local_valid_pose_steps"]} / {ow["head_valid_pose_steps"]}（最多64=32步×2臂）'],
          ['归一化对照','显式发布 EEF stats；std 缩放、不 clip' if mora['status']=='executed' else '缺 EEF，未执行',f'片段 smoke quantile；XYZ 最大分量裁剪变化 {ow["clipped_xyz_max_m"]*1000:.3f} mm']]
        parts += [table(['项目','MoraVLA','OpenWAM'],rows)]
        if m['plot']:
            dist=m.get('same_time_endpoint_distance_m')
            parts += [f'<p>共同目标源帧：{m["shared_target_count"]} 个。'+(f'同一时刻的世界系 EEF/TCP 端点距离：平均 {dist["mean"]*1000:.3f} mm，最大 {dist["max"]*1000:.3f} mm。<strong>这是不同端点的距离，不是预测误差，也不能作为标定误差指标。</strong>' if dist else '')+'</p>']
            if m['video']:
                parts += [f'<h4>固定当前头相机：处理后的图像 + 恢复绝对位置后的轨迹</h4><video controls preload="none" playsinline poster="{p}gt_projection_compare.jpg"><source src="{p}gt_projection_compare.mp4" type="video/mp4"></video>',
                  f'<p><small>{m["video_frames"]} 帧 / {m["video_duration_s"]} 秒；固定观测图像，逐步展开未来目标。播放速度仅为展示，并非源视频实时回放。{link(p+"gt_projection_compare.mp4","下载对比 MP4")}</small></p>']
            else:
                parts += ['<p class="warn">此条头相机未通过 strict calibration eligibility，头相机投影与 head 分支保持 MASKED。存在数值矩阵不代表允许可信投影。</p>']
            parts += [f'<h4>Relative pose：上排当前 head axes；下排各自当前端点 local axes</h4><a href="{p}gt_comparison.png"><img style="width:100%;height:auto" loading="lazy" src="{p}gt_comparison.png" alt="{esc(ident)} relative EEF TCP trajectories"></a>',
              '<p>head 两边方向基底相同，但各自减去当前端点的位置；local 分别以 Mora EEF / OpenWAM TCP 为原点及方向基底。原点都画零，不意味着它们是同一物理点。显示曲线使用报告侧几何/source-mask 与可信相机检查；实际 composer loss mask 单独存于 NPZ，不能把可画曲线等同于实际有效 loss 槽。图中姿态轴是恢复后的未来朝向，旋转来自矩阵乘法。Mora 归一化后再还原曲线在数值精度内重合，因此不重复画线。</p>']
        else:
            parts += ['<p class="warn"><strong>本条不画伪造的双侧 TCP 对照：</strong>'+esc(ow['tcp_reason'])+'。Mora EEF 状态：'+esc(mora['status'])+'；完整数值和 mask 仍保留在证据文件。</p>']
        parts += ['<p>'+' · '.join([link(p+'image_metrics.json','图像证据 / hash'),link(p+'gt_evidence.json','GT / stats / mask 证据'),link(p+'gt_trajectories.npz','两边实际数组 NPZ'),link('#'+ident,'查看原有滑动锚点视频')]+([link(p+'gt_render_metrics.json','投影指标')] if m['plot'] else []))+'</p></article>']
    parts += ['<p>生成脚本：'+ ' · '.join(link('comparison/'+n,l) for n,l in [('prepare_images.py','真实图像 transforms'),('prepare_gt.py','真实 GT / composer'),('render_pairs.py','轨迹绘图 / 投影'),('image_results.json','图像汇总'),('gt_results.json','GT 汇总'),('render_results.json','渲染汇总')])+'</p></section>',
      '''<script>(function(){const select=document.getElementById('pairSelect');const cards=[...document.querySelectorAll('.pair-card')];function show(id){if(!cards.some(c=>c.id===id))return;select.value=id;cards.forEach(c=>{c.hidden=c.id!==id;if(c.hidden)c.querySelectorAll('video').forEach(v=>v.pause())})}select.addEventListener('change',()=>show(select.value));function hash(){const id=decodeURIComponent(location.hash.slice(1));if(id.startsWith('pair_'))show(id)}window.addEventListener('hashchange',hash);hash()})();</script>''']
    wrist=(base/'abc_wrist.html').read_text() if (base/'abc_wrist.html').exists() else ''
    return (base/'analysis.html').read_text()+'\n'+wrist+'\n'+'\n'.join(parts)
