"""Self-contained HTML content, relative MP4/image assets, no CDN/fetch dependency."""
from pathlib import Path
import json,html,collections,hashlib,subprocess,shutil
from PIL import Image
from comparison_html import build_comparison_html
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
def esc(x):return html.escape(str(x),quote=True)
def load(p):return json.loads(p.read_text())
def link(url,label):return f'<a href="{esc(url)}">{esc(label)}</a>'
def fmt(v):return 'MASK' if v is None else '['+', '.join('—' if x is None else f'{x:.5f}' for x in v)+']'
def table(headers,rows):return '<div class="scroll"><table><thead><tr>'+''.join(f'<th>{esc(h)}</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{c}</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table></div>'
manifest=load(OUT/'manifest.json');audit=load(OUT/'source_contract_audit.json');results=manifest['results'];assert len(results)==46 and all(r['status']=='ok' for r in results)
for r in results:
 d=OUT/'clips'/r['id'];r['input']=load(d/'input_audit.json');r['render']=load(d/'render_metrics.json');r['config']=(d/'data.yaml').read_text();r['source_contract']=next(x for x in audit['repositories'] if x['repo']==r['repo'])
head=sum(r['input']['head_supervised'] for r in results);tcp=sum(r['input']['tcp_local_supervised'] for r in results);joint=sum(bool(r['input']['joint_supervised_dimensions']) for r in results)
maxpos=max(r['render']['source_pose_roundtrip_max_position_m'] or 0 for r in results);maxrot=max(r['render']['source_pose_roundtrip_max_rotation_matrix_error'] or 0 for r in results)
provenance={'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'branch':subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),'tracked_worktree_changes':subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).splitlines(),'source_yaml_sha256':manifest['yaml_sha256'],'generated_utc':subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip(),'executed':'production reader, relative stats, CPU pixel preprocess, projection render; no model forward / no training'}
evidence=OUT/'evidence';evidence.mkdir(exist_ok=True)
source_files=['openwam/dataloader/bases/lerobot_v3_reader.py','openwam/dataloader/mora_relative_lerobot.py','openwam/dataloader/utils/mora_relative.py','openwam/dataloader/utils/video_io.py','openwam/dataloader/transforms/multiview.py','openwam/dataloader/utils/normalization.py','openwam/model/video_backbone/wan/preprocess.py','scripts/data/prepare_openwam_relative_stats.py','tests/dataloader/test_lerobot_prompt_schema.py']
for name in source_files:
 p=ROOT/name;target=evidence/name.replace('/','__');target.write_bytes(p.read_bytes());provenance.setdefault('code_sha256',{})[name]=hashlib.sha256(p.read_bytes()).hexdigest()
shutil.copyfile(manifest['yaml'],evidence/'pretrain_v0_3_curated_smoke.yaml')
(OUT/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2))
style='''*{box-sizing:border-box}body{margin:0;background:#101722;color:#e0e8f3;font:15px/1.6 system-ui,-apple-system,"Microsoft YaHei",sans-serif}main{max-width:1340px;margin:auto;padding:28px}h1{font-size:30px;line-height:1.3}h2{font-size:23px;margin-top:34px}h3{font-size:19px}a{color:#84c6ff}code,pre{font:13px/1.5 ui-monospace,monospace}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#111c2b;padding:16px;border-radius:8px;max-height:500px;overflow:auto}small,.muted{color:#9caec3}.panel,.clip{background:#192333;border:1px solid #304055;border-radius:14px;padding:22px;margin:18px 0}.stats{display:flex;gap:12px;flex-wrap:wrap}.stat{flex:1;min-width:145px;padding:16px;background:#22334b;border-radius:12px}.stat strong{font-size:30px;display:block;color:#8cded0}.tag{display:inline-block;border-radius:20px;padding:2px 10px;background:#26394e;font-size:12px;margin:2px}.yes{color:#99efc9}.no{color:#e6bd78}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border-bottom:1px solid #344153;text-align:left;padding:9px;vertical-align:top}th{background:#233249}.scroll{overflow:auto}video{width:100%;max-height:830px;background:#0c1420;border-radius:8px}details{margin-top:13px}summary{cursor:pointer;font-weight:600;color:#afd2ff}.gallery{display:flex;overflow:auto;gap:12px;padding:10px 0}.gallery figure{margin:0;flex:0 0 190px}.gallery img{width:100%;height:auto;border-radius:5px}.gallery figcaption{font-size:12px;color:#a9b9cc}.filter{position:sticky;top:0;padding:12px;background:#152236ef;backdrop-filter:blur(10px);z-index:3;border-bottom:1px solid #4c5b70}select,input,button{background:#24364d;color:white;border:1px solid #4a6486;border-radius:6px;padding:8px;margin:3px;max-width:100%}.formula{border-left:3px solid #78d6cf;padding-left:18px}.warn{border-left:3px solid #e5b267;padding-left:16px}.twocol{display:grid;grid-template-columns:1fr 1fr;gap:18px}.flow{display:flex;gap:8px;align-items:stretch;flex-wrap:wrap}.node{flex:1;min-width:170px;background:#23364f;border-radius:10px;padding:14px}.arrow{align-self:center;color:#72d5cd;font-size:22px}.view-map{width:320px;height:384px;display:grid;grid-template-rows:256px 128px;grid-template-columns:160px 160px;border:1px solid #89a5c4;margin:auto}.view-map div{display:flex;align-items:center;justify-content:center;text-align:center;border:1px dashed #7295b8}.view-map .head{grid-column:span 2;background:#213e52}.view-map .left{background:#45452d}.view-map .right{background:#47344a}@media(max-width:760px){main{padding:14px}.twocol{grid-template-columns:1fr}.panel,.clip{padding:14px}h1{font-size:25px}}'''
parts=[f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OpenWAM · curated smoke 输入与 relative TCP 审计</title><style>{style}</style><main>',
'<h1>Curated smoke：实际输入处理与 relative TCP 轨迹</h1>',
f'<p>2026-10-10 · <code>{esc(provenance["branch"])}</code> · HEAD <code>{provenance["commit"][:7]}</code> + 当前工作区。使用当前 112D reader，保留 32 步未来目标。</p>',
'<p class="warn"><strong>本报告已执行数据 reader、GT 转换、像素预处理和视频渲染；未启动训练、未执行 VAE / 模型 forward。</strong>原有视频为裁剪前物理 GT；新增对比中粉色标记为裁剪后反归一化目标。全部是 GT，不是模型预测或过拟合结果。</p>',
f'<div class="stats"><div class="stat"><strong>46 / 46</strong>源片段通过 reader</div><div class="stat"><strong>12</strong>数据 family</div><div class="stat"><strong>{tcp}</strong>TCP-local 可监督片段</div><div class="stat"><strong>{head}</strong>Head-relative 可监督片段</div><div class="stat"><strong>{joint}</strong>具有 arm joint 监督</div></div>',
'<p>跳转：<a href="#mora-comparison">Mora / OpenWAM 详细对比</a> · <a href="#paired-visuals">同帧图像与轨迹</a> · <a href="#pipeline">图像处理链路</a> · <a href="#gt">GT 转换</a> · <a href="#coverage">覆盖率</a> · <a href="#clips">46 段视频</a> · <a href="#verify">验证与复现</a></p>',
'<section class="panel"><h2>严格保留指定的数据范围</h2>',
f'<p>源配置：<code>{esc(manifest["yaml"])}</code> 的 108–298 行 <code>segments_by_repo</code>。共 46 条，每条是原始 episode 内的 <code>[start, end)</code>，各 150 帧。</p>',
f'<p>虽然文件名是 v0_3，实际配置的根目录是 <code>{esc(manifest["source_root"])}</code>。这次按配置读取 V0_2，未混入其他版本的 TCP 补充字段。</p>',
'<p>沿用 OpenWAM 的 <strong>30 Hz 源状态</strong>：每窗 33 个 state，目标 t+1…t+32（最长 1.067 s），视频 stride=4，模型视频 9 帧。没有套用 Mora resolver 的 15 Hz 下采样。报告视频单独每两源帧显示一帧，15 fps、5 秒，不改变训练输入时序。</p>',
'</section><section class="panel" id="pipeline"><h2>① 图像：从源视频到模型像素张量</h2>',
'<div class="flow"><div class="node">选定 episode / 150 帧区间<br><small>保留原视频 shard 偏移</small></div><div class="arrow">→</div><div class="node">PyAV 按帧索引解码<br><small>t + [0,4,8,…,32]</small></div><div class="arrow">→</div><div class="node">PIL LANCZOS 拉伸 resize<br><small>head 320×256<br>wrist 各 160×128</small></div><div class="arrow">→</div><div class="node">三相机拼接<br><small>320×384，缺失槽位黑色<br>assembler 同尺寸 BILINEAR</small></div><div class="arrow">→</div><div class="node">RGB uint8 → bf16<br><small>2×pixel/255 − 1<br>[1,3,9,384,320]</small></div></div>',
'<div class="twocol"><div><h3>本次真实开关</h3>',
table(['操作','本次行为'],[['Crop','未启用；没有 center/random crop'],['Resize','各相机直接拉伸到槽位尺寸；不保持源宽高比'],['颜色增强','color_jitter 未配置，关闭'],['随机视角','固定指定 head camera；未启用随机 EE / 相机模式'],['空间 pad','无等比例 letterbox；缺相机整槽填黑'],['时间 pad','不足窗口：视频重复最后一张已采样真帧，video_mask=false；GT 补零且 action_mask=false'],['首帧条件','first_frame_image=[video[0]]；vace_video=None'],['后续 VAE / 文本编码 / 加噪','本次未运行，不计作已验证模型 forward']]),
'</div><div><h3>实际 canvas（宽×高）</h3><div class="view-map"><div class="head">Head · 320×256</div><div class="left">Left wrist<br>160×128</div><div class="right">Right wrist<br>160×128</div></div></div></div>',
'<p>每段详情包含源相机第一帧预览、resize 后槽位和全部 9 张实际 reader canvas。源预览保存为 JPEG；resize / canvas 保存为 PNG。已经逐像素核对原始解码 → resize，以及 9 张重建 canvas → reader 输出。</p>',
'<p class="warn">视频左上角为便于观察而采用等比例缩放、留边的头相机预览；右上角才是模型实际拉伸 canvas（附 GT overlay）。详情中的 PNG 是无 overlay 的实际输入。不要把报告展示的留边当作训练预处理。</p>',
'</section><section class="panel" id="gt"><h2>② GT：同一个物理 TCP，两套 relative pose</h2>',
'<div class="flow"><div class="node">源 state[t…t+32]<br><small>不读取预计算 action[t]</small></div><div class="arrow">→</div><div class="node">统一物理 TCP / world<br><small>原生 TCP 或已验证 wrist→TCP<br>移动基座逐帧转换</small></div><div class="arrow">→</div><div class="node">固定锚点 t<br><small>TCP-local / head-relative<br>joint q[t+k]−q[t]</small></div><div class="arrow">→</div><div class="node">raw 52D + validity mask<br><small>缺 TCP：两套 pose 都 mask</small></div><div class="arrow">→</div><div class="node">quantile → clip → 112D<br><small>时间 pad / 缺失维度 mask</small></div></div>',
'''<div class="formula"><p>记世界下当前 TCP 为 <code>T₀=(R₀,p₀)</code>，未来为 <code>Tₖ=(Rₖ,pₖ)</code>；当前头相机 world→camera 为 <code>C₀=(R꜀,t꜀)</code>。窗口内 C₀ 与 T₀ 固定，滑动到下一显示帧时更新锚点。</p><pre>TCP-local： Δp_tcp = R₀ᵀ (pₖ − p₀)       ΔR_tcp = R₀ᵀ Rₖ
Head-relative：Δp_head = R꜀ (pₖ − p₀)     ΔR_head = R꜀ Rₖ R₀ᵀ R꜀ᵀ
Joint：       Δq = q[t+k] − q[t]         k = 1…32
Gripper：     g = g[t+k]                 绝对未来开合量
Rotation6D：  concat(R[:,0], R[:,1])      旋转矩阵前两列，绝非旋转值直接相减</pre></div>''',
'<p>头相机投影必须先恢复锚点：<code>p_camera[k] = (R꜀ p₀ + t꜀) + Δp_head[k]</code>，再做 pinhole <code>u ~ K p_camera[k]</code>；不能直接投影相对位移。TCP-local 图中原点就是当前 TCP，终点 RGB 三轴展示未来姿态。两套 pose 均使用同一个 TCP 接触中心。</p>',
'<h3>52D → 112D 的实际占位（0-based，闭区间）</h3>',
table(['112D 槽位','内容','缺失处理'],[['0–8 / 9','左 TCP-local xyz + rotation6D / 左 gripper','缺 TCP 只 mask pose；gripper 独立'],['10–33','旧 80D 中未使用槽','零 / mask'],['34–42 / 43','右 TCP-local xyz + rotation6D / 右 gripper','与左侧相同'],['44–79','旧 80D 中未使用槽','零 / mask'],['80–86 / 87–93','左 / 右 7D relative joint','6 自由度臂的第 7 槽 mask'],['94–102 / 103–111','左 / 右 head-relative xyz + rotation6D','缺同一 TCP 或可信头相机外参时 mask']]),
'<p>归一化：<code>clip(2(x−q01)/max(q99−q01, ε)−1, −1, 1)</code>。本次每个 150 帧片段独立生成 smoke stats，采集片段内所有有效锚点/未来步；rot6D 使用固定范围。这些不是全量生产训练统计。下方每段均列出越过 quantile 范围的比例和反算后的物理变化。</p>',
'<p>proprio 使用当前帧的绝对 TCP / joint；head proprio 是当前相机下的绝对 TCP。相对未来目标不意味着 proprio 也全为零。</p></section>',
'<section class="panel" id="coverage"><h2>③ 覆盖率与 mask 原因</h2>']
reason_cn={
'Genrobot':'现有 EEF 是 finger_device_base_link；偏移仍是候选值，未作为可信 TCP 监督。',
'LightWheel':'人手/腕部数据，没有用户要求的机械臂物理 TCP，两套 hand pose 均 mask。',
'abc_130k':'由记录的 MJCF 接触中心→腕部变换及轴向置换反算 TCP；4 段 strict camera eligibility 均失败，head mask。',
'egodex':'人手虚拟帧不作为物理 TCP；手部 pose / arm joint mask。',
'egoverse':'人手虚拟帧不作为物理 TCP；手部 pose / arm joint mask。',
'interndata_a1':'原生 xyz+wxyz TCP，经逐帧 robot_to_env 转为 world；4 段 head / TCP 都可用。',
'mecka':'指定 V0_2 源没有其他版本的 TCP 补充字段；人手 pose / arm joint mask。',
'morphirobot0930':'源 EEF 为 arm08_link 法兰/腕相机安装处，无独立确认的物理 TCP 变换。',
'robochallenge_v1':'这 4 段缺少明确物理 TCP / 可追溯 wrist→TCP 标定；保留可用 joint、gripper。',
'robochallenge_v2':'3 段 DOS-W1 由记录接触中心变换恢复 TCP 并可投影；1 段 ALOHA 缺接触 TCP。',
'robomind_v1':'ARX5 几何接触中心仍是候选值，未确认为标定 TCP；保留真实 joint 与归一化 gripper。',
'robomind_v2':'同上；此版本 joint 索引与 v1 不同，使用 L0:6、R6:12，gripper12/13。'}
family_rows=[]
for family in dict.fromkeys(r['family'] for r in results):
 rr=[r for r in results if r['family']==family]
 family_rows.append([esc(family),str(len(rr)),str(sum(r['input']['tcp_local_supervised'] for r in rr)),str(sum(r['input']['head_supervised'] for r in rr)),str(sum(bool(r['input']['joint_supervised_dimensions']) for r in rr)),esc(reason_cn[family])])
parts+=[table(['Family','片段','TCP-local','Head','Joint','原因 / 数据合同'],family_rows),'<p>“有视频”不等于“有可信相机标定”；“有 URDF / 网格”不等于“已确认同一物理 TCP”。本次没有为缺失合同的数据猜测端点。详细字段、变换矩阵、原始证据见 '+link('source_contract_audit.json','source contract audit')+'。</p></section>',
'<h2 id="clips">④ 视频与逐条输入记录</h2><p>每段 5 秒。上方：当前头相机下未来 32 步 TCP 轨迹；右侧：实际拼接 canvas 加同一轨迹；下方：左 / 右当前 TCP 坐标系下的 relative EE 位置与姿态。黄色左臂、紫色右臂、白色锚点；姿态轴红 X / 绿 Y / 蓝 Z。未满足监督条件的轨迹位置明确显示 MASKED。</p>',
'<div class="filter"><select id="family"><option value="all">全部数据集</option>'+''.join(f'<option>{esc(f)}</option>' for f in dict.fromkeys(r['family'] for r in results))+'</select><select id="coverageFilter"><option value="all">全部覆盖类型</option><option value="head">Head 可监督</option><option value="tcp">TCP-local 可监督</option><option value="masked">缺 TCP / masked</option></select><input id="search" placeholder="搜索 repo / episode"><button id="pause">暂停所有视频</button><span id="visible"></span></div>']
# Show usable projections first, then source order; original YAML ID remains stable.
for r in sorted(results,key=lambda x:(not x['input']['head_supervised'],not x['input']['tcp_local_supervised'],x['id'])):
 ident=r['id'];base='clips/'+ident+'/';a=r['input'];m=r['render'];sourceinfo=load(Path(r['source'])/'meta/info.json');cfg=r['config']
 parts.append(f'<article class="clip" id="{ident}" data-family="{esc(r["family"])}" data-head="{int(a["head_supervised"])}" data-tcp="{int(a["tcp_local_supervised"])}" data-search="{esc(r["repo"])} episode {r["episode"]}"><h3>{esc(ident)} · episode {r["episode"]} · [{r["start"]}, {r["end"]})</h3><p class="muted">{esc(r["repo"])}</p>')
 for name,on in [('TCP-local',a['tcp_local_supervised']),('Head',a['head_supervised']),('Joint',bool(a['joint_supervised_dimensions']))]:parts.append(f'<span class="tag {"yes" if on else "no"}">{name}: {"有效" if on else "MASK"}</span>')
 parts.append(f'<span class="tag">joint {a["joint_supervised_dimensions"]}D · gripper {a["gripper_supervised_dimensions"]}D</span><p>{esc(reason_cn[r["family"]])}</p>')
 parts.append(f'<video controls preload="none" playsinline poster="{base}poster.jpg"><source src="{base}trajectory.mp4" type="video/mp4"></video>')
 parts.append(f'<p><small>Head 有效监督 {m["head_supervised_frames"]}/{m["frame_count"]} 显示帧；画内可见轨迹 {m["head_projected_frames"]}/{m["frame_count"]} 帧。TCP 回代最大位置误差：{("—（masked）" if m["source_pose_roundtrip_max_position_m"] is None else format(m["source_pose_roundtrip_max_position_m"],".3g")+" m")}。{link(base+"trajectory.mp4","下载 MP4")}</small></p>')
 parts.append('<details><summary>展开实际图像处理链路：源帧 → resize → 9 张模型输入</summary><div class="gallery">')
 for side,cam in [('head',r['head']),('left',r['left']),('right',r['right'])]:
  if cam and (OUT/base/f'source_{side}.jpg').exists():
   shape=sourceinfo['features'][cam]['shape'];native_size=Image.open(OUT/base/f'source_{side}.jpg').size;shape_note=f'{native_size[0]}×{native_size[1]}' + (f'（metadata 写 {shape[1]}×{shape[0]}，不符）' if tuple(shape[:2][::-1]) != native_size else '');parts.append(f'<figure><a href="{base}source_{side}.jpg"><img loading="lazy" src="{base}source_{side}.jpg"></a><figcaption>源 {esc(side)} · {esc(shape_note)}<br>{esc(cam)}</figcaption></figure><figure><img loading="lazy" src="{base}resized_{side}.png"><figcaption>LANCZOS resize · {"320×256" if side=="head" else "160×128"}</figcaption></figure>')
  else:parts.append(f'<figure><p>{esc(side)} 缺失 → 黑色槽位</p></figure>')
 parts.append('</div><div class="gallery">')
 for k in range(9):parts.append(f'<figure><a href="{base}model_frame_{k}.png"><img loading="lazy" src="{base}model_frame_{k}.png"></a><figcaption>模型 video[{k}] · 源帧 {r["start"]+k*4}<br>320×384 RGB，无 overlay</figcaption></figure>')
 parts.append('</div><p>像素输出：<code>[1,3,9,384,320]</code> bf16；实际范围 '+esc(m['prevae']['range'])+'。原帧 resize 与 9 张 canvas 均逐像素一致。此张量是 VAE 前像素输入，不是 latent。</p></details>')
 parts.append('<details><summary>展开真实数值、GT mask、归一化裁剪与尾部 padding</summary>')
 rows=[]
 for s in a['examples']:rows.append([str(s['future_step']),str(s['source_frame']),esc(fmt(s['left_tcp_xyz_m'])),esc(fmt(s['right_tcp_xyz_m'])),esc(fmt(s['left_head_delta_m'])),esc(fmt(s['right_head_delta_m']))])
 parts.append(table(['k','源帧','左 TCP-local Δp / m','右 TCP-local Δp / m','左 head Δp / m','右 head Δp / m'],rows))
 parts.append(table(['k','joint14 = q[t+k] − q[t]，rad（— 为 masked）'],[[str(s['future_step']),esc(fmt(s['joint14_rad']))] for s in a['examples']]))
 parts.append(table(['组别','有效 cell（片段所有锚点）','被 quantile 裁剪','最大物理变化'],[[esc(k),str(v['valid_cells']),f'{v["clipped_cells"]} ({v["clipped_cells"]/v["valid_cells"]*100:.2f}%)' if v['valid_cells'] else '—',f'{v["max_physical_change"]:.6g}' if v['valid_cells'] else '—'] for k,v in a['clipping'].items()]))
 parts.append('<p>clipping 统计包含重叠窗口；上面的轨迹是裁剪前 GT。输入记录 JSON 的 examples 同时保留完整 112D 模型值和 112D mask，NPZ 则包含完整首窗口及 proprio。</p>')
 parts.append(table(['本地锚点 / 源帧','真实未来步','action 有效 cell','video_mask','wrist 解码失败累计'],[[f'{c["anchor_local"]} / {c["anchor_source"]}',str(c['real_future_steps']),str(c['valid_action_cells']),esc(c['video_mask']),str(c['wrist_decode_failures'])] for c in r['checks']]))
 parts.append('<p>'+' · '.join(link(base+file,label) for file,label in [('input_audit.json','数值和 clipping JSON'),('model_inputs.npz','模型输入 NPZ'),('source.npz','源 state NPZ'),('trace.json','reader trace'),('stats.npy','本片段 smoke stats'),('render_metrics.json','视频验证'),('video_offsets.json','原视频帧偏移')])+'</p></details>')
 parts.append('<details><summary>展开 reader 配置与 TCP 来源</summary><p class="warn">这是有界报告 fixture。必须通过 <code>prepare.build_clip_reader(config)</code> 恢复原 merged-video offset；不要单独调用 build_dataset(data.yaml)，否则视频从 shard 开头读取。此文件不是可直接启动全量训练的配置。</p><pre>'+esc(cfg)+'</pre><pre>'+esc(json.dumps(r['source_contract']['tcp'],ensure_ascii=False,indent=2))+'</pre><p>额外 report-only adapter：</p><pre>'+esc(json.dumps(r['adapter'],ensure_ascii=False,indent=2))+'</pre></details></article>')
parts+=['<section class="panel" id="verify"><h2>⑤ 验证、限制与复现</h2>',f'<p><strong>46 / 46</strong> 指定区间读取成功，184 个首/中/尾窗口检查通过；46 个 32×112 输入与生产归一化 + scatter 精确一致；有 joint 的样例均验证固定锚点差值。原有 46 段 MP4 解码为 75 帧 / 5 秒；新增 7 段固定 anchor 对比动画各 52 帧 / 5.2 秒。</p>',f'<p>11 个可监督 TCP 片段的物理 GT 回代最大坐标分量误差 <code>{maxpos:.6g} m</code>，最大旋转矩阵元素误差 <code>{maxrot:.6g}</code>。这是坐标转换的数值自洽检查，<strong>不是相机外参精度或真实 TCP 标定精度的测量</strong>。</p>',
'<p>本次修复了真实 task parquet 的兼容问题：读取显式 <code>task</code> 列，兼容旧文本索引，并校验空文本 / 冲突 task ID。报告 fixture 还补回了原视频 shard 内的 episode 累积偏移，逐相机与 from_timestamp×fps 校验。这一适配只用于报告，不改生产视频索引规则。</p>',
'<p class="warn">现有 relative reader 尚未消费源 confidence / frame-valid / EEF-mask 字段。所选 Genrobot 的 alignment / frame validity、Mecka hand / camera validity 均有效；Eg odex 有低置信值，但本次其 pose/joint 已被缺物理 TCP 策略 mask。不能把这些字段描述成已经参与训练过滤。源预计算 actions 的 mask 也不能直接套给重新定义的 next-state 目标。</p>',
'<p>本次不执行 URDF/FK loss、模型 forward 或优化器步骤；因此本报告不能用来判断过拟合是否收敛。图像 reader / GT 使用指定 Python 环境，绘图使用系统 Python 的 matplotlib；两者重建 canvas 已做像素一致性检查。</p>',
'<pre>/root/miniconda3/envs/openwam/bin/python outputs/curated_smoke_20261010/prepare.py\n/root/miniconda3/envs/openwam/bin/python outputs/curated_smoke_20261010/audit_inputs.py\n/usr/bin/python3 outputs/curated_smoke_20261010/render.py\n/usr/bin/python3 outputs/curated_smoke_20261010/build_report.py</pre>',
'<p>直接打开本地 <code>index.html</code>；移动到个人电脑时请保留同目录 <code>clips/</code>、<code>comparison/</code> 及相对路径。无需 CDN 或在线 JavaScript。服务器源数据路径仍是复现 reader 所必需，浏览报告则不需要挂载源数据。</p>',
'<p>'+ ' · '.join(link(p,l) for p,l in [('manifest.json','完整片段清单'),('source_contract_audit.json','源数据合同'),('input_audit.json','全部输入审计'),('provenance.json','代码版本与 SHA256'),('evidence/pretrain_v0_3_curated_smoke.yaml','原 YAML'),('prepare.py','reader driver'),('render.py','render driver'),('audit_inputs.py','数值验证脚本')])+'</p><details><summary>代码快照（本次实际版本）</summary><ul>'+''.join('<li>'+link('evidence/'+x.replace('/','__'),x)+'</li>' for x in source_files)+'</ul></details></section>',
'<script>const cards=[...document.querySelectorAll(".clip")];function filter(){let f=document.getElementById("family").value,c=document.getElementById("coverageFilter").value,s=document.getElementById("search").value.toLowerCase(),n=0;for(let x of cards){let show=(f==="all"||x.dataset.family===f)&&(c==="all"||c==="head"&&x.dataset.head==="1"||c==="tcp"&&x.dataset.tcp==="1"||c==="masked"&&x.dataset.tcp==="0")&&x.dataset.search.toLowerCase().includes(s);x.hidden=!show;if(show)n++;else x.querySelector("video").pause()}document.getElementById("visible").textContent=n+" / 46 片段"}for(let id of ["family","coverageFilter","search"])document.getElementById(id).addEventListener("input",filter);document.getElementById("pause").onclick=()=>document.querySelectorAll("video").forEach(v=>v.pause());document.querySelectorAll("video").forEach(v=>v.addEventListener("play",()=>document.querySelectorAll("video").forEach(w=>{if(w!==v)w.pause()})));filter();</script></main></html>']
document='\n'.join(parts)
marker='<section class="panel" id="pipeline">'
assert document.count(marker)==1
document=document.replace(marker,build_comparison_html()+'\n'+marker)
(OUT/'index.html').write_text(document);print('Wrote',OUT/'index.html','TCP',tcp,'head',head,'joint',joint,'max position',maxpos)
