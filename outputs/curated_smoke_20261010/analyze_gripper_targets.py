"""Evidence-only state/command comparison. Does not change training targets."""
from pathlib import Path
import json,base64,html,csv,hashlib,os
from fractions import Fraction
import numpy as np
import cv2,av
os.environ.setdefault('MPLCONFIGDIR','/tmp/openwam-gripper-comparison-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

ROOT=Path(__file__).resolve().parents[2]
BASE=Path(__file__).resolve().parent/'gripper_state_cmd_20261013'
OUT=BASE/'target_comparison';OUT.mkdir(exist_ok=True)
TRAIN=ROOT/'outputs/curated_train_20261011'
audit=json.loads((BASE/'audit.json').read_text());lookup={r['id']:r for r in audit}
HUMAN={'LightWheel','egodex','egoverse','mecka'}
def calc(x):
 d=np.diff(x);d=d[np.isfinite(d)]
 return {'min':float(np.nanmin(x)),'max':float(np.nanmax(x)),
         'd1_rms':float(np.sqrt(np.mean(d*d))),'d1_max':float(np.max(abs(d))),
         'd1_p95':float(np.quantile(abs(d),.95)),'jumps_gt_0_1':int(np.sum(abs(d)>.1))}
results=[]
for r in audit:
 a=np.load(BASE/r['id']/'source_gripper.npz')
 for j,side in enumerate(['left','right']):
  state=a[r['training_state']+'::'+side]
  cmd=a[r['command_keys'][j]] if r['command_keys'] else None
  entry={'id':r['id'],'family':r['family'],'side':side,'robot':r['family'] not in HUMAN,'frames':len(state),'state':calc(state),'command_key':r['command_keys'][j] if cmd is not None else None,'candidate':calc(cmd) if cmd is not None else None}
  if cmd is None:entry['provenance']='missing'
  elif r['family'] in HUMAN:entry['provenance']='human_proxy_not_robot_cmd'
  else:
   entry['candidate_vs_next_state_max_abs']=float(np.max(abs(cmd[:-1]-state[1:])))
   entry['candidate_equals_next_state']=bool(np.array_equal(cmd[:-1],state[1:]))
   if r['family'].startswith('robochallenge') and entry['candidate_equals_next_state']:entry['provenance']='future_state_identical_not_independent'
   elif r['family']=='interndata_a1':entry['provenance']='master_openness'
   elif r['family']=='abc_130k':entry['provenance']='preserved_ee_action'
   elif r['family'].startswith('robomind'):entry['provenance']='preserved_master_requires_scale_timing_audit'
   else:entry['provenance']='preserved_action_position_requires_timing_audit'
   settled=np.zeros(len(state)-1,bool)
   for t in range(15,len(state)-1):settled[t]=np.isfinite(cmd[t-15:t+2]).all() and np.ptp(cmd[t-15:t+2])<=.01
   d=np.diff(state)
   entry['settled_command']={'definition':'cmd[t-15:t+2] range <=0.01; t>=15; diagnostic, not proof of zero motion','pairs':int(settled.sum()),'state_d1_rms':float(np.sqrt(np.mean(d[settled]**2))) if settled.any() else None,'state_jumps_gt_0_1':int(np.sum(settled&(abs(d)>.1)))}
  results.append(entry)
(OUT/'metrics.json').write_text(json.dumps(results,indent=2))
with (OUT/'metrics.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['id','side','provenance','state_d1_rms','candidate_d1_rms','state_max_jump','candidate_max_jump','settled_state_big_jumps'])
 for e in results:w.writerow([e['id'],e['side'],e['provenance'],e['state']['d1_rms'],e['candidate']['d1_rms'] if e['candidate'] else '',e['state']['d1_max'],e['candidate']['d1_max'] if e['candidate'] else '',e.get('settled_command',{}).get('state_jumps_gt_0_1','')])

# Reconstruct the source calibration to test the noise-amplification hypothesis.
r=lookup['23_interndata_a1'];a=np.load(BASE/r['id']/'source_gripper.npz')
cal=r['source_gripper_normalization']['position_calibration']['states.right_gripper.position']
native=a['states.right_gripper.position::right'];state=a[r['training_state']+'::right']
recomputed=np.clip((native-cal['closed_position'])/cal['range'],0,1)
err=float(np.max(abs(recomputed-state)));assert err<2e-4
event=int(np.argmax(abs(np.diff(state))))+1
mechanism={'id':r['id'],'side':'right','calibration':cal,'gain':1/cal['range'],'raw_range':float(np.ptp(native)),
 'formula_max_abs_error':err,'tolerance_note':'float64 metadata reconstruction versus source float32 under narrow calibration',
 'event':{'before_frame':int(a['frame_index'][event-1]),'after_frame':int(a['frame_index'][event]),
          'raw_before':float(native[event-1]),'raw_after':float(native[event]),'state_before':float(state[event-1]),'state_after':float(state[event]),'cmd_before':1.,'cmd_after':1.}}
assert np.all(a[r['command_keys'][1]]==1)
(OUT/'calibration_evidence.json').write_text(json.dumps(mechanism,indent=2))

CASES=[
 ('23_interndata_a1','right',42,'标定放大：夹爪保持张开，state 接近 0→1 跳变',
  '右腕视频在 frame 130→131 没有可见的对应全幅开合；cmd 全段恒为 1。原始 position 仅由 0.992100656→0.992344260，源库却用约 0.000112307 的区间映射 0～1。当前 state GT 会让模型学习这类标定放大的波动。应修复输入 state 标定；有可信 cmd 时，动作 GT 更适合用 cmd。'),
 ('23_interndata_a1','left',75,'控制意图与达到的位置：cmd=0，state≈0.525',
  '闭合指令持续为 0 时，state 停在约 0.525。它们描述不同的量：想闭合到哪里、实际达到哪里。仅凭这段视频无法把差值完全归因于接触，标定偏移也可能参与；不能把测得的停留位置直接等同于下发指令。'),
 ('08_abc_130k','right',22,'反例：state 本身平滑，开合变化不是噪声',
  'ABC 这段 state 跟随动作连续变化，没有 A1 那类指令不变时的大幅跳变。cmd 的最大相邻帧变化反而略大。不能因为 cmd 有台阶就判它更差，也不能因为 state 是反馈就判它更难学；这里优先 cmd 的理由是控制语义，而不是曲线一定更平滑。'),
 ('26_morphirobot0930','right',35,'轻微 state 波动：不需要把所有 state 都判为坏标签',
  'cmd 在本段恒为 1，state 在 0.990970～1 之间变化，最大单帧变化约 0.001516。这与 A1 接近全量程的跳变不是一个级别；夹爪输入保留反馈有价值。保留的 action.position 时间语义还需核对，不能凭常值片段校准命令延迟。'),
 ('40_robomind_v1','right',30,'反例：换成 cmd 不保证曲线更平滑或 loss 更低',
  '这一段保留 action 的变化幅度大于 state。两路还使用各自的 min/max 标定，不能直接把幅值差归因于测量噪声。若用 cmd 监督，必须先确认主从夹爪尺度、方向与时序；“cmd 一定更好学”没有证据。'),
 ('35_robochallenge_v2','right',35,'字段名陷阱：保留 action 与下一帧 state 完全相同',
  '本片段两侧全部 149 对值均满足 actions.source_action[t] == state[t+1]。当前抽样的 RoboChallenge v1/v2 共 8 段都如此。因此此处改字段不会改变可比范围内的 GT，也不能把它当作一组独立真实 cmd 来证明优势。')]

plt.rcParams.update({'font.size':12,'axes.facecolor':'#162234','figure.facecolor':'#101722','text.color':'#e3ebf7','axes.labelcolor':'#e3ebf7','xtick.color':'#bdcbdc','ytick.color':'#bdcbdc','axes.edgecolor':'#5b6b7f','grid.color':'#38485b','savefig.facecolor':'#101722'})
videos=[]
for name,side,focus,title,note in CASES:
 r=lookup[name];j=['left','right'].index(side);data=np.load(BASE/name/'source_gripper.npz')
 state=data[r['training_state']+'::'+side];cmd=data[r['command_keys'][j]];gt=np.r_[state[1:],np.nan]
 ids=data['frame_index'];t=np.arange(len(state))/r['fps'];native_key=f'states.{side}_gripper.position::{side}'
 fig,axes=plt.subplots(2,1,figsize=(9.2,8.8),dpi=100);fig.subplots_adjust(left=.16,right=.97,top=.93,bottom=.09,hspace=.43)
 axes[0].plot(t,state,color='#66d992',label='state[t]',lw=2)
 command_label='preserved action[t]' if name.startswith('35_') else 'cmd candidate[t]'
 axes[0].plot(t,cmd,color='#ffb454',label=command_label,lw=2,ls='--')
 axes[0].plot(t,gt,color='#c49aff',label='current GT = state[t+1]',lw=1.5,ls=':')
 axes[0].set(ylim=(-.05,1.05),ylabel='Released [0,1] scale',title=f'{side.upper()} gripper | source frame {ids[0]} - {ids[-1]}')
 axes[0].legend(loc='upper right',facecolor='#162234',labelcolor='#e3ebf7',fontsize=10)
 if name=='23_interndata_a1':
  raw=data[native_key];c=r['source_gripper_normalization']['position_calibration'][native_key.split('::')[0]]
  axes[1].plot(t,raw,color='#74bcff',lw=1.5,label='raw position (own source units)')
  axes[1].axhline(c['closed_position'],color='#ff8b91',ls='--',label='source q05 = closed boundary')
  axes[1].axhline(c['open_position'],color='#ffdb80',ls='--',label='source q95 = open boundary')
  axes[1].set(ylabel='Raw position',title=f"Source calibration gain = {1/c['range']:.1f}")
  axes[1].ticklabel_format(axis='y',useOffset=False,style='plain');axes[1].legend(loc='best',facecolor='#162234',labelcolor='#e3ebf7',fontsize=9)
 else:
  axes[1].plot(t[1:],np.diff(state),color='#66d992',label='state[i] - state[i-1]',lw=1.5)
  axes[1].plot(t[1:],np.diff(cmd),color='#ffb454',label='candidate[i] - candidate[i-1]',lw=1.3,ls='--')
  axes[1].set(ylabel='Adjacent-frame difference',title='Variation is NOT automatically noise')
  axes[1].legend(loc='best',facecolor='#162234',labelcolor='#e3ebf7',fontsize=10)
 for ax in axes:ax.set(xlim=(0,t[-1]),xlabel='Time in clip (s)');ax.grid(alpha=.6)
 dest=OUT/f'{name}_{side}';dest.mkdir(exist_ok=True)
 fig.savefig(dest/'curves.svg')
 canvas=FigureCanvasAgg(fig);canvas.draw();chart=cv2.cvtColor(np.asarray(canvas.buffer_rgba()),cv2.COLOR_RGBA2BGR)
 plot_boxes=[tuple(map(int,ax.get_window_extent().bounds)) for ax in axes];plt.close(fig)
 frames=np.load(TRAIN/'data'/name/'canvases.npy',mmap_mode='r');assert len(frames)==150
 path=dest/'focus.mp4';out=av.open(str(path),'w',options={'movflags':'+faststart'})
 stream=out.add_stream('libx264',rate=Fraction(str(r['fps'])));stream.width=1600;stream.height=1080;stream.pix_fmt='yuv420p';stream.options={'crf':'22','preset':'fast','threads':'4'}
 for i,rgb in enumerate(frames):
  frame=np.full((1080,1600,3),(34,23,16),np.uint8)
  frame[115:115+chart.shape[0],680:680+chart.shape[1]]=chart
  def txt(s,x,y,color=(237,236,230),size=.65):cv2.putText(frame,s,(x,y),cv2.FONT_HERSHEY_SIMPLEX,size,color,1,cv2.LINE_AA)
  txt(f'{name} | {side.upper()} GRIPPER',20,40,size=.98)
  txt(f'episode {r["episode"]} | source frame {ids[i]} | {i/r["fps"]:.3f}s',20,76)
  wrist=rgb[256:384,j*160:(j+1)*160];other=rgb[256:384,(1-j)*160:(2-j)*160]
  frame[130:642,20:660]=cv2.cvtColor(cv2.resize(wrist,(640,512),interpolation=cv2.INTER_CUBIC),cv2.COLOR_RGB2BGR)
  frame[684:940,20:340]=cv2.cvtColor(rgb[:256],cv2.COLOR_RGB2BGR)
  frame[684:940,340:660]=cv2.cvtColor(cv2.resize(other,(320,256),interpolation=cv2.INTER_CUBIC),cv2.COLOR_RGB2BGR)
  txt('FOCUS WRIST (enlarged from actual model input)',20,116,size=.59)
  txt('HEAD / PRIMARY',20,675,size=.58);txt('OTHER WRIST',350,675,size=.58)
  for x,y,w,h in plot_boxes:
   xx=680+x+round(i*w/(len(frames)-1));top=115+chart.shape[0]-y-h
   cv2.line(frame,(xx,top),(xx,top+h),(210,215,235),1)
  txt(f'state[t] = {state[i]:.6f}',20,977,(146,217,102),.85)
  txt(f'candidate[t] = {cmd[i]:.6f}',20,1015,(84,180,255),.85)
  txt('GT state[t+1] = '+(f'{gt[i]:.6f}' if np.isfinite(gt[i]) else 'N/A'),20,1053,(255,154,196),.8)
  txt('Source CMD timing must be verified before changing GT.',710,1020,size=.61)
  txt('Label comparison only; not a training A/B result.',710,1050,size=.59)
  for packet in stream.encode(av.VideoFrame.from_ndarray(frame,format='bgr24')):out.mux(packet)
  if i==focus:cv2.imwrite(str(dest/'preview.jpg'),frame)
 for packet in stream.encode():out.mux(packet)
 out.close()
 with av.open(str(path)) as reader:
  decoded=list(reader.decode(video=0));assert len(decoded)==150;assert float(reader.streams.video[0].average_rate)==r['fps']
  hashes={hashlib.sha256(f.to_ndarray(format='rgb24')[140:630,20:660].tobytes()).hexdigest() for f in decoded}
 assert len(hashes)>1
 videos.append({'id':name,'side':side,'folder':dest.name,'focus':focus,'focus_source_frame':int(ids[focus]),'fps':r['fps'],'frames':150,'moving_focus_frames':len(hashes),'title':title,'note':note})
 print(json.dumps(videos[-1],ensure_ascii=False),flush=True)

esc=lambda x:html.escape(str(x))
parts=['''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Gripper：state 还是 cmd？</title><style>body{font:16px/1.7 system-ui,sans-serif;background:#101722;color:#e5eefb;max-width:1450px;margin:30px auto;padding:0 22px}a{color:#91caff}h1,h2{line-height:1.35}section{background:#1b283b;padding:22px;border-radius:12px;margin:24px 0}video,img{width:100%;max-height:850px;background:#0a111e}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #43536a;padding:8px;text-align:left}code{color:#ffcc86;overflow-wrap:anywhere}.notice{border-left:5px solid #efb65f;padding:15px 20px;background:#243144}button,select{background:#283e5d;color:#edf4ff;border:1px solid #7190b9;padding:8px;margin:6px;border-radius:5px}small{color:#b4c3d9}summary{cursor:pointer}.scroll{overflow:auto}</style><h1>Gripper：state 输入保留反馈，动作 GT 优先可信 cmd</h1><p>依据当前 46 个 curated smoke 片段的逐帧数据与视频。这里只分析夹爪；训练代码和正在运行的训练目标未改动。</p><div class="notice"><b>建议：</b>如果模型输出要直接下发给夹爪控制器，输入使用经过可靠标定的当前 <code>state[t]</code>，GT 优先使用与部署控制接口一致、时间对齐的真实 <code>cmd</code>。<b>不要按字段名一刀切替换，也不要把曲线更平滑等同于控制更正确。</b></div><p>本次确实看到了 state 的异常跳变，但主要明确证据来自 A1 的窄分位数标定。ABC 等片段的 state 很平滑；RoboMind 也有保留 action 比 state 变化更大的反例。还没有做 state/cmd 两种监督的同条件训练或机器人成功率 A/B，不能宣称换 cmd 一定 loss 更低或收敛更快。</p>''']
parts.append(f'<section><h2>最明确的证据：A1 右夹爪</h2><p>原始 position 全段范围只有 <b>{mechanism["raw_range"]:.9f}</b>；源库把 <code>{cal["closed_position"]:.9f}～{cal["open_position"]:.9f}</code> 当成闭合到张开的全量程，等效增益约 <b>{mechanism["gain"]:.0f}</b>。因此微小波动被放大。</p><table><tr><th>source frame</th><th>原始 position</th><th>reader state</th><th>master openness</th></tr>')
e=mechanism['event']
for pfx in ['before','after']:parts.append(f'<tr><td>{e[pfx+"_frame"]}</td><td>{e["raw_"+pfx]:.9f}</td><td>{e["state_"+pfx]:.9f}</td><td>{e["cmd_"+pfx]:.1f}</td></tr>')
parts.append(f'</table><p>按源元数据公式重算 state 的最大偏差为 {err:.7f}，支持“窄标定区间放大波动”的解释；小残差可能与窄区间下的浮点精度有关。视频中右夹爪保持张开，没有可见的相应全幅开合。仅 clip 到 [0,1] 无法消除此问题，输入 state 的标定同样需要修复。</p></section>')
parts.append('''<section><h2>两种目标的含义</h2><table><tr><th></th><th>未来 state</th><th>真实 cmd</th></tr><tr><td>模型在学什么</td><td>执行后达到的开合，包括响应、接触约束和测量/标定误差</td><td>控制器应接收的开合目标；仍需正确的尺度和方向</td></tr><tr><td>抓住物体时</td><td>实际夹爪可能停在物体宽度，并非完全闭合</td><td>闭合指令可持续有效，维持闭合意图</td></tr><tr><td>抖动与跳变</td><td>可以平滑，也可能含反馈噪声或标定放大</td><td>可以连续也可以离散；正确开闭切换产生的台阶应保留</td></tr><tr><td>缺少可靠 cmd</td><td>可作为明确标记的伪 action 监督；需检查标定</td><td>不能凭 actions.source_action 名称假设其存在；可 mask 命令监督</td></tr></table><p>对基于位置误差的底层控制器，指令与反馈的差值会影响施力；把已达到的位置直接当指令，未必能复现原控制行为。ACT 原论文也区分 follower 状态输入和 leader 指令监督，见 <a href="https://arxiv.org/html/2304.13705v1#S4">ACT 第 IV 节</a>。这是控制语义依据，不是对本数据集抓取力或成功率的实测结论。</p><p><b>时间对齐：</b>GT 的第 0 项应对应观察 t 后执行的命令；原始 cmd[t] 是否就是这条命令，需要逐数据源核查。有些保存列已经做过移位，不应再机械加 1 或加 3。下面橙线只按原始行号展示，不代表已确定部署对齐。</p></section>''')
for v in videos:
 key=v['folder'];r=lookup[v['id']];m=next(e for e in results if e['id']==v['id'] and e['side']==v['side'])
 b64=base64.b64encode((OUT/key/'focus.mp4').read_bytes()).decode()
 parts.append(f'<section id="{key}"><h2>{esc(v["title"])}</h2><p>{esc(v["note"])}</p><small>{esc(v["id"])} · {v["side"]} · episode {r["episode"]} · source frames {r["start"]}–{r["end"]-1}</small><video id="v{key}" data-fps="{v["fps"]}" controls muted loop playsinline preload="none" src="data:video/mp4;base64,{b64}"></video><div><button onclick="seekVideo(\'v{key}\',{max(0,v["focus"]-3)/v["fps"]})">定位关键帧附近（{v["focus_source_frame"]}）</button><button onclick="stepVideo(\'v{key}\',-1)">上一帧</button><button onclick="stepVideo(\'v{key}\',1)">下一帧</button><select onchange="document.getElementById(\'v{key}\').playbackRate=Number(this.value)"><option value="1">1×</option><option value="0.5">0.5×</option><option value="0.25">0.25×</option></select></div>')
 parts.append(f'<p>绿色 state[t]；紫色当前 GT=state[t+1]；橙色保留的命令候选。字段：<code>{esc(m["command_key"])}</code>。')
 if 'robochallenge' in v['id']:parts.append('<b>这段橙线与紫线完全重合，不是独立命令对照。</b>')
 parts.append('</p><table><tr><th>指标（当前发布尺度）</th><th>state</th><th>命令候选</th></tr>')
 for label,k in [('最大单帧变化','d1_max'),('单帧变化 RMS','d1_rms'),('单帧变化 P95','d1_p95')]:parts.append(f'<tr><td>{label}</td><td>{m["state"][k]:.6f}</td><td>{m["candidate"][k]:.6f}</td></tr>')
 parts.append('</table><p><small>这是曲线变化量，不是模型 loss，也不自动等于噪声。跨数据源/不同标定尺度不可直接排名。</small></p>')
 svg=base64.b64encode((OUT/key/'curves.svg').read_bytes()).decode()
 parts.append(f'<details><summary>展开静态曲线（可导出）与来源</summary><img src="data:image/svg+xml;base64,{svg}"><p><code>{esc(r["source"])}</code></p><p><a href="target_comparison/{key}/curves.svg">SVG</a> · <a href="target_comparison/{key}/focus.mp4">MP4</a></p></details></section>')
parts.append('''<section><h2>各数据源建议</h2><table><tr><th>数据源</th><th>建议</th></tr><tr><td>A1</td><td>优先已知语义的 master_actions.*_gripper.openness 做动作 GT；先修复 state 的窄分位数标定，不能让输入继续虚假跳变。</td></tr><tr><td>ABC</td><td>有原始 /left-ee-action.position、/right-ee-action.position 来源，适合作为 cmd 候选。state 本身正常，优先 cmd 是为了匹配控制语义；核对时间对齐。</td></tr><tr><td>RoboMind v1/v2</td><td>有 master 来源，但主从量程/归一化与时间偏移需先校验。不能直接比较现有两路数值来决定优劣。</td></tr><tr><td>Morphi</td><td>部分片段保留 action.position；本次常值样例不足以确定时序。另有片段未识别到独立命令，需分来源处理。</td></tr><tr><td>RoboChallenge v1/v2</td><td>8 个抽样片段的保留 action 与 state[t+1] 完全一致；这批字段切换没有监督差异。需找到更上游真正的控制命令才能比较。</td></tr><tr><td>Genrobot</td><td>这批字段中未识别到独立 cmd；保留质量合格的 state 伪监督或 mask，不虚构命令。</td></tr><tr><td>LightWheel / EgoDex / EgoVerse / Mecka</td><td>人手数据的 gripper/proxy 不等于机器人执行器 cmd，应与机器人命令监督明确区分。</td></tr></table></section>''')
parts.append('<section><h2>完整逐侧量化结果</h2><p>46 片段 × 左右两侧。只统计数值，不把未确认来源标为真实 cmd。大跳变阈值是相邻帧 |Δ|&gt;0.1；“稳定指令”条件是此前 15 帧至下一帧的 cmd 范围≤0.01，仍不保证机器人绝对静止。</p><div class="scroll"><table><tr><th>片段 / 侧</th><th>候选来源判断</th><th>state 最大 Δ</th><th>候选最大 Δ</th><th>稳定指令段内 state 大跳变数</th></tr>')
for e in results:
 c=f'{e["candidate"]["d1_max"]:.6f}' if e['candidate'] else 'N/A'
 parts.append(f'<tr><td>{e["id"]} / {e["side"]}</td><td>{esc(e["provenance"])}</td><td>{e["state"]["d1_max"]:.6f}</td><td>{c}</td><td>{e.get("settled_command",{}).get("state_jumps_gt_0_1","—")}</td></tr>')
parts.append('''</table></div><p><a href="target_comparison/metrics.csv">逐侧指标 CSV</a> · <a href="target_comparison/metrics.json">详细 JSON</a> · <a href="target_comparison/calibration_evidence.json">A1 标定重算证据</a> · <a href="index.html">前一版全部字段视频</a></p></section><p>方法与限制：每段 150 连续帧；视频来源为已对齐源 frame_index 的训练 RGB 缓存。腕图放大用于观察，没有生成新像素细节。数据片段有限，不能代表整个大数据集。没有重训、没有控制器回放、没有测力或任务成功率验证。视频和曲线已内嵌，本 HTML 下载后可直接用浏览器打开。</p><script>function seekVideo(id,t){const v=document.getElementById(id);const go=()=>{v.currentTime=t;v.pause()};if(v.readyState>=1)go();else{v.addEventListener('loadedmetadata',go,{once:true});v.load()}}function stepVideo(id,n){const v=document.getElementById(id);seekVideo(id,Math.max(0,Math.min((v.duration||5)-1/Number(v.dataset.fps),v.currentTime+n/Number(v.dataset.fps))))}</script></html>''')
(BASE/'state_vs_cmd.html').write_text(''.join(parts))
(OUT/'validation.json').write_text(json.dumps({'source_streams':len(results),'robot_streams':sum(e['robot'] for e in results),'videos':videos,'calibration_max_error':err,'no_training_changes':True},ensure_ascii=False,indent=2))
print('REPORT',BASE/'state_vs_cmd.html',flush=True)
