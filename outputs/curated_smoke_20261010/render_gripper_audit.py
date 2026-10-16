"""Render real moving video + exact source gripper fields + production reader GT."""
from pathlib import Path
import sys, json, csv, base64, html, types, importlib.util, hashlib
from fractions import Fraction
import numpy as np
import cv2, av
from PIL import Image
from omegaconf import OmegaConf

ROOT=Path(__file__).resolve().parents[2]
TRAIN=ROOT/'outputs/curated_train_20261011'
OUT=Path(__file__).resolve().parent/'gripper_state_cmd_20261013'
sys.path[:0]=[str(ROOT),str(Path(__file__).resolve().parent)]
from prepare import build_clip_reader
spec=importlib.util.spec_from_file_location('old_gripper_reader',TRAIN/'reader_before_gripper01.py')
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
SELECT=['38_robomind_v1','40_robomind_v1','20_interndata_a1','23_interndata_a1','26_morphirobot0930','08_abc_130k','00_Genrobot','42_robomind_v2']
W,H=1600,1040
BG=(27,22,18);FG=(230,231,235);MUTED=(170,169,163)
GREEN=(135,221,98);ORANGE=(75,174,255);BLUE=(247,170,91);PURPLE=(225,136,223);GRAY=(170,170,170)
blank=Image.new('RGB',(320,384))
def skip_video(self,*a,**kw):return [blank]*self._num_video_frames
def text(img,s,x,y,color=FG,scale=.61):cv2.putText(img,str(s),(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)
def num(v):return f'{v:.6f}' if np.isfinite(v) else 'N/A'
def safejson(v):
 if isinstance(v,np.integer):return int(v)
 if isinstance(v,np.ndarray):return [safejson(x) for x in v.tolist()]
 if isinstance(v,(float,np.floating)):return float(v) if np.isfinite(v) else None
 if isinstance(v,dict):return {k:safejson(x) for k,x in v.items()}
 if isinstance(v,(tuple,list)):return [safejson(x) for x in v]
 return v
def graph(base,series,box,limits,title):
 x,y,w,h=box;lo,hi=limits
 text(base,title,x,y-13,scale=.63)
 for v in np.linspace(lo,hi,5):
  yy=int(y+h-(v-lo)/(hi-lo)*h)
  cv2.line(base,(x,yy),(x+w,yy),(61,56,48),1)
  text(base,f'{v:.3f}',x-62,yy+4,MUTED,.45)
 cv2.rectangle(base,(x,y),(x+w,y+h),(92,86,78),1)
 for n,(name,a,color) in enumerate(series):
  for i in range(len(a)-1):
   if not np.isfinite(a[i:i+2]).all():continue
   p=(x+round(i*w/(len(a)-1)),round(y+h-(a[i]-lo)/(hi-lo)*h))
   q=(x+round((i+1)*w/(len(a)-1)),round(y+h-(a[i+1]-lo)/(hi-lo)*h))
   # Distinct line patterns keep coincident state/target curves readable.
   if n==1 and i%5>=3:continue
   if n==2 and i%4>=2:continue
   if n==3 and i%6>=2:continue
   cv2.line(base,p,q,color,2,cv2.LINE_AA)
 return {'series':series,'box':box,'limits':limits}
def cursor(img,g,i,n):
 x,y,w,h=g['box'];lo,hi=g['limits'];xx=x+round(i*w/(n-1))
 cv2.line(img,(xx,y),(xx,y+h),(224,228,233),1)
 for _,a,c in g['series']:
  if np.isfinite(a[i]):cv2.circle(img,(xx,round(y+h-(a[i]-lo)/(hi-lo)*h)),4,c,-1)

audit=json.loads((OUT/'audit.json').read_text());rows={r['id']:r for r in audit}
manifest={r['id']:r for r in json.loads((TRAIN/'data_manifest.json').read_text())}
results=[]
for name in SELECT:
 r=rows[name];dest=OUT/name;source=dict(np.load(dest/'source_gripper.npz'));n=len(source['frame_index'])
 cfg=OmegaConf.load(TRAIN/'data'/name/'data.yaml')
 new=build_clip_reader(cfg);old=legacy.MoraRelativeJointDataset.from_config(cfg)
 for reader in [new,old]:reader._decode_window_video=types.MethodType(skip_video,reader)
 frames=np.load(TRAIN/'data'/name/'canvases.npy',mmap_mode='r')
 assert frames.shape==(n,384,320,3)
 gt=np.full((n,2),np.nan);old_gt=gt.copy();state=np.stack([source[f"{r['training_state']}::{s}"] for s in ['left','right']],axis=1)
 for i in range(len(new)):
  now,before=new[i],old[i]
  for side,slot in enumerate([9,43]):
   if now['action_mask'][0,slot]:gt[i,side]=float(now['action'][0,slot])
   if before['action_mask'][0,slot]:old_gt[i,side]=float(before['action'][0,slot])
  np.testing.assert_allclose(now['proprio'][0,[9,43]].numpy(),np.clip(state[i],0,1),rtol=0,atol=1e-7)
 np.testing.assert_allclose(gt[:-1],np.clip(state[1:],0,1),rtol=0,atol=1e-7)
 raw_keys={}
 for s in ['left','right']:
  if r['family']=='robomind_v1':raw_keys[s]=[f'observation.states.eef::{s}',f'actions.eef.source_native::{s}']
  elif r['family']=='interndata_a1':raw_keys[s]=[f'states.{s}_gripper.position::{s}',f'master_actions.{s}_gripper.position::{s}']
  elif r['family']=='morphirobot0930':raw_keys[s]=[f'observation.state.position::{s}',f'action.position::{s}']
  elif r['family']=='robomind_v2':raw_keys[s]=[f'raw.puppet.end_effector_{s}_position_align::{s}',f'raw.master.end_effector_{s}_position_align::{s}']
  else:raw_keys[s]=[]
 data={**source,'reader.state.left':state[:,0],'reader.state.right':state[:,1],
       'reader.gt_next.left':gt[:,0],'reader.gt_next.right':gt[:,1],
       'old_quantile.gt_next.left':old_gt[:,0],'old_quantile.gt_next.right':old_gt[:,1]}
 with (dest/'frames.csv').open('w') as f:
  writer=csv.writer(f);writer.writerow(data)
  for i in range(n):writer.writerow([int(a[i]) if k=='frame_index' else num(a[i]) for k,a in data.items()])
 base=np.full((H,W,3),BG,np.uint8)
 text(base,name,20,34,scale=.95)
 text(base,f"episode={r['episode']} | source frames [{r['start']}, {r['end']}) | {r['fps']} fps",20,64,scale=.65)
 text(base,'STATE: selected source field (no per-clip scaling)',710,35,GREEN,.62)
 text(base,'CMD: preserved command/action, if available',710,59,ORANGE,.62)
 text(base,'ACTION: released column (may be future state)',710,83,BLUE,.62)
 text(base,'GT[0]: actual reader output = state[t+1]',710,107,PURPLE,.62)
 graphs=[]
 for j,s in enumerate(['left','right']):
  yy=164+j*278
  cmd=source[r['command_keys'][j]] if r['command_keys'] else np.full(n,np.nan)
  act=source[f"{r['released_action']}::{s}"] if r['released_action'] else np.full(n,np.nan)
  series=[('state',state[:,j],GREEN),('cmd',cmd,ORANGE),('action',act,BLUE),('GT',gt[:,j],PURPLE)]
  g=graph(base,series,(755,yy,810,154),(-.06,1.06),s.upper()+'  |  reader-source scale');graphs.append(g)
  text(base,'0 s',755,yy+175,MUTED,.46);text(base,f'{(n-1)/r["fps"]:.2f} s',1510,yy+175,MUTED,.46)
  keys=raw_keys[s]
  if keys:
   rawseries=[('raw state',source[keys[0]],GREEN),('raw cmd/action',source[keys[1]],ORANGE)]
  else:rawseries=[('old quantile GT',old_gt[:,j],GRAY),('new GT',gt[:,j],PURPLE)]
  values=np.concatenate([a[np.isfinite(a)] for _,a,_ in rawseries]);lo=float(values.min());hi=float(values.max());pad=max((hi-lo)*.12,1e-5)
  if not keys:lo=min(lo,-1.);hi=max(hi,1.);pad=.05
  xx=755+j*435
  title=(s.upper()+'  |  native fields (own scale)') if keys else (s.upper()+'  |  old/new training GT')
  graphs.append(graph(base,rawseries,(xx,787,375,132),(lo-pad,hi+pad),title))
 text(base,'HEAD / PRIMARY CAMERA',34,153,scale=.6)
 text(base,'LEFT WRIST',34,670,scale=.6);text(base,'RIGHT WRIST',355,670,scale=.6)
 text(base,'Camera/labels share source frame_index. CMD is a recorded field; hardware timing is not inferred.',20,1014,MUTED,.62)
 if not r['command_keys']:text(base,'Independent CMD unavailable in this source.',710,727,ORANGE,.61)
 else:text(base,'CMD source: '+r['command_label'],710,727,ORANGE,.57)
 # Draw actual moving RGB camera canvas, not repeated anchor images.
 path=dest/'gripper.mp4';out=av.open(str(path),'w',options={'movflags':'+faststart'});stream=out.add_stream('libx264',rate=Fraction(str(r['fps'])))
 stream.width=W;stream.height=H;stream.pix_fmt='yuv420p';stream.options={'crf':'22','preset':'fast','threads':'4'}
 for i in range(n):
  frame=base.copy();rgb=cv2.resize(np.asarray(frames[i]),(640,768),interpolation=cv2.INTER_CUBIC)
  frame[130:898,20:660]=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
  for label,x,y in [('HEAD / PRIMARY',30,152),('LEFT WRIST',30,665),('RIGHT WRIST',350,665)]:
   cv2.rectangle(frame,(x-4,y-19),(x+200,y+6),BG,-1);text(frame,label,x,y,scale=.57)
  text(frame,f'source frame {int(source["frame_index"][i])} | clip time {i/r["fps"]:.3f} s',20,108,scale=.7)
  for g in graphs:cursor(frame,g,i,n)
  for j,s in enumerate(['left','right']):
   yy=164+j*278
   cmd=source[r['command_keys'][j]][i] if r['command_keys'] else np.nan
   vals=[('state',state[i,j],GREEN),('cmd',cmd,ORANGE),('action',source[f"{r['released_action']}::{s}"][i] if r['released_action'] else np.nan,BLUE),('GT[0]',gt[i,j],PURPLE)]
   for k,(label,v,c) in enumerate(vals):text(frame,label+' '+num(v),710+(k%2)*435,yy+204+(k//2)*26,c,.62)
   text(frame,f'old quantile GT: {num(old_gt[i,j])}',30,924+j*24,GRAY,.59)
   gg=graphs[j*2+1]
   for k,(label,a,c) in enumerate(gg['series']):text(frame,label+': '+num(a[i]),710+j*435,946+k*24,c,.5)
  for packet in stream.encode(av.VideoFrame.from_ndarray(frame,format='bgr24')):out.mux(packet)
  if i in [0,n//2]:cv2.imwrite(str(dest/f'preview_{i}.jpg'),frame)
 for packet in stream.encode():out.mux(packet)
 out.close()
 with av.open(str(path)) as verify:
  decoded=[f.to_ndarray(format='rgb24') for f in verify.decode(video=0)]
 assert len(decoded)==n
 hashes={hashlib.sha256(f[160:640,20:660].tobytes()).hexdigest() for f in decoded}
 raw_hashes={hashlib.sha256(np.asarray(f).tobytes()).hexdigest() for f in frames}
 assert len(hashes)>1 and len(raw_hashes)>1,'Video is static'
 result={'id':name,'frames':n,'fps':r['fps'],'seconds':n/r['fps'],'distinct_camera_frames':len(hashes),'distinct_source_canvases':len(raw_hashes),'state_gt_alignment_verified':True,'raw_keys':raw_keys,'example':{k:safejson(v[0]) for k,v in data.items()}}
 results.append(result);print(json.dumps(result),flush=True)

notes={
 '38_robomind_v1':'旧 EEF 字段出现大于 1 的数值，但元数据明确标注 source_native_unconfirmed；视频下方单独画其原始尺度。当前 reader 从 observation.state 取归一化夹爪，pose 仍来自原 EEF。',
 '40_robomind_v1':'包含真实负数的旧 EEF 样例：左侧约 −0.1085，右侧范围约 −0.1364～4.0896。它们不是此前测试中的 −0.2；不能把旧 EEF 与归一化 openness 混为一谈。',
 '20_interndata_a1':'对照原始 position、master command openness 与 reader 的 state。源库先用每个 repo 的标定参数将 position 转为 openness；新 reader 保持其 [0,1] 值。',
 '23_interndata_a1':'原始左 position 最大约 1.0010。另一个值得复核的问题是：右 position 仅约 0.991958～0.992350，却经源库 p05/p95 标定变成几乎 0～1；右侧标定分母只有约 0.0001123。这说明范围正确不等于物理标定正确，本次只展示证据，未修改该标定。',
 '26_morphirobot0930':'原始左 position=1.000021，源库已 clip 为 state=1。当前 GT 保持 1；旧的片段 quantile 会把这个恒定值变成 −1。橙线用保留的 action.position；发布的 action 是未来 3/4 帧 state，单独用蓝线。',
 '08_abc_130k':'有明确 openness 字段。橙线 actions.source_action 是保留下来的 action；蓝线发布 action 对齐未来 3 帧；紫线为当前 OpenWAM 实际读取的下一帧 state。',
 '00_Genrobot':'没有识别到独立保留的 cmd 字段，因此标注 N/A，不将 actions.eef 伪称为硬件 cmd。蓝线仅展示该发布 action 字段。',
 '42_robomind_v2':'下方展示 raw.puppet/master 原始 position。源库元数据采用 clip(0,10)/10，这段 reader 数值因此只有约 0.002～0.004；是否确实对应 0～10 的物理量程，需要结合视频再复核，不能仅凭数值落在 [0,1] 就认定正确。',
}
esc=lambda x:html.escape(str(x))
parts=['''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>夹爪原始 state / cmd 同步视频核对</title><style>body{font:16px/1.65 system-ui,sans-serif;background:#101722;color:#e6edf5;max-width:1450px;margin:32px auto;padding:0 22px}h1,h2{line-height:1.3}a{color:#8dc9ff}code{color:#ffd18a;overflow-wrap:anywhere}section,.notice{background:#1b2636;padding:22px;border-radius:12px;margin:24px 0}.notice{border-left:5px solid #f5bb64}video{width:100%;max-height:850px;background:#000}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #3a485a;text-align:left;padding:7px;overflow-wrap:anywhere}summary{cursor:pointer}small{color:#b7c2d1}pre{white-space:pre-wrap;overflow-wrap:anywhere}.scroll{overflow:auto}</style><h1>夹爪原始 state / cmd：同步视频核对</h1><p>2026-10-13 · 当前 curated smoke 的 46 个片段、12 类数据源 · 每片段 150 帧</p><div class="notice"><b>先纠正来源：1.4 和 −0.2 是回归测试人工构造的数值，不是从真实数据中发现的异常帧。</b><p>出处：<code>tests/dataloader/test_mora_relative_reader.py:163</code>，测试写入 <code>[.35, -.2, .25, 1.4, NaN, Inf]</code> 验证 clip/mask。它们没有对应的真实拍摄视频。下面视频全部来自真实源片段，原始字段不做裁剪。</p></div>''']
finite=sum(r['field_stats'][f"{r['training_state']}::{s}"]['finite'] for r in audit for s in ['left','right'])
outside=sum(r['field_stats'][f"{r['training_state']}::{s}"]['outside_01'] for r in audit for s in ['left','right'])
parts.append(f'<p>逐帧读取源 Parquet：当前 reader 选用的夹爪 state 共 <b>{finite}</b> 个有限值，超出 [0,1] 的数量为 <b>{outside}</b>。这只覆盖本次 46 个训练片段，不能外推为整个大规模数据集。另有旧 EEF、原始 position、velocity/effort 等字段超出该范围；不同字段的量纲与含义不能混用。</p>')
parts.append('''<p><b>绿色</b>：reader 选用的源 state[t]；<b>橙色</b>：保留的 cmd/action 字段；<b>蓝色</b>：发布数据里的 action；<b>紫色</b>：当前生产 reader 实际输出的 action chunk 第 0 步 GT = clip(state[t+1],0,1)。当前 32 步 chunk 其余步未在图中叠加。源库 action 可能已经被未来 state 替换，不等同于执行器命令。曲线重合时以同步数值为准。</p><p>每个视频都包含连续移动的头/主相机和左右腕相机、原始 frame_index、左右 state/cmd 数值、时间游标；未配置的相机保留原模型画布空区。最后一帧没有可训练的下一帧窗口，GT 标为 N/A。可用播放器减速；本 HTML 内嵌所有 MP4，下载后可直接用浏览器打开。</p>''')
for result in results:
 name=result['id'];r=rows[name];dest=OUT/name
 encoded=base64.b64encode((dest/'gripper.mp4').read_bytes()).decode()
 parts.append(f'<section id="{name}"><h2>{esc(name)}</h2><p>{esc(notes[name])}</p><p><small>episode {r["episode"]} · source frames [{r["start"]},{r["end"]}) · {r["fps"]} fps · {result["frames"]} 连续帧</small></p><video controls loop muted playsinline preload="none" src="data:video/mp4;base64,{encoded}"></video><p>State：<code>{esc(r["training_state"])}</code><br>CMD：<code>{esc(r["command_label"])}</code><br>发布 action：<code>{esc(r["released_action"])}</code>；source target offsets：<code>{esc(r["target_offsets"])}</code></p>')
 parts.append(f'<p>原数据：<code>{esc(r["source"])}</code></p><details><summary>原始字段范围、元数据与首帧数值</summary><div class="scroll"><table><tr><th>字段</th><th>min</th><th>max</th><th>超出 [0,1] 的帧数</th></tr>')
 for k,v in r['field_stats'].items():
  if '.mask' in k or 'velocity' in k or 'effort' in k:continue
  parts.append(f'<tr><td>{esc(k)}</td><td>{num(v["min"])}</td><td>{num(v["max"])}</td><td>{v["outside_01"]}</td></tr>')
 parts.append('</table></div><pre>'+esc(json.dumps({'normalization_metadata':r['source_gripper_normalization'],'field_metadata':r['field_metadata'],'frame0':result['example']},ensure_ascii=False,indent=2))+'</pre></details>')
 parts.append(f'<p><a href="{name}/gripper.mp4">独立 MP4</a> · <a href="{name}/frames.csv">逐帧 CSV（原始精度显示 6 位小数）</a> · <a href="{name}/source_gripper.npz">原始精度 NPZ</a></p></section>')
parts.append('<section><h2>全部 46 个片段核对结果</h2><p>“无越界”不代表闭合/张开的物理量程已验证。A1 窄分位数标定和 RoboMind v2 源量程仍值得进一步核实。</p><div class="scroll"><table><tr><th>片段</th><th>reader state 范围 L / R</th><th>cmd 字段</th><th>其他越界字段（排除 mask/速度/力）</th></tr>')
for r in audit:
 ranges=' / '.join(f'{r["field_stats"][r["training_state"]+"::"+s]["min"]:.6f}…{r["field_stats"][r["training_state"]+"::"+s]["max"]:.6f}' for s in ['left','right'])
 other=[k for k,v in r['field_stats'].items() if v['outside_01'] and not any(t in k for t in ['.mask','velocity','effort'])]
 parts.append(f'<tr><td>{esc(r["id"])}</td><td>{ranges}</td><td>{esc(r["command_label"])}</td><td>{esc(", ".join(other)) or "无"}</td></tr>')
parts.append('</table></div></section><p>构建脚本：outputs/curated_smoke_20261010/gripper_audit.py、render_gripper_audit.py。GT 来自当前生产 reader，并逐帧与源 state[t+1] 校验；视频使用已与源视频帧号对齐的训练 RGB 缓存。</p></html>')
(OUT/'index.html').write_text(''.join(parts))
(OUT/'video_validation.json').write_text(json.dumps({'scope':'46 curated clips; no full-corpus claim','state_finite':finite,'state_outside_01':outside,'synthetic_only_values':[-.2,1.4],'videos':results},ensure_ascii=False,indent=2))
print('REPORT',OUT/'index.html',flush=True)
