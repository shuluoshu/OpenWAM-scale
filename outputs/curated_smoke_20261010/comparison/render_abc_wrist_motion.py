"""Real moving wrist-camera videos with per-frame current-anchor TCP forecasts."""
from pathlib import Path
import os,sys,json,hashlib
os.environ.setdefault('MPLCONFIGDIR','/tmp/openwam-abc-motion-mpl')
import numpy as np
import cv2,av
OUT=Path(__file__).resolve().parent;REPORT=OUT.parent
sys.path.insert(0,str(REPORT))
from render import LocalPlot
from render_pairs import draw_projection,cvtext,write

def render(r):
 ident=r['id'];dest=OUT/ident
 source=json.loads((dest/'abc_wrist_motion_source.json').read_text())
 gate=json.loads((dest/'wrist_geometry_eligibility.json').read_text())
 z=dict(np.load(dest/'abc_wrist_motion_arrays.npz'))
 count=source['display_frames'];assert count==75
 xyz=z['local'][...,:3,3];limit=max(.025,float(np.max(np.abs(xyz[z['valid']])))*1.15)
 plot=LocalPlot(True,limit)
 filename='abc_wrist_tcp_motion.mp4'
 output=av.open(str(dest/filename),'w',options={'movflags':'+faststart'})
 stream=output.add_stream('libx264',rate=15);stream.width=1280;stream.height=1000;stream.pix_fmt='yuv420p';stream.options={'crf':'19','preset':'fast'}
 cameras={s:{'projected_frames':0,'visible_pose_points':0,'background_hashes':[]} for s in ['left','right']}
 colors=[(90,220,255),(220,120,240)];posters=[]
 for k in range(count):
  current=int(z['anchor_source_frame_indices'][k]);horizon=int(z['horizon'][k])
  canvas=np.full((1000,1280,3),(43,28,17),np.uint8)
  cvtext(canvas,f'{ident} | REAL CAMERA VIDEO | source frame {current} | future +1..+{horizon}',(12,25),.64)
  cvtext(canvas,'DIAGNOSTIC: wrist calibration NOT strict-QA-passed. Images, TCP anchor and camera pose update together.',(12,55),.54,(110,185,255))
  for a,side in enumerate(['left','right']):
   meta=source['images'][side];path=dest/meta['directory']/f'{k:03d}.png'
   image=cv2.imread(str(path));h,w=image.shape[:2];x=640*a+(640-w)//2;y=80+(480-h)//2
   sha=hashlib.sha256(image[:,:,::-1].tobytes()).hexdigest()
   assert sha==meta['frame_sha256'][k], 'background must be the matching decoded source frame'
   cameras[side]['background_hashes'].append(sha);canvas[y:y+h,x:x+w]=image
   geom=(x,y,w,h,w,h);C=z[side+'_world_to_camera'][k];K=z[side+'_K'][k]
   numeric=bool(np.isfinite(C).all() and np.isfinite(K).all())
   enabled=bool(gate['wrist_geometry_available']) and numeric
   if enabled:
    assert np.allclose(C[:3,:3]@C[:3,:3].T,np.eye(3),atol=1e-4)
    assert np.isclose(np.linalg.det(C[:3,:3]),1,atol=1e-4)
    poses=C[None]@z['world_recovered'][k,:horizon+1,a]
    inside=draw_projection(canvas,poses[:,:3,3],K,meta['native_size_wh'],geom,colors[a],z['valid'][k,:horizon+1,a],poses[:,:3,:3],axis_length=.025)
    cameras[side]['projected_frames']+=1;cameras[side]['visible_pose_points']+=inside
   else:cvtext(canvas,'Wrist extrinsics unavailable: original video only',(640*a+12,315),.56,(90,160,255))
   cvtext(canvas,('LEFT wrist / LEFT TCP' if a==0 else 'RIGHT wrist / RIGHT TCP')+f' | {(current-r["start"])/r["fps"]:.2f}s',(640*a+12,584),.55,colors[a])
  chart=plot.render(z['local'][k,:horizon+1],z['valid'][k,:horizon+1])
  canvas[600:1000]=cv2.resize(chart,(1280,400),interpolation=cv2.INTER_AREA)
  cvtext(canvas,'Current TCP-local frame is re-anchored each frame. White=current TCP. RGB=terminal pose axes.',(12,997),.48)
  if k in [0,37,74]:posters.append(canvas.copy())
  frame=av.VideoFrame.from_ndarray(canvas,format='bgr24')
  for packet in stream.encode(frame):output.mux(packet)
 for packet in stream.encode():output.mux(packet)
 output.close();plot.close()
 with av.open(str(dest/filename)) as f:
  decoded=[frame.to_ndarray(format='bgr24') for frame in f.decode(video=0)]
 assert len(decoded)==count
 assert np.mean(np.abs(decoded[0][80:560].astype(float)-decoded[37][80:560]))>0
 for side in cameras:
  hashes=cameras[side].pop('background_hashes');cameras[side]['unique_background_frames']=len(set(hashes))
  assert len(set(hashes))>1
 cv2.imwrite(str(dest/'abc_wrist_tcp_motion.jpg'),posters[1])
 cv2.imwrite(str(dest/'abc_wrist_tcp_motion_contact_sheet.jpg'),np.vstack([cv2.resize(p,(960,750)) for p in posters]))
 result={'id':ident,'video':filename,'frames':count,'fps':15,'duration_s':5.0,'moving_background_verified':True,
  'source_frame_start':int(z['anchor_source_frame_indices'][0]),'source_frame_last_displayed':int(z['anchor_source_frame_indices'][-1]),
  'camera_and_tcp_anchor':'updated per displayed source frame; future trajectory projected using that current camera',
  'future_horizon':32,'tail':'shorten future horizon to valid remaining source rows, no invented padding trajectory',
  'cameras':cameras,'world_recovery_max_matrix_error':source['world_recovery_max_matrix_error'],
  'training_masks_changed':False,'calibration_qa':gate}
 write(dest/'abc_wrist_tcp_motion_metrics.json',result)
 print(ident,'75 frames / 5 sec',cameras,flush=True);return result

if __name__=='__main__':
 manifest=json.loads((REPORT/'manifest.json').read_text())['results']
 rows=[render(r) for r in manifest if r['family']=='abc_130k'];write(OUT/'abc_wrist_motion_results.json',rows)
 parts=['<section class="panel" id="abc-wrist-tcp"><h2>ABC：真实腕部相机视频 + 每帧 TCP relative 轨迹</h2>',
 '<p><strong>已替换固定首帧动画：现在背景是真实连续腕相机视频。</strong>读取原150帧片段，每两源帧显示一次，75帧 / 15fps / 5秒。左右图像、TCP锚点、腕相机内外参按同一源帧同步更新；每帧叠加当前时刻之后最多32步的轨迹。片段末尾按剩余真实步数缩短。</p>',
 '<p>上排是左右腕相机真实画面，下排是各自<strong>当前 TCP-local</strong>坐标系的未来轨迹。投影先计算 <code>T_world_future = T_world_tcp(t) × T_relative(t,k)</code>，再乘 <code>T_wristCamera_world(t)</code>；两者均随播放更新。白点=当前 TCP；黄/紫=左/右轨迹；RGB=未来姿态轴。当前 TCP 与腕相机刚性连接，白点可能在图像上近似固定，这是正常现象；环境画面在真实运动。</p>',
 '<p class="warn">08、10、11 仍仅作诊断：使用原数据 QA rejected 的腕相机标定。09 外参为 NaN，播放真实腕相机视频和 TCP-local 曲线，但不伪造图像投影。没有调整 TCP 偏移或训练 mask。</p>']
 for r in rows:
  ident=r['id'];p='comparison/'+ident+'/'
  parts += [f'<h3>{ident} · 源帧 {r["source_frame_start"]}…{r["source_frame_last_displayed"]}</h3>',
   f'<video controls preload="none" playsinline poster="{p}abc_wrist_tcp_motion.jpg"><source src="{p}abc_wrist_tcp_motion.mp4" type="video/mp4"></video>',
   '<p>真实运动检查：左右相机分别有 '+str(r['cameras']['left']['unique_background_frames'])+' / '+str(r['cameras']['right']['unique_background_frames'])+' 张不同的背景图像，图像 hash 与对应源帧逐条核对。</p>',
   '<p>'+' · '.join(f'<a href="{p}{file}">{label}</a>' for file,label in [('abc_wrist_tcp_motion.mp4','下载连续视频'),('abc_wrist_tcp_motion_contact_sheet.jpg','首 / 中 / 尾画面对照'),('abc_wrist_tcp_motion_metrics.json','运动与投影验证'),('abc_wrist_motion_source.json','源帧时刻与图像 hash'),('abc_wrist_motion_arrays.npz','逐帧 pose / 相机数组')])+'</p>']
 parts += ['</section>'];(OUT/'abc_wrist.html').write_text('\n'.join(parts))
