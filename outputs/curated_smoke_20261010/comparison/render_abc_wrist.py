"""Diagnostic ABC wrist projections using stored, explicitly non-QA-passed calibration."""
from pathlib import Path
import json,html,os
os.environ.setdefault("MPLCONFIGDIR","/tmp/openwam-abc-wrist-mpl")
import numpy as np
import cv2,av
import matplotlib.pyplot as plt
from render_pairs import fit_background,project_display,draw_projection,axes3d,cvtext,BG,write

OUT=Path(__file__).resolve().parent
REPORT=OUT.parent

def render(row):
 dest=OUT/row['id'];z=dict(np.load(dest/'gt_trajectories.npz'));cal=dict(np.load(dest/'wrist_calibration_anchor.npz'))
 gate=json.loads((dest/'wrist_geometry_eligibility.json').read_text())
 local=np.concatenate([np.tile(np.eye(4),(1,2,1,1)),z['ow_local']],axis=0)
 world=z['ow_tcp_anchor'][None]@local
 direct=np.concatenate([z['ow_tcp_anchor'][None],z['ow_tcp_world']],axis=0)
 error=float(np.max(np.abs(world-direct)));assert error<2e-6
 valid=np.concatenate([z['ow_anchor_mask'][None],z['ow_local_mask']],axis=0)
 assert not z['ow_head_mask'].any()
 cameras=[];details={};arrays={'tcp_local':local,'tcp_world_recovered':world,'tcp_pose_valid':valid}
 colors=[(90,220,255),(220,120,240)]
 for a,side in enumerate(['left','right']):
  prefix=f'observation.states.cameras.cam_{side}_wrist.'
  K=cal[prefix+'intrinsic'].reshape(3,3).astype(float);C=cal[prefix+'world_to_camera'].reshape(4,4).astype(float)
  numeric=bool(np.isfinite(C).all() and np.isfinite(K).all())
  if numeric:
   assert np.allclose(C[:3,:3]@C[:3,:3].T,np.eye(3),atol=1e-4)
   assert np.isclose(np.linalg.det(C[:3,:3]),1,atol=1e-4)
  enabled=numeric and bool(gate['wrist_geometry_available'])
  path=dest/f'image_source_{side}.png';native=cv2.imread(str(path));nh,nw=native.shape[:2]
  bg,g=fit_background(path,640*a,80,640,480)
  poses=C[None]@world[:,a] if enabled else None
  info={'numeric_calibration':numeric,'diagnostic_projection_enabled':enabled,'strict_wrist_camera_eef_eligible':False,
        'source_image_size_wh':[nw,nh],'K':K.tolist(),'C':C.tolist() if numeric else None}
  if enabled:
   uv,inside=project_display(poses[:,:3,3],K,[nw,nh],g);inside &= valid[:,a]
   info.update(anchor_pixel_native=((uv[0]-np.array(g[:2])+.5)*np.array([nw/g[2],nh/g[3]])-.5).tolist(),
      anchor_depth_m=float(poses[0,2,3]),visible_pose_count=int(inside.sum()),pose_count=len(poses))
   arrays[side+'_tcp_in_fixed_wrist']=poses
  cameras.append((bg,g,K,[nw,nh],poses,enabled));details[side]=info
 fig=plt.figure(figsize=(12.8,4.4),dpi=100,facecolor=BG);axes=[];dots=[];lines=[];triads=[];axis_lengths=[]
 for a in range(2):
  ax=fig.add_subplot(1,2,a+1,projection='3d',facecolor=BG);axes.append(ax)
  xyz=local[:,a,:3,3];limit=max(.025,float(np.max(np.abs(xyz)))*1.2)
  ax.set(xlim=(-limit,limit),ylim=(-limit,limit),zlim=(-limit,limit),xlabel='X / m',ylabel='Y / m',zlabel='Z / m')
  ax.set_title(('LEFT' if a==0 else 'RIGHT')+' TCP-local: inv(T0) @ Tk',color='white',fontsize=11)
  ax.set_box_aspect((1,1,1));ax.view_init(23,-58);ax.tick_params(colors='white',labelsize=7)
  for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:axis.label.set_color('white');axis.set_pane_color((.07,.11,.17,1))
  ax.plot(*xyz.T,color='#f7d858' if a==0 else '#e879db',alpha=.25,lw=1)
  line,=ax.plot([],[],[],color='#f7d858' if a==0 else '#e879db',lw=2);lines.append(line)
  ax.scatter([0],[0],[0],color='white',s=22)
  dot,=ax.plot([],[],[],'o',color='#f7d858' if a==0 else '#e879db',markersize=5);dots.append(dot)
  triads.append([ax.plot([],[],[],color=c,lw=1.5)[0] for c in ['#ff6262','#72e49b','#6caeff']]);axis_lengths.append(limit*.17)
  axes3d(ax,np.zeros(3),np.eye(3),limit*.17)
 fig.subplots_adjust(left=.03,right=.97,top=.87,bottom=.12,wspace=.1)
 fig.text(.5,.025,'Pre-clip physical GT | White: current TCP; colored dot: future TCP | RGB axes: XYZ orientation',ha='center',color='white',fontsize=10)
 video=av.open(str(dest/'abc_wrist_tcp.mp4'),'w',options={'movflags':'+faststart'});stream=video.add_stream('libx264',rate=10)
 stream.width=1280;stream.height=1040;stream.pix_fmt='yuv420p';stream.options={'crf':'19','preset':'fast'}
 for k in range(33):
  canvas=np.full((1040,1280,3),(43,28,17),np.uint8)
  cvtext(canvas,f'{row["id"]} | FIXED wrist images at source {row["start"]} | reveal t+{k} / 32',(12,25),.64)
  cvtext(canvas,'DIAGNOSTIC ONLY: wrist calibration NOT strict-QA-passed. Training masks unchanged.',(12,55),.62,(110,185,255))
  for a,(bg,g,K,native,poses,enabled) in enumerate(cameras):
   x,y,w,h,_,_=g;canvas[y:y+h,x:x+w]=bg
   if enabled:
    draw_projection(canvas,poses[:k+1,:3,3],K,native,g,colors[a],valid[:k+1,a],poses[:k+1,:3,:3],axis_length=.025)
    uv,ok=project_display(poses[k:k+1,:3,3],K,native,g)
    if ok[0] and valid[k,a]:cv2.circle(canvas,tuple(np.rint(uv[0]).astype(int)),5,colors[a],-1)
   else:cvtext(canvas,'NO WRIST EXTRINSICS: projection unavailable',(a*640+15,325),.58,(90,160,255))
   cvtext(canvas,('LEFT wrist / LEFT TCP' if a==0 else 'RIGHT wrist / RIGHT TCP'),(a*640+12,585),.57,colors[a])
   xyz=local[:k+1,a,:3,3];lines[a].set_data_3d(*xyz.T);dots[a].set_data_3d(*xyz[-1:,].T)
   for j,axis_line in enumerate(triads[a]):
    line=np.stack([xyz[-1],xyz[-1]+local[k,a,:3,j]*axis_lengths[a]]);axis_line.set_data_3d(*line.T)
  cvtext(canvas,'White=current TCP. Yellow=left; purple=right. Axes: X red / Y green / Z blue.',(12,612),.52)
  fig.canvas.draw();chart=np.asarray(fig.canvas.buffer_rgba())[:,:,:3][:,:,::-1]
  canvas[600:1040]=chart
  # Footer legend is placed in the chart margin rather than over source images.
  frame=av.VideoFrame.from_ndarray(canvas,format='bgr24')
  for packet in stream.encode(frame):video.mux(packet)
 for packet in stream.encode():video.mux(packet)
 video.close();plt.close(fig);cv2.imwrite(str(dest/'abc_wrist_tcp.jpg'),canvas)
 with av.open(str(dest/'abc_wrist_tcp.mp4')) as handle:count=sum(1 for _ in handle.decode(video=0))
 assert count==33
 np.savez_compressed(dest/'abc_wrist_tcp_arrays.npz',**arrays)
 metrics={'id':row['id'],'episode':row['episode'],'anchor_source_frame':row['start'],'video_frames':33,'display_fps':10,
  'physical_horizon_s':32/30,'display_duration_s':3.3,'fixed_current_camera':True,'source_gate':gate,'cameras':details,
  'world_recovery_max_matrix_error':error,'head_training_mask_unchanged':True,'projection_is_diagnostic_not_training_eligible':True}
 write(dest/'abc_wrist_tcp_metrics.json',metrics);print(row['id'],details,flush=True);return metrics

if __name__=='__main__':
 manifest=json.loads((REPORT/'manifest.json').read_text())['results']
 rows=[render(r) for r in manifest if r['family']=='abc_130k'];write(OUT/'abc_wrist_results.json',rows)
 parts=['<section class="panel" id="abc-wrist-tcp"><h2>ABC：腕部相机视角下的 TCP relative pose</h2>',
 '<p>上排左右腕相机固定在当前 t0，分别投影对应手臂的未来32步 TCP；下排展示同一条轨迹在当前 TCP-local 坐标系中的位置。视频逐步展开未来目标，原图固定，不是随动相机回放。</p>',
 '<p><code>T_world_future = T_world_tcp0 × T_tcp_relative</code>，再用当前腕相机 <code>T_camera_world(t0)</code> 转换并投影；没有把 relative xyz 直接当成相机坐标。轨迹为裁剪前物理 GT，端点是当前 reader 恢复的双指接触中心。白点为当前 TCP；黄/紫为左/右轨迹，红绿蓝轴为 XYZ 姿态方向。</p>',
 '<p class="warn"><strong>仅供诊断验收：</strong>08、10、11 有标定数值，但发布 QA 为 rejected；09 的腕相机外参为 NaN，不能投影。这些标定不作为已验证的监督依据，未修改 head mask、训练配置或 TCP 偏移。可观察当前白点是否落在夹爪末端虚拟中心；若不落在目标点，不能靠数学回代自洽宣称标定准确。</p>']
 for r in rows:
  ident=r['id'];p='comparison/'+ident+'/'
  parts += [f'<h3>{ident} · episode {r["episode"]} · anchor {r["anchor_source_frame"]}</h3>',
   f'<video controls preload="none" playsinline poster="{p}abc_wrist_tcp.jpg"><source src="{p}abc_wrist_tcp.mp4" type="video/mp4"></video>',
   '<p>'+('诊断投影：使用原数据标定，QA rejected。' if r['cameras']['left']['diagnostic_projection_enabled'] else '腕相机外参缺失（NaN），上排只显示原图；下排 TCP-local 轨迹有效。')+f'显示3.3秒，实际未来跨度1.067秒；恢复世界位姿最大矩阵元素差 {r["world_recovery_max_matrix_error"]:.3g}。</p>',
   '<p>'+ ' · '.join(f'<a href="{p}{name}">{label}</a>' for name,label in [('abc_wrist_tcp.mp4','下载 MP4'),('abc_wrist_tcp.jpg','完整轨迹图'),('abc_wrist_tcp_metrics.json','标定状态与投影数值'),('abc_wrist_tcp_arrays.npz','轨迹数组'),('wrist_calibration_anchor.npz','原始腕相机 K / 外参')])+'</p>']
 parts += ['</section>'];(OUT/'abc_wrist.html').write_text('\n'.join(parts))
