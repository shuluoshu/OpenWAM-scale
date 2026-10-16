"""Videos and mathematical checks sourced from the actual production reader targets."""
from pathlib import Path
import sys,json,argparse,traceback,os,importlib.util
os.environ.setdefault("MPLCONFIGDIR","/tmp/openwam-curated-matplotlib")
import numpy as np
import pyarrow.parquet as pq
import torch
import av,cv2
torch.set_num_threads(2)
cv2.setNumThreads(1)
from PIL import Image
from omegaconf import OmegaConf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from prepare import build_clip_reader
from openwam.dataloader.utils.video_io import decode_video_frames
from openwam.dataloader.transforms.multiview import assemble_multiview_layout
from openwam.dataloader.utils.mora_relative import _rotation
spec=importlib.util.spec_from_file_location('wan_pixel_preprocess',ROOT/'openwam/model/video_backbone/wan/preprocess.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);preprocess_video=module.preprocess_video
from scripts.data.urdf_fk_audit import pose7_to_matrix,eef20_to_matrix
from scripts.data.trajectory_projection import project_points
OUT=Path(__file__).resolve().parent
COLORS=[(60,210,255),(230,125,230)]
def write(path,x):path.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def text(im,s,xy,color=(245,245,245),scale=.5):cv2.putText(im,s,xy,cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)
def pose_from_raw(x):
 t=np.broadcast_to(np.eye(4),(*x.shape[:-1],4,4)).copy();t[...,:3,3]=x[...,:3];t[...,:3,:3]=_rotation(x[...,3:9])[0];return t

def absolute_tcp(cfg,src):
 n=len(src['frame_index']);t=np.broadcast_to(np.eye(4),(n,2,4,4)).copy();valid=np.zeros((n,2),bool)
 for a,side in enumerate(['left','right']):
  if side in cfg.get('tcp_pose_keys',{}):t[:,a]=pose7_to_matrix(src[cfg.tcp_pose_keys[side]]);valid[:,a]=True
  elif side in cfg.get('tcp_from_eef',{}):t[:,a]=eef20_to_matrix(src['observation.states.eef'])[:,a]@np.array(cfg.tcp_from_eef[side]);valid[:,a]=True
 if cfg.get('tcp_reference_to_world_key'):t=pose7_to_matrix(src[cfg.tcp_reference_to_world_key])[:,None]@t
 return t,valid

class LocalPlot:
 def __init__(self,enabled,limit):
  self.fig=plt.figure(figsize=(9.6,3),dpi=100,facecolor='#101927');self.axes=[];self.lines=[];self.markers=[];self.pose_lines=[]
  for a in range(2):
   ax=self.fig.add_subplot(1,2,a+1,projection='3d',facecolor='#101927');self.axes.append(ax)
   ax.set(xlim=(-limit,limit),ylim=(-limit,limit),zlim=(-limit,limit),xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)',title=('Left' if a==0 else 'Right')+' current TCP frame')
   ax.set_box_aspect((1,1,1));ax.view_init(elev=24,azim=-60);ax.tick_params(labelsize=6,colors='white');ax.title.set_color('white')
   for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:axis.label.set_color('white');axis.set_pane_color((.08,.12,.17,1))
   color='#ffd23c' if a==0 else '#e67de6'
   self.lines.append(ax.plot([],[],[],color=color,lw=2)[0]);self.markers.append(ax.plot([0],[0],[0],'wo',markersize=4)[0])
   pl=[]
   for c in ['#ff6565','#63ed94','#65a8ff']:pl.append(ax.plot([],[],[],color=c,lw=1.7)[0])
   self.pose_lines.append(pl)
   if not enabled:ax.text2D(.15,.55,'TCP MASKED / unavailable',transform=ax.transAxes,color='white',fontsize=11)
  self.fig.subplots_adjust(left=.04,right=.96,bottom=.13,top=.87,wspace=.16)
  self.length=max(.015,min(.06,limit*.3));self.enabled=enabled;self.cached=None
 def render(self,local,valid):
  if self.cached is not None and not self.enabled:return self.cached.copy()
  for a in range(2):
   p=local[:,a,:3,3].copy();p[~valid[:,a]]=np.nan
   self.lines[a].set_data_3d(p[:,0],p[:,1],p[:,2])
   for j,line in enumerate(self.pose_lines[a]):
    if valid[:,a].any():
     k=np.flatnonzero(valid[:,a])[-1];o=p[k];end=o+local[k,a,:3,j]*self.length;v=np.stack([o,end]);line.set_data_3d(v[:,0],v[:,1],v[:,2])
    else:line.set_data_3d([],[],[])
  self.fig.canvas.draw();rgb=np.asarray(self.fig.canvas.buffer_rgba())[...,:3].copy();bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR);self.cached=bgr;return bgr.copy()
 def close(self):plt.close(self.fig)

def render(identifier):
 dest=OUT/'clips'/identifier;r=json.loads((dest/'result.json').read_text());cfg=OmegaConf.load(dest/'data.yaml');reader=build_clip_reader(cfg);rep=reader.representation
 src=dict(np.load(dest/'source.npz',allow_pickle=False));info=json.loads((Path(r['source'])/'meta/info.json').read_text());meta=json.loads((dest/'source_episode.json').read_text())
 table=pq.read_table(dest/'dataset/data/chunk-000/file-000.parquet');columns={k:np.asarray(table[k].to_pylist()) for k in rep.columns};columns['_rows']=np.arange(len(table))
 poses,pose_valid=absolute_tcp(cfg,src);_,abs_mask=rep._absolute_values(columns);pose_valid &= abs_mask[:,[0,10]];n=len(table);indices=list(range(0,n-1,2));slots={};video_errors={};head_preview=[]
 cams=[r['head'],r['left'],r['right']]
 for index,cam in enumerate(cams):
  if cam is None:continue
  pre=f'videos/{cam}/';path=Path(r['source'])/info['video_path'].format(video_key=cam,chunk_index=meta[pre+'chunk_index'],file_index=meta[pre+'file_index'])
  offset=json.loads((dest/'video_offsets.json').read_text())[cam]['clip_base'];ids=[offset+i for i in indices]
  h,w=(256,320) if index==0 else (128,160)
  try:slots[cam]=decode_video_frames(str(path),ids,h,w)
  except Exception as e:
   if index==0:raise
   video_errors[cam]=f'{type(e).__name__}: {e}';continue
  with av.open(str(path)) as container:
   native_h,native_w=container.streams.video[0].height,container.streams.video[0].width
  native=decode_video_frames(str(path),[offset],native_h,native_w)[0]
  np.testing.assert_array_equal(np.asarray(native.resize((w,h),Image.Resampling.LANCZOS)),np.asarray(slots[cam][0]))
  native.save(dest/(['source_head.jpg','source_left.jpg','source_right.jpg'][index]),quality=92)
  slots[cam][0].save(dest/(['resized_head.png','resized_left.png','resized_right.png'][index]))
  if index==0:
   nh,nw=native_h,native_w;scale=min(640/nw,480/nh);ph,pw=round(nh*scale),round(nw*scale)
   head_preview=decode_video_frames(str(path),ids,ph,pw)
   display=(pw,ph,(640-pw)//2,(480-ph)//2,nw,nh)
 canvases=[assemble_multiview_layout({cam:frames[k] for cam,frames in slots.items()},[cams[0],cams[1] or '__missing_left__',cams[2] or '__missing_right__'],384,320) for k in range(len(indices))]
 for k in range(9):np.testing.assert_array_equal(np.array(canvases[indices.index(k*4)]),np.array(Image.open(dest/f'model_frame_{k}.png')))
 frames=[Image.open(dest/f'model_frame_{k}.png').convert('RGB') for k in range(9)]
 prevae=preprocess_video(frames,dtype=torch.bfloat16,device='cpu');assert list(prevae.shape)==[1,3,9,384,320]
 preprocessing=dict(shape=list(prevae.shape),dtype=str(prevae.dtype),range=[float(prevae.min()),float(prevae.max())],actual_function='openwam.model.video_backbone.wan.preprocess.preprocess_video',model_forward_or_vae_encoding_executed=False,all_nine_reader_frames_match_reconstructed_pipeline=True)
 # Fixed local-axis scale over all sliding windows, in metres.
 local_all=[]
 for j in indices:
  action,mask,_,_=rep.encode_window({k:v[j:min(n,j+33)] for k,v in columns.items()})
  for a in range(2):
   if mask[:,a*10].any():local_all.append(action[mask[:,a*10],a*10:a*10+3])
 limit=max(.04,max((float(abs(x).max()) for x in local_all),default=.04)*1.2)
 plot=LocalPlot(bool(pose_valid.any()),limit)
 output=av.open(str(dest/'trajectory.mp4'),'w');stream=output.add_stream('libx264',rate=15);stream.width=960;stream.height=832;stream.pix_fmt='yuv420p';stream.options={'crf':'22','preset':'fast'}
 max_position_error=0.;max_rotation_error=0.;projected_frames=0;head_valid_steps=0;tcp_valid_steps=0;head_supervised_frames=0;head_positive_depth_points=0;head_in_image_points=0
 kkey=next((k for k in [f'observation.states.cameras.{r["head"].split(".")[-1]}.intrinsic','observation.states.camera.intrinsic','head_camera_intrinsics'] if k in src),None)
 for vi,j in enumerate(indices):
  ids=np.arange(j,min(n,j+33));action,mask,_,_=rep.encode_window({k:v[ids] for k,v in columns.items()})
  local=np.broadcast_to(np.eye(4),(len(ids),2,4,4)).copy();valid=np.zeros((len(ids),2),bool)
  for a in range(2):
   local[1:,a]=pose_from_raw(action[:,a*10:a*10+9]);valid[:,a]=np.r_[pose_valid[j,a],mask[:,a*10]]
   if valid[:,a].any():
    restored=poses[j,a]@local[:,a];sel=valid[:,a]
    max_position_error=max(max_position_error,float(abs(restored[sel,:3,3]-poses[ids,a][sel,:3,3]).max()))
    max_rotation_error=max(max_rotation_error,float(abs(restored[sel,:3,:3]-poses[ids,a][sel,:3,:3]).max()))
  assert max_position_error<3e-5 and max_rotation_error<3e-5
  image=np.zeros((832,960,3),np.uint8);image[:]=(16,25,39)
  preview=cv2.cvtColor(np.asarray(head_preview[vi]),cv2.COLOR_RGB2BGR)
  pw,ph,px,py,nw,nh=display;image[32+py:32+py+ph,px:px+pw]=preview
  image[55:55+384,640:960]=cv2.cvtColor(np.asarray(canvases[vi]),cv2.COLOR_RGB2BGR)
  text(image,f'{identifier} | source frame {r["start"]+j} | GT future +1..+{len(ids)-1}',(10,22))
  text(image,'Reader canvas + GT overlay',(647,49),scale=.42)
  text(image,'video offsets: 0,4,...,32',(645,459),scale=.38)
  text(image,'white origin; L yellow / R purple',(645,479),scale=.33)
  text(image,'pose axes: X red / Y green / Z blue',(645,499),scale=.32)
  projected=False;supervised=False
  if mask[:,34:].any() and cfg.head_world_to_camera_key in columns and kkey:
   c=np.asarray(columns[cfg.head_world_to_camera_key][j]).reshape(4,4);K=np.asarray(src[kkey][j]).reshape(3,3)
   for a in range(2):
    hs=34+a*9;ok=mask[:,hs];head_valid_steps+=int(ok.sum())
    if not ok.any():continue
    supervised=True
    origin=c[:3,:3]@poses[j,a,:3,3]+c[:3,3]
    # Head deltas are vectors: restore anchor translation before pinhole projection.
    points=origin+np.vstack([np.zeros((1,3)),action[:,hs:hs+3]])
    target=poses[ids,a,:3,3]@c[:3,:3].T+c[:3,3]
    valid_head=np.r_[pose_valid[j,a],ok]
    np.testing.assert_allclose(points[valid_head],target[valid_head],atol=3e-5,rtol=0)
    head_rot=_rotation(action[:,hs+3:hs+9])[0]@(c[:3,:3]@poses[j,a,:3,:3])
    np.testing.assert_allclose(head_rot[ok],(c[:3,:3]@poses[ids[1:],a,:3,:3])[ok],atol=3e-5,rtol=0)
    pixels,pv=project_points(points,K);pv&=valid_head;head_positive_depth_points+=int(pv.sum())
    native_pixels=pixels.copy()
    model_pixels=native_pixels*np.array([320/nw,256/nh])+np.array([640,55])
    model_inside=pv&(model_pixels[:,0]>=640)&(model_pixels[:,0]<960)&(model_pixels[:,1]>=55)&(model_pixels[:,1]<311)
    for k in range(1,len(model_pixels)):
     if model_inside[k-1] and model_inside[k]:cv2.line(image,tuple(np.rint(model_pixels[k-1]).astype(int)),tuple(np.rint(model_pixels[k]).astype(int)),COLORS[a],1,cv2.LINE_AA)
    if model_inside[0]:cv2.circle(image,tuple(np.rint(model_pixels[0]).astype(int)),3,(255,255,255),-1)
    if pv.any():
     pixels[:,0]=pixels[:,0]*pw/nw+px;pixels[:,1]=pixels[:,1]*ph/nh+py+32
     inside=pv & (pixels[:,0]>=px)&(pixels[:,0]<px+pw)&(pixels[:,1]>=py+32)&(pixels[:,1]<py+32+ph)
     for k in range(1,len(pixels)):
      if inside[k-1] and inside[k]:cv2.line(image,tuple(np.rint(pixels[k-1]).astype(int)),tuple(np.rint(pixels[k]).astype(int)),COLORS[a],2,cv2.LINE_AA)
     if inside[0]:cv2.circle(image,tuple(np.rint(pixels[0]).astype(int)),5,(255,255,255),-1)
     head_in_image_points+=int(inside.sum());projected |= bool(inside.any())
  tcp_valid_steps+=int(valid[1:].sum());projected_frames+=int(projected);head_supervised_frames+=int(supervised)
  text(image,('Head TCP: pre-clip GT in CURRENT camera' if projected else 'Head TCP supervised; trajectory outside view / behind camera') if supervised else 'Head TCP MASKED / unavailable calibration or TCP',(10,523),scale=.47)
  image[532:832]=plot.render(local,valid)
  frame=av.VideoFrame.from_ndarray(image,format='bgr24')
  for packet in stream.encode(frame):output.mux(packet)
  if vi==0:cv2.imwrite(str(dest/'poster.jpg'),image)
 for packet in stream.encode():output.mux(packet)
 output.close();plot.close()
 with av.open(str(dest/'trajectory.mp4')) as check:
  count=sum(1 for f in check.decode(video=0));assert count==len(indices)
 metrics=dict(id=identifier,frame_count=count,duration_seconds=count/15,display_fps=15,source_fps=info['fps'],display_source_indices=[r['start']+x for x in indices],training_frame_sampling='30Hz source kept; video stride4; display alone uses every2 frames',prevae=preprocessing,source_pose_roundtrip_max_position_m=max_position_error if tcp_valid_steps else None,source_pose_roundtrip_max_rotation_matrix_error=max_rotation_error if tcp_valid_steps else None,head_projected_frames=projected_frames,head_supervised_frames=head_supervised_frames,head_positive_depth_points=head_positive_depth_points,head_in_image_points=head_in_image_points,head_valid_target_arm_steps=head_valid_steps,tcp_valid_target_arm_steps=tcp_valid_steps,projection_model='pinhole (only supported A1/DOSW1 head targets); ABC strict calibration masked',wrist_decode_errors=video_errors,supervision='Physical GT before quantile clipping; no model predictions',local_plot_limit_m=limit)
 write(dest/'render_metrics.json',metrics);print(identifier,'VIDEO OK',count,'head',projected_frames,flush=True);return metrics

def main():
 p=argparse.ArgumentParser();p.add_argument('--ids',nargs='*');args=p.parse_args();results=[]
 dirs=[p for p in sorted((OUT/'clips').iterdir()) if (p/'result.json').exists()]
 for d in dirs:
  if args.ids and d.name not in args.ids:continue
  try:results.append(render(d.name))
  except Exception as e:print(d.name,'FAIL',traceback.format_exc(),flush=True);results.append(dict(id=d.name,error=f'{type(e).__name__}: {e}'))
 write(OUT/'render_results.json',results)
 if any('error'in r for r in results):raise SystemExit(1)
if __name__=='__main__':main()
