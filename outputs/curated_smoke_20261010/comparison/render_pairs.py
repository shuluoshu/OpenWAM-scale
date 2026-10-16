"""Matched source-anchor plots: canonical EEF versus declared TCP, never renamed."""
from pathlib import Path
import sys,os,json,argparse
os.environ.setdefault('MPLCONFIGDIR','/tmp/openwam-paired-mpl')
import numpy as np
import cv2,av
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent;REPORT=OUT.parent;ROOT=REPORT.parents[1]
sys.path.insert(0,str(ROOT))
from scripts.data.trajectory_projection import project_points
cv2.setNumThreads(1)
MORA='#69dbe9';OW='#ffbf63';POST='#ff657d';BG='#111c2b'

def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def cvtext(im,s,xy,scale=.52,color=(235,235,235)):
 cv2.putText(im,s,xy,cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)
def curve(ax,poses,mask,color,label,style='-',alpha=1):
 p=poses[:,:3,3].copy();p[~mask]=np.nan;p=np.vstack([np.zeros(3),p]);ax.plot(p[:,0],p[:,1],p[:,2],style,color=color,lw=1.8,alpha=alpha,label=label)
def axes3d(ax,origin,rotation,length,alpha=1):
 for j,c in enumerate(['#ff6262','#72e49b','#6caeff']):
  line=np.stack([origin,origin+rotation[:,j]*length]);ax.plot(*line.T,color=c,lw=1,alpha=alpha)
def plot_pose_comparison(dest,z):
 fig=plt.figure(figsize=(14,9),dpi=110,facecolor=BG);axs=[]
 for row,branch in enumerate(['head','local']):
  for a in range(2):
   ax=fig.add_subplot(2,2,row*2+a+1,projection='3d',facecolor=BG);axs.append(ax)
   parts=[]
   for prefix in ['mora','ow']:
    poses=z[f'{prefix}_{branch}'][:,a];mask=z[f'{prefix}_{branch}_mask'][:,a]
    if mask.any():parts.append(poses[mask,:3,3])
   limit=max(.025,max((float(np.max(np.abs(x))) for x in parts),default=.025)*1.2)
   ax.set(xlim=(-limit,limit),ylim=(-limit,limit),zlim=(-limit,limit),xlabel='X / m',ylabel='Y / m',zlabel='Z / m')
   ax.set_box_aspect((1,1,1));ax.view_init(elev=23,azim=-58);ax.tick_params(colors='white',labelsize=7)
   ax.set_title(('L' if a==0 else 'R')+(' - current HEAD axes' if branch=='head' else ' - own current endpoint LOCAL axes'),color='white',fontsize=11)
   for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:axis.label.set_color('white');axis.set_pane_color((.07,.11,.17,1))
   any_curve=False
   for prefix,color,label in [('mora',MORA,'Mora canonical EEF (50)'),('ow',OW,'OpenWAM physical TCP (32)')]:
    poses=z[f'{prefix}_{branch}'][:,a];mask=z[f'{prefix}_{branch}_mask'][:,a]
    if mask.any():
     curve(ax,poses,mask,color,label,alpha=.45);any_curve=True
     shared=z['shared_'+('mora' if prefix=='mora' else 'ow')+'_indices'];sm=np.zeros(len(mask),bool);sm[shared]=True;sm &=mask
     if sm.any():curve(ax,poses,sm,color,'same target frames' if prefix=='mora' else None)
     last=np.flatnonzero(mask)[-1];R=poses[last,:3,:3]
     if branch=='head':R=R@z['world_to_camera'][:3,:3]@z['mora_eef_anchor' if prefix=='mora' else 'ow_tcp_anchor'][a,:3,:3]
     axes3d(ax,poses[last,:3,3],R,limit*.16)
   om=z[f'ow_{branch}_mask'][:,a]
   if om.any():curve(ax,z[f'ow_model_{branch}'][:,a],om,POST,'OpenWAM after quantile clip','--')
   ax.scatter([0],[0],[0],color='white',s=18)
   if not any_curve:ax.text2D(.05,.5,'MASKED / no trusted geometry',transform=ax.transAxes,color='white')
   else:ax.legend(loc='upper left',fontsize=7,facecolor='#24334a',labelcolor='white',framealpha=.8)
 fig.suptitle(dest.name+' | same source anchor | physical-unit GT comparison',color='white',fontsize=15)
 fig.text(.5,.02,'Solid emphasis = identical target source frames. Endpoint separation is not a calibration-error metric.\nHead panels share camera axes but each subtracts its own current endpoint; local panels use different endpoint frames.',ha='center',color='#b7c8dc',fontsize=10)
 fig.subplots_adjust(left=.04,right=.96,top=.92,bottom=.09,hspace=.23,wspace=.05)
 fig.savefig(dest/'gt_comparison.png',facecolor=BG);plt.close(fig)

def fit_background(path,x0,y0,w,h):
 image=cv2.imread(str(path));ih,iw=image.shape[:2];scale=min(w/iw,h/ih);fw,fh=round(iw*scale),round(ih*scale);resized=cv2.resize(image,(fw,fh),interpolation=cv2.INTER_LINEAR)
 return resized,(x0+(w-fw)//2,y0+(h-fh)//2,fw,fh,iw,ih)
def project_display(points,K,native_wh,geom):
 uv,valid=project_points(points,K);nw,nh=native_wh;x,y,w,h,iw,ih=geom
 # Two image resizes, no crop: each uses a half-pixel center coordinate convention.
 uv=(uv+.5)*np.array([iw/nw,ih/nh])-.5
 uv=(uv+.5)*np.array([w/iw,h/ih])-.5+np.array([x,y])
 inside=valid&(uv[:,0]>=x)&(uv[:,0]<x+w)&(uv[:,1]>=y)&(uv[:,1]<y+h)
 return uv,inside

def draw_projection(image,points,K,native_wh,geom,color,mask,rotations=None,axis_length=.025):
 uv,inside=project_display(points,K,native_wh,geom);inside &=mask
 for j in range(1,len(uv)):
  if inside[j-1] and inside[j]:cv2.line(image,tuple(np.rint(uv[j-1]).astype(int)),tuple(np.rint(uv[j]).astype(int)),color,2,cv2.LINE_AA)
 if inside[0]:cv2.circle(image,tuple(np.rint(uv[0]).astype(int)),4,(255,255,255),-1)
 if rotations is not None:
  for ix in sorted(set([0,len(points)-1])):
   if not inside[ix]:continue
   for k,col in enumerate([(90,90,255),(125,240,140),(255,155,100)]):
    endpoint=points[ix]+rotations[ix,:,k]*axis_length;axis_uv,ok=project_display(np.stack([points[ix],endpoint]),K,native_wh,geom)
    if ok.all():cv2.line(image,tuple(np.rint(axis_uv[0]).astype(int)),tuple(np.rint(axis_uv[1]).astype(int)),col,1,cv2.LINE_AA)
 return int(inside.sum())

def video_comparison(dest,z,imeta):
 mora_image,mg=fit_background(dest/'mora_head_final.png',0,58,640,410)
 ow_image,og=fit_background(REPORT/'clips'/dest.name/'resized_head.png',640,58,640,410)
 native=imeta['cameras']['head']['native_size_wh'];K=z['intrinsic'];c=z['world_to_camera'];anchor=int(z['anchor_source_frame_index']);mids=z['mora_target_source_frame_indices'];oids=z['ow_target_source_frame_indices'];shared=set(z['shared_target_source_frame_indices'].tolist())
 furthest=int(max(np.max(mids),np.max(oids)));max_step=furthest-anchor
 if not 1<=max_step<=120:raise ValueError(f'Unexpected target time range {max_step}')
 out=av.open(str(dest/'gt_projection_compare.mp4'),'w',options={'movflags':'+faststart'});stream=out.add_stream('libx264',rate=10);stream.width=1280;stream.height=690;stream.pix_fmt='yuv420p';stream.options={'crf':'20','preset':'fast'}
 totals={'mora_visible_points':0,'ow_visible_points':0};first_full=None
 for step in range(1,max_step+1):
  image=np.full((690,1280,3),(35,25,17),np.uint8)
  for bg,g in [(mora_image,mg),(ow_image,og)]:x,y,w,h,_,_=g;image[y:y+h,x:x+w]=bg
  cvtext(image,'MoraVLA | canonical EEF | actual Qwen-processed head',(12,25),.57)
  cvtext(image,'OpenWAM | declared TCP | actual head slot',(652,25),.57)
  cvtext(image,f'FIXED image / source anchor {anchor}; reveal targets up to source frame {anchor+step}',(12,49),.52)
  for prefix,g,ids,col in [('mora',mg,mids,(235,220,105)),('ow',og,oids,(100,190,255))]:
   take=(ids<=anchor+step)&(ids>=0);indices=np.flatnonzero(take)
   for a in range(2):
    mask=z[f'{prefix}_head_mask'][:,a];anchor_pose=z['mora_eef_anchor' if prefix=='mora' else 'ow_tcp_anchor'][a];origin=c[:3,:3]@anchor_pose[:3,3]+c[:3,3];rot0=c[:3,:3]@anchor_pose[:3,:3]
    points=origin+np.vstack([np.zeros(3),z[f'{prefix}_head'][indices,a,:3,3]])
    rotations=np.concatenate([rot0[None],z[f'{prefix}_head'][indices,a,:3,:3]@rot0])
    valid=np.r_[bool(z['mora_anchor_mask' if prefix=='mora' else 'ow_anchor_mask'][a]),mask[indices]]
    color=col if a==0 else tuple(int(v*.7) for v in col)
    totals[prefix+'_visible_points']+=draw_projection(image,points,K,native,g,color,valid,rotations)
    if prefix=='ow' and len(indices):
     mp=origin+np.vstack([np.zeros(3),z['ow_model_head'][indices,a,:3,3]])
     # Sparse magenta markers denote de-normalized, quantile-clipped model GT.
     pixels,inside=project_display(mp,K,native,g);inside &=valid
     for k in np.flatnonzero(inside)[::3]:cv2.circle(image,tuple(np.rint(pixels[k]).astype(int)),2,(160,95,255),-1)
  cvtext(image,f'Mora targets: {mids[0]}..{mids[-1]} (50) | OpenWAM: {oids[0]}..{oids[-1]} (32)',(16,500),.6)
  cvtext(image,f'Matched target-frame intersection: {min(shared) if shared else "none"}..{max(shared) if shared else "none"} ({len(shared)} steps)',(16,532),.55)
  cvtext(image,'White: current endpoint. Axes: X red / Y green / Z blue. Light=L; darker=R.',(16,564),.52)
  cvtext(image,'Mora cyan = pre-existing EEF; OpenWAM orange = TCP. Pink dots = OpenWAM post-clip GT.',(16,596),.52)
  cvtext(image,'Fixed head camera at anchor; poses restored before projection. Reveal speed is display-only.',(16,628),.52)
  cvtext(image,'This video compares different endpoints and time horizons; it is NOT a prediction/error overlay.',(16,661),.5)
  frame=av.VideoFrame.from_ndarray(image,format='bgr24')
  for packet in stream.encode(frame):out.mux(packet)
  first_full=image
 for packet in stream.encode():out.mux(packet)
 out.close();cv2.imwrite(str(dest/'gt_projection_compare.jpg'),first_full)
 with av.open(str(dest/'gt_projection_compare.mp4')) as handle:count=sum(1 for _ in handle.decode(video=0))
 assert count==max_step
 return {'video_frames':count,'video_duration_s':count/10,'display_fps':10,'display_semantic':'fixed observed image and anchor; progressively reveal target frames; NOT real-time source playback',**totals}

def render(dest):
 e=json.loads((dest/'gt_evidence.json').read_text());imeta=json.loads((dest/'image_metrics.json').read_text());z=dict(np.load(dest/'gt_trajectories.npz'))
 if not z['ow_local_mask'].any():return {'id':dest.name,'plot':False,'video':False,'reason':'OpenWAM pose masked; no fabricated paired geometry'}
 assert 'mora_local' in z
 plot_pose_comparison(dest,z);metrics={'id':dest.name,'plot':True,'video':False,'anchor_source_frame':int(z['anchor_source_frame_index']),'endpoint_note':'Mora canonical EEF and OpenWAM TCP are not the same physical point'}
 if z['shared_ow_indices'].size:
  oi,mi=z['shared_ow_indices'],z['shared_mora_indices'];sep=np.linalg.norm(z['ow_tcp_world'][oi,:,:3,3]-z['mora_eef_world'][mi,:,:3,3],axis=-1)
  valid=z['ow_local_mask'][oi]&z['mora_local_mask'][mi]
  metrics['same_time_endpoint_distance_m']={'mean':float(sep[valid].mean()),'max':float(sep[valid].max()),'interpretation':'physical endpoint discrepancy, not a calibration-error metric'} if valid.any() else None
 metrics['anchor_endpoint_distance_m']=np.linalg.norm(z['ow_tcp_anchor'][:,:3,3]-z['mora_eef_anchor'][:,:3,3],axis=-1).tolist()
 metrics['shared_target_count']=len(z['shared_ow_indices'])
 if e['calibration']['trusted_for_projection'] and z['ow_head_mask'].any() and z['mora_head_mask'].any():metrics.update(video=True,**video_comparison(dest,z,imeta))
 write(dest/'gt_render_metrics.json',metrics);print(dest.name,'paired plot',metrics['video'],flush=True);return metrics
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--ids',nargs='*');args=p.parse_args();results=[]
 for d in sorted(OUT.iterdir()):
  if not d.is_dir() or not (d/'gt_evidence.json').exists() or not (d/'image_metrics.json').exists():continue
  if args.ids and d.name not in args.ids:continue
  results.append(render(d))
 write(OUT/'render_results.json',results)
