"""Read real time-varying wrist frames/calibration and production relative targets."""
from pathlib import Path
import sys,json,hashlib,argparse
import numpy as np
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from PIL import Image
from omegaconf import OmegaConf
HERE=Path(__file__).resolve().parent;REPORT=HERE.parent
sys.path.insert(0,str(REPORT));sys.path.insert(0,str(REPORT.parents[1]))
from prepare import build_clip_reader
from openwam.dataloader.utils.mora_relative import _rotation
from openwam.dataloader.utils.video_io import decode_video_frames

def poses(raw):
 t=np.broadcast_to(np.eye(4),(*raw.shape[:-1],4,4)).copy()
 t[...,:3,3]=raw[...,:3];t[...,:3,:3]=_rotation(raw[...,3:9])[0];return t
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n')

def prepare(r):
 ident=r['id'];clip=REPORT/'clips'/ident;dest=HERE/ident
 cfg=OmegaConf.load(clip/'data.yaml');reader=build_clip_reader(cfg);rep=reader.representation
 table=pq.read_table(clip/'dataset/data/chunk-000/file-000.parquet')
 columns={k:np.asarray(table[k].to_pylist()) for k in rep.columns};columns['_rows']=np.arange(len(table))
 absolute,abs_mask=rep._absolute_values(columns)
 world_all=np.stack([poses(absolute[:,:9]),poses(absolute[:,10:19])],axis=1)
 n=len(table);indices=np.arange(0,n-1,2);count=len(indices)
 local=np.broadcast_to(np.eye(4),(count,33,2,4,4)).copy();world=local.copy();valid=np.zeros((count,33,2),bool)
 horizon=np.zeros(count,dtype=int);target_ids=np.full((count,33),-1,dtype=int);maxerr=0.
 for vi,j in enumerate(indices):
  end=min(n,j+33);action,mask,_,_=rep.encode_window({k:v[j:end] for k,v in columns.items()});steps=end-j
  assert not mask[:,34:].any(), 'ABC head mask must remain disabled'
  for a,start in enumerate([0,10]):
   local[vi,1:steps,a]=poses(action[:,start:start+9]);valid[vi,0,a]=abs_mask[j,start];valid[vi,1:steps,a]=mask[:,start]
   world[vi,:steps,a]=world_all[j,a]@local[vi,:steps,a]
   if valid[vi,:steps,a].any():
    err=np.max(np.abs(world[vi,:steps,a][valid[vi,:steps,a]]-world_all[j:end,a][valid[vi,:steps,a]]));maxerr=max(maxerr,float(err))
  horizon[vi]=steps-1;target_ids[vi,:steps]=r['start']+np.arange(j,end)
 assert maxerr<2e-6
 calibration_columns=['frame_index']+[f'observation.states.cameras.cam_{s}_wrist.{field}' for s in ['left','right'] for field in ['intrinsic','world_to_camera']]
 expr=(ds.field('episode_index')==r['episode'])&(ds.field('frame_index')>=r['start'])&(ds.field('frame_index')<r['end'])
 cal=ds.dataset(r['source_parquet'],format='parquet').to_table(columns=calibration_columns,filter=expr).sort_by([('frame_index','ascending')])
 np.testing.assert_array_equal(cal['frame_index'].to_numpy(),np.arange(r['start'],r['end']))
 arrays={'local':local,'world_recovered':world,'valid':valid,'horizon':horizon,'target_source_frame_indices':target_ids,
         'anchor_source_frame_indices':r['start']+indices,'anchor_local_frame_indices':indices}
 info=json.loads((Path(r['source'])/'meta/info.json').read_text());episode=json.loads((clip/'source_episode.json').read_text());offsets=json.loads((clip/'video_offsets.json').read_text())
 images={}
 for side in ['left','right']:
  key=r[side];prefix=f'videos/{key}/';path=Path(r['source'])/info['video_path'].format(video_key=key,chunk_index=episode[prefix+'chunk_index'],file_index=episode[prefix+'file_index'])
  base=offsets[key]['clip_base'];video_ids=(base+indices).tolist()
  native_image=Image.open(dest/f'image_source_{side}.png').convert('RGB');nw,nh=native_image.size
  # Explicit read lets sandbox errors propagate instead of hiding them as decode failures.
  with path.open('rb') as f:f.read(1)
  native_first=decode_video_frames(str(path),[base],nh,nw)[0].convert('RGB')
  np.testing.assert_array_equal(np.asarray(native_first),np.asarray(native_image))
  scale=min(640/nw,480/nh);w,h=round(nw*scale),round(nh*scale)
  frames=decode_video_frames(str(path),video_ids,h,w);assert len(frames)==count
  directory=dest/f'wrist_motion_{side}';directory.mkdir(exist_ok=True)
  hashes=[];motion=[];prev=None
  for i,image in enumerate(frames):
   rgb=np.asarray(image.convert('RGB'));hashes.append(hashlib.sha256(rgb.tobytes()).hexdigest())
   if prev is not None:motion.append(float(np.abs(rgb.astype(float)-prev).mean()))
   prev=rgb.astype(float);image.save(directory/f'{i:03d}.png')
  assert len(set(hashes))>1, f'{ident}/{side} camera frames are frozen'
  cprefix=f'observation.states.cameras.cam_{side}_wrist.'
  K=np.asarray(cal[cprefix+'intrinsic'].to_pylist()).reshape(n,3,3)[indices]
  C=np.asarray(cal[cprefix+'world_to_camera'].to_pylist()).reshape(n,4,4)[indices]
  arrays[side+'_K']=K;arrays[side+'_world_to_camera']=C
  images[side]={'directory':directory.name,'native_size_wh':[nw,nh],'display_size_wh':[w,h],
   'video_path':str(path),'video_frame_indices':video_ids,'source_frame_indices':(r['start']+indices).tolist(),
   'source_timestamps_s':(float(episode[prefix+'from_timestamp'])+(r['start']+indices)/r['fps']).tolist(),
   'frame_sha256':hashes,'unique_background_frames':len(set(hashes)),'mean_adjacent_pixel_difference':float(np.mean(motion)),
   'first_source_frame_matches_existing_anchor':True,
   'finite_extrinsic_frames':int(np.isfinite(C).all(axis=(1,2)).sum()),
   'extrinsic_max_change_from_first':float(np.nanmax(np.abs(C-C[:1]))) if np.isfinite(C).any() else None}
 np.savez_compressed(dest/'abc_wrist_motion_arrays.npz',**arrays)
 result={'id':ident,'episode':r['episode'],'source_start':r['start'],'source_end_exclusive':r['end'],
  'source_fps':r['fps'],'display_fps':15,'display_frames':count,'duration_s':count/15,
  'background':'real decoded source frames, advancing two source frames per displayed frame',
  'window':'re-anchor TCP-local and current wrist extrinsics at each displayed source frame; future +1..+32',
  'world_recovery_max_matrix_error':maxerr,'head_mask_unchanged':True,'images':images}
 write(dest/'abc_wrist_motion_source.json',result);print(ident,{s:images[s]['unique_background_frames'] for s in images},'recovery',maxerr,flush=True)
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--ids',nargs='*');args=p.parse_args()
 rows=json.loads((REPORT/'manifest.json').read_text())['results']
 results=[prepare(r) for r in rows if r['family']=='abc_130k' and (not args.ids or r['id'] in args.ids)]
 write(HERE/'abc_wrist_motion_preparation.json',results)
