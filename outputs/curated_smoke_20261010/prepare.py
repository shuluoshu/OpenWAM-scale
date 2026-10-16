"""Run the production OpenWAM reader on exact source clips, without changing source data."""
from pathlib import Path
import sys, json, hashlib, argparse, copy, traceback
import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from omegaconf import OmegaConf
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from openwam.dataloader.registry import build_dataset
from openwam.dataloader.utils.lerobotv3 import compute_file_local_offsets
from openwam.dataloader.utils.mora_relative import _rotation
from scripts.data.prepare_openwam_relative_stats import prepare_relative_stats
OUT=Path(__file__).resolve().parent
YAML=(ROOT/'../../pretrain/launch/moravla/train_files/pretrain_v0_3_curated_smoke.yaml').resolve()

def safe(x):
 if isinstance(x,(np.ndarray,np.generic)):return x.tolist()
 if isinstance(x,Path):return str(x)
 raise TypeError(type(x).__name__)
def write(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=safe)+'\n')

def source_profile(repo,info):
 f=info['features'];family=repo.split('/')[0]
 videos=[k for k,v in f.items() if v.get('dtype')=='video']
 head=next((k for k in ['observation.images.cam_high','observation.images.cam_front','observation.images.camera_front','observation.images.head_left','observation.images.egocentric','observation.images.front_1'] if k in videos),None)
 if head is None:raise ValueError(f'No explicitly recognized head camera: {videos}')
 left=next((k for k in ['observation.images.cam_left_wrist','observation.images.camera_left'] if k in videos),None)
 right=next((k for k in ['observation.images.cam_right_wrist','observation.images.camera_right'] if k in videos),None)
 cfg=dict(type='canonical_eef_lerobot',action_representation='head_tcp_relative_joint112',state_key='observation.states.eef',joint_state_key='observation.states.joint',tcp_pose_keys={},head_camera=head,left_wrist_camera=left,right_wrist_camera=right,head_world_to_camera_key='observation.states.camera.world_to_camera',num_frames=33,video_stride=4,window_stride=1,height=384,width=320,multiview=True,normalize_mode='quantile',unify_action=True,unify_action_map=['0-9','34-43','80-93','94-111'],episode_indices=[0],max_episodes=1,max_windows=None,return_trace=True)
 reason='No declared physical TCP or verified fixed transform in this source profile; pose supervision masked.'
 if family=='interndata_a1':
  cfg['tcp_pose_keys']={s:f'states.{s}_tcp_to_robot_pose' for s in ['left','right']};cfg['tcp_reference_to_world_key']='states.robot_to_env_pose';reason='Native TCP pose columns, mapped robot -> world using per-row robot_to_env.'
 if family=='abc_130k':
  hist=info.get('starvla_metadata',{}).get('rectification_history',[])
  h=next((x for x in hist if x.get('schema_version')=='eef_tcp_to_wrist_0804_v1'),None)
  if h:
   basis=np.eye(4)
   if any(x.get('schema_version')=='abc_130k_eef_axis_x_forward_z_up_v1' for x in hist):basis[:3,:3]=np.eye(3)[:,[2,0,1]]
   cfg['tcp_from_eef']={x['arm']:(np.linalg.inv(basis)@np.linalg.inv(np.array(x['T_source_wrist_row_major']).reshape(4,4))).tolist() for x in h['transforms']}
   reason='TCP restored from recorded official MJCF contact-center -> wrist transform and recorded axis basis; source EEF is wrist.'
 if family in ['abc_130k','robomind_v1','robomind_v2']:
  cfg['joint_state_key']='observation.state'
  feature=f.get('observation.state',{});names=feature.get('names') or {}
  if isinstance(names,dict):names=names.get('motors',[])
  if family=='robomind_v1':cfg['joint_indices']={'left':list(range(6)),'right':list(range(7,13))}
  elif not names:raise ValueError(f'Need explicit joint ordering: {repo}')
 if family in ['LightWheel','mecka','egodex','egoverse']:
  reason='Human wrist/hand or virtual hand frame; no physical TCP configured under current no-TCP-mask policy. No arm joint supervision.'
 audit=next(x for x in json.loads((OUT/'source_contract_audit.json').read_text())['repositories'] if x['repo']==repo)
 overrides=dict(audit['recommended_reader_configuration']);head_ok=overrides.pop('head_camera_available',True)
 cfg.update(overrides)
 if not head_ok:cfg['head_world_to_camera_key']='audit.unavailable_head_calibration'
 reason=audit['tcp']['status']+': '+audit['tcp'].get('endpoint','')
 return cfg,reason

def source_video_offsets(source, cameras, episode, start, fps):
 """Preserve original merged-video ordinals when extracting a one-episode fixture."""
 files=sorted((source/'meta/episodes').rglob('*.parquet'))
 columns=['episode_index','length','dataset_from_index']
 for cam in cameras:
  if cam:columns.extend(f'videos/{cam}/{key}' for key in ['chunk_index','file_index','from_timestamp'])
 eps=ds.dataset([str(p) for p in files],format='parquet').to_table(columns=columns).to_pandas()
 selected=np.flatnonzero(eps.episode_index.to_numpy()==episode)
 assert len(selected)==1
 offsets={}
 for cam in cameras:
  if not cam:continue
  prefix=f'videos/{cam}/';base=int(compute_file_local_offsets(eps,prefix+'chunk_index',prefix+'file_index')[selected[0]])
  timestamp_base=round(float(eps.iloc[selected[0]][prefix+'from_timestamp'])*fps)
  if base!=timestamp_base:raise ValueError(f'Video cumsum/timestamp disagreement {cam}: {base} != {timestamp_base}')
  offsets[cam]={'original_episode_base':base,'clip_base':base+start,'timestamp_base_verified':timestamp_base}
 return offsets

def build_clip_reader(config):
 """Report-only fixture adapter; production decode/representation are unchanged.

 data.yaml alone addresses the local Parquet; this wrapper is required to restore
 original video offsets, whose calculation otherwise loses preceding episodes.
 """
 reader=build_dataset(config)
 offsets=json.loads((Path(config.dataset_dir).parent/'video_offsets.json').read_text())
 for cam,entry in offsets.items():reader._ep_video_frame_offsets[cam][0]=entry['clip_base']
 return reader

def read_clip(source,info,episode,start,end,needed):
 files=sorted((source/'meta/episodes').rglob('*.parquet'))
 names=pq.ParquetFile(files[0]).schema_arrow.names
 columns=[k for k in names if not k.startswith('stats/')]
 table=ds.dataset([str(p) for p in files],format='parquet').to_table(columns=columns,filter=ds.field('episode_index')==episode)
 if len(table)!=1:raise ValueError(f'Expected one episode {episode}, got {len(table)}')
 meta=table.to_pylist()[0]
 if not 0<=start<end<=meta['length']:raise ValueError(f'Segment {start}:{end} outside episode length {meta["length"]}')
 data=source/info['data_path'].format(chunk_index=meta['data/chunk_index'],file_index=meta['data/file_index'])
 names=pq.ParquetFile(data).schema_arrow.names
 columns=[k for k in names if k in needed]
 expr=(ds.field('episode_index')==episode)&(ds.field('frame_index')>=start)&(ds.field('frame_index')<end)
 rows=ds.dataset(str(data),format='parquet').to_table(columns=columns,filter=expr).sort_by([('frame_index','ascending')])
 np.testing.assert_array_equal(rows['frame_index'].to_numpy(),np.arange(start,end))
 return meta,rows,data

def prepare(number,repo,segment,root):
 episode,start,end=segment;identifier=f'{number:02d}_{repo.split("/")[0]}';dest=OUT/'clips'/identifier;dest.mkdir(parents=True,exist_ok=True)
 source=root/repo;info=json.loads((source/'meta/info.json').read_text());cfg,reason=source_profile(repo,info)
 needed={'frame_index','episode_index','index','timestamp','task_index',cfg['state_key'],cfg['joint_state_key'],'states.robot_to_env_pose','states.left_tcp_to_robot_pose','states.right_tcp_to_robot_pose',cfg['head_world_to_camera_key']}
 needed.update(k for k in info['features'] if any(s in k for s in ['intrinsic','distortion','canonical.frame_valid','training.valid','source.camera.world_to_camera_valid','observation.states.eef.mask']))
 if repo.split('/')[0] in ['robomind_v1','robomind_v2','robochallenge_v1']:needed.add('observation.state')
 meta,table,data_path=read_clip(source,info,episode,start,end,needed)
 original={k:np.asarray(table[k].to_pylist()) for k in table.column_names}
 np.savez_compressed(dest/'source.npz',**original)
 write(dest/'source_episode.json',meta)
 write(dest/'video_offsets.json',source_video_offsets(source,[cfg['head_camera'],cfg['left_wrist_camera'],cfg['right_wrist_camera']],episode,start,info['fps']))
 adapter=[]
 if repo.split('/')[0] in ['robomind_v1','robomind_v2','robochallenge_v1']:
  state=np.asarray(table['observation.state'].to_pylist(),dtype=np.float32)
  feature=info['features']['observation.state'];names=feature.get('names') or {}
  if isinstance(names,dict):names=names.get('motors',[])
  grippers=[next(i for i,name in enumerate(names) if side in str(name).lower() and 'grip' in str(name).lower()) for side in ['left','right']]
  eef=np.zeros((len(table),20),np.float32);eef[:,[9,19]]=state[:,grippers]
  if not np.isfinite(eef[:,[9,19]]).all() or np.any((eef[:,[9,19]]<0)|(eef[:,[9,19]]>1)):raise ValueError('Source normalized gripper out of range')
  cfg['state_key']='audit.normalized_gripper_eef20'
  table=table.append_column(cfg['state_key'],pa.array(eef.tolist(),type=pa.list_(pa.float32(),20)))
  info['features'][cfg['state_key']]={'dtype':'float32','shape':[20],'names':None}
  adapter.append(dict(operation='gripper-only EEF20 for existing reader interface; no fabricated pose',source='observation.state',source_indices=grippers,destination_indices=[9,19],reason='Use normalized source joint-state grippers; source native EEF grippers unavailable/unconfirmed'))
 local=dest/'dataset';(local/'meta/episodes/chunk-000').mkdir(parents=True,exist_ok=True);(local/'data/chunk-000').mkdir(parents=True,exist_ok=True)
 length=end-start
 for key,values in [('frame_index',np.arange(length)),('index',np.arange(length)),('episode_index',np.zeros(length,dtype=int)),('timestamp',np.arange(length,dtype=float)/info['fps'])]:
  i=table.column_names.index(key);table=table.set_column(i,key,pa.array(values,type=table.schema.field(key).type))
 pq.write_table(table,local/'data/chunk-000/file-000.parquet')
 lm=copy.deepcopy(meta);lm.update(episode_index=0,length=length,dataset_from_index=0,dataset_to_index=length)
 lm['data/chunk_index']=0;lm['data/file_index']=0
 for k in list(lm):
  if k.startswith('videos/') and k.endswith('/from_timestamp'):lm[k]=float(meta[k])+start/info['fps']
  elif k.startswith('videos/') and k.endswith('/to_timestamp'):
   lm[k]=float(meta[k.replace('/to_timestamp','/from_timestamp')])+end/info['fps']
 pq.write_table(pa.Table.from_pylist([lm]),local/'meta/episodes/chunk-000/file-000.parquet')
 li=copy.deepcopy(info);li.update(total_episodes=1,total_frames=length,splits={'train':'0:1'},data_path='data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet')
 li['features']={k:v for k,v in li['features'].items() if k in table.column_names or v.get('dtype')=='video'}
 write(local/'meta/info.json',li)
 for name in ['tasks.parquet','stats.json']:
  target=local/'meta'/name
  if not target.exists() and not target.is_symlink() and (source/'meta'/name).exists():target.symlink_to(source/'meta'/name)
 cfg.update(dataset_id=identifier,dataset_dir=str(local),video_root=str(source),normalization_stats_path=str(dest/'stats.npy'))
 config=OmegaConf.create(cfg);OmegaConf.save(config,dest/'data.yaml');prepare_relative_stats(config)
 reader=build_clip_reader(config)
 checks=[]
 for anchor in [0,(length-33)//2,length-33,length-2]:
  s=reader[anchor];trace=s['trace'];assert s['action'].shape==(32,112)
  actual=min(32,length-anchor-1)
  assert not s['action_mask'][actual:].any()
  assert trace['action_target_frame_indices']==[anchor+i+1 if i<actual else None for i in range(32)]
  if anchor==0:
   np.savez_compressed(dest/'model_inputs.npz',action=s['action'].numpy(),action_mask=s['action_mask'].numpy(),proprio=s['proprio'].numpy(),proprio_mask=s['proprio_mask'].numpy(),video_mask=s['video_mask'].numpy(),raw_action_eef20=trace['raw_action_eef20'],raw_action_joint14=trace['raw_action_joint14'],raw_action_head18=trace['raw_action_head18'])
   for index,img in enumerate(s['video']):img.save(dest/f'model_frame_{index}.png')
   write(dest/'trace.json',trace)
  checks.append(dict(anchor_local=anchor,anchor_source=start+anchor,real_future_steps=actual,valid_action_cells=int(s['action_mask'].sum()),video_mask=s['video_mask'].tolist(),wrist_decode_failures=reader._wrist_fail_count))
 result=dict(id=identifier,repo=repo,family=repo.split('/')[0],episode=episode,start=start,end=end,fps=info['fps'],source=str(source),source_parquet=str(data_path),head=cfg['head_camera'],left=cfg['left_wrist_camera'],right=cfg['right_wrist_camera'],tcp_reason=reason,tcp_configured=bool(cfg.get('tcp_pose_keys') or cfg.get('tcp_from_eef')),joint_indices=reader.representation.joint_indices,adapter=adapter,checks=checks,status='ok',source_info_sha256=hashlib.sha256((source/'meta/info.json').read_bytes()).hexdigest())
 write(dest/'result.json',result);print(identifier,'OK',reason,flush=True);return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--ids',nargs='*',type=int);args=p.parse_args()
 c=OmegaConf.to_container(OmegaConf.load(YAML),resolve=True)['datasets']['vla_data']['pipeline'];root=Path(c['dataset']['data_root_dir'])
 results=[]
 for number,(repo,segments) in enumerate(c['resolver']['segments_by_repo'].items()):
  if args.ids is not None and number not in args.ids:continue
  assert len(segments)==1
  try:result=prepare(number,repo,segments[0],root)
  except Exception as e:
   result=dict(id=f'{number:02d}_{repo.split("/")[0]}',repo=repo,status='failed',error=f'{type(e).__name__}: {e}',traceback=traceback.format_exc());print(json.dumps(result),flush=True)
  results.append(result);write(OUT/('manifest.json' if args.ids is None else 'manifest_subset.json'),dict(yaml=str(YAML),yaml_sha256=hashlib.sha256(YAML.read_bytes()).hexdigest(),source_root=str(root),selection='segments_by_repo, lines108-298; half-open original source-frame intervals',source_fps_preserved=True,mora_15fps_not_applied=True,results=results))
 if any(x['status']!='ok' for x in results):raise SystemExit(1)
if __name__=='__main__':main()
