"""Audit actual source state/command fields; synthetic test bounds are not data."""
from pathlib import Path
import sys,json
import numpy as np
import pyarrow.parquet as pq
import pyarrow.dataset as ds
from omegaconf import OmegaConf
ROOT=Path(__file__).resolve().parents[2];TRAIN=ROOT/'outputs/curated_train_20261011'
OUT=Path(__file__).resolve().parent/'gripper_state_cmd_20261013';OUT.mkdir(exist_ok=True)

def indices(key,feature):
 shape=feature.get('shape',[]);names=feature.get('names') or []
 if isinstance(names,dict):names=names.get('motors',names.get('eef',[]))
 if shape in ([],[1]) and ('gripper' in key.lower() or ('end_effector_' in key and key.endswith('_position_align'))):
  return {side:None for side in ['left','right'] if side in key}
 if len(shape)!=1:return {}
 if isinstance(names,list) and len(names)==shape[0]:
  found={}
  for j,name in enumerate(names):
   n=str(name).lower()
   if 'grip' not in n:continue
   if 'left' in n or n.startswith('lgrip'):found['left']=j
   if 'right' in n or n.startswith('rgrip'):found['right']=j
  if found:return found
 if shape==[20] and ('eef' in key or names==['eef']):return {'left':9,'right':19}
 return {}

def summary(x,frames):
 good=np.isfinite(x);v=x[good];outside=good&((x<0)|(x>1))
 ids=np.flatnonzero(outside)
 return {'min':float(v.min()) if len(v) else None,'max':float(v.max()) if len(v) else None,
         'finite':int(good.sum()),'nonfinite':int((~good).sum()),'outside_01':len(ids),
         'examples':[{'frame':int(frames[i]),'value':float(x[i])} for i in ids[:3]]}

rows=json.loads((TRAIN/'data_manifest.json').read_text());report=[]
for row in rows:
 dest=OUT/row['id'];dest.mkdir(exist_ok=True)
 info=json.loads((Path(row['source'])/'meta/info.json').read_text());features=info['features']
 names=pq.ParquetFile(row['source_parquet']).schema_arrow.names
 mappings={k:indices(k,features.get(k,{})) for k in names if any(t in k for t in ['action','state','grip','cmd','end_effector_'])}
 mappings={k:v for k,v in mappings.items() if v}
 extra=[k for k in ['action.state_target_frame_offset','action.eef_target_frame_offset','action.state_target_frame_index'] if k in names]
 filt=(ds.field('episode_index')==row['episode'])&(ds.field('frame_index')>=row['start'])&(ds.field('frame_index')<row['end'])
 table=ds.dataset(row['source_parquet'],format='parquet').to_table(columns=['frame_index',*mappings,*extra],filter=filt).sort_by([('frame_index','ascending')])
 frames=table['frame_index'].to_numpy();np.testing.assert_array_equal(frames,np.arange(row['start'],row['end']))
 arrays={'frame_index':frames};stats={};metadata={}
 for key,mapping in mappings.items():
  raw=np.asarray(table[key].to_pylist(),dtype=float)
  for side,j in mapping.items():
   x=raw.reshape(-1) if j is None else raw[:,j];tag=f'{key}::{side}';arrays[tag]=x;stats[tag]=summary(x,frames)
  metadata[key]={k:features[key][k] for k in ['gripper_unit','unit','source_unit','unit_conversion','gripper_correction','status','source_key'] if k in features.get(key,{})}
 for key in extra:arrays[key]=table[key].to_numpy()
 cfg=OmegaConf.load(TRAIN/'data'/row['id']/'data.yaml')
 training_state='observation.state' if str(cfg.state_key).startswith('audit.') else str(cfg.state_key)
 assert all(f'{training_state}::{side}' in arrays for side in ['left','right'])
 # Name preserved source action explicitly: a released future-state target is
 # never presented as a hardware command.
 if all(f'master_actions.{side}_gripper.openness::{side}' in arrays for side in ['left','right']):
  cmd_keys=[f'master_actions.{side}_gripper.openness::{side}' for side in ['left','right']];cmd_label='master_actions.*_gripper.openness'
 elif all(f'actions.source_action::{side}' in arrays for side in ['left','right']):
  cmd_keys=[f'actions.source_action::{side}' for side in ['left','right']];cmd_label='actions.source_action (preserved)'
 elif all(f'action.position::{side}' in arrays for side in ['left','right']):
  cmd_keys=[f'action.position::{side}' for side in ['left','right']];cmd_label='action.position (preserved diagnostic command)'
 elif all(f'action::{side}' in arrays for side in ['left','right']) and not extra and info.get('starvla_metadata',{}).get('action_source')!='observation.state_timestamp_lookup' and info.get('starvla_metadata',{}).get('action',{}).get('semantic')!='absolute_future_observation_state':
  cmd_keys=[f'action::{side}' for side in ['left','right']];cmd_label='action (hardware command unverified)'
 else:cmd_keys=[];cmd_label='No independent command field identified'
 canonical=next((k for k in ['action','actions.eef'] if all(f'{k}::{s}' in arrays for s in ['left','right'])),None)
 result={**{k:row[k] for k in ['id','family','repo','source','episode','start','end','fps']},'training_state':training_state,
         'command_keys':cmd_keys,'command_label':cmd_label,'released_action':canonical,'action_source':info.get('starvla_metadata',{}).get('action_source'),
         'action_semantic':info.get('starvla_metadata',{}).get('action'),
         'source_gripper_normalization':info.get('starvla_metadata',{}).get('gripper_normalization'),
         'field_stats':stats,'field_metadata':metadata,'target_offsets':{k:np.unique(arrays[k]).tolist() for k in extra if 'offset' in k}}
 np.savez_compressed(dest/'source_gripper.npz',**arrays);(dest/'metadata.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));report.append(result)
 train_out=sum(stats[f'{training_state}::{s}']['outside_01'] for s in ['left','right'])
 command_out=sum(stats[k]['outside_01'] for k in cmd_keys)
 raw_out=[k for k,v in stats.items() if v['outside_01']]
 print(json.dumps({'id':row['id'],'state_outside':train_out,'cmd_outside':command_out,'cmd':cmd_label,'other_outside_fields':raw_out}),flush=True)
(OUT/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print('AUDITED',len(report),flush=True)
