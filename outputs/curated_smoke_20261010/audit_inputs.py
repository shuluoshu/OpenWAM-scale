"""Verify saved actual model targets and quantify pre-clip versus model GT."""
from pathlib import Path
import sys,json
import numpy as np
import pyarrow.parquet as pq
from omegaconf import OmegaConf
from prepare import build_clip_reader,write
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from openwam.dataloader.utils.normalization import apply_normalization,NORM_EPS
OUT=Path(__file__).resolve().parent
GROUPS={'tcp_position_m':[0,1,2,10,11,12],'tcp_rotation6d':list(range(3,9))+list(range(13,19)), 'gripper':[9,19],'joint_rad':list(range(20,34)), 'head_position_m':[34,35,36,43,44,45], 'head_rotation6d':list(range(37,43))+list(range(46,52))}

def audit(dest):
 cfg=OmegaConf.load(dest/'data.yaml');reader=build_clip_reader(cfg);rep=reader.representation
 table=pq.read_table(dest/'dataset/data/chunk-000/file-000.parquet');columns={k:np.asarray(table[k].to_pylist()) for k in rep.columns};columns['_rows']=np.arange(len(table))
 model=np.load(dest/'model_inputs.npz');src=np.load(dest/'source.npz');raw,mask,_,_=rep.encode_window({k:v[:33] for k,v in columns.items()})
 expected=np.zeros((32,112),np.float32);expected[:,rep.scatter]=np.where(mask,apply_normalization(raw,reader._normalization_stats,'quantile'),0)
 expected_mask=np.zeros((32,112),bool);expected_mask[:,rep.scatter]=mask
 np.testing.assert_array_equal(expected_mask,model['action_mask']);np.testing.assert_array_equal(expected,model['action'])
 assert np.all(model['action'][~model['action_mask']]==0)
 joint_error=0.;joint_dims=0
 for a,side in enumerate(['left','right']):
  idx=rep.joint_indices[side];joint_dims+=len(idx)
  if idx:
   joints=np.asarray(src[rep.joint_state_key],np.float32)[:,idx];delta=joints[1:33]-joints[0]
   np.testing.assert_allclose(raw[:,20+a*7:20+a*7+len(idx)],delta,atol=1e-6,rtol=0)
   joint_error=max(joint_error,float(np.max(np.abs(raw[:,20+a*7:20+a*7+len(idx)]-delta))))
 qlo=reader._normalization_stats['q01'];qhi=reader._normalization_stats['q99'];scale=np.maximum(qhi-qlo,NORM_EPS)
 summary={k:{'valid_cells':0,'clipped_cells':0,'max_physical_change':0.} for k in GROUPS}
 for j in range(len(table)-1):
  x,m,_,_=rep.encode_window({k:v[j:min(j+33,len(table))] for k,v in columns.items()})
  scaled=(x-qlo)/scale*2-1;clipped=(scaled<-1)|(scaled>1)
  physical=(np.clip(scaled,-1,1)+1)*.5*scale+qlo
  for group,ids in GROUPS.items():
   valid=m[:,ids];entry=summary[group];entry['valid_cells']+=int(valid.sum());entry['clipped_cells']+=int((clipped[:,ids]&valid).sum())
   entry['max_physical_change']=max(entry['max_physical_change'],float(np.max(np.abs(physical[:,ids]-x[:,ids])[valid])) if valid.any() else 0.)
 def sample(k):
  return {'future_step':k+1,'source_frame':int(src['frame_index'][k+1]),'left_tcp_xyz_m':raw[k,:3].tolist() if mask[k,0] else None,'right_tcp_xyz_m':raw[k,10:13].tolist() if mask[k,10] else None,'left_head_delta_m':raw[k,34:37].tolist() if mask[k,34] else None,'right_head_delta_m':raw[k,43:46].tolist() if mask[k,43] else None,'joint14_rad':np.where(mask[k,20:34],raw[k,20:34],np.nan).tolist(),'model_action112':model['action'][k].tolist(),'model_action_mask112':model['action_mask'][k].tolist()}
 samples=[sample(k) for k in [0,15,31]]
 for sample in samples:sample['joint14_rad']=[None if not np.isfinite(x) else x for x in sample['joint14_rad']]
 validity={}
 for key in src.files:
  if 'valid' in key or key.endswith('.mask'):
   v=src[key];validity[key]={'shape':list(v.shape),'all_finite':bool(np.isfinite(v).all()),'minimum':float(np.min(v)),'maximum':float(np.max(v)),'zero_cells':int((v==0).sum()),'consumed_by_reader':False}
 result={'actual_normalized_scatter_exact_match':True,'joint_vs_fixed_anchor_max_error_rad':joint_error if joint_dims else None,'joint_supervised_dimensions':joint_dims,'tcp_local_supervised':bool(mask[:,[0,10]].any()),'head_supervised':bool(mask[:,[34,43]].any()),'gripper_supervised_dimensions':int(mask[0,[9,19]].sum()),'first_window_valid_action_cells':int(mask.sum()),'clip_stats_scope':'all 149 local anchors, only futures remaining inside the selected 150 source rows','clipping':summary,'examples':samples,'available_source_validity_fields':validity,'source_validity_limitation':'Production relative reader currently checks finite/rigid transforms and configured schema; it does not consume source confidence/validity masks.'}
 write(dest/'input_audit.json',result);return result
if __name__=='__main__':
 results=[]
 for d in sorted((OUT/'clips').iterdir()):
  if (d/'result.json').exists():results.append({'id':d.name,**audit(d)})
 write(OUT/'input_audit.json',{'count':len(results),'results':results});print('Verified',len(results),'actual normalized 32x112 targets, mask, fixed-anchor joints and clip statistics')
