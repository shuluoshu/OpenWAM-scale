"""Independently re-decode exact current/target frames for the same-case report."""
import argparse
from pathlib import Path
import sys
import numpy as np
import torch
import yaml
CODE=Path(__file__).resolve().parents[2];sys.path.insert(0,str(CODE))
from scripts.experiments.infer_stage1_samecase import dump, sample_identity
from openwam.dataloader.curated_stage2 import CuratedStage2Dataset
from openwam.dataloader.stage2_diagnostics import report_goal_target
from openwam.model.goal.observation import clean_multiview


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(2);out=a.run/'report/stage1_samecase';out.mkdir(exist_ok=True,parents=True)
    samples=torch.load(a.run/'data/samples.pt',weights_only=False,map_location='cpu')
    cfg=yaml.safe_load((a.run/'comparison_config.yaml').read_text())['dataloader']
    cfg['normalization_registry']=None;cfg['metadata_only']=True
    ds=CuratedStage2Dataset.from_config(cfg);repos={r['repo']:r for r in ds.repos};results=[]
    for i,s in enumerate(samples):
        tr=s['trace'];r=repos[tr['identity'][0]];m=ds._episode(r,tr['episode'])
        goal=s['report_target']['goal_source_frame']
        if goal!=s['instruction_end_exclusive']-1:raise ValueError('wrong instruction endpoint')
        if str(m['root'])!=tr['source_repo']:raise ValueError('source repository mismatch')
        views,alignment=ds._decode_views(m,[tr['anchor']]);cameras=m['profile']['cameras']
        current=clean_multiview([views[0].get(c) for c in cameras],s['goal_view_valid'])
        if not torch.equal(current,s['goal_observed_multiview']):raise ValueError(f'current source mismatch: case {i}')
        saved=s['_eval']['report_goal']
        # Re-read the unique source goal row, endpoint and camera and recompute UV.
        image,gt=report_goal_target(ds,r,tr['episode'],m,s)
        if image is None or not np.array_equal(np.asarray(image),np.asarray(saved)):raise ValueError('goal diagnostic frame mismatch')
        old=s['report_target'].get('projection');new=gt.get('projection');residual=None
        if old is not None:
            if new is None:raise ValueError('source projection became unavailable')
            for key in ('projection_valid','in_frame'):
                if not np.array_equal(old[key],new[key]):raise ValueError('source projection validity changed')
            mask=old['projection_valid'] & old['in_frame']
            if mask.any():
                residual=float(np.abs((old['uv']-new['uv'])[mask]*[319,255]).max())
                if residual>1e-4:raise ValueError('source projection UV changed')
        row=dict(case=i,status='verified',identity=sample_identity(s),anchor_frame=tr['anchor'],goal_frame=goal,
            current_source_equal=True,target_source_equal=True,projection_max_abs_px=residual,
            source_alignment=alignment,recomputed_report_target=gt,
            scope='Exact source RGB decode and unique source goal-row calibration reprojection; not independent physical calibration.')
        folder=out/f'case_{i:03d}';folder.mkdir(exist_ok=True);dump(folder/'source_audit.json',row)
        results.append(dict(case=i,status='verified',current_source_equal=True,target_source_equal=True,projection_max_abs_px=residual))
        dump(out/'source_audit_summary.json',dict(status='running',cases=results));print(i,r['family'],'verified',flush=True)
    dump(out/'source_audit_summary.json',dict(status='complete',cases=results))

if __name__=='__main__':main()
