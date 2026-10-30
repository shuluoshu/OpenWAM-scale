"""Materialize bounded native task/subtask anchors from exactly three episodes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import hashlib
import numpy as np
import torch
import yaml

CODE=Path(__file__).resolve().parents[2];sys.path.insert(0,str(CODE))
from openwam.dataloader.curated_stage2 import CuratedStage2Dataset
from openwam.train.stage2_eval import NativeFixedCases, _json_safe
from scripts.experiments.fullspan_samples import DecodedFrameCache


def select_descriptors(dataset, per_level=12):
    identities={(r['family'],r['repo'],ep) for r in dataset.repos for ep,_,_ in r['ranges']}
    if len(identities)!=3 or len({f for f,_,_ in identities})!=3:
        raise ValueError('requires exactly three dataset families and one episode per family')
    descriptors=[]
    for repo in dataset.repos:
        episode=repo['ranges'][0][0];m=dataset._episode(repo,episode)
        for level in ('task','subtask'):
            options=dataset._options(m,repo['ranges'],level)
            candidates=[(o,a) for o in options for a in range(o['start'],o['end_exclusive'])
                        if o['last_source']-a>=32]
            if not candidates:continue
            # Midpoints of equal partitions avoid boundary-only observations.
            n=min(per_level,len(candidates))
            indices=np.floor((np.arange(n)+.5)*len(candidates)/n).astype(int)
            for index in indices:
                option,anchor=candidates[index]
                descriptors.append(dict(repo=repo['repo'],family=repo['family'],episode=episode,
                    level=level,option=option,anchor=anchor))
    if {d['family'] for d in descriptors}!={f for f,_,_ in identities}:
        raise ValueError('at least one episode has no eligible full action window')
    return descriptors


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--anchors-per-level',type=int,default=12)
    args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():raise FileExistsError('immutable prepared data already exists')
    if args.anchors_per_level<1:raise ValueError('anchors-per-level must be positive')
    torch.set_num_threads(2)
    import scripts.data.prepare_subtask_goals as decoder
    decoder.decode_frames=DecodedFrameCache(decoder.decode_frames,max_bytes=512*1024**2)
    cfg=yaml.safe_load(args.config.read_text());ds=CuratedStage2Dataset.from_config(cfg['dataloader'])
    cases=NativeFixedCases(ds,run_dir=out,seed=42,max_cases=128,max_search_episodes=1)
    cases.descriptors=select_descriptors(ds,args.anchors_per_level)
    cases.coverage=dict(mode='three episodes; bounded evenly spaced task/subtask training anchors',
        selected_cases=len(cases.descriptors),available_cases=len(cases.descriptors),families=sorted({d['family'] for d in cases.descriptors}))
    cases.publish_descriptors();samples=[];rows=[]
    for i,d in enumerate(cases.descriptors):
        sample=cases[i]
        if not sample['action_mask'].any():raise ValueError(f'no supervised actions: {d}')
        path=out/f'case_{i:03d}.pt';torch.save(sample,path);samples.append(sample)
        rows.append(dict(index=i,family=d['family'],repo=d['repo'],episode=d['episode'],anchor=d['anchor'],
            granularity=d['level'],prompt=sample['prompt'],path=str(path.resolve()),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),valid_action_elements=int(sample['action_mask'].sum()),
            report_target=sample['report_target']))
        print(json.dumps(dict(event='prepared',index=i,total=len(cases),family=d['family'],anchor=d['anchor'])),flush=True)
    torch.save(samples,out/'samples.pt')
    # Three representative anchors; all materialized cases remain trainable.
    eval_indices=[]
    for family in sorted({r['family'] for r in rows}):
        choices=[r['index'] for r in rows if r['family']==family and r['granularity']=='task']
        eval_indices.append(choices[len(choices)//2])
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    manifest=dict(schema='openwam.z_action.fixed_cpu.v1',source_yaml=ds.selection['source_yaml'],
        source_sha256=ds.source_yaml_sha256,normalization_registry=ds.registry_path,
        normalization_sha256=sha(ds.registry_path),normalization_horizon=32,action_horizon=32,
        video_stride=4,video_frames=9,coverage=cases.coverage,cases=rows,evaluation_indices=eval_indices,
        samples_path=str((out/'samples.pt').resolve()),samples_sha256=sha(out/'samples.pt'))
    (out/'manifest.json').write_text(json.dumps(_json_safe(manifest),indent=2)+'\n')
    print(json.dumps(dict(status='complete',samples=len(samples),evaluation_indices=eval_indices)),flush=True)


if __name__=='__main__':main()
