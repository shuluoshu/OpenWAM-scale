"""Prepare real immutable CPU Stage2 samples without model/VAE execution."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from openwam.dataloader.curated_stage2 import CuratedStage2Dataset
from openwam.dataloader.native_stage2_stats import prepare_native_stats, lookup_stats, normalize
from openwam.dataloader.native_stage2_io import merge_source_masks
from openwam.train.stage2_eval import NativeFixedCases, _json_safe


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-yaml',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--registry');p.add_argument('--horizon',type=int,default=32)
    a=p.parse_args();out=Path(a.output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    ds=CuratedStage2Dataset(str(Path(a.source_yaml).resolve()),seed=42,observation_augmentation={'enabled':False},history={'enabled':True,'frame_dropout_prob':0.0},max_cache_bytes=1024**3)
    registry=a.registry or prepare_native_stats(ds,out/'statistics',max_episodes_per_profile=2,max_windows_per_profile=32)
    from openwam.dataloader.native_stage2_stats import read_registry
    ds.registry_path=str(Path(registry).resolve());ds.registry=read_registry(registry,ds.stats_contract())
    cases=NativeFixedCases(ds,run_dir=out,seed=42,max_cases=128,max_search_episodes=32)
    # Family labels can contain distinct robot platforms; cover every declared platform.
    represented=set()
    for d in cases.descriptors:
        r=next(r for r in ds.repos if r['repo']==d['repo'])
        d['required_platform']=ds._episode(r,d['episode'])['profile']['robot_type']
        represented.add((d['family'],d['required_platform']))
    for r in ds.repos:
        episode=r['ranges'][0][0];m=ds._episode(r,episode);key=(r['family'],m['profile']['robot_type'])
        if key in represented:continue
        opts=ds._options(m,[x for x in r['ranges'] if x[0]==episode],'task')
        if not opts:raise ValueError(f'No valid task window for platform {key}')
        cases.descriptors.append(dict(repo=r['repo'],episode=episode,anchor=opts[0]['start'],level='task',option=opts[0],family=r['family'],required_platform=key[1]))
        represented.add(key)
    # Audit actual targets; metadata-valid windows can still have every action masked.
    for d in cases.descriptors:
        found=False
        for repo in [r for r in ds.repos if r['family']==d['family'] and (not d.get('required_platform') or ds._episode(r,r['ranges'][0][0])['profile']['robot_type']==d['required_platform'])]:
            for episode in sorted({e for e,_,_ in repo['ranges']}):
                m=ds._episode(repo,episode)
                opts=ds._options(m,[r for r in repo['ranges'] if r[0]==episode],d['level'])
                for o in opts:
                    anchors=list(range(o['start'],o['end_exclusive']))
                    anchors.sort(key=lambda x:(o['last_source']-x<a.horizon,abs(x-(o['start']+o['end_exclusive'])//2)))
                    for anchor in anchors:
                        _,raw,mask,_,_,_=ds.raw_window(m,o,anchor)
                        if mask.any():
                            d.update(repo=repo['repo'],episode=episode,option=o,anchor=anchor);found=True;break
                    if found:break
                if found:break
            if found:break
        d['action_supervision_status']='available' if found else 'unavailable: all selected targets semantically masked'
        print('target audit',d['family'],d['level'],d['action_supervision_status'],flush=True)
    cases.coverage.update(selected_cases=len(cases.descriptors),available_cases=len(cases.descriptors),platform_count=len(represented),platform_policy='all declared family+robot_type pairs')
    cases.publish_descriptors();rows=[];samples=[]
    for i,d in enumerate(cases.descriptors):
        print('preparing',i,d['family'],d['level'],flush=True)
        repo=next(r for r in ds.repos if r['repo']==d['repo']);m=ds._episode(repo,d['episode'])
        reuse=out.parent/'data_v3'/f'case_{i:03d}.pt'
        sample=None
        if reuse.exists():
            candidate=torch.load(reuse,weights_only=False,map_location='cpu')
            t=candidate['trace']
            current_stats=lookup_stats(ds.registry,repo,m)
            if (t['source_repo']==str(m['root']) and t['episode']==d['episode'] and t['anchor']==d['anchor'] and candidate['granularity']==d['level']
                    and t['profile_id']==m['profile']['profile_id'] and candidate['_eval']['action_stats']==current_stats['action']):
                sample=candidate
                sample['trace']['stats_registry']=ds.registry_path
                sample['trace']['normalization_population']=current_stats['population']
        if sample is None:sample=cases[i]
        action=sample['action'].numpy();am=sample['action_mask'].numpy();raw=sample['trace']['raw_action']
        assert a.horizon==32, 'Native Stage2 contract requires horizon32'
        assert am.any() and np.isfinite(action).all() and np.all(action[~am]==0)
        path=out/f'case_{i:03d}.pt';torch.save(sample,path)
        row=dict(action_supervision_status=d.get('action_supervision_status'),robot_type=m['profile']['robot_type'],profile_id=m['profile']['profile_id'],mask_reasons=m['profile']['mask_reasons'],index=i,family=d['family'],granularity=d['level'],repo=d['repo'],episode=d['episode'],anchor=d['anchor'],episode_length=m['length'],prompt=sample['prompt'],instruction_interval=[sample['instruction_start'],sample['instruction_end_exclusive']],path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),valid_action_dims=np.flatnonzero(am.any(0)).tolist(),valid_action_elements=int(am.sum()),action_horizon=a.horizon,real_action_steps=len(raw),video_source_indices=sample['trace']['video_source_indices'],video_mask=sample['video_mask'].tolist(),head_history=sample['trace'].get('head_history'),report_target=sample['report_target'])
        samples.append(sample);rows.append(row);(out/f'case_{i:03d}_trace.json').write_text(json.dumps(_json_safe(sample['trace']),indent=2))
        (out/'preparation_progress.json').write_text(json.dumps(_json_safe(rows),indent=2))
    manifest=dict(schema='openwam.z_action.fixed_cpu.v1',source_yaml=str(Path(a.source_yaml).resolve()),source_sha256=ds.source_yaml_sha256,normalization_registry=ds.registry_path,normalization_sha256=hashlib.sha256(Path(ds.registry_path).read_bytes()).hexdigest(),normalization_horizon=32,action_horizon=a.horizon,video_stride=4,video_frames=9,coverage=cases.coverage,cases=rows)
    torch.save(samples,out/'samples.pt')
    manifest['samples_path']=str(out/'samples.pt')
    manifest['samples_sha256']=hashlib.sha256((out/'samples.pt').read_bytes()).hexdigest()
    manifest['coverage'].update(action_supervised_cases=sum(bool(s['action_mask'].any()) for s in samples),action_unavailable_cases=[i for i,s in enumerate(samples) if not s['action_mask'].any()])
    manifest['tcp_conventions']={'Genrobot':{'offset_m':[.1015,-.0003,-.0398],'axes':'inherited finger-device base frame','endpoint':'audited device TCP; not human fingertip','independent_physical_calibration_verified':False}}
    (out/'manifest.json').write_text(json.dumps(_json_safe(manifest),indent=2));print('COMPLETE',len(rows),flush=True)

if __name__=='__main__':main()
