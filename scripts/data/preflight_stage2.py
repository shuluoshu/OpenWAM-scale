"""CPU-only native Stage 2 metadata/statistics/bounded video preflight."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
import yaml
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from openwam.dataloader.curated_stage2 import CuratedStage2Dataset
from openwam.dataloader.native_stage2_stats import audit_selection,prepare_native_stats,read_registry


def clean(value):
    if isinstance(value,(np.ndarray,torch.Tensor)):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,Path):return str(value)
    raise TypeError(type(value).__name__)


def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=clean)+'\n')


def export_sample(dataset,repo,episode,m,option,anchor,level,folder,*,observation_mode='head'):
    from PIL import Image
    folder.mkdir(parents=True,exist_ok=True)
    sample=dataset._sample_selected(repo,episode,m,option,anchor,level)
    views,_=dataset._decode_views(m,[anchor]);head=views[0].get(m['profile']['cameras'][0])
    if head is not None:head.save(folder/'source_observed.png')
    observed=sample['goal_observed_multiview'] if observation_mode=='multiview' else sample['goal_observed_head']
    if observed is not None:
        rgb=observed[:,0].permute(1,2,0).add(1).mul(127.5).clamp(0,255).byte().numpy()
        Image.fromarray(rgb).save(folder/'goal_input.png')
    for i,image in enumerate(sample['video']):image.save(folder/f'canvas_{i:02d}.png')
    goal_tensors=({'goal_observed_multiview':observed.numpy(),'goal_view_valid':sample['goal_view_valid'].numpy()}
                  if observation_mode=='multiview' else {})
    np.savez_compressed(folder/'supervision.npz',**goal_tensors,action=sample['action'].numpy(),action_mask=sample['action_mask'].numpy(),
        proprio=sample['proprio'].numpy(),proprio_mask=sample['proprio_mask'].numpy(),video_mask=sample['video_mask'].numpy(),
        video_view_mask=sample['video_view_mask'].numpy(),raw_action=sample['trace']['raw_action'],raw_state=sample['trace']['raw_state'])
    report={k:sample[k] for k in ('trace','report_target','granularity','prompt','goal_seed','goal_head_valid')}
    report['goal_observation_mode']=observation_mode
    if observation_mode=='multiview':report['goal_view_valid']=sample['goal_view_valid'].tolist()
    report['fk_supervision']=sample.get('fk_supervision');report['augmentation']=sample.get('_observation_augmentation')
    import hashlib
    report['preprocessing']=dict(goal=dict(dtype='float32',range=[-1,1],shape=[3,1,384 if observation_mode=='multiview' else 256,320]),
        canvas=dict(dtype='PIL uint8 RGB',range=[0,255],shape=[9,384,320,3],resize='bilinear stretch per view'),
        action=dict(dtype='float32',shape=[32,112],normalization='native profile quantile; gripper physical clip[0,1]'))
    report['source_code_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [ROOT/'openwam/dataloader/curated_stage2.py',ROOT/'openwam/dataloader/native_stage2_io.py',
                  ROOT/'openwam/dataloader/native_stage2_profiles.py',ROOT/'openwam/dataloader/native_stage2_stats.py',
                  ROOT/'openwam/dataloader/utils/mora_relative.py']}
    report['artifact_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob('*.png')}
    report['normalization_identity']=dataset.normalization_identity()
    from openwam.dataloader.stage2_diagnostics import report_goal_target
    goal_image, report['report_target'] = report_goal_target(dataset, repo, episode, m, sample)
    if goal_image is not None: goal_image.save(folder/'report_goal.png')
    write(folder/'trace.json',report)
    return dict(repo=repo['repo'],episode=episode,anchor=anchor,granularity=level,folder=str(folder),
                action_valid=int(sample['action_mask'].sum()),video_valid=int(sample['video_mask'].sum()),
                view_valid=sample['video_view_mask'].sum(0).tolist(),fk_arms=len((sample.get('fk_supervision') or {}).get('arms',[])))


def load_config(path):
    """Compose Hydra defaults when present; accept standalone dataset YAML too."""
    from omegaconf import OmegaConf
    path=Path(path).resolve();config=OmegaConf.load(path)
    if 'defaults' in config:
        from hydra import compose, initialize_config_dir
        root=next((p for p in path.parents if p.name=='configs'),path.parent)
        name=path.relative_to(root).with_suffix('').as_posix()
        with initialize_config_dir(config_dir=str(root),version_base=None):config=compose(config_name=name)
    return OmegaConf.to_container(config,resolve=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--scope-only',action='store_true')
    parser.add_argument('--output-dir',required=True);parser.add_argument('--prepare-stats',action='store_true')
    parser.add_argument('--max-episodes-per-profile',type=int,default=4);parser.add_argument('--max-windows-per-profile',type=int,default=128)
    parser.add_argument('--max-export-episodes-per-profile',type=int,default=4)
    parser.add_argument('--split',choices=['train','val','test'],default='train')
    parser.add_argument('--skip-video-export',action='store_true',help='Report bounded native-array/statistics audit without video claims')
    args=parser.parse_args();output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    full_config=load_config(args.config)
    from openwam.model.goal.runtime import goal_observation_mode
    observation_mode=goal_observation_mode(full_config.get('goal',{}))
    config=full_config.get('dataloader',full_config)
    config=dict(config)
    if args.scope_only or args.prepare_stats:config['metadata_only']=True
    dataset=CuratedStage2Dataset.from_config(config,split=args.split)
    audit=audit_selection(dataset);write(output/'metadata_audit.json',audit)
    summary=dict(dataset.scope_summary(),metadata_errors=audit['errors'],normalization_contracts=len(audit['groups']),metadata_repositories=audit['metadata_repositories'])
    write(output/'summary.json',summary)
    if audit['errors']:raise SystemExit(1)
    if args.scope_only:return
    if args.prepare_stats:
        registry=prepare_native_stats(dataset,output/'statistics',max_episodes_per_profile=args.max_episodes_per_profile,max_windows_per_profile=args.max_windows_per_profile,audit=audit)
        dataset.registry=read_registry(registry,dataset.stats_contract());dataset.registry_path=registry;summary['normalization_registry']=registry
    if dataset.registry is None:raise ValueError('normalization_registry required or pass --prepare-stats')
    if args.skip_video_export:
        summary.update(statistics_ready=True,exports=[],video_exports_skipped=True)
        write(output/'summary.json',summary);return
    exports=[];failures=[];repos={r['repo']:r for r in dataset.repos}
    for number,(key,group) in enumerate(sorted(audit['groups'].items())):
        found=set();attempts=0
        for name in group['repos']:
            repo=repos[name]
            for episode in dict.fromkeys(e for e,_,_ in repo['ranges']):
                if attempts>=args.max_export_episodes_per_profile:break
                attempts+=1
                try:
                    m=dataset._episode(repo,episode);ranges=[r for r in repo['ranges'] if r[0]==episode]
                    for level in ('task','subtask'):
                        if level in found or not dataset._options(m,ranges,level):continue
                        option,anchor,_=dataset._choose(m,ranges,np.random.default_rng(dataset.seed),level)
                        exports.append(export_sample(dataset,repo,episode,m,option,anchor,level,output/f'{number:03d}_{repo["family"]}_{level}',observation_mode=observation_mode));found.add(level)
                except Exception as exc:failures.append(dict(repo=name,episode=episode,error=f'{type(exc).__name__}: {exc}'))
                if len(found)==2:break
            if len(found)==2 or attempts>=args.max_export_episodes_per_profile:break
        if 'task' not in found:failures.append(dict(contract=key,error='no task export in bounded episode sample'))
    summary.update(exports=exports,failures=failures,all_videos_audited=False,statistics_ready=True,
        diagnostic_scope='bounded per normalization-contract episode samples, not exhaustive source video audit')
    write(output/'summary.json',summary)
    if failures:raise SystemExit(1)

if __name__=='__main__':main()
