#!/usr/bin/env python3
"""Precompute/audit all reachable RFC 1Hz independent multiview T=1 latents.

Preferred: python scripts/data/precompute_goal_memory.py --config configs/goal/stage1_goal_memory.yaml
 --device cuda --dtype bfloat16 --batch-size 4 --report coverage.json
The config path uses exactly the training baseline VAE component and prefix.
Standalone fallback: --vae-checkpoint Wan2.2_VAE.pth --source-yaml sources.yaml --cache-dir cache.
Use the same dtype, source YAML and cache directory for training.
"""
from pathlib import Path
import argparse
import json
import os
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from openwam.dataloader.rfc_goal import RFCGoalDataset
from openwam.model.goal.memory_codec import GoalMemoryCodec


def precompute_goal_memory(dataset, *, batch_size=4, audit_only=False):
    if type(batch_size) is not int or batch_size<1:raise ValueError('batch_size must be positive integer')
    if dataset.codec is None:raise ValueError('precompute/audit requires a codec with cache identity')
    report=dict(population_identity=dataset.population_identity,vae_identity=dataset.codec.vae_identity,
                requested=0,valid=0,invalid=0,hits=0,missing=0,written=0,audit_only=audit_only)
    pending=[]
    def flush():
        if not pending:return
        pixels=torch.stack([dataset.decode_memory(request) for request in pending])
        masks=torch.tensor([r['view_valid'] for r in pending],dtype=torch.bool)
        latents=dataset.codec.encode_images(pixels,masks).detach().cpu()
        for request,latent in zip(pending,latents):
            dataset.codec.put_cached(**dataset.cache_key(request),latent=latent)
            report['written']+=1
        pending.clear()
    for request in dataset.precompute_requests():
        report['requested']+=1
        if not any(request['view_valid']):report['invalid']+=1;continue
        report['valid']+=1
        cached=dataset.codec.get_cached(**dataset.cache_key(request))
        if cached is not None:report['hits']+=1;continue
        report['missing']+=1
        if not audit_only:
            pending.append(request)
            if len(pending)>=batch_size:flush()
    flush()
    report['covered']=report['hits']+report['written']
    report['complete']=report['covered']==report['valid']
    return report


def load_vae(checkpoint,device,dtype):
    from openwam.model.video_backbone.encoder.wan22_vae import _resolve_wan22_vae_class_for_file
    from openwam.model.video_backbone.wan.shared.core.loader.model import load_model
    cls,converter=_resolve_wan22_vae_class_for_file(str(checkpoint))
    vae=load_model(cls,path=str(checkpoint),torch_dtype=dtype,device=device,state_dict_converter=converter)
    if int(vae.z_dim)!=48:raise ValueError('goal memory requires native Wan2.2 48-channel VAE')
    return vae.eval().requires_grad_(False)


def load_configured_vae(config,dtype):
    """Use exactly the training component class/kwargs and baseline prefix."""
    from openwam.train.goal_trainer import _construct,load_component_weights
    vae=_construct(config['components']['vae'])
    load_component_weights(vae,config['baseline_checkpoint'],'video_backbone.vae.')
    return vae.to(device='cpu',dtype=dtype).eval().requires_grad_(False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--config',help='training YAML; uses its exact baseline VAE and dataset settings')
    mode.add_argument('--vae-checkpoint',help='standalone native VAE checkpoint fallback')
    parser.add_argument('--source-yaml');parser.add_argument('--cache-dir')
    parser.add_argument('--vae-identity',help='standalone fallback identity override only')
    parser.add_argument('--data-root');parser.add_argument('--path-aliases')
    parser.add_argument('--device',default='cpu');parser.add_argument('--dtype',choices=['float32','bfloat16'])
    parser.add_argument('--batch-size',type=int,default=4);parser.add_argument('--recent-frames',type=int)
    parser.add_argument('--epoch-size',type=int);parser.add_argument('--seed',type=int)
    parser.add_argument('--subtask-probability',type=float)
    parser.add_argument('--audit-only',action='store_true');parser.add_argument('--report',required=True)
    args=parser.parse_args()
    if args.config:
        if args.vae_identity:parser.error('--config derives the training VAE identity; do not override it')
        from openwam.train.goal_memory import load_config
        from openwam.train.goal_memory_data import make_dataset
        config=load_config(args.config)
        dtype=getattr(torch,args.dtype or ('float32' if args.device=='cpu' else config.get('dtype','bfloat16')))
        for arg,key in [('source_yaml','source_yaml'),('data_root','data_root'),('path_aliases','path_aliases'),
                        ('epoch_size','epoch_size'),('seed','seed'),('subtask_probability','subtask_probability')]:
            value=getattr(args,arg)
            if value is not None:config['dataset'][key]=value
        if args.recent_frames is not None:config.setdefault('history',{})['recent_frames']=args.recent_frames
        cache_dir=args.cache_dir or config['memory_cache']['directory']
        vae=load_configured_vae(config,dtype)
    else:
        if not args.source_yaml or not args.cache_dir:parser.error('standalone mode requires --source-yaml and --cache-dir')
        dtype=getattr(torch,args.dtype or ('float32' if args.device=='cpu' else 'bfloat16'))
        cache_dir=args.cache_dir
        vae=load_vae(args.vae_checkpoint,'cpu',dtype)
    from openwam.model.goal.dependencies import module_identity
    vae_identity=args.vae_identity or module_identity(vae)['fingerprint']
    vae.to(device=args.device,dtype=dtype)
    codec=GoalMemoryCodec(vae,cache_dir,vae_identity=vae_identity,device=args.device,dtype=dtype,
                          max_encode_batch_size=args.batch_size)
    if args.config:
        dataset=make_dataset(config,codec)
    else:
        dataset=RFCGoalDataset(args.source_yaml,Path(cache_dir)/'reader',codec=codec,
            epoch_size=args.epoch_size if args.epoch_size is not None else 1000,
            seed=args.seed if args.seed is not None else 42,
            subtask_probability=args.subtask_probability if args.subtask_probability is not None else .7,
            recent_frames=args.recent_frames if args.recent_frames is not None else 2,
            data_root=args.data_root,path_aliases=args.path_aliases,dtype=dtype)
    report=precompute_goal_memory(dataset,batch_size=args.batch_size,audit_only=args.audit_only)
    destination=Path(args.report);destination.parent.mkdir(parents=True,exist_ok=True)
    fd,temporary=tempfile.mkstemp(dir=destination.parent,prefix='.coverage-')
    try:
        with os.fdopen(fd,'w') as stream:json.dump(report,stream,indent=2);stream.write('\n')
        os.replace(temporary,destination)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)
    print(json.dumps(report,sort_keys=True))
    return 0 if report['complete'] else 2

if __name__=='__main__':raise SystemExit(main())
