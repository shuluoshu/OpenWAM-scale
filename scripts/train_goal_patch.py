#!/usr/bin/env python3
"""Separate Stage A: jointly flow-match terminal image and a 4D TCP patch."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--smoke',action='store_true')
    mode.add_argument('--overfit',action='store_true')
    parser.add_argument('--config',default=str(ROOT/'configs/goal/g1_patch_stage_a.yaml'))
    parser.add_argument('--mixed-manifest',action='store_true',help='balanced task/subtask diagnostic dataset')
    parser.add_argument('--manifest',default=str(ROOT/'outputs/g1_per_dataset_overfit_20261018_tcpfix/manifest.json'))
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--dtype',choices=['float32','bfloat16'])
    parser.add_argument('--steps',type=int)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--diagnostic-override',action='store_true')
    parser.add_argument('--allow-unreviewed-geometry',action='store_true')
    args=parser.parse_args(argv)
    import torch
    torch.set_num_threads(2)
    from openwam.train.goal_patch import make_smoke_data,train_patch
    output=Path(args.output_dir)
    if output.exists() and any(output.iterdir()):raise ValueError('nonempty output directory; choose a new run')
    if args.smoke:
        expert,batches,records=make_smoke_data(args.device)
        result=train_patch(expert,batches=batches,records=records,output_dir=output,
            steps=args.steps if args.steps is not None else 4,eval_every=2,sampling_steps=2,
            learning_rate=.001,seed=args.seed,precomputed_text_fingerprint=hashlib.sha256(batches[0]['context'].cpu().numpy().tobytes()).hexdigest(),
            training_metadata=dict(synthetic_diagnostic=True,real_wan_blocks=True))
    else:
        from omegaconf import OmegaConf
        from openwam.train.goal_overfit import validate_overfit_manifest,prepare_overfit_images,cache_text_contexts
        from openwam.train.goal_trainer import bootstrap_components,resolve_tokenizer_config,_seed
        from openwam.model.goal.patch_expert import PatchGoalExpert
        from openwam.model.goal.dependencies import dependency_identity
        if args.mixed_manifest:
            from openwam.dataloader.mixed_goal import load_mixed_manifest
            validator=load_mixed_manifest
        else: validator=validate_overfit_manifest
        records=validator(args.manifest,diagnostic_override=args.diagnostic_override,
            allow_unreviewed_geometry=args.allow_unreviewed_geometry)
        config=OmegaConf.to_container(OmegaConf.load(args.config),resolve=True)
        if config.get('batch_size',1)!=1:raise ValueError('fixed-case overfit requires batch_size 1')
        config['tokenizer']=resolve_tokenizer_config(config)
        dtype=getattr(torch,args.dtype or config.get('dtype','bfloat16'));_seed(args.seed)
        # Borrow only loaded DiT/codec; original head and role embeddings are discarded.
        print(json.dumps(dict(event='bootstrap_started',samples=len(records),mixed=args.mixed_manifest)),flush=True)
        original,text_encoder,tokenizer=bootstrap_components(config,device=args.device,dtype=dtype)
        print(json.dumps(dict(event='model_loaded')),flush=True)
        expert=PatchGoalExpert(original.dit,original.codec,dit_config=original.dit_config,
            shift=config.get('shift',5.),lambda_image=config.get('lambda_image',.5),
            lambda_tcp=config.get('lambda_tcp',1.),use_gradient_checkpointing=config.get('gradient_checkpointing',True))
        del original
        batches=prepare_overfit_images(records,device=args.device,dtype=dtype)
        for b,r in zip(batches,records):
            b['sources']=[r.get('metric_key',r['family'])]
            b.update(family=r['family'],granularity=r.get('granularity','task'),instruction=r['instruction'])
        count=cache_text_contexts(batches,records,text_encoder=text_encoder,tokenizer=tokenizer,device=args.device,dtype=dtype)
        print(json.dumps(dict(event='text_cache_ready',unique_prompts=count)),flush=True)
        text_encoder.to('cpu')
        identities=dependency_identity(expert.codec.vae,text_dim=expert.dit.text_embedding[0].in_features,
            text_encoder=text_encoder,tokenizer=tokenizer)
        del text_encoder,tokenizer
        if torch.device(args.device).type=='cuda':torch.cuda.empty_cache()
        result=train_patch(expert,batches=batches,records=records,output_dir=output,
            steps=args.steps if args.steps is not None else config['max_steps'],eval_every=config['eval_every'],
            sampling_steps=config['sampling_steps'],learning_rate=config['learning_rate'],seed=args.seed,
            dependency_identities=identities,dependency_metadata=dict(components=config['components'],tokenizer=config['tokenizer']),
            training_metadata=dict(config=config,manifest_sha256=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
                sampling_policy='family_level_balanced' if args.mixed_manifest else 'round_robin',
                cached_unique_prompts=count,geometry_review='unreviewed numerical candidates',records=records))
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','config','evaluations')},indent=2),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
