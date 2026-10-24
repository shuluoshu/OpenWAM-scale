#!/usr/bin/env python3
"""Explicit preflight / synthetic smoke / single-process curated Stage A training."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--preflight',action='store_true',help='CPU real-data image/geometry audit; no model weights')
    modes.add_argument('--smoke',action='store_true',help='tiny real Wan blocks, synthetic inputs, no model weights')
    modes.add_argument('--train',action='store_true',help='explicitly launch single-process real training')
    parser.add_argument('--config',default=str(ROOT/'configs/goal/stage1_joint_sample.yaml'))
    parser.add_argument('--source-yaml',help='absolute source-YAML override')
    parser.add_argument('--data-root',help='optional source dataset root override')
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--device',default=None)
    parser.add_argument('--dtype',choices=['float32','bfloat16'])
    parser.add_argument('--steps',type=int)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--goal-representation', choices=['patch_denoise', 'query_latent'],
                        help='override model.goal_representation')
    for key in ('num_image_tokens', 'num_tcp_tokens', 'dim', 'num_heads', 'num_layers', 'ffn_dim'):
        parser.add_argument('--query-' + key.replace('_', '-'), type=int, default=None)
    parser.add_argument('--tokenizer-path',help='explicit tokenizer directory override')
    parser.add_argument('--init-goal-checkpoint',help='explicit checked v1 head -> multiview initialization; fresh optimizer')
    args=parser.parse_args(argv)
    import torch
    from openwam.train.goal_curated import load_config,require_single_process,preflight,train_curated,TextContextCache,initialize_goal_checkpoint
    require_single_process();torch.set_num_threads(2)
    if args.source_yaml and not Path(args.source_yaml).is_absolute():parser.error('--source-yaml must be absolute')
    if args.init_goal_checkpoint and args.preflight:parser.error('--init-goal-checkpoint requires --train or --smoke')
    config=load_config(args.config,source_yaml=args.source_yaml,data_root=args.data_root)
    model = config.setdefault('model', {})
    if args.goal_representation:
        model['goal_representation'] = args.goal_representation
    representation = model.get('goal_representation', 'patch_denoise')
    query_options = model.setdefault('query_goal', {})
    for key in ('num_image_tokens', 'num_tcp_tokens', 'dim', 'num_heads', 'num_layers', 'ffn_dim'):
        value = getattr(args, 'query_' + key)
        if value is not None:
            if representation != 'query_latent':
                parser.error('--query-* settings require query_latent')
            query_options[key] = value
    if representation == 'query_latent':
        from openwam.model.goal.query_expert import QueryConfig
        QueryConfig(**query_options)
        if args.init_goal_checkpoint:
            parser.error('legacy --init-goal-checkpoint cannot initialize query_latent topology')
    observation_mode=config.get('observation_mode','head')
    flow_settings=dict(shift=config.get('shift',5.),lambda_image=config.get('lambda_image',.5),
                       lambda_tcp=config.get('lambda_tcp',1.))
    def build_expert(dit, codec, dit_config, **kwargs):
        from openwam.model.goal.patch_expert import PatchGoalExpert
        if representation == 'query_latent':
            from openwam.model.goal.query_expert import QueryGoalExpert
            return QueryGoalExpert(dit, codec, dit_config=dit_config, query_config=query_options,
                lambda_query_image=config.get('lambda_query_image', .1),
                lambda_query_tcp=config.get('lambda_query_tcp', .1), **kwargs)
        return PatchGoalExpert(dit, codec, dit_config=dit_config, **kwargs)
    if args.init_goal_checkpoint and observation_mode!='multiview':
        parser.error('--init-goal-checkpoint requires a multiview config')
    output=Path(args.output_dir)
    if output.exists() and any(output.iterdir()):raise ValueError('nonempty output directory; choose a new run')
    if args.smoke:
        from openwam.train.goal_patch import make_smoke_data
        from openwam.model.goal.patch_expert import PatchGoalExpert
        expert,batches,records=make_smoke_data(args.device or 'cpu',observation_mode=observation_mode)
        fingerprint=hashlib.sha256(batches[0]['context'].cpu().numpy().tobytes()).hexdigest()
        if args.init_goal_checkpoint:
            expert=initialize_goal_checkpoint(args.init_goal_checkpoint,vae=expert.codec.vae,dit_config=expert.dit_config,
                observation_mode=observation_mode,text_dim=expert.dit.text_embedding[0].in_features,
                precomputed_text_fingerprint=fingerprint,device=args.device or 'cpu',**flow_settings)
        else:
            expert=build_expert(expert.dit,expert.codec,dit_config=expert.dit_config,
                observation_mode=observation_mode,**flow_settings)
        smoke_dtype = getattr(torch, args.dtype or 'float32')
        expert.to(device=args.device or 'cpu', dtype=smoke_dtype)
        expert.codec.vae.to(device=args.device or 'cpu', dtype=smoke_dtype)
        records[0].update(metric_key='synthetic:task:0',granularity='task')
        class SyntheticDataset:
            def evaluation_samples(self):return list(zip(records,batches))
            def sample(self,step):return records[0],batches[0]
            def scope_summary(self):return dict(synthetic=True,families=['synthetic'],observation_mode=observation_mode)
        result=train_curated(expert,dataset=SyntheticDataset(),context_cache=None,output_dir=output,
            steps=args.steps if args.steps is not None else 2,eval_every=500,sampling_steps=2,learning_rate=.001,seed=args.seed,
            precomputed_text_fingerprint=fingerprint,
            training_metadata=dict(synthetic_diagnostic=True,real_wan_blocks=True,config=config))
    else:
        from openwam.dataloader.curated_goal import CuratedGoalDataset
        dataset_config=dict(config['dataset'])
        cache_root=Path(dataset_config.pop('cache_dir',output.parent/'data_cache'))
        dataset_config['cache_dir']=str(cache_root/(output.name+'-'+uuid.uuid4().hex))
        dtype=getattr(torch,args.dtype or config.get('dtype','bfloat16'))
        # Preflight always exports the actual BF16 input grid, independently of model dtype.
        dataset=CuratedGoalDataset(**dataset_config,seed=args.seed,dtype=torch.bfloat16 if args.preflight else dtype)
        if args.preflight:
            result=preflight(dataset,output)
        else:
            from openwam.train.goal_trainer import bootstrap_components,resolve_tokenizer_config,_seed
            from openwam.model.goal.patch_expert import PatchGoalExpert
            from openwam.model.goal.dependencies import dependency_identity
            from openwam.model.video_backbone.wan.encode import encode_text
            device=args.device or 'cuda'
            if config.get('batch_size',1)!=1:raise ValueError('curated streaming requires batch_size 1')
            if not Path(config['baseline_checkpoint']).is_file():raise FileNotFoundError(config['baseline_checkpoint'])
            if args.tokenizer_path:
                config.setdefault('tokenizer',{}).setdefault('kwargs',{})['name']=str(Path(args.tokenizer_path).resolve())
            config['tokenizer']=resolve_tokenizer_config(config)
            _seed(args.seed)
            # CPU bootstrap avoids temporary full T5 occupancy on the accelerator.
            original,text_encoder,tokenizer=bootstrap_components(config,device='cpu',dtype=dtype)
            if args.init_goal_checkpoint:
                # Release the baseline DiT before constructing the checkpoint DiT.
                # Initialize on CPU alongside the shared text/VAE dependencies,
                # then place the expert and VAE on their training device.
                vae,dit_config=original.codec.vae,original.dit_config
                del original
                expert=initialize_goal_checkpoint(args.init_goal_checkpoint,vae=vae,dit_config=dit_config,
                    observation_mode=observation_mode,text_encoder=text_encoder,tokenizer=tokenizer,dtype=dtype,
                    use_gradient_checkpointing=config.get('gradient_checkpointing',True),**flow_settings)
                expert.to(device=device,dtype=dtype)
            else:
                expert=build_expert(original.dit,original.codec,dit_config=original.dit_config,
                    **flow_settings,
                    use_gradient_checkpointing=config.get('gradient_checkpointing',True),
                    observation_mode=observation_mode).to(device=device,dtype=dtype)
                del original
            expert.codec.vae.to(device=device,dtype=dtype).eval().requires_grad_(False)
            text_device=config.get('text_device','cpu')
            text_encoder.to(device=text_device,dtype=dtype).eval().requires_grad_(False)
            identities=dependency_identity(expert.codec.vae,text_dim=expert.dit.text_embedding[0].in_features,
                                           text_encoder=text_encoder,tokenizer=tokenizer)
            def encode(prompt):
                return encode_text([prompt],tokenizer=tokenizer,text_encoder=text_encoder,device=text_device)
            cache=TextContextCache(encode,max_entries=config.get('text_cache_size',32))
            result=train_curated(expert,dataset=dataset,context_cache=cache,output_dir=output,
                steps=args.steps if args.steps is not None else config['max_steps'],eval_every=config.get('eval_every',500),
                sampling_steps=config.get('sampling_steps',20),learning_rate=config['learning_rate'],seed=args.seed,
                dependency_identities=identities,dependency_metadata=dict(components=config['components'],tokenizer=config['tokenizer']),
                training_metadata=dict(config=config,source_yaml_sha256=hashlib.sha256(Path(config['dataset']['source_yaml']).read_bytes()).hexdigest()))
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','config','evaluations')},indent=2),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
