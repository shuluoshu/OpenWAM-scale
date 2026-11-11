#!/usr/bin/env python3
"""Load the portable RFC policy and infer normalized32x112 actions.

Inputs are verified frozen-VAE latents and frozen-T5 text tensors, plus normalized
proprio112. Robot-specific denormalization belongs to the matching stats registry.
"""
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--dependencies',required=True,help='verified frozen dependency identities JSON')
    parser.add_argument('--inputs',required=True,help='trusted torch tensor dictionary: current_latents,memory_latents,text,proprio,masks/time/IDs')
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--dtype',choices=['float32','bfloat16'],default='bfloat16')
    parser.add_argument('--steps',type=int,default=10)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--memory-kv-cache',action='store_true',help='requires episode_id and frame IDs; low precision defaults off')
    args=parser.parse_args(argv)
    if args.steps<1:parser.error('positive --steps required')
    import torch
    from openwam.train.goal_memory_stage2 import load_policy_bundle
    from openwam.model.goal.goal_memory_cache import GoalMemoryRuntimeCache
    dtype=getattr(torch,args.dtype);dependencies=json.loads(Path(args.dependencies).read_text())
    policy=load_policy_bundle(args.checkpoint,dependencies=dependencies,device=args.device,dtype=dtype).eval()
    raw=torch.load(args.inputs,map_location='cpu',weights_only=True)
    required={'current_latents','memory_latents','text','proprio'}
    if not isinstance(raw,dict) or required-raw.keys():raise ValueError('input tensor dictionary missing required fields')
    allowed=required|{'text_mask','memory_valid','observed_view_valid','memory_view_valid','memory_ages',
                      'memory_frame_ids','goal_horizon_s','future_view_valid','episode_id'}
    if raw.keys()-allowed:raise ValueError(f'unknown inference inputs: {sorted(raw.keys()-allowed)}')
    batch={k:(v.to(device=args.device,dtype=dtype if v.is_floating_point() else v.dtype)
              if torch.is_tensor(v) else v) for k,v in raw.items()}
    if args.memory_kv_cache:
        if 'episode_id' not in batch or 'memory_frame_ids' not in batch:
            raise ValueError('memory KV cache requires explicit episode_id and memory_frame_ids')
        batch['runtime_cache']=GoalMemoryRuntimeCache(model_identity=str(Path(args.checkpoint).resolve()))
    torch.manual_seed(args.seed)
    b=len(batch['current_latents']);shape=policy.video.config.latent_shape
    noise=torch.randn(b,32,112,device=args.device,dtype=dtype)
    future=None
    if policy.attention_mask_mode=='action_sees_video':
        future=torch.randn(b,48,2,*shape[-2:],device=args.device,dtype=dtype)
    actions=policy.infer_actions(noise,steps=args.steps,future_noise=future,**batch)
    if not torch.isfinite(actions).all():raise FloatingPointError('nonfinite inferred actions')
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(normalized_actions=actions.cpu(),steps=args.steps,seed=args.seed,
                    format='openwam.goal_memory.actions.v1'),output)
    print(json.dumps(dict(output=str(output),shape=list(actions.shape),normalized=True)),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
