#!/usr/bin/env python3
"""Opt-in RFC goal-memory Stage1/Stage2. Never modifies a running experiment."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def smoke(config, device, dtype):
    import torch
    from openwam.model.goal.goal_memory import GoalMemoryExpert, GoalMemoryConfig
    from openwam.model.video_backbone.wan.models.dit import WanModel
    from openwam.train.goal_trainer import TINY_DIT_CONFIG
    torch.manual_seed(config.get('seed', 42))
    dit_config = {**TINY_DIT_CONFIG, 'num_layers': 2}
    expert = GoalMemoryExpert(WanModel(**dit_config), GoalMemoryConfig(**config.get('memory', {})), dit_config=dit_config)
    expert.to(device=device, dtype=dtype)
    generator = torch.Generator().manual_seed(71)
    def tensor(*shape):
        return torch.randn(*shape, generator=generator).to(device=device, dtype=dtype)
    # Inputs are deliberately full 120-token, three-view latent canvases.
    fixed = dict(current_latents=tensor(1,48,1,24,20), memory_latents=tensor(1,3,48,1,24,20),
                 target_latents=tensor(1,48,1,24,20), text=tensor(1,8,16),
                 target_uv=torch.tensor([[[.3,.4],[.6,.7]]],device=device),
                 target_image_valid=torch.ones(1,dtype=torch.bool,device=device),
                 target_view_valid=torch.ones(1,3,dtype=torch.bool,device=device),
                 target_point_valid=torch.ones(1,2,dtype=torch.bool,device=device),
                 target_in_frame=torch.ones(1,2,dtype=torch.bool,device=device),
                 memory_ages=torch.tensor([[10.,2.,1.]],device=device),
                 memory_frame_ids=torch.tensor([[0,8,9]],device=device),
                 goal_horizon_s=torch.tensor([3.],device=device))
    if config['stage'] == 2:
        from openwam.train.goal_memory_stage2 import make_smoke_policy
        expert = make_smoke_policy(expert,config)
        fixed.update(actions=tensor(1,32,112),action_mask=torch.ones(1,32,112,dtype=torch.bool,device=device),
                     proprio=tensor(1,112))
        if config.get('policy',{}).get('future_video_mode') == 'joint':
            fixed.update(future_latents=tensor(1,48,2,24,20),
                         future_view_valid=torch.ones(1,2,3,dtype=torch.bool,device=device))
    return expert, lambda update: fixed, dit_config, dict(diagnostic=True, native_wan=True,
                                                         text='synthetic',vae='latents-in'), 'synthetic-native-latents'


def main(argv=None):
    import torch
    from openwam.train.goal_memory import load_config, train_updates
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--smoke',action='store_true',help='tiny native blocks; synthetic latents; no weights or VAE')
    mode.add_argument('--preflight',action='store_true',help='source-native metadata, image/mask/label audit only')
    mode.add_argument('--train',action='store_true',help='explicit single-device training')
    parser.add_argument('--config',default=str(ROOT/'configs/goal/stage1_goal_memory.yaml'))
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--device',default=None)
    parser.add_argument('--dtype',choices=['float32','bfloat16'])
    parser.add_argument('--steps',type=int,help='total schedule budget; must match when resuming')
    parser.add_argument('--stop-after',type=int,help='bound this invocation without changing total schedule budget')
    parser.add_argument('--resume',help='new-format resume.pt; restores model/master/moments/RNG/cursor')
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.steps is not None:
        if args.steps <= 0: parser.error('--steps must be positive')
        config['max_updates'] = args.steps
    elif args.smoke:
        config['max_updates'] = 2
    device = args.device or ('cpu' if args.smoke or args.preflight else 'cuda')
    dtype = getattr(torch,args.dtype or ('float32' if device=='cpu' else config.get('dtype','bfloat16')))
    if args.preflight:
        from openwam.train.goal_memory_data import preflight
        result = preflight(config, args.output_dir)
    else:
        if args.smoke:
            model,provider,dit_config,dependencies,population = smoke(config, device, dtype)
        else:
            from openwam.train.goal_memory_data import build_training
            model,provider,dit_config,dependencies,population = build_training(config,device=device,dtype=dtype)
        result = train_updates(model,provider,output_dir=args.output_dir,config=config,dependencies=dependencies,
                               dit_config=dit_config,population_identity=population,resume=args.resume,stop_after=args.stop_after)
    print(json.dumps(result,indent=2),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
