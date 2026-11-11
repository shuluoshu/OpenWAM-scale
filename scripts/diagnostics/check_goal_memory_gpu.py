#!/usr/bin/env python3
"""Bounded native GPU validation; no optimizer allocation or training job changes.

Loads only the pretrained VideoDiT component. Text and latents are synthetic;
this checks production geometry/routing/backward/cache, not dataset quality.
"""
from pathlib import Path
import argparse
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def run_expert(model, inputs, cache=None):
    extra = dict(runtime_cache=cache, episode_id="native-smoke") if cache is not None else {}
    return model.outputs(model.run_state(model.prepare_state(**inputs, **extra)))


def check_stage2(video, config, inputs, dtype, device):
    import torch
    from openwam.train.goal_memory_stage2 import build_native_action
    from openwam.model.architectures.dual_system.goal_memory import GoalMemoryPolicy
    action,proprio,initialization=build_native_action(video,config)
    policy=GoalMemoryPolicy(video,action,proprio).to(device=device,dtype=dtype)
    base={k:v for k,v in inputs.items() if k not in ('noisy_target_latents','timestep')}
    base['proprio']=torch.randn(1,112,device=device,dtype=dtype)
    noise=torch.randn(1,32,112,device=device,dtype=dtype)
    t=torch.full((1,),500.,device=device,dtype=dtype)
    policy.eval()
    with torch.no_grad():
        full=policy(noise,t,**base,noisy_target_latents=inputs['noisy_target_latents'],target_timestep=t)['action_velocity']
        cache=policy.prefill(**base)
        cached=policy.action_from_prefill(cache,noise,t,base['text'],base['proprio'])
        difference=float((full.float()-cached.float()).norm()/full.float().norm().clamp_min(1e-8))
        if dtype==torch.float32 and difference>1e-4:
            raise AssertionError(f'isolated action prefill mismatch: {difference}')
        actions=policy.infer_actions(noise,steps=3,**base)
        if actions.shape!=(1,32,112) or not torch.isfinite(actions).all():
            raise AssertionError('invalid cached action inference')
    policy.train();torch.cuda.reset_peak_memory_stats(device)
    timings={}
    for mode in ('none','joint'):
        policy.future_video_mode=mode
        policy.attention_mask_mode='isolated' if mode=='none' else 'action_sees_video'
        policy.zero_grad(set_to_none=True)
        extra={} if mode=='none' else dict(future_latents=torch.randn(1,48,2,24,20,device=device,dtype=dtype),
                                          future_view_valid=torch.ones(1,2,3,device=device,dtype=torch.bool))
        torch.cuda.synchronize(device);start=time.perf_counter()
        result=policy.training_loss(actions=noise,action_mask=torch.ones_like(noise,dtype=torch.bool),
            **base,target_latents=inputs['noisy_target_latents'],
            target_image_valid=torch.ones(1,device=device,dtype=torch.bool),
            target_view_valid=torch.ones(1,3,device=device,dtype=torch.bool),**extra)
        result['loss'].backward();torch.cuda.synchronize(device)
        for name,p in [('G',policy.video.goal_queries),('native_action',policy.action_backbone.action_encoder.weight)]:
            if p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().sum()==0:
                raise AssertionError(f'invalid {mode} {name} gradient')
        timings[mode]=dict(forward_backward_s=time.perf_counter()-start,loss=float(result['loss'].detach()))
    return dict(initialization=initialization,action_shape=[1,32,112],prefill_relative_l2=difference,
                modes=timings,peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
                optimizer_step=False,synthetic_inputs=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(ROOT/'configs/goal/stage1_goal_memory.yaml'))
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--dtype',choices=['float32','bfloat16'],default='float32')
    parser.add_argument('--allow-approximate-low-precision-cache',action='store_true')
    parser.add_argument('--stage2',action='store_true',help='also load native action weights, check isolated inference and joint backward')
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args()
    import torch
    from openwam.train.goal_memory import load_config
    from openwam.train.goal_trainer import _construct,load_component_weights
    from openwam.model.goal.goal_memory import GoalMemoryExpert,GoalMemoryConfig
    from openwam.model.goal.goal_memory_cache import GoalMemoryRuntimeCache
    if args.repeats < 1: parser.error('positive repeats required')
    if args.dtype != 'float32' and not args.allow_approximate_low_precision_cache:
        parser.error('low-precision cross-chunk KV is approximate; pass --allow-approximate-low-precision-cache or use float32')
    config=load_config(args.config);device=torch.device(args.device);dtype=getattr(torch,args.dtype)
    torch.manual_seed(101)
    started=time.perf_counter()
    dit=_construct(config['components']['dit'])
    load_component_weights(dit,config['baseline_checkpoint'],'video_backbone.dit.')
    model=GoalMemoryExpert(dit,GoalMemoryConfig(**{**config.get('memory',{}),'history_dropout':0.,'history_noise_mix':0.}),
                           dit_config=config['components']['dit']['extra_kwargs'])
    model.to(device=device,dtype=dtype)
    load_s=time.perf_counter()-started
    def tensor(*shape): return torch.randn(*shape,device=device,dtype=dtype)
    inp=dict(current_latents=tensor(1,48,1,24,20),memory_latents=tensor(1,3,48,1,24,20),
             noisy_target_latents=tensor(1,48,1,24,20),timestep=torch.tensor([500.],device=device),
             text=tensor(1,8,model.context_width),memory_frame_ids=torch.tensor([[0,8,9]],device=device),
             memory_ages=torch.tensor([[10.,2.,1.]],device=device),goal_horizon_s=torch.tensor([3.],device=device))
    def run(cache=None):
        return run_expert(model, inp, cache)
    def timed(cache=None):
        torch.cuda.synchronize(device);t=time.perf_counter();out=run(cache)
        torch.cuda.synchronize(device);return out,time.perf_counter()-t
    model.eval();cache=GoalMemoryRuntimeCache(model_identity='loaded-native-smoke')
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        off,_=timed();cached,cold_s=timed(cache)
        hot=[];uncached=[]
        for _ in range(args.repeats):
            cached,elapsed=timed(cache);hot.append(elapsed)
            _,elapsed=timed();uncached.append(elapsed)
        diffs={k:dict(max_abs=float((off[k].float()-cached[k].float()).abs().max()),
                       relative_l2=float((off[k].float()-cached[k].float()).norm()/off[k].float().norm().clamp_min(1e-8))) for k in off}
        for key in off:
            if not torch.isfinite(cached[key]).all():raise AssertionError(f'nonfinite cached {key}')
            # FP32 tiny-native tests separately enforce tight numerical parity.
            if diffs[key]['relative_l2'] > (.02 if dtype==torch.bfloat16 else 1e-4):
                raise AssertionError(f'cache mismatch {key}: {diffs[key]}')
    if cache.hits == 0 or cache.misses == 0:
        raise AssertionError('cache benchmark did not exercise both cold and hot paths')
    model.train();torch.cuda.synchronize(device);t=time.perf_counter()
    train={k:v for k,v in inp.items() if k not in ('noisy_target_latents','timestep')}
    result=model.training_loss(**train,target_latents=inp['noisy_target_latents'],
        target_uv=torch.tensor([[[.3,.4],[.6,.7]]],device=device),
        target_image_valid=torch.ones(1,dtype=torch.bool,device=device),
        target_view_valid=torch.ones(1,3,dtype=torch.bool,device=device),
        target_point_valid=torch.ones(1,2,dtype=torch.bool,device=device),
        target_in_frame=torch.ones(1,2,dtype=torch.bool,device=device),update=100,total_updates=100)
    result['loss'].backward();torch.cuda.synchronize(device)
    backward_s=time.perf_counter()-t
    for name,p in [('G',model.goal_queries),('native_first_q',model.dit.blocks[0].self_attn.q.weight)]:
        if p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().sum()==0:
            raise AssertionError(f'invalid {name} gradient')
    report=dict(gpu=torch.cuda.get_device_name(device),dtype=args.dtype,dim=dit.dim,layers=len(dit.blocks),
                goal_tokens=model.config.num_goal_tokens,target_tokens=120,synthetic_inputs=True,
                pretrained_video_weights=True,load_s=load_s,cold_cache_s=cold_s,hot_cache_s=hot,
                cache_off_s=uncached,cache_difference=diffs,cache_hits=cache.hits,cache_misses=cache.misses,
                forward_backward_s=backward_s,loss=float(result['loss'].detach()),
                peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
                optimizer_step=False)
    if args.stage2:
        model.zero_grad(set_to_none=True)
        del result, off, cached, cache
        report['stage2']=check_stage2(model,config,inp,dtype,device)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
