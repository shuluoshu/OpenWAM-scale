#!/usr/bin/env python3
"""DSW tiny-native-block verification, never a full-model quality/speed benchmark.

Uses the repository's deterministic tiny VAE/T5 fixtures; the visual-history
encoder, Wan transformer, ActionDiT, adapter, FM loss and optimizer are real.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--dtype',choices=['float32','bfloat16'],default='float32')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    sys.path[:0]=[str(root),str(root/'tests/goal')]
    os.environ['OPENWAM_TEST_DEVICE']=args.device
    os.environ['OPENWAM_TEST_DTYPE']=args.dtype
    import torch
    from test_visual_history_aux_cache import model,batch
    from test_visual_history_stage2 import load_encoder
    from test_visual_history import history
    from test_multiview_patch import multiview_batch
    torch.manual_seed(42);torch.set_num_threads(2)
    dtype=getattr(torch,args.dtype)
    if args.device=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
        torch.cuda.reset_peak_memory_stats()
    with tempfile.TemporaryDirectory(prefix='cstar-smoke-') as tmp:
        _,encoder,_=load_encoder(Path(tmp))
        a=model();a.goal_expert=encoder.to(device=args.device,dtype=dtype)
        obs,_,_,views=multiview_batch(1)
        h={k:v.to(device=args.device,dtype=dtype if v.is_floating_point() else v.dtype) for k,v in history().items()}
        with torch.no_grad():
            c=a.goal_expert.condition_tokens(obs.to(device=args.device,dtype=dtype),
                observed_view_valid=views.to(args.device),**h)
        x=batch();x['goal_latent']=c
        frozen={n:p.detach().clone() for n,p in a.goal_expert.named_parameters()}
        reps={'video':a.video_backbone.dit.patch_embedding.weight,
              'action':a.action_backbone.action_encoder.weight,
              'cstar_adapter':a.goal_adapter.projection.weight}
        before={n:p.detach().clone() for n,p in reps.items()}
        a.train()
        kwargs={k:v for k,v in x.items() if k not in ('noisy_actions','action_timestep','timestep','latents')}
        opt=torch.optim.AdamW([p for p in a.parameters() if p.requires_grad],lr=1e-3,foreach=False)
        losses=a.compute_loss(actions=x['noisy_actions'],input_latents=x['latents'],**kwargs)
        losses['loss'].backward()
        gradients={n:dict(finite=bool(torch.isfinite(p.grad).all()),norm=float(p.grad.float().norm())) for n,p in reps.items()}
        assert all(v['finite'] and v['norm']>0 for v in gradients.values())
        opt.step();opt.zero_grad(set_to_none=True)
        changed={n:int((p!=before[n]).sum()) for n,p in reps.items()}
        assert all(changed.values())
        frozen_ok=all(p.grad is None and not p.requires_grad and torch.equal(p,frozen[n]) for n,p in a.goal_expert.named_parameters())
        assert frozen_ok
        a.eval();counts={'video_layers':0,'adapter':0}
        orig=a.video_backbone.pre_attn_at_layer
        def pre(*args,**kw):counts['video_layers']+=1;return orig(*args,**kw)
        a.video_backbone.pre_attn_at_layer=pre
        handle=a.goal_adapter.register_forward_hook(lambda *args:counts.__setitem__('adapter',counts['adapter']+1))
        with torch.no_grad():
            expected=[];actions=x['noisy_actions'].clone()
            for step in (900.,700.,500.,300.,100.):
                t=torch.full_like(x['action_timestep'],step)
                v=a(**{**x,'noisy_actions':actions,'action_timestep':t})[1]
                expected.append(v);actions=actions-.2*v
            plain_counts=dict(counts);counts.update(video_layers=0,adapter=0)
            cache=a.prepare_action_kv_cache({k:v for k,v in x.items() if k not in ('noisy_actions','action_timestep','timestep')})
            actions=x['noisy_actions'].clone();errors=[]
            for i,step in enumerate((900.,700.,500.,300.,100.)):
                t=torch.full_like(x['action_timestep'],step)
                actual=a.forward_action_kv_cache(actions,t,cache)
                errors.append(float((actual.float()-expected[i].float()).abs().max()))
                tol=.025 if dtype==torch.bfloat16 else 3e-5
                torch.testing.assert_close(actual,expected[i],atol=tol,rtol=tol)
                actions=actions-.2*actual
        handle.remove()
        assert counts==dict(video_layers=2,adapter=1)
        result=dict(passed=True,device=args.device,dtype=args.dtype,torch=torch.__version__,
            gpu=torch.cuda.get_device_name(0) if args.device=='cuda' else None,
            scope='tiny native blocks; synthetic latents; fixture VAE/T5; no full checkpoint/dataset or throughput claim',
            shape=dict(layers=2,hidden=48,action_horizon=32,action_dim=112,cstar_tokens=120,cstar_width=48),
            losses={k:float(losses[k].detach()) for k in ('loss','loss_video','loss_action')},
            gradients=gradients,changed_elements=changed,frozen_encoder_unchanged=frozen_ok,
            velocity_max_abs_error_per_step=errors,uncached_calls=plain_counts,cached_calls=counts,
            peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device=='cuda' else None)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))


if __name__=='__main__':main()
