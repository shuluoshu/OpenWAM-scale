"""Sharded GPU-style FP32 ownership, parity and exact resume on two Gloo ranks."""
import copy
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _groups(params):
    return [dict(params=params[:2],lr=1e-5,name='goal'),dict(params=params[2:],lr=3e-4,name='history')]


def _worker(rank, world, init, folder):
    from openwam.train.goal_optimizer import ShardedFP32MasterAdamW
    from openwam.train.goal_overfit import FP32MasterAdamW
    from openwam.train.goal_epoch import optimizer_payload,restore_optimizer
    dist.init_process_group('gloo',init_method='file://'+init,rank=rank,world_size=world)
    try:
        params=[torch.nn.Parameter(torch.ones(n,dtype=torch.bfloat16)) for n in (7,11,13,17)]
        ref=[torch.nn.Parameter(p.detach().clone()) for p in params]
        opt=ShardedFP32MasterAdamW(_groups(params),betas=(.9,.95))
        full=FP32MasterAdamW([{**g,'betas':(.9,.95)} for g in _groups(ref)])
        ownership=[None]*world
        dist.all_gather_object(ownership,[i for i,p in enumerate(params) if p in opt.local_optimizer.state])
        assert sorted(i for part in ownership for i in part)==list(range(len(params)))
        assert all(0<len(part)<len(params) for part in ownership)
        for step in range(18):
            for i,(p,q) in enumerate(zip(params,ref)):
                grad=None if (i==2 and step%3==0) else torch.full_like(p,.2+i*.1+step*.01)
                p.grad=grad;q.grad=None if grad is None else grad.clone()
            opt.step();full.step()
            for p,q in zip(params,ref):torch.testing.assert_close(p,q,rtol=0,atol=0)
            for i,p in enumerate(params):
                if p in opt.local_optimizer.state:
                    state=opt.local_optimizer.state[p]
                    torch.testing.assert_close(state['master'],full.master_parameters[i],rtol=0,atol=0)
                    assert state['master'].dtype==torch.float32
                    assert state.get('exp_avg',state['master']).dtype==torch.float32
                assert p.grad is None
            if step==5:
                opt.prepare_checkpoint()
                if rank==0:torch.save(optimizer_payload(opt),Path(folder)/'optimizer.pt')
                dist.barrier()
                payload=torch.load(Path(folder)/'optimizer.pt',weights_only=True)
                new_params=[torch.nn.Parameter(p.detach().clone()) for p in params]
                restored=ShardedFP32MasterAdamW(_groups(new_params),betas=(.9,.95))
                restore_optimizer(restored,payload)
                params,opt=new_params,restored
        opt.prepare_checkpoint()
        if rank==0:
            saved=optimizer_payload(opt)
            assert saved['format']=='openwam.zero1_fp32.v1'
            for i,master in enumerate(full.master_parameters):
                torch.testing.assert_close(saved['adam']['state'][i]['master'],master,rtol=0,atol=0)
    finally:dist.destroy_process_group()


def test_sharded_master_matches_full_precision_updates_and_resume(tmp_path):
    import openwam.train.goal_optimizer as module
    assert hasattr(module,'ShardedFP32MasterAdamW')
    mp.spawn(_worker,args=(2,str(tmp_path/'init'),str(tmp_path)),nprocs=2,join=True)


def test_optimizer_config_rejects_cpu_sharding_and_invalid_betas():
    from openwam.train.goal_optimizer import optimizer_config
    with pytest.raises(ValueError,match='GPU|model'):
        optimizer_config(dict(optimizer_sharding='zero1',optimizer_device='cpu'))
    with pytest.raises(ValueError,match='betas'):
        optimizer_config(dict(adam_betas=[.9,1.]))
    assert optimizer_config({})['betas']==(.9,.999)

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA scalar optimizer resume regression')
def test_cuda_scalar_gate_resume_keeps_master_and_moments_on_gpu(tmp_path):
    from openwam.train.goal_optimizer import ShardedFP32MasterAdamW
    from openwam.train.goal_epoch import optimizer_payload,restore_optimizer
    dist.init_process_group('gloo',init_method='file://'+str(tmp_path/'cuda_init'),rank=0,world_size=1)
    try:
        p=torch.nn.Parameter(torch.ones((),dtype=torch.bfloat16,device='cuda'))
        opt=ShardedFP32MasterAdamW([dict(params=[p],lr=1e-5)])
        p.grad=torch.ones_like(p);opt.step();opt.prepare_checkpoint()
        q=torch.nn.Parameter(p.detach().clone())
        restored=ShardedFP32MasterAdamW([dict(params=[q],lr=1e-5)])
        restore_optimizer(restored,optimizer_payload(opt))
        state=restored.local_optimizer.state[q]
        assert all(state[k].device.type=='cuda' and state[k].dtype==torch.float32 for k in ('master','exp_avg','exp_avg_sq'))
        assert state['step'].device.type=='cpu'
        p.grad=torch.ones_like(p);q.grad=torch.ones_like(q)
        opt.step();restored.step()
        torch.testing.assert_close(p,q,rtol=0,atol=0)
        torch.testing.assert_close(opt.local_optimizer.state[p]['master'],state['master'],rtol=0,atol=0)
    finally:dist.destroy_process_group()
