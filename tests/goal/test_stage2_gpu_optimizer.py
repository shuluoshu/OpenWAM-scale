"""Native Stage2 selects single-tensor GPU AdamW only in new GPU recipes."""
from types import SimpleNamespace
import copy
import pytest
import torch
from omegaconf import OmegaConf


def _trainer(monkeypatch, parameter, foreach):
    from openwam.train.openwam_trainer import OpenWAMTrainer
    import openwam.train.utils.stage2 as stage2
    monkeypatch.setattr(stage2,'build_stage2_parameter_groups',lambda model,training:[{'params':[parameter],'lr':1e-4}])
    trainer=OpenWAMTrainer.__new__(OpenWAMTrainer)
    trainer._stage2=True
    trainer.architecture=SimpleNamespace()
    training=dict(learning_rate=1e-4,weight_decay=.01,adam_betas=[.9,.95],offload_optimizer_device='none')
    if foreach is not None:training['adam_foreach']=foreach
    trainer.cfg=OmegaConf.create({'training':training})
    return trainer


def test_native_gpu_optimizer_backend_is_explicit_and_legacy_default_unchanged(monkeypatch):
    p=torch.nn.Parameter(torch.ones(3))
    assert _trainer(monkeypatch,p,False).build_optimizer().defaults['foreach'] is False
    assert _trainer(monkeypatch,p,False).build_optimizer().defaults['fused'] is False
    assert _trainer(monkeypatch,p,None).build_optimizer().defaults['foreach'] is None


@pytest.mark.parametrize('values',[{'adam_foreach':'false'}, {'adam_foreach':0}])
def test_optimizer_rejects_invalid_backend_option(values):
    from openwam.train.utils.stage2 import stage2_optimizer_options
    with pytest.raises(ValueError,match='adam_foreach'):
        stage2_optimizer_options(OmegaConf.create(values))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='native GPU AdamW resume parity')
def test_native_gpu_optimizer_matches_fp32_reference_after_resume(monkeypatch):
    p=torch.nn.Parameter(torch.linspace(-1,1,37,device='cuda',dtype=torch.float32))
    q=torch.nn.Parameter(p.detach().clone())
    opt=_trainer(monkeypatch,p,False).build_optimizer()
    ref=torch.optim.AdamW([q],lr=1e-4,betas=(.9,.95),weight_decay=.01,foreach=False)
    for step in range(4):
        gradient=torch.linspace(.1,.8,37,device='cuda')*(step+1)
        p.grad=gradient.clone();q.grad=gradient.clone()
        opt.step();ref.step();opt.zero_grad();ref.zero_grad()
        if step==1:
            payload=copy.deepcopy(opt.state_dict())
            p=torch.nn.Parameter(p.detach().clone())
            opt=_trainer(monkeypatch,p,False).build_optimizer()
            opt.load_state_dict(payload)
    torch.testing.assert_close(p,q,rtol=1e-6,atol=1e-7)
    assert all(opt.state[p][k].device.type=='cuda' for k in ('exp_avg','exp_avg_sq'))
