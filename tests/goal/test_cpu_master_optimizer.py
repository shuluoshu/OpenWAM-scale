"""H100-size memory budget: CPU FP32 masters with device-resident models."""
import pytest
import torch


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_cpu_masters_match_reference_and_resume(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from openwam.train.goal_overfit import FP32MasterAdamW
    from openwam.train.goal_epoch import optimizer_payload,restore_optimizer
    model=torch.nn.Parameter(torch.tensor([1.,2.],dtype=torch.bfloat16,device=device))
    opt=FP32MasterAdamW([{'params':[model],'lr':1e-5}],master_device='cpu')
    reference=torch.nn.Parameter(model.detach().float().cpu().clone())
    expected=torch.optim.AdamW([reference],lr=1e-5,weight_decay=.01,foreach=False)
    for step in range(12):
        grad=torch.tensor([.3+step*.1,.7],dtype=model.dtype,device=device)
        model.grad=grad;reference.grad=grad.float().cpu()
        opt.step();expected.step();expected.zero_grad(set_to_none=True)
        master=opt.master_parameters[0]
        assert master.device.type=='cpu' and master.dtype==torch.float32
        torch.testing.assert_close(master,reference,rtol=0,atol=0)
        torch.testing.assert_close(model.cpu(),master.to(torch.bfloat16),rtol=0,atol=0)
        assert model.grad is None and master.grad is None
        assert opt.optimizer.state[master]['exp_avg'].device.type=='cpu'
    clone=torch.nn.Parameter(torch.zeros_like(model))
    restored=FP32MasterAdamW([{'params':[clone],'lr':1e-5}],master_device='cpu')
    restore_optimizer(restored,optimizer_payload(opt))
    model.grad=torch.ones_like(model);clone.grad=model.grad.clone()
    opt.step();restored.step()
    torch.testing.assert_close(model,clone,rtol=0,atol=0)
    torch.testing.assert_close(opt.master_parameters[0],restored.master_parameters[0],rtol=0,atol=0)


def test_h100_preset_enables_cpu_optimizer_without_changing_data_model_or_augmentation():
    from pathlib import Path
    from openwam.train.goal_curated import load_config
    root=Path(__file__).resolve().parents[2]/'configs/goal'
    h200=load_config(root/'stage1_query_history_epoch_3fps.yaml')
    h100=load_config(root/'stage1_query_history_epoch_3fps_h100.yaml')
    assert h100.pop('optimizer_device')=='cpu'
    assert h100==h200
