"""The comparison's single-GPU optimizer retains native trainable ownership."""
import sys
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from scripts import compare_z_action_stage2 as comparison
from openwam.train.goal_contract import G1_LRS


class TinyArchitecture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.video_backbone=torch.nn.Linear(2,2)
        self.proprio_encoder=torch.nn.Linear(2,2)
        self.action_backbone=torch.nn.Module()
        self.action_backbone.body=torch.nn.Linear(2,2)
        self.action_backbone.action_encoder=torch.nn.Linear(2,2)
        self.action_backbone.action_decoder=torch.nn.Linear(2,2)
        self.goal_adapter=torch.nn.Linear(2,2)
        self.goal_expert=torch.nn.Linear(2,2).requires_grad_(False)
        self.to(dtype=torch.bfloat16)
        with torch.no_grad():
            for p in self.parameters():p.fill_(1.)


def training():
    return OmegaConf.create(dict(parameter_lrs=G1_LRS,adam_betas=[.8,.92],weight_decay=.02))


def test_master_optimizer_native_groups_accumulate_bf16_updates_and_release_gradients():
    arch=TinyArchitecture();optimizer,parameters=comparison.build_master_optimizer(arch,training())
    assert {id(p) for p in parameters}=={id(p) for p in arch.parameters() if p.requires_grad}
    assert len(parameters)==len(set(map(id,parameters)))
    assert {g['name']:g['lr'] for g in optimizer.optimizer.param_groups}==G1_LRS
    assert all(g['betas']==(.8,.92) and g['weight_decay']==.02 for g in optimizer.optimizer.param_groups)
    assert optimizer.optimizer.defaults['foreach'] is False
    assert all(master.dtype==torch.float32 for _,master in optimizer.parameter_pairs)
    teacher={name:p.clone() for name,p in arch.goal_expert.named_parameters()}
    before=arch.goal_adapter.weight.clone()
    for step in range(40):
        loss=sum(p.float().sum() for p in parameters);loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters,1.,foreach=False,error_if_nonfinite=True)
        optimizer.step()
        assert all(p.grad is None and master.grad is None for p,master in optimizer.parameter_pairs)
    assert not torch.equal(before,arch.goal_adapter.weight)
    for name,p in arch.goal_expert.named_parameters():
        assert p.grad is None and torch.equal(p,teacher[name])


def test_single_gpu_factory_disables_deepspeed_without_preparing_a_model(monkeypatch):
    import accelerate
    monkeypatch.setenv('WORLD_SIZE','1');monkeypatch.setenv('ACCELERATE_USE_DEEPSPEED','true')
    captured=[]
    accelerator=SimpleNamespace(num_processes=1,distributed_type=SimpleNamespace(name='NO'),device=torch.device('cuda:0'))
    monkeypatch.setattr(accelerate,'Accelerator',lambda **kw:captured.append(kw) or accelerator)
    assert comparison.build_single_gpu_accelerator() is accelerator
    assert captured==[dict(mixed_precision='bf16',device_placement=False,gradient_accumulation_steps=1)]
    import os
    assert os.environ['ACCELERATE_USE_DEEPSPEED']=='false'
    monkeypatch.setenv('WORLD_SIZE','2')
    with pytest.raises(ValueError,match='exactly one process'):comparison.build_single_gpu_accelerator()


def test_single_gpu_flag_parses_without_starting_worker(tmp_path,monkeypatch):
    monkeypatch.setattr(sys,'argv',['compare_z_action_stage2.py','--config','unused.yaml',
        '--samples','unused.pt','--output',str(tmp_path),'--arm','joint','--single-gpu-master',
        '--lr-scale','1.5','--warmup-updates','100','--no-legacy-report'])
    captured=[]
    monkeypatch.setattr(comparison,'worker',lambda args:captured.append(args))
    comparison.main()
    assert captured[0].single_gpu_master and captured[0].lr_scale==1.5
    assert captured[0].warmup_updates==100


@pytest.mark.parametrize('mode',['joint','none'])
def test_native_stage2_bf16_cpu_loss_updates_master_owned_adapter(tmp_path,mode):
    from test_visual_history_stage2 import load_encoder, attach
    from test_patch_stage2_memory import inputs
    _,encoder,_=load_encoder(tmp_path);arch=attach(encoder)
    arch.cfg['future_video_mode']=mode
    arch.set_dtype_device(torch.bfloat16,torch.device('cpu'))
    optimizer,parameters=comparison.build_master_optimizer(arch,training())
    x=inputs()
    x={key:value.to(dtype=torch.bfloat16) for key,value in x.items()}
    x['first_frame_latents']=x['latents'][:,:,:1].clone()
    z=torch.randn(1,120,48,dtype=torch.bfloat16,requires_grad=True)
    result=arch(**x,goal_latent=z,goal_token_valid=torch.ones(1,120,dtype=torch.bool))
    loss=result[1].float().square().mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert arch.goal_adapter.projection.weight.grad.abs().sum()>0
    before=arch.goal_adapter.projection.weight.clone()
    torch.nn.utils.clip_grad_norm_(parameters,1.,foreach=False,error_if_nonfinite=True)
    optimizer.step()
    assert not torch.equal(before,arch.goal_adapter.projection.weight)
    assert z.grad is None and all(p.grad is None for p in encoder.parameters())
    assert all(p.grad is None and master.grad is None for p,master in optimizer.parameter_pairs)


@pytest.mark.parametrize('mode',['joint','none'])
def test_checkpointed_native_outputs_and_gradients_match_eager(mode):
    import copy
    from test_patch_stage2_memory import make_arch, inputs
    from openwam.model.goal.conditioning import VisualHistoryMemoryAdapter
    a=make_arch();a.cfg['future_video_mode']=mode
    a.goal_adapter=VisualHistoryMemoryAdapter(a.video_backbone,dim=48)
    b=copy.deepcopy(a);b.build_mot_driver()
    x=inputs();x['first_frame_latents']=x['latents'][:,:,:1].clone()
    x.update(goal_latent=torch.randn(1,120,48),goal_token_valid=torch.ones(1,120,dtype=torch.bool))
    eager=a(**x,use_gradient_checkpointing=False)
    checked=b(**x,use_gradient_checkpointing=True)
    for u,v in zip(eager,checked):
        if u is None:assert v is None
        else:torch.testing.assert_close(u,v,rtol=0,atol=0)
    sum(y.square().mean() for y in eager if y is not None).backward()
    sum(y.square().mean() for y in checked if y is not None).backward()
    for (name,p),(other,q) in zip(a.named_parameters(),b.named_parameters()):
        assert name==other
        if p.grad is None:assert q.grad is None,name
        else:torch.testing.assert_close(p.grad,q.grad,rtol=1e-5,atol=1e-6,msg=name)


def test_action_only_checkpoint_releases_unused_video_output_every_update(monkeypatch):
    import gc
    import weakref
    from test_patch_stage2_memory import make_arch,inputs
    from openwam.model.goal.conditioning import VisualHistoryMemoryAdapter
    a=make_arch();a.cfg['future_video_mode']='none'
    a.goal_adapter=VisualHistoryMemoryAdapter(a.video_backbone,dim=48)
    x=inputs();x['first_frame_latents']=x['latents'][:,:,:1].clone()
    x.update(goal_latent=torch.randn(1,120,48),goal_token_valid=torch.ones(1,120,dtype=torch.bool))
    original=a.video_backbone.post_attn_at_layer;references=[]
    def record(layer_id,state,attention,post):
        result=original(layer_id,state,attention,post)
        if layer_id==a.video_backbone.num_layers-1:
            references.append(weakref.ref(result.hidden_states))
        return result
    monkeypatch.setattr(a.video_backbone,'post_attn_at_layer',record)
    optimizer=torch.optim.SGD([p for p in a.parameters() if p.requires_grad],lr=1e-4)
    for step in range(12):
        output=a(**x,use_gradient_checkpointing=True)
        loss=output[1].square().mean();loss.backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
        del output,loss
        gc.collect()
        assert references and all(reference() is None for reference in references),step


def test_comparison_never_publishes_failed_optimizer_update(tmp_path,monkeypatch):
    import contextlib
    import openwam.train.openwam_trainer as trainers
    import openwam.train.utils.checkpointing as checkpoints
    import openwam.train.utils.stage2_assets as assets
    import openwam.dataloader.registry as registry
    cfg=OmegaConf.create(dict(project=dict(seed=42),model=dict(architecture=dict(future_video_mode='none')),dataloader={},
        training={**dict(training()),'max_grad_norm':1.,'finetune_ckpt_path':'baseline'}))
    monkeypatch.setattr(comparison,'validate_inputs',lambda *a:{})
    monkeypatch.setattr(comparison,'render',lambda *a:None)
    monkeypatch.setattr(comparison,'evaluate',lambda *a:None)
    monkeypatch.setattr(OmegaConf,'load',lambda *a:cfg)
    sample=dict(action_mask=torch.ones(1,dtype=torch.bool),_eval=dict(family='tiny'))
    sample_file=tmp_path/'samples.pt';sample_file.write_bytes(b'fixture')
    config_file=tmp_path/'config.yaml';config_file.write_text('fixture')
    monkeypatch.setattr(torch,'load',lambda *a,**kw:[sample])
    monkeypatch.setattr(torch.cuda,'manual_seed_all',lambda *a:None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0)
    monkeypatch.setattr(torch.cuda,'max_memory_reserved',lambda:0)
    arch=TinyArchitecture()
    arch.set_dtype_device=lambda dtype,device:arch.to(dtype=dtype,device=device)
    arch.move_frozen_to_device=lambda *a:None
    arch.save_assets_for_deployment=lambda *a:None
    trainer=SimpleNamespace(architecture=arch,_ckpt_source_dir=None,
        compute_loss=lambda batch:dict(total=sum(p.float().sum() for p in arch.parameters() if p.requires_grad)))
    monkeypatch.setattr(trainers,'OpenWAMTrainer',lambda *a,**kw:trainer)
    accelerator=SimpleNamespace(device=torch.device('cpu'),unwrap_model=lambda value:value,
        accumulate=lambda *a:contextlib.nullcontext(),backward=lambda loss:loss.backward())
    monkeypatch.setattr(comparison,'build_single_gpu_accelerator',lambda:accelerator)
    real_build=comparison.build_master_optimizer
    def failing_optimizer(*args):
        optimizer,parameters=real_build(*args);original=optimizer.step;calls=[]
        def step():
            calls.append(1)
            if len(calls)==2:raise RuntimeError('simulated partial optimizer update')
            original()
        optimizer.step=step
        return optimizer,parameters
    monkeypatch.setattr(comparison,'build_master_optimizer',failing_optimizer)
    monkeypatch.setattr(registry,'build_dataset',lambda *a,**kw:None)
    monkeypatch.setattr(assets,'save_native_normalization',lambda *a:None)
    monkeypatch.setattr(checkpoints,'save_config',lambda *a:None)
    saved=[];monkeypatch.setattr(checkpoints,'save_weights',lambda *a,**kw:saved.append((a,kw)))
    args=SimpleNamespace(output=str(tmp_path/'run'),arm='none',config=str(config_file),samples=str(sample_file),
        seed=42,smoke=True,smoke_case_indices=None,smoke_case_limit=0,single_gpu_master=True,
        lr_scale=1.,warmup_updates=0,seconds_per_arm=60,smoke_updates=2,eval_every=100)
    with pytest.raises(RuntimeError,match='simulated partial optimizer update'):comparison.worker(args)
    assert saved==[]
