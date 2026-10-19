"""Exact goal latent geometry through actual Wan + action transformer blocks."""
from types import SimpleNamespace
import copy
import torch
from tests.goal.test_expert import TINY_DIT, TinyVAE
from openwam.model.video_backbone.wan.models.dit import WanModel
from openwam.model.video_backbone.wan_backbone import Wan22Ti2v
from openwam.model.architectures.dual_system.joint_self_attn import DualSystemSelfAttnArchitecture


def make_arch():
    torch.manual_seed(31)
    arch=DualSystemSelfAttnArchitecture(dict(video_dim=48,dim=48,ffn_dim=64,num_heads=2,
        attn_head_dim=24,bridge_interval=1,num_dit_layers=2,action_dim=112,text_dim=16,
        attention_mask_mode='mutual',video_attention_mask_mode='first_frame_causal'))
    arch.video_backbone=Wan22Ti2v(SimpleNamespace(dit=WanModel(**{**TINY_DIT,'num_layers':2}),vae=TinyVAE()))
    arch.set_dtype_device(torch.float32,torch.device('cpu'));arch.build_mot_driver()
    return arch


def inputs():
    return dict(noisy_actions=torch.randn(1,32,112),action_timestep=torch.tensor([500.]),
        latents=torch.randn(1,48,3,24,20),timestep=torch.tensor([500.]),context=torch.randn(1,4,16))


def test_memory_is_read_only_obs_isolated_and_adapter_has_gradient():
    from openwam.model.goal.conditioning import GoalMemoryAdapter
    a=make_arch();a.goal_adapter=GoalMemoryAdapter(a.video_backbone)
    assert a.goal_adapter.patch.in_features==192 and a.goal_adapter.patch.out_features==512
    x=inputs();g=torch.randn(1,48,1,16,20);valid=torch.ones(1,80,dtype=torch.bool)
    base=a(**x);disabled=a(**x,goal_latent=None)
    assert all(torch.equal(v,w) for v,w in zip(base,disabled))
    out=a(**x,goal_latent=g,goal_token_valid=valid)
    changed=a(**x,goal_latent=g+3,goal_token_valid=valid)
    assert torch.allclose(out[0][:,:,:1],changed[0][:,:,:1],atol=1e-6)
    assert not torch.allclose(out[0][:,:,1:],changed[0][:,:,1:])
    assert not torch.allclose(out[1],changed[1])
    (out[0].square().mean()+out[1].square().mean()).backward()
    assert a.goal_adapter.patch.weight.grad.abs().sum()>0
    assert a.action_backbone.action_encoder.weight.grad.abs().sum()>0
    assert not any('norm_k' in k for k in a.goal_adapter.state_dict())
    assert all('q' not in name for name,_ in a.goal_adapter.named_parameters())


def test_checkpoint_goal_tensors_and_gradients_match():
    from openwam.model.goal.conditioning import GoalMemoryAdapter
    a=make_arch();a.goal_adapter=GoalMemoryAdapter(a.video_backbone);b=copy.deepcopy(a);b.build_mot_driver()
    x=inputs();x.update(goal_latent=torch.randn(1,48,1,16,20),goal_token_valid=torch.ones(1,80,dtype=torch.bool))
    y=a(**x);z=b(**x,use_gradient_checkpointing=True)
    for v,w in zip(y,z):torch.testing.assert_close(v,w)
    sum(v.square().mean() for v in y).backward();sum(v.square().mean() for v in z).backward()
    torch.testing.assert_close(a.goal_adapter.patch.weight.grad,b.goal_adapter.patch.weight.grad)


def test_compiled_joint_loop_consumes_explicit_goal_tensors(monkeypatch):
    from openwam.model.goal.conditioning import GoalMemoryAdapter
    import openwam.model.architectures.dual_system.joint_self_attn as joint
    a=make_arch().eval();a.goal_adapter=GoalMemoryAdapter(a.video_backbone)
    x=inputs();x.update(goal_latent=torch.randn(1,48,1,16,20),goal_token_valid=torch.ones(1,80,dtype=torch.bool))
    with torch.no_grad():expected=a(**x)
    # Real Dynamo capture with portable eager backend, not a fake compiled callable.
    monkeypatch.setattr(joint,'torch_compile_kwargs',lambda *args,**kwargs:dict(backend='eager',fullgraph=True))
    a.apply_compile_optimizations({'enabled':True})
    with torch.no_grad():actual=a(**x)
    assert a._compiled_mot_run_joint_loop is not None, 'compile silently fell back to eager'
    for u,v in zip(expected,actual):torch.testing.assert_close(u,v)
    with torch.no_grad():changed=a(**{**x,'goal_latent':x['goal_latent']+1})
    assert not torch.allclose(actual[1],changed[1])


def test_goal_mask_preserves_rectangular_base_and_invalid_columns():
    from openwam.model.goal.conditioning import append_goal_mask
    base=torch.rand(12,17)>.5;valid=torch.ones(2,80,dtype=torch.bool);valid[1,4:]=False
    mask=append_goal_mask(base,valid,5)
    assert mask.shape==(2,1,12,97)
    assert torch.equal(mask[0,0,:,:17],base) and torch.equal(mask[1,0,:,:17],base)
    assert not mask[:,:,:5,17:].any()
    assert mask[0,0,5:,17:].all() and not mask[1,0,5:,21:].any()


def test_explicit_row_column_and_goal_type_embeddings_receive_gradients():
    from openwam.model.goal.conditioning import GoalMemoryAdapter
    a=make_arch();adapter=GoalMemoryAdapter(a.video_backbone)
    assert hasattr(adapter,'row_position') and hasattr(adapter,'column_position') and hasattr(adapter,'goal_type')
    k,v=adapter(torch.randn(1,48,1,16,20),a.video_backbone)
    (k.square().mean()+v.square().mean()).backward()
    assert adapter.row_position.shape==(1,8,1,512)
    assert adapter.column_position.shape==(1,1,10,512)
    assert adapter.goal_type.shape==(1,1,512)
    for p in (adapter.row_position,adapter.column_position,adapter.goal_type):
        assert p.grad.abs().sum()>0


def test_frozen_patch_teacher_optimizer_and_disabled_memory():
    from openwam.model.goal.conditioning import GoalMemoryAdapter
    from openwam.train.goal_contract import build_g1_parameter_groups
    from openwam.train.goal_patch import make_smoke_data
    a=make_arch(); a.goal_adapter=GoalMemoryAdapter(a.video_backbone)
    teacher,_,_=make_smoke_data(); a.goal_expert=teacher.freeze()
    a.video_backbone.text_encoder=torch.nn.Linear(16,16)
    a.train()
    assert a.training and a.video_backbone.dit.training and a.action_backbone.training
    assert not a.goal_expert.training and not a.video_backbone.vae.training and not a.video_backbone.text_encoder.training
    assert not ({id(p) for p in teacher.dit.parameters()} & {id(p) for p in a.video_backbone.dit.parameters()})
    groups=build_g1_parameter_groups(a)
    owned=[id(p) for group in groups for p in group['params']]
    assert len(owned)==len(set(owned))==sum(p.requires_grad for p in a.parameters())
    assert not set(owned)&{id(p) for p in teacher.parameters()}
    before={n:p.detach().clone() for n,p in teacher.named_parameters()}
    x=inputs(); g=torch.randn(1,48,1,16,20,requires_grad=True)
    # All invalid goal columns reproduce the exact base mutual output.
    with torch.no_grad():
        base=a(**x,goal_latent=g,goal_token_valid=torch.zeros(1,80,dtype=torch.bool))
        saved=a.goal_expert; del a.goal_expert
        disabled=a(**x); a.goal_expert=saved
    for u,v in zip(base,disabled): torch.testing.assert_close(u,v,atol=1e-6,rtol=1e-6)
    selected=[a.video_backbone.dit.patch_embedding.weight,
              a.action_backbone.action_encoder.weight,a.goal_adapter.patch.weight]
    initial=[p.detach().clone() for p in selected]
    optimizer=torch.optim.AdamW(groups)
    result=a(**x,goal_latent=g,goal_token_valid=torch.ones(1,80,dtype=torch.bool))
    sum(y.square().mean() for y in result).backward()
    assert g.grad is None
    for p in selected: assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
    optimizer.step()
    for p,old in zip(selected,initial): assert not torch.equal(p,old)
    for n,p in teacher.named_parameters():
        assert p.grad is None and torch.equal(p,before[n])


def test_prepare_clean_head_missing_rows_and_diagnostics(monkeypatch):
    from openwam.model.goal import runtime
    from openwam.model.goal.types import GoalCondition
    a=SimpleNamespace(goal_config={'num_steps':1},goal_expert=object(),goal_cache=None,goal_asset_identity='tiny')
    seen=[]
    def predict(**kwargs):
        seen.append(kwargs)
        b=kwargs['observed'].shape[0]
        return GoalCondition(torch.ones(b,48,1,16,20),torch.ones(b,80,dtype=torch.bool),torch.ones(b,2,2)),{}
    monkeypatch.setattr(runtime,'predict_goals',predict)
    ctx=torch.zeros(2,4,16); head=torch.rand(2,3,1,256,320)
    got=runtime.prepare_goal_inputs(a,head,['a','b'],ctx,head_valid=torch.tensor([True,False]),seeds=[17,23])
    assert set(got)=={'goal_latent','goal_token_valid'}
    assert got['goal_token_valid'][0].all() and not got['goal_token_valid'][1].any()
    assert seen[0]['seeds']==[17] and torch.equal(seen[0]['observed'],head[:1])
    assert not a.goal_diagnostics['predicted_uv'].requires_grad
    seen.clear()
    got=runtime.prepare_goal_inputs(a,None,['a','b'],ctx,head_valid=torch.zeros(2,dtype=torch.bool),seeds=[17,23])
    assert not seen and not got['goal_token_valid'].any()
    import pytest
    with pytest.raises(ValueError,match='explicit clean'):
        runtime.prepare_goal_inputs(a,None,['a','b'],ctx,head_valid=torch.ones(2,dtype=torch.bool),seeds=[17,23])


def test_prepare_inputs_threads_explicit_clean_head(monkeypatch):
    import pytest
    from openwam.model.goal import runtime
    a=make_arch(); a.goal_expert=torch.nn.Linear(1,1)
    a._pipeline_transform_instance=SimpleNamespace(apply=lambda sample: {**sample,'video':'augmented'})
    context=torch.zeros(1,4,16)
    a.preprocess=lambda **kwargs: {'context':context}
    head=torch.ones(3,1,256,320)
    sample=dict(video='raw',prompt='subtask',goal_observed_head=head,goal_head_valid=True,goal_seed=91)
    class Reached(Exception): pass
    def capture(arch,observed,prompts,ctx,**kwargs):
        assert arch is a and torch.equal(observed,head[None]) and prompts==['subtask'] and ctx is context
        assert kwargs['seeds']==[91] and kwargs['head_valid'].tolist()==[True]
        raise Reached
    monkeypatch.setattr(runtime,'prepare_goal_inputs',capture)
    with pytest.raises(Reached): a.prepare_inputs([sample])
    sample.pop('goal_observed_head')
    with pytest.raises(ValueError,match='explicit clean'): a.prepare_inputs([sample])


def test_configure_patch_teacher_from_versioned_asset(tmp_path):
    from openwam.model.goal.runtime import configure_goal
    from openwam.train.goal_patch import make_smoke_data,save_patch_bundle
    a=make_arch(); teacher,_,_=make_smoke_data()
    a.video_backbone.vae=teacher.codec.vae
    path=tmp_path/'patch.safetensors'; save_patch_bundle(teacher,path,precomputed_text_fingerprint='a'*64)
    configure_goal(a,{'goal':{'enabled':True,'checkpoint':str(path),'precomputed_text_fingerprint':'a'*64}})
    a.train()
    assert not a.goal_expert.training and not a.video_backbone.vae.training
    assert a.goal_adapter.training and a.goal_asset_identity
