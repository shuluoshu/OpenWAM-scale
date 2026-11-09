"""Later images must never update an earlier history representation."""
import pytest
import torch

from test_expert import TINY_DIT,TinyVAE
from test_multiview_patch import multiview_batch
from test_visual_history import history
from test_text_visual_history import text_config
from openwam.model.goal.codec import GoalCodec
from openwam.model.goal.visual_history import HistoryTokenPool,VisualHistoryGoalExpert
from openwam.model.video_backbone.wan.models.dit import WanModel


def make_causal_model():
    torch.manual_seed(123)
    return VisualHistoryGoalExpert(WanModel(**TINY_DIT),GoalCodec(TinyVAE()),
        dit_config=TINY_DIT,num_history_tokens=20,pool_dim=32,pool_heads=4,
        language_conditioning=True,causal_history=True)


def causal_config():
    return {**text_config(),'format':'openwam.visual_history.v3','causal_history':True}


def test_pool_prefix_causality_and_masked_empty_prefixes():
    torch.manual_seed(7)
    pool=HistoryTokenPool(48,20,32,4,causal=True).eval()
    patches=torch.randn(2,10,80,48);h=history(2)
    ages,valid=h['history_ages'],h['history_valid']
    a=pool(patches,ages,valid)
    changed=patches.clone();changed[:,5:]+=7
    b=pool(changed,ages,valid)
    torch.testing.assert_close(a[:,:10],b[:,:10],rtol=0,atol=0)
    assert not torch.allclose(a[:,10:],b[:,10:])
    valid[0,:3]=False;valid[1]=False
    a=pool(patches,ages,valid)
    patches[~valid]=float('nan');ages[~valid]=float('nan')
    torch.testing.assert_close(a,pool(patches,ages,valid),rtol=0,atol=0)
    assert torch.isfinite(a).all() and not a[0,:6].any() and not a[1].any()
    assert pool.token_valid(valid).tolist()==[[False]*6+[True]*14,[False]*20]


def test_fusion_history_cannot_read_current_or_later_history():
    e=make_causal_model().eval();obs,ctx,_,views=multiview_batch(1);h=history()
    captured=[]
    handle=e.visual_adapter.register_forward_hook(lambda m,i,o:captured.append(o.detach().clone()))
    c=e.condition_tokens(obs,ctx,observed_view_valid=views,**h)
    e.condition_tokens(obs*.4,ctx,observed_view_valid=views,**h)
    torch.testing.assert_close(captured[0][:,:20],captured[1][:,:20],rtol=0,atol=0)
    changed={**h,'history_latents':h['history_latents'].clone()}
    changed['history_latents'][:,5:]+=5
    later=e.condition_tokens(obs,ctx,observed_view_valid=views,**changed)
    handle.remove()
    torch.testing.assert_close(captured[0][:,:10],captured[2][:,:10],rtol=0,atol=0)
    assert not torch.allclose(c,later)
    assert not torch.allclose(c,e.condition_tokens(obs,ctx+3,observed_view_valid=views,**h))
    assert c.shape==(1,120,48)


def test_causal_empty_history_and_terminal_gradients():
    e=make_causal_model().train();e.use_gradient_checkpointing=True
    obs,ctx,target,views=multiview_batch(2);h=history(2)
    h['history_valid'][0]=False;h['history_latents'][0]=float('nan')
    out=e(obs,ctx,target,observed_view_valid=views,context_lens=torch.tensor([2,3]),**h)
    assert torch.isfinite(out['condition_tokens']).all() and torch.isfinite(out['loss'])
    out['loss'].backward()
    for module in (e.history_pool,e.visual_adapter,e.condition_text_embedding):
        grads=[p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum() for g in grads)>0
    e.eval();h['history_valid'].zero_();h['history_latents'].fill_(float('nan'))
    a=e.condition_tokens(obs,ctx,observed_view_valid=views,**h)
    b=e.condition_tokens(obs,ctx,observed_view_valid=views)
    torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_causal_bundle_runtime_and_compatibility(tmp_path):
    from openwam.train.goal_visual_history import save_visual_history_bundle
    from openwam.model.goal import runtime
    from test_visual_history_stage2 import make_arch
    e=make_causal_model().freeze();path=tmp_path/'causal.safetensors'
    metadata=save_visual_history_bundle(e,path,precomputed_text_fingerprint='a'*64)
    assert metadata['format']=='openwam.visual_history.v3' and metadata['history']['causal_history']
    encoder=runtime.load_stage2_goal(path,vae=e.codec.vae,text_encoder=None,tokenizer=None,
        text_dim=16,device='cpu',dtype=torch.float32,precomputed_text_fingerprint='a'*64)
    assert encoder.causal_history
    obs,ctx,_,views=multiview_batch(1);h=history()
    torch.testing.assert_close(e.condition_tokens(obs,ctx,observed_view_valid=views,**h),
        encoder(obs,ctx,observed_view_valid=views,**h),rtol=0,atol=0)
    cfg={**causal_config(),'checkpoint':str(path),'precomputed_text_fingerprint':'a'*64}
    a=make_arch();a.video_backbone.vae=e.codec.vae
    runtime.configure_goal(a,dict(goal=cfg,model=dict(goal_representation=e.representation,query_history={'enabled':True})))
    assert runtime.goal_prediction_policy(a)['causal_history']
    with pytest.raises(ValueError,match='causal|format'):
        runtime.validate_goal_compatibility(causal_config(),text_config())
    with pytest.raises(ValueError,match='causal|format'):
        runtime.goal_representation({**causal_config(),'causal_history':False})


@pytest.mark.parametrize('mode',['joint','none'])
def test_causal_presets_and_stage12_contract(mode):
    from pathlib import Path
    from hydra import compose,initialize_config_dir
    from omegaconf import OmegaConf
    from openwam.train.goal_curated import load_config
    from openwam.train.stage12 import validate_stage1_stage2_contract
    from openwam.train.utils.stage2 import validate_stage2_config
    root=Path(__file__).resolve().parents[2]
    with initialize_config_dir(config_dir=str(root/'configs'),version_base=None):
        cfg=compose(config_name='goal/stage2_causal_text_visual_history_'+mode+'_gpu')
    stage2=OmegaConf.to_container(cfg,resolve=True)
    for suffix in ('production','gpu_zero1'):
        s1=load_config(root/f'configs/goal/stage1_causal_text_visual_history_relative112_{suffix}.yaml')
        validate_stage1_stage2_contract(s1,stage2)
    validate_stage2_config(cfg)
    assert stage2['goal']['visual_history']==dict(dim=3072,num_tokens=120)
    stage2['goal']['causal_history']=False
    with pytest.raises(ValueError,match='causal'):
        validate_stage1_stage2_contract(s1,stage2)
