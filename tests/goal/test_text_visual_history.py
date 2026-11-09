"""Text must affect exported C*, including when history is absent or masked."""
import pytest
import torch

from test_expert import TINY_DIT, TinyVAE
from test_multiview_patch import multiview_batch
from test_visual_history import history
from test_visual_history_stage2 import config as visual_config, make_arch
from openwam.model.goal import runtime
from openwam.model.goal.codec import GoalCodec
from openwam.model.goal.visual_history import VisualHistoryGoalExpert
from openwam.model.video_backbone.wan.models.dit import WanModel
from openwam.train.goal_visual_history import save_visual_history_bundle


def make_text_model():
    torch.manual_seed(123)
    return VisualHistoryGoalExpert(WanModel(**TINY_DIT), GoalCodec(TinyVAE()),
        dit_config=TINY_DIT, num_history_tokens=5, pool_dim=32, pool_heads=4,
        language_conditioning=True)


def text_config():
    return {**visual_config(), 'format':'openwam.visual_history.v2',
            'language_conditioning':True}


@pytest.mark.parametrize('history_mode', ['present', 'masked', 'absent'])
def test_text_affects_cstar_without_a_history_dependency(history_mode):
    expert=make_text_model().eval()
    obs,ctx,_,views=multiview_batch(2)
    h={} if history_mode=='absent' else history(2)
    if history_mode=='masked':
        h['history_valid'].zero_();h['history_latents'].fill_(float('nan'))
        h['history_ages'].fill_(float('nan'))
    a=expert.condition_tokens(obs,ctx,observed_view_valid=views,**h)
    b=expert.condition_tokens(obs,ctx+3,observed_view_valid=views,**h)
    assert a.shape==(2,120,48) and torch.isfinite(a).all()
    assert not torch.allclose(a,b)
    if history_mode=='masked':
        torch.testing.assert_close(a,expert.condition_tokens(obs,ctx,observed_view_valid=views),rtol=0,atol=0)


def test_text_padding_is_ignored_and_context_is_required():
    e=make_text_model().eval();obs,ctx,_,views=multiview_batch(2)
    kwargs=dict(observed_view_valid=views,context_lens=torch.tensor([2,3]),**history(2))
    expected=e.condition_tokens(obs,ctx,**kwargs)
    padded=ctx.clone();padded[0,2:]=float('nan');padded[1,3:]=float('nan')
    torch.testing.assert_close(expected,e.condition_tokens(obs,padded,**kwargs),rtol=0,atol=0)
    with pytest.raises(ValueError,match='context|text'):
        e.condition_tokens(obs,observed_view_valid=views)
    for lens in (torch.tensor([0,2]),torch.tensor([2.,2.5]),torch.tensor([5,2])):
        with pytest.raises(ValueError,match='length'):
            e.condition_tokens(obs,ctx,**{**kwargs,'context_lens':lens})
    padded[0,0]=float('nan')
    with pytest.raises(ValueError,match='finite'):
        e.condition_tokens(obs,padded,**kwargs)


@pytest.mark.parametrize('checkpointing',[False,True])
def test_terminal_supervision_reaches_text_conditioner(checkpointing):
    e=make_text_model().train();e.use_gradient_checkpointing=checkpointing
    obs,ctx,targets,views=multiview_batch(1);h=history()
    out=e(obs,ctx,targets,observed_view_valid=views,context_lens=torch.tensor([2]),**h)
    expected=e.condition_tokens(obs,ctx,observed_view_valid=views,context_lens=torch.tensor([2]),**h)
    torch.testing.assert_close(out['condition_tokens'],expected)
    for loss_name in ('fm_loss','tcp_loss'):
        e.zero_grad();out[loss_name].backward(retain_graph=True)
        for module in (e.condition_text_embedding,e.visual_adapter.cross_attn,e.history_pool):
            grads=[p.grad for p in module.parameters() if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads)
            assert sum(g.abs().sum() for g in grads)>0


def test_bundle_stage2_runtime_and_cache_preserve_text(tmp_path):
    e=make_text_model().freeze();path=tmp_path/'text.safetensors'
    meta=save_visual_history_bundle(e,path,precomputed_text_fingerprint='a'*64)
    assert meta['format']=='openwam.visual_history.v2' and meta['history']['language_conditioning']
    encoder=runtime.load_stage2_goal(path,vae=e.codec.vae,text_encoder=None,tokenizer=None,
        text_dim=16,device='cpu',dtype=torch.float32,precomputed_text_fingerprint='a'*64)
    assert encoder.language_conditioning and hasattr(encoder,'condition_text_embedding')
    assert not hasattr(encoder.dit,'blocks') and not hasattr(encoder.dit,'text_embedding')
    obs,ctx,_,views=multiview_batch(1);h=history()
    kwargs=dict(observed_view_valid=views,context_lens=torch.tensor([2]),**h)
    torch.testing.assert_close(e.condition_tokens(obs,ctx,**kwargs),encoder(obs,ctx,**kwargs),rtol=0,atol=0)
    a=make_arch();a.goal_expert=encoder;a.goal_config=text_config()
    runtime.configure_goal_adapter(a,a.goal_config);a.goal_cache=runtime.GoalPredictionCache()
    a.goal_asset_identity='text-fixture'
    def run(context,prompt):
        return runtime.prepare_goal_inputs(a,None,[prompt],context,head_valid=[True],seeds=[3],
            context_lens=torch.tensor([2]),goal_observed_multiview=obs,goal_view_valid=views,**h)['goal_latent']
    first=run(ctx,'task');second=run(ctx+3,'subtask')
    assert not torch.allclose(first,second)
    torch.testing.assert_close(first,run(ctx,'task'),rtol=0,atol=0)
    policy=runtime.goal_prediction_policy(a)
    assert policy['format']=='openwam.visual_history.v2' and policy['language_conditioning']
    assert not any(p.requires_grad for p in encoder.parameters())
    cfg={**text_config(),'checkpoint':str(path),'precomputed_text_fingerprint':'a'*64}
    b=make_arch();b.video_backbone.vae=e.codec.vae
    runtime.configure_goal(b,dict(goal=cfg,model=dict(goal_representation=cfg['representation'],query_history={'enabled':True})))
    with pytest.raises(ValueError,match='format|language'):
        runtime.configure_goal(b,dict(goal={**cfg,'format':'openwam.visual_history.v1','language_conditioning':False},
            model=dict(goal_representation=cfg['representation'],query_history={'enabled':True})))


def test_language_mode_is_a_strict_configuration_contract():
    runtime.goal_representation(text_config())
    with pytest.raises(ValueError,match='language|format'):
        runtime.goal_representation({**text_config(),'language_conditioning':False})
    with pytest.raises(ValueError,match='language|format'):
        runtime.goal_representation({**visual_config(),'language_conditioning':True})
    with pytest.raises(ValueError,match='language|format'):
        runtime.validate_goal_compatibility(text_config(),visual_config())
    from openwam.train.stage12 import validate_stage1_stage2_contract
    s1=dict(model=dict(goal_representation='visual_history_masked_sa',visual_history={'language_conditioning':True}),
        observation_mode='multiview',components=dict(dit=dict(extra_kwargs=dict(dim=48))),dataset=dict(history={}))
    s2=dict(model=dict(goal_representation='visual_history_masked_sa'),goal=text_config(),dataloader=dict(history={}))
    validate_stage1_stage2_contract(s1,s2)
    with pytest.raises(ValueError,match='language'):
        validate_stage1_stage2_contract(s1,{**s2,'goal':visual_config()})


@pytest.mark.parametrize('mode',['joint','none'])
def test_text_presets_preserve_the_production_contract(mode):
    from pathlib import Path
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from openwam.train.goal_curated import load_config,goal_parameter_groups
    from openwam.train.stage12 import validate_stage1_stage2_contract
    from openwam.train.utils.stage2 import validate_stage2_config
    root=Path(__file__).resolve().parents[2]
    with initialize_config_dir(config_dir=str(root/'configs'),version_base=None):
        cfg=compose(config_name='goal/stage2_text_visual_history_'+mode+'_gpu')
    stage2=OmegaConf.to_container(cfg,resolve=True)
    for suffix in ('production','gpu_zero1'):
        stage1=load_config(root/f'configs/goal/stage1_text_visual_history_relative112_{suffix}.yaml')
        validate_stage1_stage2_contract(stage1,stage2)
    validate_stage2_config(cfg)
    assert stage2['goal']['visual_history']==dict(dim=3072,num_tokens=120)
    e=make_text_model().train_stage_a()
    groups=goal_parameter_groups(e,1e-5,1e-4)
    parameters=[id(p) for g in groups for p in g['params']]
    assert len(parameters)==len(set(parameters))
    assert set(parameters)=={id(p) for p in e.parameters() if p.requires_grad}


@pytest.mark.parametrize('declared_format,language',[
    ('openwam.visual_history.v1',True),('openwam.visual_history.v2',False),
    ('openwam.visual_history.v2','true')])
def test_bundle_rejects_language_format_corruption(tmp_path,declared_format,language):
    import json
    from safetensors import safe_open
    from safetensors.torch import save_file
    from openwam.train.goal_visual_history import load_visual_history_bundle
    e=make_text_model();path=tmp_path/'broken.safetensors'
    meta=save_visual_history_bundle(e,path,precomputed_text_fingerprint='a'*64)
    meta['format']=declared_format;meta['history']['language_conditioning']=language
    with safe_open(str(path),framework='pt') as f:
        weights={k:f.get_tensor(k) for k in f.keys()}
    save_file(weights,str(path),metadata={'visual_history':json.dumps(meta)})
    with pytest.raises(ValueError,match='format|language'):
        load_visual_history_bundle(path,vae=e.codec.vae,text_dim=16,precomputed_text_fingerprint='a'*64)
