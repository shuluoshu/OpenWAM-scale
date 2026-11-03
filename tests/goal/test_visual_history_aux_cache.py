"""Auxiliary video + C*: native tiny Wan/ActionDiT, on CPU or opt-in CUDA."""
import copy
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir

from test_patch_stage2_memory import make_arch, inputs
from test_visual_history_stage2 import load_encoder
from openwam.model.goal.conditioning import VisualHistoryMemoryAdapter

DEVICE = os.environ.get('OPENWAM_TEST_DEVICE', 'cpu')
DTYPE = torch.bfloat16 if os.environ.get('OPENWAM_TEST_DTYPE') == 'bfloat16' else torch.float32


def model(mode='isolated', future='joint'):
    a = make_arch()
    a.cfg['future_video_mode'] = future
    a.cfg['attention_mask_mode'] = mode
    a._mot_driver_kwargs['attention_mask_mode'] = mode
    a.build_mot_driver()
    from openwam.model.video_backbone.wan.shared.diffusion.flow_match import FlowMatchScheduler
    a.video_backbone._scheduler = FlowMatchScheduler(template='Wan')
    a.init_training_schedulers(10)
    a.goal_adapter = VisualHistoryMemoryAdapter(a.video_backbone, dim=48)
    a.to(device=DEVICE, dtype=DTYPE)
    a.set_dtype_device(DTYPE, torch.device(DEVICE))
    return a


def batch(b=1):
    x = inputs()
    x = {k: v.repeat(b, *([1] * (v.ndim-1))).to(device=DEVICE, dtype=DTYPE) for k,v in x.items()}
    x['first_frame_latents'] = x['latents'][:, :, :1].clone()
    x['goal_latent'] = torch.randn(b,120,48,device=DEVICE,dtype=DTYPE)
    x['goal_token_valid'] = torch.ones(b,120,device=DEVICE,dtype=torch.bool)
    x['goal_token_valid'][0,80:] = False
    if b > 1:
        x['goal_token_valid'][1] = False
    x['seq_lens'] = torch.tensor([2]*b,device=DEVICE)
    return x


def assert_close(a,b):
    tol = 0.025 if DTYPE == torch.bfloat16 else 3e-5
    torch.testing.assert_close(a,b,atol=tol,rtol=tol)


def test_auxiliary_video_independent_actions_and_live_cstar():
    a=model().eval(); x=batch()
    y=a(**x)
    changed={**x,'latents':x['latents'].clone(),'timestep':x['timestep']*.1}
    changed['latents'][:,:,1:] += 5
    z=a(**changed)
    assert_close(y[1],z[1])
    assert not torch.allclose(y[0][:,:,1:].float(),z[0][:,:,1:].float())
    invisible=x['goal_latent'].clone(); invisible[:,80:]+=25
    assert_close(y[1],a(**{**x,'goal_latent':invisible})[1])
    visible=x['goal_latent'].clone(); visible[:,:80]+=5
    assert not torch.allclose(y[1].float(),a(**{**x,'goal_latent':visible})[1].float())
    changed={**x,'latents':x['latents'].clone()};changed['latents'][:,:,:1]+=2
    changed['first_frame_latents']=changed['latents'][:,:,:1]
    assert not torch.allclose(y[1].float(),a(**changed)[1].float())


def test_auxiliary_video_loss_reaches_shared_backbone_and_adapter(monkeypatch):
    a=model().train();x=batch();x['goal_latent'].requires_grad_(True)
    # Exercise the actual FM objective, including timestep sampling and masks.
    kwargs={k:v for k,v in x.items() if k not in ('noisy_actions','action_timestep','timestep','latents')}
    video_losses=[]
    original=a._compute_video_loss
    def capture(*args,**kw):
        loss=original(*args,**kw);video_losses.append(loss);return loss
    monkeypatch.setattr(a,'_compute_video_loss',capture)
    out=a.compute_loss(actions=x['noisy_actions'],input_latents=x['latents'],**kwargs)
    assert out['loss_video'].isfinite() and out['loss_video'].item()>0
    video_losses[0].backward()
    assert a.video_backbone.dit.patch_embedding.weight.grad.float().abs().sum()>0
    assert a.goal_adapter.projection.weight.grad.float().abs().sum()>0
    assert x['goal_latent'].grad is None


def test_auxiliary_preset_and_rejected_incompatible_memory():
    from openwam.train.utils.stage2 import validate_stage2_config
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[2]/'configs'),version_base=None):
        cfg=compose(config_name='goal/stage2_visual_history_aux_gpu')
    validate_stage2_config(cfg)
    assert cfg.model.architecture.attention_mask_mode=='isolated'
    assert cfg.model.architecture.future_video_mode=='joint'
    assert cfg.training.lambda_video>0 and cfg.model.architecture.action_kv_cache
    cfg.model.architecture.video_attention_mask_mode='bidirectional'
    with pytest.raises(ValueError):validate_stage2_config(cfg)


@pytest.mark.parametrize('mode,future',[('isolated','joint'),('mutual','none')])
def test_cached_native_steps_match_uncached_and_masks(mode,future):
    a=model(mode,future).eval();x=batch(2)
    cache=a.prepare_action_kv_cache({k:v for k,v in x.items() if k not in ('noisy_actions','action_timestep','timestep')})
    for step in (800.,400.,100.):
        x['action_timestep'].fill_(step)
        with torch.no_grad():
            expected=a(**x)[1]
            actual=a.forward_action_kv_cache(x['noisy_actions'],x['action_timestep'],cache)
        assert_close(actual,expected)
        x['noisy_actions']=x['noisy_actions']-.1*actual
    assert all(not t.requires_grad for pair in cache.video_kv for t in pair)
    assert cache.goal_keys is not None


def test_cache_eligibility_preserves_unsupported_paths():
    a=model().eval()
    assert a.can_cache_action_only(decode_video=False,cfg_scale=1.)
    assert not a.can_cache_action_only(decode_video=True,cfg_scale=1.)
    assert not a.can_cache_action_only(decode_video=False,cfg_scale=2.)
    a._mot_driver_kwargs['attention_mask_mode']='mutual';a.build_mot_driver()
    assert not a.can_cache_action_only(decode_video=False,cfg_scale=1.)
    a.cfg['future_video_mode']='none'
    assert a.can_cache_action_only(decode_video=False,cfg_scale=1.)
    a.train()
    with pytest.raises(ValueError,match='eval'):
        a.prepare_action_kv_cache(batch())


def test_warmstart_optin_never_relaxes_resume():
    from omegaconf import OmegaConf
    from openwam.train.utils.stage2 import validate_stage2_baseline_architecture
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[2]/'configs'),version_base=None):
        cfg=compose(config_name='goal/stage2_visual_history_aux_gpu')
    baseline=copy.deepcopy(cfg);baseline.model.architecture.attention_mask_mode='mutual'
    validate_stage2_baseline_architecture(cfg,baseline,resume=False)
    with pytest.raises(ValueError,match='attention_mask_mode'):
        validate_stage2_baseline_architecture(cfg,baseline,resume=True)
    cfg.training.allow_attention_mask_warmstart=False
    with pytest.raises(ValueError,match='attention_mask_mode'):
        validate_stage2_baseline_architecture(cfg,baseline,resume=False)


def test_generate_cache_matches_real_cstar_pipeline_and_prefills_once(tmp_path,monkeypatch):
    from test_visual_history import history
    from test_multiview_patch import multiview_batch
    from test_visual_history_stage2 import config as goal_config
    _,encoder,_=load_encoder(tmp_path)
    a=model().eval();a.goal_expert=encoder.to(device=DEVICE,dtype=DTYPE)
    a.goal_config=goal_config();a.goal_asset_identity='aux-cache-test'
    from openwam.model.goal.runtime import GoalPredictionCache
    a.goal_cache=GoalPredictionCache()
    x=batch()
    def preprocess(**kwargs):
        n=1 if kwargs['num_frames']==1 else 3
        return dict(context=x['context'].clone(),seq_lens=x['seq_lens'],
            latents=x['latents'][:,:,:n].clone(),first_frame_latents=x['first_frame_latents'].clone())
    monkeypatch.setattr(a.video_backbone,'preprocess_input_for_inference',preprocess)
    obs,_,_,views=multiview_batch(1)
    obs=obs.to(device=DEVICE,dtype=DTYPE);views=views.to(DEVICE)
    h={k:v.to(device=DEVICE,dtype=DTYPE if v.is_floating_point() else v.dtype) for k,v in history().items()}
    counts=dict(video=0,adapter=0)
    orig=a.video_backbone.pre_attn_at_layer
    def video(*args,**kwargs):counts['video']+=1;return orig(*args,**kwargs)
    monkeypatch.setattr(a.video_backbone,'pre_attn_at_layer',video)
    handle=a.goal_adapter.register_forward_hook(lambda *args:counts.__setitem__('adapter',counts['adapter']+1))
    active=torch.zeros(112,dtype=torch.bool,device=DEVICE);active[:10]=True
    kwargs=dict(schedule=[(1000.,1000.),(600.,600.),(200.,200.),(0.,0.)],prompt='move',
        action_num_frames=33,num_frames=9,seed=73,decode_video=False,
        goal_observed_multiview=obs,goal_view_valid=views,active_action_mask=active,**h)
    expected=a.generate(**kwargs,use_action_kv_cache=False)
    assert counts==dict(video=6,adapter=3)
    counts.update(video=0,adapter=0)
    actual=a.generate(**kwargs,use_action_kv_cache=True)
    assert counts==dict(video=2,adapter=1)
    assert_close(torch.from_numpy(actual['normalized_actions']),torch.from_numpy(expected['normalized_actions']))
    assert actual['video'] is None and np.all(actual['normalized_actions'][:,10:]==0)
    assert all(p.grad is None and not p.requires_grad for p in a.goal_expert.parameters())
    # Each request must refresh visual and goal conditions, never retain old KV.
    x['first_frame_latents']+=2;x['latents'][:,:,:1]=x['first_frame_latents']
    fresh=a.generate(**{**kwargs,'goal_observed_multiview':obs*.3},use_action_kv_cache=True)
    reference=a.generate(**{**kwargs,'goal_observed_multiview':obs*.3},use_action_kv_cache=False)
    assert_close(torch.from_numpy(fresh['normalized_actions']),torch.from_numpy(reference['normalized_actions']))
    assert not np.allclose(actual['normalized_actions'],fresh['normalized_actions'])
    with pytest.raises(ValueError,match='predicted goals only'):
        a.generate(**kwargs,use_action_kv_cache=True,goal_latent=x['goal_latent'])
    handle.remove()


def test_cached_context_padding_and_proprio_follow_native_path():
    a=model().eval()
    a._init_proprio_context(dict(use_proprioception=True,state_dim=112),text_dim=16)
    a.proprio_encoder.to(device=DEVICE,dtype=DTYPE)
    x=batch(2);x['proprio']=torch.randn(2,1,112,device=DEVICE,dtype=DTYPE)
    x['_proprio_sample_mask']=torch.tensor([[True],[False]],device=DEVICE)
    with torch.no_grad():
        y=a(**x)[1]
        cache=a.prepare_action_kv_cache({k:v for k,v in x.items() if k not in ('noisy_actions','action_timestep','timestep')})
        assert_close(y,a.forward_action_kv_cache(x['noisy_actions'],x['action_timestep'],cache))
        changed={**x,'context':x['context'].clone()}
        changed['context'][:,2:]+=100
        cache2=a.prepare_action_kv_cache({k:v for k,v in changed.items() if k not in ('noisy_actions','action_timestep','timestep')})
        assert_close(y,a.forward_action_kv_cache(x['noisy_actions'],x['action_timestep'],cache2))
        changed={**x,'proprio':x['proprio']+2}
        cache3=a.prepare_action_kv_cache({k:v for k,v in changed.items() if k not in ('noisy_actions','action_timestep','timestep')})
        out=a.forward_action_kv_cache(x['noisy_actions'],x['action_timestep'],cache3)
        assert_close(out,a(**changed)[1])
        assert not torch.allclose(y[0].float(),out[0].float())
        assert_close(y[1],out[1])


def test_action_gradients_exclude_future_video_and_joint_checkpointing_matches():
    a=model().train();b=copy.deepcopy(a);b.build_mot_driver();x=batch()
    x['latents'].requires_grad_(True)
    expected=a(**x)
    expected[1].float().square().mean().backward()
    assert x['latents'].grad[:,:,1:].count_nonzero()==0
    assert a.action_backbone.action_encoder.weight.grad.float().abs().sum()>0
    assert a.goal_adapter.projection.weight.grad.float().abs().sum()>0
    actual=b(**x,use_gradient_checkpointing=True)
    for u,v in zip(expected,actual):assert_close(u,v)
    actual[1].float().square().mean().backward()
    assert_close(a.goal_adapter.projection.weight.grad,b.goal_adapter.projection.weight.grad)


def test_isolated_runtime_loads_original_frozen_cstar_asset(tmp_path):
    from openwam.model.goal import runtime
    from test_visual_history_stage2 import config as goal_config
    full,encoder,path=load_encoder(tmp_path)
    # Dependency identity includes dtype: save the fixture in its loading dtype.
    from openwam.train.goal_visual_history import save_visual_history_bundle
    full.to(dtype=DTYPE)
    full.codec.vae.to(dtype=DTYPE)  # shared dependency is deliberately not a child module
    save_visual_history_bundle(full,path,precomputed_text_fingerprint='a'*64)
    a=model();a.video_backbone.vae=full.codec.vae.to(device=DEVICE,dtype=DTYPE)
    cfg=goal_config();cfg.update(checkpoint=str(path),precomputed_text_fingerprint='a'*64)
    runtime.configure_goal(a,dict(goal=cfg,model=dict(goal_representation='visual_history_masked_sa',query_history={'enabled':True})))
    assert not a.goal_expert.training
    assert all(not p.requires_grad for p in a.goal_expert.parameters())
    assert a.goal_expert.representation=='visual_history_masked_sa'
    assert not hasattr(a.goal_expert.dit,'blocks')


def test_generate_without_clean_prefix_falls_back_to_full_path(monkeypatch):
    a=model().eval();x=batch();seen=[]
    def preprocess(**kwargs):
        seen.append(kwargs['num_frames'])
        return dict(context=x['context'].clone(),latents=x['latents'].clone())
    monkeypatch.setattr(a.video_backbone,'preprocess_input_for_inference',preprocess)
    kwargs=dict(schedule=[(1000.,1000.),(500.,500.),(0.,0.)],prompt='move',
                action_num_frames=33,num_frames=9,seed=73,decode_video=False)
    expected=a.generate(**kwargs,use_action_kv_cache=False)
    actual=a.generate(**kwargs,use_action_kv_cache=True)
    assert seen==[9,9]
    assert_close(torch.from_numpy(expected['normalized_actions']),torch.from_numpy(actual['normalized_actions']))


def test_generate_multiframe_prefix_override_falls_back(monkeypatch):
    a=model().eval();x=batch()
    def preprocess(**kwargs):
        return dict(context=x['context'].clone(),latents=x['latents'].clone(),
                    first_frame_latents=x['first_frame_latents'].clone())
    monkeypatch.setattr(a.video_backbone,'preprocess_input_for_inference',preprocess)
    kwargs=dict(schedule=[(1000.,1000.),(500.,500.),(0.,0.)],prompt='move',
                action_num_frames=33,num_frames=9,seed=73,decode_video=False,
                first_frame_latents=x['latents'][:,:,:2].clone())
    expected=a.generate(**kwargs,use_action_kv_cache=False)
    actual=a.generate(**kwargs,use_action_kv_cache=True)
    assert_close(torch.from_numpy(expected['normalized_actions']),torch.from_numpy(actual['normalized_actions']))
