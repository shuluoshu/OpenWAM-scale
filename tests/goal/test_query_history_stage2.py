import pytest
import torch
from openwam.model.goal.runtime import goal_cache_key, validate_goal_model_config, validate_goal_compatibility


def config(enabled):
    return dict(representation='query_latent', format='openwam.goal.query.v1', history={'enabled':enabled})


def test_history_model_contract():
    with pytest.raises(ValueError, match='history'):
        validate_goal_model_config(config(True), {'query_history':{'enabled':False}})


def test_history_checkpoint_contract():
    with pytest.raises(ValueError, match='history'):
        validate_goal_compatibility(config(True), config(False))


def test_history_cache_binds_ages_masks_and_features():
    kwargs=dict(observed=torch.zeros(1),context=torch.zeros(1),context_len=None,prompt='x',seed=0,num_steps=1,asset_identity='a',shift=1)
    h=dict(history_latents=torch.zeros(10,48,1,16,20),history_ages=torch.arange(10,0,-1).float(),history_valid=torch.ones(10,dtype=torch.bool))
    first=goal_cache_key(**kwargs,**h)
    for field in h:
        changed={k:v.clone() for k,v in h.items()}
        changed[field].flatten()[0] = False if field=='history_valid' else 42
        assert goal_cache_key(**kwargs,**changed)!=first


def test_actual_frozen_runtime_cache_and_missing_head_contract():
    from test_query_stage2 import attach
    from test_query_history import history_expert, history
    from test_multiview_patch import multiview_batch
    from openwam.model.goal.runtime import prepare_goal_inputs, goal_prediction_policy
    from dataclasses import asdict
    a=attach(); a.goal_expert=history_expert().freeze()
    a.goal_config['history']=asdict(a.goal_expert.history_config)
    obs,ctx,_,views=multiview_batch(1); h=history()
    def run(**extra):
        return prepare_goal_inputs(a,None,['a'],ctx,head_valid=[True],seeds=[1],
            goal_observed_multiview=obs,goal_view_valid=views,**extra)
    first=run(**h)['goal_latent']
    assert not first.requires_grad
    assert all(not p.requires_grad for p in a.goal_expert.parameters())
    assert len(a.goal_cache.entries)==1
    torch.testing.assert_close(run(**h)['goal_latent'],first,rtol=0,atol=0)
    h['history_ages']*=2
    assert not torch.allclose(run(**h)['goal_latent'],first)
    assert len(a.goal_cache.entries)==2
    assert goal_prediction_policy(a)['history']['enabled']
    with pytest.raises(ValueError,match='history'):
        prepare_goal_inputs(a,None,['a'],ctx,head_valid=[False],seeds=[1],
            goal_observed_multiview=obs,goal_view_valid=torch.zeros_like(views))
    h['history_valid'][:]=False
    assert torch.isfinite(run(**h)['goal_latent']).all()


def test_history_encoder_only_checkpoint_runtime(tmp_path):
    from test_query_history import history_expert, history
    from test_multiview_patch import multiview_batch
    from openwam.train.goal_query import save_query_bundle
    from openwam.model.goal.runtime import load_stage2_goal, predict_goals, GoalPredictionCache
    expert=history_expert().freeze(); path=tmp_path/'history.safetensors'
    save_query_bundle(expert,path,precomputed_text_fingerprint="a"*64)
    restored=load_stage2_goal(path,vae=expert.codec.vae,text_encoder=None,tokenizer=None,
        text_dim=16,device='cpu',dtype=torch.float32,precomputed_text_fingerprint='a'*64)
    assert restored.history_config==expert.history_config
    assert not hasattr(restored,'dit')
    obs,ctx,_,views=multiview_batch(1); h=history()
    kwargs=dict(observed=obs,context=ctx,context_lens=None,seeds=[0],num_steps=1,
        cache=GoalPredictionCache(),asset_identity='fixture',prompts=['move'],augmented=False,
        observed_view_valid=views,**h)
    before,_=predict_goals(expert=expert,**kwargs)
    after,_=predict_goals(expert=restored,**{**kwargs,'cache':GoalPredictionCache()})
    torch.testing.assert_close(before.latent,after.latent,rtol=0,atol=0)
