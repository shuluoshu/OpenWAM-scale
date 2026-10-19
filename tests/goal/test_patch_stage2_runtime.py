import pytest
import torch
from safetensors.torch import save_file
from openwam.model.goal.patch_expert import PatchGoalPrediction
from openwam.model.goal.runtime import GoalPredictionCache, goal_cache_key, load_stage2_goal, predict_goals, to_goal_condition
from openwam.train.goal_patch import make_smoke_data, save_patch_bundle

def prediction():
    return PatchGoalPrediction(torch.zeros(1,48,1,16,20,requires_grad=True),torch.zeros(1,2,2,dtype=torch.bfloat16,requires_grad=True),torch.zeros(1,192),torch.ones(1,81,dtype=torch.bool))

def test_condition_contract():
    p=prediction(); p.token_valid[:,7]=False; c=to_goal_condition(p)
    assert c.image_token_valid.shape==(1,80) and not c.image_token_valid[0,7]
    assert c.uv.dtype==torch.float32 and not c.uv.requires_grad and not c.latent.requires_grad
    p.token_valid=p.token_valid[:,:80]
    with pytest.raises(ValueError,match='81'): to_goal_condition(p)

def test_cache_bounds():
    c=to_goal_condition(prediction()); size=sum(t.numel()*t.element_size() for t in vars(c).values())
    cache=GoalPredictionCache(maxsize=2,max_bytes=size*2)
    cache.put('a',c); cache.put('b',c); cache.get('a'); cache.put('c',c)
    assert cache.get('b') is None and cache.bytes_used==size*2
    hit=cache.get('a'); hit.latent.fill_(9)
    assert cache.get('a').latent.sum()==0
    assert all(t.device.type=='cpu' for t in vars(hit).values())
    tiny=GoalPredictionCache(maxsize=2,max_bytes=size-1); tiny.put('x',c)
    assert tiny.get('x') is None and tiny.bytes_used==0

def test_cache_invalidation():
    kw=dict(observed=torch.zeros(3,1,256,320),context=torch.zeros(2,8),context_len=2,prompt='move',seed=1,num_steps=1,asset_identity='a',shift=5.)
    key=goal_cache_key(**kw)
    for name,value in dict(observed=torch.ones(3,1,256,320),context=torch.ones(2,8),context_len=1,prompt='stop',seed=2,num_steps=2,asset_identity='b',shift=4.,preprocessing_version='next').items():
        assert goal_cache_key(**{**kw,name:value})!=key

def test_bundle_and_generation(tmp_path):
    torch.set_num_threads(1)
    expert,batches,_=make_smoke_data(); b=batches[0]; path=tmp_path/'patch.safetensors'
    save_patch_bundle(expert,path,precomputed_text_fingerprint='a'*64)
    deps=dict(vae=expert.codec.vae,text_encoder=None,tokenizer=None,text_dim=b['context'].shape[-1],device='cpu',dtype=torch.float32)
    loaded=load_stage2_goal(path,**deps,precomputed_text_fingerprint='a'*64)
    assert not loaded.training and all(not p.requires_grad for p in loaded.parameters())
    loaded.train(); assert not loaded.training
    with pytest.raises(ValueError,match='text dependency'): load_stage2_goal(path,**deps,precomputed_text_fingerprint='b'*64)
    legacy=tmp_path/'legacy.safetensors'; save_file({'x':torch.zeros(1)},str(legacy))
    with pytest.raises(ValueError,match='patch'): load_stage2_goal(legacy,**deps)
    observed=b['observed'].repeat(2,1,1,1,1); context=b['context'].repeat(2,1,1)
    args=dict(expert=loaded,observed=observed,context=context,context_lens=None,seeds=[7,9],num_steps=1,cache=GoalPredictionCache(),asset_identity='tiny',prompts=['a','b'],augmented=False,microbatch_size=2)
    first,stats=predict_goals(**args); assert stats['generated_samples']==2
    _,stats=predict_goals(**args); assert stats['generated_samples']==0
    second,stats=predict_goals(**{**args,'seeds':[9,7],'prompts':['b','a'],'observed':observed.flip(0),'context':context.flip(0),'augmented':True,'microbatch_size':1})
    assert stats['generated_samples']==2
    torch.testing.assert_close(first.latent,second.latent.flip(0),atol=2e-5,rtol=2e-5)
    torch.testing.assert_close(first.uv,second.uv.flip(0),atol=2e-5,rtol=2e-5)

def test_cached_latent_preserves_expert_dtype():
    class BF16Expert:
        shift=5.
        def freeze(self): return self
        def generate(self, observed, context, **kwargs):
            p=prediction(); p.latent=p.latent.to(torch.bfloat16)
            return p
    args=dict(expert=BF16Expert(),observed=torch.zeros(1,3,1,256,320),
        context=torch.zeros(1,2,8),context_lens=None,seeds=[1],num_steps=1,
        cache=GoalPredictionCache(),asset_identity='bf16',prompts=['x'],augmented=False)
    miss,_=predict_goals(**args); hit,_=predict_goals(**args)
    assert miss.latent.dtype==hit.latent.dtype==torch.bfloat16
