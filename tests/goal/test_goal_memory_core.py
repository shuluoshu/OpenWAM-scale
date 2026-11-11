import pytest
import torch
from openwam.model.video_backbone.wan.models.dit import WanModel
from openwam.model.goal.goal_memory import GoalMemoryConfig, GoalMemoryExpert
from openwam.model.goal.goal_memory_cache import GoalMemoryRuntimeCache, PersistentFrameLatentCache


def fixture():
    torch.manual_seed(41)
    d = WanModel(dim=24, in_dim=48, ffn_dim=48, out_dim=48, text_dim=16,
        freq_dim=16, eps=1e-6, patch_size=(1,2,2), num_heads=2, num_layers=2,
        has_image_input=False, require_vae_embedding=False, require_clip_embedding=False,
        fuse_vae_embedding_in_latents=True)
    model = GoalMemoryExpert(d, GoalMemoryConfig(num_goal_tokens=3, latent_shape=(48,1,6,4), history_dropout=0)).eval()
    args = dict(current_latents=torch.randn(2,48,1,6,4), memory_latents=torch.randn(2,2,48,1,6,4),
        noisy_target_latents=torch.randn(2,48,1,6,4), timestep=torch.tensor([123.,812.]),
        text=torch.randn(2,5,16), goal_horizon_s=torch.tensor([3.,8.]))
    return model,args


def run(model,args,**extra):
    state=model.prepare_state(**(args|extra)); model.run_state(state)
    return state,model.outputs(state)


def test_causality_and_fixed_g_bottleneck():
    m,a=fixture()
    s,o=run(m,a)
    s2,o2=run(m,a,noisy_target_latents=a['noisy_target_latents']*5,timestep=torch.tensor([999.,1.]))
    torch.testing.assert_close(o['goal_tokens'],o2['goal_tokens'],atol=0,rtol=0)
    assert not torch.allclose(o['velocity'],o2['velocity'])
    # Pin G at every layer, including its K/V: history and text have no route to Y.
    base=m.prepare_state(**a)
    fixed={}
    for i in range(len(m.dit.blocks)):
        q,k,v,p=m.pre_attn_at_layer(i,base)
        fixed[i]=(k[:,base.goal_slice].detach(),v[:,base.goal_slice].detach())
        m.post_attn_at_layer(i,base,m.dit.blocks[i].self_attn.attn(q,k,v,attn_mask=base.attention_mask),p)
    changed=m.prepare_state(**(a|dict(memory_latents=a['memory_latents']*7,text=a['text']*-3)))
    for i in range(len(m.dit.blocks)):
        q,k,v,p=m.pre_attn_at_layer(i,changed)
        k=k.clone();v=v.clone(); k[:,changed.goal_slice],v[:,changed.goal_slice]=fixed[i]
        m.post_attn_at_layer(i,changed,m.dit.blocks[i].self_attn.attn(q,k,v,attn_mask=changed.attention_mask),p)
    torch.testing.assert_close(m.outputs(base)['velocity'],m.outputs(changed)['velocity'],atol=0,rtol=0)
    _,altered=run(m,a,text=a['text']*-3)
    assert not torch.allclose(o['goal_tokens'],altered['goal_tokens'])
    assert not torch.allclose(o['velocity'],altered['velocity'])


def test_native_block_equivalence():
    m,a=fixture();s=m.prepare_state(**a)
    s.text_query_gate.fill_(True)
    expected=m.dit.blocks[0](s.tokens,s.context,s.modulation,s.frequencies,
        context_mask=s.context_mask,self_attn_mask=s.attention_mask)
    q,k,v,p=m.pre_attn_at_layer(0,s)
    m.post_attn_at_layer(0,s,m.dit.blocks[0].self_attn.attn(q,k,v,attn_mask=s.attention_mask),p)
    torch.testing.assert_close(s.tokens,expected,atol=0,rtol=0)


def test_empty_padding_and_gradients():
    m,a=fixture();m.train()
    text=a['text'].clone();text[0]=float('nan')
    mem=a['memory_latents'].clone();mem[0]=float('nan')
    target=a['noisy_target_latents'].clone();target[0]=float('nan')
    result=m.training_loss(a['current_latents'],mem,target,text,torch.tensor([[[float('nan')]*2]*2,[[.2,.3],[.8,.5]]]),
        text_mask=torch.tensor([[False]*5,[True]*3+[False]*2]),
        memory_valid=torch.tensor([[False,False],[True,True]]),target_image_valid=torch.tensor([False,True]),
        target_point_valid=torch.tensor([[False,False],[True,True]]),target_in_frame=torch.ones(2,2,dtype=torch.bool),
        total_updates=100,update=100)
    assert torch.isfinite(result['loss']);result['loss'].backward()
    assert m.goal_queries.grad is not None and m.goal_queries.grad.abs().sum()>0
    assert m.dit.blocks[0].self_attn.q.weight.grad.abs().sum()>0
    assert m.uv_head[-1].weight.grad.abs().sum()>0
    _,o=run(m,a,memory_latents=a['memory_latents'][:,:0],text=a['text'][:,:0])
    assert torch.isfinite(o['velocity']).all()


def test_cache_and_window_equivalence():
    m,a=fixture();cache=GoalMemoryRuntimeCache(model_identity='test')
    with torch.no_grad():
        s,o=run(m,a)
        cs,co=run(m,a,runtime_cache=cache,episode_id='e',memory_frame_ids=torch.tensor([[1,2],[1,2]]))
        for key in o: torch.testing.assert_close(o[key],co[key],atol=2e-6,rtol=2e-6)
        hits=cache.hits
        _,again=run(m,a,runtime_cache=cache,episode_id='e',memory_frame_ids=torch.tensor([[1,2],[1,2]]))
        assert cache.hits>hits
        window=a|dict(memory_latents=a['memory_latents'][:,1:])
        _,uncached=run(m,window)
        _,cached=run(m,window,runtime_cache=cache,episode_id='e',memory_frame_ids=torch.tensor([[2],[2]]))
        for key in uncached: torch.testing.assert_close(uncached[key],cached[key],atol=2e-6,rtol=2e-6)
        # Frame evolution alone equals its block in a multi-frame window.
        solo,_=run(m,window)
        torch.testing.assert_close(s.tokens[:,s.memory_slices[1]],solo.tokens[:,solo.memory_slices[0]],atol=2e-6,rtol=2e-6)
        cache.set_episode('different'); assert len(cache)==0


def test_persistent_identity_and_atomic_roundtrip(tmp_path):
    c=PersistentFrameLatentCache(tmp_path,vae_identity='v1',preprocess_identity='p1',latent_shape=(48,1,6,4))
    x=torch.randn(48,1,6,4)
    c.put(source='s',episode='e',frame=1,view='canvas',latent=x)
    torch.testing.assert_close(c.get(source='s',episode='e',frame=1,view='canvas'),x)
    other=PersistentFrameLatentCache(tmp_path,vae_identity='v2',preprocess_identity='p1',latent_shape=(48,1,6,4))
    assert other.get(source='s',episode='e',frame=1,view='canvas') is None
    with pytest.raises(ValueError): PersistentFrameLatentCache(tmp_path,vae_identity='',preprocess_identity='p')


def test_invalid_views_and_all_empty_loss():
    m,a=fixture();m.train()
    cv=torch.tensor([[False,False,False],[True,False,True]])
    current=a['current_latents'].clone();current[0]=float('nan');current[1,:,:,4:,:2]=float('nan')
    result=m.training_loss(current,a['memory_latents']*float('nan'),a['noisy_target_latents']*float('nan'),
        a['text']*float('nan'),torch.full((2,2,2),float('nan')),
        observed_view_valid=cv,memory_valid=torch.zeros(2,2,dtype=torch.bool),text_mask=torch.zeros(2,5,dtype=torch.bool),
        target_image_valid=torch.zeros(2,dtype=torch.bool),target_point_valid=torch.zeros(2,2,dtype=torch.bool),target_in_frame=torch.zeros(2,2,dtype=torch.bool))
    assert result['loss'].item()==0
    result['loss'].backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())


def test_warmup_horizon_and_public_kv_override():
    m,a=fixture()
    assert m.warmup_probability(0,100)==.5
    assert m.warmup_probability(10,100)==.25
    assert m.warmup_probability(20,100)==0
    s,o=run(m,a)
    _,changed=run(m,a,goal_horizon_s=torch.tensor([10.,99.]))
    torch.testing.assert_close(o['goal_tokens'],changed['goal_tokens'],rtol=0,atol=0)
    assert not torch.allclose(o['velocity'],changed['velocity'])
    warm=m.prepare_state(**a,warmup_y_text=torch.tensor([True,False]))
    assert warm.text_query_gate[0,warm.target_slice].all()
    assert not warm.text_query_gate[1,warm.target_slice].any()
    q,k,v,p=m.pre_attn_at_layer(0,s)
    override={0:(torch.zeros_like(k[:,s.goal_slice]),torch.ones_like(v[:,s.goal_slice]))}
    s=m.prepare_state(**a,goal_kv_override=override)
    _,k,v,_=m.pre_attn_at_layer(0,s)
    assert (k[:,s.goal_slice]==0).all() and (v[:,s.goal_slice]==1).all()


def test_cache_invalidates_after_parameter_update():
    m,a=fixture();cache=GoalMemoryRuntimeCache(model_identity='test')
    kwargs=dict(runtime_cache=cache,episode_id='e',memory_frame_ids=torch.tensor([[1,2],[1,2]]))
    with torch.no_grad():
        run(m,a,**kwargs); misses=cache.misses
        m.dit.blocks[0].self_attn.k.weight.add_(.01)
        _,co=run(m,a,**kwargs);_,o=run(m,a)
    assert cache.misses>misses
    for key in o: torch.testing.assert_close(o[key],co[key],atol=2e-6,rtol=2e-6)


def test_thirty_native_layers_and_120_target_tokens():
    m,_=fixture()
    import copy
    m.dit.blocks=torch.nn.ModuleList([copy.deepcopy(m.dit.blocks[0]) for _ in range(30)])
    m.config.latent_shape=(48,1,24,20)
    args=dict(current_latents=torch.randn(1,48,1,24,20),memory_latents=torch.randn(1,1,48,1,24,20),
              noisy_target_latents=torch.randn(1,48,1,24,20),timestep=torch.tensor([500.]),text=torch.randn(1,3,16))
    with torch.no_grad(): s,o=run(m,args)
    assert s.target_slice.stop-s.target_slice.start==120
    assert o['velocity'].shape==(1,48,1,24,20)
    assert torch.isfinite(o['velocity']).all()


def test_missing_target_views_have_zero_loss_and_gradient():
    m,a=fixture();m.train();m.config.history_dropout=0;m.config.history_noise_mix=0
    views=torch.tensor([[True,False,False],[False,False,False]])
    target=a['noisy_target_latents'].clone();target[0,:,:,4:]=float('nan');target[1]=float('nan')
    noise=torch.randn_like(target);noise[0,:,:,4:]=float('nan');noise[1]=float('nan')
    kwargs=dict(target_image_valid=torch.ones(2,dtype=torch.bool),target_view_valid=views,
        target_point_valid=torch.zeros(2,2,dtype=torch.bool),target_in_frame=torch.zeros(2,2,dtype=torch.bool),
        sigma=torch.tensor([.5,.5]),noise=noise,update=100,total_updates=100)
    r=m.training_loss(a['current_latents'],a['memory_latents'],target,a['text'],torch.zeros(2,2,2),**kwargs)
    expected=(r['velocity'][0,:,:,:4]-(noise[0,:,:,:4]-target[0,:,:,:4])).square().mean()
    torch.testing.assert_close(r['fm_loss'],expected)
    assert torch.isfinite(r['loss'])
    r['loss'].backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())


def test_wrist_only_targets_exclude_head_camera_uv_supervision():
    m,a=fixture();m.train();m.config.history_dropout=0;m.config.history_noise_mix=0
    target=a['noisy_target_latents']
    noise=torch.randn_like(target)
    r=m.training_loss(a['current_latents'],a['memory_latents'],target,a['text'],
        torch.tensor([[[.2,.3],[.8,.5]]]*2),
        target_image_valid=torch.ones(2,dtype=torch.bool),
        target_view_valid=torch.tensor([[False,True,True]]*2),
        target_point_valid=torch.ones(2,2,dtype=torch.bool),
        target_in_frame=torch.ones(2,2,dtype=torch.bool),
        sigma=torch.tensor([.5,.5]),noise=noise,update=100,total_updates=100)
    expected=(r['velocity'][:,:,:,4:]-(noise[:,:,:,4:]-target[:,:,:,4:])).square().mean()
    torch.testing.assert_close(r['fm_loss'],expected)
    assert r['fm_loss'].item()>0
    assert r['valid_points'].item()==0
    assert r['uv_loss'].item()==0
    r['loss'].backward()
    assert m.dit.head.head.weight.grad.abs().sum()>0
    assert m.uv_head[-1].weight.grad.abs().sum()==0
