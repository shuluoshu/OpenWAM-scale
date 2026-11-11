import pytest
import torch
from test_goal_memory_core import fixture
from openwam.model.architectures.dual_system.goal_memory import GoalMemoryPolicy
from openwam.model.action_backbone.separate_action_dit import ActionDiT


def setup(mode='isolated', future='none'):
    v,a=fixture()
    ab=ActionDiT(action_dim=112,dim=24,ffn_dim=48,num_heads=2,num_layers=2,video_dim=24,bridge_layers=[0,1],variant='joint_self_attn',attn_head_dim=12,text_dim=16)
    p=GoalMemoryPolicy(v,ab,torch.nn.Linear(112,16),attention_mask_mode=mode,future_video_mode=future).eval()
    a['target_timestep']=a.pop('timestep')
    a.update(noisy_actions=torch.randn(2,32,112),action_timestep=torch.tensor([100.,500.]),proprio=torch.randn(2,112))
    return p,a


def test_isolated_deletion_causality_and_cache():
    p,a=setup(future='joint');a['noisy_future_latents']=torch.randn(2,48,2,6,4)
    with torch.no_grad():
        x=p(**a);b={k:v for k,v in a.items() if k not in ('noisy_target_latents','noisy_future_latents')};y=p(**b)
        torch.testing.assert_close(x['action_velocity'],y['action_velocity'],atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(x['goal_tokens'],y['goal_tokens'],atol=1e-6,rtol=1e-5)
        c=p.prefill(**b);z=p.action_from_prefill(c,a['noisy_actions'],a['action_timestep'],a['text'],a['proprio'])
        torch.testing.assert_close(z,y['action_velocity'],atol=1e-6,rtol=1e-5)


def test_video_never_reads_action_and_future_route():
    p,a=setup('action_sees_video','joint');a['noisy_future_latents']=torch.randn(2,48,2,6,4)
    x=p(**a);y=p(**(a|{'noisy_actions':a['noisy_actions']*3}))
    torch.testing.assert_close(x['future_velocity'],y['future_velocity'],atol=0,rtol=0)
    z=p(**(a|{'noisy_future_latents':a['noisy_future_latents']*3}))
    assert not torch.allclose(x['action_velocity'],z['action_velocity'])
    x['action_velocity'].square().mean().backward()
    assert p.video.goal_queries.grad.abs().sum()>0


def loss_batch(a):
    b={k:v for k,v in a.items() if k not in ('noisy_actions','action_timestep','noisy_target_latents','target_timestep')}
    b.update(actions=a['noisy_actions'],action_mask=torch.ones_like(a['noisy_actions'],dtype=torch.bool),target_latents=a['noisy_target_latents'],
        target_image_valid=torch.ones(2,dtype=torch.bool),target_view_valid=torch.ones(2,3,dtype=torch.bool))
    return b


def test_losses_frozen_noaux_and_padding():
    p,a=setup(future='joint');p.train();b=loss_batch(a)
    b.update(future_latents=torch.randn(2,48,2,6,4),future_view_valid=torch.ones(2,2,3,dtype=torch.bool))
    b['actions'][0]=float('nan');b['action_mask'][0]=False
    b['future_latents'][0]=float('nan');b['future_view_valid'][0]=False
    out=p.training_loss(**b)
    assert torch.isfinite(out['loss']);out['loss'].backward()
    assert p.video.goal_queries.grad.abs().sum()>0
    p.zero_grad();p.lambda_keyframe=0;p.freeze_video=True;p.video.requires_grad_(False)
    out=p.training_loss(**b);out['loss'].backward()
    assert all(t.grad is None for t in p.video.parameters())
    assert p.action_backbone.action_encoder.weight.grad.abs().sum()>0
    torch.testing.assert_close(out['loss'],out['action_loss'])
    assert not out['keyframe_fm_loss'].requires_grad


def test_action_goal_dropout_and_history_sensitivity():
    p,a=setup();a.pop('noisy_target_latents');x=p(**a)
    z=p(**(a|{'memory_latents':a['memory_latents']*3}))
    assert not torch.allclose(x['action_velocity'],z['action_velocity'])
    p.goal_dropout=1
    x=p(**a);x['action_velocity'].square().mean().backward()
    assert p.video.goal_queries.grad is None or not p.video.goal_queries.grad.any()
    p.zero_grad();p.goal_dropout=0
    z=p(**a);assert not torch.allclose(x['action_velocity'],z['action_velocity'])
    z['action_velocity'].square().mean().backward();assert p.video.goal_queries.grad.abs().sum()>0


def test_native_stage1_baseline_migration_and_portable_bundle(tmp_path):
    from safetensors.torch import save_file
    from openwam.train.goal_memory_checkpoint import save_memory_bundle
    from openwam.train.goal_memory_stage2 import build_stage2,save_policy_bundle,load_policy_bundle
    p,a=setup()
    dc=dict(dim=24,in_dim=48,ffn_dim=48,out_dim=48,text_dim=16,freq_dim=16,eps=1e-6,patch_size=(1,2,2),num_heads=2,num_layers=2,has_image_input=False,require_vae_embedding=False,require_clip_embedding=False,fuse_vae_embedding_in_latents=True)
    p.video.dit_config=dc
    save_memory_bundle(p.video,tmp_path/'s1.safetensors',dit_config=dc,dependencies={'id':'immutable'})
    # Actual native 80D baseline, strictly padded through the existing migration.
    ab=ActionDiT(action_dim=80,dim=24,ffn_dim=96,num_heads=2,num_layers=2,video_dim=24,bridge_layers=[0,1],variant='joint_self_attn',attn_head_dim=12,text_dim=16,freq_dim=16)
    weights={'action_backbone.'+k:v for k,v in ab.state_dict().items()}
    weights.update({'proprio_encoder.'+k:v for k,v in torch.nn.Linear(80,16).state_dict().items()})
    save_file(weights,str(tmp_path/'base.safetensors'))
    cfg=dict(stage1_checkpoint=str(tmp_path/'s1.safetensors'),baseline_checkpoint=str(tmp_path/'base.safetensors'),initialization={'expand_action_dim_from_checkpoint':True})
    p=build_stage2(cfg,dit=None,dependencies={'id':'immutable'},device='cpu',dtype=torch.float32).eval()
    assert not p.action_backbone.action_encoder.weight[:,80:].any()
    torch.testing.assert_close(p.action_backbone.action_encoder.weight[:,:80],ab.action_encoder.weight,atol=0,rtol=0)
    save_policy_bundle(p,tmp_path/'policy.safetensors',dependencies={'id':'immutable'})
    loaded=load_policy_bundle(tmp_path/'policy.safetensors',dependencies={'id':'immutable'}).eval()
    with torch.no_grad():torch.testing.assert_close(p(**a)['action_velocity'],loaded(**a)['action_velocity'],atol=0,rtol=0)


def test_checkpoint_gradients_and_inference_prefill_once():
    import copy
    p,a=setup('action_sees_video','joint');p.train();p.video.config.history_noise_mix=0
    a['noisy_future_latents']=torch.randn(2,48,2,6,4)
    q=copy.deepcopy(p);q.video.config.use_gradient_checkpointing=False
    x=p(**a);y=q(**a)
    x['action_velocity'].square().mean().backward();y['action_velocity'].square().mean().backward()
    for (n,v),(n2,w) in zip(p.named_parameters(),q.named_parameters()):
        assert n==n2
        if v.grad is not None:torch.testing.assert_close(v.grad,w.grad,atol=1e-6,rtol=1e-5)
    p,a=setup();p.eval();count=[]
    original=p.video.pre_attn_at_layer
    def call(*args,**kwargs):count.append(1);return original(*args,**kwargs)
    p.video.pre_attn_at_layer=call
    batch={k:v for k,v in a.items() if k not in ('noisy_actions','action_timestep','noisy_target_latents','target_timestep')}
    output=p.infer_actions(a['noisy_actions'],steps=4,**batch)
    assert output.shape==(2,32,112) and len(count)==2


def test_runtime_memory_cache_equivalence_and_clear():
    from openwam.model.goal.goal_memory_cache import GoalMemoryRuntimeCache
    p,a=setup();a.pop('noisy_target_latents')
    cache=GoalMemoryRuntimeCache(model_identity='unit')
    a.update(memory_frame_ids=torch.tensor([[0,8],[0,8]]),episode_id='ep1')
    with torch.no_grad():
        x=p(**a);y=p(**(a|{'runtime_cache':cache}))
        torch.testing.assert_close(x['action_velocity'],y['action_velocity'],atol=1e-6,rtol=1e-5)
        # Current/history window change exercises pre-RoPE memory reuse.
        a['memory_frame_ids']=torch.tensor([[8,9],[8,9]])
        a['memory_latents']=a['memory_latents'].flip(1)
        x=p(**a);y=p(**(a|{'runtime_cache':cache}))
        torch.testing.assert_close(x['action_velocity'],y['action_velocity'],atol=1e-6,rtol=1e-5)


@pytest.mark.parametrize('mode',['none','joint'])
def test_real_native_provider_aligns_source_and_crosses_epoch(tmp_path,mode):
    import av
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tests.dataloader.test_curated_stage2 import make_native_source
    from openwam.dataloader.curated_stage2 import CuratedStage2Dataset
    from openwam.dataloader.native_stage2_stats import prepare_native_stats
    from openwam.dataloader.rfc_goal import RFCGoalDataset
    from openwam.train.goal_memory_data import LatentBatchProvider
    from openwam.train.goal_memory_stage2 import stage2_provider
    from openwam.model.goal.memory_codec import GoalMemoryCodec
    from openwam.train.goal_curated import TextContextCache
    from test_rfc_goal_data import VAE
    path=make_native_source(tmp_path)
    root=tmp_path/'source/interndata_a1/unit';camera='observation.images.cam_high'
    epfile=root/'meta/episodes/chunk-000/file-000.parquet';ep=pq.read_table(epfile).to_pylist()[0]
    ep.update({'dataset_from_index':0,f'videos/{camera}/chunk_index':0,f'videos/{camera}/file_index':0,f'videos/{camera}/from_timestamp':0.})
    pq.write_table(pa.Table.from_pylist([ep]),epfile)
    video=root/f'videos/{camera}/0.mp4';video.parent.mkdir(parents=True)
    with av.open(str(video),'w') as container:
        stream=container.add_stream('mpeg4',rate=30);stream.width=40;stream.height=32;stream.pix_fmt='yuv420p'
        for i in range(24):
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.full((32,40,3),i*5,dtype=np.uint8),format='rgb24')):container.mux(packet)
        for packet in stream.encode():container.mux(packet)
    native=CuratedStage2Dataset(path,future_video_mode='none')
    registry=prepare_native_stats(native,tmp_path/'stats',max_episodes_per_profile=1,max_windows_per_profile=16)
    class TemporalVAE(VAE):
        def encode(self,videos,device):
            assert len(videos)==1 and videos[0].shape==(3,9,384,320)
            return torch.zeros(1,48,3,24,20,device=device)
    codec=GoalMemoryCodec(TemporalVAE(),tmp_path/'latent-cache',vae_identity='fixture')
    dataset=RFCGoalDataset(path,tmp_path/'reader-cache',codec=codec,require_cache=False,epoch_size=10,subtask_probability=.7)
    text=TextContextCache(lambda _: (torch.ones(1,5,16),torch.tensor([5])))
    base=LatentBatchProvider(dataset,codec,text,device='cpu',dtype=torch.float32)
    config=dict(dataset={'source_yaml':str(path)},policy={'future_video_mode':mode},action_data={'normalization_registry':str(registry)})
    provide=stage2_provider(config,base,dataset,codec,device='cpu',dtype=torch.float32)
    for step in (0,1,9,10,21):
        batch=provide(step);selected=dataset.selection_at(step%10)
        assert batch['actions'].shape==(1,32,112) and batch['action_mask'].any()
        assert batch['_metadata']['action_source']['anchor']==selected['anchor']
        assert batch['_metadata']['instruction']==selected['segment']['text']
        assert batch['_metadata']['granularity']==selected['level']
    if mode=='none':assert 'future_latents' not in batch
    else:
        assert batch['future_latents'].shape==(1,48,2,24,20)
        assert batch['future_view_valid'].shape==(1,2,3)
    Path=__import__('pathlib').Path
    Path(registry).write_text(Path(registry).read_text()+'\n')
    import pytest
    with pytest.raises(ValueError,match='immutable'):provide(22)


@pytest.mark.parametrize('mode',['none','joint'])
def test_bfloat16_training_preserves_native_input_dtype(mode):
    p,a=setup(future=mode);p=p.to(dtype=torch.bfloat16).train()
    batch=loss_batch(a)
    batch={k:v.to(dtype=torch.bfloat16) if torch.is_tensor(v) and v.is_floating_point() else v for k,v in batch.items()}
    if mode=='joint':batch.update(future_latents=torch.randn(2,48,2,6,4,dtype=torch.bfloat16),future_view_valid=torch.ones(2,2,3,dtype=torch.bool))
    out=p.training_loss(**batch)
    assert torch.isfinite(out['loss'])
    out['loss'].backward()


def test_swapped_goal_changes_action_and_no_g_closes_that_route():
    p,a=setup();a.pop('noisy_target_latents')
    override=torch.randn(2,3,24)*4
    x=p(**a)['action_velocity'];y=p(**(a|{'goal_tokens_override':override}))['action_velocity']
    assert not torch.allclose(x,y)
    p.goal_dropout=1
    x=p(**a)['action_velocity'];y=p(**(a|{'goal_tokens_override':override}))['action_velocity']
    torch.testing.assert_close(x,y,atol=0,rtol=0)


@pytest.mark.parametrize('seed_fp32_cache', [False, True])
def test_autocast_default_bypasses_runtime_cache(seed_fp32_cache):
    from openwam.model.goal.goal_memory_cache import GoalMemoryRuntimeCache
    p,a=setup();a.pop('noisy_target_latents')
    a.update(memory_frame_ids=torch.tensor([[0,8],[0,8]]),episode_id='autocast')
    cache=GoalMemoryRuntimeCache(model_identity='autocast')
    with torch.no_grad():
        if seed_fp32_cache:p(**(a|{'runtime_cache':cache}))
        before=(len(cache),cache.hits,cache.misses)
        with torch.autocast('cpu',dtype=torch.bfloat16):
            expected=p(**a)['action_velocity']
            actual=p(**(a|{'runtime_cache':cache}))['action_velocity']
        assert (len(cache),cache.hits,cache.misses)==before
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)


def test_approximate_autocast_cache_invalidates_before_fp32_reuse():
    from openwam.model.goal.goal_memory_cache import GoalMemoryRuntimeCache
    p,a=setup();p.allow_approximate_low_precision_cache=True;a.pop('noisy_target_latents')
    a.update(memory_frame_ids=torch.tensor([[0,8],[0,8]]),episode_id='autocast')
    cache=GoalMemoryRuntimeCache(model_identity='autocast')
    with torch.no_grad():
        with torch.autocast('cpu',dtype=torch.bfloat16):
            p(**(a|{'runtime_cache':cache}))
            assert len(cache)==4
            assert any(t.dtype==torch.bfloat16 for entry in cache.entries.values() for layer in entry for t in layer)
            p(**(a|{'runtime_cache':cache}))
            assert cache.hits==4
        hits,misses=cache.hits,cache.misses
        actual=p(**(a|{'runtime_cache':cache}))['action_velocity']
        assert cache.hits==hits and cache.misses==misses+4
        expected=p(**a)['action_velocity']
        torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)


def test_joint_inference_defaults_future_camera_mask_from_observed():
    from openwam.model.goal.goal_memory_masks import view_token_mask
    p,a=setup('action_sees_video','joint')
    batch={k:v for k,v in a.items() if k not in ('noisy_actions','action_timestep','noisy_target_latents','target_timestep')}
    batch['observed_view_valid']=torch.tensor([[True,False,True],[True,True,False]])
    future=torch.randn(2,48,2,6,4)
    valid=batch['observed_view_valid'][:,None].expand(-1,2,-1)
    pixels=view_token_mask(valid,3,2).reshape(2,2,3,2).repeat_interleave(2,-2).repeat_interleave(2,-1)[:,None]
    changed=torch.where(pixels,future,future+20)
    x=p.infer_actions(a['noisy_actions'],steps=2,future_noise=future,**batch)
    y=p.infer_actions(a['noisy_actions'],steps=2,future_noise=changed,**batch)
    torch.testing.assert_close(x,y,atol=0,rtol=0)
    explicit=p.infer_actions(a['noisy_actions'],steps=2,future_noise=future,**(batch|{'future_view_valid':valid}))
    torch.testing.assert_close(x,explicit,atol=0,rtol=0)
