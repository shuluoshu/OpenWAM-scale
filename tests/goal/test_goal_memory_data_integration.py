import torch


def test_source_reader_codec_text_core_training_and_preflight(tmp_path):
    from test_rfc_goal_data import source,VAE
    from openwam.train.goal_memory_data import make_dataset,LatentBatchProvider,preflight
    from openwam.model.goal.memory_codec import GoalMemoryCodec
    from openwam.train.goal_curated import TextContextCache
    from scripts.data.precompute_goal_memory import precompute_goal_memory
    from openwam.model.goal.goal_memory import GoalMemoryConfig,GoalMemoryExpert
    from openwam.model.video_backbone.wan.models.dit import WanModel
    from openwam.train.goal_trainer import TINY_DIT_CONFIG
    from openwam.train.goal_memory import train_updates
    path,root=source(tmp_path)
    config=dict(stage=1,max_updates=10,checkpoint_every=1,learning_rate=1e-4,new_learning_rate=1e-3,
                dataset=dict(source_yaml=str(path),cache_dir=str(tmp_path/'reader'),subtask_probability=.7),
                memory_cache=dict(require_cache=True))
    vae=VAE();codec=GoalMemoryCodec(vae,tmp_path/'cache',vae_identity='fixture')
    dataset=make_dataset(config,codec)
    assert precompute_goal_memory(dataset)['complete']
    textcache=TextContextCache(lambda text:(torch.ones(1,8,16),torch.tensor([3])))
    provider=LatentBatchProvider(dataset,codec,textcache,device='cpu',dtype=torch.float32)
    batch=provider(0)
    assert batch['text_mask'].sum()==3
    assert batch['current_latents'].shape==(1,48,1,24,20)
    assert batch['target_latents'].shape==(1,48,1,24,20)
    assert vae.calls==2  # one precompute, one batch of current+target
    assert batch['_metadata']['granularity'] in ('task','subtask')
    ditconfig={**TINY_DIT_CONFIG,'num_layers':2}
    model=GoalMemoryExpert(WanModel(**ditconfig),GoalMemoryConfig())
    result=train_updates(model,provider,output_dir=tmp_path/'train',config=config,
                         dit_config=ditconfig,dependencies={'vae':'fixture'},
                         population_identity=dataset.population_identity,stop_after=1)
    assert result['update']==1
    assert result['last']['granularity']==batch['_metadata']['granularity']
    report=preflight(config,tmp_path/'preflight')
    assert report['model_loaded'] is False and len(report['samples'])==10


def test_configured_reader_episode_cache_bound_is_effective(tmp_path):
    from test_rfc_goal_data import source
    from openwam.train.goal_memory_data import make_dataset
    path, _ = source(tmp_path)
    config = dict(max_updates=1, dataset=dict(source_yaml=str(path), cache_dir=str(tmp_path/'reader')))
    assert make_dataset(config).reader.max_cached_episodes == 8
    config['dataset']['max_cached_episodes'] = 2
    assert make_dataset(config).reader.max_cached_episodes == 2
