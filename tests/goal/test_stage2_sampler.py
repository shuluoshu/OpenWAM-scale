import pytest
import torch
from accelerate.data_loader import BatchSamplerShard
from torch.utils.data import BatchSampler
from openwam.train.utils.stage2 import GlobalDrawSampler, Stage2Progress, required_draws


def test_accelerate_shards_global_draws_once_without_padding():
    length=required_draws(max_updates=7, accumulation=3, batch_size=2, world_size=2)
    sampler=GlobalDrawSampler(length, seed=42, selection_identity='fixture')
    batches=BatchSampler(sampler, batch_size=2, drop_last=True)
    ranks=[list(BatchSamplerShard(batches,num_processes=2,process_index=rank,split_batches=False)) for rank in range(2)]
    ids=[{i for batch in rank for i in batch} for rank in ranks]
    assert ids[0].isdisjoint(ids[1])
    assert ids[0]|ids[1]==set(range(length))
    assert len(ranks[0])==7*3


def test_resume_uses_committed_offset_not_prefetch_cursor():
    sampler=GlobalDrawSampler(24,seed=42,selection_identity='fixture')
    iterator=iter(sampler)
    for _ in range(15):next(iterator)
    state=sampler.state_dict(committed_offset=8)
    restored=GlobalDrawSampler(24,seed=42,selection_identity='fixture')
    restored.load_state_dict(state)
    assert next(iter(restored))==8
    restored.set_epoch(1)
    assert next(iter(restored))==24
    for key,value in [('seed',43),('selection_identity','changed'),('draws_per_epoch',48),('version','wrong')]:
        with pytest.raises(ValueError,match=key):restored.load_state_dict({**state,key:value})


def test_progress_counts_only_successful_updates_and_global_consumption():
    progress=Stage2Progress(batch_size=2,world_size=2,accumulation=2)
    assert not progress.consume(sync_gradients=False,optimizer_step_skipped=False)
    assert progress.micro_step==1 and progress.opt_step==0 and progress.consumed_draws==4
    assert not progress.consume(sync_gradients=True,optimizer_step_skipped=True)
    assert progress.micro_step==2 and progress.opt_step==0
    assert not progress.consume(sync_gradients=False,optimizer_step_skipped=False)
    assert progress.consume(sync_gradients=True,optimizer_step_skipped=False)
    assert progress.opt_step==1 and progress.consumed_draws==16
    state=progress.state_dict()
    restored=Stage2Progress(batch_size=2,world_size=2,accumulation=2)
    restored.load_state_dict(state)
    assert restored.state_dict()==state
    with pytest.raises(ValueError,match='world_size'):
        Stage2Progress(batch_size=2,world_size=1,accumulation=2).load_state_dict(state)
    with pytest.raises(ValueError,match='update boundary'):
        restored.load_state_dict({**state,'micro_step':3})


def test_draw_identity_changes_augmentation_but_keeps_same_observed_goal(tmp_path,monkeypatch):
    import numpy as np
    from PIL import Image
    from tests.dataloader.test_curated_stage2 import make_native_source
    from openwam.dataloader.curated_stage2 import CuratedStage2Dataset
    from openwam.dataloader.native_stage2_stats import prepare_native_stats
    selection=make_native_source(tmp_path)
    preparation=CuratedStage2Dataset(selection)
    registry=prepare_native_stats(preparation,tmp_path/'stats',max_episodes_per_profile=1,max_windows_per_profile=16)
    dataset=CuratedStage2Dataset(selection,normalization_registry=registry,
        observation_augmentation={'enabled':True,'color_jitter':{'p':1,'brightness':.4,'contrast':.2,'saturation':.2}})
    monkeypatch.setattr(dataset,'_decode_views',lambda m,indices:([{'observation.images.cam_high':Image.new('RGB',(40,32),(100,120,140))} for _ in indices],{}))
    selected=dataset._draw(4)
    a=dataset._sample_selected(*selected,draw_id=4)
    again=dataset._sample_selected(*selected,draw_id=4)
    b=dataset._sample_selected(*selected,draw_id=5)
    assert a['goal_seed']==b['goal_seed']
    assert torch.equal(a['goal_observed_head'],b['goal_observed_head'])
    np.testing.assert_array_equal(np.asarray(a['video'][0]),np.asarray(again['video'][0]))
    assert not np.array_equal(np.asarray(a['video'][0]),np.asarray(b['video'][0]))
