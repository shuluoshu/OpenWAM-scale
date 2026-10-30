from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from openwam.dataloader.head_history import HistorySamplingConfig, select_history_frames
from openwam.model.goal.history import HeadFeatureCache, move_history


def test_fixed_ten_seconds_and_dropout_preserve_real_ages():
    cfg=HistorySamplingConfig(enabled=True,jitter_fraction=0.,frame_dropout_prob=.3)
    idx,age,valid=select_history_frames(600,30,0,cfg,np.random.default_rng(42),True)
    assert 0<valid.sum()<10
    np.testing.assert_array_equal(idx[valid],np.arange(300,600,30)[valid])
    np.testing.assert_allclose(age[valid],(600-idx[valid])/30)
    assert (age[~valid]==0).all() and (idx[~valid]==-1).all()
    idx,age,valid=select_history_frames(600,30,0,cfg,np.random.default_rng(42),False)
    assert valid.all()
    np.testing.assert_array_equal(age,np.arange(10,0,-1))


def test_start_and_short_history_never_wrap():
    cfg=HistorySamplingConfig(enabled=True,jitter_fraction=0.,frame_dropout_prob=0.)
    idx,age,valid=select_history_frames(0,30,0,cfg,np.random.default_rng(0),True)
    assert not valid.any() and (idx==-1).all() and (age==0).all()
    idx,age,valid=select_history_frames(75,30,0,cfg,np.random.default_rng(0),True)
    assert idx.tolist()==[-1]*8+[15,45]
    assert age.tolist()==[0]*8+[2,1]


def test_full_history_cache_cleared_on_episode_switch_even_identical_pixels():
    calls=[]
    def encode(x):
        calls.append(len(x)); return x.new_zeros((len(x),48,1,16,20))
    codec=SimpleNamespace(vae=torch.nn.Linear(1,1),encode=encode)
    cache=HeadFeatureCache(256)
    images=torch.zeros(1,10,3,1,256,320);valid=torch.ones(1,10,dtype=torch.bool)
    for ep,expected in [(10,1),(10,1),(20,2),(10,3)]:
        cache.encode(codec,images,valid,episode_ids=torch.tensor([ep]))
        assert len(calls)==expected
    assert cache.episode_id==10


def test_mixed_batch_clears_at_each_episode_and_missing_ids_do_not_reuse():
    calls=[]
    def encode(x):
        calls.append(1);return x.new_zeros((len(x),48,1,16,20))
    codec=SimpleNamespace(vae=torch.nn.Linear(1,1),encode=encode);cache=HeadFeatureCache(256)
    images=torch.zeros(3,10,3,1,256,320);valid=torch.ones(3,10,dtype=torch.bool)
    cache.encode(codec,images,valid,episode_ids=torch.tensor([1,2,1]))
    assert len(calls)==3
    cache.encode(codec,images[:1],valid[:1]);cache.encode(codec,images[:1],valid[:1])
    assert len(calls)==5


def test_episode_id_transfer_preserves_integer_precision():
    ids=torch.tensor([2**60+7],dtype=torch.int64)
    assert torch.equal(move_history({'history_episode_id':ids},'cpu',torch.bfloat16)['history_episode_id'],ids)


def test_one_history_switch_controls_reader_and_model(tmp_path):
    from openwam.train.goal_curated import load_config
    root=Path(__file__).resolve().parents[2]
    for enabled in [True,False]:
        p=tmp_path/'config.yaml'
        p.write_text(f'base_config: {root}/configs/goal/stage1_query_history_epoch_3fps.yaml\nhistory_enabled: {str(enabled).lower()}\n')
        cfg=load_config(p)
        assert cfg['model']['query_history']['enabled'] is enabled
        assert cfg['dataset']['history']['enabled'] is enabled
        assert cfg['dataset']['history']['jitter_fraction']==0.
        assert cfg['dataset']['history']['frame_dropout_prob']==.2


def test_stage2_episode_switch_flushes_both_caches():
    from test_query_stage2 import attach
    from test_query_history import history_expert, history
    from test_multiview_patch import multiview_batch
    from openwam.model.goal.runtime import prepare_goal_inputs
    a=attach();a.goal_expert=history_expert().freeze()
    obs,ctx,_,views=multiview_batch(1);h=history()
    def run(ep):
        return prepare_goal_inputs(a,None,['move'],ctx,head_valid=[True],seeds=[0],
            goal_observed_multiview=obs,goal_view_valid=views,**{**h,'history_episode_id':torch.tensor([ep])})
    first=run(1)['goal_latent'];oldkeys=set(a.goal_cache.entries)
    a.goal_expert._history_cache.entries[b'old']=torch.ones(1)
    second=run(2)['goal_latent']
    torch.testing.assert_close(first,second)
    assert not (oldkeys & set(a.goal_cache.entries))
    assert b'old' not in a.goal_expert._history_cache.entries


def test_dropout_extremes_reproducibility_and_fractional_fps():
    cfg=HistorySamplingConfig(enabled=True,frame_dropout_prob=1.)
    idx,age,valid=select_history_frames(900,29.97,0,cfg,np.random.default_rng(42),True)
    assert not valid.any() and (age==0).all()
    idx,age,valid=select_history_frames(900,29.97,0,cfg,np.random.default_rng(42),False)
    assert valid.all() and (age<=10.).all() and (age>0.).all()
    cfg=HistorySamplingConfig(enabled=True,frame_dropout_prob=.5)
    a=select_history_frames(900,30,0,cfg,np.random.default_rng(1),True)
    b=select_history_frames(900,30,0,cfg,np.random.default_rng(1),True)
    for x,y in zip(a,b): np.testing.assert_array_equal(x,y)
    for probability in [-.1,1.1,float('nan')]:
        with pytest.raises(ValueError,match='dropout'): HistorySamplingConfig(frame_dropout_prob=probability)


@pytest.mark.parametrize('preset',['sample','3000h'])
@pytest.mark.parametrize('enabled',[True,False])
def test_stage2_single_history_switch(preset,enabled):
    from hydra import compose,initialize_config_dir
    root=Path(__file__).resolve().parents[2]
    with initialize_config_dir(config_dir=str(root/'configs'),version_base=None):
        cfg=compose(config_name=f'goal/stage2_query_history_{preset}',overrides=[f'history_enabled={str(enabled).lower()}'])
    assert cfg.model.query_history.enabled is enabled
    assert cfg.goal.history.enabled is enabled
    assert cfg.dataloader.history.enabled is enabled
