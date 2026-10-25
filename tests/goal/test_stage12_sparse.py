"""Sparse Stage1 sampling and single-node 32 rank launch contract."""
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[2]


def test_3fps_cli_prepare_uses_config_without_loading_model(tmp_path,monkeypatch):
    from scripts import train_goal_epoch
    import openwam.train.goal_curated as curated
    import openwam.dataloader.curated_goal as source
    import openwam.dataloader.goal_epoch as epoch
    monkeypatch.delenv('WORLD_SIZE',raising=False)
    monkeypatch.setattr(curated,'load_config',lambda *a,**kw:dict(model={'goal_representation':'query_latent'},dataset={},sample_fps=3.))
    sentinel=object();monkeypatch.setattr(source,'CuratedGoalDataset',lambda **kw:sentinel)
    seen=[]
    def build(dataset,path,*,sample_fps,workers,max_pending,commit_every):
        assert (workers,max_pending,commit_every)==(32,64,256)
        assert dataset is sentinel;seen.append(sample_fps);return dict(total_samples=7,sample_fps=sample_fps)
    monkeypatch.setattr(epoch,'build_epoch_index',build)
    assert train_goal_epoch.main(['--prepare','--config','unused','--index',str(tmp_path/'index.db'),'--output-dir',str(tmp_path/'out')])==0
    assert seen==[3.]


def test_32rank_tail_retains_exact_population():
    from openwam.train.goal_epoch import rank_sample
    seen=[]
    for cursor in (0,32,64):
        batch=[rank_sample(cursor,70,rank,32) for rank in range(32)]
        seen.extend(i for i,w in batch if w)
        assert sum(w for _,w in batch)==pytest.approx(32.)
    assert seen==list(range(70))


def test_single_node_32_launch_and_insufficient_device_rejection():
    from openwam.train.stage12 import single_node_torchrun
    command=single_node_torchrun(sys.executable,32,visible_devices=32)
    assert command==[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=32']
    with pytest.raises(ValueError,match='visible'):
        single_node_torchrun(sys.executable,32,visible_devices=8)
    for invalid in (True,0,3.5):
        with pytest.raises(ValueError):single_node_torchrun(sys.executable,invalid)


def test_old_and_new_presets_keep_source_and_history_but_separate_outputs():
    from openwam.train.stage12 import load_pipeline
    from openwam.train.goal_curated import load_config
    from openwam.dataloader.goal_epoch import epoch_sample_fps
    old=load_pipeline(ROOT/'configs/goal/stage12_query_history_posttrain.yaml')
    new=load_pipeline(ROOT/'configs/goal/stage12_query_history_posttrain_3fps_32gpu.yaml')
    assert old['world_size']==8 and new['world_size']==32
    assert old['source_yaml']==new['source_yaml']
    assert old['output_dir']!=new['output_dir']
    before=load_config(old['stage1_config']);after=load_config(new['stage1_config'])
    assert epoch_sample_fps(before)==15 and epoch_sample_fps(after)==3
    assert before['dataset']['history']==after['dataset']['history']
    assert before['model']==after['model']
    assert old['stage2_config']==new['stage2_config']
