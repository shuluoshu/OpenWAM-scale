"""The documented pipeline uses bounded parallel offline preparation by default."""
import pytest

@pytest.mark.parametrize('flags,expected',[
    ([],dict(workers=32,max_pending=64,commit_every=256)),
    (['--index-workers','8','--index-max-pending','24','--index-commit-every','64'],
     dict(workers=8,max_pending=24,commit_every=64)),
])
def test_prepare_cli_passes_parallel_defaults_and_overrides(tmp_path,monkeypatch,flags,expected):
    from scripts import train_goal_epoch
    import openwam.train.goal_curated as curated
    import openwam.dataloader.curated_goal as source
    import openwam.dataloader.goal_epoch as epoch
    monkeypatch.setenv('WORLD_SIZE','1');monkeypatch.setenv('RANK','0')
    monkeypatch.setattr(curated,'load_config',lambda *a,**kw:dict(
        model={'goal_representation':'query_latent'},dataset={},sample_fps=3.))
    monkeypatch.setattr(source,'CuratedGoalDataset',lambda **kw:object())
    seen={}
    def prepare(dataset,index,**kwargs):seen.update(kwargs);return dict(complete=True)
    monkeypatch.setattr(epoch,'build_epoch_index',prepare)
    assert train_goal_epoch.main(['--prepare','--config','unused','--index',str(tmp_path/'index.db'),
        '--output-dir',str(tmp_path/'out'),*flags])==0
    assert seen==dict(sample_fps=3.,**expected)
