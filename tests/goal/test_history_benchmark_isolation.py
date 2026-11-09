"""The in-process performance reference must use its pinned codec classes."""
import importlib.util
from pathlib import Path
import subprocess

import torch


def test_paired_baseline_uses_its_own_absolute_codec_dependencies(tmp_path):
    from test_expert import TINY_DIT, TinyVAE
    from openwam.model.video_backbone.wan.models.dit import WanModel
    repo=Path(__file__).resolve().parents[2]
    destination=tmp_path/'openwam/model/goal'
    destination.mkdir(parents=True)
    for filename in ['codec.py','history.py','patch_expert.py','visual_history.py','observation.py']:
        content=subprocess.check_output(['git','show','10dc74e:openwam/model/goal/'+filename],cwd=repo)
        (destination/filename).write_bytes(content)
    spec=importlib.util.spec_from_file_location('paired_benchmark',repo/'scripts/experiments/benchmark_history_vae_paired.py')
    driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    old=driver.load_baseline_goal(tmp_path)
    codec=old.codec.GoalCodec(TinyVAE());codec.enable_cache(max_bytes=100000)
    model=old.visual.VisualHistoryGoalExpert(WanModel(**TINY_DIT),codec,dit_config=TINY_DIT)
    assert type(model.observation_codec) is old.codec.GoalCodec
    assert model.observation_codec.cache is codec.cache
    output=model.observation_codec.encode(torch.zeros(1,3,1,384,320))
    assert output.shape==(1,48,1,24,20)
