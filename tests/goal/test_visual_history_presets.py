from pathlib import Path
import pytest
from hydra import initialize_config_dir,compose
from omegaconf import OmegaConf
from openwam.train.goal_curated import load_config
from openwam.train.stage12 import validate_stage1_stage2_contract
from openwam.model.goal.runtime import validate_goal_model_config


@pytest.mark.parametrize('mode',['joint','none'])
def test_visual_history_presets_pass_both_pipeline_and_runtime_contracts(mode):
    root=Path(__file__).resolve().parents[2]
    stage1=load_config(root/'configs/goal/stage1_visual_history_overfit.yaml')
    with initialize_config_dir(config_dir=str(root/'configs'),version_base=None):
        cfg=compose(config_name='goal/stage2_visual_history_'+mode)
    stage2=OmegaConf.to_container(cfg,resolve=True)
    validate_stage1_stage2_contract(stage1,stage2)
    validate_goal_model_config(stage2['goal'],stage2['model'])
    assert stage2['model']['architecture']['future_video_mode']==mode
