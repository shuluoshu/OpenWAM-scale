from pathlib import Path
import pytest
from scripts import train_goal_curated as module
CODE=Path(__file__).resolve().parents[2]

def test_visual_history_rejects_legacy_patch_warm_start(tmp_path,capsys):
    with pytest.raises(SystemExit) as stopped:
        module.main(['--smoke','--config',str(CODE/'configs/goal/stage1_visual_history_overfit.yaml'),
            '--init-goal-checkpoint',str(tmp_path/'absent-patch.safetensors'),'--output-dir',str(tmp_path/'out')])
    assert stopped.value.code==2
    assert 'visual_history' in capsys.readouterr().err
