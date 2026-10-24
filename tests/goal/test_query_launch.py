"""Launch contracts for old/new models and both data selections."""
from pathlib import Path
import subprocess
import sys

import pytest
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('data', ['sample', '3000h'])
def test_both_stage1_and_stage2_presets(data):
    from openwam.train.goal_curated import load_config
    from openwam.train.utils.stage2 import validate_stage2_config
    for kind, model in [('multiview', 'patch_denoise'), ('query', 'query_latent')]:
        stage1 = load_config(ROOT / f'configs/goal/stage1_{kind}_{data}.yaml')
        assert stage1['model']['goal_representation'] == model
        assert stage1['dataset']['observation_mode'] == 'multiview'
        suffix = 'smoke.yaml' if data == 'sample' else '3000h.yaml'
        assert stage1['dataset']['source_yaml'].endswith(suffix)
        with initialize_config_dir(config_dir=str(ROOT/'configs'), version_base=None):
            stage2 = compose(config_name=f'goal/stage2_{kind}_{data}')
        validate_stage2_config(stage2)
        assert stage2.model.goal_representation == model
        assert stage2.dataloader.source_yaml.endswith(suffix)


def test_curated_cli_rejects_unknown_model(tmp_path):
    cmd = [sys.executable, str(ROOT/'scripts/train_goal_curated.py'), '--smoke',
           '--output-dir', str(tmp_path/'run'), '--goal-representation', 'oops']
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode != 0 and 'invalid choice' in result.stderr


def test_unknown_yaml_representation_rejected(tmp_path):
    from openwam.train.goal_curated import load_config
    path = tmp_path/'bad.yaml'
    path.write_text('model:\n  goal_representation: typo\n')
    with pytest.raises(ValueError, match='representation'):
        load_config(path)
