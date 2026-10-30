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


@pytest.mark.parametrize('data',['sample','3000h','epoch','epoch_3fps','epoch_sample'])
def test_c_stage1_presets_select_prefix_with_matching_history_and_lr(data):
    from openwam.train.goal_curated import load_config
    cfg=load_config(ROOT/f'configs/goal/stage1_query_prefix_{data}.yaml')
    assert cfg['model']['goal_representation']=='query_latent'
    assert cfg['model']['query_conditioning']=='prefix_self_attention'
    assert cfg['condition_learning_rate']==1e-4
    assert cfg['learning_rate']==1e-5
    assert cfg['dataset']['history']['num_frames']==10
    assert cfg['dataset']['history']['mean_interval_s']==1.
    assert cfg['dataset']['history']['frame_dropout_prob']==.2
    assert cfg['model']['query_history']['enabled']
    assert cfg['dataset']['cache_dir'].startswith('/mnt/data/cpfs/')


@pytest.mark.parametrize('data',['sample','posttrain_3fps_4x8gpu'])
def test_c_pipeline_stage2_loads_matching_encoder_contract(data):
    from openwam.train.stage12 import load_pipeline
    from openwam.train.goal_curated import load_config
    from omegaconf import OmegaConf
    pipeline=load_pipeline(ROOT/f'configs/goal/stage12_query_prefix_{data}.yaml')
    stage1=load_config(pipeline['stage1_config'],source_yaml=pipeline['source_yaml'])
    with initialize_config_dir(config_dir=str(ROOT/'configs'),version_base=None):
        stage2=OmegaConf.to_container(compose(config_name=pipeline['stage2_config']),resolve=True)
    assert stage1['sample_fps']==3.
    assert stage1['model']['query_goal']==stage2['model']['query_goal']
    assert stage1['model']['query_history']==stage2['model']['query_history']
    assert stage2['goal']['conditioning']=='query_memory'
    assert pipeline['output_dir'].startswith('/mnt/data/cpfs/')
    assert pipeline['world_size']==(1 if data=='sample' else 32)
