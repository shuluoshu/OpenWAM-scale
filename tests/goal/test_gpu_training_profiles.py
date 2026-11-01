"""Production profiles keep a common sample budget and validated hold launch."""
from pathlib import Path
import json
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest

ROOT=Path(__file__).resolve().parents[2]

@pytest.mark.parametrize('gpu,mode',[('80g','joint'),('140g','none')])
def test_gpu_profile_composes_masks_budgets_and_reservation(gpu,mode):
    from openwam.train.stage12 import load_pipeline,validate_stage1_stage2_contract
    from openwam.train.goal_curated import load_config
    from openwam.train.goal_optimizer import optimizer_config
    from openwam.train.utils.stage2 import validate_stage2_config
    from scripts.pai.prepare_stage12_dlc import build_request
    p=load_pipeline(ROOT/'configs/goal'/f'stage12_gpu_zero1_{gpu}_{mode}_4x8gpu.yaml')
    s1=load_config(p['stage1_config'],source_yaml=p['source_yaml'])
    with initialize_config_dir(config_dir=str(ROOT/'configs'),version_base=None):
        s2=compose(config_name=p['stage2_config'],overrides=[f'training.max_steps={p["stage2_max_steps"]}',*p['stage2_overrides']])
    validate_stage2_config(s2)
    validate_stage1_stage2_contract(s1,OmegaConf.to_container(s2,resolve=True))
    assert optimizer_config(s1)['sharding']=='zero1'
    assert s1['dataset_pin_memory'] and s1['batch_size']*p['world_size']==1024
    assert s1['dataset_num_workers']*s1['prefetch_factor']==s1['batch_size']
    assert s2.training.batch_size*p['world_size']==256
    assert s2.training.offload_optimizer_device=='none'
    assert s2.training.adam_foreach is False
    assert s2.dataloader.future_video_mode==mode
    assert s1['convergence']['min_updates']==50000 and s1['convergence']['eval_every']==10000
    assert s2.model.architecture.action_dim==112 and s1['sample_fps']==3
    assert 'stage12_gpu_optimizer_' in p['output_dir']
    site=json.loads((ROOT/'configs/pai'/f'stage12_gpu_zero1_{gpu}.site.json').read_text())
    site['commit']='a'*40
    req=build_request(site)
    assert req['UserCommand']=='sleep 36000000' and req['Priority']==9
    assert req['Envs']['STAGE12_CONFIG'].endswith(f'{gpu}_{mode}_4x8gpu.yaml')
