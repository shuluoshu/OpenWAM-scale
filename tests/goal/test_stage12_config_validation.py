"""Reject scalar launch errors before offline preparation or prepared admission."""
from pathlib import Path

import pytest
import torch

ROOT=Path(__file__).resolve().parents[2]


@pytest.fixture
def pipeline(tmp_path,monkeypatch):
    import openwam.train.stage12 as module
    from openwam.dataloader.curated_stage2 import CuratedStage2Dataset
    config=module.load_pipeline(ROOT/'configs/goal/stage12_visual_history_relative112_4x8gpu.yaml')
    config.update(output_dir=str(tmp_path/'run'),source_yaml=str(tmp_path/'unread-source.yaml'))
    monkeypatch.setattr(module,'load_pipeline',lambda path:config)
    monkeypatch.setattr(CuratedStage2Dataset,'from_config',lambda *a,**kw:pytest.fail('metadata read before scalar admission'))
    monkeypatch.setattr(torch.cuda,'_lazy_init',lambda:pytest.fail('CUDA initialized'))
    return module,config


@pytest.mark.parametrize('mode',['--prepare','--check'])
@pytest.mark.parametrize('override,reason',[
    ('evaluation.every_updates=0','every_updates'),
    ('evaluation.every_updates=true','every_updates'),
    ('evaluation.every_updates=500.1','every_updates'),
    ('model.architecture.action_dim=80','action_dim'),
    ('training.max_steps=0','max_steps'),
])
def test_invalid_stage2_scalar_fails_before_data_or_prepared_admission(pipeline,mode,override,reason):
    module,config=pipeline
    config['stage2_overrides']=[override]
    with pytest.raises(ValueError,match=reason):
        module.main(['--config','unused',mode],_runner=lambda *a,**kw:pytest.fail('Stage1 launched'))


@pytest.mark.parametrize('mode',['--prepare','--check'])
def test_pipeline_budget_is_validated_before_data_or_prepared_admission(pipeline,mode):
    module,config=pipeline
    config['stage2_max_steps']=0
    with pytest.raises(ValueError,match='max_steps'):
        module.main(['--config','unused',mode],_runner=lambda *a,**kw:pytest.fail('Stage1 launched'))


@pytest.mark.parametrize('mode',['--prepare','--check'])
@pytest.mark.parametrize('cadence,budget,extra,expected',[(10000,154000,[],154000),(2,0,['training.max_steps=2'],2)])
def test_effective_stage2_config_is_validated_with_final_override_precedence(pipeline,monkeypatch,mode,cadence,budget,extra,expected):
    from openwam.train.utils import stage2
    module,config=pipeline
    config.update(stage2_max_steps=budget,stage2_overrides=[f'evaluation.every_updates={cadence}',*extra])
    validate=stage2.validate_stage2_config
    class AdmissionComplete(Exception):pass
    def admitted(cfg):
        validate(cfg)
        assert cfg.evaluation.every_updates==cadence
        assert cfg.training.max_steps==expected
        raise AdmissionComplete
    monkeypatch.setattr(stage2,'validate_stage2_config',admitted)
    with pytest.raises(AdmissionComplete):
        module.main(['--config','unused',mode],_runner=lambda *a,**kw:pytest.fail('Stage1 launched'))
