"""The orchestrator must not promote failed or partial Stage1 work."""
import json
from pathlib import Path
import subprocess
import pytest


def test_stage2_requires_success_and_complete_verified_epoch(tmp_path):
    from openwam.train.stage12 import execute_stages
    calls=[]
    def failed(command):calls.append(command);raise subprocess.CalledProcessError(1,command)
    with pytest.raises(subprocess.CalledProcessError):
        execute_stages(tmp_path,{},['stage1'],['stage2'],runner=failed)
    assert calls==[['stage1']]
    calls.clear()
    with pytest.raises(FileNotFoundError):
        execute_stages(tmp_path,{},['stage1'],['stage2'],runner=lambda c:calls.append(c),resume=True)
    assert calls==[['stage1']]


def test_completed_stage1_resume_is_verified_and_preserved_on_stage2_error(tmp_path,monkeypatch):
    import openwam.train.stage12 as module
    calls=[];proof=dict(checkpoint='/immutable/query.safetensors')
    monkeypatch.setattr(module,'verify_completion',lambda *a:proof)
    (tmp_path/'stage1').mkdir();(tmp_path/'stage1/epoch_complete.json').write_text('{}')
    def run(command):calls.append(command);raise subprocess.CalledProcessError(9,command)
    with pytest.raises(subprocess.CalledProcessError):
        module.execute_stages(tmp_path,{},['stage1'],['stage2','{goal_checkpoint}'],runner=run,resume=True)
    assert calls==[['stage2','/immutable/query.safetensors']]
    assert (tmp_path/'stage1/epoch_complete.json').exists()
    assert json.loads((tmp_path/'pipeline_state.json').read_text())['phase']=='stage2_failed'


def test_successful_handoff_only_passes_verified_checkpoint(tmp_path,monkeypatch):
    import openwam.train.stage12 as module
    calls=[]
    monkeypatch.setattr(module,'verify_completion',lambda *a:dict(checkpoint='/verified/query.safetensors'))
    module.execute_stages(tmp_path,{},['stage1'],['stage2','{goal_checkpoint}'],runner=lambda c:calls.append(c))
    assert calls==[['stage1'],['stage2','/verified/query.safetensors']]
    assert json.loads((tmp_path/'pipeline_state.json').read_text())['phase']=='complete'


def test_completed_pipeline_rejects_changed_stage2_contract(tmp_path,monkeypatch):
    import openwam.train.stage12 as module
    monkeypatch.setattr(module,'verify_completion',lambda *a:dict(checkpoint='/verified/query.safetensors'))
    module.execute_stages(tmp_path,{},['stage1'],['stage2'],runner=lambda c:None,pipeline_identity='original')
    with pytest.raises(ValueError,match='pipeline identity'):
        module.execute_stages(tmp_path,{},['stage1'],['stage2'],runner=lambda c:pytest.fail('must not run'),
            resume=True,pipeline_identity='changed')
