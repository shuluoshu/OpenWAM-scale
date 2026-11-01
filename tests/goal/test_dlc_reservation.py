"""Reservation and explicit per-pod training have separate lifecycle semantics."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

import pytest
import scripts.pai.prepare_stage12_dlc as renderer


def site(tmp_path):
    return dict(workspace_id='workspace', quota_id='quota', image='image', cpu=144,
        memory='1500Gi', shared_memory='64Gi', python=sys.executable,
        code_dir=str(tmp_path / "code 'quoted' $(false)"), commit='a' * 40,
        pod_count=4, gpus_per_pod=8, data_sources=[{'DataSourceId': 'source'}],
        user_vpc={'VpcId': 'vpc'}, display_name='test',
        prepared_marker=str(tmp_path / "prepared 'marker'.json"),
        pipeline_config="configs/pipeline 'quoted'.yaml")


def render(tmp_path, monkeypatch, cfg):
    manifest=tmp_path/'site.json';manifest.write_text(json.dumps(cfg))
    validator=tmp_path/'validator.py'
    validator.write_text('def validate_request(request): return []\ndef build_command(request, **kwargs): return ["echo", "submit"]\n')
    out=tmp_path/'rendered'
    monkeypatch.setattr(sys,'argv',['renderer','--site',str(manifest),'--out',str(out),'--validator',str(validator)])
    renderer.main()
    return out


def test_default_request_only_reserves_and_pins_launch_environment(tmp_path):
    cfg=site(tmp_path)
    cfg['runtime_archive']=dict(path='/missing/archive.tar', sha256='b'*64, prefix='/missing/runtime')
    cfg['python']='/missing/runtime/bin/python'
    request=renderer.build_request(cfg)
    assert request['UserCommand']=='sleep 36000000'
    assert request['Envs']['OPENWAM_EXPECTED_COMMIT']==cfg['commit']
    assert request['Envs']['STAGE12_PREPARED_MARKER']==cfg['prepared_marker']
    assert request['Envs']['STAGE12_CONFIG']==cfg['pipeline_config']


@pytest.mark.parametrize('mode',['auto', '', None, False])
def test_unknown_startup_mode_is_rejected(tmp_path, mode):
    with pytest.raises(ValueError,match='startup_mode'):
        renderer.build_request(dict(site(tmp_path),startup_mode=mode))


def test_hold_render_and_submission_do_not_require_completed_preparation(tmp_path,monkeypatch):
    import scripts.pai.preparation_gate as gate
    monkeypatch.setattr(gate,'require_pipeline_prepared',lambda *a:pytest.fail('reservation must not check training readiness'))
    cfg=site(tmp_path)
    out=render(tmp_path,monkeypatch,cfg)
    assert json.loads((out/'request.json').read_text())['UserCommand']=='sleep 36000000'
    assert 'preparation_gate' not in (out/'submit.sh').read_text()
    launcher=(out/'launch_train.sh').read_text()
    assert 'stage12_entry.sh' in launcher
    assert cfg['commit'] in launcher
    assert shlex.quote(cfg['prepared_marker']) in launcher
    assert not Path(cfg['prepared_marker']).exists()
    instructions=(out/'LAUNCH.md').read_text()
    assert 'each of the 4' in instructions
    assert 'nohup bash' in instructions
    assert 'launch_train.sh' in instructions
    assert (out/'launch_train.sh').stat().st_mode & 0o111


def fake_entry(cfg,tmp_path):
    Path(cfg['prepared_marker']).write_text('{}')
    entry=Path(cfg['code_dir'])/'scripts/pai/stage12_entry.sh';entry.parent.mkdir(parents=True)
    runs=tmp_path/'runs';ready=tmp_path/'ready';release=tmp_path/'release'
    entry.write_text('''#!/usr/bin/env bash
set -eu
[[ "$1" == train ]]
[[ "$OPENWAM_EXPECTED_COMMIT" == '''+shlex.quote(cfg['commit'])+''' ]]
[[ "$STAGE12_PREPARED_MARKER" == '''+shlex.quote(cfg['prepared_marker'])+''' ]]
printf 'run\\n' >> '''+shlex.quote(str(runs))+'''
touch '''+shlex.quote(str(ready))+'''
while [[ ! -f '''+shlex.quote(str(release))+''' ]]; do sleep 0.02; done
echo 'training output'
exit "${FAKE_TRAIN_EXIT:-0}"
''')
    return runs,ready,release


def test_launcher_holds_duplicate_guard_through_child_and_releases_on_failure(tmp_path,monkeypatch):
    cfg=site(tmp_path);out=render(tmp_path,monkeypatch,cfg)
    runs,ready,release=fake_entry(cfg,tmp_path)
    env=dict(os.environ, FAKE_TRAIN_EXIT='7')
    first=subprocess.Popen(['bash',str(out/'launch_train.sh')],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=env)
    try:
        deadline=time.monotonic()+5
        while not ready.exists() and time.monotonic()<deadline:
            if first.poll() is not None:break
            time.sleep(.02)
        assert ready.exists(),first.communicate(timeout=2)
        duplicate=subprocess.run(['bash',str(out/'launch_train.sh')],capture_output=True,text=True,timeout=2)
        assert duplicate.returncode==75
        assert 'already running' in duplicate.stderr
        assert runs.read_text()=='run\n'
    finally:
        release.touch()
        output=first.communicate(timeout=5)[0]
    assert first.returncode==7
    assert 'training output' in output
    assert 'exit status 7' in output
    retry=subprocess.run(['bash',str(out/'launch_train.sh')],capture_output=True,text=True,timeout=5)
    assert retry.returncode==0,retry.stdout+retry.stderr
    assert runs.read_text()=='run\nrun\n'
    logs=list((Path(cfg['code_dir'])/'outputs/dlc_launch').glob('*.log'))
    assert logs and any('exit status 7' in p.read_text() for p in logs)


@pytest.mark.parametrize('valid',[True,False])
def test_explicit_hold_launcher_checks_runtime_hash_before_training(tmp_path,monkeypatch,valid):
    import hashlib
    import tarfile
    cfg=site(tmp_path)
    archive=tmp_path/"runtime 'archive' $(false).tar"
    payload=tmp_path/'payload';payload.write_text('verified')
    with tarfile.open(archive,'w') as stream:stream.add(payload,arcname='payload')
    prefix=tmp_path/"runtime 'prefix' $(false)"
    cfg['python']=str(prefix/'bin/python')
    cfg['runtime_archive']=dict(path=str(archive),prefix=str(prefix),sha256=hashlib.sha256(archive.read_bytes()).hexdigest() if valid else '0'*64)
    out=render(tmp_path,monkeypatch,cfg)
    runs,ready,release=fake_entry(cfg,tmp_path);release.touch()
    result=subprocess.run(['bash',str(out/'launch_train.sh')],capture_output=True,text=True,timeout=5)
    assert (result.returncode==0) is valid,result.stdout+result.stderr
    assert runs.exists() is valid
    assert (prefix/'payload').exists() is valid
    if not valid:assert 'Runtime archive SHA256 mismatch' in result.stdout+result.stderr


@pytest.mark.parametrize('required',[True,0,-1,'bad',float('inf')])
def test_gpu_memory_requirement_is_a_positive_number(tmp_path,required):
    with pytest.raises(ValueError,match='expected_min_gpu_memory_gib'):
        renderer.build_request(dict(site(tmp_path),expected_min_gpu_memory_gib=required))


@pytest.mark.parametrize('memory,count,success',[(81920,8,True),(40960,8,False),(81920,7,False)])
def test_gpu_inventory_checks_memory_before_runtime_bootstrap(tmp_path,monkeypatch,memory,count,success):
    cfg=site(tmp_path);cfg['expected_min_gpu_memory_gib']=70
    Path(cfg['prepared_marker']).write_text('{}')
    cfg['runtime_archive']=dict(path='/missing/runtime.tar',sha256='b'*64,prefix='/missing/runtime')
    cfg['python']='/missing/runtime/bin/python'
    bin_dir=tmp_path/'bin';bin_dir.mkdir()
    nvidia=bin_dir/'nvidia-smi';nvidia.write_text('#!/bin/sh\n'+''.join(f"echo 'Fixture GPU, {memory}'\n" for _ in range(count)));nvidia.chmod(0o755)
    out=render(tmp_path,monkeypatch,cfg)
    result=subprocess.run(['bash',str(out/'launch_train.sh')],capture_output=True,text=True,timeout=5,env=dict(os.environ,PATH=str(bin_dir)+':'+os.environ['PATH']))
    output=result.stdout+result.stderr
    assert 'Fixture GPU' in output
    if success:
        assert '/missing/runtime.tar' in output
        assert 'GPU memory requirement failed' not in output
    else:
        assert 'GPU memory requirement failed' in output
        assert '/missing/runtime.tar' not in output
    assert result.returncode!=0


def test_manual_launch_requires_prepared_marker_before_runtime_extraction(tmp_path,monkeypatch):
    cfg=site(tmp_path)
    cfg['runtime_archive']=dict(path='/missing/runtime.tar',sha256='b'*64,prefix='/missing/runtime')
    cfg['python']='/missing/runtime/bin/python'
    out=render(tmp_path,monkeypatch,cfg)
    result=subprocess.run(['bash',str(out/'launch_train.sh')],capture_output=True,text=True,timeout=5)
    output=result.stdout+result.stderr
    assert result.returncode!=0
    assert 'Prepared marker missing' in output
    assert '/missing/runtime.tar' not in output


def test_legacy_auto_start_enforces_gpu_floor_before_archive_or_training(tmp_path):
    cfg=site(tmp_path)
    cfg.update(startup_mode='train',expected_min_gpu_memory_gib=70,
        runtime_archive=dict(path='/missing/runtime.tar',sha256='b'*64,prefix='/missing/runtime'),
        python='/missing/runtime/bin/python')
    runs,_,_=fake_entry(cfg,tmp_path)
    bin_dir=tmp_path/'bin';bin_dir.mkdir()
    nvidia=bin_dir/'nvidia-smi'
    nvidia.write_text('#!/bin/sh\n'+"echo 'Undersized GPU, 40960'\n"*8)
    nvidia.chmod(0o755)
    request=renderer.build_request(cfg)
    result=subprocess.run(['bash','-c',request['UserCommand']],capture_output=True,text=True,timeout=5,
        env=dict(os.environ,PATH=str(bin_dir)+':'+os.environ['PATH']))
    output=result.stdout+result.stderr
    assert result.returncode!=0
    assert 'Undersized GPU' in output
    assert 'GPU memory requirement failed' in output
    assert '/missing/runtime.tar' not in output
    assert not runs.exists()
