"""Optional pinned runtime extraction stays quoted and fails before launch."""
import copy
import hashlib
import os
from pathlib import Path
import subprocess
import tarfile
import pytest
from scripts.pai.prepare_stage12_dlc import build_request


def site(tmp_path):
    return dict(workspace_id='workspace',quota_id='quota',image='image',cpu=144,
        memory='1500Gi',shared_memory='500Gi',python='/usr/bin/python',
        code_dir=str(tmp_path/'code'),commit='a'*40,pod_count=4,gpus_per_pod=8,
        data_sources=[{'DataSourceId':'source'}],user_vpc={'VpcId':'vpc'},display_name='test')


@pytest.mark.parametrize('case',['not_mapping','missing','bad_sha','relative_path','relative_prefix','python_mismatch'])
def test_runtime_archive_rejects_bad_manifest(tmp_path,case):
    cfg=site(tmp_path)
    cfg['runtime_archive']=dict(path='/runtime.tar',sha256='a'*64,prefix='/tmp/runtime/env')
    cfg['python']='/tmp/runtime/env/bin/python'
    if case=='not_mapping':cfg['runtime_archive']='bad'
    elif case=='missing':del cfg['runtime_archive']['sha256']
    elif case=='bad_sha':cfg['runtime_archive']['sha256']='not pinned'
    elif case=='relative_path':cfg['runtime_archive']['path']='runtime.tar'
    elif case=='relative_prefix':cfg['runtime_archive']['prefix']='runtime/env'
    else:cfg['python']='/some/other/python'
    with pytest.raises(ValueError,match='runtime_archive'):
        build_request(cfg)


@pytest.mark.parametrize('valid',[True,False])
def test_quoted_runtime_archive_extracts_only_after_checksum(tmp_path,valid):
    cfg=site(tmp_path)
    archive=tmp_path/"runtime 'quoted' $(false).tar"
    prefix=tmp_path/"runtime 'prefix' $(false)"/'env'
    content=tmp_path/'payload';content.write_text('verified runtime')
    with tarfile.open(archive,'w') as stream:stream.add(content,arcname='payload')
    cfg['runtime_archive']=dict(path=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest() if valid else '0'*64,prefix=str(prefix))
    cfg['python']=str(prefix/'bin/python')
    entry=Path(cfg['code_dir'])/'scripts/pai/stage12_entry.sh';entry.parent.mkdir(parents=True)
    result_file=tmp_path/'entry-ran'
    entry.write_text('#!/bin/bash\nprintf launched > '+__import__('shlex').quote(str(result_file))+'\n')
    request=build_request(cfg)
    result=subprocess.run(['bash','-c',request['UserCommand']],capture_output=True,text=True)
    assert (result.returncode==0) is valid
    assert result_file.exists() is valid
    assert (prefix/'payload').exists() is valid
    if valid:assert (prefix/'payload').read_text()=='verified runtime'


def test_without_archive_keeps_plain_entry_command(tmp_path):
    import shlex
    cfg=site(tmp_path)
    assert build_request(cfg)['UserCommand']==shlex.join(['bash',str(Path(cfg['code_dir'])/'scripts/pai/stage12_entry.sh'),'train'])


def test_submit_preflight_uses_host_python_before_runtime_exists(tmp_path, monkeypatch):
    import json
    import shlex
    import sys
    import scripts.pai.prepare_stage12_dlc as renderer
    import scripts.pai.preparation_gate as gate
    cfg=site(tmp_path)
    cfg.update(runtime_archive=dict(path='/archive.tar',sha256='a'*64,prefix='/nonexistent/dlc/runtime'),
               python='/nonexistent/dlc/runtime/bin/python',prepared_marker='/prepared.json')
    manifest=tmp_path/'site.json';manifest.write_text(json.dumps(cfg))
    validator=tmp_path/'validator.py'
    validator.write_text('def validate_request(request): return []\ndef build_command(request, **kwargs): return ["echo", "submit"]\n')
    output=tmp_path/'rendered'
    monkeypatch.setattr(gate,'require_pipeline_prepared',lambda *a:None)
    monkeypatch.setattr(sys,'argv',['renderer','--site',str(manifest),'--out',str(output),'--validator',str(validator)])
    renderer.main()
    preflight=(output/'submit.sh').read_text().splitlines()[2]
    assert shlex.split(preflight)[0]==sys.executable
    assert cfg['python'] not in preflight
