"""GPU submissions must consume complete, shared offline data artifacts."""
import hashlib
import json
from pathlib import Path
import sqlite3
import pytest


def _prepared(root):
    source=root/'source.yaml';source.write_text('selection: fixture\n')
    index=root/'population.sqlite'
    with sqlite3.connect(index) as db:
        db.execute('CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT)')
        db.executemany('INSERT INTO metadata VALUES (?,?)', [
            ('complete','true'),('total_samples','12'),('sample_fps','3.0'),
            ('source_yaml_sha256',json.dumps(hashlib.sha256(source.read_bytes()).hexdigest()))])
    raw=json.dumps(dict(entries={},sources={},contract=dict(
        selection_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))).encode()
    stats=root/('registry-'+hashlib.sha256(raw).hexdigest()+'.json');stats.write_bytes(raw)
    marker=root/'prepared.json'
    marker.write_text(json.dumps(dict(source_yaml=str(source),index=str(index),stats_registry=str(stats))))
    return marker,index,source,stats


def test_offline_gate_rejects_incomplete_index_and_changed_source(tmp_path):
    from scripts.pai.preparation_gate import require_prepared
    marker,index,source,stats=_prepared(tmp_path)
    assert require_prepared(marker)['index']==str(index)
    with sqlite3.connect(index) as db:db.execute("UPDATE metadata SET value='false' WHERE key='complete'")
    with pytest.raises(ValueError,match='incomplete'):require_prepared(marker)
    with sqlite3.connect(index) as db:db.execute("UPDATE metadata SET value='true' WHERE key='complete'")
    source.write_text('changed: true\n')
    with pytest.raises(ValueError,match='source'):require_prepared(marker)


def test_offline_gate_missing_or_tampered_artifacts(tmp_path):
    from scripts.pai.preparation_gate import require_prepared
    with pytest.raises(FileNotFoundError):require_prepared(tmp_path/'prepared.json')
    marker,index,source,stats=_prepared(tmp_path)
    stats.write_text('{}')
    with pytest.raises(ValueError,match='registry'):require_prepared(marker)


def test_offline_gate_rejects_non_shared_marker_before_read():
    from scripts.pai.preparation_gate import require_prepared
    with pytest.raises(ValueError,match='CPFS'):require_prepared('/root/train/prepared.json')


def test_offline_gate_rejects_wrong_rate_and_registry_selection(tmp_path):
    from scripts.pai.preparation_gate import require_prepared
    marker,index,source,stats=_prepared(tmp_path)
    with pytest.raises(ValueError,match='sample_fps'):require_prepared(marker,sample_fps=15.)
    payload=json.loads(stats.read_text());payload['contract']['selection_sha256']='another-source'
    raw=json.dumps(payload).encode()
    other=tmp_path/('registry-'+hashlib.sha256(raw).hexdigest()+'.json');other.write_bytes(raw)
    data=json.loads(marker.read_text());data['stats_registry']=str(other);marker.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='registry.*selection'):require_prepared(marker,sample_fps=3.)


def test_pipeline_gate_rejects_marker_rate_and_registry_override(tmp_path,monkeypatch):
    from scripts.pai.preparation_gate import require_pipeline_prepared
    import openwam.train.stage12 as pipeline
    import openwam.train.goal_curated as goal
    marker,index,source,stats=_prepared(tmp_path)
    config=dict(output_dir=str(tmp_path),stage1_config='fixture',source_yaml=str(source),stats_registry=None)
    stage1=dict(sample_fps=3.)
    monkeypatch.setattr(pipeline,'load_pipeline',lambda _:config)
    monkeypatch.setattr(goal,'load_config',lambda *a,**k:stage1)
    assert require_pipeline_prepared(marker,'fixture')['stats_registry']==str(stats)
    config['stats_registry']=str(tmp_path/'other.json')
    with pytest.raises(ValueError,match='registry override'):require_pipeline_prepared(marker,'fixture')
    config['stats_registry']=None;stage1['sample_fps']=15.
    with pytest.raises(ValueError,match='sample_fps'):require_pipeline_prepared(marker,'fixture')
    with pytest.raises(ValueError,match='marker'):require_pipeline_prepared(tmp_path/'other.json','fixture')


def test_request_priority_is_explicit_and_checked():
    from scripts.pai.prepare_stage12_dlc import build_request
    site=dict(workspace_id='1140257',quota_id='quota1r415ahnha3',image='image',cpu=144,
              memory='1600Gi',shared_memory='64Gi',python='/cpfs/python',code_dir='/cpfs/code',
              commit='abc',pod_count=4,gpus_per_pod=8,data_sources=[{'Uri':'cpfs://test'}],
              user_vpc={'VpcId':'vpc'},display_name='test',priority=9)
    assert build_request(site)['Priority']==9
    for priority in (0,10,True):
        with pytest.raises(ValueError,match='priority'):build_request(dict(site,priority=priority))
