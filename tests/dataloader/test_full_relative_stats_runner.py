import json
from pathlib import Path
import numpy as np
import pytest
from tests.dataloader.test_curated_stage2 import make_native_source
from openwam.dataloader.curated_stage2 import CuratedStage2Dataset
from openwam.dataloader.native_stage2_stats import audit_selection


def runner():
    import scripts.data.prepare_full_relative_stats as module
    return module


def test_exhaustive_population_matches_native(tmp_path):
    r=runner(); ds=CuratedStage2Dataset(make_native_source(tmp_path)); repo=ds.repos[0]
    m=ds._episode(repo,0); action=r.StreamingStats(); state=r.StreamingStats()
    counts=r.accumulate_episode(ds,m,repo['ranges'],action,state,batch_size=3)
    expected_action=r.StreamingStats(); expected_state=r.StreamingStats(); anchors=0
    for level in ('task','subtask'):
        for option in ds._options(m,repo['ranges'],level):
            for anchor in range(option['start'],option['end_exclusive']):
                _,a,mask,s,sm,_=ds.raw_window(m,option,anchor)
                expected_action.update(a,mask); expected_state.update(s,sm); anchors+=1
    assert counts['anchors']==anchors
    for actual,expected in ((action,expected_action),(state,expected_state)):
        np.testing.assert_array_equal(actual.count,expected.count)
        np.testing.assert_array_equal(actual.hist,expected.hist)
        np.testing.assert_allclose(actual.mean,expected.mean,atol=1e-12)


def test_checkpoint_atomic_resume_no_double_count_and_mutation(tmp_path):
    r=runner(); source=tmp_path/'source'; source.write_text('original')
    files=r.snapshot_files([source]); out=tmp_path/'shard'; out.mkdir()
    a=r.StreamingStats(); s=r.StreamingStats(); x=np.ones((3,52)); a.update(x,x.astype(bool))
    identity={'population_policy':r.POPULATION_POLICY,'test':'x'}
    r.save_checkpoint(out,identity,[7],files,a,s,{'anchors':3},done=False)
    # Uncommitted in-memory work and stray file must not count on restart.
    a.update(x,x.astype(bool)); (out/'uncommitted.npz').write_bytes(b'crash')
    loaded=r.load_checkpoint(out,identity)
    assert loaded['completed_episodes']==[7]
    np.testing.assert_array_equal(loaded['action'].count,np.full(52,3))
    with pytest.raises(ValueError,match='identity'):
        r.load_checkpoint(out,{'test':'different'})
    source.write_text('mutated source')
    with pytest.raises(ValueError,match='source changed'):
        r.load_checkpoint(out,identity)


def test_publication_requires_all_selected_and_full_policy(tmp_path):
    r=runner(); ds=CuratedStage2Dataset(make_native_source(tmp_path)); audit=audit_selection(ds)
    with pytest.raises(ValueError,match='incomplete'):
        r.require_complete(ds,audit,{})
    name=ds.repos[0]['repo']; episodes=sorted({v[0] for v in ds.repos[0]['ranges']})
    done={name:{'done':True,'completed_episodes':episodes,'identity':{'population_policy':'sampled'}}}
    with pytest.raises(ValueError,match='population'):
        r.require_complete(ds,audit,done)
    done[name]['identity']['population_policy']=r.POPULATION_POLICY
    r.require_complete(ds,audit,done)
    audit['errors'].append({'error':'missing selected repo'})
    with pytest.raises(ValueError,match='incomplete'):
        r.require_complete(ds,audit,done)


def test_repository_resume_publish_and_native_lookup(tmp_path):
    r=runner(); ds=CuratedStage2Dataset(make_native_source(tmp_path)); audit=audit_selection(ds)
    out=tmp_path/'stats'; out.mkdir(); repo=ds.repos[0]; code=r.code_identity()
    directory=r.shard_directory(out,repo['repo'])
    job=dict(repo=repo,source=audit['sources'][repo['repo']],code=code,directory=str(directory),batch_size=3,checkpoint_episodes=1)
    first=r.run_repository(job,ds); second=r.run_repository(job,ds)
    assert first==second
    ready=r.publish(ds,audit,out,code)
    assert ready['status']=='complete'
    from openwam.dataloader.native_stage2_stats import read_registry,lookup_stats
    registry=read_registry(ready['stats_registry'],ds.stats_contract())
    entry=lookup_stats(registry,repo,ds._episode(repo,0))
    assert all(type(v) is int for v in entry['action']['count'])
    assert entry['action']['min'][3]==-1
    assert entry['empirical']['action']['min'][3] is not None
    assert 'lower' in entry['quantile_intervals']['action']['q01']
    assert ready['coverage']['episodes']==len({x[0] for x in repo['ranges']})
    tcp=registry['informational_tcp_summary']
    dimensions=[*range(20),*range(34,52)]
    assert tcp['normalization'] is False and tcp['raw_dimensions']==dimensions
    assert tcp['action']['count']==[entry['action']['count'][i] for i in dimensions]
    assert tcp['state']['count']==[entry['state']['count'][i] for i in dimensions]
    assert tcp['local_raw_dimensions']==list(range(20))
    assert tcp['head_raw_dimensions']==list(range(34,52))


def test_worker_restart_after_committed_episode_does_not_count_twice(tmp_path,monkeypatch):
    r=runner(); ds=CuratedStage2Dataset(make_native_source(tmp_path)); audit=audit_selection(ds)
    repo=ds.repos[0]; directory=tmp_path/'shard'
    job=dict(repo=repo,source=audit['sources'][repo['repo']],code=r.code_identity(),directory=str(directory),batch_size=4,checkpoint_episodes=1)
    save=r.save_checkpoint
    def crash_after_pointer(*args,**kwargs):
        save(*args,**kwargs)
        raise RuntimeError('simulated process crash')
    monkeypatch.setattr(r,'save_checkpoint',crash_after_pointer)
    with pytest.raises(RuntimeError,match='simulated process crash'): r.run_repository(job,ds)
    saved=json.loads((directory/'checkpoint.json').read_text())
    monkeypatch.setattr(r,'save_checkpoint',save)
    result=r.run_repository(job,ds)
    assert result['counts']==saved['counts']
    checkpoint=r.load_checkpoint(directory,r.repository_identity(ds,repo,job['source'],job['code']))
    assert checkpoint['done']
    assert checkpoint['completed_episodes']==saved['completed_episodes']


def test_spawned_cli_finishes_fixture(tmp_path):
    import subprocess
    import sys
    r=runner(); source=make_native_source(tmp_path); ds=CuratedStage2Dataset(source)
    audit=tmp_path/'audit.json'; audit.write_text(json.dumps(audit_selection(ds)))
    output=tmp_path/'result'
    result=subprocess.run([sys.executable,'-m','scripts.data.prepare_full_relative_stats',
        '--source-yaml',str(source),'--output',str(output),'--metadata-audit',str(audit),
        '--workers','1','--batch-size','3','--checkpoint-episodes','1'],capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stdout+result.stderr
    assert json.loads((output/'readiness.json').read_text())['status']=='complete'
    assert json.loads((output/'progress.json').read_text())['completed_repositories']==1


def test_code_identity_binds_external_kinematics_dependency(monkeypatch):
    r=runner(); original=r.sha256; before=r.code_identity()
    def changed_dependency(path):
        if Path(path).as_posix().endswith('/openwam/kinematics.py'):
            return 'changed numerical kinematics implementation'
        return original(path)
    monkeypatch.setattr(r,'sha256',changed_dependency)
    assert r.code_identity()!=before


def test_fully_masked_declared_joints_cannot_publish_readiness(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    r=runner(); source=make_native_source(tmp_path)
    root=tmp_path/'source/interndata_a1/unit'
    info=json.loads((root/'meta/info.json').read_text())
    info['features']['actions.joint.mask']={'shape':[14]}
    (root/'meta/info.json').write_text(json.dumps(info))
    data=root/'data/episode-0.parquet'; table=pq.read_table(data)
    pq.write_table(table.append_column('actions.joint.mask',pa.array([[0.]*14]*len(table))),data)
    ds=CuratedStage2Dataset(source); audit=audit_selection(ds); repo=ds.repos[0]
    out=tmp_path/'stats'; out.mkdir(); code=r.code_identity()
    directory=r.shard_directory(out,repo['repo'])
    job=dict(repo=repo,source=audit['sources'][repo['repo']],code=code,directory=str(directory),batch_size=3,checkpoint_episodes=1)
    r.run_repository(job,ds)
    with pytest.raises(ValueError,match='stats_insufficient_valid_population'):
        r.publish(ds,audit,out,code)
    assert not (out/'readiness.json').exists()
    assert not list(out.glob('registry-*.json'))
    checkpoint=r.load_checkpoint(directory,r.repository_identity(ds,repo,job['source'],code))
    assert checkpoint['done'] and not checkpoint['action'].count[20:34].any()


def test_repository_episode_load_uses_scoped_existing_metadata_cache(tmp_path,monkeypatch):
    from openwam.dataloader import episode_metadata
    r=runner(); ds=CuratedStage2Dataset(make_native_source(tmp_path)); audit=audit_selection(ds)
    repo=ds.repos[0]; job=dict(repo=repo,source=audit['sources'][repo['repo']],code=r.code_identity(),
        directory=str(tmp_path/'shard'),batch_size=3,checkpoint_episodes=1)
    original=r.load_episode; enabled_during_load=[]
    previous=getattr(episode_metadata._LOCAL,'enabled',False)
    def checked_load(*args,**kwargs):
        enabled_during_load.append(getattr(episode_metadata._LOCAL,'enabled',False))
        return original(*args,**kwargs)
    monkeypatch.setattr(r,'load_episode',checked_load)
    r.run_repository(job,ds)
    assert enabled_during_load and all(enabled_during_load)
    assert getattr(episode_metadata._LOCAL,'enabled',False)==previous
    assert any(key[0]==repo['path'] for key in episode_metadata._CACHE.entries)
