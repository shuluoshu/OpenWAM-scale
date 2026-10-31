import copy
import json
from pathlib import Path
import pytest


def module():
    from scripts.data import finalize_relative_stats_subset
    return finalize_relative_stats_subset


def fixture(tmp_path):
    import yaml
    from tests.dataloader.test_curated_stage2 import make_native_source
    from openwam.dataloader.curated_stage2 import CuratedStage2Dataset
    from openwam.dataloader.native_stage2_stats import audit_selection
    new=make_native_source(tmp_path); doc=yaml.safe_load(new.read_text())
    olddoc=copy.deepcopy(doc)
    olddoc['resolver']['segments_by_repo']['interndata_a1/missing']=[[0,0,2]]
    olddoc['resolver']['repo_weights']['interndata_a1/missing']=1
    old=tmp_path/'old.yaml'; old.write_text(yaml.safe_dump(olddoc))
    oldds=CuratedStage2Dataset(old); newds=CuratedStage2Dataset(new)
    return oldds,newds,audit_selection(oldds),audit_selection(newds)


def test_only_exact_retained_selection_allowed(tmp_path):
    m=module(); old,new,oa,na=fixture(tmp_path)
    m.validate_subset(old,new,oa,na)
    for field,value in [('weight',2),('ranges',[(0,2,4)])]:
        saved=new.repos[0][field]; new.repos[0][field]=value
        with pytest.raises(ValueError,match='retained repository changed'): m.validate_subset(old,new,oa,na)
        new.repos[0][field]=saved
    with pytest.raises(ValueError,match='strict subset'): m.validate_subset(new,new,na,na)


def test_transfer_preserves_blobs_and_publishes_native_registry(tmp_path):
    m=module(); old,new,oa,na=fixture(tmp_path); r=m.runner
    source=tmp_path/'oldstats'; dest=tmp_path/'newstats'; source.mkdir(); dest.mkdir()
    repo=new.repos[0]; code=r.code_identity(); olddir=r.shard_directory(source,repo['repo'])
    r.run_repository(dict(repo=repo,source=oa['sources'][repo['repo']],code=code,directory=str(olddir),batch_size=4,checkpoint_episodes=1),old)
    before=(olddir/'checkpoint.json').read_bytes()
    m.validate_subset(old,new,oa,na)
    assert m.transfer_repository(old,new,oa,na,source,dest,repo,code)
    newdir=r.shard_directory(dest,repo['repo']); transferred=json.loads((newdir/'checkpoint.json').read_text())
    assert (olddir/'checkpoint.json').read_bytes()==before
    assert transferred['migration']['old_checkpoint_sha256']==r.sha256(olddir/'checkpoint.json')
    for blob in transferred['statistics'].values():
        assert (olddir/blob['file']).read_bytes()==(newdir/blob['file']).read_bytes()
    ready=m.finalize(old,new,oa,na,source,dest,poll_seconds=.01)
    assert m.verify_native_consumer(new.selection['source_yaml'],ready,na)['status']=='complete'


def test_missing_checkpoint_and_stopped_old_runner_fails(tmp_path):
    m=module(); old,new,oa,na=fixture(tmp_path)
    source=tmp_path/'oldstats'; source.mkdir(); dest=tmp_path/'newstats'
    with pytest.raises(ValueError,match='old runner stopped'):
        m.finalize(old,new,oa,na,source,dest,poll_seconds=.01)
    assert not (dest/'readiness.json').exists()
    assert json.loads((dest/'migration_progress.json').read_text())['status']=='failed'


def test_failed_old_repository_is_not_silently_waited(tmp_path):
    m=module(); old,new,oa,na=fixture(tmp_path)
    source=tmp_path/'oldstats'; dest=tmp_path/'newstats'
    directory=m.runner.shard_directory(source,new.repos[0]['repo']); directory.mkdir(parents=True)
    (directory/'progress.json').write_text(json.dumps({'status':'failed','error':'bad source'}))
    with pytest.raises(ValueError,match='old repository failed'):
        m.finalize(old,new,oa,na,source,dest,poll_seconds=.01)
    assert not (dest/'readiness.json').exists()


def test_changed_source_refuses_shard_transfer(tmp_path):
    m=module(); old,new,oa,na=fixture(tmp_path); r=m.runner
    source=tmp_path/'oldstats'; dest=tmp_path/'newstats'; source.mkdir(); dest.mkdir()
    repo=new.repos[0]; code=r.code_identity(); directory=r.shard_directory(source,repo['repo'])
    r.run_repository(dict(repo=repo,source=oa['sources'][repo['repo']],code=code,directory=str(directory),batch_size=4,checkpoint_episodes=1),old)
    data=Path(repo['path'])/'data/episode-0.parquet'; data.touch()
    with pytest.raises(ValueError,match='source changed'):
        m.transfer_repository(old,new,oa,na,source,dest,repo,code)
    assert not (r.shard_directory(dest,repo['repo'])/'checkpoint.json').exists()


def test_live_watcher_waits_then_reuses_newly_completed_shard(tmp_path,monkeypatch):
    import fcntl
    m=module(); old,new,oa,na=fixture(tmp_path); r=m.runner
    source=tmp_path/'oldstats'; source.mkdir(); dest=tmp_path/'newstats'
    lock=(source/'.runner.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert m.old_runner_active(source)
    repo=new.repos[0]; code=r.code_identity(); waited=[]
    def finish_old_job(delay):
        waited.append(delay)
        r.run_repository(dict(repo=repo,source=oa['sources'][repo['repo']],code=code,
            directory=str(r.shard_directory(source,repo['repo'])),batch_size=4,checkpoint_episodes=1),old)
        fcntl.flock(lock,fcntl.LOCK_UN); lock.close()
    monkeypatch.setattr(m.time,'sleep',finish_old_job)
    ready=m.finalize(old,new,oa,na,source,dest,poll_seconds=.01)
    assert waited==[.01] and ready['status']=='complete'
    assert not m.old_runner_active(source)


def test_completion_between_poll_and_old_lock_check_is_rechecked(tmp_path,monkeypatch):
    m=module(); old,new,oa,na=fixture(tmp_path); r=m.runner
    source=tmp_path/'oldstats'; source.mkdir(); dest=tmp_path/'newstats'
    repo=new.repos[0]; code=r.code_identity()
    r.run_repository(dict(repo=repo,source=oa['sources'][repo['repo']],code=code,
        directory=str(r.shard_directory(source,repo['repo'])),batch_size=4,checkpoint_episodes=1),old)
    transfer=m.transfer_repository; calls=[]
    def racing_transfer(*args):
        calls.append(1)
        return False if len(calls)==1 else transfer(*args)
    monkeypatch.setattr(m,'transfer_repository',racing_transfer)
    assert m.finalize(old,new,oa,na,source,dest,poll_seconds=.01)['status']=='complete'
    assert len(calls)==2


def test_finalizer_own_code_change_before_publication_fails(tmp_path,monkeypatch):
    m=module(); old,new,oa,na=fixture(tmp_path); r=m.runner
    source=tmp_path/'oldstats'; source.mkdir(); dest=tmp_path/'newstats'
    repo=new.repos[0]; code=r.code_identity()
    r.run_repository(dict(repo=repo,source=oa['sources'][repo['repo']],code=code,
        directory=str(r.shard_directory(source,repo['repo'])),batch_size=4,checkpoint_episodes=1),old)
    transfer=m.transfer_repository; sha=r.sha256; changed=[]
    def changed_code(path):
        if changed and Path(path).resolve()==Path(m.__file__).resolve(): return 'changed code'
        return sha(path)
    def transfer_then_change(*args):
        result=transfer(*args); changed.append(True); return result
    monkeypatch.setattr(r,'sha256',changed_code)
    monkeypatch.setattr(m,'transfer_repository',transfer_then_change)
    with pytest.raises(ValueError,match='finalizer code changed'):
        m.finalize(old,new,oa,na,source,dest,poll_seconds=.01)
    assert not (dest/'readiness.json').exists()
    assert not list(dest.glob('registry-*.json'))
