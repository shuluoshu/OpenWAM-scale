"""Finite population semantics, without decoding or model initialization."""
import json
import sqlite3
from pathlib import Path
import numpy as np
import pytest
from openwam.dataloader import goal_epoch


class Source:
    def __init__(self, tmp_path, fps=30, length=12):
        p=tmp_path/'source.yaml';p.write_text('fixture')
        self.selection={'source_yaml':str(p)}
        self.repos=[dict(repo='fixture',path=str(tmp_path),family='fixture',ranges=[(0,1,length)])]
        self.seed=13;self._evaluating=False;self.length=length;self.fps=fps;self.calls=0
        self.valid=np.ones(length,bool)
    def _episode(self, repo, episode):
        self.calls+=1
        return dict(length=self.length,info={'fps':self.fps},tasks=[dict(start=0,end_exclusive=self.length,text='task')],
                    subtasks=[dict(start=2,end_exclusive=8,text='sub')],history_source_valid=self.valid)
    def _materialize(self, repo, episode, m, segment, anchor, goal, level):
        return dict(anchor=anchor,goal=goal,level=level,draw=self._history_draw_id), {'text':segment['text']}
    def scope_summary(self):return {'repositories':1}


def test_exact_grid_both_levels_and_source_flags(tmp_path):
    d=Source(tmp_path);d.valid[4]=False
    summary=goal_epoch.build_epoch_index(d,tmp_path/'index.db')
    finite=goal_epoch.FiniteGoalDataset(d,tmp_path/'index.db')
    pairs=[finite.sample(i)[0] for i in range(len(finite))]
    assert [(x['level'],x['anchor'],x['goal']) for x in pairs]==[
        ('task',2,11),('task',6,11),('task',8,11),('task',10,11),('subtask',2,7),('subtask',6,7)]
    assert summary['excluded_invalid_anchor_pairs']==2
    assert summary['total_samples']==6 and summary['complete']
    assert len(summary['index_sha256'])==len(summary['source_yaml_sha256'])==64
    assert finite.sample(3)==finite.sample(3)
    with pytest.raises(IndexError):finite.sample(6)


def test_nondivisible_and_slow_source_no_duplicate_frames(tmp_path):
    d=Source(tmp_path,fps=20);goal_epoch.build_epoch_index(d,tmp_path/'index.db')
    f=goal_epoch.FiniteGoalDataset(d,tmp_path/'index.db')
    anchors=[f.sample(i)[0]['anchor'] for i in range(len(f)) if f.sample(i)[0]['level']=='task']
    assert anchors==[1,2,4,5,6,8,9,10]


def test_source_identity_and_incomplete_index_rejected(tmp_path):
    d=Source(tmp_path);p=tmp_path/'index.db';goal_epoch.build_epoch_index(d,p)
    d.repos[0]['ranges']=[(0,2,12)]
    with pytest.raises(ValueError,match='identity'):goal_epoch.validate_index(d,p)
    d.repos[0]['ranges']=[(0,1,12)]
    with sqlite3.connect(p) as c:c.execute("UPDATE metadata SET value='false' WHERE key='complete'")
    with pytest.raises(ValueError,match='incomplete'):goal_epoch.validate_index(d,p)


def test_old_annotation_segmentation_index_requires_rebuild(tmp_path):
    d=Source(tmp_path);p=tmp_path/'index.db';goal_epoch.build_epoch_index(d,p)
    with sqlite3.connect(p) as c:c.execute("UPDATE metadata SET value='2' WHERE key='version'")
    with pytest.raises(ValueError,match='identity'):goal_epoch.validate_index(d,p)
    with pytest.raises(ValueError,match='identity'):goal_epoch.build_epoch_index(d,p)


def test_resume_committed_episode_and_compact_storage(tmp_path):
    d=Source(tmp_path,length=100_000);d.repos[0]['ranges'].append((1,0,100_000))
    original=d._episode
    def fail(repo,episode):
        if episode==1:raise OSError('source temporarily unavailable')
        return original(repo,episode)
    d._episode=fail;p=tmp_path/'index.db'
    with pytest.raises(OSError):goal_epoch.build_epoch_index(d,p)
    with sqlite3.connect(p) as c:assert c.execute('SELECT COUNT(*) FROM episodes').fetchone()[0]==1
    seen=[]
    def resume(repo,episode):seen.append(episode);return original(repo,episode)
    d._episode=resume
    summary=goal_epoch.build_epoch_index(d,p)
    assert seen==[1] and summary['complete']
    with sqlite3.connect(p) as c:assert c.execute('SELECT COUNT(*) FROM runs').fetchone()[0]<=4
    assert p.stat().st_size<100_000


def test_curated_global_flags_without_history(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from test_curated_goal import fixture
    from openwam.dataloader.curated_goal import CuratedGoalDataset
    config,root=fixture(tmp_path)
    rows=pq.read_table(root/'data.parquet').to_pylist()
    for row in rows:row['training.valid']=row['frame_index']!=6
    pq.write_table(pa.Table.from_pylist(rows),root/'data.parquet')
    d=CuratedGoalDataset(config,tmp_path/'cache')
    summary=goal_epoch.build_epoch_index(d,tmp_path/'index.db')
    assert summary['total_samples']==1 and summary['excluded_invalid_anchor_pairs']==1


def test_native_absolute_selection_external_annotation_and_alias(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import yaml
    from test_curated_goal import fixture
    from openwam.dataloader.curated_goal import CuratedGoalDataset,parse_curated_yaml
    config,root=fixture(tmp_path)
    info=json.loads((root/'meta/info.json').read_text())
    info['features']={'observation.images.head':dict(dtype='video',shape=[24,32,3])}
    (root/'meta/info.json').write_text(json.dumps(info))
    old='/missing/original/native';ann=tmp_path/'external.parquet'
    pq.write_table(pa.Table.from_pylist([dict(repo=old,episode_index=0,subtask_index=0,start_frame=0,end_frame=8,subtask='external')]),ann)
    config.write_text(yaml.safe_dump(dict(data_root_dir=str(tmp_path),selection=dict(segments_by_repo={old:[[0,1,7]]},repo_weights={old:1}),
        curation=dict(post_training_additions=dict(sources={old:dict(cameras=['observation.images.head',None,None],annotation_file=str(ann),annotation_filter={'repo':old})})))))
    d=CuratedGoalDataset(config,tmp_path/'cache',path_aliases={old:str(root)})
    assert d.repos[0]['repo']==old and d.repos[0]['path']==str(root)
    m=d._episode(d.repos[0],0)
    assert m['subtasks']==[dict(start=0,end_exclusive=8,text='external',subtask_index=0)]
    assert m['head']=='observation.images.head'
    goal_epoch.build_epoch_index(d,tmp_path/'index.db')
    assert len(goal_epoch.FiniteGoalDataset(d,tmp_path/'index.db'))==12


def test_metadata_and_annotation_changes_invalidate_index(tmp_path):
    d=Source(tmp_path);(tmp_path/'meta').mkdir();p=tmp_path/'meta/info.json';p.write_text('{"fps":30}')
    goal_epoch.build_epoch_index(d,tmp_path/'index.db')
    p.write_text('{"fps":60}')
    with pytest.raises(ValueError,match='identity'):goal_epoch.validate_index(d,tmp_path/'index.db')


def test_evaluation_uses_finite_valid_samples_and_restores_training_mode(tmp_path):
    d=Source(tmp_path);d.valid[2]=False
    goal_epoch.build_epoch_index(d,tmp_path/'index.db');f=goal_epoch.FiniteGoalDataset(d,tmp_path/'index.db')
    samples=f.evaluation_samples()
    assert {r['level'] for r,_ in samples}=={'task','subtask'}
    assert all(r['anchor']!=2 for r,_ in samples) and d._evaluating is False


@pytest.mark.parametrize('pixel_xy,inside',[([4.,8.],True),([31.5,8.],False)])
def test_native_materialization_dispatches_verified_projection(tmp_path,monkeypatch,pixel_xy,inside):
    import sys
    import types
    import pyarrow as pa
    import pyarrow.parquet as pq
    import yaml
    av=pytest.importorskip('av')
    from test_curated_goal import fixture
    from openwam.dataloader.curated_goal import CuratedGoalDataset
    config,root=fixture(tmp_path)
    info=json.loads((root/'meta/info.json').read_text());info['features']={'observation.images.head':dict(dtype='video',shape=[24,32,3])}
    (root/'meta/info.json').write_text(json.dumps(info))
    epfile=root/'meta/episodes/chunk-000/file-000.parquet'
    rows=pq.read_table(epfile).to_pylist();rows=[{k.replace('egocentric','head'):v for k,v in r.items()} for r in rows]
    pq.write_table(pa.Table.from_pylist(rows),epfile)
    with av.open(str(root/'video.mp4'),'w') as container:
        stream=container.add_stream('mpeg4',rate=10);stream.width=32;stream.height=24;stream.pix_fmt='yuv420p'
        for i in range(10):
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.full((24,32,3),i*20,np.uint8),format='rgb24')):container.mux(packet)
        for packet in stream.encode():container.mux(packet)
    key=str(root);config.write_text(yaml.safe_dump(dict(data_root_dir='/',selection=dict(segments_by_repo={key:[[0,1,2]]},repo_weights={key:1}),curation=dict(post_training_additions=dict(sources={key:{}})))))
    calls=[]
    def project(info,ep,columns,index,camera):
        calls.append((index,camera))
        return np.array([pixel_xy,[10,12]],np.float32),np.array([True,False]),{'definition_id':'verified_native_fixture'}
    monkeypatch.setitem(sys.modules,'openwam.dataloader.posttrain_physical',types.SimpleNamespace(project_posttrain_goal=project))
    d=CuratedGoalDataset(config,tmp_path/'cache');record,batch=d.sample(0)
    assert calls==[(0,'observation.images.head')]
    targets=np.load(record['target_path'])
    assert targets['projection_valid'].tolist()==[True,False]
    expected=(np.array(pixel_xy)+.5)*[320/32,256/24]-.5
    assert targets['in_frame'].tolist()==[inside,False]
    np.testing.assert_allclose(targets['uv'][0],expected/[319.,255.] if inside else [0.,0.],rtol=1e-6)
    np.testing.assert_array_equal(targets['uv'][1],[0.,0.])
    assert record['provenance']['endpoint']['definition_id']=='verified_native_fixture'


def test_canonical_task_and_frame_flag_changes_invalidate_index(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from test_curated_goal import fixture
    from openwam.dataloader.curated_goal import CuratedGoalDataset
    config,root=fixture(tmp_path);path=tmp_path/'index.db'
    d=CuratedGoalDataset(config,tmp_path/'cache');goal_epoch.build_epoch_index(d,path)
    frames=pq.read_table(root/'data.parquet').to_pylist()
    for row in frames:row['training.valid']=row['frame_index']!=6
    pq.write_table(pa.Table.from_pylist(frames),root/'data.parquet')
    with pytest.raises(ValueError,match='source.*changed|identity'):goal_epoch.validate_index(d,path)
    with pytest.raises(ValueError,match='source.*changed|identity'):goal_epoch.build_epoch_index(d,path)


def test_task_mapping_content_and_new_annotations_invalidate_index(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from test_curated_goal import fixture
    from openwam.dataloader.curated_goal import CuratedGoalDataset
    config,root=fixture(tmp_path);path=tmp_path/'index.db'
    d=CuratedGoalDataset(config,tmp_path/'cache');goal_epoch.build_epoch_index(d,path)
    mapping=root/'meta/tasks.parquet';original=mapping.read_bytes()
    rows=pq.read_table(mapping).to_pylist();rows[0]['task']='changed task'
    pq.write_table(pa.Table.from_pylist(rows),mapping)
    with pytest.raises(ValueError,match='identity'):goal_epoch.validate_index(d,path)
    mapping.write_bytes(original)
    pq.write_table(pa.Table.from_pylist([dict(episode_index=0,subtask_index=0,subtask='new')]),root/'meta/sub_tasks.parquet')
    with pytest.raises(ValueError,match='identity'):goal_epoch.validate_index(d,path)


def test_runtime_source_mutation_and_stale_reader_cache_are_rejected(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import os
    from test_curated_goal import fixture
    from openwam.dataloader.curated_goal import CuratedGoalDataset
    config,root=fixture(tmp_path);path=tmp_path/'index.db'
    d=CuratedGoalDataset(config,tmp_path/'cache');goal_epoch.build_epoch_index(d,path)
    f=goal_epoch.FiniteGoalDataset(d,path)
    source=root/'data.parquet';stat=source.stat();rows=pq.read_table(source).to_pylist()
    # Keep the source schema and size stable: only replace one task index.
    rows[6]['task_index']=1
    pq.write_table(pa.Table.from_pylist(rows),source)
    os.utime(source,ns=(stat.st_atime_ns,stat.st_mtime_ns))
    with pytest.raises(ValueError,match='source.*changed|identity'):f.sample(1)


def test_interrupted_resume_checks_committed_episode_dependencies(tmp_path):
    d=Source(tmp_path);d.repos[0]['ranges'].append((1,0,12));dependency=tmp_path/'labels.bin';dependency.write_bytes(b'original')
    original=d._episode
    def fail(repo,episode):
        if episode==1:raise OSError('interrupted')
        return dict(original(repo,episode),data=dependency)
    d._episode=fail;path=tmp_path/'index.db'
    with pytest.raises(OSError):goal_epoch.build_epoch_index(d,path)
    dependency.write_bytes(b'changed labels')
    d._episode=original
    with pytest.raises(ValueError,match='source dependency changed'):goal_epoch.build_epoch_index(d,path)


@pytest.mark.parametrize('change',['validity','instruction'])
def test_projected_content_digest_rejects_change_without_file_stat_change(tmp_path,change):
    d=Source(tmp_path);path=tmp_path/'index.db';goal_epoch.build_epoch_index(d,path)
    f=goal_epoch.FiniteGoalDataset(d,path)
    if change=='validity':d.valid[2]=False
    else:
        original=d._episode
        def changed(repo,episode):
            m=original(repo,episode);m['tasks'][0]['text']='changed without file stamp';return m
        d._episode=changed
    with pytest.raises(ValueError,match='source labels changed'):f.sample(0)


def test_three_fps_keeps_terminal_goals_and_rejects_other_rate_index(tmp_path):
    d=Source(tmp_path,length=61)
    original=d._episode
    def episode(repo,index):
        m=original(repo,index);m['subtasks']=[dict(start=2,end_exclusive=28,text='sub')];return m
    d._episode=episode;path=tmp_path/'index.db'
    summary=goal_epoch.build_epoch_index(d,path,sample_fps=3.)
    f=goal_epoch.FiniteGoalDataset(d,path,sample_fps=3.)
    pairs=[f.sample(i)[0] for i in range(len(f))]
    assert [(r['level'],r['anchor'],r['goal']) for r in pairs]==[
        *[('task',a,60) for a in (10,20,30,40,50)],('subtask',10,27),('subtask',20,27)]
    assert summary['sample_fps']==3.
    with pytest.raises(ValueError,match='sample_fps'):goal_epoch.validate_index(d,path,sample_fps=15.)
    with pytest.raises(ValueError,match='sample_fps'):goal_epoch.FiniteGoalDataset(d,path,sample_fps=15.)


@pytest.mark.parametrize('bad',[0,-1,float('nan'),float('inf'),True,'3',None])
def test_invalid_configured_epoch_sample_rate_rejected(bad):
    with pytest.raises(ValueError,match='sample_fps'):goal_epoch.epoch_sample_fps({'sample_fps':bad})


class ParallelSource(Source):
    """Metadata reader that exposes accidental shared LRU use and completion order."""
    def __init__(self, tmp_path):
        from collections import OrderedDict
        import threading
        super().__init__(tmp_path,length=31)
        self.repos[0]['ranges']=[(e,1,15) for e in range(7)]+[(e,18,31) for e in range(7)]
        self._episodes=OrderedDict();self.max_cached_episodes=2
        self.probe={'lock':threading.Lock(),'active':0,'peak':0,'reads':[], 'owners':{}, 'fail':None}
        self.dependency=tmp_path/'dependency.bin'
        if not self.dependency.exists():self.dependency.write_bytes(b'fixed source')
    def _episode(self,repo,episode):
        import threading
        import time
        probe=self.probe;owner=threading.get_ident()
        with probe['lock']:
            probe['active']+=1;probe['peak']=max(probe['peak'],probe['active'])
            probe['reads'].append(episode)
            assert probe['owners'].setdefault(id(self._episodes),owner)==owner
        try:
            time.sleep(.03 if episode%2==0 else .002)
            if probe['fail']==episode:raise OSError('injected metadata failure')
            m=super()._episode(repo,episode)
            m['info']['fps']=[20,30,2][episode%3]
            m['source_valid']=np.arange(self.length)%7!=episode%7
            # Adjacent equal IDs/text are distinct v3 occurrences and goals.
            m['subtasks']=[dict(start=a,end_exclusive=b,text='repeat',subtask_index=1)
                for a,b in ((0,8),(8,16),(16,31))]
            m['data']=self.dependency
            self._episodes[(repo['path'],episode)]=m
            while len(self._episodes)>self.max_cached_episodes:self._episodes.popitem(last=False)
            return m
        finally:
            with probe['lock']:probe['active']-=1


def _index_rows(path):
    with sqlite3.connect(path) as connection:
        return {table:connection.execute(f'SELECT * FROM {table} ORDER BY 1,2').fetchall()
            for table in ('metadata','episodes','runs','sources','episode_sources')}


@pytest.mark.parametrize('workers,max_pending',[(3,4),(4,2)])
def test_parallel_index_matches_serial_rows_and_bounds_metadata_readers(tmp_path,workers,max_pending):
    serial=ParallelSource(tmp_path)
    serial_path=tmp_path/'serial.sqlite'
    goal_epoch.build_epoch_index(serial,serial_path,sample_fps=3.)
    parallel=ParallelSource(tmp_path);progress=tmp_path/'progress.json'
    parallel_path=tmp_path/'parallel.sqlite'
    summary=goal_epoch.build_epoch_index(parallel,parallel_path,sample_fps=3.,
        workers=workers,max_pending=max_pending,commit_every=3,progress_path=progress)
    assert _index_rows(serial_path)==_index_rows(parallel_path)
    assert 1<parallel.probe['peak']<=min(workers,max_pending)
    assert not parallel._episodes  # No original dataset cache is touched by workers.
    assert len(parallel.probe['owners'])<=min(workers,max_pending)
    status=json.loads(progress.read_text())
    assert status['status']=='complete'
    assert status['completed_episodes']==status['total_episodes']==7
    assert status['total_samples']==summary['total_samples']
    assert status['newly_completed_episodes']==7 and status['episodes_per_second']>0
    assert status['commits']==3
    assert not list(tmp_path.glob('progress.json.tmp*'))


def test_parallel_failure_rolls_back_batch_and_resumes_exact_v3_rows(tmp_path):
    source=ParallelSource(tmp_path);path=tmp_path/'resumed.sqlite';progress=tmp_path/'progress.json'
    source.probe['fail']=3
    with pytest.raises(OSError,match='injected metadata failure'):
        goal_epoch.build_epoch_index(source,path,workers=3,max_pending=4,commit_every=2,progress_path=progress)
    with sqlite3.connect(path) as connection:
        assert connection.execute('SELECT episode FROM episodes ORDER BY episode').fetchall()==[(0,),(1,)]
        assert json.loads(connection.execute("SELECT value FROM metadata WHERE key='complete'").fetchone()[0]) is False
        before=connection.execute('SELECT * FROM runs ORDER BY sample_start').fetchall()
    failed=json.loads(progress.read_text())
    assert failed['status']=='failed' and failed['completed_episodes']==2
    assert 'injected metadata failure' in failed['error']
    source.probe.update(fail=None,reads=[],owners={})
    goal_epoch.build_epoch_index(source,path,workers=2,max_pending=2,commit_every=3,progress_path=progress)
    assert sorted(source.probe['reads'])==[2,3,4,5,6]
    with sqlite3.connect(path) as connection:
        assert connection.execute('SELECT * FROM runs ORDER BY sample_start LIMIT ?', (len(before),)).fetchall()==before
    status=json.loads(progress.read_text())
    assert status['resumed_episodes']==2 and status['newly_completed_episodes']==5
    source.probe['owners']={}
    goal_epoch.build_epoch_index(source,tmp_path/'serial.sqlite')
    assert _index_rows(path)==_index_rows(tmp_path/'serial.sqlite')


def test_parallel_resume_rejects_committed_dependency_change(tmp_path):
    source=ParallelSource(tmp_path);path=tmp_path/'index.sqlite'
    source.probe['fail']=2
    with pytest.raises(OSError):
        goal_epoch.build_epoch_index(source,path,workers=2,commit_every=2)
    source.dependency.write_bytes(b'mutated source')
    source.probe.update(fail=None,reads=[])
    with pytest.raises(ValueError,match='source dependency changed'):
        goal_epoch.build_epoch_index(source,path,workers=2,commit_every=2)
    assert source.probe['reads']==[]


def test_parallel_source_stamp_is_checked_after_prefetch(tmp_path,monkeypatch):
    source=ParallelSource(tmp_path);path=tmp_path/'index.sqlite'
    original=goal_epoch._episode_sources
    changed=False
    def capture_then_mutate(metadata):
        nonlocal changed
        sources=original(metadata)
        if not changed:
            changed=True
            source.dependency.write_bytes(b'changed after metadata was prepared')
        return sources
    monkeypatch.setattr(goal_epoch,'_episode_sources',capture_then_mutate)
    with pytest.raises(ValueError,match='source dependency changed'):
        goal_epoch.build_epoch_index(source,path,workers=2,commit_every=2)
    with sqlite3.connect(path) as connection:
        assert connection.execute('SELECT COUNT(*) FROM episodes').fetchone()[0]==0


@pytest.mark.parametrize('kwargs',[{'workers':0},{'workers':True},{'workers':1.5},
    {'max_pending':0},{'max_pending':False},{'commit_every':0},{'commit_every':2.5}])
def test_index_preparation_limits_must_be_positive_integers(tmp_path,kwargs):
    with pytest.raises(ValueError,match='positive integer'):
        goal_epoch.build_epoch_index(Source(tmp_path),tmp_path/'index.sqlite',**kwargs)


def test_global_source_checks_are_bounded_parallel_but_sample_checks_are_serial(tmp_path,monkeypatch):
    import threading
    import time
    connection=sqlite3.connect(':memory:')
    connection.executescript('CREATE TABLE sources(path TEXT,size INTEGER,mtime_ns INTEGER); CREATE TABLE episode_sources(repo_id INTEGER,episode INTEGER,path TEXT);')
    for i in range(9):
        path=tmp_path/f'source_{i}';path.write_text(str(i))
        connection.execute('INSERT INTO sources VALUES (?,?,?)',(str(path),*goal_epoch._file_stamp(path)))
        connection.execute('INSERT INTO episode_sources VALUES (0,0,?)',(str(path),))
    lock=threading.Lock();active=0;peak=0;original=goal_epoch._file_stamp
    def delayed(path):
        nonlocal active,peak
        with lock:active+=1;peak=max(peak,active)
        try:
            time.sleep(.005)
            return original(path)
        finally:
            with lock:active-=1
    monkeypatch.setattr(goal_epoch,'_file_stamp',delayed)
    goal_epoch._check_sources(connection,workers=4)
    assert 1<peak<=4
    peak=0
    goal_epoch._check_sources(connection,0,0,workers=4)
    assert peak==1
    (tmp_path/'source_2').unlink()
    for workers in (1,4):
        with pytest.raises(ValueError,match='source dependency changed:.*source_2'):
            goal_epoch._check_sources(connection,workers=workers)
    connection.close()


def test_parallel_identity_preserves_order_and_hashes_shared_sidecar_once(tmp_path,monkeypatch):
    import threading
    source=Source(tmp_path);shared=tmp_path/'annotation';shared.write_bytes(b'shared')
    source.repos=[]
    for i in range(9):
        root=tmp_path/f'repo_{i}';(root/'meta').mkdir(parents=True)
        (root/'meta/info.json').write_text(str(i))
        source.repos.append(dict(repo=str(i),path=str(root),ranges=[(0,1,12)],native_metadata={'annotation_file':str(shared)}))
    expected=goal_epoch._identity(source,workers=1)
    original=goal_epoch.sha256_file;seen=[];lock=threading.Lock()
    def counted(path):
        with lock:seen.append(str(Path(path).resolve()))
        return original(path)
    monkeypatch.setattr(goal_epoch,'sha256_file',counted)
    assert goal_epoch._identity(source,workers=4)==expected
    assert seen.count(str(shared.resolve()))==1


def test_existing_index_is_warmed_before_sqlite_scan(tmp_path,monkeypatch):
    source=Source(tmp_path);path=tmp_path/'index.sqlite'
    goal_epoch.build_epoch_index(source,path)
    original=goal_epoch._warm_index;warmed=[]
    def warm(path):
        warmed.append(Path(path));return original(path)
    monkeypatch.setattr(goal_epoch,'_warm_index',warm)
    original_connect=goal_epoch.sqlite3.connect
    def connect(*args,**kwargs):
        assert warmed and warmed[-1]==path
        return original_connect(*args,**kwargs)
    monkeypatch.setattr(goal_epoch.sqlite3,'connect',connect)
    goal_epoch.validate_index(source,path)
    warmed.clear()
    goal_epoch.build_epoch_index(source,path)


def test_resume_count_is_visible_before_identity_and_sources_check(tmp_path,monkeypatch):
    source=Source(tmp_path);path=tmp_path/'index.sqlite';progress=tmp_path/'progress.json'
    goal_epoch.build_epoch_index(source,path,progress_path=progress)
    original=goal_epoch._check_identity;seen=[]
    def inspect(*args,**kwargs):
        status=json.loads(progress.read_text())
        if status['status']=='checking_sources':
            assert status['completed_episodes']==status['resumed_episodes']==1
            seen.append(status)
        return original(*args,**kwargs)
    monkeypatch.setattr(goal_epoch,'_check_identity',inspect)
    goal_epoch.build_epoch_index(source,path,progress_path=progress)
    assert seen


def test_validated_summary_skips_global_scan_but_retains_sample_checks(tmp_path,monkeypatch):
    source=Source(tmp_path);dependency=tmp_path/'labels';dependency.write_bytes(b'original')
    original=source._episode
    source._episode=lambda *args:dict(original(*args),data=dependency)
    path=tmp_path/'index.sqlite';summary=goal_epoch.build_epoch_index(source,path)
    def unexpected(*args,**kwargs):raise AssertionError('global validation repeated')
    monkeypatch.setattr(goal_epoch,'validate_index',unexpected)
    finite=goal_epoch.FiniteGoalDataset(source,path,sample_fps=15.,_validated_summary=summary)
    assert len(finite)==summary['total_samples']
    dependency.write_bytes(b'changed')
    with pytest.raises(ValueError,match='source dependency changed'):finite.sample(0)


@pytest.mark.parametrize('field,value',[('index_path','/wrong/index'),('index_sha256','0'*64),
    ('source_yaml_sha256','0'*64),('version',2),('sample_fps',3.),('complete',False),
    ('total_samples',0),('selected_repos_sha256','0'*64)])
def test_validated_summary_rejects_incompatible_broadcast(tmp_path,field,value):
    source=Source(tmp_path);path=tmp_path/'index.sqlite';summary=goal_epoch.build_epoch_index(source,path)
    with pytest.raises(ValueError,match='validated summary'):
        goal_epoch.FiniteGoalDataset(source,path,sample_fps=15.,_validated_summary=dict(summary,**{field:value}))


@pytest.mark.parametrize('mutate_episode,expected_committed',[(2,0),(6,6)])
def test_source_change_after_write_before_batch_commit_rolls_back(tmp_path,monkeypatch,mutate_episode,expected_committed):
    source=ParallelSource(tmp_path);path=tmp_path/'index.sqlite';progress=tmp_path/'progress.json'
    original=goal_epoch._record_sources
    def write_then_mutate(connection,repo_id,episode,*args,**kwargs):
        original(connection,repo_id,episode,*args,**kwargs)
        if episode==mutate_episode:source.dependency.write_bytes(b'changed immediately before batch commit')
    monkeypatch.setattr(goal_epoch,'_record_sources',write_then_mutate)
    with pytest.raises(ValueError,match='source dependency changed'):
        goal_epoch.build_epoch_index(source,path,workers=3,commit_every=3,progress_path=progress)
    with sqlite3.connect(path) as connection:
        assert connection.execute('SELECT COUNT(*) FROM episodes').fetchone()[0]==expected_committed
        assert json.loads(connection.execute("SELECT value FROM metadata WHERE key='complete'").fetchone()[0]) is False
    status=json.loads(progress.read_text())
    assert status['status']=='failed' and status['completed_episodes']==expected_committed


def test_batch_source_rechecks_deduplicate_shared_paths(tmp_path,monkeypatch):
    import threading
    source=ParallelSource(tmp_path);path=tmp_path/'index.sqlite'
    original=goal_epoch._file_stamp;seen=[];lock=threading.Lock()
    def count(path):
        with lock:seen.append(str(path))
        return original(path)
    monkeypatch.setattr(goal_epoch,'_file_stamp',count)
    goal_epoch.build_epoch_index(source,path,workers=3,commit_every=3)
    # Seven worker snapshots, one recheck for each of three transaction batches,
    # then one final global validation. The shared dependency is never re-statted
    # for every occurrence while inserting SQL rows.
    assert seen.count(str(source.dependency.resolve()))==7+3+1


def test_parallel_metadata_error_identifies_original_source(tmp_path):
    class BrokenSource(Source):
        def _episode(self,repo,episode):raise ValueError('source frame indices must be dense unique complete')
    path=tmp_path/'index.sqlite';progress=tmp_path/'progress.json'
    with pytest.raises(ValueError,match='repo_id=0.*repo=fixture.*episode=0.*dense unique complete'):
        goal_epoch.build_epoch_index(BrokenSource(tmp_path),path,workers=2,progress_path=progress)
    error=json.loads(progress.read_text())['error']
    assert 'repo_id=0' in error and 'repo=fixture' in error and 'episode=0' in error


def test_all_episode_data_parts_are_stamped_and_mutation_is_rejected(tmp_path):
    source=Source(tmp_path);first=tmp_path/'part0.parquet';second=tmp_path/'part1.parquet'
    first.write_bytes(b'first');second.write_bytes(b'second')
    original=source._episode
    source._episode=lambda *args:dict(original(*args),data=first,data_files=(first,second))
    path=tmp_path/'index.sqlite';goal_epoch.build_epoch_index(source,path,workers=2)
    with sqlite3.connect(path) as connection:
        assert {row[0] for row in connection.execute('SELECT path FROM sources')}=={str(first.resolve()),str(second.resolve())}
    second.write_bytes(b'changed continuation')
    with pytest.raises(ValueError,match='source dependency changed'):
        goal_epoch.validate_index(source,path)


def test_single_data_part_keeps_existing_v3_digest_and_rows(tmp_path):
    source=Source(tmp_path);data=tmp_path/'data.parquet';data.write_bytes(b'original')
    original=source._episode
    source._episode=lambda *args:dict(original(*args),data=data)
    goal_epoch.build_epoch_index(source,tmp_path/'old.sqlite')
    source._episode=lambda *args:dict(original(*args),data=data,data_files=(data,))
    goal_epoch.build_epoch_index(source,tmp_path/'new.sqlite')
    assert _index_rows(tmp_path/'old.sqlite')==_index_rows(tmp_path/'new.sqlite')
