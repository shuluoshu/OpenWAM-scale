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
