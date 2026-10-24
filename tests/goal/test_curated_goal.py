"""Source semantics without model weights or accelerator allocations."""
import json
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml

from openwam.dataloader.curated_goal import CuratedGoalDataset,parse_curated_yaml


def fixture(tmp_path,annotations=False):
    import pyarrow as pa
    import pyarrow.parquet as pq
    root=tmp_path/'mecka'/'fixture';(root/'meta/episodes/chunk-000').mkdir(parents=True)
    info=dict(fps=10,features={'observation.images.egocentric':dict(dtype='video',shape=[24,32,3])},
        data_path='data.parquet',video_path='video.mp4')
    (root/'meta/info.json').write_text(json.dumps(info))
    ep={'episode_index':0,'length':10,'data/chunk_index':0,'data/file_index':0,
        'videos/observation.images.egocentric/chunk_index':0,'videos/observation.images.egocentric/file_index':0,
        'videos/observation.images.egocentric/from_timestamp':0.}
    pq.write_table(pa.Table.from_pylist([ep]),root/'meta/episodes/chunk-000/file-000.parquet')
    rows=[dict(episode_index=0,frame_index=i,task_index=0 if i<8 else 1,subtask_index=0 if i<3 or i>=6 else 1) for i in range(10)]
    pq.write_table(pa.Table.from_pylist(rows),root/'data.parquet')
    pq.write_table(pa.Table.from_pylist([dict(task_index=0,task='first task'),dict(task_index=1,task='second task')]),root/'meta/tasks.parquet')
    if annotations:
        pq.write_table(pa.Table.from_pylist([dict(episode_index=0,subtask_index=0,subtask='repeat'),dict(episode_index=0,subtask_index=1,subtask='middle')]),root/'meta/sub_tasks.parquet')
    config=tmp_path/'source.yaml'
    config.write_text(yaml.safe_dump(dict(data_root_dir=str(tmp_path),selection=dict(segments_by_repo={'mecka/fixture':[[0,1,2],[0,6,7]]},repo_weights={'mecka/fixture':1,'missing':0}))))
    return config,root


def test_parse_weight_and_disjoint_contract(tmp_path):
    config,_=fixture(tmp_path)
    parsed=parse_curated_yaml(config)
    assert len(parsed['repos'])==1
    assert parsed['repos'][0]['ranges']==[(0,1,2),(0,6,7)]
    d=CuratedGoalDataset(config,tmp_path/'cache')
    assert d.scope_summary()['selected_frames']==2


def test_original_task_interval_and_absent_annotations(tmp_path,monkeypatch):
    config,_=fixture(tmp_path)
    d=CuratedGoalDataset(config,tmp_path/'cache',subtask_probability=1)
    monkeypatch.setattr(d,'_materialize',lambda repo,episode,m,segment,anchor,goal,level:(segment,anchor,goal,level))
    for step in range(20):
        segment,anchor,goal,level=d.sample(step)
        assert anchor in (1,6)
        assert goal==7 and level=='task'
        assert segment['start']==0 and segment['end_exclusive']==8


def test_repeated_subtasks_remain_separate(tmp_path,monkeypatch):
    config,_=fixture(tmp_path,annotations=True)
    d=CuratedGoalDataset(config,tmp_path/'cache',subtask_probability=1)
    monkeypatch.setattr(d,'_materialize',lambda repo,episode,m,segment,anchor,goal,level:(segment,anchor,goal,level))
    seen=set()
    for step in range(30):
        segment,anchor,goal,level=d.sample(step);seen.add((anchor,goal))
        assert level=='subtask' and segment['text']=='repeat'
    assert seen=={(1,2),(6,7)}


def test_present_invalid_annotation_fails(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    config,root=fixture(tmp_path,annotations=True)
    pq.write_table(pa.Table.from_pylist([dict(episode_index=0,subtask_index=0,subtask='')]),root/'meta/sub_tasks.parquet')
    with pytest.raises(ValueError,match='malformed present'):
        CuratedGoalDataset(config,tmp_path/'cache').sample(0)


@pytest.mark.parametrize('rows',[
    [dict(episode_index=0,subtask_index=17,subtask='wrong index')],
    [dict(episode_index=0,subtask_index=0,subtask='repeat',start_frame=1,end_frame=2),
     dict(episode_index=0,subtask_index=1,subtask='middle',start_frame=3,end_frame=6)],
])
def test_present_annotation_mapping_conflicts_fail(tmp_path,rows):
    import pyarrow as pa
    import pyarrow.parquet as pq
    config,root=fixture(tmp_path,annotations=True)
    pq.write_table(pa.Table.from_pylist(rows),root/'meta/sub_tasks.parquet')
    with pytest.raises(ValueError,match='inconsistent subtask annotation'):
        CuratedGoalDataset(config,tmp_path/'cache',subtask_probability=1).sample(0)


@pytest.mark.parametrize('mode',['empty_episode','explicitly_unannotated'])
def test_present_annotations_allow_true_task_only(tmp_path,monkeypatch,mode):
    import pyarrow as pa
    import pyarrow.parquet as pq
    config,root=fixture(tmp_path,annotations=True)
    rows=[dict(episode_index=1 if mode=='empty_episode' else 0,subtask_index=17,subtask='other')]
    pq.write_table(pa.Table.from_pylist(rows),root/'meta/sub_tasks.parquet')
    if mode=='explicitly_unannotated':
        frames=pq.read_table(root/'data.parquet').to_pylist()
        for row in frames:row['subtask_unannotated']=True
        pq.write_table(pa.Table.from_pylist(frames),root/'data.parquet')
    d=CuratedGoalDataset(config,tmp_path/'cache',subtask_probability=1)
    monkeypatch.setattr(d,'_materialize',lambda repo,episode,m,segment,anchor,goal,level:level)
    assert d.sample(0)=='task'


def test_real_decode_missing_geometry_masks(tmp_path):
    av=pytest.importorskip('av')
    config,root=fixture(tmp_path)
    with av.open(str(root/'video.mp4'),'w') as container:
        stream=container.add_stream('mpeg4',rate=10);stream.width=32;stream.height=24;stream.pix_fmt='yuv420p'
        for index in range(10):
            frame=av.VideoFrame.from_ndarray(np.full((24,32,3),index*20,dtype=np.uint8),format='rgb24')
            for packet in stream.encode(frame):container.mux(packet)
        for packet in stream.encode():container.mux(packet)
    d=CuratedGoalDataset(config,tmp_path/'cache')
    record,batch=d.sample(0)
    assert record['goal_source_frame']==7
    assert batch['observed'].device.type=='cpu' and batch['observed'].dtype==torch.bfloat16
    assert batch['observed'].shape==(1,3,1,256,320)
    assert batch['targets'].uv.dtype==torch.float32
    assert not batch['targets'].point_valid.any()
    assert record['provenance']['goal_video']['source_frame_index']==7


def test_history_reader_shapes_range_and_instruction_cache(tmp_path):
    av=pytest.importorskip('av')
    config,root=fixture(tmp_path,annotations=True)
    with av.open(str(root/'video.mp4'),'w') as container:
        stream=container.add_stream('mpeg4',rate=10);stream.width=32;stream.height=24;stream.pix_fmt='yuv420p'
        for index in range(10):
            frame=av.VideoFrame.from_ndarray(np.full((24,32,3),index*20,dtype=np.uint8),format='rgb24')
            for packet in stream.encode(frame):container.mux(packet)
        for packet in stream.encode():container.mux(packet)
    from openwam.train.goal_curated import TextContextCache,prepare_batch
    captured=[]
    def encode(text):
        captured.append(text)
        return torch.zeros(1,2,4),torch.tensor([2])
    cache=TextContextCache(encode)
    d=CuratedGoalDataset(config,tmp_path/'cache',history={'enabled':True},subtask_probability=1)
    record,batch=d.sample(0)
    assert batch['history_images'].shape==(1,10,3,1,256,320)
    assert batch['history_valid'].dtype==torch.bool
    assert batch['history_ages'].dtype==torch.float32
    trace=record['provenance']['head_history']
    assert trace['selected_start'] in (1,6)
    assert all(i==-1 or trace['selected_start']<=i<record['anchor_source_frame'] for i in trace['indices'])
    prepare_batch(record,batch,device='cpu',dtype=torch.float32,context_cache=cache)
    assert captured==['repeat'] and record['instruction']==batch['instruction']=='repeat'
    d.subtask_probability=0
    record,batch=d.sample(1)
    prepare_batch(record,batch,device='cpu',dtype=torch.float32,context_cache=cache)
    assert captured==['repeat','first task']


def test_history_reads_global_source_flags_only_when_enabled(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    config,root=fixture(tmp_path)
    table=pq.read_table(root/'data.parquet')
    table=table.append_column('canonical.frame_valid',pa.array([1.,0.,float('nan')]+[1.]*7))
    table=table.append_column('training.valid',pa.array([1.]*3+[float('inf'),.5]+[1.]*5))
    pq.write_table(table,root/'data.parquet')
    disabled=CuratedGoalDataset(config,tmp_path/'disabled')
    assert 'history_source_valid' not in disabled._episode(disabled.repos[0],0)
    enabled=CuratedGoalDataset(config,tmp_path/'enabled',history={'enabled':True})
    metadata=enabled._episode(enabled.repos[0],0)
    np.testing.assert_array_equal(metadata['history_source_valid'],[True,False,False,False,False]+[True]*5)
