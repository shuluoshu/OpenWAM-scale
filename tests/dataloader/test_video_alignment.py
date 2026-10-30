"""Real encoded VFR frames must follow source ordinals, never rounded PTS."""
from fractions import Fraction
import hashlib

import numpy as np
import pytest


def encode_video(path, *, count=384, fps=30, variable=True):
    av=pytest.importorskip('av')
    with av.open(str(path),'w') as container:
        stream=container.add_stream('mpeg4',rate=fps)
        stream.width=32;stream.height=24;stream.pix_fmt='yuv420p'
        stream.codec_context.time_base=Fraction(1,30000)
        pts=660  # Nonzero first PTS, like the released Mecka videos.
        for index in range(count):
            rgb=np.empty((24,32,3),np.uint8)
            rgb[:]=[index%251,(index*17)%251,(index*37)%251]
            frame=av.VideoFrame.from_ndarray(rgb,format='rgb24')
            frame.pts=pts;frame.time_base=Fraction(1,30000)
            for packet in stream.encode(frame):container.mux(packet)
            pts+=30000//fps+(73 if variable and index%5==0 else 0)
        for packet in stream.encode():container.mux(packet)
    with av.open(str(path)) as container:
        stream=container.streams.video[0]
        rate=float(stream.average_rate)
        reference=[(frame.pts,np.asarray(frame.to_image().convert('RGB'))) for frame in container.decode(stream)]
    assert len(reference)==count and reference[0][0]>0
    if variable:assert abs(rate-fps)>1e-4
    return reference


def test_sequential_vfr_uses_every_decoded_ordinal_through_terminal(tmp_path):
    from scripts.data.prepare_subtask_goals import decode_frames
    path=tmp_path/'vfr.mp4';reference=encode_video(path)
    images,alignment=decode_frames(path,0.,30,[383,0,116,60],sequential=True)
    for index in (0,60,116,383):
        np.testing.assert_array_equal(np.asarray(images[index]),reference[index][1])
        assert alignment[index]['pts']==reference[index][0]
        assert alignment[index]['video_ordinal']==index
        assert alignment[index]['source_rgb_sha256']==hashlib.sha256(reference[index][1].tobytes()).hexdigest()
    with pytest.raises(ValueError,match='incomplete video extraction'):
        decode_frames(path,0.,30,[384],sequential=True)


def test_cfr_path_remains_strict_about_fps_and_pts(tmp_path):
    from scripts.data.prepare_subtask_goals import decode_frames
    path=tmp_path/'vfr.mp4';encode_video(path,count=20)
    with pytest.raises(ValueError,match='FPS mismatch'):
        decode_frames(path,0.,30,[0,19])
    import av
    with av.open(str(path)) as container:rate=float(container.streams.video[0].average_rate)
    with pytest.raises(ValueError,match='nonuniform PTS'):
        decode_frames(path,0.,rate,[0,19])
    path=tmp_path/'cfr.mp4';reference=encode_video(path,count=20,variable=False)
    images,alignment=decode_frames(path,2/30,30,[0,17])
    for index in (0,17):
        np.testing.assert_array_equal(np.asarray(images[index]),reference[index+2][1])
        assert alignment[index]['video_ordinal']==index+2


@pytest.mark.parametrize('offset',[1/30,1e-7])
def test_sequential_rejects_nonzero_time_offset_without_ordinal_origin(tmp_path,offset):
    from scripts.data.prepare_subtask_goals import decode_frames
    path=tmp_path/'cfr.mp4';encode_video(path,count=8,variable=False)
    with pytest.raises(ValueError,match='nonzero time offset'):
        decode_frames(path,offset,30,[0],sequential=True)


def test_explicit_raw_ordinals_preserve_episode_indices_and_source_pixels(tmp_path):
    from scripts.data.prepare_subtask_goals import decode_frames
    path=tmp_path/'vfr.mp4';reference=encode_video(path,count=20)
    images,alignment=decode_frames(path,0.,30,[0,6,9],sequential=True,
        video_ordinals={0:3,6:9,9:12})
    for index in (0,6,9):
        np.testing.assert_array_equal(np.asarray(images[index]),reference[index+3][1])
        assert alignment[index]['source_frame_index']==index
        assert alignment[index]['video_ordinal']==index+3
        assert alignment[index]['ordinal_mapping']=='explicit_source_frame_index'


@pytest.mark.parametrize('change',[
    {'source_frame_index':[3,4,6]}, {'source_frame_index':[3,3,4]},
    {'source_frame_index':[3.,4.,5.]}, {'source_frame_index':[0,1,2]},
    {'frame_index':[0,2,3]}, {'source_frame_index':None},
])
def test_mecka_mapping_rejects_missing_or_inconsistent_source_rows(change):
    from openwam.dataloader.video_alignment import mecka_video_ordinals
    columns=dict(frame_index=[0,1,2],source_frame_index=[3,4,5]);columns.update(change)
    with pytest.raises(ValueError,match='Mecka'):
        mecka_video_ordinals(columns,dict(length=3,source_frame_offset=3))


def test_mecka_mapping_uses_verified_source_frame_indices():
    from openwam.dataloader.video_alignment import mecka_video_ordinals
    result=mecka_video_ordinals(dict(frame_index=[0,1,2],source_frame_index=[3,4,5]),
        dict(length=3,source_frame_offset=3))
    assert result.tolist()==[3,4,5]


def test_sequential_history_rejects_ambiguous_offset_even_without_past_slots(tmp_path):
    from openwam.dataloader.head_history import HistorySamplingConfig,materialize_head_history
    (tmp_path/'video.mp4').touch()
    ep=dict(episode_index=0,length=3,**{'videos/head/chunk_index':0,
        'videos/head/file_index':0,'videos/head/from_timestamp':1e-7})
    with pytest.raises(ValueError,match='nonzero time offset'):
        materialize_head_history(tmp_path,dict(fps=30,video_path='video.mp4'),ep,'head',0,0,
            HistorySamplingConfig(enabled=True),np.random.default_rng(0),False,
            sequential=True,video_ordinals=np.arange(3)+3)


def test_mecka_current_goal_history_and_stage2_share_raw_video_mapping(tmp_path):
    import json
    import pyarrow as pa
    import pyarrow.parquet as pq
    import yaml
    from tests.goal.test_curated_goal import fixture
    from openwam.dataloader.curated_goal import CuratedGoalDataset
    from openwam.dataloader.curated_stage2 import CuratedStage2Dataset

    config,root=fixture(tmp_path)
    info=json.loads((root/'meta/info.json').read_text());info['fps']=2
    (root/'meta/info.json').write_text(json.dumps(info))
    ep_path=root/'meta/episodes/chunk-000/file-000.parquet'
    ep=pq.read_table(ep_path).to_pylist()[0]
    ep.update(source_frame_offset=3,dataset_from_index=0)
    pq.write_table(pa.Table.from_pylist([ep]),ep_path)
    rows=pq.read_table(root/'data.parquet').to_pylist()
    for row in rows:row['source_frame_index']=row['frame_index']+3
    pq.write_table(pa.Table.from_pylist(rows),root/'data.parquet')
    selection=yaml.safe_load(config.read_text())
    selection['selection']['segments_by_repo']['mecka/fixture']=[[0,0,10]]
    config.write_text(yaml.safe_dump(selection))
    reference=encode_video(root/'video.mp4',count=16,fps=2)
    stage1=CuratedGoalDataset(config,tmp_path/'cache',history={'enabled':True})
    stage1._evaluating=True
    repo=stage1.repos[0];m=stage1._episode(repo,0)
    record,batch=stage1._materialize(repo,0,m,m['tasks'][0],6,7,'task')
    traces=record['provenance']
    for name,index in [('observed_video',6),('goal_video',7)]:
        assert traces[name]['video_ordinal']==index+3
        assert traces[name]['source_rgb_sha256']==hashlib.sha256(reference[index+3][1].tobytes()).hexdigest()
    history=traces['head_history']
    assert history['indices']==[-1]*7+[0,2,4]
    assert history['timestamp_source']=='source_row_ordinal_over_nominal_fps'
    for index in (0,2,4):
        assert history['alignment'][index]['video_ordinal']==index+3
        assert history['alignment'][index]['source_rgb_sha256']==hashlib.sha256(reference[index+3][1].tobytes()).hexdigest()
    assert batch['history_valid'].sum()==3

    stage2=CuratedStage2Dataset(config)
    m2=stage2._episode(stage2.repos[0],0)
    assert m2['video_ordinals'].tolist()==list(range(3,13))
    views,trace=stage2._decode_views(m2,[0,4,8])
    camera='observation.images.egocentric'
    for view,index in zip(views,(0,4,8)):
        np.testing.assert_array_equal(np.asarray(view[camera]),reference[index+3][1])
        assert trace[camera]['alignment'][index]['video_ordinal']==index+3
