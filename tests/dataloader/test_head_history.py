import numpy as np
import pytest
import torch
from openwam.dataloader.head_history import HistorySamplingConfig, select_history_frames, materialize_head_history


def test_fixed_gaps_oldest_first_padding_and_real_ages():
    indices, ages, valid = select_history_frames(35, 10, 10, HistorySamplingConfig(enabled=True), np.random.default_rng(1), False)
    assert indices.tolist() == [-1]*8+[15,25]
    assert valid.tolist() == [False]*8+[True,True]
    np.testing.assert_array_equal(ages, [0]*8+[2.,1.])
    assert ages.dtype == np.float32


def test_random_cumulative_gaps_reproducible_and_variable():
    cfg = HistorySamplingConfig(enabled=True,jitter_fraction=.5)
    a=select_history_frames(1000,30,0,cfg,np.random.default_rng(7),True)
    b=select_history_frames(1000,30,0,cfg,np.random.default_rng(7),True)
    c=select_history_frames(1000,30,0,cfg,np.random.default_rng(8),True)
    np.testing.assert_array_equal(a[0],b[0])
    assert not np.array_equal(a[0],c[0])
    gaps=np.diff(np.r_[a[0],1000])/30
    assert (gaps>=.5-1/30).all() and (gaps<=1.5+1/30).all()
    assert np.unique(gaps).size>1
    np.testing.assert_allclose(a[1],(1000-a[0])/30)


def test_low_fps_duplicate_slots_masked_and_current_excluded():
    indices,ages,valid=select_history_frames(3,.2,0,HistorySamplingConfig(enabled=True),np.random.default_rng(0),False)
    chosen=indices[valid]
    assert len(set(chosen))==len(chosen)
    assert ((chosen>=0)&(chosen<3)).all()
    assert (indices[~valid]==-1).all() and (ages[~valid]==0).all()


def test_disabled_has_no_fields_or_decoding():
    fields,trace=materialize_head_history(None,None,None,None,0,0,HistorySamplingConfig(),np.random.default_rng(0),False)
    assert fields=={} and trace=={}


def test_materializer_real_decode_offset_and_clean_resize(tmp_path):
    av=pytest.importorskip('av')
    path=tmp_path/'head.mp4'
    with av.open(str(path),'w') as container:
        stream=container.add_stream('mpeg4',rate=10);stream.width=32;stream.height=24;stream.pix_fmt='yuv420p'
        for index in range(50):
            frame=av.VideoFrame.from_ndarray(np.full((24,32,3),index*4,np.uint8),format='rgb24')
            for packet in stream.encode(frame):container.mux(packet)
        for packet in stream.encode():container.mux(packet)
    info=dict(fps=10,video_path='head.mp4');head='head';prefix='videos/head/'
    ep=dict(episode_index=0,length=30,**{prefix+'chunk_index':0,prefix+'file_index':0,prefix+'from_timestamp':2.,prefix+'to_timestamp':5.})
    fields,trace=materialize_head_history(tmp_path,info,ep,head,25,0,HistorySamplingConfig(enabled=True),np.random.default_rng(0),False)
    assert fields['history_images'].shape==(10,3,1,256,320)
    assert fields['history_ages'].dtype==torch.float32
    assert fields['history_valid'].dtype==torch.bool
    assert trace['indices']==[-1]*8+[5,15]
    assert trace['timestamp_source']=='validated_video_cfr_ordinal_over_fps'
    assert fields['history_images'][-1].mean()>fields['history_images'][-2].mean()
    assert fields['history_images'][:8].count_nonzero()==0
    ep[prefix+'from_timestamp']=float('nan')
    with pytest.raises(ValueError,match='metadata'):
        materialize_head_history(tmp_path,info,ep,head,25,0,HistorySamplingConfig(enabled=True),np.random.default_rng(0),False)


def test_materializer_checks_nonintegral_offset_even_without_past_slots(tmp_path):
    (tmp_path/'head.mp4').touch()
    ep={'episode_index':0,'length':10,'videos/head/chunk_index':0,'videos/head/file_index':0,'videos/head/from_timestamp':.15}
    with pytest.raises(ValueError,match='nonintegral video offset'):
        materialize_head_history(tmp_path,{'fps':10,'video_path':'head.mp4'},ep,'head',0,0,
            HistorySamplingConfig(enabled=True),np.random.default_rng(0),False)


def test_materializer_masks_source_invalid_before_decode(tmp_path,monkeypatch):
    from PIL import Image
    from openwam.dataloader.native_stage2_io import source_frame_validity
    (tmp_path/'head.mp4').touch()
    ep={'episode_index':0,'length':22,'videos/head/chunk_index':0,'videos/head/file_index':0,'videos/head/from_timestamp':0.}
    columns={'canonical.frame_valid':np.ones(22),'training.valid':np.ones(22)}
    columns['canonical.frame_valid'][19]=0
    columns['training.valid'][18]=np.nan
    decoded=[]
    def decode(path,offset,fps,indices,**kwargs):
        decoded.extend(indices)
        return {i:Image.new('RGB',(32,24)) for i in indices},{}
    monkeypatch.setattr('scripts.data.prepare_subtask_goals.decode_frames',decode)
    fields,trace=materialize_head_history(tmp_path,{'fps':1,'video_path':'head.mp4'},ep,'head',20,12,
        HistorySamplingConfig(enabled=True),np.random.default_rng(0),False,
        source_valid=source_frame_validity(columns,22))
    assert decoded==list(range(12,18))
    assert trace['indices']==[-1,-1,12,13,14,15,16,17,-1,-1]
    assert fields['history_valid'].tolist()==[False,False]+[True]*6+[False,False]
    assert fields['history_ages'][-2:].count_nonzero()==0
    assert fields['history_images'][-2:].count_nonzero()==0


def test_episode_boundary_in_shared_video_never_reuses_previous_history(tmp_path):
    av=pytest.importorskip('av')
    path=tmp_path/'head.mp4'
    with av.open(str(path),'w') as container:
        stream=container.add_stream('mpeg4',rate=1)
        stream.width=32; stream.height=24; stream.pix_fmt='yuv420p'
        for index in range(32):
            rgb=np.zeros((24,32,3),np.uint8)
            rgb[:,:,(0 if index<16 else 2)]=240
            frame=av.VideoFrame.from_ndarray(rgb,format='rgb24')
            for packet in stream.encode(frame): container.mux(packet)
        for packet in stream.encode(): container.mux(packet)
    info=dict(fps=1,video_path='head.mp4')
    cfg=HistorySamplingConfig(enabled=True)
    def read(episode,anchor):
        ep=dict(episode_index=episode,length=16,**{'videos/head/chunk_index':0,
            'videos/head/file_index':0,'videos/head/from_timestamp':episode*16.,
            'videos/head/to_timestamp':(episode+1)*16.})
        return materialize_head_history(tmp_path,info,ep,'head',anchor,0,cfg,np.random.default_rng(0),False)
    old,_=read(0,12)
    assert old['history_valid'].all()
    first,trace=read(1,0)
    assert first['history_episode_id'].dtype==torch.int64
    assert first['history_episode_id']!=old['history_episode_id']
    assert trace['indices']==[-1]*10
    assert not first['history_valid'].any()
    assert first['history_images'].count_nonzero()==0
    partial,trace=read(1,3)
    assert partial['history_episode_id']==first['history_episode_id']
    assert trace['indices']==[-1]*7+[0,1,2]
    assert partial['history_images'][:7].count_nonzero()==0
    # Episode A is red; B is blue despite sharing the same video file.
    assert old['history_images'][:,0].mean()>old['history_images'][:,2].mean()+1
    assert partial['history_images'][-3:,2].mean()>partial['history_images'][-3:,0].mean()+1
