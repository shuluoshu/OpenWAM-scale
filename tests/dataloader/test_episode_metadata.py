from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Lock
import time

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest

from openwam.dataloader.native_stage2_io import read_episode_metadata as read_filtered_metadata


def read_episode_metadata(root,episode):
    from openwam.dataloader.episode_metadata import offline_metadata_cache
    with offline_metadata_cache():return read_filtered_metadata(root,episode)


def metadata_repo(tmp_path, name='repo', episodes=(0, 1)):
    root=tmp_path/name
    path=root/'meta/episodes/chunk-000/file-000.parquet'
    path.parent.mkdir(parents=True)
    pq.write_table(pa.table({'episode_index':episodes,'length':[10]*len(episodes),
                            'nested':[[1,2]]*len(episodes),'stats/ignore':[3]*len(episodes)}),path)
    return root,path


def test_metadata_reuses_one_repository_scan_and_returns_private_rows(tmp_path, monkeypatch):
    root,_=metadata_repo(tmp_path);original=ds.dataset;calls=[]
    def scan(*args,**kwargs):
        calls.append(args);return original(*args,**kwargs)
    monkeypatch.setattr(ds,'dataset',scan)
    first=read_episode_metadata(root,0);first['nested'].append(99)
    assert read_episode_metadata(root,1)==dict(episode_index=1,length=10,nested=[1,2])
    assert read_episode_metadata(root,0)==dict(episode_index=0,length=10,nested=[1,2])
    assert len(calls)==1


@pytest.mark.parametrize('change',['rewrite','add','remove'])
def test_metadata_cache_rejects_changed_file_stamp_or_file_list(tmp_path,change):
    root,path=metadata_repo(tmp_path)
    read_episode_metadata(root,0)
    if change=='rewrite':pq.write_table(pa.table({'episode_index':[0,1],'length':[11,11]}),path)
    elif change=='add':pq.write_table(pa.table({'episode_index':[2],'length':[10]}),path.with_name('file-001.parquet'))
    else:path.unlink()
    with pytest.raises(ValueError,match='metadata.*changed'):
        read_episode_metadata(root,0)


def test_metadata_initial_read_rejects_source_change(tmp_path,monkeypatch):
    root,path=metadata_repo(tmp_path);original=ds.dataset
    class ChangingDataset:
        def __init__(self,*args,**kwargs):
            self.dataset=original(*args,**kwargs);self.schema=self.dataset.schema
        def to_table(self,*args,**kwargs):
            table=self.dataset.to_table(*args,**kwargs)
            pq.write_table(pa.table({'episode_index':[0,1],'length':[12,12]}),path)
            return table
    monkeypatch.setattr(ds,'dataset',ChangingDataset)
    with pytest.raises(ValueError,match='metadata.*changed'):
        read_episode_metadata(root,0)


def test_metadata_concurrent_first_reads_share_scan(tmp_path,monkeypatch):
    root,_=metadata_repo(tmp_path);original=ds.dataset;calls=[];lock=Lock();barrier=Barrier(16)
    def scan(*args,**kwargs):
        with lock:calls.append(args)
        time.sleep(.02)
        return original(*args,**kwargs)
    def read(index):
        barrier.wait();return read_episode_metadata(root,index%2)
    monkeypatch.setattr(ds,'dataset',scan)
    with ThreadPoolExecutor(max_workers=16) as pool:rows=list(pool.map(read,range(16)))
    assert [row['episode_index'] for row in rows]==[i%2 for i in range(16)]
    assert len(calls)==1


def test_metadata_duplicate_rejection_only_affects_requested_episode(tmp_path):
    root,_=metadata_repo(tmp_path,episodes=(0,1,1))
    assert read_episode_metadata(root,0)['episode_index']==0
    for episode in (1,2):
        with pytest.raises(ValueError,match='nonunique episode metadata'):
            read_episode_metadata(root,episode)


def test_metadata_cache_evicts_repositories_to_both_limits(tmp_path,monkeypatch):
    from openwam.dataloader import episode_metadata
    cache=episode_metadata.EpisodeMetadataCache(max_repos=2,max_bytes=4096)
    monkeypatch.setattr(episode_metadata,'_CACHE',cache)
    for index in range(3):
        root,_=metadata_repo(tmp_path,name=str(index));read_episode_metadata(root,0)
    assert len(cache.entries)==2
    assert all(key[0]!=str(tmp_path/'0') for key in cache.entries)
    assert cache.bytes<=4096
    assert len(cache.inflight)==0


def test_metadata_oversize_uses_uncached_filtered_read(tmp_path,monkeypatch):
    from openwam.dataloader import episode_metadata
    cache=episode_metadata.EpisodeMetadataCache(max_repos=2,max_bytes=1)
    monkeypatch.setattr(episode_metadata,'_CACHE',cache)
    root,_=metadata_repo(tmp_path)
    assert read_episode_metadata(root,1)==dict(episode_index=1,length=10,nested=[1,2])
    assert not cache.entries and cache.bytes==0 and not cache.inflight


def test_metadata_blocks_and_byte_budget_evict_independently(tmp_path,monkeypatch):
    from openwam.dataloader import episode_metadata
    cache=episode_metadata.EpisodeMetadataCache(max_repos=8,max_bytes=4096)
    monkeypatch.setattr(episode_metadata,'_CACHE',cache)
    root,_=metadata_repo(tmp_path,episodes=(0,1024,2048))
    assert read_episode_metadata(root,0)['episode_index']==0
    one_block=cache.bytes;cache.max_bytes=one_block+one_block//2
    for episode in (1024,2048):
        assert read_episode_metadata(root,episode)['episode_index']==episode
        assert len(cache.entries)==1 and cache.bytes<=cache.max_bytes
    assert next(iter(cache.entries))==(str(root),2)


def test_metadata_new_block_still_rejects_changed_existing_repository(tmp_path):
    root,path=metadata_repo(tmp_path,episodes=(0,1024))
    read_episode_metadata(root,0)
    pq.write_table(pa.table({'episode_index':[0,1024],'length':[11,11]}),path)
    with pytest.raises(ValueError,match='metadata.*changed'):
        read_episode_metadata(root,1024)


def test_metadata_singleflight_failure_releases_all_readers(tmp_path,monkeypatch):
    from openwam.dataloader import episode_metadata
    cache=episode_metadata.EpisodeMetadataCache()
    monkeypatch.setattr(episode_metadata,'_CACHE',cache)
    root,_=metadata_repo(tmp_path);barrier=Barrier(8)
    def fail(*args,**kwargs):
        time.sleep(.02);raise OSError('source read failed')
    def read(index):
        barrier.wait()
        with pytest.raises(OSError,match='source read failed'):read_episode_metadata(root,index%2)
    monkeypatch.setattr(cache,'_load',fail)
    with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(read,range(8)))
    assert not cache.entries and not cache.inflight and cache.bytes==0


def test_regular_metadata_reads_stay_filtered_without_repository_cache(tmp_path,monkeypatch):
    root,path=metadata_repo(tmp_path);original=ds.dataset;filters=[]
    class RecordingDataset:
        def __init__(self,*args,**kwargs):
            self.dataset=original(*args,**kwargs);self.schema=self.dataset.schema
        def to_table(self,*args,**kwargs):
            filters.append(str(kwargs['filter']));return self.dataset.to_table(*args,**kwargs)
    monkeypatch.setattr(ds,'dataset',RecordingDataset)
    assert read_filtered_metadata(root,0)['length']==10
    pq.write_table(pa.table({'episode_index':[0,1],'length':[11,11]}),path)
    assert read_filtered_metadata(root,0)['length']==11
    assert filters==['(episode_index == 0)','(episode_index == 0)']


def test_offline_cache_opt_in_is_thread_local_and_restores_on_exception(tmp_path,monkeypatch):
    from openwam.dataloader.episode_metadata import offline_metadata_cache
    root,_=metadata_repo(tmp_path);original=ds.dataset;calls=[]
    def scan(*args,**kwargs):
        calls.append(args);return original(*args,**kwargs)
    monkeypatch.setattr(ds,'dataset',scan)
    with pytest.raises(RuntimeError,match='leave context'):
        with offline_metadata_cache():
            read_filtered_metadata(root,0)
            with offline_metadata_cache():read_filtered_metadata(root,1)
            with ThreadPoolExecutor(max_workers=1) as pool:
                assert pool.submit(read_filtered_metadata,root,0).result()['episode_index']==0
            read_filtered_metadata(root,0)
            raise RuntimeError('leave context')
    read_filtered_metadata(root,0)
    assert len(calls)==3  # One cached scan, other thread's read, then restored read.
