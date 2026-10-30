"""Bounded metadata blocks with source checks on every cache access."""
from collections import OrderedDict
from concurrent.futures import Future
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import sys
from threading import Lock,local


def _snapshot(root):
    rows=[]
    for path in sorted((root/'meta/episodes').glob('chunk-*/*.parquet')):
        try:stamp=path.stat()
        except FileNotFoundError as exc:
            raise ValueError(f'{root}: episode metadata source changed during scan') from exc
        rows.append((str(path),stamp.st_size,stamp.st_mtime_ns))
    return tuple(rows)


class EpisodeMetadataCache:
    """At most eight 1,024-episode blocks, shared by metadata reader threads.

    Bounding blocks also bounds the number of retained repositories. A block
    larger than the byte budget uses the original filtered read without caching.
    Arrow tables stay private and every returned row is an independent copy.
    """
    def __init__(self,max_repos=8,max_bytes=256*1024*1024):
        if min(max_repos,max_bytes)<1:raise ValueError('metadata cache limits must be positive')
        self.max_repos=max_repos;self.max_bytes=max_bytes
        self.entries=OrderedDict();self.inflight={};self.bytes=0;self.lock=Lock()

    def _dataset(self,stamp):
        import pyarrow.dataset as ds
        dataset=ds.dataset([path for path,_,_ in stamp],format='parquet')
        return dataset,[name for name in dataset.schema.names if not name.startswith('stats/')]

    def _load(self,root,block,stamp):
        import pyarrow.dataset as ds
        dataset,columns=self._dataset(stamp)
        start=block*1024
        table=dataset.to_table(columns=columns,filter=(ds.field('episode_index')>=start)&(ds.field('episode_index')<start+1024))
        if _snapshot(root)!=stamp:raise ValueError(f'{root}: episode metadata source changed during read')
        locations={}
        for index,episode in enumerate(table['episode_index'].to_pylist()):
            locations[episode]=-1 if episode in locations else index
        size=table.get_total_buffer_size()+sys.getsizeof(locations)
        size+=sum(sys.getsizeof(k)+sys.getsizeof(v) for k,v in locations.items())
        size+=sys.getsizeof(stamp)+sum(sys.getsizeof(row)+sum(sys.getsizeof(v) for v in row) for row in stamp)
        if size>self.max_bytes:return None
        return stamp,table,locations,size

    def _uncached(self,root,episode,stamp):
        import pyarrow.dataset as ds
        dataset,columns=self._dataset(stamp)
        table=dataset.to_table(columns=columns,filter=ds.field('episode_index')==episode)
        if _snapshot(root)!=stamp:raise ValueError(f'{root}: episode metadata source changed during read')
        if len(table)!=1:raise ValueError(f'{root}:{episode}: nonunique episode metadata')
        return deepcopy(table.to_pylist()[0])

    def read(self,root,episode):
        root=Path(root);stamp=_snapshot(root);key=(str(root),int(episode)//1024)
        with self.lock:
            # A new block must agree with any other retained block of this root.
            previous=[value[0] for k,value in self.entries.items() if k[0]==key[0]]
            previous.extend(value[0] for k,value in self.inflight.items() if k[0]==key[0])
            if any(value!=stamp for value in previous):
                raise ValueError(f'{root}: episode metadata source changed')
            if not stamp:raise ValueError(f'{root}:{episode}: missing episode metadata')
            entry=self.entries.get(key)
            if entry is not None:self.entries.move_to_end(key)
            else:
                owner=key not in self.inflight
                if owner:self.inflight[key]=(stamp,Future())
                future=self.inflight[key][1]
        if entry is None:
            if owner:
                try:
                    entry=self._load(root,key[1],stamp)
                    with self.lock:
                        if entry is not None:
                            while self.entries and (len(self.entries)>=self.max_repos or self.bytes+entry[3]>self.max_bytes):
                                _,old=self.entries.popitem(last=False);self.bytes-=old[3]
                            self.entries[key]=entry;self.bytes+=entry[3]
                        future.set_result(entry);del self.inflight[key]
                except BaseException as exc:
                    with self.lock:
                        future.set_exception(exc);del self.inflight[key]
                    raise
            else:entry=future.result()
        if entry is None:return self._uncached(root,episode,stamp)
        index=entry[2].get(episode,-1)
        if index<0:raise ValueError(f'{root}:{episode}: nonunique episode metadata')
        return deepcopy(entry[1].slice(index,1).to_pylist()[0])


_CACHE=EpisodeMetadataCache()
_LOCAL=local()


@contextmanager
def offline_metadata_cache():
    """Enable block reuse only for this thread's offline preparation scope."""
    previous=getattr(_LOCAL,'enabled',False);_LOCAL.enabled=True
    try:yield
    finally:_LOCAL.enabled=previous


def read_episode_metadata(root,episode):
    if getattr(_LOCAL,'enabled',False):return _CACHE.read(root,episode)
    # Random training accesses retain the original single-episode projection.
    import pyarrow.dataset as ds
    root=Path(root);files=sorted((root/'meta/episodes').glob('chunk-*/*.parquet'))
    if not files:raise ValueError(f'{root}:{episode}: missing episode metadata')
    dataset=ds.dataset([str(path) for path in files],format='parquet')
    columns=[name for name in dataset.schema.names if not name.startswith('stats/')]
    table=dataset.to_table(columns=columns,filter=ds.field('episode_index')==episode)
    if len(table)!=1:raise ValueError(f'{root}:{episode}: nonunique episode metadata')
    return table.to_pylist()[0]
