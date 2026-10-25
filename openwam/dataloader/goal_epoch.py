"""Persistent finite goal population, represented by compact source-grid runs.

Grid k selects floor(k * source_fps / sample_fps), aligned to episode zero.
Sources slower than the requested rate retain every source frame exactly once.
Metadata readers and pending compact results are bounded. Ordered transaction
batches preserve global IDs and resume from the last durable episode prefix.
"""
from __future__ import annotations
from fractions import Fraction
from collections import OrderedDict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import closing
import copy
import hashlib
import json
import math
import numbers
import os
from pathlib import Path
import sqlite3
import threading
import time
import numpy as np

# v3 preserves adjacent same-ID annotation occurrences and their goal frames.
# Existing v2 populations must be rebuilt, including partially prepared indices.
VERSION=3


def epoch_sample_fps(config):
    """Current-frame sampling only; historical-frame and Stage2 rates are separate."""
    value=config.get('sample_fps',15.0)
    if isinstance(value,bool) or not isinstance(value,numbers.Real) or not math.isfinite(value) or value<=0:
        raise ValueError('sample_fps must be a positive finite number')
    return float(value)


def sha256_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def _json(value):return json.dumps(value,sort_keys=True,separators=(',',':'))


def _ordered_parallel(function,items,workers):
    """Consume bounded futures in input order; the iterator stays on its owner thread."""
    if isinstance(workers,bool) or not isinstance(workers,int) or workers<1:
        raise ValueError('workers must be a positive integer')
    if workers==1:
        for item in items:yield function(item)
        return
    executor=ThreadPoolExecutor(max_workers=workers,thread_name_prefix='epoch-validate')
    pending=deque();items=iter(items)
    try:
        for _ in range(2*workers):
            item=next(items,None)
            if item is None:break
            pending.append(executor.submit(function,item))
        while pending:
            yield pending.popleft().result()
            item=next(items,None)
            if item is not None:pending.append(executor.submit(function,item))
    finally:
        for future in pending:future.cancel()
        executor.shutdown(wait=True,cancel_futures=True)


def _warm_index(path):
    """Sequential readahead prevents cold CPFS SQLite scans issuing tiny reads."""
    if Path(path).is_file():
        with Path(path).open('rb') as stream:
            while stream.read(8*1024*1024):pass


def _selected_repos_digest(dataset):
    digest=hashlib.sha256()
    for repo in dataset.repos:digest.update(_json(repo).encode())
    return digest.hexdigest()


def _identity(dataset,*,workers=32):
    source=sha256_file(dataset.selection['source_yaml'])
    digest=hashlib.sha256(source.encode())
    sidecars={};lock=threading.Lock()
    def read_repo(repo):
        values=[]
        paths=[Path(repo['path'])/'meta'/name for name in ('info.json','tasks.parquet','sub_tasks.parquet')]
        annotation=repo.get('native_metadata',{}).get('annotation_file')
        if annotation:paths.append(Path(annotation))
        for path in paths:
            key=str(path.resolve())
            with lock:
                owner=key not in sidecars
                if owner:sidecars[key]=Future()
                future=sidecars[key]
            if owner:
                try:future.set_result(sha256_file(path) if path.is_file() else 'MISSING')
                except BaseException as exc:
                    future.set_exception(exc)
                    raise
            values.append([key,future.result()])
        return _json(repo),values
    for repo,values in _ordered_parallel(read_repo,dataset.repos,workers):
        digest.update(repo.encode())
        for value in values:digest.update(_json(value).encode())
    return {'version':VERSION,'source_yaml_sha256':source,'selection_sha256':digest.hexdigest()}


def _episode_digest(metadata):
    """Content hash of the projected population inputs, independent of file stats."""
    digest=hashlib.sha256()
    for key in ('length','info','ep','tasks','subtasks'):
        digest.update(_json([key,metadata.get(key)]).encode())
    valid=np.asarray(metadata.get('source_valid',metadata.get('history_source_valid',
        np.ones(metadata['length'],bool))),dtype=bool)
    digest.update(valid.tobytes())
    digest.update(str(metadata.get('data','')).encode())
    return digest.hexdigest()


def _file_stamp(path):
    try:
        st=Path(path).stat();return st.st_size,st.st_mtime_ns
    except FileNotFoundError:return -1,-1


def _episode_sources(metadata):
    paths=set()
    if metadata.get('root') is not None:
        root=Path(metadata['root'])
        paths.update(root/'meta'/name for name in ('info.json','tasks.parquet','sub_tasks.parquet'))
        paths.update((root/'meta/episodes').glob('chunk-*/*.parquet'))
    for key in ('data','annotation_file'):
        if metadata.get(key):paths.add(Path(metadata[key]))
    paths.update(Path(path) for path in metadata.get('data_files',()))
    return [(str(p.resolve()),*_file_stamp(p)) for p in sorted(paths)]


def _check_source_rows(rows,*,workers=32):
    def check(row):
        path,size,mtime=row
        if _file_stamp(path)!=(size,mtime):raise ValueError(f'epoch index source dependency changed: {path}')
    for _ in _ordered_parallel(check,rows,workers):pass


def _check_sources(connection,repo_id=None,episode=None,*,workers=32):
    if repo_id is None:rows=connection.execute('SELECT path,size,mtime_ns FROM sources')
    else:rows=connection.execute('SELECT s.path,s.size,s.mtime_ns FROM sources s JOIN episode_sources e ON e.path=s.path WHERE e.repo_id=? AND e.episode=?',(repo_id,episode))
    _check_source_rows(rows,workers=workers if repo_id is None else 1)


def _record_sources(connection,repo_id,episode,sources,pending_sources):
    for path,size,mtime in sources:
        existing=connection.execute('SELECT size,mtime_ns FROM sources WHERE path=?',(path,)).fetchone()
        if existing is not None and existing!=(size,mtime):raise ValueError(f'epoch index source dependency changed during build: {path}')
        connection.execute('INSERT OR IGNORE INTO sources VALUES (?,?,?)',(path,size,mtime))
        connection.execute('INSERT INTO episode_sources VALUES (?,?,?)',(repo_id,episode,path))
        # The writer keeps only this transaction's unique dependencies. Files are
        # rechecked concurrently immediately before the batch becomes durable.
        pending_sources[path]=(size,mtime)


def _metadata(connection):return {k:json.loads(v) for k,v in connection.execute('SELECT key,value FROM metadata')}


def _set(connection,key,value):connection.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',(key,_json(value)))


def _check_identity(dataset,metadata):
    if any(metadata.get(k)!=v for k,v in _identity(dataset).items()):raise ValueError('epoch index source identity mismatch')


def validate_index(dataset,index_path,*,sample_fps=None):
    """Validate identity/completion and interval coverage; never decode images."""
    path=Path(index_path).resolve()
    _warm_index(path)
    with closing(sqlite3.connect(f'{path.as_uri()}?mode=ro',uri=True)) as c:
        m=_metadata(c);_check_identity(dataset,m)
        if sample_fps is not None and m.get('sample_fps')!=epoch_sample_fps({'sample_fps':sample_fps}):
            raise ValueError('epoch index sample_fps identity mismatch')
        if not m.get('complete'):raise ValueError('epoch index incomplete')
        _check_sources(c)
        expected=sum(len({e for e,_,_ in r['ranges']}) for r in dataset.repos)
        if c.execute('SELECT COUNT(*) FROM episodes').fetchone()[0]!=expected:raise ValueError('epoch index incomplete episodes')
        last=0
        for start,count in c.execute('SELECT sample_start,count FROM runs ORDER BY sample_start'):
            if start!=last or count<=0:raise ValueError('epoch index corrupt sample coverage')
            last+=count
        if last!=m['total_samples']:raise ValueError('epoch index corrupt sample count')
        m['samples_by_level']={level:count for level,count in c.execute('SELECT level,SUM(count) FROM runs GROUP BY level')}
        m['completed_episodes']=expected
        m['run_count']=c.execute('SELECT COUNT(*) FROM runs').fetchone()[0]
    return dict(m,index_sha256=sha256_file(path),index_path=str(path),selected_repos_sha256=_selected_repos_digest(dataset))


def _prepare_episode(dataset,item,sample_fps):
    ri,repo,episode,_=item
    try:return _prepare_episode_result(dataset,item,sample_fps)
    except Exception as exc:
        message=f'epoch source repo_id={ri} repo={repo["repo"]} episode={episode}: {exc}'
        if isinstance(exc,ValueError):raise ValueError(message) from exc
        if isinstance(exc,OSError):raise OSError(message) from exc
        raise RuntimeError(message) from exc


def _prepare_episode_result(dataset,item,sample_fps):
    ri,repo,episode,ranges=item
    metadata=dataset._episode(repo,episode);n=int(metadata['length']);fps=float(metadata['info']['fps'])
    sources=_episode_sources(metadata)
    if not math.isfinite(fps) or fps<=0:raise ValueError('invalid source fps')
    ratio=max(Fraction(str(fps))/Fraction(str(sample_fps)),Fraction(1))
    num,den=ratio.numerator,ratio.denominator
    valid=np.asarray(metadata.get('source_valid',metadata.get('history_source_valid',np.ones(n,bool))),dtype=bool)
    if valid.shape!=(n,):raise ValueError('invalid source validity shape')
    episode_excluded=0;runs=[]
    for level in ('task','subtask'):
        for segment in metadata['tasks' if level=='task' else 'subtasks']:
            goal=segment['end_exclusive']-1
            for e,a,b in ranges:
                if not 0<=a<b<=n:raise ValueError('curated range exceeds original episode')
                start=max(a,segment['start']);end=min(b,goal)
                if start>=end:continue
                lo=(start*den+num-1)//num;hi=(end*den+num-1)//num
                # Only invalid source indices expand; the result retains compact runs.
                invalid=np.flatnonzero(~valid[start:end])+start
                run_start=lo
                for frame in invalid:
                    k=(int(frame)*den+num-1)//num
                    if k>=hi or k*num//den!=frame:continue
                    episode_excluded+=1
                    if k>run_start:
                        runs.append((k-run_start,ri,episode,level,_json(segment),run_start,num,den))
                    run_start=k+1
                if run_start<hi:
                    runs.append((hi-run_start,ri,episode,level,_json(segment),run_start,num,den))
    return ri,episode,runs,episode_excluded,_episode_digest(metadata),sources


def _pending_episodes(connection,dataset):
    for ri,repo in enumerate(dataset.repos):
        completed={row[0] for row in connection.execute('SELECT episode FROM episodes WHERE repo_id=?',(ri,))}
        ranges_by_episode={}
        for item in sorted(repo['ranges']):ranges_by_episode.setdefault(item[0],[]).append(item)
        for episode,ranges in ranges_by_episode.items():
            if episode not in completed:yield ri,repo,episode,ranges


def _prepared_episodes(dataset,items,sample_fps,workers,max_pending):
    if workers==1:
        for item in items:yield _prepare_episode(dataset,item,sample_fps)
        return
    local=threading.local()
    def prepare(item):
        if not hasattr(local,'dataset'):
            # Selection/configuration are read-only; CuratedGoalDataset's mutable
            # metadata LRU belongs exclusively to this worker and retains one episode.
            local.dataset=copy.copy(dataset)
            local.dataset._episodes=OrderedDict()
            local.dataset.max_cached_episodes=1
        return _prepare_episode(local.dataset,item,sample_fps)
    executor=ThreadPoolExecutor(max_workers=workers,thread_name_prefix='epoch-index')
    pending=deque();items=iter(items)
    try:
        for _ in range(max_pending):
            item=next(items,None)
            if item is None:break
            pending.append(executor.submit(prepare,item))
        while pending:
            # Consume selection order, regardless of worker completion order.
            yield pending.popleft().result()
            item=next(items,None)
            if item is not None:pending.append(executor.submit(prepare,item))
    finally:
        for future in pending:future.cancel()
        executor.shutdown(wait=True,cancel_futures=True)


def _write_progress(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(f'{path.name}.tmp.{os.getpid()}')
    try:
        with temporary.open('w') as stream:
            stream.write(_json(value)+'\n');stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
        descriptor=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def build_epoch_index(dataset,index_path,*,sample_fps=15.0,workers=1,max_pending=None,
                      commit_every=1,progress_path=None):
    """Prepare v3 rows in order, optionally reading metadata concurrently.

    Each commit contains complete episodes and their source stamps. An exception
    rolls back the current batch; at most ``commit_every - 1`` completed episodes
    need rereading after failure. Defaults preserve serial, per-episode commits.
    """
    sample_fps=epoch_sample_fps({'sample_fps':sample_fps})
    if max_pending is None:max_pending=workers*2
    for name,value in (('workers',workers),('max_pending',max_pending),('commit_every',commit_every)):
        if isinstance(value,bool) or not isinstance(value,int) or value<1:
            raise ValueError(f'{name} must be a positive integer')
    path=Path(index_path).resolve();path.parent.mkdir(parents=True,exist_ok=True)
    progress_path=Path(progress_path) if progress_path is not None else Path(str(path)+'.progress.json')
    if progress_path.resolve()==path:raise ValueError('progress_path must differ from index_path')
    started=time.monotonic()
    expected=sum(len({e for e,_,_ in repo['ranges']}) for repo in dataset.repos)
    durable=dict(completed_episodes=0,total_samples=0,excluded_invalid_anchor_pairs=0)
    resumed=0;commits=0
    def progress(status,**extra):
        elapsed=time.monotonic()-started;new=durable['completed_episodes']-resumed
        _write_progress(progress_path,dict(index_path=str(path),version=VERSION,status=status,
            total_episodes=expected,resumed_episodes=resumed,newly_completed_episodes=new,
            elapsed_seconds=elapsed,episodes_per_second=new/elapsed if elapsed else 0.,
            updated_at_unix=time.time(),workers=workers,max_pending=max_pending,
            commit_every=commit_every,commits=commits,**durable,**extra))
    progress('checking_sources')
    try:
        _warm_index(path)
        with closing(sqlite3.connect(path)) as c:
            c.executescript('''CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
              CREATE TABLE IF NOT EXISTS episodes(repo_id INTEGER,episode INTEGER,excluded INTEGER,digest TEXT,PRIMARY KEY(repo_id,episode));
              CREATE TABLE IF NOT EXISTS sources(path TEXT PRIMARY KEY,size INTEGER,mtime_ns INTEGER);
              CREATE TABLE IF NOT EXISTS episode_sources(repo_id INTEGER,episode INTEGER,path TEXT,PRIMARY KEY(repo_id,episode,path));
              CREATE TABLE IF NOT EXISTS runs(sample_start INTEGER PRIMARY KEY,count INTEGER NOT NULL,repo_id INTEGER,
              episode INTEGER,level TEXT,segment TEXT,grid_start INTEGER,numerator INTEGER,denominator INTEGER);''')
            m=_metadata(c)
            resumed=c.execute('SELECT COUNT(*) FROM episodes').fetchone()[0]
            total=c.execute('SELECT COALESCE(MAX(sample_start+count),0) FROM runs').fetchone()[0]
            excluded=c.execute('SELECT COALESCE(SUM(excluded),0) FROM episodes').fetchone()[0]
            durable.update(completed_episodes=resumed,total_samples=total,excluded_invalid_anchor_pairs=excluded)
            progress('checking_sources')
            if m:
                _check_identity(dataset,m)
                if m['sample_fps']!=sample_fps:raise ValueError('epoch index sample_fps identity mismatch')
                _check_sources(c)
            else:
                for k,v in dict(_identity(dataset),sample_fps=sample_fps,complete=False,total_samples=0,
                    grid_policy='episode-zero floor(k*source_fps/sample_fps); slower sources retain all frames',
                    excluded_invalid_anchor_pairs=0).items():_set(c,k,v)
                c.commit()
            progress('preparing')
            if not m.get('complete',False):
                completed=resumed;uncommitted=0;pending_sources={}
                prepared=_prepared_episodes(dataset,_pending_episodes(c,dataset),sample_fps,workers,max_pending)
                try:
                    for ri,episode,runs,episode_excluded,digest,sources in prepared:
                        # All writes remain on this connection's owning thread.
                        for run in runs:
                            c.execute('INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?)',(total,*run))
                            total+=run[0]
                        _record_sources(c,ri,episode,sources,pending_sources)
                        c.execute('INSERT INTO episodes VALUES (?,?,?,?)',(ri,episode,episode_excluded,digest))
                        excluded+=episode_excluded;completed+=1;uncommitted+=1
                        if uncommitted>=commit_every:
                            _check_source_rows((path,*stamp) for path,stamp in pending_sources.items())
                            _set(c,'total_samples',total);_set(c,'excluded_invalid_anchor_pairs',excluded)
                            c.commit();commits+=1;uncommitted=0;pending_sources.clear()
                            durable.update(completed_episodes=completed,total_samples=total,excluded_invalid_anchor_pairs=excluded)
                            progress('preparing')
                    if pending_sources:_check_source_rows((path,*stamp) for path,stamp in pending_sources.items())
                    _set(c,'total_samples',total);_set(c,'excluded_invalid_anchor_pairs',excluded)
                    _set(c,'complete',True);c.commit()
                    if uncommitted:commits+=1
                    durable.update(completed_episodes=completed,total_samples=total,excluded_invalid_anchor_pairs=excluded)
                except BaseException:
                    c.rollback()
                    raise
                finally:prepared.close()
            progress('validating')
        summary=validate_index(dataset,path)
        progress('complete')
        return summary
    except BaseException as exc:
        progress('failed',error=f'{type(exc).__name__}: {exc}')
        raise


class FiniteGoalDataset:
    def __init__(self,dataset,index_path,*,sample_fps=None,_validated_summary=None):
        self.dataset=dataset;self.index_path=Path(index_path).resolve()
        if _validated_summary is None:self.summary=validate_index(dataset,self.index_path,sample_fps=sample_fps)
        else:
            # Only for a summary freshly validated and broadcast by the launcher's
            # rank zero. Every rank still verifies its bytes/configuration locally.
            summary=dict(_validated_summary)
            total=summary.get('total_samples')
            if (summary.get('index_path')!=str(self.index_path) or summary.get('version')!=VERSION
                or summary.get('complete') is not True or isinstance(total,bool) or not isinstance(total,int) or total<=0
                or (sample_fps is not None and summary.get('sample_fps')!=epoch_sample_fps({'sample_fps':sample_fps}))
                or summary.get('selected_repos_sha256')!=_selected_repos_digest(dataset)
                or summary.get('source_yaml_sha256')!=sha256_file(dataset.selection['source_yaml'])
                or summary.get('index_sha256')!=sha256_file(self.index_path)):
                raise ValueError('epoch index validated summary identity mismatch')
            with closing(sqlite3.connect(f'{self.index_path.as_uri()}?mode=ro',uri=True)) as c:
                if any(summary.get(k)!=v for k,v in _metadata(c).items()):
                    raise ValueError('epoch index validated summary metadata mismatch')
            self.summary=summary
        self._connection=None;self._evaluations=None
        self._validated=OrderedDict()
    def __len__(self):return self.summary['total_samples']
    def _db(self):
        if self._connection is None:self._connection=sqlite3.connect(f'{self.index_path.as_uri()}?mode=ro',uri=True)
        return self._connection
    def sample(self,global_index):
        if isinstance(global_index,bool) or not isinstance(global_index,(int,np.integer)) or not 0<=global_index<len(self):raise IndexError(global_index)
        row=self._db().execute('SELECT * FROM runs WHERE sample_start<=? ORDER BY sample_start DESC LIMIT 1',(int(global_index),)).fetchone()
        start,count,ri,episode,level,segment,grid,num,den=row
        anchor=(grid+int(global_index)-start)*num//den;segment=json.loads(segment)
        repo=self.dataset.repos[ri];key=(ri,episode)
        _check_sources(self._db(),ri,episode)
        if key not in self._validated:
            # Discard metadata cached before index verification; reread projected inputs.
            if hasattr(self.dataset,'_episodes'):self.dataset._episodes.pop((repo['path'],episode),None)
            m=self.dataset._episode(repo,episode)
            expected=self._db().execute('SELECT digest FROM episodes WHERE repo_id=? AND episode=?',key).fetchone()[0]
            if _episode_digest(m)!=expected:raise ValueError(f'epoch index source labels changed: {repo["repo"]}:{episode}')
            self._validated[key]=m
            while len(self._validated)>getattr(self.dataset,'max_cached_episodes',8):self._validated.popitem(last=False)
        else:
            self._validated.move_to_end(key);m=self.dataset._episode(repo,episode)
            if self._validated[key] is not m:
                expected=self._db().execute('SELECT digest FROM episodes WHERE repo_id=? AND episode=?',key).fetchone()[0]
                if _episode_digest(m)!=expected:raise ValueError(f'epoch index source labels changed: {repo["repo"]}:{episode}')
                self._validated[key]=m
        _check_sources(self._db(),ri,episode)
        self.dataset._history_draw_id=int(global_index)
        return self.dataset._materialize(repo,episode,m,segment,anchor,segment['end_exclusive']-1,level)
    def evaluation_samples(self):
        if self._evaluations is None:
            ids=[];seen=set()
            for start,ri,level in self._db().execute('SELECT sample_start,repo_id,level FROM runs ORDER BY sample_start'):
                key=(self.dataset.repos[ri]['family'],level)
                if key not in seen:seen.add(key);ids.append(start)
            previous=self.dataset._evaluating;self.dataset._evaluating=True
            try:self._evaluations=[self.sample(i) for i in ids]
            finally:self.dataset._evaluating=previous
        return self._evaluations
    def scope_summary(self):return dict(self.dataset.scope_summary(),finite_epoch=self.summary)
