"""Persistent finite goal population, represented by compact source-grid runs.

Grid k selects floor(k * source_fps / sample_fps), aligned to episode zero.
Sources slower than the requested rate retain every source frame exactly once.
Only one episode's labels/validity are resident while building. Each episode is
committed atomically, so a failed build resumes without changing global IDs.
"""
from __future__ import annotations
from fractions import Fraction
from collections import OrderedDict
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import numpy as np

VERSION=2


def sha256_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def _json(value):return json.dumps(value,sort_keys=True,separators=(',',':'))


def _identity(dataset):
    source=sha256_file(dataset.selection['source_yaml'])
    digest=hashlib.sha256(source.encode())
    sidecars={}
    for repo in dataset.repos:
        digest.update(_json(repo).encode())
        paths=[Path(repo['path'])/'meta'/name for name in ('info.json','tasks.parquet','sub_tasks.parquet')]
        annotation=repo.get('native_metadata',{}).get('annotation_file')
        if annotation:paths.append(Path(annotation))
        for path in paths:
            key=str(path.resolve())
            if key not in sidecars:sidecars[key]=sha256_file(path) if path.is_file() else 'MISSING'
            digest.update(_json([key,sidecars[key]]).encode())
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
    return [(str(p.resolve()),*_file_stamp(p)) for p in sorted(paths)]


def _check_sources(connection,repo_id=None,episode=None):
    if repo_id is None:rows=connection.execute('SELECT path,size,mtime_ns FROM sources')
    else:rows=connection.execute('SELECT s.path,s.size,s.mtime_ns FROM sources s JOIN episode_sources e ON e.path=s.path WHERE e.repo_id=? AND e.episode=?',(repo_id,episode))
    for path,size,mtime in rows:
        if _file_stamp(path)!=(size,mtime):raise ValueError(f'epoch index source dependency changed: {path}')


def _record_sources(connection,repo_id,episode,metadata):
    for path,size,mtime in _episode_sources(metadata):
        existing=connection.execute('SELECT size,mtime_ns FROM sources WHERE path=?',(path,)).fetchone()
        if existing is not None and existing!=(size,mtime):raise ValueError(f'epoch index source dependency changed during build: {path}')
        connection.execute('INSERT OR IGNORE INTO sources VALUES (?,?,?)',(path,size,mtime))
        connection.execute('INSERT INTO episode_sources VALUES (?,?,?)',(repo_id,episode,path))


def _metadata(connection):return {k:json.loads(v) for k,v in connection.execute('SELECT key,value FROM metadata')}


def _set(connection,key,value):connection.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',(key,_json(value)))


def _check_identity(dataset,metadata):
    if any(metadata.get(k)!=v for k,v in _identity(dataset).items()):raise ValueError('epoch index source identity mismatch')


def validate_index(dataset,index_path):
    """Validate identity/completion and interval coverage; never decode images."""
    path=Path(index_path).resolve()
    with sqlite3.connect(f'{path.as_uri()}?mode=ro',uri=True) as c:
        m=_metadata(c);_check_identity(dataset,m)
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
    return dict(m,index_sha256=sha256_file(path),index_path=str(path))


def build_epoch_index(dataset,index_path,*,sample_fps=15.0):
    if not math.isfinite(sample_fps) or sample_fps<=0:raise ValueError('sample_fps must be positive finite')
    path=Path(index_path);path.parent.mkdir(parents=True,exist_ok=True)
    with sqlite3.connect(path) as c:
        c.executescript('''CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS episodes(repo_id INTEGER,episode INTEGER,excluded INTEGER,digest TEXT,PRIMARY KEY(repo_id,episode));
          CREATE TABLE IF NOT EXISTS sources(path TEXT PRIMARY KEY,size INTEGER,mtime_ns INTEGER);
          CREATE TABLE IF NOT EXISTS episode_sources(repo_id INTEGER,episode INTEGER,path TEXT,PRIMARY KEY(repo_id,episode,path));
          CREATE TABLE IF NOT EXISTS runs(sample_start INTEGER PRIMARY KEY,count INTEGER NOT NULL,repo_id INTEGER,
          episode INTEGER,level TEXT,segment TEXT,grid_start INTEGER,numerator INTEGER,denominator INTEGER);''')
        m=_metadata(c)
        if m:
            _check_identity(dataset,m)
            if m['sample_fps']!=sample_fps:raise ValueError('epoch index sample_fps identity mismatch')
            _check_sources(c)
        else:
            for k,v in dict(_identity(dataset),sample_fps=sample_fps,complete=False,total_samples=0,
                grid_policy='episode-zero floor(k*source_fps/sample_fps); slower sources retain all frames',
                excluded_invalid_anchor_pairs=0).items():_set(c,k,v)
            c.commit()
        if not m.get('complete',False):
            total=c.execute('SELECT COALESCE(MAX(sample_start+count),0) FROM runs').fetchone()[0]
            excluded=c.execute('SELECT COALESCE(SUM(excluded),0) FROM episodes').fetchone()[0]
            for ri,repo in enumerate(dataset.repos):
                ranges_by_episode={}
                for item in sorted(repo['ranges']):ranges_by_episode.setdefault(item[0],[]).append(item)
                for episode,ranges in ranges_by_episode.items():
                    if c.execute('SELECT 1 FROM episodes WHERE repo_id=? AND episode=?',(ri,episode)).fetchone():continue
                    metadata=dataset._episode(repo,episode);n=int(metadata['length']);fps=float(metadata['info']['fps'])
                    if not math.isfinite(fps) or fps<=0:raise ValueError('invalid source fps')
                    ratio=max(Fraction(str(fps))/Fraction(str(sample_fps)),Fraction(1))
                    num,den=ratio.numerator,ratio.denominator
                    valid=np.asarray(metadata.get('source_valid',metadata.get('history_source_valid',np.ones(n,bool))),dtype=bool)
                    if valid.shape!=(n,):raise ValueError('invalid source validity shape')
                    episode_excluded=0
                    # An episode transaction contains its runs and completion marker together.
                    with c:
                        for level in ('task','subtask'):
                            for segment in metadata['tasks' if level=='task' else 'subtasks']:
                                goal=segment['end_exclusive']-1
                                for e,a,b in ranges:
                                    if not 0<=a<b<=n:raise ValueError('curated range exceeds original episode')
                                    start=max(a,segment['start']);end=min(b,goal)
                                    if start>=end:continue
                                    lo=(start*den+num-1)//num;hi=(end*den+num-1)//num
                                    # Only invalid source indices need expansion; no population-sized array.
                                    invalid=np.flatnonzero(~valid[start:end])+start
                                    run_start=lo
                                    for frame in invalid:
                                        k=(int(frame)*den+num-1)//num
                                        if k>=hi or k*num//den!=frame:continue
                                        episode_excluded+=1
                                        if k>run_start:
                                            count=k-run_start
                                            c.execute('INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?)',
                                                (total,count,ri,episode,level,_json(segment),run_start,num,den))
                                            total+=count
                                        run_start=k+1
                                    if run_start<hi:
                                        count=hi-run_start
                                        c.execute('INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?)',
                                            (total,count,ri,episode,level,_json(segment),run_start,num,den))
                                        total+=count
                        _record_sources(c,ri,episode,metadata)
                        c.execute('INSERT INTO episodes VALUES (?,?,?,?)',(ri,episode,episode_excluded,_episode_digest(metadata)))
                        excluded+=episode_excluded
                        _set(c,'total_samples',total);_set(c,'excluded_invalid_anchor_pairs',excluded)
            with c:_set(c,'complete',True)
    return validate_index(dataset,path)


class FiniteGoalDataset:
    def __init__(self,dataset,index_path):
        self.dataset=dataset;self.index_path=Path(index_path).resolve()
        self.summary=validate_index(dataset,self.index_path);self._connection=None;self._evaluations=None
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
