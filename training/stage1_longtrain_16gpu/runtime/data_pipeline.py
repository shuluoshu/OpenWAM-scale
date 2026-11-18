"""Full-source, common-draw Stage1 input provider for native G / causal C*.

The old finite population is read-only. Sampling is an unbounded keyed stream:
exact shuffled 3:7 grain blocks -> native YAML repository marginal -> uniform
eligible 3fps anchor. Every rank and architecture sees the same draw mapping.
Workers own bounded native readers and decode CPU RGB. The main process batches
only frozen VAE/T5 cache misses; every VAE input remains independent T=1.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from contextlib import contextmanager, closing
import copy
import fcntl
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sqlite3
import string
import sys
import tempfile
import time
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler

ROOT=Path(__file__).resolve().parent
DEFAULT_CODE=Path('/mnt/data/cpfs/b5/yeqingda/CodeX/runtime/openwam-kf1-endpoint-221464d-dlc')
DEFAULT_SOURCE=Path('/mnt/data/cpfs/b5/yeqingda/openwam-0911/OpenWAM/outputs/stage12_subset_continuation_20261109/source_exclude_missing_133.yaml')
DEFAULT_ORIGINAL_INDEX=Path('/mnt/data/cpfs/b5/yeqingda/openwam-0911/OpenWAM/outputs/causal_cstar_h100_switch_20261109/h100_causal_text/population.sqlite')
DEFAULT_INDEX=Path('/mnt/data/cpfs/b5big/yeqingda/CodeX/longtrain_16gpu_20261112/population_raw_policy.sqlite')
DEFAULT_CACHE=Path('/mnt/data/cpfs/b5big/yeqingda/CodeX/longtrain_16gpu_20261112/cache')
DEFAULT_G_VAL=Path('/mnt/data/cpfs/b5/yeqingda/CodeX/experiments/rfc_stage1_compare_20261111/data')
DEFAULT_C_VAL=Path('/mnt/data/cpfs/b5/yeqingda/CodeX/experiments/rfc_stage1_compare_8gpu_20261111/baseline_cstar/data')
G_FIELDS=('memory_valid','memory_view_valid','memory_ages','memory_frame_ids',
    'observed_view_valid','target_uv','target_image_valid','target_view_valid',
    'target_point_valid','target_in_frame','goal_horizon_s')
STREAM_VERSION='native-repo-marginal-3fps-exact-3to7-keyed-v1'


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),default=str)
def digest_json(value):return hashlib.sha256(canonical(value).encode()).hexdigest()
def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.'+path.name,dir=path.parent)
    try:
        with os.fdopen(fd,'w') as f:json.dump(value,f,indent=2,default=str);f.write('\n');f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def bind_native(code_dir=DEFAULT_CODE):
    code=str(Path(code_dir).resolve())
    if 'openwam' in sys.modules:
        actual=Path(sys.modules['openwam'].__file__).resolve().parent.parent
        if actual!=Path(code):raise ValueError(f'wrong native runtime already imported: {actual}')
    if code not in sys.path:sys.path.insert(0,code)
    return code


def file_generation(path):
    p=Path(path).resolve();s=p.stat()
    # st_dev is deliberately excluded: the same CPFS inode has mount-local dev.
    return dict(path=str(p),ino=s.st_ino,size=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)


class CommonDrawStream:
    """Compact run catalog; no per-frame objects, permutation, or episode load."""
    def __init__(self,index_path,repos,*,seed=42):
        if type(seed) is not int or not 0<=seed<2**64:raise ValueError('seed must be unsigned 64-bit integer')
        self.path=Path(index_path).resolve();self.repos=repos;self.seed=seed;self._connection=None
        self.groups={};self.weights={};self.repo_ids={};self._grain_blocks=OrderedDict()
        with closing(sqlite3.connect(f'{self.path.as_uri()}?mode=ro',uri=True)) as c:
            self.metadata={k:json.loads(v) for k,v in c.execute('SELECT key,value FROM metadata')}
            if self.metadata.get('version')!=3 or self.metadata.get('complete') is not True:
                raise ValueError('native v3 completed full population required')
            if self.metadata.get('sample_fps')!=3.:raise ValueError('full-source population must be native 3fps')
            self.total=int(self.metadata['total_samples'])
            for level in ('task','subtask'):
                ids=[];weights=[];groups=[]
                # Repo count is small. SQLite run scan is done once per provider,
                # never 203k native _episode calls or materialized Python pools.
                rows=c.execute('SELECT repo_id,sample_start,count FROM runs WHERE level=? ORDER BY repo_id,sample_start',(level,))
                current=None;starts=[];counts=[]
                def finish():
                    if current is None:return
                    if not 0<=current<len(repos):raise ValueError('run repository outside selected YAML')
                    ids.append(current);weights.append(float(repos[current]['weight']))
                    groups.append((np.asarray(starts,dtype=np.int64),np.cumsum(counts,dtype=np.int64)))
                for ri,start,count in rows:
                    if count<=0:raise ValueError('nonpositive native run')
                    if ri!=current:
                        finish();current=ri;starts=[];counts=[]
                    starts.append(start);counts.append(count)
                finish()
                if not groups or not all(w>0 and math.isfinite(w) for w in weights):
                    raise ValueError(f'no eligible positive-weight {level} population')
                self.groups[level]=groups;self.repo_ids[level]=ids
                w=np.asarray(weights,dtype=np.float64);self.weights[level]=w/w.sum()
        self.contract=dict(version=STREAM_VERSION,seed=seed,grain_block=10,task_per_block=3,subtask_per_block=7,
            sample_fps=3.,index_metadata=self.metadata,repo_marginals=[dict(repo=r['repo'],weight=r['weight']) for r in repos],
            grain_then_repository_then_uniform_valid_grid_anchor=True,numpy_version=np.__version__,
            cursor='absolute-global-draw; no epoch reset; with-replacement')
        self.identity=digest_json(self.contract)

    def __getstate__(self):return dict(self.__dict__,_connection=None,_grain_blocks=OrderedDict())
    def _db(self):
        if self._connection is None:self._connection=sqlite3.connect(f'{self.path.as_uri()}?mode=ro',uri=True)
        return self._connection
    def level_at(self,draw):
        if type(draw) is not int or draw<0:raise IndexError(draw)
        block,slot=divmod(draw,10)
        if block not in self._grain_blocks:
            levels=np.asarray(['task']*3+['subtask']*7)
            np.random.default_rng(np.random.SeedSequence([self.seed,block,3719])).shuffle(levels)
            self._grain_blocks[block]=levels
            while len(self._grain_blocks)>16:self._grain_blocks.popitem(last=False)
        return str(self._grain_blocks[block][slot])
    def selection(self,draw):
        level=self.level_at(draw)
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,draw,3727]))
        group_index=int(rng.choice(len(self.groups[level]),p=self.weights[level]))
        starts,cumulative=self.groups[level][group_index]
        offset=int(rng.integers(int(cumulative[-1])))
        run=int(np.searchsorted(cumulative,offset,side='right'))
        sample_id=int(starts[run])+offset-(int(cumulative[run-1]) if run else 0)
        row=self._db().execute('SELECT * FROM runs WHERE sample_start=?',(int(starts[run]),)).fetchone()
        start,count,ri,episode,actual_level,segment,grid,num,den=row
        if actual_level!=level or ri!=self.repo_ids[level][group_index]:raise ValueError('native sampling catalog changed')
        segment=json.loads(segment);anchor=(grid+sample_id-start)*num//den;goal=int(segment['end_exclusive'])-1
        if not int(segment['start'])<=anchor<goal:raise ValueError('native matching-grain endpoint invariant failed')
        return dict(global_draw=draw,sample_id=sample_id,repo_id=ri,episode=episode,level=level,
            segment=segment,anchor=anchor,goal=goal)


def local_draw_positions(update,local_batch_size,accumulation_steps,*,rank,world):
    if any(type(v) is not int or v<0 for v in (update,rank)) or not 0<=rank<world:
        raise ValueError('invalid update/rank/world')
    if any(type(v) is not int or v<1 for v in (local_batch_size,accumulation_steps,world)):
        raise ValueError('batch, accumulation and world must be positive integers')
    cursor=update*local_batch_size*accumulation_steps*world
    return [[cursor+a*world*local_batch_size+rank*local_batch_size+i for i in range(local_batch_size)]
            for a in range(accumulation_steps)]


class BoundedDiskStore:
    """Immutable content-keyed payloads with a strict shared on-disk byte budget.

    SQLite contains only cache accounting, never training data. flock serializes
    writers/eviction and prevents readers losing a file during torch.load. DELETE
    journaling avoids relying on CPFS WAL shared-memory support. A put commits
    atomic bytes and evicts the oldest insertion; no full-directory scan per step.
    """
    def __init__(self,root,*,max_bytes):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        if type(max_bytes) is not int or max_bytes<1:raise ValueError('disk cache budget must be positive integer')
        self.max_bytes=max_bytes;self._connection=None
        with self._lock(True):
            c=self._db();c.execute('CREATE TABLE IF NOT EXISTS entries(key TEXT PRIMARY KEY,bytes INTEGER,created REAL)')
            c.execute('CREATE TABLE IF NOT EXISTS config(key TEXT PRIMARY KEY,value INTEGER)')
            c.execute('CREATE INDEX IF NOT EXISTS entries_age ON entries(created,key)')
            row=c.execute("SELECT value FROM config WHERE key='max_bytes'").fetchone()
            if row is not None and row[0]!=max_bytes:raise ValueError('shared cache budget differs; use a new cache directory')
            c.execute("INSERT OR IGNORE INTO config VALUES ('max_bytes',?)",(max_bytes,))
            c.execute("INSERT OR IGNORE INTO config SELECT 'used_bytes',COALESCE(SUM(bytes),0) FROM entries");c.commit()
            self._recover_pending()
    def __getstate__(self):return dict(self.__dict__,_connection=None)
    @contextmanager
    def _lock(self,exclusive):
        with (self.root/'.lock').open('a+b') as f:
            fcntl.flock(f.fileno(),fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            try:yield
            finally:fcntl.flock(f.fileno(),fcntl.LOCK_UN)
    def _db(self):
        if self._connection is None:
            self._connection=sqlite3.connect(self.root/'catalog.sqlite',timeout=120)
            self._connection.execute('PRAGMA journal_mode=DELETE')
        return self._connection
    @property
    def disk_bytes(self):
        with self._lock(False):return int(self._db().execute("SELECT value FROM config WHERE key='used_bytes'").fetchone()[0])
    def _path(self,key):return self.root/key[:2]/(key+'.pt')
    def _recover_pending(self):
        journal=self.root/'.pending.json'
        if not journal.is_file():return
        item=json.loads(journal.read_text());p=self._path(item['key']);c=self._db()
        if p.is_file() and c.execute('SELECT 1 FROM entries WHERE key=?',(item['key'],)).fetchone() is None:
            c.execute('INSERT INTO entries VALUES (?,?,?)',(item['key'],p.stat().st_size,item['created']))
        if item.get('temporary'):Path(item['temporary']).unlink(missing_ok=True)
        used=int(c.execute('SELECT COALESCE(SUM(bytes),0) FROM entries').fetchone()[0])
        while used>self.max_bytes:
            key,n=c.execute('SELECT key,bytes FROM entries ORDER BY created,key LIMIT 1').fetchone()
            self._path(key).unlink(missing_ok=True);c.execute('DELETE FROM entries WHERE key=?',(key,));used-=n
        c.execute("UPDATE config SET value=? WHERE key='used_bytes'",(used,));c.commit();journal.unlink()
    def get(self,identity):
        if (self.root/'.pending.json').exists():
            with self._lock(True):self._recover_pending()
        key=digest_json(identity);p=self._path(key)
        with self._lock(False):
            if not p.is_file():return None
            payload=torch.load(p,map_location='cpu',weights_only=True)
        if payload.get('identity')!=identity:raise ValueError(f'disk cache identity mismatch: {p}')
        value=payload['value']
        if not torch.is_tensor(value) or not torch.isfinite(value).all():raise ValueError(f'disk cache tensor is not finite: {p}')
        return value
    def put(self,identity,value):
        if not torch.is_tensor(value) or not torch.isfinite(value).all():raise ValueError('disk cache requires finite tensor')
        key=digest_json(identity);p=self._path(key);p.parent.mkdir(exist_ok=True)
        # Serialize under the lock so simultaneous pending writes are included in
        # the bound. One oversized value bypasses cache without affecting output.
        with self._lock(True):
            self._recover_pending()
            if p.is_file():
                if self._db().execute('SELECT 1 FROM entries WHERE key=?',(key,)).fetchone() is None:
                    raise ValueError('unaccounted cache file; explicit cache repair required: '+str(p))
                return
            fd,tmp=tempfile.mkstemp(prefix='.pending-',dir=p.parent)
            journal=self.root/'.pending.json'
            atomic_json(journal,dict(key=key,created=time.time(),temporary=tmp))
            committed=False
            try:
                with os.fdopen(fd,'wb') as f:
                    torch.save(dict(identity=identity,value=value.detach().cpu().contiguous()),f);f.flush();os.fsync(f.fileno())
                size=os.stat(tmp).st_size
                if size>self.max_bytes:return
                c=self._db();used=int(c.execute("SELECT value FROM config WHERE key='used_bytes'").fetchone()[0])
                while used+size>self.max_bytes:
                    old=c.execute('SELECT key,bytes FROM entries ORDER BY created,key LIMIT 1').fetchone()
                    if old is None:raise RuntimeError('cache accounting corrupt')
                    self._path(old[0]).unlink(missing_ok=True);c.execute('DELETE FROM entries WHERE key=?',(old[0],));used-=old[1]
                os.replace(tmp,p);c.execute('INSERT OR REPLACE INTO entries VALUES (?,?,?)',(key,size,time.time()))
                c.execute("UPDATE config SET value=? WHERE key='used_bytes'",(used+size,));c.commit();committed=True
            except BaseException:
                self._db().rollback();raise
            finally:
                if os.path.exists(tmp):os.unlink(tmp)
                # Keep the journal if the final bytes exist but the accounting
                # commit failed; the next writer/constructor repairs under lock.
                if committed or not p.is_file():
                    journal.unlink(missing_ok=True)


class TensorLRU:
    def __init__(self,max_bytes):self.max_bytes=int(max_bytes);self.nbytes=0;self.values=OrderedDict()
    def get(self,key):
        if key not in self.values:return None
        self.values.move_to_end(key);return self.values[key]
    def put(self,key,value):
        n=value.numel()*value.element_size()
        if n>self.max_bytes:return
        if key in self.values:self.nbytes-=self.values.pop(key).numel()*value.element_size()
        while self.values and self.nbytes+n>self.max_bytes:
            _,old=self.values.popitem(last=False);self.nbytes-=old.numel()*old.element_size()
        self.values[key]=value;self.nbytes+=n


def cached_or_decode(store,identity,decode):
    latent=store.get(identity)
    if latent is not None:return dict(identity=identity,latent=latent)
    return dict(identity=identity,pixels=decode())


def _video_path(m,episode,camera):
    from openwam.dataloader.rfc_goal import _video
    return _video(m,episode,camera)


class FullSourceDataset(Dataset):
    """Lazy native episode reader. GPU objects never enter DataLoader workers."""
    def __init__(self,config,arm,*,frame_store=None,dependencies=None,audit=None):
        self.config=config;self.options=config.get('data_pipeline',{});self.arm=normalize_arm(arm)
        bind_native(config.get('code_dir',DEFAULT_CODE))
        if self.options.get('metadata_view_manifest'):
            from frozen_metadata_view import make_reader
            reader_policy=dict(metadata_view_manifest=self.options['metadata_view_manifest'])
        else:
            from raw_metadata_overlay import make_reader
            reader_policy=dict(overlay_manifest=self.options.get('raw_metadata_overlay',ROOT/'raw_metadata_overlay.json'))
        self.source=Path(self.options.get('source_yaml',DEFAULT_SOURCE)).resolve()
        self.index=Path(self.options.get('population_index',DEFAULT_INDEX)).resolve()
        self.reader=make_reader(self.source,self.options.get('reader_cache',ROOT/'reader_cache'),
            **reader_policy,
            seed=int(config.get('seed',42)),subtask_probability=.7,dtype=torch.float32,
            observation_mode='multiview',materialize_multiview_target=True,
            max_cached_episodes=int(self.options.get('max_cached_episodes',2)),
            data_root=self.options.get('data_root'),path_aliases=self.options.get('path_aliases'))
        self.stream=CommonDrawStream(self.index,self.reader.repos,seed=int(config.get('seed',42)))
        self.audit=audit;self.frame_store=frame_store;self.dependencies=dependencies
        self._validated=OrderedDict();self._source_generations=OrderedDict()
    def __len__(self):return 2**63-1  # DataLoader sampler is infinite; no arbitrary epoch/pilot stop.
    def __getstate__(self):return dict(self.__dict__,_validated=OrderedDict(),_source_generations=OrderedDict())
    def _metadata(self,item):
        from openwam.dataloader.goal_epoch import _check_sources,_episode_digest
        repo=self.reader.repos[item['repo_id']];key=(item['repo_id'],item['episode'])
        c=self.stream._db();_check_sources(c,*key)
        m=self.reader._episode(repo,item['episode'])
        expected=c.execute('SELECT digest FROM episodes WHERE repo_id=? AND episode=?',key).fetchone()
        if expected is None:raise ValueError('sample episode absent from verified native index')
        if key not in self._validated or self._validated[key] is not m:
            if _episode_digest(m)!=expected[0]:raise ValueError(f'native source labels changed: {repo["repo"]}:{item["episode"]}')
            self._validated[key]=m
            while len(self._validated)>self.reader.max_cached_episodes:self._validated.popitem(last=False)
        self._validated.move_to_end(key)
        return repo,m,expected[0]
    def _source(self,repo,m,episode,label_digest):
        paths=set(m.get('data_files',()))
        dependencies=self.reader.metadata_dependencies(repo,episode) if hasattr(self.reader,'metadata_dependencies') else None
        if dependencies is None:paths.update(m['root']/'meta'/name for name in ('info.json','tasks.parquet'))
        else:paths.update(Path(p) for p in dependencies)
        if m.get('annotation_file'):paths.add(Path(m['annotation_file']))
        paths.update(_video_path(m,episode,c)[0] for c in m['cameras'] if c is not None)
        generations=[file_generation(p) for p in sorted(paths,key=str)]
        key=(repo['path'],episode)
        previous=self._source_generations.get(key)
        if previous is not None and previous!=generations:raise ValueError('selected immutable source generation changed: '+str(key))
        self._source_generations[key]=generations;self._source_generations.move_to_end(key)
        while len(self._source_generations)>self.reader.max_cached_episodes:self._source_generations.popitem(last=False)
        source=dict(native_label_digest=label_digest,repo=str(Path(m['root']).resolve()),episode=episode,
            cameras=m['cameras'],ep=m['ep'],files=generations,
            population_selection=self.stream.metadata.get('selection_sha256'))
        return digest_json(source),generations
    def _frame_key(self,source,episode,frame,mode,views):
        if mode=='head':views=[bool(views[0]),False,False]
        return dict(format='openwam.longtrain.frozen-frame.v1',source=source,episode=int(episode),frame=int(frame),
            view='head' if mode=='head' else 'head+left_wrist+right_wrist',view_valid=[bool(v) for v in views],
            preprocess='rgb-bilinear-head256x320-t1-v1' if mode=='head' else 'rgb-bilinear-head256x320-wrists128x160-canvas384x320-t1-v1',
            codec='wan22_vae48_native_independent_t1',vae=self.dependencies['vae']['fingerprint'],
            dtype=self.options.get('encoding_dtype',self.config.get('dtype','bfloat16')))
    def _decode_frame(self,repo,m,episode,frame,mode,views):
        from scripts.data.prepare_subtask_goals import decode_frames
        from openwam.model.goal.runtime import clean_goal_head
        from openwam.model.goal.observation import clean_multiview
        images=[None]*3
        for i,camera in enumerate(m['cameras']):
            if mode=='head' and i:break
            if not views[i]:continue
            path,offset=_video_path(m,episode,camera)
            values,_=decode_frames(path,offset,float(m['info']['fps']),[frame],
                sequential=repo['family']=='mecka',video_ordinals=m.get('video_ordinals'))
            images[i]=values[frame]
        return clean_goal_head(images[0]) if mode=='head' else clean_multiview(images,torch.tensor(views,dtype=torch.bool))
    def _slot(self,source,repo,m,episode,frame,mode,views,*,pixels=None):
        identity=self._frame_key(source,episode,frame,mode,views)
        def decode():return pixels if pixels is not None else self._decode_frame(repo,m,episode,frame,mode,views)
        slot=cached_or_decode(self.frame_store,identity,decode) if self.frame_store is not None else dict(identity=identity,pixels=decode())
        if 'latent' in slot and tuple(slot['latent'].shape)!=(48,1,24 if mode=='multiview' else 16,20):
            raise ValueError('native frozen frame cache latent shape differs from codec: '+mode)
        slot['mode']=mode;return slot
    def __getitem__(self,draw):
        from openwam.dataloader.rfc_goal import select_history_frames
        from openwam.dataloader.head_history import select_history_frames as select_head,HistorySamplingConfig
        item=self.stream.selection(int(draw));repo,m,label_digest=self._metadata(item)
        source,generations=self._source(repo,m,item['episode'],label_digest)
        anchor=item['anchor'];goal=item['goal'];fps=float(m['info']['fps']);episode=item['episode']
        if not bool(m['source_valid'][anchor]):raise ValueError('indexed anchor became invalid')
        self.reader._history_draw_id=int(draw)
        # Native geometry/endpoints and RGB preprocessing remain authoritative.
        # Histories below consult persistent frozen latents BEFORE RGB decoding.
        record,b=self.reader._materialize(repo,episode,m,item['segment'],anchor,goal,item['level'])
        target=b['targets'];observed_views=b['observed_view_valid'][0]
        target_views=b['target_view_valid'][0]
        slots={};slots['current']=self._slot(source,repo,m,episode,anchor,'multiview',observed_views,pixels=b['observed'][0])
        target_mode='multiview' if self.arm=='G' else 'head'
        target_valid=bool(b['target_image_valid'][0]) if self.arm=='G' else bool(record['goal_image_valid'])
        if target_valid:
            pixels=b['target_image'][0] if self.arm=='G' else b['target_image'][0][:,:,:256,:].contiguous()
            # C* encodes normalized RAW head RGB with native GoalCodec. It never
            # crops/reuses the G multiview target latent (different convolution).
            slots['target']=self._slot(source,repo,m,episode,goal,target_mode,target_views,pixels=pixels)
        history_slots=[]
        row=dict(instruction=record['instruction'],granularity=item['level'],record=record,slots=slots,
            global_draw=int(draw),sample_id=item['sample_id'],memory_source_identity=source,
            observed_view_valid=observed_views,target_uv=target.uv[0],
            target_image_valid=torch.tensor(target_valid,dtype=torch.bool),target_view_valid=target_views,
            target_point_valid=target.point_valid[0],target_in_frame=target.in_frame[0],
            goal_horizon_s=torch.tensor((goal-anchor)/fps,dtype=torch.float32))
        if self.arm=='G':
            ids=select_history_frames(anchor,fps,2);count=3
            valid=torch.zeros(count,dtype=torch.bool);views=torch.zeros(count,3,dtype=torch.bool)
            ages=torch.zeros(count);frames=torch.full((count,),-1,dtype=torch.long)
            for i,frame in enumerate(ids):
                view=[c is not None and bool(m['source_valid'][frame]) for c in m['cameras']]
                views[i]=torch.tensor(view);valid[i]=any(view);ages[i]=(anchor-frame)/fps;frames[i]=frame
                if valid[i]:history_slots.append((i,self._slot(source,repo,m,episode,frame,'multiview',view)))
            row.update(memory_valid=valid,memory_view_valid=views,memory_ages=ages,memory_frame_ids=frames)
        else:
            start=next(a for e,a,bound in repo['ranges'] if e==episode and a<=anchor<bound)
            ids,ages,valid=select_head(anchor,fps,start,HistorySamplingConfig(enabled=True),np.random.default_rng(0),True)
            valid &= m['head'] is not None
            selected=np.flatnonzero(valid);valid[selected]&=m['source_valid'][ids[selected]]
            ids[~valid]=-1;ages[~valid]=0
            for i in np.flatnonzero(valid):
                history_slots.append((int(i),self._slot(source,repo,m,episode,int(ids[i]),'head',[True,False,False])))
            episode_identity=f'{Path(m["root"]).resolve()}:{episode}'
            eid=int.from_bytes(hashlib.sha256(episode_identity.encode()).digest()[:8],'big')&((1<<63)-1)
            row.update(history_valid=torch.from_numpy(valid),history_ages=torch.from_numpy(ages),
                history_frame_ids=torch.from_numpy(ids),history_episode_id=torch.tensor(eid,dtype=torch.long))
            if not target_valid:row['target_point_valid'].zero_();row['target_in_frame'].zero_()
        row['history_slots']=history_slots
        for generation in generations:
            if file_generation(generation['path'])!=generation:raise ValueError('source changed during materialization: '+generation['path'])
        return row


def normalize_arm(arm):
    if arm in ('G','g','default','new_g','newG'):return 'G'
    if arm in ('Cstar','cstar','causal_cstar','old_cstar','oldCstar'):return 'Cstar'
    raise ValueError('architecture must be G or Cstar')


def resolve_cache_directories(options,arm,*,rank,local_rank=None,environ=None):
    """Resolve rank-private frame roots once in the launcher/main process.

    Only {arch}, {rank}, {local_rank} are supported in frame_cache_dir.
    text_cache_dir is an explicit shared path. Missing overrides preserve the
    original cache_dir/{frames,text} layout and its existing cache identities.
    """
    if type(rank) is not int or rank<0:raise ValueError('cache rank must be nonnegative integer')
    base=Path(options.get('cache_dir',DEFAULT_CACHE)).expanduser()
    frame=options.get('frame_cache_dir')
    if frame is None:frames=base/'frames'
    else:
        frame=str(frame);fields=[]
        for _,field,spec,conversion in string.Formatter().parse(frame):
            if field is None:continue
            if field not in ('arch','rank','local_rank'):raise ValueError('unknown frame_cache_dir variable: '+str(field))
            if '{' in spec or '}' in spec:raise ValueError('nested frame_cache_dir format variables are unsupported')
            fields.append(field)
        if local_rank is None:local_rank=options.get('local_rank')
        environment=os.environ if environ is None else environ
        if local_rank is None and 'LOCAL_RANK' in environment:
            try:local_rank=int(environment['LOCAL_RANK'])
            except (ValueError,TypeError) as exc:raise ValueError('LOCAL_RANK must be a nonnegative integer') from exc
        if 'local_rank' in fields and (type(local_rank) is not int or local_rank<0):
            raise ValueError('frame_cache_dir local_rank requires LOCAL_RANK or explicit nonnegative local_rank')
        try:frames=Path(frame.format_map(dict(arch=normalize_arm(arm),rank=rank,local_rank=local_rank))).expanduser()
        except (KeyError,ValueError,TypeError) as exc:raise ValueError('invalid frame_cache_dir format: '+str(exc)) from exc
    text=Path(options['text_cache_dir']).expanduser() if options.get('text_cache_dir') is not None else base/'text'
    return dict(frames=frames.resolve(),text=text.resolve())


class DrawBatchSampler(Sampler):
    def __init__(self,*,start_update,local_batch_size,accumulation_steps,rank,world):
        self.start_update=start_update;self.local_batch_size=local_batch_size
        self.accumulation_steps=accumulation_steps;self.rank=rank;self.world=world
    def __iter__(self):
        update=self.start_update
        while True:
            for positions in local_draw_positions(update,self.local_batch_size,self.accumulation_steps,rank=self.rank,world=self.world):yield positions
            update+=1


def worker_init(_):
    torch.set_num_threads(1)
    try:
        import pyarrow as pa
        pa.set_cpu_count(1);pa.set_io_thread_count(1)
    except ImportError:pass


def list_collate(rows):return rows


class FullSourceProvider:
    def __init__(self,config,arm,rank,world,device,dtype,*,vae=None,text_encoder=None,tokenizer=None,dependencies=None):
        self.run_config=copy.deepcopy(config);self.config=resolve_native_config(config)
        self.arm=normalize_arm(arm);self.rank=rank;self.world=world
        self.device=torch.device(device);self.dtype=dtype;self.options=self.config.get('data_pipeline',{})
        bind_native(self.config.get('code_dir',DEFAULT_CODE))
        audit_path=Path(self.options.get('audit_file',ROOT/'data_audit.json'))
        if not audit_path.is_file():raise FileNotFoundError(f'full-source metadata audit required before training: {audit_path}; run data_pipeline.py audit')
        self.audit=json.loads(audit_path.read_text())
        index=Path(self.options.get('population_index',DEFAULT_INDEX)).resolve()
        source=Path(self.options.get('source_yaml',DEFAULT_SOURCE)).resolve()
        if self.audit.get('format')!='openwam.longtrain.full-source-audit.v1':raise ValueError('unknown data audit format')
        if self.audit['source_yaml_sha256']!=sha256_file(source) or self.audit['population']['index_file_stamp']!=dict(size=index.stat().st_size,mtime_ns=index.stat().st_mtime_ns):
            raise ValueError('full-source YAML/index changed since metadata audit')
        cache=Path(self.options.get('cache_dir',DEFAULT_CACHE))
        if (self.options.get('frame_cache_dir') is None or self.options.get('text_cache_dir') is None) and not cache.parent.exists():
            raise FileNotFoundError(f'cache parent missing (no automatic fallback to nearly full b5): {cache.parent}')
        self.cache_directories=resolve_cache_directories(self.options,self.arm,rank=rank)
        self.frame_store=BoundedDiskStore(self.cache_directories['frames'],max_bytes=int(self.options.get('frame_disk_gb',384)*1024**3))
        self.text_store=BoundedDiskStore(self.cache_directories['text'],max_bytes=int(self.options.get('text_disk_gb',64)*1024**3))
        self.host=TensorLRU(int(self.options.get('host_cache_mb',256)*1024**2))
        self.vae=vae;self.text_encoder=text_encoder;self.tokenizer=tokenizer
        self.dependencies=dependencies
        self._loader=None;self._iterator=None;self._next_update=None;self._shape=None
        self._codecs=None;self._encoder_signatures=None
        self.stats=dict(frame_hits=0,frame_misses=0,text_hits=0,text_misses=0)
        self.dataset=None;self.identity=None
        if vae is not None:self.attach_encoders(vae,text_encoder,tokenizer,dependencies=dependencies)
    def attach_encoders(self,vae,text_encoder,tokenizer,*,dependencies=None):
        from openwam.model.goal.dependencies import dependency_identity
        from openwam.model.goal.codec import GoalCodec,GoalCodecSpec,vae_cache_identity
        vae.eval().requires_grad_(False);text_encoder.eval().requires_grad_(False)
        self.vae=vae;self.text_encoder=text_encoder;self.tokenizer=tokenizer
        self.dependencies=dependencies or dependency_identity(vae,text_dim=self.config['components']['dit']['extra_kwargs']['text_dim'],
            text_encoder=text_encoder,tokenizer=tokenizer)
        # Hash actual resident tensors once. A caller supplying identities accepts
        # responsibility for its verified foundation load; signatures reject later edits.
        self._encoder_signatures=(vae_cache_identity(vae),vae_cache_identity(text_encoder))
        batch=int(self.options.get('vae_batch_size',4))
        self._codecs={mode:GoalCodec(vae,GoalCodecSpec(name='wan22_multiview_t1' if mode=='multiview' else 'wan22_vae48_single_frame',
            image_shape=(3,1,384 if mode=='multiview' else 256,320),latent_shape=(48,1,24 if mode=='multiview' else 16,20)),
            max_encode_batch_size=batch) for mode in ('multiview','head')}
        if getattr(self,'_profile_only',False):return self
        self.dataset=FullSourceDataset(self.config,self.arm,frame_store=self.frame_store,dependencies=self.dependencies,audit=self.audit)
        from openwam.dataloader.goal_epoch import _selected_repos_digest
        if _selected_repos_digest(self.dataset.reader)!=self.audit['population']['selected_repos_sha256']:
            raise ValueError('provider selected repositories/RAW overlay differ from full native audit')
        if self.dataset.reader.raw_overlay!=self.audit.get('raw_camera_policy'):
            raise ValueError('provider RAW camera policy differs from full native audit')
        if any(self.dataset.stream.metadata.get(k)!=v for k,v in self.audit['population'].items() if k in self.dataset.stream.metadata):
            raise ValueError('provider population metadata differs from full native audit')
        self.identity=dict(format='openwam.longtrain.common-data.v1',population=self.audit['population'],
            stream=self.dataset.stream.contract,vae=self.dependencies['vae']['fingerprint'],text=self.dependencies['text']['fingerprint'],
            independent_t1=True,training_source='full retained source YAML, not prepared 256 cohort',
            validation='fixed64 fit diagnostic; MAY overlap full training; NOT held-out')
        self.identity['fingerprint']=digest_json(self.identity)
        return self
    def _ensure_encoders(self):
        if self._codecs is not None:return
        from openwam.train.goal_trainer import _construct,load_component_weights
        from openwam.model.goal.dependencies import resolve_tokenizer_config
        encoder_device=torch.device(self.options.get('encoder_device',str(self.device)))
        text_device=torch.device(self.options.get('text_device',str(self.device)))
        vae=_construct(self.config['components']['vae']);load_component_weights(vae,self.config['baseline_checkpoint'],'video_backbone.vae.')
        text=_construct(self.config['components']['text_encoder']);load_component_weights(text,self.config['baseline_checkpoint'],'video_backbone.text_encoder.')
        vae.to(device=encoder_device,dtype=self.dtype);text.to(device=text_device,dtype=self.dtype)
        tc=resolve_tokenizer_config(self.config);mod,name=tc['model_class'].rsplit('.',1)
        tokenizer=getattr(importlib.import_module(mod),name)(**tc['kwargs'])
        self.attach_encoders(vae,text,tokenizer)
    def _start(self,update,b,a):
        if self._iterator is not None:
            shutdown=getattr(self._iterator,'_shutdown_workers',None)
            if shutdown:shutdown()
        sampler=DrawBatchSampler(start_update=update,local_batch_size=b,accumulation_steps=a,rank=self.rank,world=self.world)
        workers=int(self.options.get('num_workers',4))
        kwargs=dict(dataset=self.dataset,batch_sampler=sampler,collate_fn=list_collate,
            num_workers=workers,pin_memory=self.device.type=='cuda',worker_init_fn=worker_init)
        if workers:
            kwargs.update(multiprocessing_context='spawn',persistent_workers=True,prefetch_factor=int(self.options.get('prefetch_factor',2)))
        self._loader=DataLoader(**kwargs);self._iterator=iter(self._loader);self._next_update=update;self._shape=(b,a)
    def state_dict(self):
        return dict(identity=self.identity,next_update=self._next_update,shape=self._shape,
            cursor_policy='get_update(update,B,A) computes absolute cursor update*world*B*A; loader prefetch is uncommitted')
    def load_state_dict(self,state):
        if state['identity']!=self.identity:raise ValueError('provider resume identity mismatch')
        self._next_update=None  # get_update restarts at checkpoint committed update.
    @torch.no_grad()
    def _encode_rows(self,rows):
        from openwam.model.goal.codec import vae_cache_identity
        if (vae_cache_identity(self.vae),vae_cache_identity(self.text_encoder))!=self._encoder_signatures:
            raise ValueError('frozen VAE/text dependency changed after cache binding')
        pending={};all_slots=[]
        for row in rows:
            all_slots.extend(row['slots'].values());all_slots.extend(slot for _,slot in row['history_slots'])
        for slot in all_slots:
            key=digest_json(slot['identity'])
            if 'latent' in slot:self.stats['frame_hits']+=1;continue
            value=self.host.get(key)
            if value is None:value=self.frame_store.get(slot['identity'])
            if value is not None:slot['latent']=value;self.stats['frame_hits']+=1
            else:pending.setdefault((slot['mode'],key),[]).append(slot)
        vae_device=next(self.vae.parameters()).device
        for mode in ('multiview','head'):
            groups=[slots for (kind,_),slots in pending.items() if kind==mode]
            limit=int(self.options.get('vae_batch_size',4))
            for start in range(0,len(groups),limit):
                chunk=groups[start:start+limit]
                pixels=torch.stack([group[0]['pixels'] for group in chunk]).to(device=vae_device,dtype=self.dtype,non_blocking=True)
                latents=self._codecs[mode].encode(pixels,device=vae_device).detach().cpu()
                for group,latent in zip(chunk,latents):
                    self.frame_store.put(group[0]['identity'],latent);self.host.put(digest_json(group[0]['identity']),latent)
                    for slot in group:slot['latent']=latent
                    self.stats['frame_misses']+=1
        missing={};text_identity=self.dependencies['text']['fingerprint']
        for row in rows:
            identity=dict(format='openwam.longtrain.trimmed-text.v1',text=text_identity,
                prompt=row['instruction'],dtype=self.options.get('encoding_dtype',self.config.get('dtype','bfloat16')))
            key=digest_json(identity);context=self.host.get(key)
            if context is None:context=self.text_store.get(identity)
            if context is None:missing.setdefault(key,(identity,[]))[1].append(row)
            else:row['context']=context;self.stats['text_hits']+=1
        from openwam.model.video_backbone.wan.encode import encode_text
        text_device=next(self.text_encoder.parameters()).device
        pairs=list(missing.items());limit=int(self.options.get('text_batch_size',4))
        for start in range(0,len(pairs),limit):
            chunk=pairs[start:start+limit];prompts=[item[1][0]['prompt'] for item in chunk]
            contexts,lengths=encode_text(prompts,tokenizer=self.tokenizer,text_encoder=self.text_encoder,device=text_device)
            for i,(key,(identity,values)) in enumerate(chunk):
                context=contexts[i,:int(lengths[i])].detach().cpu().contiguous()
                if len(context)<1:raise ValueError('native causal text requires nonempty instruction')
                self.text_store.put(identity,context);self.host.put(key,context)
                for row in values:row['context']=context
                self.stats['text_misses']+=1
        return rows
    def _collate(self,rows):
        def stack(k):
            values=torch.stack([r[k] for r in rows]);return values.to(device=self.device,non_blocking=True)
        def latent(k,shape):
            return torch.stack([r['slots'][k]['latent'] if k in r['slots'] else torch.zeros(shape,dtype=self.dtype) for r in rows]).to(self.device,dtype=self.dtype,non_blocking=True)
        width=rows[0]['context'].shape[-1];lengths=[len(r['context']) for r in rows];n=max(lengths)
        text=torch.zeros(len(rows),n,width,device=self.device,dtype=self.dtype)
        mask=torch.zeros(len(rows),n,device=self.device,dtype=torch.bool)
        for i,row in enumerate(rows):text[i,:lengths[i]]=row['context'].to(self.device,dtype=self.dtype);mask[i,:lengths[i]]=True
        current=latent('current',(48,1,24,20))
        if self.arm=='G':
            memory=torch.zeros(len(rows),3,48,1,24,20,device=self.device,dtype=self.dtype)
            for i,row in enumerate(rows):
                for slot,value in row['history_slots']:memory[i,slot]=value['latent'].to(self.device,dtype=self.dtype)
            batch={k:stack(k) for k in G_FIELDS}
            batch.update(current_latents=current,target_latents=latent('target',(48,1,24,20)),
                memory_latents=memory,text=text,text_mask=mask)
        else:
            from openwam.model.goal.types import GoalTargets
            history=torch.zeros(len(rows),10,48,1,16,20,device=self.device,dtype=self.dtype)
            for i,row in enumerate(rows):
                for slot,value in row['history_slots']:history[i,slot]=value['latent'].to(self.device,dtype=self.dtype)
            targets=GoalTargets(image=latent('target',(48,1,16,20)),image_valid=stack('target_image_valid'),
                uv=stack('target_uv').float(),point_valid=stack('target_point_valid'),in_frame=stack('target_in_frame'),
                visibility=torch.full((len(rows),2),-1,dtype=torch.long,device=self.device))
            batch=dict(observed=current,observed_view_valid=stack('observed_view_valid'),context=text,
                context_lens=torch.tensor(lengths,dtype=torch.long,device=self.device),targets=targets,
                history_latents=history,history_ages=stack('history_ages'),history_valid=stack('history_valid'),
                history_episode_id=stack('history_episode_id'))
        metadata=[dict(id=r['record']['id'],global_draw=r['global_draw'],sample_id=r['sample_id'],
            granularity=r['granularity'],family=r['record']['family'],source_episode=r['record']['source_episode'],
            anchor_source_frame=r['record']['anchor_source_frame'],goal_source_frame=r['record']['goal_source_frame'],
            source_identity=r['memory_source_identity']) for r in rows]
        return batch,metadata
    @torch.no_grad()
    def warm_dependencies(self,local_batch_size=1,accumulation=1):
        """Real resident foundation VAE/T5 + bounded independent-T1 workspaces.

        Uses content-verified fixed64 raw current/target RGB and native prompts.
        The return explicitly excludes source-video decode and backward memory;
        keep this provider alive during the training profile to retain modules.
        """
        self._ensure_encoders()
        path=Path(self.options.get('validation_g',DEFAULT_G_VAL))
        manifest=json.loads((path/'manifest.json').read_text());entries=manifest['records']['val']
        vae_device=next(self.vae.parameters()).device;text_device=next(self.text_encoder.parameters()).device
        if vae_device.type=='cuda':torch.cuda.synchronize(vae_device);torch.cuda.reset_peak_memory_stats(vae_device)
        t=time.perf_counter();images=[];head=[];prompts=[]
        for i in range(local_batch_size*accumulation):
            entry=entries[i%len(entries)];p=path/entry['file']
            if sha256_file(p)!=manifest['files'][entry['file']]:raise ValueError('warm fixture SHA mismatch')
            row=torch.load(p,map_location='cpu',weights_only=True)
            current=row['raw_current_rgb'].float().div(127.5).sub(1).to(dtype=self.dtype)
            target=row['raw_target_rgb'].float().div(127.5).sub(1).to(dtype=self.dtype)
            if self.arm=='G':images.extend((current,target,current,current,current))
            else:
                images.append(current)
                head.extend([target[:,:,:256,:].contiguous()]+[current[:,:,:256,:].contiguous()]*10)
            prompts.append(row['instruction'])
        chunk_size=int(self.options.get('vae_batch_size',4))
        for start in range(0,len(images),chunk_size):
            self._codecs['multiview'].encode(torch.stack(images[start:start+chunk_size]).to(vae_device),device=vae_device)
        if self.arm=='Cstar':
            for start in range(0,len(head),chunk_size):
                self._codecs['head'].encode(torch.stack(head[start:start+chunk_size]).to(vae_device),device=vae_device)
        from openwam.model.video_backbone.wan.encode import encode_text
        text_limit=int(self.options.get('text_batch_size',4))
        for start in range(0,len(prompts),text_limit):encode_text(prompts[start:start+text_limit],tokenizer=self.tokenizer,text_encoder=self.text_encoder,device=text_device)
        if vae_device.type=='cuda':torch.cuda.synchronize(vae_device)
        return dict(scope='real frozen VAE/T5 resident, independent-T1 current+terminal fixture RGB; native Cstar head VAE; native T5 batch; excludes video decode/backward',
            local_batch_size=local_batch_size,accumulation=accumulation,vae_batch_size=chunk_size,text_batch_size=text_limit,
            vae_device=str(vae_device),text_device=str(text_device),dependencies=self.dependencies,
            fixture_rgb_role_reuse=True,actual_encode_canvas_count=len(images),actual_encode_head_count=len(head),
            workspace_policy='G current+target+3history canvases; Cstar current canvas+target+10history head frames; real fixture RGB reused to fill configured native VAE chunks',
            resident_parameter_bytes=sum(p.numel()*p.element_size() for module in (self.vae,self.text_encoder) for p in module.parameters()),
            encoder_peak_allocated=torch.cuda.max_memory_allocated(vae_device) if vae_device.type=='cuda' else None,
            encoder_peak_reserved=torch.cuda.max_memory_reserved(vae_device) if vae_device.type=='cuda' else None,elapsed_seconds=time.perf_counter()-t)
    def get_local_microbatches(self,update,global_batch_size,accumulation=1):
        divisor=self.world*accumulation
        if global_batch_size%divisor:raise ValueError('global_batch_size must divide world*accumulation')
        return self.get_update(update,global_batch_size//divisor,accumulation)
    def get_update(self,update,local_batch_size,accumulation_steps):
        self._ensure_encoders()
        if self._iterator is None or self._next_update!=update or self._shape!=(local_batch_size,accumulation_steps):
            self._start(update,local_batch_size,accumulation_steps)
        all_rows=[next(self._iterator) for _ in range(accumulation_steps)]
        self._encode_rows([row for rows in all_rows for row in rows])
        out=[self._collate(rows) for rows in all_rows];self._next_update=update+1
        expected=local_draw_positions(update,local_batch_size,accumulation_steps,rank=self.rank,world=self.world)
        if [[r['global_draw'] for r in metadata] for _,metadata in out]!=expected:raise ValueError('prefetch cursor drift')
        return out
    def validation_batches(self,indices=None,*,batch_size=1):
        """Fixed64 fitting diagnostic; source overlap with full train is allowed."""
        yield from fixed_validation_batches(self.config,self.arm,device=self.device,dtype=self.dtype,
            indices=indices,batch_size=batch_size,dependencies=self.dependencies)
    def close(self):
        if self._iterator is not None:
            shutdown=getattr(self._iterator,'_shutdown_workers',None)
            if shutdown:shutdown()
        self._iterator=None;self._loader=None


def resolve_native_config(config):
    """Accept a native model config or trainer run config with model_config."""
    if 'components' in config:native=copy.deepcopy(config)
    else:
        value=config.get('model_config')
        if isinstance(value,dict):native=copy.deepcopy(value)
        elif value:
            import yaml
            native=yaml.safe_load(Path(value).read_text())
        else:
            prepared=Path(config.get('prepared_g',DEFAULT_G_VAL))
            native=json.loads((prepared/'manifest.json').read_text())['config']
        for key in ('seed','dtype','baseline_checkpoint','code_dir'):
            if key in config:native[key]=config[key]
        native['data_pipeline']=copy.deepcopy(config.get('data_pipeline',{}))
        if 'prepared_g' in config:native['data_pipeline'].setdefault('validation_g',config['prepared_g'])
        if 'prepared_cstar' in config:native['data_pipeline'].setdefault('validation_cstar',config['prepared_cstar'])
    return native


def create_provider(config,arm,rank,world,device,dtype,**kwargs):
    provider=FullSourceProvider(config,arm,rank,world,device,dtype,**kwargs)
    provider._ensure_encoders()
    return provider
build_provider=create_provider


def warm_frozen_dependencies(config,arch,device,dtype,local_batch_size=1,accumulation=1):
    """Standalone profile helper: actual frozen modules, no full-source audit.

    Retain returned vae/text_encoder references through the training profile.
    The modules may later be passed into create_provider to avoid a second load.
    """
    obj=FullSourceProvider.__new__(FullSourceProvider)
    obj.config=resolve_native_config(config);obj.options=obj.config.get('data_pipeline',{})
    obj.arm=normalize_arm(arch);obj.device=torch.device(device);obj.dtype=dtype
    obj._profile_only=True;obj._codecs=None;obj._encoder_signatures=None
    obj.vae=None;obj.text_encoder=None;obj.tokenizer=None;obj.dependencies=None
    bind_native(obj.config.get('code_dir',DEFAULT_CODE))
    obj._ensure_encoders()
    receipt=obj.warm_dependencies(local_batch_size,accumulation)
    return dict(vae=obj.vae,text_encoder=obj.text_encoder,tokenizer=obj.tokenizer,
        dependencies=obj.dependencies,receipt=receipt,provider=obj)


def fixed_validation_batches(config,arm,*,device,dtype,indices=None,batch_size=1,dependencies=None):
    """Reuse immutable real fixtures; never treat these 64 as a held-out split."""
    arm=normalize_arm(arm);opts=config.get('data_pipeline',{})
    gpath=Path(opts.get('validation_g',DEFAULT_G_VAL));cpath=Path(opts.get('validation_cstar',DEFAULT_C_VAL))
    path=gpath if arm=='G' else cpath
    manifest=json.loads((path/'manifest.json').read_text());gmanifest=json.loads((gpath/'manifest.json').read_text())
    if dependencies is not None:
        for kind in ('vae','text'):
            if dependencies[kind]['fingerprint']!=gmanifest['dependencies'][kind]['fingerprint']:
                raise ValueError('fixed validation frozen dependency mismatch: '+kind)
    entries=manifest['records']['val'];indices=list(range(len(entries))) if indices is None else list(indices)
    if arm=='G':
        # Frozen collator only; no old training helper or preparation is mutated.
        module_path=Path('/mnt/data/cpfs/b5/yeqingda/CodeX/experiments/rfc_stage1_compare_8gpu_20261111/frozen_v1/rfc_compare.py')
        spec=importlib.util.spec_from_file_location('_longtrain_frozen_collator',module_path)
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        for start in range(0,len(indices),batch_size):
            rows=[]
            for index in indices[start:start+batch_size]:
                entry=entries[index];p=path/entry['file']
                if sha256_file(p)!=manifest['files'][entry['file']]:raise ValueError('fixed validation SHA mismatch')
                row=torch.load(p,map_location='cpu',weights_only=True)
                cp=path/row['context_file']
                if sha256_file(cp)!=manifest['files'][row['context_file']]:raise ValueError('fixed context SHA mismatch')
                rows.append(row)
            yield mod.collate_prepared(rows,path,device=device,dtype=dtype)
    else:
        # Local equivalent of frozen native C* collate avoids importing execution
        # migration helpers just to load immutable validation tensors.
        from openwam.model.goal.types import GoalTargets
        for start in range(0,len(indices),batch_size):
            rows=[];contexts=[]
            for index in indices[start:start+batch_size]:
                entry=entries[index];p=path/entry['file']
                if sha256_file(p)!=manifest['files'][entry['file']]:raise ValueError('fixed C* validation SHA mismatch')
                r=torch.load(p,map_location='cpu',weights_only=True);rows.append(r)
                cp=Path(r['context_file']);cp=cp if cp.is_absolute() else path/cp
                # C* context belongs to content-pinned parent; verify parent SHA.
                rel=str(cp.relative_to(gpath))
                if sha256_file(cp)!=gmanifest['files'][rel]:raise ValueError('fixed C* text SHA mismatch')
                contexts.append(torch.load(cp,map_location='cpu',weights_only=True))
            def stack(k):
                v=torch.stack([r[k] for r in rows]);return v.to(device=device,dtype=dtype if 'latent' in k else v.dtype)
            lens=torch.tensor([len(c) for c in contexts],device=device,dtype=torch.long)
            text=torch.zeros(len(rows),int(lens.max()),contexts[0].shape[-1],device=device,dtype=dtype)
            for i,c in enumerate(contexts):text[i,:len(c)]=c.to(device=device,dtype=dtype)
            targets=GoalTargets(image=stack('target_latent'),image_valid=stack('target_image_valid'),uv=stack('target_uv').float(),
                point_valid=stack('target_point_valid'),in_frame=stack('target_in_frame'),
                visibility=torch.full((len(rows),2),-1,device=device,dtype=torch.long))
            batch=dict(observed=stack('observed_latent'),observed_view_valid=stack('observed_view_valid'),context=text,
                context_lens=lens,targets=targets,history_latents=stack('history_latents'),history_ages=stack('history_ages'),
                history_valid=stack('history_valid'),history_episode_id=stack('history_episode_id'))
            yield batch,[dict(r['record'],granularity=r['granularity']) for r in rows]


def audit_full_source(output=ROOT/'data_audit.json',*,source_yaml=DEFAULT_SOURCE,index_path=DEFAULT_INDEX,code_dir=DEFAULT_CODE,metadata_view_manifest=None):
    bind_native(code_dir)
    if metadata_view_manifest:
        from frozen_metadata_view import make_reader
        policy=dict(metadata_view_manifest=metadata_view_manifest)
    else:
        from raw_metadata_overlay import make_reader
        policy={}
    from openwam.dataloader.goal_epoch import validate_index
    reader=make_reader(source_yaml,ROOT/'audit_reader_cache',**policy,subtask_probability=.7,
        max_cached_episodes=2,observation_mode='multiview',materialize_multiview_target=True)
    t=time.perf_counter();summary=validate_index(reader,index_path,sample_fps=3.)
    stream=CommonDrawStream(index_path,reader.repos,seed=42)
    groups={}
    for grain in ('task','subtask'):
        groups[grain]=[dict(repo_id=ri,repo=reader.repos[ri]['repo'],family=reader.repos[ri]['family'],
            yaml_weight=reader.repos[ri]['weight'],conditional_repo_probability=float(w),
            eligible_3fps_anchors=int(group[1][-1])) for ri,w,group in zip(stream.repo_ids[grain],stream.weights[grain],stream.groups[grain])]
    with closing(sqlite3.connect(f'{Path(index_path).resolve().as_uri()}?mode=ro',uri=True)) as c:
        source_count=c.execute('SELECT COUNT(*) FROM sources').fetchone()[0]
    report=dict(format='openwam.longtrain.full-source-audit.v1',source_yaml=str(Path(source_yaml).resolve()),
        source_yaml_sha256=sha256_file(source_yaml),population=summary,source_dependencies_checked=source_count,
        scope=reader.scope_summary(),sampling=stream.contract,grain_repositories=groups,
        raw_camera_policy=reader.raw_overlay,
        native_source_validation='native validate_index: source YAML, content identity, complete interval coverage, EVERY recorded source size+mtime; no video decode',
        lazy_runtime_validation='selected native labels episode digest; recorded source stamps; video generations before+after materialization',
        corpus_rgb_content_hash=False,source_generation_contract='immutable CPFS files; normal inode/size/mtime/ctime changes fail or produce a new cache key',
        exclusions='only previously authorized retained source YAML excludes missing 133; no additional exclusions/redraws',
        validation='fixed64 from prior comparison is fitting feedback; may overlap full training; not held-out',
        elapsed_seconds=time.perf_counter()-t)
    atomic_json(output,report);return report


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    a=sub.add_parser('audit');a.add_argument('--output',default=str(ROOT/'data_audit.json'))
    a.add_argument('--metadata-view-manifest');a.add_argument('--source-yaml',default=str(DEFAULT_SOURCE));a.add_argument('--index-path',default=str(DEFAULT_INDEX));a.add_argument('--code-dir',default=str(DEFAULT_CODE))
    a=sub.add_parser('sample');a.add_argument('--count',type=int,default=100);a.add_argument('--cursor',type=int,default=0);a.add_argument('--output',default=str(ROOT/'sampling_preview.json'))
    args=parser.parse_args()
    if args.command=='audit':
        report=audit_full_source(args.output,source_yaml=args.source_yaml,index_path=args.index_path,code_dir=args.code_dir,metadata_view_manifest=args.metadata_view_manifest)
        print(canonical(dict(output=args.output,total_samples=report['population']['total_samples'],episodes=report['population']['completed_episodes'],elapsed=report['elapsed_seconds'])))
    else:
        bind_native();from openwam.dataloader.curated_goal import parse_curated_yaml
        selection=parse_curated_yaml(DEFAULT_SOURCE);stream=CommonDrawStream(DEFAULT_INDEX,selection['repos'])
        values=[stream.selection(i) for i in range(args.cursor,args.cursor+args.count)]
        atomic_json(args.output,dict(contract=stream.contract,selections=values));print(canonical(dict(output=args.output,count=len(values))))

if __name__=='__main__':main()
