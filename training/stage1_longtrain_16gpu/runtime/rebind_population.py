"""Rebind a NEW copy of native compact index after metadata-only repair.

No original source/index mutation. Every affected selected episode is recomputed
with the pinned native v3 builder. Run tuples (including instruction text,
endpoints, eligibility, 3fps grid) must match byte-for-byte before snapshots can
advance. All untouched SQL table rows are hashed before/after. A complete native
validate_index then checks every source and full coverage of the new copy.
"""
from __future__ import annotations
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import copy
from contextlib import closing
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import threading
import time
import os

import data_pipeline as dp


def row_digest(rows):
    h=hashlib.sha256()
    for row in rows:h.update(dp.canonical(row).encode());h.update(b'\n')
    return h.hexdigest()


def assert_same_runs(old,new):
    # New native builder omits only the preexisting global sample_start. These
    # offsets are retained unchanged in the copied index and checked globally.
    if old!=new:
        for i in range(max(len(old),len(new))):
            a=old[i] if i<len(old) else None;b=new[i] if i<len(new) else None
            if a!=b:raise ValueError('semantic native population changed: '+dp.canonical(dict(run=i,old=a,new=b)))
        raise AssertionError('unreachable run comparison')


def rebind(*,old_index=dp.DEFAULT_ORIGINAL_INDEX,source_yaml=dp.DEFAULT_SOURCE,
           new_index=Path('/mnt/data/cpfs/b5big/yeqingda/CodeX/longtrain_16gpu_20261112/population_raw_policy.sqlite'),workers=8):
    from openwam.dataloader.curated_goal import CuratedGoalDataset
    from openwam.dataloader.goal_epoch import (_prepare_episode_result,_identity,_selected_repos_digest,validate_index)
    start=time.perf_counter();old_index=Path(old_index).resolve();new_index=Path(new_index).resolve()
    if new_index==old_index or new_index.exists():raise ValueError('new index must be a NEW path')
    source_yaml=Path(source_yaml).resolve()
    from raw_metadata_overlay import make_reader,load_overlay_manifest
    overlay=load_overlay_manifest()
    reader=make_reader(source_yaml,dp.ROOT/'rebind_reader_cache',
        max_cached_episodes=2,observation_mode='multiview',materialize_multiview_target=True)
    old_stamp=dp.file_generation(old_index);old_sha=dp.sha256_file(old_index)
    old=sqlite3.connect(f'{old_index.as_uri()}?mode=ro',uri=True)
    metadata={k:json.loads(v) for k,v in old.execute('SELECT key,value FROM metadata')}
    if metadata.get('source_yaml_sha256')!=dp.sha256_file(source_yaml) or metadata.get('sample_fps')!=3. or metadata.get('total_samples')!=48547815:
        raise ValueError('wrong authorized original full-source population')
    rows=old.execute('SELECT path,size,mtime_ns FROM sources').fetchall()
    def check(row):
        path,size,mtime=row;p=Path(path)
        try:s=p.stat();now=(s.st_size,s.st_mtime_ns)
        except FileNotFoundError:now=(-1,-1)
        return dict(path=path,expected=[size,mtime],actual=list(now)) if now!=(size,mtime) else None
    with ThreadPoolExecutor(workers*4) as pool:changed=[r for r in pool.map(check,rows) if r]
    if not changed:raise ValueError('no changed dependencies: rebind unnecessary')
    if any(not r['path'].endswith(('/meta/info.json','/meta/episodes/chunk-000/file-000.parquet')) or min(r['actual'])<0 for r in changed):
        raise ValueError('only present audited info/episode metadata changes may be rebound')
    changed_paths={r['path'] for r in changed}
    affected=set()
    for path in changed_paths:affected.update(old.execute('SELECT repo_id,episode FROM episode_sources WHERE path=?',(path,)).fetchall())
    affected_repos={ri for ri,_ in affected}
    expected_repos={str(Path('/mnt/data/cpfs/b5big/post_train_data/instruct_move')/x) for x in ('conterfactual_20261016','conterfactual_before_1008/im')}
    if {reader.repos[ri]['path'] for ri in affected_repos}!=expected_repos or len(changed)!=4:
        raise ValueError('changed source scope differs from reviewed 4 files / 2 instruct_move repositories')
    # Added video features can change the native preference without changing
    # a single language/grid run. The recoverable original metadata must also
    # prove the exact camera generation, before any snapshot can be refreshed.
    from openwam.dataloader.native_cameras import resolve_native_cameras
    import pyarrow.parquet as pq
    camera_proofs=[]
    for ri in sorted(affected_repos):
        repo=reader.repos[ri];root=Path(repo['path']);backup=root/'meta/backup_before_undistorted_20261112'
        if not (backup/'info.json').is_file():raise ValueError('cannot prove original RGB generation: missing immutable info backup')
        before=json.loads((backup/'info.json').read_text());after=json.loads((root/'meta/info.json').read_text())
        old_cameras=resolve_native_cameras(repo['repo'],before);new_cameras=resolve_native_cameras(repo['repo'],after)
        effective_cameras=resolve_native_cameras(repo['repo'],json.loads((Path(overlay['sources'][repo['path']]['view_root'])/'meta/info.json').read_text()))
        old_table=pq.read_table(backup/'episodes/chunk-000/file-000.parquet')
        new_table=pq.read_table(root/'meta/episodes/chunk-000/file-000.parquet')
        changed_columns=[k for k in old_table.column_names if k not in new_table.column_names or not old_table[k].equals(new_table[k])]
        proof=dict(repo_id=ri,repo=repo['repo'],backup_info_sha256=dp.sha256_file(backup/'info.json'),
            backup_metadata_sha256=dp.sha256_file(backup/'episodes/chunk-000/file-000.parquet'),
            old_cameras=old_cameras,new_cameras=new_cameras,effective_raw_cameras=effective_cameras,changed_original_metadata_columns=changed_columns,
            added_metadata_columns=[k for k in new_table.column_names if k not in old_table.column_names],
            added_info_features=[k for k in after['features'] if k not in before['features']],camera_alignment=after.get('camera_alignment'))
        camera_proofs.append(proof)
    dp.atomic_json(dp.ROOT/'population_rebind_rgb_gate.json',dict(format='openwam.longtrain.rgb-generation-gate.v1',
        old_index=str(old_index),changed_files=changed,proofs=camera_proofs,
        permitted=all(p['old_cameras']==p['effective_raw_cameras'] and not p['changed_original_metadata_columns'] for p in camera_proofs)))
    if any(p['old_cameras']!=p['effective_raw_cameras'] or p['changed_original_metadata_columns'] for p in camera_proofs):
        raise ValueError('effective RAW camera generation changed despite overlay; see population_rebind_rgb_gate.json')
    print(dp.canonical(dict(event='rebind_scope',affected_episodes=len(affected),affected_repositories=sorted(affected_repos),changed=changed)),flush=True)
    # One scan: no unindexed per-episode runs lookup over the 794k-row table.
    oldruns={key:[] for key in affected}
    for row in old.execute('SELECT count,repo_id,episode,level,segment,grid_start,numerator,denominator FROM runs ORDER BY sample_start'):
        key=(row[1],row[2])
        if key in affected:oldruns[key].append(tuple(row))
    base_hashes={
        'runs':row_digest(old.execute('SELECT * FROM runs ORDER BY sample_start')),
        'metadata_except_selection':row_digest(old.execute("SELECT * FROM metadata WHERE key!='selection_sha256' ORDER BY key")),
        'episode_sources':row_digest(old.execute('SELECT * FROM episode_sources ORDER BY repo_id,episode,path')),
        'sources_unchanged':row_digest(row for row in old.execute('SELECT * FROM sources ORDER BY path') if row[0] not in changed_paths),
        'episodes_unaffected':row_digest(row for row in old.execute('SELECT * FROM episodes ORDER BY repo_id,episode') if (row[0],row[1]) not in affected),
    }
    local=threading.local()
    def compute(key):
        if not hasattr(local,'reader'):
            local.reader=copy.copy(reader);local.reader._episodes=OrderedDict();local.reader._evaluations=None
        ri,episode=key;repo=reader.repos[ri]
        ranges=[r for r in repo['ranges'] if r[0]==episode]
        result=_prepare_episode_result(local.reader,(ri,repo,episode,ranges),3.)
        actual=result[2]
        assert_same_runs(oldruns[key],actual)
        prior=old_expected[key]
        if result[3]!=prior['excluded']:raise ValueError('source validity exclusion count changed')
        # Even if bytes/stat changed, no dependencies may appear/disappear.
        if {r[0] for r in result[5]}!=prior['sources']:raise ValueError('affected episode source dependency set changed')
        if result[4]!=prior['digest']:raise ValueError('effective native RAW episode digest differs from ORIGINAL index: '+str(key))
        m=local.reader._episode(repo,episode)
        if m['cameras']!=overlay['sources'][repo['path']]['cameras']:raise ValueError('RAW camera policy escaped')
        raw=[dp.file_generation(dp._video_path(m,episode,camera)[0]) for camera in m['cameras'] if camera is not None]
        return result,raw
    old_expected={key:dict(excluded=excluded,digest=digest,sources=set()) for ri,episode,excluded,digest in old.execute('SELECT * FROM episodes') if (key:=(ri,episode)) in affected}
    for ri,episode,path in old.execute('SELECT * FROM episode_sources'):
        if (ri,episode) in affected:old_expected[(ri,episode)]['sources'].add(path)
    results=[];raw_videos={};episode_raw_paths={}
    with ThreadPoolExecutor(workers) as pool:
        for result,raw in pool.map(compute,sorted(affected)):
            results.append(result);episode_raw_paths[(result[0],result[1])]=[g['path'] for g in raw]
            for generation in raw:
                previous=raw_videos.setdefault(generation['path'],generation)
                if previous!=generation:raise ValueError('RAW source changed during snapshot')
            if len(results)%500==0:print(dp.canonical(dict(event='native_episodes_verified',complete=len(results),total=len(affected))),flush=True)
    for generation in raw_videos.values():
        if dp.file_generation(generation['path'])!=generation:raise ValueError('RAW source changed during full affected episode proof')
    extra_paths=set(raw_videos)
    from raw_metadata_overlay import OVERLAY_MANIFEST
    extra_paths.add(str(OVERLAY_MANIFEST.resolve()))
    for entry in overlay['sources'].values():extra_paths.update(f['path'] for f in entry['files'])
    extra_stamps={p:(Path(p).stat().st_size,Path(p).stat().st_mtime_ns) for p in extra_paths}
    new_identity=_identity(reader,workers=workers*4)
    if new_identity['source_yaml_sha256']!=metadata['source_yaml_sha256'] or new_identity['version']!=3:
        raise ValueError('source YAML or builder version changed')
    # No clone is created until ALL affected semantic checks have succeeded.
    new_index.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(old_index,new_index)
    os.chmod(new_index,0o600)
    new=sqlite3.connect(new_index)
    with new:
        for ri,episode,runs,excluded,digest,sources in results:
            new.execute('UPDATE episodes SET digest=? WHERE repo_id=? AND episode=?',(digest,ri,episode))
        for item in changed:
            if check((item['path'],*item['actual'])) is not None:raise ValueError('source changed again during rebind')
            new.execute('UPDATE sources SET size=?,mtime_ns=? WHERE path=?',(*item['actual'],item['path']))
        new.execute("UPDATE metadata SET value=? WHERE key='selection_sha256'",(json.dumps(new_identity['selection_sha256']),))
        for path,stamp in extra_stamps.items():new.execute('INSERT OR IGNORE INTO sources VALUES (?,?,?)',(path,*stamp))
        for ri,episode in sorted(affected):
            entry=overlay['sources'][reader.repos[ri]['path']]
            paths=[str(OVERLAY_MANIFEST.resolve()),*[f['path'] for f in entry['files']],*episode_raw_paths[(ri,episode)]]
            for path in paths:new.execute('INSERT OR IGNORE INTO episode_sources VALUES (?,?,?)',(ri,episode,path))
    post_hashes={
        'runs':row_digest(new.execute('SELECT * FROM runs ORDER BY sample_start')),
        'metadata_except_selection':row_digest(new.execute("SELECT * FROM metadata WHERE key!='selection_sha256' ORDER BY key")),
        'episode_sources':row_digest(row for row in new.execute('SELECT * FROM episode_sources ORDER BY repo_id,episode,path') if row[2] not in extra_paths),
        'sources_unchanged':row_digest(row for row in new.execute('SELECT * FROM sources ORDER BY path') if row[0] not in changed_paths and row[0] not in extra_paths),
        'episodes_unaffected':row_digest(row for row in new.execute('SELECT * FROM episodes ORDER BY repo_id,episode') if (row[0],row[1]) not in affected),
    }
    if base_hashes!=post_hashes:raise ValueError('unexpected unchanged-table mutation during new-index rebind')
    old.close();new.close()
    summary=validate_index(reader,new_index,sample_fps=3.)
    if summary['total_samples']!=48547815 or summary['run_count']!=794012:raise ValueError('full population cardinality changed')
    if dp.file_generation(old_index)!=old_stamp or dp.sha256_file(old_index)!=old_sha:raise ValueError('original index changed during rebind')
    report=dict(format='openwam.longtrain.population-rebind.v1',old_index=str(old_index),old_sha256=old_sha,
        new_index=str(new_index),new_sha256=summary['index_sha256'],source_yaml=str(source_yaml),
        changed_files=changed,affected_repositories=[dict(repo_id=ri,repo=reader.repos[ri]['repo'],path=reader.repos[ri]['path'],
            selected_episodes=sum(r==ri for r,e in affected)) for ri in sorted(affected_repos)],
        affected_episodes=len(affected),affected_runs=sum(len(v) for v in oldruns.values()),
        exact_semantic_run_equality=True,unchanged_tables_sha256=base_hashes,after_unchanged_tables_sha256=post_hashes,
        old_selection_sha256=metadata['selection_sha256'],new_selection_sha256=new_identity['selection_sha256'],
        population=summary,old_index_unchanged=True,source_files_modified=False,exclusions_added=0,
        original_metadata_camera_proofs=camera_proofs,raw_camera_overlay=overlay,
        all_affected_native_episode_digests_equal_original=True,raw_current_video_files=len(raw_videos),
        raw_video_generation_contract='Fresh current RAW snapshot; historical video SHA/stamps were not recorded by original index. Every selected raw path/locator is original; current files exist and were stable before/after proof.',
        raw_video_snapshot_sha256=dp.digest_json(raw_videos),added_source_snapshot_paths=sorted(extra_paths),
        elapsed_seconds=time.perf_counter()-start)
    dp.atomic_json(dp.ROOT/'raw_video_current_snapshot.json',dict(format='openwam.longtrain.current-raw-video-snapshot.v1',videos=raw_videos))
    dp.atomic_json(dp.ROOT/'population_rebind_receipt.json',report)
    print(dp.canonical(dict(event='rebind_complete',new_index=str(new_index),episodes=len(affected),elapsed=report['elapsed_seconds'])),flush=True)
    return report


if __name__=='__main__':
    dp.bind_native();rebind()
