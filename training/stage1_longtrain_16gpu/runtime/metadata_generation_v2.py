"""Bounded proof/publish tools for immutable instruct_move metadata generation.

Modes proof and videos can run independently after the metadata view is copied.
Publish never updates the original source, original index, prior manifest/audit,
or any model. It retains the full population and copies a new immutable index.
"""
from __future__ import annotations
import argparse
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import shutil
import sqlite3
import threading
import time

import data_pipeline as dp
from frozen_metadata_view import GENERATION_ROOT,load_manifest,make_reader,_stamp
from rebind_population import row_digest,assert_same_runs

def reader(manifest):
    return make_reader(dp.DEFAULT_SOURCE,GENERATION_ROOT/'reader_cache',metadata_view_manifest=manifest,
        max_cached_episodes=2,observation_mode='multiview',materialize_multiview_target=True)


def prove_pnp(workers=8):
    from openwam.dataloader.goal_epoch import _prepare_episode_result
    manifest=GENERATION_ROOT/'manifest.json';m=load_manifest(manifest);d=reader(manifest)
    ri=5564;repo=d.repos[ri]
    if not repo['repo'].endswith('/conterfactual_before_1008/pnp'):raise ValueError('wrong newly changed source')
    old=sqlite3.connect(f'{dp.DEFAULT_ORIGINAL_INDEX.as_uri()}?mode=ro',uri=True)
    expected={int(ep):dict(digest=digest,excluded=excluded,runs=[]) for ep,excluded,digest in old.execute('SELECT episode,excluded,digest FROM episodes WHERE repo_id=?',(ri,))}
    for row in old.execute('SELECT count,repo_id,episode,level,segment,grid_start,numerator,denominator FROM runs WHERE repo_id=? ORDER BY sample_start',(ri,)):
        expected[row[2]]['runs'].append(tuple(row))
    local=threading.local();t=time.perf_counter()
    def compute(episode):
        if not hasattr(local,'dataset'):
            local.dataset=copy.copy(d);local.dataset._episodes=OrderedDict();local.dataset._evaluations=None
        ranges=[r for r in repo['ranges'] if r[0]==episode]
        result=_prepare_episode_result(local.dataset,(ri,repo,episode,ranges),3.)
        assert_same_runs(expected[episode]['runs'],result[2])
        if result[3]!=expected[episode]['excluded'] or result[4]!=expected[episode]['digest']:
            raise ValueError('new pnp original native episode/eligibility digest differs: '+str(episode))
        metadata=local.dataset._episode(repo,episode)
        entry=m['sources'][repo['repo']]
        if metadata['cameras']!=entry['cameras'] or str(metadata['root'])!=entry['physical_root']:
            raise ValueError('new pnp camera/physical source contract differs')
        return dict(episode=episode,digest=result[4],run_count=len(result[2]))
    results=[]
    with ThreadPoolExecutor(workers) as pool:
        for result in pool.map(compute,sorted(expected)):
            results.append(result)
            if len(results)%500==0:print(dp.canonical(dict(event='pnp_native_proof',complete=len(results),total=len(expected))),flush=True)
    report=dict(format='openwam.longtrain.pnp-full-native-proof.v2',manifest_sha256=dp.sha256_file(manifest),
        original_index_sha256=m['original_index_sha256'],repo_id=ri,selected_episodes=len(results),runs=sum(x['run_count'] for x in results),
        every_native_episode_digest_equals_original=True,every_compact_run_tuple_equals_original=True,
        rows=results,elapsed_seconds=time.perf_counter()-t)
    dp.atomic_json(GENERATION_ROOT/'pnp_proof.json',report);print(dp.canonical(dict(event='pnp_native_proof_complete',episodes=len(results),elapsed=report['elapsed_seconds'])),flush=True)


def snapshot_videos(workers=32):
    import pyarrow.parquet as pq
    manifest=load_manifest(GENERATION_ROOT/'manifest.json');old=sqlite3.connect(f'{dp.DEFAULT_ORIGINAL_INDEX.as_uri()}?mode=ro',uri=True)
    paths={};episode_paths={};t=time.perf_counter()
    for entry in manifest['sources'].values():
        ri=entry['repo_id'];selected={r[0] for r in old.execute('SELECT episode FROM episodes WHERE repo_id=?',(ri,))}
        view=Path(entry['view_root']);info=json.loads((view/'meta/info.json').read_text());cameras=[c for c in entry['cameras'] if c is not None]
        columns=['episode_index']+[f'videos/{camera}/{field}' for camera in cameras for field in ('chunk_index','file_index','from_timestamp','to_timestamp')]
        for file in entry['files']:
            if '/episodes/' not in file['relative']:continue
            table=pq.read_table(file['path'],columns=columns)
            for ep in table.to_pylist():
                episode=int(ep['episode_index'])
                if episode not in selected:continue
                values=[]
                for camera in cameras:
                    prefix='videos/'+camera+'/'
                    path=(Path(entry['physical_root'])/info['video_path'].format(video_key=camera,episode_index=episode,
                        chunk_index=int(ep[prefix+'chunk_index']),file_index=int(ep[prefix+'file_index'])))
                    values.append(str(path));paths[str(path)]=None
                episode_paths[f'{ri}:{episode}']=values
    names=sorted(paths);paths={};canonical_paths={}
    with ThreadPoolExecutor(workers) as pool:
        for requested,generation in zip(names,pool.map(dp.file_generation,names)):
            paths[generation['path']]=generation;canonical_paths[requested]=generation['path']
    episode_paths={key:[canonical_paths[path] for path in values] for key,values in episode_paths.items()}
    report=dict(format='openwam.longtrain.physical-video-snapshot.v2',manifest_sha256=dp.sha256_file(GENERATION_ROOT/'manifest.json'),
        physical_video_files=len(paths),selected_episodes=len(episode_paths),videos=paths,episode_paths=episode_paths,
        historical_video_identity='not claimed; fresh current original effective video generation, fixed original camera paths/locators',elapsed_seconds=time.perf_counter()-t)
    dp.atomic_json(GENERATION_ROOT/'physical_video_snapshot.json',report)
    print(dp.canonical(dict(event='physical_video_snapshot_complete',files=len(paths),episodes=len(episode_paths),elapsed=report['elapsed_seconds'])),flush=True)


def publish():
    from openwam.dataloader.goal_epoch import _identity,validate_index
    manifest_path=GENERATION_ROOT/'manifest.json';manifest=load_manifest(manifest_path)
    pnp=json.loads((GENERATION_ROOT/'pnp_proof.json').read_text());videos=json.loads((GENERATION_ROOT/'physical_video_snapshot.json').read_text())
    if pnp['manifest_sha256']!=dp.sha256_file(manifest_path) or videos['manifest_sha256']!=dp.sha256_file(manifest_path):raise ValueError('generation proofs differ')
    if pnp['selected_episodes']!=7035 or pnp['every_native_episode_digest_equals_original'] is not True:raise ValueError('new pnp proof incomplete')
    regression=json.loads((GENERATION_ROOT/'native_probe.json').read_text())
    if regression['manifest_sha256']!=dp.sha256_file(manifest_path) or regression['all_frozen_roots_tested']!=15 or len(regression['native_digest_probes'])!=30 or len(regression['native_render_probes'])!=15:
        raise ValueError('all-root native digest/renderer regression incomplete')
    previous=json.loads((dp.ROOT/'population_rebind_receipt.json').read_text())
    if previous['affected_episodes']!=6718 or previous['all_affected_native_episode_digests_equal_original'] is not True or previous['exact_semantic_run_equality'] is not True:
        raise ValueError('reused original 6718 proof incomplete')
    if previous['old_sha256']!=manifest['original_index_sha256']:raise ValueError('reused proof original population differs')
    current=dp.DEFAULT_INDEX;newpath=GENERATION_ROOT/'population.sqlite'
    if newpath.exists():raise ValueError('new index path already exists; inspect before retry')
    old_sha=dp.sha256_file(current)
    if old_sha!=previous['new_sha256']:raise ValueError('prior index changed')
    d=reader(manifest_path);old=sqlite3.connect(f'{current.as_uri()}?mode=ro',uri=True)
    scope={entry['repo_id'] for entry in manifest['sources'].values()}
    mapping={file['old_path']:file['path'] for entry in manifest['sources'].values() for file in entry['files']}
    prior_raw=json.loads((dp.ROOT/'raw_video_current_snapshot.json').read_text())['videos']
    obsolete=set(previous['added_source_snapshot_paths'])-set(prior_raw)
    before=dict(runs=row_digest(old.execute('SELECT * FROM runs ORDER BY sample_start')),
        episodes=row_digest(old.execute('SELECT * FROM episodes ORDER BY repo_id,episode')),
        metadata_except_selection=row_digest(old.execute("SELECT * FROM metadata WHERE key!='selection_sha256' ORDER BY key")),
        episode_sources_outside_frozen_scope=row_digest(row for row in old.execute('SELECT * FROM episode_sources ORDER BY repo_id,episode,path') if row[0] not in scope))
    # Before publication, prove all physical videos still match their snapshot.
    with ThreadPoolExecutor(32) as pool:
        for generation in pool.map(dp.file_generation,sorted(videos['videos'])):
            if generation!=videos['videos'][generation['path']]:raise ValueError('physical video changed during generation proof')
    identity=_identity(d,workers=32);shutil.copyfile(current,newpath);newpath.chmod(0o600);new=sqlite3.connect(newpath)
    with new:
        new.execute('CREATE INDEX IF NOT EXISTS episode_sources_path_v2 ON episode_sources(path)')
        for path in obsolete:
            new.execute('DELETE FROM episode_sources WHERE path=?',(path,));new.execute('DELETE FROM sources WHERE path=?',(path,))
        for original,frozen in mapping.items():
            stamp=_stamp(frozen)
            new.execute('UPDATE episode_sources SET path=? WHERE path=?',(frozen,original))
            new.execute('UPDATE sources SET path=?,size=?,mtime_ns=? WHERE path=?',(frozen,*stamp,original))
        for path in (str(manifest_path),manifest['aliases_file']):
            new.execute('INSERT OR IGNORE INTO sources VALUES (?,?,?)',(path,*_stamp(path)))
            for ri in scope:
                new.execute('INSERT OR IGNORE INTO episode_sources SELECT repo_id,episode,? FROM episodes WHERE repo_id=?',(path,ri))
        for path,generation in videos['videos'].items():
            expected=new.execute('SELECT size,mtime_ns FROM sources WHERE path=?',(path,)).fetchone()
            value=(generation['size'],generation['mtime_ns'])
            if expected is not None and expected!=value:raise ValueError('existing physical video guard changed; no stamp refresh allowed')
            new.execute('INSERT OR IGNORE INTO sources VALUES (?,?,?)',(path,*value))
        new.executemany('INSERT OR IGNORE INTO episode_sources VALUES (?,?,?)',
            ((int(key.split(':')[0]),int(key.split(':')[1]),path) for key,paths in videos['episode_paths'].items() for path in paths))
        new.execute("UPDATE metadata SET value=? WHERE key='selection_sha256'",(json.dumps(identity['selection_sha256']),))
    after=dict(runs=row_digest(new.execute('SELECT * FROM runs ORDER BY sample_start')),
        episodes=row_digest(new.execute('SELECT * FROM episodes ORDER BY repo_id,episode')),
        metadata_except_selection=row_digest(new.execute("SELECT * FROM metadata WHERE key!='selection_sha256' ORDER BY key")),
        episode_sources_outside_frozen_scope=row_digest(row for row in new.execute('SELECT * FROM episode_sources ORDER BY repo_id,episode,path') if row[0] not in scope))
    if before!=after:raise ValueError('population rows/episode labels/outside-scope dependencies changed')
    # No unread original metadata dependency may survive in the new catalog.
    for original in mapping:
        if new.execute('SELECT 1 FROM sources WHERE path=?',(original,)).fetchone():raise ValueError('unread original metadata remains a false dependency')
    unchanged_paths=set(row[0] for row in old.execute('SELECT path FROM sources'))-set(mapping)-obsolete
    def source_digest(connection):return row_digest(row for row in connection.execute('SELECT * FROM sources ORDER BY path') if row[0] in unchanged_paths)
    if source_digest(old)!=source_digest(new):raise ValueError('untouched physical/source guards changed')
    new.close();old.close();summary=validate_index(d,newpath,sample_fps=3.)
    if (summary['total_samples'],summary['completed_episodes'],summary['run_count'])!=(48547815,203242,794012):raise ValueError('full population changed')
    if dp.sha256_file(current)!=old_sha or dp.sha256_file(dp.DEFAULT_ORIGINAL_INDEX)!=manifest['original_index_sha256']:raise ValueError('old index modified')
    newpath.chmod(0o444)
    report=dict(format='openwam.longtrain.frozen-metadata-generation.v2',metadata_manifest=str(manifest_path),metadata_manifest_sha256=dp.sha256_file(manifest_path),
        new_index=str(newpath),population=summary,reused_previous_full_episode_proof=6718,unchanged_byte_equivalent_episodes=24036,
        newly_full_proved_pnp_episodes=7035,effective_metadata_scope_episodes=37789,
        all_global_episode_digests_and_runs_unchanged=True,before_row_hashes=before,after_row_hashes=after,
        replaced_original_metadata_paths=len(mapping),obsolete_v1_metadata_dependencies_removed=len(obsolete),
        physical_video_files=videos['physical_video_files'],original_indices_unchanged=True,source_files_modified=False,
        native_selection_identity_reads_frozen_sidecars=True,unread_original_metadata_dependencies_remaining=0)
    dp.atomic_json(GENERATION_ROOT/'receipt.json',report)
    audit=dp.audit_full_source(GENERATION_ROOT/'audit.json',index_path=newpath,metadata_view_manifest=manifest_path)
    options=dict(source_yaml=str(dp.DEFAULT_SOURCE),population_index=str(newpath),audit_file=str(GENERATION_ROOT/'audit.json'),
        metadata_view_manifest=str(manifest_path),path_aliases=manifest['aliases_file'])
    dp.atomic_json(GENERATION_ROOT/'production_data_options_v2.json',options)
    for file in ('receipt.json','audit.json','production_data_options_v2.json','copy_proof.json','pnp_proof.json','native_probe.json','physical_video_snapshot.json'):
        (GENERATION_ROOT/file).chmod(0o444)
    print(dp.canonical(dict(event='metadata_generation_v2_published',index=str(newpath),index_sha256=summary['index_sha256'],audit=str(GENERATION_ROOT/'audit.json'))),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['proof','videos','publish']);args=parser.parse_args()
    {'proof':prove_pnp,'videos':snapshot_videos,'publish':publish}[args.mode]()
if __name__=='__main__':main()
