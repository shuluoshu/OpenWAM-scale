#!/usr/bin/env python3
"""Exhaustive CPU native fixed-anchor relative112 statistics, resumable by episode.

Run as ``python -m scripts.data.prepare_full_relative_stats --help``. Only a
complete selected population can publish a normalization registry. A smoke
subset leaves reusable repository checkpoints and a partial progress report.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time
import uuid
import numpy as np

from openwam.dataloader.curated_stage2 import CuratedStage2Dataset
from openwam.dataloader.episode_metadata import offline_metadata_cache
from openwam.dataloader.native_relative_population import RelativeEpisodeEncoder
from openwam.dataloader.native_stage2_io import load_episode
from openwam.dataloader.native_stage2_profiles import fingerprint
from openwam.dataloader.native_stage2_stats import audit_selection, ROTATION_DIMS
from openwam.dataloader.relative_streaming_stats import StreamingStats
from openwam.dataloader.utils.mora_relative import MoraHeadTCPRepresentation

POPULATION_POLICY = {
    'version': 'exhaustive_fixed_anchor_relative112_v1',
    'selection': 'all positive-weight YAML repositories and every selected episode/range',
    'enumeration': 'all native _options task and subtask anchors, distinct level/windows, unweighted',
    'level_weighting': 'both available levels contribute; not the training 1:1 level mixture',
    'repository_weighting': 'positive inclusion only; no resampling or numerical weighting',
    'action': 'each valid future cell at offsets 1..32 relative to fixed anchor; gripper absolute',
    'state': 'one state per eligible level/window anchor',
    'sampling': False, 'clipping': False, 'raw_dimensions':52, 'unified_dimensions':112, 'horizon':32,
    'histogram': {'bins':32768, 'relative_bin_width':.001, 'scale':1e-6},
}


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()


def atomic_json(path,payload):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('w') as stream:
        json.dump(payload,stream,sort_keys=True,separators=(',',':'),allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor=os.open(path.parent,os.O_RDONLY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


def snapshot_files(paths):
    result={}
    for path in sorted({str(Path(p).resolve()) for p in paths}):
        stat=Path(path).stat()
        result[path]={'size':stat.st_size,'mtime_ns':stat.st_mtime_ns}
    return result


def verify_files(snapshot):
    for path,expected in snapshot.items():
        try: actual=snapshot_files([path])[path]
        except OSError as exc: raise ValueError(f'source changed: {path}: {exc}') from exc
        if actual!=expected: raise ValueError(f'source changed: {path}')


def repository_files(repo):
    root=Path(repo['path'])
    # Metadata, annotations and scalar parquet only: never touch video payloads.
    paths=[p for p in (root/'meta').rglob('*') if p.is_file()]
    paths.extend((root/'data').rglob('*.parquet'))
    annotation=repo.get('native_metadata',{}).get('annotation_file')
    if annotation: paths.append(Path(annotation))
    return snapshot_files(paths)


def code_identity():
    root=Path(__file__).resolve().parents[2]
    paths=[*sorted((root/'openwam/dataloader').rglob('*.py')),
           Path(__file__).resolve(),root/'scripts/data/prepare_goal_labels.py',
           root/'scripts/data/prepare_subtask_goals.py',root/'openwam/kinematics.py',root/'openwam/train/stage12.py']
    return fingerprint({str(p.relative_to(root)):sha256(p) for p in paths})


def accumulate_episode(dataset,metadata,ranges,action,state,*,batch_size,progress=None):
    if batch_size<1: raise ValueError('batch_size must be positive')
    encoder=RelativeEpisodeEncoder(metadata)
    counts={'anchors':0,'task_anchors':0,'subtask_anchors':0,'options':0}
    for level in ('task','subtask'):
        for option in dataset._options(metadata,ranges,level):
            counts['options']+=1
            for start in range(option['start'],option['end_exclusive'],batch_size):
                anchors=np.arange(start,min(start+batch_size,option['end_exclusive']))
                a,mask,s,sm=encoder.encode(anchors,np.full(len(anchors),option['last_source']+1))
                action.update(a,mask); state.update(s,sm)
                counts['anchors']+=len(anchors); counts[level+'_anchors']+=len(anchors)
                if progress: progress(counts)
    return counts


def save_checkpoint(directory,identity,completed_episodes,files,action,state,counts,*,done):
    """Write new immutable stats first, then atomically switch the sole pointer.

    A crash before the pointer replacement replays only uncommitted episodes.
    Old checkpoint blobs remain recoverable until the new pointer is durable.
    """
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    old=json.loads((directory/'checkpoint.json').read_text()) if (directory/'checkpoint.json').exists() else None
    generation=f'{len(completed_episodes):08d}-{uuid.uuid4().hex}'
    stats={}
    for name,obj in (('action',action),('state',state)):
        path=directory/f'{generation}-{name}.npz'; obj.save(path)
        stats[name]={'file':path.name,'sha256':sha256(path)}
    payload={'identity':identity,'completed_episodes':list(completed_episodes),
             'source_files':files,'statistics':stats,'counts':counts,'done':done,'updated_at':time.time()}
    atomic_json(directory/'checkpoint.json',payload)
    if old:
        for item in old['statistics'].values():
            (directory/item['file']).unlink(missing_ok=True)
    return payload


def load_checkpoint(directory,identity):
    directory=Path(directory); path=directory/'checkpoint.json'
    if not path.exists(): return None
    payload=json.loads(path.read_text())
    if fingerprint(payload['identity'])!=fingerprint(identity): raise ValueError('checkpoint identity mismatch')
    verify_files(payload['source_files'])
    for name,item in payload['statistics'].items():
        blob=directory/item['file']
        if sha256(blob)!=item['sha256']: raise ValueError('checkpoint statistics hash mismatch')
        payload[name]=StreamingStats.load(blob)
    return payload


def verify_repository(dataset,repo,expected):
    # Re-audit the source's native representation and every selected metadata row.
    from types import SimpleNamespace
    one=SimpleNamespace(repos=[repo],filter_config=dataset.filter_config,stats_contract=dataset.stats_contract)
    actual=audit_selection(one)
    if actual['errors'] or actual['sources'].get(repo['repo'])!=expected:
        raise ValueError(f'{repo["repo"]}: audited source identity changed: {actual["errors"][:2]}')


def repository_identity(dataset,repo,source,code):
    return {'population_policy':POPULATION_POLICY,'contract':dataset.stats_contract(),
            'repo':repo,'source':source,'code_sha256':code}


_WORKER_DATASET=None


def initialize_worker(source_yaml,path_aliases):
    global _WORKER_DATASET
    import torch
    torch.set_num_threads(1)
    _WORKER_DATASET=CuratedStage2Dataset(source_yaml,path_aliases=path_aliases)


@offline_metadata_cache()
def run_repository(job,dataset=None):
    dataset=dataset or _WORKER_DATASET
    repo=job['repo']; directory=Path(job['directory']); directory.mkdir(parents=True,exist_ok=True)
    identity=repository_identity(dataset,repo,job['source'],job['code'])
    if code_identity()!=job['code']: raise ValueError('runner dependency code changed')
    verify_repository(dataset,repo,job['source'])
    current_files=repository_files(repo)
    loaded=load_checkpoint(directory,identity)
    if loaded and loaded['source_files']!=current_files: raise ValueError('source changed: repository file inventory')
    action=loaded['action'] if loaded else StreamingStats()
    state=loaded['state'] if loaded else StreamingStats()
    completed=loaded['completed_episodes'] if loaded else []
    counts=dict(loaded['counts']) if loaded else dict(anchors=0,task_anchors=0,subtask_anchors=0,options=0)
    episodes=sorted({r[0] for r in repo['ranges']})
    if completed!=episodes[:len(completed)]: raise ValueError('checkpoint episodes not a selected prefix')
    if loaded and loaded['done']:
        if completed!=episodes: raise ValueError('incomplete done checkpoint')
        return {'repo':repo['repo'],'directory':str(directory),'episodes':len(completed),'counts':counts}
    last_progress=0.
    def report(episode,local=None,force=False):
        nonlocal last_progress
        now=time.monotonic()
        if not force and now-last_progress<15: return
        last_progress=now
        atomic_json(directory/'progress.json',dict(repo=repo['repo'],status='running',current_episode=episode,
            completed_episodes=len(completed),selected_episodes=len(episodes),counts=counts,current_episode_counts=local,
            action_valid_cells=int(action.count.sum()),state_valid_cells=int(state.count.sum()),updated_at=time.time()))
    try:
        for episode in episodes[len(completed):]:
            report(episode,force=True)
            m,_=load_episode(repo,episode,dataset.filter_config,256*1024*1024)
            if (m['info_hash']!=job['source']['info_sha256'] or
                m['calibration_sha256']!=job['source']['calibration_sha256'] or
                fingerprint(m['representation'].contract)!=job['source']['entry']):
                raise ValueError('loaded episode representation identity changed')
            # All files used by the projected reader must have been snapshotted
            # before loading, including shards spanning this episode.
            used=[str(Path(p).resolve()) for p in m['data_files']]
            if any(p not in current_files for p in used): raise ValueError('source data outside initial file inventory')
            verify_files({p:current_files[p] for p in used})
            local=accumulate_episode(dataset,m,[r for r in repo['ranges'] if r[0]==episode],action,state,
                                     batch_size=job['batch_size'],progress=lambda c:report(episode,c))
            verify_files({p:current_files[p] for p in used})
            for key,value in local.items(): counts[key]+=value
            completed.append(episode)
            del m
            if len(completed)%job['checkpoint_episodes']==0:
                verify_files(current_files)
                save_checkpoint(directory,identity,completed,current_files,action,state,counts,done=False)
        verify_repository(dataset,repo,job['source']); verify_files(current_files)
        if repository_files(repo)!=current_files: raise ValueError('source changed: repository file inventory')
        if code_identity()!=job['code']: raise ValueError('runner dependency code changed')
        save_checkpoint(directory,identity,completed,current_files,action,state,counts,done=True)
        atomic_json(directory/'progress.json',dict(repo=repo['repo'],status='complete',completed_episodes=len(completed),
            selected_episodes=len(episodes),counts=counts,action_valid_cells=int(action.count.sum()),
            state_valid_cells=int(state.count.sum()),updated_at=time.time()))
        return {'repo':repo['repo'],'directory':str(directory),'episodes':len(completed),'counts':counts}
    except Exception as exc:
        atomic_json(directory/'progress.json',dict(repo=repo['repo'],status='failed',error=f'{type(exc).__name__}: {exc}',
            completed_episodes=len(completed),counts=counts,updated_at=time.time()))
        raise


def require_complete(dataset,audit,checkpoints):
    selected={r['repo'] for r in dataset.repos}
    if (audit['errors'] or set(audit['sources'])!=selected or set(checkpoints)!=selected):
        raise ValueError('incomplete selected repository metadata/population')
    grouped=[name for group in audit['groups'].values() for name in group['repos']]
    if len(grouped)!=len(selected) or set(grouped)!=selected:
        raise ValueError('incomplete or duplicated profile grouping')
    for repo in dataset.repos:
        checkpoint=checkpoints[repo['repo']]
        if checkpoint['identity'].get('population_policy')!=POPULATION_POLICY:
            raise ValueError('statistics population is not exhaustive; legacy samples cannot claim full coverage')
        if not checkpoint['done'] or checkpoint['completed_episodes']!=sorted({v[0] for v in repo['ranges']}):
            raise ValueError('incomplete selected episodes')


def json_vectors(summary):
    return {key:[None if not np.isfinite(v) else int(v) if key=='count' else float(v) for v in values]
            for key,values in summary.items()}


def registry_vectors(stats):
    result=stats.summary(); missing=stats.count==0
    for name in ('min','q01'): result[name][missing]=-1
    for name in ('max','q99','std'): result[name][missing]=1
    result['mean'][missing]=0
    for name in ('min','q01'): result[name][ROTATION_DIMS]=-1
    for name in ('max','q99','std'): result[name][ROTATION_DIMS]=1
    result['mean'][ROTATION_DIMS]=0
    return json_vectors(result)


def quantile_intervals(stats):
    result={}
    for label,q in (('q01',.01),('q99',.99)):
        estimate,low,high=stats.quantile(q)
        result[label]=json_vectors(dict(estimate=estimate,lower=low,upper=high,half_width=(high-low)/2))
    return result


TCP_DIMENSIONS=[*range(20),*range(34,52)]


def tcp_projection(stats):
    """Project before merging: joint cells never enter cross-profile statistics."""
    result=StreamingStats(len(TCP_DIMENSIONS),bins=stats.bins,relative_error=stats.relative_error,scale=stats.scale)
    for field in ('hist','count','mean','m2','minimum','maximum'):
        setattr(result,field,getattr(stats,field)[TCP_DIMENSIONS].copy())
    return result


def shard_directory(output,name): return Path(output)/'repos'/hashlib.sha256(name.encode()).hexdigest()


def publish(dataset,audit,output,code):
    output=Path(output); checkpoints={}
    # Gate before allocating/merging histogram profiles or writing any registry.
    for repo in dataset.repos:
        path=shard_directory(output,repo['repo'])/'checkpoint.json'
        if path.exists(): checkpoints[repo['repo']]=json.loads(path.read_text())
    require_complete(dataset,audit,checkpoints)
    tcp_action=StreamingStats(len(TCP_DIMENSIONS)); tcp_state=StreamingStats(len(TCP_DIMENSIONS))
    entries={}; coverage={'repositories':len(dataset.repos),'episodes':0,'anchors':0,'task_anchors':0,'subtask_anchors':0}
    for key,group in sorted(audit['groups'].items()):
        action=StreamingStats(); state=StreamingStats(); population=[]
        for name in sorted(group['repos']):
            repo=next(r for r in dataset.repos if r['repo']==name)
            verify_repository(dataset,repo,audit['sources'][name])
            identity=repository_identity(dataset,repo,audit['sources'][name],code)
            shard=load_checkpoint(shard_directory(output,name),identity)
            if shard['source_files']!=repository_files(repo): raise ValueError('source changed: repository file inventory')
            action.merge(shard['action']); state.merge(shard['state'])
            record=dict(repo=name,episodes=len(shard['completed_episodes']),**shard['counts'])
            population.append(record)
            coverage['episodes']+=record['episodes']
            for k in ('anchors','task_anchors','subtask_anchors'): coverage[k]+=record[k]
        tcp_action.merge(tcp_projection(action)); tcp_state.merge(tcp_projection(state))
        entries[key]=dict(contract=group['contract'],action=registry_vectors(action),state=registry_vectors(state),
            empirical=dict(action=json_vectors(action.summary()),state=json_vectors(state.summary())),
            quantile_intervals=dict(action=quantile_intervals(action),state=quantile_intervals(state)),
            coverage=dict(action_missing_raw_dimensions=np.flatnonzero(action.count==0).tolist(),
                state_missing_raw_dimensions=np.flatnonzero(state.count==0).tolist(),
                action_head_valid_cells=int(action.count[34:].sum()),state_head_valid_cells=int(state.count[34:].sum())),
            population=population,scope='complete selected unweighted distinct task/subtask window population')
    # Apply the same readiness checks as DLC reuse, without repeating its audit.
    from openwam.train.stage12 import validate_stats_payload
    for key,entry in entries.items():
        required=[20+7*side+i for side,name in enumerate(('left','right'))
                  for i in range(len(entry['contract']['joint_indices'][name]))]
        for stream in ('action','state'):
            validate_stats_payload(entry[stream],required,source=f'{key}:{stream}')
    # Repeat file and external physical-dependency checks after all merges.

    for repo in dataset.repos:
        verify_repository(dataset,repo,audit['sources'][repo['repo']])
        if repository_files(repo)!=checkpoints[repo['repo']]['source_files']:
            raise ValueError('source changed before publication')
    if code_identity()!=code or sha256(dataset.selection['source_yaml'])!=dataset.source_yaml_sha256:
        raise ValueError('code/source selection changed before publication')
    payload=dict(contract=dataset.stats_contract(),sources=audit['sources'],entries=entries,
        preparation=dict(population_policy=POPULATION_POLICY,complete=True,code_sha256=code,
            quantile_error='Per-dimension deterministic empirical quantile intervals and half-widths. Relative bin width is not a relative quantile-error guarantee across zero.',
            normalization_policy='native rotation6d identity ranges/std and missing-dimension fillers; true moments and extrema retained under empirical'),
        informational_tcp_summary=dict(normalization=False,
            scope='informational only; complete unweighted selected task/subtask window population across profiles; joints excluded',
            raw_dimensions=TCP_DIMENSIONS,local_raw_dimensions=list(range(20)),head_raw_dimensions=list(range(34,52)),
            action=json_vectors(tcp_action.summary()),state=json_vectors(tcp_state.summary()),
            quantile_intervals=dict(action=quantile_intervals(tcp_action),state=quantile_intervals(tcp_state))),
        dimensions=dict(raw=52,unified=112,raw_to_unified=MoraHeadTCPRepresentation.scatter.tolist()),coverage=coverage)
    raw=json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode(); digest=hashlib.sha256(raw).hexdigest()
    path=output/f'registry-{digest}.json'
    if path.exists() and path.read_bytes()!=raw: raise ValueError('immutable registry collision')
    if not path.exists():
        temporary=output/(path.name+'.tmp')
        with temporary.open('wb') as stream: stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        temporary.replace(path)
    readiness=dict(status='complete',stats_registry=str(path.resolve()),registry_sha256=digest,
        source_yaml=dataset.selection['source_yaml'],source_sha256=dataset.source_yaml_sha256,
        horizon=32,dimensions=112,raw_dimensions=52,coverage=coverage)
    atomic_json(output/'readiness.json',readiness)
    return readiness


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-yaml',required=True); parser.add_argument('--output',required=True)
    parser.add_argument('--metadata-audit',required=True); parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--batch-size',type=int,default=256); parser.add_argument('--checkpoint-episodes',type=int,default=100)
    parser.add_argument('--path-aliases',help='JSON file mapping source repo names to paths')
    parser.add_argument('--only-repo',action='append',default=[],help='Reusable smoke subset; never publishes partial registry')
    args=parser.parse_args(argv)
    if min(args.workers,args.batch_size,args.checkpoint_episodes)<1: parser.error('worker/batch/checkpoint counts must be positive')
    output=Path(args.output).resolve(); output.mkdir(parents=True,exist_ok=True)
    # Serialize writers to prevent two runs from replacing a checkpoint pointer.
    import fcntl
    lock=(output/'.runner.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (output/'readiness.json').unlink(missing_ok=True)
    aliases=json.loads(Path(args.path_aliases).read_text()) if args.path_aliases else None
    dataset=CuratedStage2Dataset(args.source_yaml,path_aliases=aliases)
    audit=json.loads(Path(args.metadata_audit).read_text())
    if audit['contract']!=dataset.stats_contract(): raise ValueError('metadata audit contract mismatch')
    names={r['repo'] for r in dataset.repos}
    if set(args.only_repo)-names: raise ValueError('unknown --only-repo')
    if set(audit['sources'])-names: raise ValueError('audit sources outside selected population')
    code=code_identity(); failures=[]; results={}; started=time.time()
    selected=[r for r in dataset.repos if not args.only_repo or r['repo'] in args.only_repo]
    jobs=iter([dict(repo=r,source=audit['sources'][r['repo']],code=code,
        directory=str(shard_directory(output,r['repo'])),batch_size=args.batch_size,checkpoint_episodes=args.checkpoint_episodes)
        for r in selected if r['repo'] in audit['sources']])
    def progress(status):
        atomic_json(output/'progress.json',dict(status=status,started_at=started,updated_at=time.time(),
            source_yaml=dataset.selection['source_yaml'],source_sha256=dataset.source_yaml_sha256,
            population_policy=POPULATION_POLICY,selected_repositories=len(dataset.repos),requested_repositories=len(selected),
            completed_repositories=len(results),completed_episodes=sum(x['episodes'] for x in results.values()),
            counts={k:sum(x['counts'][k] for x in results.values()) for k in ('anchors','task_anchors','subtask_anchors')},
            metadata_errors=audit['errors'],failures=failures,per_repository_progress='repos/<sha256(repo)>/progress.json'))
    progress('running')
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),
            initializer=initialize_worker,initargs=(str(Path(args.source_yaml).resolve()),aliases)) as pool:
        pending={}
        def submit():
            job=next(jobs,None)
            if job is not None: pending[pool.submit(run_repository,job)]=job['repo']['repo']
        for _ in range(args.workers): submit()
        while pending:
            finished,_=wait(pending,timeout=30,return_when=FIRST_COMPLETED)
            for future in finished:
                name=pending.pop(future)
                try: results[name]=future.result()
                except Exception as exc: failures.append(dict(repo=name,error=f'{type(exc).__name__}: {exc}'))
                submit()
            progress('running')
            if finished: print(json.dumps(dict(completed=len(results),failed=len(failures),selected=len(dataset.repos))),flush=True)
    if failures:
        progress('failed'); return 1
    try: readiness=publish(dataset,audit,output,code)
    except ValueError as exc:
        failures.append(dict(publication=str(exc)))
        status='partial' if str(exc).startswith('incomplete') else 'failed'
        progress(status); print(json.dumps(dict(status=status,reason=str(exc))),flush=True)
        return 0 if args.only_repo and status=='partial' else 1
    progress('complete'); print(json.dumps(readiness),flush=True); return 0


if __name__=='__main__': raise SystemExit(main())
