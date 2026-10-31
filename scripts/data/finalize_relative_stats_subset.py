#!/usr/bin/env python3
"""Verify/rebind unchanged complete repository shards to a strict YAML subset.

The old scan remains untouched. Run this watcher in the foreground or under the
usual process supervisor; it polls pending repository checkpoints until complete
and fails if the old runner stops with unfinished retained repositories.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import time

from scripts.data import prepare_full_relative_stats as runner
from openwam.dataloader.curated_stage2 import CuratedStage2Dataset
from openwam.dataloader.native_stage2_profiles import fingerprint


def validate_subset(old,new,old_audit,new_audit):
    old_repos={repo['repo']:repo for repo in old.repos}
    new_repos={repo['repo']:repo for repo in new.repos}
    if not set(new_repos)<set(old_repos): raise ValueError('new selection must be a strict subset')
    for name,repo in new_repos.items():
        if fingerprint(repo)!=fingerprint(old_repos[name]):
            raise ValueError(f'retained repository changed: {name}')
    old_contract=old.stats_contract(); new_contract=new.stats_contract()
    if old_audit['contract']!=old_contract or new_audit['contract']!=new_contract:
        raise ValueError('audit contract mismatch')
    old_contract.pop('selection_sha256'); new_contract.pop('selection_sha256')
    if old_contract!=new_contract: raise ValueError('reader contract changed beyond selection hash')
    if new_audit['errors'] or set(new_audit['sources'])!=set(new_repos):
        raise ValueError('new metadata audit is incomplete')
    grouped=[name for group in new_audit['groups'].values() for name in group['repos']]
    if len(grouped)!=len(new_repos) or set(grouped)!=set(new_repos):
        raise ValueError('new metadata profile grouping is incomplete/duplicated')
    for name,source in new_audit['sources'].items():
        if source!=old_audit['sources'].get(name): raise ValueError(f'audited retained source changed: {name}')
        key=source['entry']
        group=new_audit['groups'].get(key)
        if (not group or name not in group['repos'] or
            group['contract']!=old_audit['groups'][key]['contract']):
            raise ValueError(f'retained representation changed: {name}')


def old_runner_active(output):
    path=Path(output)/'.runner.lock'
    if not path.exists(): return False
    with path.open('r') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: return True
        fcntl.flock(lock,fcntl.LOCK_UN)
    return False


def migration_identity(old,new,old_directory):
    return dict(format='openwam.relative_stats_subset_migration.v1',
        old_source_yaml=old.selection['source_yaml'],old_source_sha256=old.source_yaml_sha256,
        new_source_yaml=new.selection['source_yaml'],new_source_sha256=new.source_yaml_sha256,
        old_checkpoint=str((old_directory/'checkpoint.json').resolve()),
        old_checkpoint_sha256=runner.sha256(old_directory/'checkpoint.json'),
        migration_code_sha256=runner.sha256(__file__),
        rule='strict subset; unchanged retained paths/ranges/weights/metadata; verified immutable histogram blobs')


def transfer_repository(old,new,old_audit,new_audit,old_output,new_output,repo,code):
    name=repo['repo']; old_directory=runner.shard_directory(old_output,name)
    directory=runner.shard_directory(new_output,name)
    pointer=old_directory/'checkpoint.json'
    if not pointer.exists(): return False
    header=json.loads(pointer.read_text())
    if not header['done']: return False
    old_repo=next(item for item in old.repos if item['repo']==name)
    old_identity=runner.repository_identity(old,old_repo,old_audit['sources'][name],code)
    old_pointer_hash=runner.sha256(pointer)
    shard=runner.load_checkpoint(old_directory,old_identity)
    expected=sorted({value[0] for value in repo['ranges']})
    if shard['completed_episodes']!=expected: raise ValueError(f'incomplete done checkpoint: {name}')
    if runner.repository_files(repo)!=shard['source_files']: raise ValueError(f'source inventory changed: {name}')
    runner.verify_repository(new,repo,new_audit['sources'][name])
    if runner.sha256(pointer)!=old_pointer_hash: raise ValueError(f'old checkpoint changed during verification: {name}')
    new_identity=runner.repository_identity(new,repo,new_audit['sources'][name],code)
    provenance=migration_identity(old,new,old_directory)
    if (directory/'checkpoint.json').exists():
        existing=runner.load_checkpoint(directory,new_identity)
        if not existing['done'] or existing.get('migration')!=provenance:
            raise ValueError(f'existing migrated checkpoint/provenance mismatch: {name}')
        return True
    directory.mkdir(parents=True,exist_ok=True)
    for blob in shard['statistics'].values():
        # Checkpoint filenames must remain local to the source and destination.
        if Path(blob['file']).name!=blob['file']: raise ValueError('invalid checkpoint blob filename')
        source=old_directory/blob['file']; target=directory/blob['file']
        if not target.exists():
            temporary=target.with_name(target.name+f'.{os.getpid()}.tmp')
            temporary.unlink(missing_ok=True)
            try: os.link(source,temporary)
            except OSError: shutil.copyfile(source,temporary)
            with temporary.open('rb') as stream: os.fsync(stream.fileno())
            if runner.sha256(temporary)!=blob['sha256']: raise ValueError(f'transferred statistics changed: {name}')
            temporary.replace(target)
        if runner.sha256(target)!=blob['sha256']: raise ValueError(f'existing statistics blob mismatch: {name}')
    payload={key:value for key,value in shard.items() if key not in ('action','state')}
    payload['identity']=new_identity; payload['migration']=provenance
    if runner.sha256(pointer)!=old_pointer_hash: raise ValueError(f'old checkpoint changed before transfer commit: {name}')
    runner.verify_files(payload['source_files'])
    runner.atomic_json(directory/'checkpoint.json',payload)
    runner.atomic_json(directory/'progress.json',dict(repo=name,status='complete',migration=provenance,
        completed_episodes=len(expected),selected_episodes=len(expected),counts=payload['counts'],updated_at=time.time()))
    return True


def verify_native_consumer(source_yaml,readiness,audit):
    from openwam.dataloader.native_stage2_stats import lookup_stats
    from openwam.train.stage12 import validate_precomputed_registry
    dataset=CuratedStage2Dataset(source_yaml,normalization_registry=readiness['stats_registry'])
    validate_precomputed_registry(dataset,audit)
    # Metadata-only lookup checks every retained native profile using audited identities.
    from types import SimpleNamespace
    for repo in dataset.repos:
        source=audit['sources'][repo['repo']]
        metadata=dict(representation=SimpleNamespace(contract=audit['groups'][source['entry']]['contract']),
            info_hash=source['info_sha256'],calibration_sha256=source['calibration_sha256'])
        lookup_stats(dataset.registry,repo,metadata)
    return readiness


def finalize(old,new,old_audit,new_audit,old_output,new_output,*,poll_seconds=30):
    if poll_seconds<=0: raise ValueError('poll_seconds must be positive')
    old_output=Path(old_output).resolve(); new_output=Path(new_output).resolve()
    if old_output==new_output: raise ValueError('old and new output directories must differ')
    new_output.mkdir(parents=True,exist_ok=True)
    lock=(new_output/'.runner.lock').open('a')
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except Exception: lock.close(); raise
    started=time.time(); completed=set(); pending={repo['repo'] for repo in new.repos}
    status_base=dict(old_output=str(old_output),new_output=str(new_output),
        old_source_sha256=old.source_yaml_sha256,new_source_sha256=new.source_yaml_sha256,
        selected_repositories=len(new.repos),excluded_repositories=sorted({r['repo'] for r in old.repos}-pending),
        migration_code_sha256=runner.sha256(__file__),started_at=started)
    def status(state,**extra):
        runner.atomic_json(new_output/'migration_progress.json',dict(status_base,status=state,
            completed_repositories=len(completed),pending_repositories=sorted(pending),updated_at=time.time(),**extra))
    try:
        validate_subset(old,new,old_audit,new_audit)
        code=runner.code_identity()
        status('running',code_sha256=code)
        def verify_code():
            if runner.code_identity()!=code: raise ValueError('statistics code changed while waiting')
            if runner.sha256(__file__)!=status_base['migration_code_sha256']:
                raise ValueError('finalizer code changed while waiting or before publication')
        def transfer_pending():
            for repo in new.repos:
                name=repo['repo']
                if name not in pending: continue
                if transfer_repository(old,new,old_audit,new_audit,old_output,new_output,repo,code):
                    completed.add(name); pending.remove(name)
                    if len(completed)%50==0: status('running',code_sha256=code)
                else:
                    progress=runner.shard_directory(old_output,name)/'progress.json'
                    if progress.exists():
                        record=json.loads(progress.read_text())
                        if record.get('status')=='failed': raise ValueError(f'old repository failed: {name}: {record.get("error")}')
        while pending:
            verify_code()
            transfer_pending()
            status('waiting' if pending else 'publishing',code_sha256=code)
            if pending:
                if not old_runner_active(old_output):
                    # The old job can commit its final pointer between our first
                    # poll and releasing its lock. Re-read once after observing
                    # the inactive lock before declaring unfinished work dead.
                    verify_code()
                    transfer_pending()
                    if pending:
                        raise ValueError(f'old runner stopped with unfinished retained repositories: {sorted(pending)}')
                    status('publishing',code_sha256=code)
                else:
                    verify_code()
                    # Bounded sleeps keep foreground supervisors responsive.
                    time.sleep(min(poll_seconds,60))
        verify_code()
        readiness=runner.publish(new,new_audit,new_output,code)
        try:
            verify_code()
            verify_native_consumer(new.selection['source_yaml'],readiness,new_audit)
        except Exception:
            (new_output/'readiness.json').unlink(missing_ok=True)
            raise
        runner.atomic_json(new_output/'migration.json',dict(status_base,status='complete',code_sha256=code,
            completed_repositories=len(completed),registry_sha256=readiness['registry_sha256'],
            stats_registry=readiness['stats_registry'],consumer_validation='native dataset contract, all source/profile lookups and Stage12 payload checks passed'))
        runner.atomic_json(new_output/'progress.json',dict(status='complete',coverage=readiness['coverage'],
            stats_registry=readiness['stats_registry'],migration='migration.json',updated_at=time.time()))
        status('complete',readiness=readiness)
        return readiness
    except Exception as exc:
        (new_output/'readiness.json').unlink(missing_ok=True)
        status('failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        fcntl.flock(lock,fcntl.LOCK_UN); lock.close()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ('old-yaml','new-yaml','old-output','new-output','old-audit','new-audit'):
        parser.add_argument('--'+flag,required=True)
    parser.add_argument('--poll-seconds',type=float,default=30)
    args=parser.parse_args(argv)
    old=CuratedStage2Dataset(args.old_yaml); new=CuratedStage2Dataset(args.new_yaml)
    result=finalize(old,new,json.loads(Path(args.old_audit).read_text()),json.loads(Path(args.new_audit).read_text()),
        args.old_output,args.new_output,poll_seconds=args.poll_seconds)
    print(json.dumps(result),flush=True)
    return 0


if __name__=='__main__': raise SystemExit(main())
