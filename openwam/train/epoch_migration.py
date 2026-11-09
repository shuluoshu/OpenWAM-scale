"""Explicit offline subset migrations; original runs are never modified.

Callers must verify the parent index against its checkpoint and that retained
repo selections/paths are unchanged, then run the normal full index validation.
This is a new experiment with inherited optimization state, not an identity bypass.
"""
from pathlib import Path
import json
import os
import shutil
import sqlite3
import uuid

from openwam.train.goal_epoch import atomic_json, sha256
from openwam.train.epoch_order import retained_parent_cursor, sampling_contract


def subset_index(parent, destination, repo_map, metadata_identity):
    parent, destination = Path(parent).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    if not repo_map or sorted(repo_map.values()) != list(range(len(repo_map))):
        raise ValueError('repository map must compact an ordered nonempty subset')
    if [repo_map[k] for k in sorted(repo_map)] != list(range(len(repo_map))):
        raise ValueError('repository map changes source order')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name+'.pending-'+uuid.uuid4().hex)
    intervals = []; total = parent_end = 0
    try:
        with sqlite3.connect(temporary) as c:
            c.execute('ATTACH DATABASE ? AS parent', (parent.as_uri()+'?mode=ro',))
            for (ddl,) in c.execute("SELECT sql FROM parent.sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall():
                c.execute(ddl)
            c.execute('CREATE TEMP TABLE mapping(old INTEGER PRIMARY KEY,new INTEGER UNIQUE)')
            c.executemany('INSERT INTO mapping VALUES (?,?)', sorted(repo_map.items()))
            c.execute('INSERT INTO metadata SELECT * FROM parent.metadata')
            c.execute('INSERT INTO episodes SELECT m.new,e.episode,e.excluded,e.digest FROM parent.episodes e JOIN mapping m ON m.old=e.repo_id')
            c.execute('INSERT INTO episode_sources SELECT m.new,e.episode,e.path FROM parent.episode_sources e JOIN mapping m ON m.old=e.repo_id')
            c.execute('INSERT INTO sources SELECT s.* FROM parent.sources s JOIN (SELECT DISTINCT path FROM episode_sources) e ON e.path=s.path')
            batch=[]
            for row in c.execute('SELECT * FROM parent.runs ORDER BY sample_start'):
                start,count,repo,*rest=row
                if start != parent_end or count <= 0:
                    raise ValueError('parent sample coverage is corrupt')
                parent_end = start+count
                if repo not in repo_map:
                    if intervals and intervals[-1][1] == start:
                        intervals[-1][1] = parent_end
                    else:
                        intervals.append([start,parent_end])
                    continue
                batch.append((total,count,repo_map[repo],*rest)); total += count
                if len(batch) >= 4096:
                    c.executemany('INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?)',batch);batch=[]
            c.executemany('INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?)',batch)
            old={k:json.loads(v) for k,v in c.execute('SELECT * FROM metadata')}
            if not old.get('complete') or old['total_samples'] != parent_end or total <= 0:
                raise ValueError('invalid parent/retained population')
            excluded=c.execute('SELECT COALESCE(SUM(excluded),0) FROM episodes').fetchone()[0]
            new={**metadata_identity,'total_samples':total,'excluded_invalid_anchor_pairs':excluded,'complete':True}
            c.executemany('INSERT OR REPLACE INTO metadata VALUES (?,?)',[(k,json.dumps(v,sort_keys=True,separators=(',',':'))) for k,v in new.items()])
            c.commit()
            if c.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                raise ValueError('derived SQLite integrity check failed')
        with temporary.open('rb') as stream:os.fsync(stream.fileno())
        temporary.rename(destination)
        fd=os.open(destination.parent,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
    finally:
        temporary.unlink(missing_ok=True)
    return dict(parent_total_samples=parent_end,total_samples=total,excluded_sample_ranges=intervals)


def migrate_checkpoint(parent, output, identity):
    """Publish new manifest/training state; inherit unmodified weight bundle bytes."""
    import torch
    parent,output=Path(parent).resolve(),Path(output).resolve()
    if (output/'latest.json').exists():raise FileExistsError(output/'latest.json')
    manifest=json.loads((parent/'manifest.json').read_text())
    stamps={}
    for name,digest in manifest['files'].items():
        if Path(name).name != name:raise ValueError('invalid checkpoint filename')
        path=parent/name;stamps[name]=(path.stat().st_size,path.stat().st_mtime_ns)
        if sha256(path)!=digest:raise ValueError('parent checkpoint checksum mismatch: '+name)
    old=manifest['identity']
    for key in ('world_size','seed','representation','data_loading'):
        if old.get(key)!=identity.get(key):raise ValueError('migration changes '+key)
    if old.get('training') is not None or identity.get('training') is not None:
        raise ValueError('only finite non-convergence epochs support subset migration')
    if old['sampling']!=sampling_contract({'seed':old['seed']},old['total_samples']):
        raise ValueError('parent must use the unchanged global shuffle/runtime')
    if identity['sampling'].get('parent_total_samples')!=old['total_samples']:
        raise ValueError('parent sampling population mismatch')
    cursor=retained_parent_cursor(identity['sampling'],manifest['cursor'])
    saved=torch.load(parent/'training.pt',map_location='cpu',weights_only=True,mmap=True)
    if saved['identity']!=old or any(saved[k]!=manifest[k] for k in ('cursor','update')):
        raise ValueError('parent training state disagrees with manifest')
    root=output/'checkpoints';root.mkdir(parents=True,exist_ok=True)
    temporary=root/('.pending-'+uuid.uuid4().hex);temporary.mkdir()
    destination=root/f'update_{manifest["update"]:09d}_migrated_{uuid.uuid4().hex[:8]}'
    try:
        saved.update(identity=identity,cursor=cursor)
        torch.save(saved,temporary/'training.pt')
        files={'training.pt':sha256(temporary/'training.pt')}
        for name,digest in manifest['files'].items():
            if name=='training.pt':continue
            if name=='convergence_state.json':raise ValueError('convergence migration is unsupported')
            # Independent files protect the original even if a later tool edits one.
            shutil.copyfile(parent/name,temporary/name)
            if sha256(temporary/name)!=digest:raise ValueError('copied bundle checksum mismatch')
            files[name]=digest
        lineage=dict(format='openwam.subset_continuation.v1',parent_checkpoint=str(parent),
                     parent_manifest_sha256=sha256(parent/'manifest.json'),parent_identity=old,
                     parent_cursor=manifest['cursor'],parent_update=manifest['update'],
                     cursor=cursor,identity=identity,optimizer='inherited unchanged',
                     bundle_metadata='unchanged parent provenance; new identity is in manifest/training.pt',
                     note='New experiment; removed data may have influenced inherited weights and moments.')
        atomic_json(temporary/'migration.json',lineage)
        files['migration.json']=sha256(temporary/'migration.json')
        atomic_json(temporary/'manifest.json',dict(identity=identity,cursor=cursor,update=manifest['update'],files=files))
        if any((parent/n).stat().st_size!=st[0] or (parent/n).stat().st_mtime_ns!=st[1] for n,st in stamps.items()):
            raise ValueError('parent checkpoint changed during migration')
        for path in temporary.iterdir():
            with path.open('rb') as stream:os.fsync(stream.fileno())
        temporary.rename(destination)
        fd=os.open(root,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
        atomic_json(output/'latest.json',dict(directory=str(destination)))
    except BaseException:
        shutil.rmtree(temporary,ignore_errors=True);raise
    return destination
