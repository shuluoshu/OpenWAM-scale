#!/usr/bin/env python3
"""Validate shared offline artifacts before allocating or starting GPU jobs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
import psutil


def read(path):
    try:return json.loads(Path(path).read_text())
    except (FileNotFoundError,json.JSONDecodeError):return {}


def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.pending');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)


def gate_status(state,marker,now):
    if state.get('status')=='failed':raise RuntimeError('CPU data preparation failed; inspect prepare.log')
    if state.get('status')=='ready' and Path(marker).is_file():return True
    if state.get('updated_at') and now-state['updated_at']>900:
        raise RuntimeError('CPU preparation heartbeat stale >15 minutes; refusing to wait indefinitely')
    return False


def require_prepared(marker,*,source_yaml=None,sample_fps=None,index_path=None):
    """Fail immediately: a running CPU preparer is not a ready GPU workload."""
    def shared(path):
        path=Path(path).resolve()
        if not path.is_relative_to('/mnt/data/cpfs'):
            raise ValueError(f'prepared data must reside on shared CPFS: {path}')
        if not path.is_file():raise FileNotFoundError(path)
        return path
    marker=shared(marker)
    data=json.loads(marker.read_text())
    source,index,registry=(shared(data[key]) for key in ('source_yaml','index','stats_registry'))
    if source_yaml is not None and source!=Path(source_yaml).resolve():raise ValueError('offline source path differs from pipeline')
    if index_path is not None and index!=Path(index_path).resolve():raise ValueError('offline index path differs from pipeline')
    with sqlite3.connect(f'{index.as_uri()}?mode=ro',uri=True) as connection:
        metadata={k:json.loads(v) for k,v in connection.execute('SELECT key,value FROM metadata')}
    if metadata.get('complete') is not True or metadata.get('total_samples',0)<=0:
        raise ValueError('offline population index incomplete or empty')
    if sample_fps is not None and metadata.get('sample_fps')!=float(sample_fps):
        raise ValueError('offline index sample_fps differs from pipeline')
    if hashlib.sha256(source.read_bytes()).hexdigest()!=metadata.get('source_yaml_sha256'):
        raise ValueError('offline source YAML differs from prepared population')
    registry_bytes=registry.read_bytes()
    if registry.name!='registry-'+hashlib.sha256(registry_bytes).hexdigest()+'.json':
        raise ValueError('offline statistics registry checksum mismatch')
    if json.loads(registry_bytes).get('contract',{}).get('selection_sha256')!=metadata['source_yaml_sha256']:
        raise ValueError('offline statistics registry selection differs from population')
    return data


def require_pipeline_prepared(marker,config_path):
    from openwam.train.stage12 import load_pipeline
    from openwam.train.goal_curated import load_config
    from openwam.dataloader.goal_epoch import epoch_sample_fps
    config=load_pipeline(config_path)
    output=Path(config['output_dir'])
    if Path(marker).resolve()!=(output/'prepared.json').resolve():
        raise ValueError('prepared marker differs from pipeline output')
    stage1=load_config(config['stage1_config'],source_yaml=config['source_yaml'])
    data=require_prepared(marker,source_yaml=config['source_yaml'],
        sample_fps=epoch_sample_fps(stage1),index_path=output/'population.sqlite')
    if config.get('stats_registry') and Path(config['stats_registry']).resolve()!=Path(data['stats_registry']).resolve():
        raise ValueError('pipeline registry override differs from prepared manifest')
    return data


def main():
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--attach-pid',type=int);mode.add_argument('--wait',action='store_true')
    mode.add_argument('--check',action='store_true',help='fail immediately unless offline CPFS artifacts are ready')
    p.add_argument('--status',type=Path);p.add_argument('--marker',type=Path,required=True)
    p.add_argument('--config',type=Path,help='bind prepared source/index/rate to this Stage12 pipeline')
    p.add_argument('--timeout',type=int,default=86400)
    args=p.parse_args();started=time.time()
    if args.check:
        if args.config:
            import sys
            sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
            require_pipeline_prepared(args.marker,args.config)
        else:require_prepared(args.marker)
        print('Offline CPFS preparation verified',flush=True)
        return 0
    if args.status is None:p.error('--status is required for process tracking')
    if args.attach_pid:
        process=psutil.Process(args.attach_pid)
        cmd=process.cmdline()
        if not any(x.endswith('train_stage12.py') for x in cmd) or '--prepare' not in cmd:
            raise RuntimeError('PID is not the intended Stage12 CPU preparation')
        while process.is_running() and process.status()!=psutil.STATUS_ZOMBIE:
            write(args.status,dict(status='running',pid=args.attach_pid,updated_at=time.time(),started_at=started))
            time.sleep(5)
        success=args.marker.is_file()
        write(args.status,dict(status='ready' if success else 'failed',pid=args.attach_pid,updated_at=time.time(),started_at=started))
        return 0 if success else 1
    while not gate_status(read(args.status),args.marker,time.time()):
        if time.time()-started>args.timeout:raise TimeoutError('CPU preparation deadline exceeded')
        print('Waiting for CPU preparation on shared storage',flush=True);time.sleep(30)
    print('CPU preparation ready',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
