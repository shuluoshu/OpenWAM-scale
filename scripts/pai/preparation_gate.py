#!/usr/bin/env python3
"""Track one CPU preparation process and gate the queued GPU job on its result."""
import argparse
import json
import os
from pathlib import Path
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


def main():
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--attach-pid',type=int);mode.add_argument('--wait',action='store_true')
    p.add_argument('--status',type=Path,required=True);p.add_argument('--marker',type=Path,required=True)
    p.add_argument('--timeout',type=int,default=86400)
    args=p.parse_args();started=time.time()
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
