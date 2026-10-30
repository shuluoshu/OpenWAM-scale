"""Node-level orchestration for Stage12; torchrun owns GPU-level process groups."""
from __future__ import annotations
from datetime import timedelta
import os
import ctypes
from pathlib import Path
import subprocess
import time
import torch
import torch.distributed as dist


def topology(config):
    world=config.get('world_size',8)
    nodes=config.get('nnodes',1)
    local=config.get('gpus_per_node',world)
    if any(isinstance(x,bool) or not isinstance(x,int) or x<1 for x in (world,nodes,local)):
        raise ValueError('world_size, nnodes and gpus_per_node must be positive integers')
    if world!=nodes*local:raise ValueError('world_size must equal nnodes * gpus_per_node')
    return nodes,local


def node_environment(config,*,required=True):
    nodes,_=topology(config)
    if required and any(not os.environ.get(k) for k in ('RANK','WORLD_SIZE','MASTER_ADDR','MASTER_PORT')):
        raise ValueError('multi-node training requires DLC RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT')
    rank=int(os.environ.get('RANK','0'))
    count=int(os.environ.get('WORLD_SIZE',nodes))
    port=int(os.environ.get('MASTER_PORT','29500'))
    if count!=nodes or not 0<=rank<nodes:
        raise ValueError('DLC WORLD_SIZE/RANK must describe nodes, not GPU ranks')
    if not 1<=port<65535:raise ValueError('MASTER_PORT must leave room for training port + 1')
    return rank,os.environ.get('MASTER_ADDR','127.0.0.1'),port


def torchrun_command(python,config,*,visible_devices=None,require_env=True):
    nodes,local=topology(config)
    if visible_devices is not None and visible_devices<local:
        raise ValueError(f'requires {local} visible CUDA devices per node; found {visible_devices}')
    if nodes==1:
        from openwam.train.stage12 import single_node_torchrun
        return single_node_torchrun(python,local,visible_devices=visible_devices)
    rank,host,port=node_environment(config,required=require_env)
    return [python,'-m','torch.distributed.run',f'--nnodes={nodes}',f'--nproc_per_node={local}',
            f'--node_rank={rank}',f'--master_addr={host}',f'--master_port={port+1}','--max_restarts=0']


def child_environment(root):
    env=os.environ.copy()
    # DLC's rank is a POD rank. Only torchrun may assign training GPU ranks.
    for key in ('RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE','GROUP_RANK','ROLE_RANK',
                'ROLE_WORLD_SIZE','MASTER_ADDR','MASTER_PORT','TORCHELASTIC_RUN_ID'):
        env.pop(key,None)
    env['PYTHONPATH']=str(root)+os.pathsep+env.get('PYTHONPATH','')
    env['OPENWAM_NODE_RANK']=str(dist.get_rank())
    return env


def _enable_subreaper():
    # torchrun creates child workers in separate sessions. Adopt orphan workers
    # if the launcher is OOM-killed so cleanup can still find their GPU processes.
    libc=ctypes.CDLL(None,use_errno=True)
    if libc.prctl(36,1,0,0,0)!=0:  # PR_SET_CHILD_SUBREAPER (Linux DLC containers)
        raise OSError(ctypes.get_errno(),'cannot enable child subreaper')


def _stop(process):
    import psutil
    # The coordinator has only this stage's subprocess tree. Subreaper adoption
    # keeps workers discoverable even when the torchrun launcher has exited.
    children=psutil.Process().children(recursive=True)
    for child in children:
        try:child.terminate()
        except psutil.NoSuchProcess:pass
    _,alive=psutil.wait_procs(children,timeout=10)
    for child in alive:
        try:child.kill()
        except psutil.NoSuchProcess:pass
    psutil.wait_procs(alive,timeout=5)
    if process is not None:
        try:process.wait(timeout=1)
        except subprocess.TimeoutExpired:process.kill();process.wait()


def _run_on_nodes(command,cwd):
    rank=dist.get_rank()
    command=[f'--node_rank={rank}' if arg.startswith('--node_rank=') else arg for arg in command]
    process=None;launch_error=None;completed=False
    try:
        try:
            _enable_subreaper()
            process=subprocess.Popen(command,cwd=cwd,env=child_environment(cwd),start_new_session=True)
        except Exception as exc:launch_error=f'{type(exc).__name__}: {exc}'
        while True:
            rc=process.poll() if process is not None else -1
            state=torch.tensor([int(rc is None),int(launch_error is not None or rc not in (None,0))],dtype=torch.int32)
            dist.all_reduce(state,op=dist.ReduceOp.MAX)
            if state[1]:raise RuntimeError(f'distributed stage failed; node {rank} exit={rc}, launch_error={launch_error}')
            if not state[0]:
                completed=True;return
            time.sleep(.2)
    finally:
        if not completed:_stop(process)


def coordinate_pipeline(leader,config,*,timeout_seconds=86400,startup_validation=None):
    """Run leader once; broadcast only GPU commands, not metadata operations or secrets.

    A peer process failure is reduced while subprocesses run, so other local
    process groups are stopped promptly rather than waiting for NCCL timeouts.
    """
    nodes,_=topology(config);rank,host,port=node_environment(config)
    if dist.is_initialized():raise RuntimeError('Stage12 parent must not already be distributed')
    # Missing/failed Pods must not spend the day in initial rendezvous. Once
    # all nodes join, leader-only metadata validation may legitimately be long.
    store=dist.TCPStore(host,port,nodes,rank==0,timeout=timedelta(seconds=min(timeout_seconds,300)))
    dist.init_process_group('gloo',store=store,rank=rank,world_size=nodes,
                            timeout=timedelta(seconds=timeout_seconds))
    try:
        startup_error=None
        try:
            if startup_validation:startup_validation()
        except Exception as exc:startup_error=f'node {rank}: {type(exc).__name__}: {exc}'
        errors=[None]*nodes;dist.all_gather_object(errors,startup_error)
        if any(errors):raise RuntimeError('node startup validation failed: '+str(errors))
        if rank:
            while True:
                message=[None];dist.broadcast_object_list(message,src=0)
                item=message[0]
                if item['kind']=='stop':
                    if item['error']:raise RuntimeError('Stage12 leader failed: '+item['error'])
                    return item.get('result',0)
                try:_run_on_nodes(item['command'],item['cwd'])
                except RuntimeError:
                    # Leader receives the same failed reduction and publishes the
                    # final error after recording pipeline_state.json.
                    continue
        def run(command,*,cwd,env=None,all_nodes=None):
            distributed=('torch.distributed.run' in command) if all_nodes is None else all_nodes
            if not distributed:
                return subprocess.run(command,cwd=cwd,env=child_environment(cwd),check=True)
            dist.broadcast_object_list([dict(kind='run',command=command,cwd=str(cwd))],src=0)
            return _run_on_nodes(command,str(cwd))
        error=None;result=None
        try:
            result=leader(run)
            return result
        except BaseException as exc:
            error=f'{type(exc).__name__}: {exc}';raise
        finally:
            dist.broadcast_object_list([dict(kind='stop',error=error,result=result)],src=0)
    finally:
        dist.destroy_process_group()
