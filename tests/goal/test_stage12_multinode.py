"""Multi-node launch and coordinator failures must not duplicate the pipeline."""
import json
import os
from pathlib import Path
import socket
import psutil
import sys
import pytest
import torch.multiprocessing as mp


def test_topology_and_dlc_node_environment(monkeypatch):
    from openwam.train.stage12_distributed import torchrun_command
    cfg=dict(world_size=32,nnodes=4,gpus_per_node=8)
    monkeypatch.setenv('MASTER_ADDR','node0')
    monkeypatch.setenv('MASTER_PORT','29500')
    monkeypatch.setenv('RANK','2');monkeypatch.setenv('WORLD_SIZE','4')
    cmd=torchrun_command(sys.executable,cfg,visible_devices=8)
    assert '--standalone' not in cmd
    assert '--nnodes=4' in cmd and '--nproc_per_node=8' in cmd and '--node_rank=2' in cmd
    assert '--master_port=29501' in cmd # parent coordinator owns 29500
    with pytest.raises(ValueError):torchrun_command(sys.executable,dict(cfg,world_size=16))
    with pytest.raises(ValueError):torchrun_command(sys.executable,cfg,visible_devices=4)
    monkeypatch.setenv('WORLD_SIZE','32')
    with pytest.raises(ValueError):torchrun_command(sys.executable,cfg)


def _coordinator_worker(rank,port,directory,fail):
    os.environ.update(RANK=str(rank),WORLD_SIZE='2',MASTER_ADDR='127.0.0.1',MASTER_PORT=str(port))
    from openwam.train.stage12_distributed import coordinate_pipeline
    root=Path(directory)
    def leader(run):
        with (root/'leader').open('a') as f:f.write('once\n')
        command=[sys.executable,'-c',
            "import os,sys,time,subprocess; from pathlib import Path; "
            "r=os.environ['OPENWAM_NODE_RANK']; "
            f"Path({str(root)!r}, 'worker'+r).write_text(os.environ.get('WORLD_SIZE','unset')); "
            +(f"child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)'],start_new_session=True); "
              f"Path({str(root)!r},'orphan'+r).write_text(str(child.pid)); "
              "sys.exit(7) if r=='1' else time.sleep(30)" if fail else "sys.exit(0)")]
        # force distributed here to test coordination without loading any GPU/model
        run(command,cwd=root,env=os.environ.copy(),all_nodes=True)
        (root/'promoted').write_text('yes')
        return 0
    try:
        coordinate_pipeline(leader,dict(nnodes=2,gpus_per_node=1,world_size=2),timeout_seconds=30)
    except RuntimeError:
        (root/f'failed{rank}').write_text('yes')
        if not fail:raise


@pytest.mark.parametrize('fail',[False,True])
def test_coordinator_leader_once_and_propagates_peer_failure(tmp_path,fail):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    mp.spawn(_coordinator_worker,args=(port,str(tmp_path),fail),nprocs=2,join=True)
    assert (tmp_path/'leader').read_text()=='once\n'
    assert all((tmp_path/f'worker{i}').read_text()=='unset' for i in range(2))
    assert (tmp_path/'promoted').exists()!=fail
    if fail:
        assert all((tmp_path/f'failed{i}').exists() for i in range(2))
        for path in tmp_path.glob('orphan*'):
            assert not psutil.pid_exists(int(path.read_text())), 'orphan worker survived cleanup'


def _torchrun_coordinator_worker(rank,port,directory):
    os.environ.update(RANK=str(rank),WORLD_SIZE='2',MASTER_ADDR='127.0.0.1',MASTER_PORT=str(port))
    from openwam.train.stage12_distributed import coordinate_pipeline,torchrun_command
    root=Path(directory);cfg=dict(nnodes=2,gpus_per_node=1,world_size=2)
    def leader(run):
        for stage in ('stage1','stage2'):
            run([*torchrun_command(sys.executable,cfg),str(root/'worker.py'),str(root),stage],cwd=root)
        # Non-distributed preflight must run once, without DLC rank variables.
        run([sys.executable,'-c',"import os; assert 'WORLD_SIZE' not in os.environ; "
             f"open({str(root/'preflight')!r},'w').write('once')"],cwd=root)
        return 0
    coordinate_pipeline(leader,cfg,timeout_seconds=60)


def test_actual_two_node_torchrun_two_stages(tmp_path):
    (tmp_path/'worker.py').write_text('''
import sys, os
from pathlib import Path
import torch
import torch.distributed as dist
from datetime import timedelta
dist.init_process_group('gloo',timeout=timedelta(seconds=30))
x=torch.tensor(dist.get_rank()+1);dist.all_reduce(x)
assert x.item()==3
Path(sys.argv[1],sys.argv[2]+'_'+str(dist.get_rank())).write_text(os.environ['WORLD_SIZE'])
dist.destroy_process_group()
''')
    # Parent control and training use adjacent ports. Reserve/test both before release.
    with socket.socket() as a:
        a.bind(('127.0.0.1',0));port=a.getsockname()[1]
        with socket.socket() as b:b.bind(('127.0.0.1',port+1))
    mp.spawn(_torchrun_coordinator_worker,args=(port,str(tmp_path)),nprocs=2,join=True)
    for stage in ('stage1','stage2'):
        assert all((tmp_path/f'{stage}_{rank}').read_text()=='2' for rank in range(2))
    assert (tmp_path/'preflight').read_text()=='once'
