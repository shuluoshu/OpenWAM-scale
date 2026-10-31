"""Only rank zero scans global source metadata; every rank receives failures."""
from datetime import timedelta
import json
from pathlib import Path
import sqlite3
import time

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


class _Source:
    def __init__(self, root, *, empty=False):
        root=Path(root)
        self.root=root;self.selection={'source_yaml':str(root/'source.yaml')}
        self.repos=[dict(repo='fixture',path=str(root),family='fixture',ranges=[(0,0,12)])]
        self.seed=13;self._evaluating=False;self.empty=empty

    def _episode(self, repo, episode):
        return dict(length=12,info={'fps':30},data=self.root/'dependency',
            tasks=[dict(start=0,end_exclusive=12,text='move')],subtasks=[],
            source_valid=np.full(12,not self.empty,bool))

    def _materialize(self, repo, episode, metadata, segment, anchor, goal, level):
        return dict(anchor=anchor,goal=goal),{}


def _worker(rank, root, mode):
    from scripts import train_goal_epoch
    from openwam.dataloader import goal_epoch
    root=Path(root);torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=(root/'rendezvous').as_uri(),
        rank=rank,world_size=2,timeout=timedelta(seconds=25))
    calls=0;validate=goal_epoch.validate_index
    def counted(*args,**kwargs):
        nonlocal calls
        calls+=1
        return validate(*args,**kwargs)
    goal_epoch.validate_index=counted
    result={}
    try:
        index=root/('other.db' if mode in ('follower_identity','prepared_follower_identity') and rank==1 else 'index.db')
        extra={}
        if mode.startswith('prepared_'):
            extra=dict(prepared_runtime=(root/'proof.txt').read_text(),config=dict(
                dataset=dict(source_yaml=str(root/'source.yaml')),sample_fps=3.))
        finite=train_goal_epoch.validated_population(_Source(root),index,sample_fps=3.,device=torch.device('cpu'),**extra)
        result=dict(status='ok',length=len(finite),summary=finite.summary)
        if mode in ('source_changed','prepared_source_changed'):
            if rank==0:(root/'dependency').write_text('changed after global validation')
            dist.barrier()
        result['sample']=finite.sample(0)[0]
    except Exception as exc:
        result=dict(status='error',error=f'{type(exc).__name__}: {exc}')
    finally:
        result['global_validation_calls']=calls
        (root/f'rank_{rank}.json').write_text(json.dumps(result))
        dist.destroy_process_group()


@pytest.mark.parametrize('mode,expected',[
    ('ok',None),('incomplete','incomplete'),('empty','empty population'),
    ('follower_identity','index'),('source_changed','source dependency changed'),
    ('prepared_ok',None),('prepared_follower_identity','index'),('prepared_source_changed','source dependency changed'),
])
def test_two_ranks_validate_once_and_share_errors(tmp_path,mode,expected):
    from openwam.dataloader.goal_epoch import build_epoch_index
    (tmp_path/'source.yaml').write_text('fixture')
    (tmp_path/'dependency').write_text('initial')
    population=build_epoch_index(_Source(tmp_path,empty=mode=='empty'),tmp_path/'index.db',sample_fps=3.)
    if mode.startswith('prepared_'):
        from openwam.train.prepared_runtime import write_prepared_runtime
        registry=tmp_path/'registry.json';registry.write_text('{}')
        audit=tmp_path/'audit.json';audit.write_text('{"errors": []}')
        baseline=tmp_path/'baseline';baseline.mkdir();(baseline/'config.yaml').write_text('model: {}')
        proof=write_prepared_runtime(tmp_path,stage1=dict(dataset=dict(source_yaml=str(tmp_path/'source.yaml')),sample_fps=3.),
            stage2_dataloader=dict(source_yaml=str(tmp_path/'source.yaml'),normalization_registry=str(registry)),
            pipeline=dict(stage2_baseline=str(baseline)),population=population,audit_path=audit,dataset_contract={})
        (tmp_path/'proof.txt').write_text(proof)
    if mode=='incomplete':
        with sqlite3.connect(tmp_path/'index.db') as connection:
            connection.execute("UPDATE metadata SET value='false' WHERE key='complete'")
    if mode in ('follower_identity','prepared_follower_identity'):
        (tmp_path/'other.db').write_bytes((tmp_path/'index.db').read_bytes())
    context=mp.spawn(_worker,args=(str(tmp_path),mode),nprocs=2,join=False)
    deadline=time.monotonic()+60
    try:
        while not context.join(timeout=1):
            assert time.monotonic()<deadline,'distributed validation hung'
    finally:
        for process in context.processes:
            if process.is_alive():process.terminate();process.join(timeout=5)
    results=[json.loads((tmp_path/f'rank_{rank}.json').read_text()) for rank in range(2)]
    assert [r['global_validation_calls'] for r in results]==([0,0] if mode.startswith('prepared_') else [1,0])
    if expected is None:
        assert all(r['status']=='ok' and r['length']==2 for r in results)
        assert results[0]['summary']==results[1]['summary']
        assert results[0]['sample']==results[1]['sample']==dict(anchor=0,goal=11)
    else:
        assert all(r['status']=='error' and expected in r['error'] for r in results)
        if mode not in ('source_changed','prepared_source_changed'):assert results[0]['error']==results[1]['error']


def test_single_process_validation_keeps_original_errors(tmp_path):
    from scripts import train_goal_epoch
    from openwam.dataloader.goal_epoch import build_epoch_index
    (tmp_path/'source.yaml').write_text('fixture');(tmp_path/'dependency').write_text('initial')
    source=_Source(tmp_path);path=tmp_path/'index.db'
    build_epoch_index(source,path,sample_fps=3.)
    finite=train_goal_epoch.validated_population(source,path,sample_fps=3.)
    assert len(finite)==2
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE metadata SET value='false' WHERE key='complete'")
    with pytest.raises(ValueError,match='incomplete'):
        train_goal_epoch.validated_population(source,path,sample_fps=3.)


def test_launcher_initializes_cuda_and_group_before_index_validation(tmp_path,monkeypatch):
    from scripts import train_goal_epoch
    import openwam.train.goal_curated as curated
    import openwam.dataloader.curated_goal as source
    import openwam.dataloader.goal_epoch as epoch
    monkeypatch.setenv('WORLD_SIZE','2');monkeypatch.setenv('RANK','0');monkeypatch.setenv('LOCAL_RANK','0')
    monkeypatch.setattr(curated,'load_config',lambda *a,**kw:dict(
        model={'goal_representation':'query_latent'},dataset={},sample_fps=3.))
    monkeypatch.setattr(source,'CuratedGoalDataset',lambda **kw:object())
    events=[]
    monkeypatch.setattr(torch.cuda,'set_device',lambda device:events.append('cuda'))
    monkeypatch.setattr(dist,'init_process_group',lambda *a,**kw:events.append('nccl'))
    monkeypatch.setattr(dist,'is_initialized',lambda:'nccl' in events)
    monkeypatch.setattr(dist,'get_world_size',lambda:2)
    monkeypatch.setattr(dist,'get_rank',lambda:0)
    monkeypatch.setattr(dist,'broadcast_object_list',lambda *a,**kw:None)
    monkeypatch.setattr(dist,'destroy_process_group',lambda:events.append('destroy'))
    def finite(*args,**kwargs):
        assert events==['cuda','nccl'],'index validated before CUDA/NCCL initialization'
        raise ValueError('planned validation failure')
    monkeypatch.setattr(epoch,'FiniteGoalDataset',finite)
    with pytest.raises(RuntimeError,match='planned validation failure'):
        train_goal_epoch.main(['--train','--config','unused','--index',str(tmp_path/'index.db'),
            '--output-dir',str(tmp_path/'out')])
    assert events==['cuda','nccl','destroy']


def test_single_process_launcher_rejects_bad_index_before_cuda(tmp_path,monkeypatch):
    from scripts import train_goal_epoch
    import openwam.train.goal_curated as curated
    import openwam.dataloader.curated_goal as source
    import openwam.dataloader.goal_epoch as epoch
    monkeypatch.setenv('WORLD_SIZE','1');monkeypatch.setenv('RANK','0');monkeypatch.setenv('LOCAL_RANK','0')
    monkeypatch.setattr(curated,'load_config',lambda *a,**kw:dict(
        model={'goal_representation':'query_latent'},dataset={},sample_fps=3.))
    monkeypatch.setattr(source,'CuratedGoalDataset',lambda **kw:object())
    events=[]
    monkeypatch.setattr(torch.cuda,'set_device',lambda device:events.append('cuda'))
    def finite(*args,**kwargs):raise ValueError('planned validation failure')
    monkeypatch.setattr(epoch,'FiniteGoalDataset',finite)
    with pytest.raises(ValueError,match='planned validation failure'):
        train_goal_epoch.main(['--train','--config','unused','--index',str(tmp_path/'index.db'),
            '--output-dir',str(tmp_path/'out')])
    assert events==[]
