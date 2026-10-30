"""Real processes verify local bounds and cross-node bootstrap overlap."""
from datetime import timedelta
import json
from pathlib import Path
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def test_32_rank_schedule_has_four_parallel_waves():
    from openwam.train.bootstrap import bootstrap_plan
    waves=bootstrap_plan(32,8,2)
    assert len(waves)==4 and all(len(w)==8 for w in waves)
    assert sorted(rank for wave in waves for rank in wave)==list(range(32))
    for wave in waves:
        assert all(sum(rank//8==node for rank in wave)==2 for node in range(4))


def worker(rank, root, mode, concurrency):
    from openwam.train.bootstrap import run_bootstrap_in_waves
    root=Path(root);torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=(root/'rendezvous').as_uri(),rank=rank,world_size=4,timeout=timedelta(seconds=25))
    events=[]
    def boot():
        start=time.monotonic();(root/f'start{rank}').write_text(str(start))
        # Both nodes must enter their same local-rank wave concurrently.
        peer=(rank+2)%4
        deadline=time.monotonic()+5
        while not (root/f'start{peer}').exists():
            if time.monotonic()>deadline:raise RuntimeError('cross-node loading was serialized')
            time.sleep(.01)
        time.sleep(.05)
        if mode=='failure' and rank==2:raise ValueError('injected loader error')
        (root/f'end{rank}').write_text(str(time.monotonic()))
        return rank+100
    try:
        result=run_bootstrap_in_waves(boot,local_rank=rank%2,local_world_size=2,
            workers_per_node=(2 if mode=='mismatch' and rank==1 else concurrency),event=events.append)
        payload=dict(result=result)
    except Exception as exc:payload=dict(error=str(exc))
    finally:
        (root/f'result{rank}.json').write_text(json.dumps({**payload,'events':events}))
        dist.destroy_process_group()


@pytest.mark.parametrize('mode,concurrency',[('ok',1),('ok',2),('failure',1),('mismatch',1)])
def test_distributed_bootstrap_concurrency_and_all_rank_failures(tmp_path,mode,concurrency):
    # Keep missing implementation failures in the parent, before spawning.
    from openwam.train.bootstrap import run_bootstrap_in_waves
    context=mp.spawn(worker,args=(str(tmp_path),mode,concurrency),nprocs=4,join=False)
    deadline=time.monotonic()+70
    try:
        while not context.join(timeout=1):
            assert time.monotonic()<deadline,'bootstrap collectives hung'
    finally:
        for process in context.processes:
            if process.is_alive():process.terminate();process.join(timeout=5)
    results=[json.loads((tmp_path/f'result{r}.json').read_text()) for r in range(4)]
    if mode=='ok':
        assert [r['result'] for r in results]==[100,101,102,103]
        if concurrency==1:
            for node in (0,2):
                assert float((tmp_path/f'end{node}').read_text()) <= float((tmp_path/f'start{node+1}').read_text())
    else:
        assert all('error' in r for r in results)
        assert len({r['error'] for r in results})==1
        if mode=='failure':
            assert 'injected loader error' in results[0]['error']
            assert not (tmp_path/'start1').exists() and not (tmp_path/'start3').exists()
        else:
            assert 'configuration' in results[0]['error']
            assert not list(tmp_path.glob('start*'))


def test_invalid_schedule_and_single_process_result():
    from openwam.train.bootstrap import bootstrap_plan,run_bootstrap_in_waves
    for args in [(32,8,0),(32,8,True),(31,8,2),(32,0,2),(32,8,9)]:
        with pytest.raises(ValueError):bootstrap_plan(*args)
    assert run_bootstrap_in_waves(lambda:123,local_rank=0,local_world_size=1,workers_per_node=1)==123


def test_query_bootstrap_defers_codec_check_until_device_placement(monkeypatch):
    from openwam.train.bootstrap import bootstrap_query_components
    from openwam.train import goal_trainer
    from openwam.model.goal import dependencies
    from test_patch_expert import make_expert
    from test_query_goal import QUERY
    original=make_expert();vae=original.codec.vae;called=[]
    encoder=torch.nn.Linear(3,3);tokenizer=object()
    def load(config,**kwargs):
        called.append(kwargs)
        assert kwargs==dict(device='cpu',dtype=torch.float32,validate_codec=False)
        return original,encoder,tokenizer
    monkeypatch.setattr(goal_trainer,'bootstrap_components',load)
    monkeypatch.setattr(dependencies,'dependency_identity',lambda *a,**kw:{'verified':True})
    expert,text,token,deps=bootstrap_query_components(dict(seed=42,model={'query_goal':QUERY},
        observation_mode='multiview',text_device='cpu'),device=torch.device('cpu'),dtype=torch.float32)
    assert len(called)==1 and vae.calls==[(1,3,1,256,320)]
    assert expert.codec.vae is vae and text is encoder and token is tokenizer
    assert deps=={'verified':True} and not encoder.training
    assert not any(p.requires_grad for p in encoder.parameters())


def test_launcher_uses_parallel_coordinator_and_passes_resume_limit(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from scripts import train_goal_epoch as launcher
    from openwam.train import bootstrap,goal_epoch,goal_curated,stage12
    from openwam.dataloader import curated_goal
    model=torch.nn.Linear(1,1);encoder=torch.nn.Linear(1,1);events=[]
    monkeypatch.setenv('WORLD_SIZE','1');monkeypatch.setenv('RANK','0');monkeypatch.setenv('LOCAL_RANK','0');monkeypatch.setenv('LOCAL_WORLD_SIZE','1')
    monkeypatch.setattr(torch.cuda,'set_device',lambda device:events.append('device'))
    config=dict(model={'goal_representation':'query_latent'},dataset={},sample_fps=3.,
                bootstrap_workers_per_node=2,learning_rate=1e-5)
    monkeypatch.setattr(goal_curated,'load_config',lambda *a,**kw:config)
    monkeypatch.setattr(curated_goal,'CuratedGoalDataset',lambda **kw:object())
    finite=SimpleNamespace(summary={})
    monkeypatch.setattr(launcher,'validated_population',lambda *a,**kw:finite)
    monkeypatch.setattr(stage12,'epoch_identity',lambda *a:{'test':True})
    def build(*args,**kwargs):
        events.append('build');return model,encoder,object(),{}
    monkeypatch.setattr(bootstrap,'bootstrap_query_components',build)
    coordinator=bootstrap.run_bootstrap_in_waves
    def run(callback,**kwargs):
        events.append('coordinator');assert kwargs['workers_per_node']==1
        return coordinator(callback,**kwargs)
    monkeypatch.setattr(bootstrap,'run_bootstrap_in_waves',run)
    def train(*args,**kwargs):
        events.append('train');assert kwargs['bootstrap_workers_per_node']==2
        return {'ok':True}
    monkeypatch.setattr(goal_epoch,'train_epoch',train)
    assert launcher.main(['--train','--config','unused','--index','unused','--output-dir',str(tmp_path)])==0
    assert events==['device','coordinator','build','train']
