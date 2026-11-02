"""Benchmark accounting and real CPU forward/backward smoke coverage."""
import copy
import json
import os

import pytest
import torch


def test_matrix_allows_only_performance_controls():
    from openwam.train.goal_benchmark import parse_cases
    base = dict(batch_size=32, text_cache_size=128, learning_rate=1e-5)
    saved = copy.deepcopy(base)
    cases = parse_cases(None, base)
    assert [(x['batch_size'], x['num_workers'], x['prefetch_factor']) for x in cases] == [(32,8,4),(64,8,8),(128,8,16)]
    assert base == saved
    assert parse_cases([dict(name='debug', batch_size=1, num_workers=0)], base)[0]['num_workers'] == 0
    for bad in [dict(name='bad', learning_rate=1.), dict(name='../escape'), dict(name='bad',batch_size=True),dict(name='bad',measured_steps=0),dict(name='bad',decode_threads=9)]:
        with pytest.raises(ValueError): parse_cases([bad], base)
    with pytest.raises(ValueError): parse_cases([dict(name='same'),dict(name='same')],base)


def test_summary_excludes_warmup_and_uses_slowest_real_rank_wall():
    from openwam.train.goal_benchmark import summarize_case
    ranks = []
    for rank, elapsed in enumerate([4., 8.]):
        rows = [dict(step=i, measured=i>1, real_samples=2, load_s=float(rank+1),prepare_s=2.,all_ready_wait_s=0.5,forward_backward_s=3.,optimizer_sync_s=1.,total_s=7.,loss=1.,gradient_norm=2.,valid_tcp_points=3, samples_with_tcp=1) for i in range(1,4)]
        rows[0]['load_s'] = 999.
        ranks.append(dict(rank=rank, rows=rows, measured_wall_s=elapsed,peak_allocated_bytes=100+rank,peak_reserved_bytes=200+rank))
    result = summarize_case(dict(name='x'),ranks)
    assert result['global_samples_per_second'] == 1.
    assert result['measured_global_samples'] == 8
    assert result['timing_s']['load_s'] == dict(max=2.,p50=1.5)
    assert result['valid_tcp_points'] == 12
    assert result['benchmark_only'] is True
    assert result['per_rank'][1]['peak_reserved_bytes'] == 201


def test_real_cpu_training_cases_reset_weights_rng_and_skip_checkpoints(tmp_path,monkeypatch):
    from openwam.train.goal_benchmark import run_benchmark, parse_cases
    from test_query_goal import make_query
    from test_multiview_patch import multiview_batch
    torch.set_num_threads(2)
    expert = make_query()
    observed, context, targets, valid = multiview_batch(1)
    monkeypatch.setenv('OPENWAM_VIDEO_DECODE_THREADS','2')
    decoder_threads=[]
    class Data:
        def __len__(self): return 12
        def sample(self,index):
            decoder_threads.append(os.environ['OPENWAM_VIDEO_DECODE_THREADS'])
            return dict(family='tiny',instruction='move',id=str(index)),dict(observed=observed,context=context,targets=targets,observed_view_valid=valid)
    config = dict(seed=42, learning_rate=1e-5, optimizer_sharding='none')
    cases = parse_cases([dict(name=name,batch_size=1,num_workers=0,warmup_steps=1,measured_steps=2,decode_threads=threads) for name,threads in [('first',1),('second',4)]],config)
    result = run_benchmark(expert,Data(),None,config,cases,tmp_path,timeout_seconds=30)
    assert len(result)==2
    assert decoder_threads==['1']*3+['4']*3
    assert os.environ['OPENWAM_VIDEO_DECODE_THREADS']=='2'
    rows = [json.loads(line) for line in (tmp_path/'rank_0.jsonl').read_text().splitlines()]
    updates = [r for r in rows if r['event']=='update']
    first = [r for r in updates if r['case']=='first']
    second = [r for r in updates if r['case']=='second']
    assert [r['loss'] for r in first] == [r['loss'] for r in second]
    assert [r['anchor_index'] for r in first] == [r['anchor_index'] for r in second]
    assert result[0]['measured_global_samples']==2
    assert all(r['valid_tcp_points']>0 for r in updates)
    assert not list(tmp_path.glob('*.pt'))
    assert not (tmp_path/'training_complete.json').exists()
    assert not (tmp_path/'epoch_complete.json').exists()


def test_case_exceeding_population_fails_before_training(tmp_path):
    from openwam.train.goal_benchmark import run_benchmark,parse_cases
    from test_query_goal import make_query
    class Data:
        def __len__(self): return 1
        def sample(self,index): raise AssertionError('must reject before reading')
    cases=parse_cases([dict(name='too_long',batch_size=1,num_workers=0,warmup_steps=0,measured_steps=2)],{})
    with pytest.raises(ValueError,match='exceeds one finite population'):
        run_benchmark(make_query(),Data(),None,dict(learning_rate=1e-5),cases,tmp_path)


def test_nonfinite_failure_records_restart_and_does_not_run_next_case(tmp_path):
    from openwam.train.goal_benchmark import run_benchmark,parse_cases
    from test_query_goal import make_query
    from test_multiview_patch import multiview_batch
    expert=make_query()
    observed,context,targets,valid=multiview_batch(1)
    def poison(module,inputs,out):
        return dict(out,loss=out['loss']*float('nan'))
    expert.register_forward_hook(poison)
    class Data:
        def __len__(self): return 2
        def sample(self,index):
            return dict(family='tiny',instruction='move'),dict(observed=observed,context=context,targets=targets,observed_view_valid=valid)
    cases=parse_cases([dict(name=name,batch_size=1,num_workers=0,warmup_steps=0,measured_steps=1) for name in ('bad','next')],{})
    with pytest.raises(FloatingPointError,match='nonfinite Stage1 loss'):
        run_benchmark(expert,Data(),None,dict(learning_rate=1e-5),cases,tmp_path)
    failure=json.loads((tmp_path/'failure_rank_0.json').read_text())
    assert failure['restart_required'] is True
    assert failure['case']['name']=='bad'
    assert not (tmp_path/'next.benchmark.json').exists()


def test_launcher_validates_untouched_base_before_cuda(tmp_path,monkeypatch):
    from scripts import benchmark_stage1,train_goal_epoch
    import openwam.train.goal_curated as curated
    import openwam.dataloader.curated_goal as data
    import torch.distributed as dist
    base=dict(optimizer_sharding='zero1',optimizer_device='model',dtype='bfloat16',
              sample_fps=3,dataset={},batch_size=32,learning_rate=1e-5)
    before=copy.deepcopy(base)
    monkeypatch.setattr(curated,'load_config',lambda *a,**k:base)
    monkeypatch.setattr(data,'CuratedGoalDataset',lambda **k:object())
    monkeypatch.setattr(dist,'init_process_group',lambda backend,**kw:None)
    monkeypatch.setattr(dist,'is_initialized',lambda:False)
    calls=[]
    def validate(*a,**kw):
        calls.append(kw)
        assert kw['config']==before
        raise ValueError('invalid prepared proof')
    monkeypatch.setattr(train_goal_epoch,'validated_population',validate)
    def no_cuda(*a,**kw):raise AssertionError('proof must precede CUDA')
    monkeypatch.setattr(torch.cuda,'set_device',no_cuda)
    with pytest.raises(ValueError,match='invalid prepared proof'):
        benchmark_stage1.main(['--config','base.yaml','--index','index.sqlite','--prepared-runtime','proof.json',
                              '--output-dir',str(tmp_path),'--case','b128_w8_p16'])
    assert len(calls)==1 and base==before


def _distributed_benchmark_worker(rank,root):
    from datetime import timedelta
    from pathlib import Path
    import torch.distributed as dist
    from openwam.train.goal_benchmark import run_benchmark,parse_cases
    from test_query_goal import make_query
    from test_multiview_patch import multiview_batch
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=(Path(root)/'group').as_uri(),rank=rank,world_size=2,timeout=timedelta(seconds=30))
    try:
        observed,context,targets,valid=multiview_batch(1)
        class Data:
            def __len__(self):return 3
            def sample(self,index):
                return dict(family='tiny',instruction='move'),dict(observed=observed,context=context,targets=targets,observed_view_valid=valid)
        config=dict(seed=42,learning_rate=1e-5,optimizer_sharding='zero1',optimizer_device='model')
        cases=parse_cases([dict(name=name,batch_size=1,num_workers=0,warmup_steps=0,measured_steps=2) for name in ('ddp_first','ddp_reset')],config)
        run_benchmark(make_query(),Data(),None,config,cases,Path(root)/'output',timeout_seconds=30)
    finally:
        dist.destroy_process_group()


def test_two_rank_zero1_executes_tail_dummy_and_resets_cases(tmp_path):
    torch.multiprocessing.spawn(_distributed_benchmark_worker,args=(str(tmp_path),),nprocs=2,join=True)
    summary=json.loads((tmp_path/'output'/'benchmark_summary.json').read_text())
    assert all(case['measured_global_samples']==3 for case in summary['cases'])
    assert all(case['valid_tcp_points']==6 for case in summary['cases'])
    assert all(len(case['per_rank'])==2 for case in summary['cases'])
    rows=[json.loads(line) for line in (tmp_path/'output'/'rank_1.jsonl').read_text().splitlines()]
    updates=[row for row in rows if row['event']=='update']
    assert [row['loss'] for row in updates[:2]]==[row['loss'] for row in updates[2:]]
    assert updates[1]['real_samples']==0 and updates[1]['valid_tcp_points']==0


@pytest.fixture(autouse=True)
def sequential_benchmark_environment(monkeypatch):
    monkeypatch.setenv('OPENWAM_MECKA_DECODE_MODE','sequential')


def test_environment_resolves_cache_policy_and_rejects_lazy_index_builds(tmp_path):
    from openwam.train.goal_benchmark import benchmark_environment
    with pytest.raises(ValueError,match='lazy'):
        benchmark_environment({})
    env=benchmark_environment(dict(OPENWAM_MECKA_DECODE_MODE='indexed',OPENWAM_VIDEO_INDEX_MODE='read',
                                  OPENWAM_VIDEO_INDEX_CACHE=str(tmp_path),OMP_NUM_THREADS='2',MKL_NUM_THREADS='1'))
    assert env['OPENWAM_VIDEO_INDEX_CACHE']==str(tmp_path.resolve())
    assert env['cache_policy']['video_index_builds_allowed'] is False
    assert env['cache_policy']['effective_video_index_mode']=='read'
    assert benchmark_environment({'OPENWAM_MECKA_DECODE_MODE':'sequential'})['cache_policy']['effective_video_index_mode']=='off'


def test_preflight_preserves_old_output_and_collectively_checks_environment(tmp_path,monkeypatch):
    from openwam.train.goal_benchmark import validate_launch_state
    import torch.distributed as dist
    artifact=tmp_path/'benchmark_manifest.json';artifact.write_text('old evidence')
    with pytest.raises(ValueError,match='fresh'):
        validate_launch_state(tmp_path)
    assert artifact.read_text()=='old evidence'
    assert sorted(p.name for p in tmp_path.iterdir())==['benchmark_manifest.json']
    artifact.unlink();(tmp_path/'control').mkdir()
    assert validate_launch_state(tmp_path)['cache_policy']['effective_video_index_mode']=='off'
    monkeypatch.setattr(dist,'is_initialized',lambda:True)
    monkeypatch.setattr(dist,'get_world_size',lambda:2)
    def gather(results,local):
        results[:]=[local,dict(local,environment=dict(local['environment'],OMP_NUM_THREADS='different'))]
    monkeypatch.setattr(dist,'all_gather_object',gather)
    with pytest.raises(ValueError,match='environment differs'):
        validate_launch_state(tmp_path)


def test_update_log_visible_before_following_update(tmp_path):
    from openwam.train.goal_benchmark import run_benchmark,parse_cases
    from test_query_goal import make_query
    from test_multiview_patch import multiview_batch
    observed,context,targets,valid=multiview_batch(1)
    class Data:
        reads=0
        def __len__(self):return 2
        def sample(self,index):
            if self.reads:
                rows=[json.loads(line) for line in (tmp_path/'rank_0.jsonl').read_text().splitlines()]
                assert any(row['event']=='update' for row in rows)
            self.reads+=1
            return dict(family='tiny',instruction='move'),dict(observed=observed,context=context,targets=targets,observed_view_valid=valid)
    config=dict(learning_rate=1e-5)
    cases=parse_cases([dict(name='flush',batch_size=1,num_workers=0,warmup_steps=0,measured_steps=2)],config)
    result=run_benchmark(make_query(),Data(),None,config,cases,tmp_path)
    assert result[0]['cache_policy']['video_index_builds_allowed'] is False


def test_startup_watchdog_is_active_before_first_cuda_call(tmp_path,monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace
    from scripts import benchmark_stage1,train_goal_epoch
    import openwam.train.goal_benchmark as benchmark
    import openwam.train.goal_curated as curated
    import openwam.dataloader.curated_goal as data
    import torch.distributed as dist
    monkeypatch.setattr(curated,'load_config',lambda *a,**k:dict(optimizer_sharding='zero1',optimizer_device='model',dtype='bfloat16',sample_fps=3,dataset={},learning_rate=1e-5))
    monkeypatch.setattr(data,'CuratedGoalDataset',lambda **k:object())
    monkeypatch.setattr(train_goal_epoch,'validated_population',lambda *a,**k:SimpleNamespace(summary={}))
    monkeypatch.setattr(dist,'init_process_group',lambda *a,**k:None)
    monkeypatch.setattr(dist,'destroy_process_group',lambda:None)
    monkeypatch.setattr(dist,'is_initialized',lambda:False)
    monkeypatch.setattr(dist,'all_gather_object',lambda dest,value:dest.__setitem__(0,value))
    active=[]
    @contextmanager
    def timer(seconds,output,rank,phase):
        active.append((phase,seconds))
        try:yield
        finally:active.pop()
    monkeypatch.setattr(benchmark,'watchdog',timer)
    def cuda(*a,**k):
        assert ('cuda-nccl-bootstrap',600) in active
        raise RuntimeError('stop before actual CUDA')
    monkeypatch.setattr(torch.cuda,'set_device',cuda)
    with pytest.raises(RuntimeError,match='stop before actual CUDA'):
        benchmark_stage1.main(['--config','base.yaml','--index','index.sqlite','--prepared-runtime','proof.json',
                              '--output-dir',str(tmp_path),'--startup-timeout-seconds','600'])


def test_launcher_old_evidence_is_not_modified_even_on_rejection(tmp_path,monkeypatch):
    from scripts import benchmark_stage1
    import openwam.train.goal_curated as curated
    import openwam.dataloader.curated_goal as data
    import torch.distributed as dist
    old=tmp_path/'benchmark_bootstrap.json';old.write_text('keep exactly')
    monkeypatch.setattr(curated,'load_config',lambda *a,**k:dict(optimizer_sharding='zero1',optimizer_device='model',dtype='bfloat16',dataset={},learning_rate=1e-5))
    monkeypatch.setattr(dist,'init_process_group',lambda *a,**k:None)
    monkeypatch.setattr(dist,'is_initialized',lambda:False)
    def forbidden(**kw):raise AssertionError('dataset must not start')
    monkeypatch.setattr(data,'CuratedGoalDataset',forbidden)
    with pytest.raises(ValueError,match='fresh'):
        benchmark_stage1.main(['--config','base.yaml','--index','index.sqlite','--prepared-runtime','proof.json','--output-dir',str(tmp_path)])
    assert old.read_text()=='keep exactly'
    assert [p.name for p in tmp_path.iterdir()]==['benchmark_bootstrap.json']


def test_launcher_restores_short_collective_deadline_before_benchmark(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from scripts import benchmark_stage1,train_goal_epoch
    import openwam.train.goal_benchmark as benchmark
    import openwam.train.goal_curated as curated
    import openwam.train.bootstrap as bootstrap
    import openwam.dataloader.curated_goal as data
    import torch.distributed as dist
    monkeypatch.setattr(curated,'load_config',lambda *a,**k:dict(optimizer_sharding='zero1',optimizer_device='model',dtype='bfloat16',sample_fps=3,dataset={},learning_rate=1e-5))
    monkeypatch.setattr(data,'CuratedGoalDataset',lambda **k:object())
    monkeypatch.setattr(train_goal_epoch,'validated_population',lambda *a,**k:SimpleNamespace(summary={}))
    transitions=[]
    monkeypatch.setattr(dist,'init_process_group',lambda backend,**kw:transitions.append((backend,kw['timeout'].total_seconds())))
    monkeypatch.setattr(dist,'destroy_process_group',lambda:None)
    monkeypatch.setattr(dist,'is_initialized',lambda:False)
    monkeypatch.setattr(dist,'all_gather_object',lambda dest,value:dest.__setitem__(0,value))
    monkeypatch.setattr(torch.cuda,'set_device',lambda *a,**k:None)
    monkeypatch.setattr(bootstrap,'run_bootstrap_in_waves',lambda *a,**k:(object(),torch.nn.Linear(1,1),object(),{}))
    monkeypatch.setattr(dist.distributed_c10d,'_set_pg_timeout',lambda timeout:transitions.append(('shorten',timeout.total_seconds())))
    def run(*a,**k):
        assert transitions==[('gloo',180),('nccl',600),('shorten',180)]
        transitions.append(('run',k['timeout_seconds']))
    monkeypatch.setattr(benchmark,'run_benchmark',run)
    assert benchmark_stage1.main(['--config','base.yaml','--index','index.sqlite','--prepared-runtime','proof.json','--output-dir',str(tmp_path)])==0
    assert transitions[-1]==('run',180)
