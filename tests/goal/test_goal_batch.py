"""True Stage1 collation, deterministic CPU readers and finite batch tails."""
from dataclasses import fields, replace
import json
import os
import pickle
import random

import numpy as np
import pytest
import torch


class TinyBatchData:
    """Picklable CPU-only stand-in for finite disk materialization."""
    def __init__(self, total=7, fail_index=None):
        from test_multiview_patch import multiview_batch
        self.total=total; self.fail_index=fail_index
        self.obs,self.ctx,self.target,self.valid=multiview_batch(1)
    def __len__(self): return self.total
    def sample(self,index):
        if index==self.fail_index: raise RuntimeError('broken CPU row')
        target=self.target if index%2 else replace(self.target,point_valid=torch.zeros_like(self.target.point_valid))
        record=dict(id=str(index),family='tiny',instruction='move',granularity='task',worker_pid=os.getpid(),
                    random_draw=[random.random(),float(np.random.rand()),float(torch.rand(()))])
        return record,dict(observed=self.obs,context=self.ctx[:,:index%4+1],targets=target,observed_view_valid=self.valid)
    def evaluation_samples(self): return [self.sample(0)]
    def scope_summary(self): return {'total_samples':self.total}


class LargeMetadataData(TinyBatchData):
    def __init__(self):
        super().__init__()
        self.metadata = b'curated-selection' * 300000

    def sample(self, index):
        record, batch = super().sample(index)
        record['metadata_bytes'] = len(self.metadata)
        return record, batch


def test_spawn_payload_is_small_and_workers_keep_full_dataset_metadata():
    from pathlib import Path
    from openwam.train.goal_batch import GoalBatchLoader
    from openwam.train.epoch_order import EpochOrder, sampling_contract
    data = LargeMetadataData()
    order = EpochOrder(sampling_contract({'seed': 42}, len(data)))
    assert len(pickle.dumps(data)) > 4 * 1024 * 1024
    with GoalBatchLoader(data, order, cursor=0, rank=0, world=1,
                         batch_size=2, num_workers=2, prefetch_factor=1) as loader:
        # The spawn pipe must not carry the full curated selection per worker:
        # a multi-MB payload blocks each parent until that child finishes imports.
        assert len(pickle.dumps(loader.wrapper)) < 4096
        snapshot = Path(loader.wrapper.dataset.path)
        assert snapshot.is_file()
        pairs = next(loader)
        assert [int(record['id']) for record, _ in pairs] == [order[0], order[1]]
        assert all(record['metadata_bytes'] == len(data.metadata) for record, _ in pairs)
        assert all(record['worker_pid'] != os.getpid() for record, _ in pairs)
    assert not snapshot.exists()


def test_collation_retains_mixed_masks_history_and_context_lengths():
    from openwam.train.goal_batch import prepare_goal_batch
    from openwam.model.goal.history import history_kwargs
    data=TinyBatchData(); pairs=[data.sample(i) for i in (0,3)]
    for i,(_,batch) in enumerate(pairs):
        batch.update(history_images=torch.zeros(1,10,3,1,256,320),history_ages=torch.arange(10,0,-1)[None].float(),
                     history_valid=torch.tensor([[True]*i+[False]*(10-i)]),history_episode_id=torch.tensor([i]))
    pairs[0][1]['observed_view_valid']=torch.tensor([[False,True,False]])
    pairs[1][1]['targets']=replace(pairs[1][1]['targets'],image_valid=torch.tensor([False]))
    result=prepare_goal_batch(pairs,device='cpu',dtype=torch.float32)
    assert result['observed'].shape==(2,3,1,384,320)
    assert result['context'].shape==(2,4,16)
    assert result['context_lens'].tolist()==[1,4]
    assert result['context'][0,1:].eq(0).all()
    assert result['targets'].point_valid.tolist()==[[False,False],[True,True]]
    assert result['targets'].image_valid.tolist()==[True,False]
    assert result['observed_view_valid'].tolist()==[[False,True,False],[True,True,False]]
    assert history_kwargs(result)['history_images'].shape[0]==2
    assert result['history_episode_id'].tolist()==[0,1]
    assert result['sources']==['tiny','tiny']


def test_real_tiny_expert_takes_collated_optimizer_update():
    from openwam.train.goal_batch import prepare_goal_batch
    from test_query_goal import make_query
    data=TinyBatchData(); batch=prepare_goal_batch([data.sample(0),data.sample(3)],device='cpu',dtype=torch.float32)
    expert=make_query().train_stage_a(); optimizer=torch.optim.AdamW(expert.parameters(),lr=1e-4)
    before=expert.dit.head.head.weight.detach().clone()
    output=expert(batch['observed'],batch['context'],batch['targets'],context_lens=batch['context_lens'],observed_view_valid=batch['observed_view_valid'])
    assert output['velocity'].shape[0]==2 and output['valid_points']==2
    output['loss'].backward();optimizer.step()
    assert not torch.equal(before,expert.dit.head.head.weight)


def test_rank_batch_partition_covers_tail_and_repeat_without_padding_loss():
    from openwam.train.goal_batch import rank_batch
    seen=[]
    for cursor in (0,6):
        ranks=[rank_batch(cursor,7,rank,3,2) for rank in range(3)]
        seen.extend(p for positions,count,weight,consumed in ranks for p in positions[:count])
        assert sum(count for _,count,_,_ in ranks)==min(6,7-cursor)
        assert sum(weight for _,_,weight,_ in ranks)==pytest.approx(3.)
        assert all(consumed==min(6,7-cursor) for _,_,_,consumed in ranks)
    assert sorted(seen)==list(range(7))
    assert rank_batch(6,7,2,3,2)==([6],0,0.,1)


@pytest.mark.parametrize('workers',[0,2])
def test_ordered_reader_resume_uses_consumed_cursor_and_seeded_cpu_workers(workers):
    from openwam.train.goal_batch import GoalBatchLoader
    from openwam.train.epoch_order import EpochOrder,sampling_contract
    data=TinyBatchData();order=EpochOrder(sampling_contract({'seed':42},7))
    with GoalBatchLoader(data,order,cursor=0,rank=0,world=1,batch_size=2,num_workers=workers,prefetch_factor=1,seed=42) as loader:
        first=next(loader); second=next(loader)
    with GoalBatchLoader(data,order,cursor=2,rank=0,world=1,batch_size=2,num_workers=workers,prefetch_factor=1,seed=42) as loader:
        resumed=list(loader)
    assert [int(r['id']) for r,_ in first]==[order[i] for i in range(2)]
    assert [r['id'] for r,_ in second]==[r['id'] for r,_ in resumed[0]]
    assert [r['random_draw'] for r,_ in second]==[r['random_draw'] for r,_ in resumed[0]]
    assert [len(batch) for batch in resumed]==[2,2,1]
    if workers: assert all(r['worker_pid']!=os.getpid() for batch in resumed for r,_ in batch)
    for batch in resumed:
        assert all(value.device.type=='cpu' for _,b in batch for value in (b['observed'],b['context'],b['targets'].image))


def test_worker_exception_reaches_caller():
    from openwam.train.goal_batch import GoalBatchLoader
    from openwam.train.epoch_order import EpochOrder,sampling_contract
    order=EpochOrder(sampling_contract({'seed':42,'sample_order':'sequential'},7))
    with GoalBatchLoader(TinyBatchData(fail_index=2),order,cursor=0,rank=0,world=1,batch_size=2,num_workers=2,prefetch_factor=1,seed=42) as loader:
        assert len(next(loader))==2
        with pytest.raises(RuntimeError,match='broken CPU row'):next(loader)


def _identity(total,world=1,workers=0):
    from openwam.train.epoch_order import sampling_contract
    return dict(index_sha256='a'*64,source_yaml_sha256='b'*64,total_samples=total,world_size=world,seed=42,
                config_sha256='c'*64,sampling=sampling_contract({'seed':42},total),
                data_loading=dict(batch_size=2,dataset_num_workers=workers,prefetch_factor=1))


@pytest.mark.parametrize('workers',[0,2])
def test_batched_trainer_resume_matches_uninterrupted_real_updates(tmp_path,workers):
    from openwam.train.goal_epoch import train_epoch
    from openwam.train.epoch_order import EpochOrder
    from openwam.model.goal.dependencies import dependency_identity
    from test_query_goal import make_query
    identity=_identity(7,workers=workers);order=EpochOrder(identity['sampling'])
    expert=make_query();deps=dependency_identity(expert.codec.vae,text_dim=expert.text_dim,precomputed_text_fingerprint='d'*64)
    kwargs=dict(identity=identity,dependencies=deps,batch_size=2,dataset_num_workers=workers,prefetch_factor=1,
                checkpoint_every=1,eval_every=20,sampling_steps=1)
    with pytest.raises(RuntimeError,match='broken CPU row'):
        train_epoch(expert,TinyBatchData(fail_index=order[4]),None,tmp_path/'run',**kwargs)
    checkpoint=json.loads((tmp_path/'run/latest.json').read_text())['directory']
    assert json.loads(open(checkpoint+'/manifest.json').read())['cursor']==4
    resumed=make_query();proof=train_epoch(resumed,TinyBatchData(),None,tmp_path/'run',resume=True,**kwargs)
    complete=make_query();train_epoch(complete,TinyBatchData(),None,tmp_path/'full',**kwargs)
    assert proof['seen_samples']==7 and proof['updates']==4
    for key,value in complete.state_dict().items():torch.testing.assert_close(value,resumed.state_dict()[key],rtol=0,atol=0)
    rows=[json.loads(line) for line in (tmp_path/'full/progress.jsonl').read_text().splitlines()]
    updates=[row for row in rows if row['event']=='update']
    assert [row['cursor'] for row in updates]==[2,4,6,7]
    assert [row['real_samples'] for row in updates]==[2,2,2,1]
    assert all(row['batch_size']==2 and row['global_batch_size']==2 for row in updates)
    assert all(row['loader_wait_s']>=0 and row['batch_prepare_host_s']>=0 and row['update_host_s']>0 for row in updates)
    assert sum(row['samples_with_tcp'] for row in updates)==3
    assert [sum(row['family_counts'].values()) for row in updates]==[2,2,2,1]


def test_finite_dataset_pickle_drops_open_readers_and_retains_parent(tmp_path):
    from test_goal_epoch_data import Source
    from openwam.dataloader.goal_epoch import build_epoch_index,FiniteGoalDataset
    source=Source(tmp_path);build_epoch_index(source,tmp_path/'index.db')
    dataset=FiniteGoalDataset(source,tmp_path/'index.db');expected=dataset.sample(0)
    clone=pickle.loads(pickle.dumps(dataset))
    assert clone._connection is None and not clone._validated
    assert clone.sample(0)==expected
    assert dataset._connection is not None


def test_curated_pickle_drops_cached_episode_and_evaluation_payloads(tmp_path):
    from test_curated_goal import fixture
    from openwam.dataloader.curated_goal import CuratedGoalDataset
    config,_=fixture(tmp_path);dataset=CuratedGoalDataset(config,tmp_path/'cache')
    dataset._episodes['unpicklable']=lambda:None;dataset._evaluations=[lambda:None]
    clone=pickle.loads(pickle.dumps(dataset))
    assert not clone._episodes and clone._evaluations is None
    assert dataset._episodes


def test_epoch_identity_records_validated_batch_and_prefetch_contract():
    from openwam.train.stage12 import epoch_identity
    population=dict(source_yaml_sha256='a'*64,index_sha256='b'*64,total_samples=7)
    identity=epoch_identity(dict(batch_size=2,dataset_num_workers=3,prefetch_factor=1),population,2)
    assert identity['data_loading']==dict(batch_size=2,dataset_num_workers=3,prefetch_factor=1)
    with pytest.raises(ValueError,match='batch_size'):epoch_identity(dict(batch_size=0),population,1)
    with pytest.raises(ValueError,match='dataset_num_workers'):epoch_identity(dict(dataset_num_workers=-1),population,1)


def _distributed_batch_worker(rank,rendezvous,output,mode):
    from pathlib import Path
    from datetime import timedelta
    import torch.distributed as dist
    from openwam.train.goal_epoch import train_epoch
    from openwam.train.epoch_order import EpochOrder
    from openwam.model.goal.dependencies import dependency_identity
    from test_query_goal import make_query
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2,timeout=timedelta(seconds=90))
    try:
        total=5 if mode=='empty_tail' else 7
        identity=_identity(total,world=2,workers=1)
        expert=make_query();deps=dependency_identity(expert.codec.vae,text_dim=expert.text_dim,precomputed_text_fingerprint='d'*64)
        fail_index=EpochOrder(identity['sampling'])[1] if mode=='error' and rank==1 else None
        kwargs=dict(identity=identity,dependencies=deps,batch_size=2,dataset_num_workers=1,prefetch_factor=1,
                    checkpoint_every=1,eval_every=20,sampling_steps=1)
        if mode=='error':
            with pytest.raises(RuntimeError,match='sample preparation failed.*broken CPU row'):
                train_epoch(expert,TinyBatchData(total,fail_index),None,output,**kwargs)
            Path(output,f'error_rank{rank}').write_text('propagated')
        else:
            train_epoch(expert,TinyBatchData(total),None,output,**kwargs)
            torch.save(expert.state_dict(),Path(output,f'rank{rank}.pt'))
    finally:dist.destroy_process_group()


@pytest.mark.parametrize('mode',['empty_tail','unequal_tail','error'])
def test_two_rank_real_batches_and_worker_failure_synchronize(tmp_path,mode):
    import torch.multiprocessing as mp
    mp.spawn(_distributed_batch_worker,args=(str(tmp_path/'rendezvous'),str(tmp_path/'run'),mode),nprocs=2,join=True)
    output=tmp_path/'run'
    if mode=='error':
        assert (output/'error_rank0').is_file() and (output/'error_rank1').is_file()
        return
    total=5 if mode=='empty_tail' else 7
    proof=json.loads((output/'epoch_complete.json').read_text())
    assert proof['seen_samples']==total and proof['updates']==2
    left=torch.load(output/'rank0.pt',weights_only=True);right=torch.load(output/'rank1.pt',weights_only=True)
    for key in left:torch.testing.assert_close(left[key],right[key],rtol=0,atol=0)
    updates=[row for line in (output/'progress.jsonl').read_text().splitlines() if (row:=json.loads(line))['event']=='update']
    assert [row['cursor'] for row in updates]==[4,total]
    assert sum(row['samples_with_tcp'] for row in updates)==total//2
    assert sum(row['valid_tcp_points'] for row in updates)==2*(total//2)
    assert [row['family_counts'] for row in updates]==[{'tiny':4},{'tiny':total-4}]


def test_repeated_population_keeps_small_tail_before_next_cycle():
    from openwam.train.goal_batch import GoalBatchLoader
    from openwam.train.epoch_order import EpochOrder,sampling_contract
    data=TinyBatchData(5);order=EpochOrder(sampling_contract({'seed':42},5))
    with GoalBatchLoader(data,order,cursor=0,rank=0,world=1,batch_size=2,seed=42,repeat_population=True,max_batches=5) as loader:
        batches=list(loader)
    assert [len(batch) for batch in batches]==[2,2,1,2,2]
    assert [int(r['id']) for batch in batches for r,_ in batch]==[order[i] for i in range(5)]+[order[i] for i in range(4)]
    assert batches[0][0][0]['random_draw']!=batches[3][0][0]['random_draw']


def test_real_finite_spawn_readers_decode_without_training_assets(tmp_path):
    from pathlib import Path
    import yaml
    av=pytest.importorskip('av')
    from test_curated_goal import fixture
    from openwam.dataloader.curated_goal import CuratedGoalDataset
    from openwam.dataloader.goal_epoch import FiniteGoalDataset,build_epoch_index
    from openwam.train.goal_batch import GoalBatchLoader
    from openwam.train.epoch_order import EpochOrder,sampling_contract
    config,root=fixture(tmp_path)
    selection=yaml.safe_load(config.read_text());selection['selection']['segments_by_repo']={'mecka/fixture':[[0,0,8]]}
    config.write_text(yaml.safe_dump(selection))
    with av.open(str(root/'video.mp4'),'w') as container:
        stream=container.add_stream('mpeg4',rate=10);stream.width=32;stream.height=24;stream.pix_fmt='yuv420p'
        for index in range(10):
            frame=av.VideoFrame.from_ndarray(np.full((24,32,3),index*20,dtype=np.uint8),format='rgb24')
            for packet in stream.encode(frame):container.mux(packet)
        for packet in stream.encode():container.mux(packet)
    source=CuratedGoalDataset(config,tmp_path/'cache',dtype=torch.float32)
    build_epoch_index(source,tmp_path/'index.db');finite=FiniteGoalDataset(source,tmp_path/'index.db')
    finite.evaluation_samples()  # Prime live SQLite and episode/evaluation caches before spawn.
    order=EpochOrder(sampling_contract({'seed':42},len(finite)))
    with GoalBatchLoader(finite,order,cursor=0,rank=0,world=1,batch_size=2,num_workers=2,prefetch_factor=1,seed=42) as loader:
        pairs=[pair for batch in loader for pair in batch]
    assert len(pairs)==len(finite)
    assert all('observed_image_path' not in record for record,_ in pairs)
    assert not list((tmp_path/'cache/workers').rglob('*.png'))
    assert not list((tmp_path/'cache/workers').rglob('*.npz'))
    assert source.cache_dir==tmp_path/'cache'
    for position,(record,batch) in enumerate(pairs):
        reference_record,reference=finite.sample(order[position])
        assert record['id']==reference_record['id']
        torch.testing.assert_close(batch['observed'],reference['observed'],rtol=0,atol=0)
        assert not batch['targets'].point_valid.any()


def _batch_contract_mismatch_worker(rank,rendezvous,output):
    import torch.distributed as dist
    from openwam.train.goal_epoch import train_epoch
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2)
    try:
        identity=_identity(7,world=2)
        identity['data_loading']['batch_size']=rank+1
        with pytest.raises(ValueError,match='differs between ranks'):
            train_epoch(object(),range(7),None,output,identity=identity,dependencies={},batch_size=rank+1,prefetch_factor=1)
    finally:dist.destroy_process_group()


def test_all_ranks_reject_different_batch_contract_before_model_setup(tmp_path):
    import torch.multiprocessing as mp
    mp.spawn(_batch_contract_mismatch_worker,args=(str(tmp_path/'rendezvous'),str(tmp_path/'run')),nprocs=2,join=True)


def test_first_training_batch_distributes_sample_reads_across_workers():
    from openwam.train.goal_batch import GoalBatchLoader
    from openwam.train.epoch_order import EpochOrder,sampling_contract
    data=TinyBatchData(7);order=EpochOrder(sampling_contract({'seed':42},7))
    with GoalBatchLoader(data,order,cursor=0,rank=0,world=1,batch_size=4,num_workers=2,prefetch_factor=1,seed=42) as loader:
        first=next(loader)
    assert [int(r['id']) for r,_ in first]==[order[i] for i in range(4)]
    assert len({r['worker_pid'] for r,_ in first})==2


def _rank_local_loading_error_worker(rank,rendezvous,output,mode):
    from datetime import timedelta
    import torch.distributed as dist
    from openwam.train.goal_epoch import train_epoch
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2,timeout=timedelta(seconds=15))
    try:
        identity=_identity(7,world=2);kwargs=dict(batch_size=2,dataset_num_workers=0,prefetch_factor=1)
        expected='data loading identity mismatch' if mode=='identity' else mode
        if rank==0:
            if mode=='identity':identity['data_loading']['batch_size']=1
            elif mode=='budget':
                kwargs['batch_size']=4;identity['data_loading']['batch_size']=4
            else:kwargs[mode]=0 if mode!='dataset_num_workers' else -1
        if mode=='budget':
            identity['training']=dict(mode='bounded_updates',max_updates=2,repeat_population=False)
            kwargs['max_updates']=2;expected='max_updates exceeds one population'
        with pytest.raises(ValueError,match='sampling initialization failed.*'+expected):
            train_epoch(object(),range(7),None,output,identity=identity,dependencies={},**kwargs)
    finally:dist.destroy_process_group()


@pytest.mark.parametrize('mode',['batch_size','dataset_num_workers','prefetch_factor','identity','budget'])
def test_rank_local_loading_error_is_exchanged_before_model_setup(tmp_path,mode):
    import torch.multiprocessing as mp
    mp.spawn(_rank_local_loading_error_worker,args=(str(tmp_path/'rendezvous'),str(tmp_path/'run'),mode),nprocs=2,join=True)


@pytest.mark.parametrize('workers',[0,2])
def test_seeded_sample_cache_matches_uncached_repeats_and_resume(workers):
    from openwam.train.goal_batch import GoalBatchLoader
    from openwam.train.epoch_order import EpochOrder,sampling_contract
    data=TinyBatchData(4);order=EpochOrder(sampling_contract({'seed':42},4))
    kwargs=dict(rank=0,world=1,batch_size=2,num_workers=workers,prefetch_factor=1,seed=42,repeat_population=True)
    with GoalBatchLoader(data,order,cursor=0,max_batches=4,sample_cache_size=8,**kwargs) as loader:
        cached=list(loader)
    with GoalBatchLoader(data,order,cursor=0,max_batches=4,sample_cache_size=0,**kwargs) as loader:
        uncached=list(loader)
    with GoalBatchLoader(data,order,cursor=4,max_batches=2,sample_cache_size=8,**kwargs) as loader:
        resumed=list(loader)
    draws=lambda batches:[r['random_draw'] for batch in batches for r,_ in batch]
    assert draws(cached)==draws(uncached)
    assert draws(cached[2:])==draws(resumed)
    assert draws(cached[:2])!=draws(cached[2:])
