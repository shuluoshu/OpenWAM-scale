"""Two real Gloo ranks run a three-example population including its partial tail."""
import json
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pytest


def _worker(rank, rendezvous, output):
    from test_query_goal import make_query
    from test_multiview_patch import multiview_batch
    from openwam.train.goal_epoch import train_epoch
    from openwam.model.goal.dependencies import dependency_identity
    from openwam.train.epoch_order import sampling_contract
    torch.set_num_threads(1);torch.manual_seed(12)
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2)
    try:
        expert=make_query();obs,ctx,target,valid=multiview_batch(1)
        class Data:
            def __len__(self):return 3
            def sample(self,index):
                return dict(id=str(index),family='tiny',instruction='move',granularity='task'),dict(observed=obs,context=ctx,targets=target,observed_view_valid=valid)
            def evaluation_samples(self):return [self.sample(0)]
            def scope_summary(self):return {'total_samples':3}
        identity=dict(index_sha256='a'*64,source_yaml_sha256='b'*64,total_samples=3,world_size=2,seed=42,config_sha256='c'*64)
        identity['sampling']=sampling_contract({'seed':42},3)
        deps=dependency_identity(expert.codec.vae,text_dim=expert.text_dim,precomputed_text_fingerprint='d'*64)
        train_epoch(expert,Data(),None,output,identity=identity,dependencies=deps,checkpoint_every=1,eval_every=2,sampling_steps=2)
        torch.save(expert.state_dict(),Path(output)/f'rank{rank}.pt')
    finally:dist.destroy_process_group()


def test_two_rank_partial_tail_and_synchronized_weights(tmp_path):
    mp.spawn(_worker,args=(str(tmp_path/'rendezvous'),str(tmp_path/'run')),nprocs=2,join=True)
    output=tmp_path/'run';proof=json.loads((output/'epoch_complete.json').read_text())
    assert proof['seen_samples']==3 and proof['updates']==2
    left=torch.load(output/'rank0.pt',weights_only=True);right=torch.load(output/'rank1.pt',weights_only=True)
    for key in left:torch.testing.assert_close(left[key],right[key],rtol=0,atol=0)
    rows=[json.loads(x) for x in (output/'progress.jsonl').read_text().splitlines()]
    assert [x['cursor'] for x in rows if x['event']=='update']==[2,3]
    updates=[x for x in rows if x['event']=='update']
    assert [x['family_counts'] for x in updates]==[{'tiny':2},{'tiny':1}]
    assert [x['granularity_counts'] for x in updates]==[{'task':2},{'task':1}]
    assert [x['valid_tcp_points'] for x in updates]==[4,2]
    assert [x['samples_with_tcp'] for x in updates]==[2,1]


def _contract_failure_worker(rank,rendezvous,output,mode):
    from openwam.train.goal_epoch import train_epoch
    from openwam.train.epoch_order import sampling_contract
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2)
    try:
        seed=42+rank
        identity=dict(total_samples=7,world_size=2,seed=seed)
        if mode=='mismatch':identity['sampling']=sampling_contract({'seed':seed},7)
        # Validation must reject both ranks before model access or DDP setup.
        with pytest.raises(ValueError,match='sampling'):
            train_epoch(object(),range(7),None,output,identity=identity,dependencies={})
    finally:dist.destroy_process_group()


@pytest.mark.parametrize('mode',['mismatch','legacy'])
def test_all_ranks_reject_mismatched_or_legacy_sample_order(tmp_path,mode):
    mp.spawn(_contract_failure_worker,args=(str(tmp_path/'rendezvous'),str(tmp_path/'run'),mode),nprocs=2,join=True)
