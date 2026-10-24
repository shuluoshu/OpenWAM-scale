"""Two real Gloo ranks run a three-example population including its partial tail."""
import json
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank, rendezvous, output):
    from test_query_goal import make_query
    from test_multiview_patch import multiview_batch
    from openwam.train.goal_epoch import train_epoch
    from openwam.model.goal.dependencies import dependency_identity
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
