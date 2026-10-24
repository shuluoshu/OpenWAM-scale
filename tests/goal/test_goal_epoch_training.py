"""Finite distributed updates, full precision resume, and fail-closed handoff."""
import json
from pathlib import Path
import pytest
import torch

def test_rank_partition_covers_tail_once_and_weights_real_samples():
    from openwam.train.goal_epoch import rank_sample
    seen=[]
    for cursor in (0,3,6,9):
        samples=[rank_sample(cursor,11,rank,3) for rank in range(3)]
        seen.extend(index for index,weight in samples if weight>0)
        assert sum(weight for _,weight in samples)==pytest.approx(3.)
    assert seen==list(range(11))
    assert rank_sample(9,11,2,3)==(10,0.)
    with pytest.raises(ValueError):rank_sample(12,11,0,3)

def test_optimizer_resume_preserves_fp32_master_and_next_update(tmp_path):
    from openwam.train.goal_epoch import optimizer_payload, restore_optimizer
    from openwam.train.goal_overfit import FP32MasterAdamW
    p=torch.nn.Parameter(torch.tensor([1.,2.],dtype=torch.bfloat16))
    first=FP32MasterAdamW([{'params':[p],'lr':1e-5}])
    p.grad=torch.tensor([.3,.7],dtype=p.dtype);first.step()
    assert not torch.equal(first.master_parameters[0],p.float())
    torch.save(optimizer_payload(first),tmp_path/'state.pt')
    q=torch.nn.Parameter(p.detach().clone());second=FP32MasterAdamW([{'params':[q],'lr':1e-5}])
    restore_optimizer(second,torch.load(tmp_path/'state.pt',weights_only=True))
    p.grad=torch.tensor([.8,.2],dtype=p.dtype);q.grad=p.grad.clone()
    first.step();second.step()
    torch.testing.assert_close(first.master_parameters[0],second.master_parameters[0],rtol=0,atol=0)
    torch.testing.assert_close(p,q,rtol=0,atol=0)

def test_checkpoint_commit_resume_identity_and_incomplete_handoff(tmp_path):
    from openwam.train.goal_epoch import save_epoch_checkpoint,restore_epoch_checkpoint,verify_completion
    from openwam.train.goal_overfit import FP32MasterAdamW
    from test_query_goal import make_query
    from test_multiview_patch import multiview_batch
    from openwam.model.goal.dependencies import dependency_identity
    expert=make_query().train_stage_a();obs,ctx,target,valid=multiview_batch(1)
    opt=FP32MasterAdamW([{'params':list(expert.parameters()),'lr':1e-5}])
    expert(obs,ctx,target,observed_view_valid=valid)['loss'].backward();opt.step()
    identity={'index_sha256':'a'*64,'source_yaml_sha256':'b'*64,'total_samples':11,'world_size':3,'seed':42,'config_sha256':'c'*64}
    deps=dependency_identity(expert.codec.vae,text_dim=expert.text_dim,precomputed_text_fingerprint='d'*64)
    save_epoch_checkpoint(expert,opt,tmp_path,identity=identity,cursor=3,update=1,dependencies=deps)
    clone=make_query().train_stage_a();opt2=FP32MasterAdamW([{'params':list(clone.parameters()),'lr':1e-5}])
    assert restore_epoch_checkpoint(clone,opt2,tmp_path,identity)==(3,1)
    for k,v in expert.state_dict().items():torch.testing.assert_close(v,clone.state_dict()[k],rtol=0,atol=0)
    with pytest.raises(ValueError,match='identity'):restore_epoch_checkpoint(clone,opt2,tmp_path,{**identity,'world_size':2})
    with pytest.raises((ValueError,FileNotFoundError)):verify_completion(tmp_path,identity)

def test_cpu_epoch_finishes_exactly_and_can_resume_without_retraining(tmp_path):
    from openwam.train.goal_epoch import train_epoch,verify_completion
    from test_query_goal import make_query
    from test_multiview_patch import multiview_batch
    from openwam.model.goal.dependencies import dependency_identity
    torch.set_num_threads(2)
    expert=make_query();obs,ctx,target,valid=multiview_batch(1)
    class Data:
        def __len__(self):return 3
        def sample(self,index):
            row=dict(id=str(index),family='tiny',instruction='move',granularity='task')
            return row,dict(observed=obs,context=ctx,targets=target,observed_view_valid=valid)
        def evaluation_samples(self):return [self.sample(0)]
        def scope_summary(self):return {'total_samples':3}
    identity={'index_sha256':'a'*64,'source_yaml_sha256':'b'*64,'total_samples':3,'world_size':1,'seed':42,'config_sha256':'c'*64}
    deps=dependency_identity(expert.codec.vae,text_dim=expert.text_dim,precomputed_text_fingerprint='d'*64)
    result=train_epoch(expert,Data(),None,tmp_path,identity=identity,dependencies=deps,
        learning_rate=1e-5,history_learning_rate=1e-4,checkpoint_every=2,eval_every=2,sampling_steps=2)
    assert result['seen_samples']==result['total_samples']==3
    assert not any(p.requires_grad for p in expert.parameters())
    proof=verify_completion(tmp_path,identity)
    assert proof['seen_samples']==3 and Path(proof['checkpoint']).is_file()
    records=[json.loads(x) for x in (tmp_path/'progress.jsonl').read_text().splitlines()]
    assert [x['cursor'] for x in records if x['event']=='update']==[1,2,3]
    assert (tmp_path/'timeline.html').is_file()
