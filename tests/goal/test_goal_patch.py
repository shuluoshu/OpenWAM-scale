import json
import pytest
import torch
from test_expert import batch, TinyVAE, TINY_DIT


def test_continuous_uv_metrics_and_invalid_mask():
    from openwam.train.goal_patch import continuous_point_metrics
    _,_,target=batch(1)
    prediction=target.uv.clone(); prediction[0,0,0]+=0.25/319
    target.point_valid[0,1]=False; target.uv[0,1]=float('nan')
    metrics=continuous_point_metrics(prediction,target,['tiny'])
    assert metrics['overall']['left']['pixel_error']==pytest.approx(.25,abs=1e-4)
    assert metrics['overall']['right']['count']==0
    assert metrics['overall']['right']['pixel_error'] is None
    prediction[0,0,0]=2.
    assert continuous_point_metrics(prediction,target,['tiny'])['overall']['left']['pixel_error']>319


def test_patch_bundle_roundtrip_rejects_wrong_dependencies(tmp_path):
    from openwam.train.goal_patch import save_patch_bundle,load_patch_bundle,make_smoke_data
    expert,batches,records=make_smoke_data()
    path=tmp_path/'patch.safetensors'
    save_patch_bundle(expert,path,precomputed_text_fingerprint='a'*64)
    loaded=load_patch_bundle(path,vae=type(expert.codec.vae)(),text_dim=16,precomputed_text_fingerprint='a'*64)
    assert all(torch.equal(v,loaded.state_dict()[k]) for k,v in expert.state_dict().items())
    assert not any(p.requires_grad for p in loaded.parameters())
    assert json.loads(path.with_suffix('.json').read_text())['format']=='openwam.goal.patch.v1'
    with pytest.raises(ValueError,match='text'):
        load_patch_bundle(path,vae=type(expert.codec.vae)(),text_dim=16,precomputed_text_fingerprint='b'*64)
    from openwam.train.goal_trainer import load_goal_bundle
    with pytest.raises(ValueError):
        load_goal_bundle(path,vae=type(expert.codec.vae)(),text_dim=16,precomputed_text_fingerprint='a'*64)


def test_training_evaluations_append_and_shared_weights_update(tmp_path):
    from openwam.train.goal_patch import train_patch,make_smoke_data
    expert,batches,records=make_smoke_data()
    before=expert.dit.head.head.weight.detach().clone()
    output=tmp_path/'run'
    result=train_patch(expert,batches=batches,records=records,output_dir=output,
        steps=2,eval_every=1,sampling_steps=1,learning_rate=1e-3,
        precomputed_text_fingerprint='a'*64)
    assert result['status']=='complete' and result['completed_steps']==2
    assert [e['step'] for e in result['evaluations']]==[0,1,2]
    assert not torch.equal(before,expert.dit.head.head.weight)
    for step in (0,1,2):
        assert (output/f'eval_step_{step:06d}/0000_00_generated.png').is_file()
    assert 'eval_step_000000' in (output/'timeline.html').read_text()
    assert 'eval_step_000002' in (output/'timeline.html').read_text()
    with pytest.raises(ValueError,match='nonempty'):
        train_patch(expert,batches=batches,records=records,output_dir=output,steps=2)
