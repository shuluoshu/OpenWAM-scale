import json
import numpy as np
import pytest
import torch
from PIL import Image
from test_expert import make_expert, batch


def manifest(tmp_path, count=2):
    Image.new('RGB',(640,480),(123,77,42)).save(tmp_path/'image.png')
    np.savez(tmp_path/'target.npz',uv=np.array([[.25,.4],[1.2,.6]],np.float32),
             projection_valid=np.array([True,True]),in_frame=np.array([True,False]),
             visibility=np.array([1,-1],np.int8))
    rows=[dict(family=f'family{i}',source_repo='/source/data',source_episode=i,
        source_episode_length=300,anchor_source_frame=20,goal_source_frame=299,
        clip_start=20,clip_end_exclusive=200,observed_image_path=str(tmp_path/'image.png'),
        goal_image_path=str(tmp_path/'image.png'),target_path=str(tmp_path/'target.npz'),
        instruction='move the object',label_kind='episode_end',success_verified=False,
        provenance={'geometry_review_status':'pending'}) for i in range(count)]
    path=tmp_path/'manifest.json';path.write_text(json.dumps(rows));return path,rows


def test_manifest_rejects_clip_end_in_place_of_full_episode_last_frame(tmp_path):
    from openwam.train.goal_overfit import validate_overfit_manifest
    path,rows=manifest(tmp_path)
    rows[0]['goal_source_frame']=199;path.write_text(json.dumps(rows))
    with pytest.raises(ValueError,match='last frame'):
        validate_overfit_manifest(path,expected_families=2,diagnostic_override=True,allow_unreviewed_geometry=True)


def test_manifest_unique_family_and_explicit_diagnostic_optins(tmp_path):
    from openwam.train.goal_overfit import validate_overfit_manifest
    path,rows=manifest(tmp_path)
    with pytest.raises(ValueError,match='diagnostic'):
        validate_overfit_manifest(path,expected_families=2)
    rows[1]['family']=rows[0]['family'];path.write_text(json.dumps(rows))
    with pytest.raises(ValueError,match='one.*family|duplicate'):
        validate_overfit_manifest(path,expected_families=2,diagnostic_override=True,allow_unreviewed_geometry=True)


def test_manifest_masks_outside_separately_and_retains_unknown_visibility(tmp_path):
    from openwam.train.goal_overfit import validate_overfit_manifest, prepare_overfit_images
    path,_=manifest(tmp_path)
    rows=validate_overfit_manifest(path,expected_families=2,diagnostic_override=True,allow_unreviewed_geometry=True)
    prepared=prepare_overfit_images(rows,device='cpu',dtype=torch.bfloat16)
    target=prepared[0]['targets']
    assert target.point_valid.tolist()==[[True,True]]
    assert target.in_frame.tolist()==[[True,False]]
    assert target.visibility.tolist()==[[1,-1]]
    assert prepared[0]['observed'].shape==(1,3,1,256,320)
    assert prepared[0]['observed'].dtype==torch.bfloat16


def test_actual_tiny_wan_keeps_learning_after_periodic_generation(tmp_path):
    from openwam.train.goal_overfit import train_overfit
    torch.set_num_threads(2)
    expert=make_expert();obs,ctx,target=batch(1)
    before=expert.dit.patch_embedding.weight.detach().clone()
    batches=[dict(observed=obs,context=ctx,targets=target,sources=['tiny'])]
    rows=[dict(family='tiny',source_repo='/synthetic',source_episode=0,anchor_source_frame=1,
        goal_source_frame=99,source_episode_length=100,clip_start=1,clip_end_exclusive=61,
        instruction='test',label_kind='episode_end',success_verified=False,provenance={})]
    result=train_overfit(expert,batches=batches,records=rows,output_dir=tmp_path,
        steps=2,eval_every=1,sampling_steps=1,backbone_lr=.001,point_head_lr=.001,
        precomputed_text_fingerprint='a'*64)
    entries=[json.loads(line) for line in (tmp_path/'progress.jsonl').read_text().splitlines()]
    updates=[row for row in entries if row['event']=='train_step']
    assert len(updates)==2 and all(row['gradient_finite'] for row in updates)
    assert not torch.equal(before,expert.dit.patch_embedding.weight)
    assert [r['step'] for r in result['evaluations']]==[0,1,2]
    ready = [row for row in entries if row['event']=='report_ready']
    assert [row['step'] for row in ready]==[0,1,2]
    for row in ready:
        assert (tmp_path/row['directory']/'index.html').exists()
        assert (tmp_path/row['directory']/'0000_00_generated.png').exists()
    assert result['formal_generalization'] is False
    assert result['evaluations'][-1]['image_mae_uint8'] >= 0
    assert result['valid_points_seen']==4
    assert (tmp_path/'eval_step_000002'/'0000_00_generated_clean.png').exists()
    with np.load(tmp_path/'inputs'/'case_0000.npz') as saved:
        np.testing.assert_array_equal(saved['target'],target.image.cpu().float().numpy())
    assert 'observed.png' in (tmp_path/'index.html').read_text()


def test_bfloat16_export_round_trips_exact_pre_vae_values(tmp_path):
    from openwam.train.goal_overfit import validate_overfit_manifest, prepare_overfit_images, export_inputs
    path,_=manifest(tmp_path)
    rows=validate_overfit_manifest(path,expected_families=2,diagnostic_override=True,allow_unreviewed_geometry=True)
    batches=prepare_overfit_images(rows,device='cpu',dtype=torch.bfloat16)
    for b in batches: b['context']=torch.zeros(1,2,16,dtype=torch.bfloat16)
    export_inputs(tmp_path,rows,batches)
    with np.load(tmp_path/'inputs'/'case_0000.npz') as saved:
        restored=torch.from_numpy(saved['target_bfloat16_bits']).view(torch.bfloat16)
        assert torch.equal(restored,batches[0]['targets'].image)
        np.testing.assert_array_equal(saved['target'],batches[0]['targets'].image.float().numpy())


def test_fp32_master_accumulates_subquantum_bfloat16_backbone_updates():
    from openwam.train.goal_overfit import FP32MasterAdamW
    initial=torch.tensor([.01,.02,.1,1.],dtype=torch.bfloat16)
    parameter=torch.nn.Parameter(initial.clone())
    direct=torch.nn.Parameter(initial.clone())
    optimizer=FP32MasterAdamW([{'params':[parameter],'lr':1e-5}],weight_decay=.01)
    baseline=torch.optim.AdamW([direct],lr=1e-5,weight_decay=.01,foreach=False)
    for step in range(250):
        parameter.grad=torch.ones_like(parameter)
        direct.grad=torch.ones_like(direct)
        optimizer.step();baseline.step();baseline.zero_grad(set_to_none=True)
        assert parameter.grad is None
        assert all(p.grad is None for p in optimizer.master_parameters)
        if step==0:
            assert torch.equal(parameter,initial)  # First update is below BF16 resolution.
            assert not torch.equal(optimizer.master_parameters[0],initial.float())
    assert torch.equal(direct,initial)  # Direct BF16 updates never accumulate.
    assert torch.all(parameter < initial)
    assert parameter.dtype==torch.bfloat16
    for master in optimizer.master_parameters:
        assert master.dtype==torch.float32
        assert optimizer.optimizer.state[master]['exp_avg'].dtype==torch.float32
        assert optimizer.optimizer.state[master]['exp_avg_sq'].dtype==torch.float32
        torch.testing.assert_close(parameter,master.to(torch.bfloat16))


def test_overfit_evaluation_defaults_ignore_unrelated_stage_a_interval():
    from openwam.train.goal_overfit import overfit_eval_every
    assert overfit_eval_every({'eval_every':100})==120
    assert overfit_eval_every({'eval_every':100,'overfit_eval_every':240})==240


def test_input_report_handles_manifest_report_field_collisions(tmp_path):
    from openwam.train.goal_overfit import validate_overfit_manifest, prepare_overfit_images, export_inputs
    path,rows=manifest(tmp_path)
    rows[0].update(id='source_case_123',valid_points='stale',mask_reasons='stale',
                   visibility='stale',target_uv_pixels='stale')
    path.write_text(json.dumps(rows))
    records=validate_overfit_manifest(path,expected_families=2,diagnostic_override=True,allow_unreviewed_geometry=True)
    batches=prepare_overfit_images(records,device='cpu',dtype=torch.bfloat16)
    for b in batches:b['context']=torch.zeros(1,2,16)
    cards=export_inputs(tmp_path,records,batches)
    assert cards[0]['id']=='case_0000'
    assert cards[0]['provenance']['manifest_id']=='source_case_123'
    assert cards[0]['valid_points']==[True,False]
    assert cards[0]['visibility']==[1,-1]
    assert rows[0]['provenance']=={'geometry_review_status':'pending'}


def test_actual_twelve_family_manifest_cpu_input_export(tmp_path):
    from pathlib import Path
    from openwam.train.goal_overfit import validate_overfit_manifest, prepare_overfit_images, export_inputs
    path=Path('/mnt/data/cpfs/b5/yeqingda/openwam-0911/OpenWAM/outputs/g1_per_dataset_overfit_20261018/manifest.json')
    if not path.exists():pytest.skip('local full twelve-family integration fixture unavailable')
    records=validate_overfit_manifest(path,diagnostic_override=True,allow_unreviewed_geometry=True)
    batches=prepare_overfit_images(records,device='cpu',dtype=torch.bfloat16)
    for b in batches:b['context']=torch.zeros(1,2,16)
    cards=export_inputs(tmp_path,records,batches)
    assert len(cards)==12
    for i,(card,row) in enumerate(zip(cards,records)):
        assert card['id']==f'case_{i:04d}'
        assert (tmp_path/'inputs'/f'case_{i:04d}.npz').is_file()
        assert card['goal_source_frame']==card['source_episode_length']-1
        if 'id' in row:assert card['provenance']['manifest_id']==row['id']
