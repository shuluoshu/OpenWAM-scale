"""Unavailable calibration masks TCP supervision while retaining the goal pair."""
import json
import numpy as np
import pytest
import torch


@pytest.mark.parametrize('field,corruption',[
    ('intrinsic','nan'),('intrinsic','inf'),('intrinsic','zero'),
    ('world_to_camera','nan'),('world_to_camera','inf'),('world_to_camera','zero'),
])
def test_invalid_present_calibration_returns_maskable_reason(field,corruption):
    from scripts.data.prepare_goal_labels import goal_camera
    row={'observation.states.cameras.cam_high.intrinsic':np.array([[100.,0,16],[0,100,12],[0,0,1]]),
         'observation.states.cameras.cam_high.world_to_camera':np.eye(4)}
    key='observation.states.cameras.cam_high.'+field
    row[key][:]=dict(nan=np.nan,inf=np.inf,zero=0.)[corruption]
    info={'features':{'observation.images.cam_high':{'shape':[24,32,3]}}}
    camera,provenance=goal_camera('morphirobot0930',row,'observation.images.cam_high',info,(32,24))
    assert camera is None
    assert provenance['missing_reasons']


def test_invalid_distortion_is_rejected_and_export_roundoff_is_preserved():
    from scripts.data.prepare_goal_labels import goal_camera
    head='observation.images.egocentric'
    info={'features':{head:{'shape':[24,32,3]}}}
    e=np.eye(4);e[3,3]=0.9999997616
    row={'observation.states.camera.intrinsic':[100,100,16,12,0,0,0,np.nan],
         'observation.states.camera.world_to_camera':e}
    camera,provenance=goal_camera('mecka',row,head,info,(32,24))
    assert camera is None and provenance['missing_reasons']
    row['observation.states.camera.intrinsic'][-1]=0
    camera,provenance=goal_camera('mecka',row,head,info,(32,24))
    assert camera is not None and not provenance['missing_reasons']
    np.testing.assert_array_equal(camera[1],e)


def test_nan_calibration_keeps_images_instruction_and_masks_tcp(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from test_multiview_curated import native_fixture
    from openwam.dataloader.curated_goal import CuratedGoalDataset
    source,root=native_fixture(tmp_path)
    path=root/'data.parquet';rows=pq.read_table(path).to_pylist()
    k=np.array([[20.,0,16],[0,20.,12],[0,0,1]])
    hands=np.zeros((2,21,3));hands[:,4]=[[0,0,2],[.2,0,2]];hands[:,8]=[[.1,0,2],[.3,0,2]]
    for row in rows:
        row.update({'observation.states.camera.intrinsic':k.ravel().tolist(),
                    'observation.states.camera.world_to_camera':np.eye(4).ravel().tolist(),
                    'observation.states.hand.position':hands.ravel().tolist()})
    pq.write_table(pa.Table.from_pylist(rows),path)
    clean_record,clean=CuratedGoalDataset(source,tmp_path/'clean',observation_mode='multiview').sample(0)
    assert clean['targets'].point_valid.all() and clean['targets'].in_frame.all()
    for row in rows:
        row['observation.states.camera.intrinsic']=[float('nan')]*9
        row['observation.states.camera.world_to_camera']=[float('nan')]*16
    pq.write_table(pa.Table.from_pylist(rows),path)
    record,batch=CuratedGoalDataset(source,tmp_path/'invalid',observation_mode='multiview').sample(0)
    assert record['instruction']==clean_record['instruction']
    assert record['goal_source_frame']==clean_record['goal_source_frame']
    torch.testing.assert_close(batch['observed'],clean['observed'],rtol=0,atol=0)
    torch.testing.assert_close(batch['targets'].image,clean['targets'].image,rtol=0,atol=0)
    assert not batch['targets'].point_valid.any() and not batch['targets'].in_frame.any()
    assert torch.isfinite(batch['targets'].uv).all()
    assert record['provenance']['camera']['missing_reasons']
    json.dumps(record,allow_nan=False)
