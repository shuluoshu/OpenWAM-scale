"""In-memory Stage1 inputs match the historical lossless PNG/NPZ reader."""
from dataclasses import fields

import numpy as np
from PIL import Image
import pytest
import torch


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
@pytest.mark.parametrize('views',[None,[True,False,True],[False,True,False]])
def test_memory_inputs_equal_exported_inputs(tmp_path,dtype,views):
    from openwam.train.goal_input import prepare_goal_images
    from openwam.train.goal_overfit import prepare_overfit_images
    rng=np.random.default_rng(13)
    observed=Image.fromarray(rng.integers(0,256,(384 if views else 256,320,3),dtype=np.uint8))
    goal=Image.fromarray(rng.integers(0,256,(79,91,3),dtype=np.uint8))
    projection=dict(uv=np.array([[.1234567,.6],[0,0]],np.float32),
        projection_valid=np.array([True,False]),in_frame=np.array([False,False]),visibility=np.array([-1,0],np.int8))
    row=dict(family='fixture',goal_image_valid=views!=[False,True,False],
        observed_image_path=str(tmp_path/'observed.png'),goal_image_path=str(tmp_path/'goal.png'),target_path=str(tmp_path/'targets.npz'))
    if views is not None:row.update(observation_mode='multiview',view_valid=views)
    observed.save(row['observed_image_path'])
    goal.resize((320,256),Image.Resampling.BILINEAR).save(row['goal_image_path'])
    np.savez_compressed(row['target_path'],**projection)
    reference=prepare_overfit_images([row],device='cpu',dtype=dtype)[0]
    actual=prepare_goal_images(row,observed,goal,projection,dtype=dtype)
    assert actual.keys()==reference.keys()
    for key in actual:
        if key=='targets':
            for field in fields(actual[key]):
                torch.testing.assert_close(getattr(actual[key],field.name),getattr(reference[key],field.name),rtol=0,atol=0)
        elif isinstance(actual[key],torch.Tensor):torch.testing.assert_close(actual[key],reference[key],rtol=0,atol=0)
        else:assert actual[key]==reference[key]


def test_custom_targets_pin_every_tensor_without_dtype_conversion(monkeypatch):
    from openwam.model.goal.types import GoalTargets
    target=GoalTargets(torch.zeros(1),torch.ones(1,dtype=torch.bool),torch.zeros(1,dtype=torch.float32),
        torch.zeros(1,dtype=torch.bool),torch.zeros(1,dtype=torch.bool),torch.zeros(1,dtype=torch.int8))
    calls=[]
    def pin(tensor):
        calls.append(tensor)
        return tensor.clone()
    monkeypatch.setattr(torch.Tensor,'pin_memory',pin)
    from torch.utils.data._utils.pin_memory import pin_memory
    result=pin_memory({'targets':target})['targets']
    assert len(calls)==6
    for field in fields(target):
        original=getattr(target,field.name);pinned=getattr(result,field.name)
        assert original is not pinned
        torch.testing.assert_close(original,pinned,rtol=0,atol=0)


@pytest.mark.parametrize('views',[('left',),('left','right')])
def test_dataset_training_writes_no_assets_but_evaluation_keeps_assets(tmp_path,views):
    from test_multiview_curated import native_fixture
    from openwam.dataloader.curated_goal import CuratedGoalDataset
    from openwam.train.goal_overfit import prepare_overfit_images
    source,_=native_fixture(tmp_path,wrists=views)
    dataset=CuratedGoalDataset(source,tmp_path/'cache',observation_mode='multiview')
    record,batch=dataset.sample(0)
    assert record['provenance']['input_materialization']=='in_memory'
    assert not any(key in record for key in ('observed_image_path','goal_image_path','target_path'))
    assert not list(dataset.cache_dir.rglob('*'))
    assert record['provenance']['observed_views']['head']['source_video']
    assert not batch['targets'].point_valid.any()  # fixture has no TCP geometry
    for row,evaluation in dataset.evaluation_samples():
        assert row['provenance']['input_materialization']=='exported_assets'
        reference=prepare_overfit_images([row],device='cpu',dtype=dataset.dtype)[0]
        torch.testing.assert_close(evaluation['observed'],reference['observed'],rtol=0,atol=0)
        for field in fields(evaluation['targets']):
            torch.testing.assert_close(getattr(evaluation['targets'],field.name),getattr(reference['targets'],field.name),rtol=0,atol=0)


def test_loader_exposes_pinning_without_allocating_gpu_memory():
    from openwam.train.goal_batch import GoalBatchLoader
    loader=GoalBatchLoader(range(1),[0],cursor=0,rank=0,world=1,pin_memory=True)
    assert loader.loader.pin_memory is True
    with pytest.raises(ValueError,match='pin_memory'):
        GoalBatchLoader(range(1),[0],cursor=0,rank=0,world=1,pin_memory='true')


def test_missing_head_keeps_wrist_and_masks_goal_in_memory(tmp_path):
    import json
    from test_multiview_curated import native_fixture
    from openwam.dataloader.curated_goal import CuratedGoalDataset
    source,root=native_fixture(tmp_path,wrists=('left',))
    path=root/'meta/info.json';info=json.loads(path.read_text())
    del info['features']['observation.images.head_left']
    path.write_text(json.dumps(info))
    record,batch=CuratedGoalDataset(source,tmp_path/'cache',observation_mode='multiview').sample(0)
    assert record['view_valid']==[False,True,False]
    assert record['provenance']['endpoint']['mask_reason']=='missing_head_camera'
    assert not batch['targets'].image_valid.any()
    assert not batch['targets'].point_valid.any()
    assert not batch['targets'].in_frame.any()
    assert batch['observed'][...,:256,:].eq(-1).all()
    assert not list((tmp_path/'cache').rglob('*'))
