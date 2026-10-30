"""No-head history regression, runnable against frozen source or staged helper."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

CODE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(CODE));sys.path.insert(0,str(CODE/'tests/dataloader'))

@pytest.fixture
def helper(monkeypatch):
    path=os.environ.get('OPENWAM_STAGED_HEAD_HISTORY',str(CODE/'openwam/dataloader/head_history.py'))
    spec=importlib.util.spec_from_file_location('staged_head_history',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    import openwam.dataloader.head_history as public
    import openwam.dataloader.curated_stage2 as stage2
    monkeypatch.setattr(public,'materialize_head_history',module.materialize_head_history)
    monkeypatch.setattr(stage2,'materialize_head_history',module.materialize_head_history)
    return module


def test_missing_head_returns_stable_masked_tensors_and_complete_trace(tmp_path,helper,monkeypatch):
    from scripts.data import prepare_subtask_goals
    monkeypatch.setattr(prepare_subtask_goals,'decode_frames',lambda *a,**kw:pytest.fail('no head video may be decoded'))
    config=helper.HistorySamplingConfig(enabled=True,frame_dropout_prob=.7)
    rng=np.random.default_rng(42);before=copy.deepcopy(rng.bit_generator.state)
    fields,trace=helper.materialize_head_history(tmp_path,{'fps':30},{'episode_index':7,'length':100},None,
        60,0,config,rng,True,sequential=True,dtype=torch.bfloat16)
    assert fields['history_images'].shape==(10,3,1,256,320)
    assert fields['history_images'].dtype==torch.bfloat16 and not fields['history_images'].any()
    assert fields['history_valid'].dtype==torch.bool and not fields['history_valid'].any()
    assert fields['history_ages'].dtype==torch.float32 and not fields['history_ages'].any()
    identity=f'{tmp_path.resolve()}:7'
    expected=int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8],'big')&((1<<63)-1)
    assert fields['history_episode_id'].dtype==torch.int64 and fields['history_episode_id'].item()==expected
    assert trace['episode_identity']==identity and trace['episode_id']==expected
    assert trace['indices']==[-1]*10 and trace['ages_s']==[0.]*10 and trace['valid']==[False]*10
    assert trace['head'] is None and trace['source_video'] is None and trace['video_offset_s'] is None
    assert trace['alignment']=={} and trace['anchor']==60 and trace['selected_start']==0
    assert trace['mask_reason']=='missing_head_camera' and trace['fps']==30
    assert rng.bit_generator.state==before
    json.dumps(trace,allow_nan=False)
    again,_=helper.materialize_head_history(tmp_path,{'fps':30},{'episode_index':7,'length':100},None,
        90,1,config,rng,False)
    assert torch.equal(again['history_episode_id'],fields['history_episode_id'])


@pytest.mark.parametrize('info,ep,anchor,start',[
    ({'fps':float('nan')},{'episode_index':0,'length':10},3,0),
    ({'fps':0},{'episode_index':0,'length':10},3,0),
    ({'fps':30},{'episode_index':0,'length':10},10,0),
    ({'fps':30},{'episode_index':0,'length':10},3,4),
    ({'fps':30},{'episode_index':-1,'length':10},3,0),
])
def test_missing_head_still_rejects_bad_bounds_or_fps(tmp_path,helper,info,ep,anchor,start):
    with pytest.raises(ValueError,match='history|bounds|fps'):
        helper.materialize_head_history(tmp_path,info,ep,None,anchor,start,
            helper.HistorySamplingConfig(enabled=True),np.random.default_rng(0),True)


def test_missing_head_still_rejects_invalid_source_valid_shape(tmp_path,helper):
    with pytest.raises(ValueError,match='source_valid'):
        helper.materialize_head_history(tmp_path,{'fps':30},{'episode_index':0,'length':10},None,3,0,
            helper.HistorySamplingConfig(enabled=True),np.random.default_rng(0),True,source_valid=np.ones(9,bool))


def test_disabled_missing_head_does_not_require_metadata(tmp_path,helper):
    assert helper.materialize_head_history(tmp_path,{}, {},None,0,0,
        helper.HistorySamplingConfig(enabled=False),np.random.default_rng(0),True)==({},{})


def native_sample(tmp_path,helper,monkeypatch):
    from test_curated_stage2 import make_native_source
    from openwam.dataloader.curated_stage2 import CuratedStage2Dataset
    from openwam.dataloader.native_stage2_stats import prepare_native_stats
    from PIL import Image
    selection=make_native_source(tmp_path,subtasks=False)
    root=tmp_path/'source/interndata_a1/unit';path=root/'meta/info.json'
    info=json.loads(path.read_text());del info['features']['observation.images.cam_high']
    wrist='observation.images.cam_left_wrist';info['features'][wrist]={'dtype':'video','shape':[32,40,3]}
    path.write_text(json.dumps(info))
    data=CuratedStage2Dataset(selection,history={'enabled':True},gripper_state_filter={'enabled':False})
    registry=prepare_native_stats(data,tmp_path/'stats',max_episodes_per_profile=1,max_windows_per_profile=16)
    data=CuratedStage2Dataset(selection,history={'enabled':True},normalization_registry=registry,
        observation_augmentation={'enabled':False},gripper_state_filter={'enabled':False})
    def decode(m,indices):
        return ([{wrist:Image.new('RGB',(40,32),(90,120,150))} for _ in indices],
            {wrist:{'alignment':{i:{'source_rgb_sha256':'a'*64,'pts':i} for i in indices}}})
    monkeypatch.setattr(data,'_decode_views',decode)
    sample=data.sample_at(0);metadata=data._episode(data.repos[0],0)
    return data,metadata,sample


def assert_wrist_only_sample(sample):
    assert not sample['goal_head_valid'] and sample['goal_observed_head'] is None
    assert sample['goal_view_valid'].tolist()==[False,True,False]
    assert not sample['history_valid'].any() and not sample['history_images'].any()
    assert sample['trace']['head_history']['indices']==[-1]*10
    assert not sample['action_mask'][:,94:112].any(), 'head-coordinate TCP targets must stay masked'
    assert sample['action_mask'][:,80:86].any(), 'independent robot joints must survive'
    assert sample['action_mask'][:,:3].any(), 'local left TCP translation must survive'
    assert sample['action_mask'][:,9].any(), 'independent left gripper command must survive'
    assert sample['proprio_mask'][:,9].any(), 'independent measured gripper state must survive'
    assert sample['video_view_mask'][:,1].any() and not sample['video_view_mask'][:,0].any()


def test_native_stage2_missing_head_keeps_wrist_and_independent_targets(tmp_path,helper,monkeypatch):
    _,_,sample=native_sample(tmp_path,helper,monkeypatch)
    assert_wrist_only_sample(sample)


def test_fullspan_restore_missing_head_preserves_all_supervision_masks(tmp_path,helper,monkeypatch):
    from scripts.experiments.fullspan_samples import restore_visual_availability
    data,metadata,sample=native_sample(tmp_path,helper,monkeypatch)
    before={key:sample[key].clone() for key in ('action_mask','proprio_mask','action','proprio')}
    restored=restore_visual_availability(data,metadata,sample)
    assert_wrist_only_sample(restored)
    for key,value in before.items():torch.testing.assert_close(restored[key],value,rtol=0,atol=0)
    assert restored['trace']['visual_availability']['native_history_indices']==[-1]*10
