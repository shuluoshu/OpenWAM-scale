import json
import numpy as np
import pytest
import torch
from PIL import Image


def test_contiguous_runs_do_not_merge_repeated_ids_or_bridge_gaps():
    from openwam.dataloader.subtask_goal import contiguous_subtasks
    frames={'frame_index':list(range(9)), 'task_index':[0]*9,'subtask_index':[4,4,-1,4,4,5,5,5,5],
            'subtask_unannotated':[False]*7+[True,False]}
    annotations=[{'subtask_index':4,'subtask':'pick cup'},{'subtask_index':5,'subtask':'place cup'}]
    rows=contiguous_subtasks(frames,annotations,9)
    assert [(r['start'],r['end_exclusive']) for r in rows]==[(0,2),(2,3),(3,5),(5,7),(7,8),(8,9)]
    assert rows[0]['subtask_index']==rows[2]['subtask_index']
    assert rows[0]['number']!=rows[2]['number']
    assert not rows[1]['valid'] and not rows[4]['valid'] and rows[5]['valid']


def test_parser_rejects_non_dense_frames_and_ambiguous_text():
    from openwam.dataloader.subtask_goal import contiguous_subtasks
    f=dict(frame_index=[0,2],task_index=[0,0],subtask_index=[1,1])
    with pytest.raises(ValueError,match='dense'):contiguous_subtasks(f,[],2)
    f['frame_index']=[0,1]
    with pytest.raises(ValueError,match='conflicting'):contiguous_subtasks(f,[dict(subtask_index=1,subtask='a'),dict(subtask_index=1,subtask='b')],2)


def make_manifest(tmp_path):
    Image.new('RGB',(320,256),(100,200,50)).save(tmp_path/'image.png')
    np.savez(tmp_path/'targets.npz',uv=np.array([[.1234567,.4],[np.nan,np.nan]],np.float32),
        projection_valid=[True,False],in_frame=[True,False],visibility=np.array([-1,-1],np.int8))
    row=dict(format='openwam.goal.subtask.v1',id='a',family='test',source_repo=str(tmp_path),source_episode=0,
        source_episode_length=20,segment_number=0,subtask_index=1,instruction='pick cup',parent_task='make tea',
        instruction_start=2,instruction_end_exclusive=8,anchor_source_frame=2,goal_source_frame=7,
        label_kind='subtask_end',success_verified=False,split='diagnostic',
        observed_image_path='image.png',goal_image_path='image.png',target_path='targets.npz',
        provenance={'goal_source_frame':7,'geometry_review_status':'pending_review'})
    path=tmp_path/'manifest.jsonl';path.write_text(json.dumps(row)+'\n');return path,row


def test_dataset_matches_joint_expert_input_contract_without_future_conditions(tmp_path):
    from openwam.dataloader.subtask_goal import SubtaskGoalDataset
    path,row=make_manifest(tmp_path)
    ds=SubtaskGoalDataset(path,allow_unreviewed_geometry=True)
    b=ds[0]
    assert b['instruction']=='pick cup' and b['observed'].shape==(1,3,1,256,320)
    assert b['observed'].dtype==torch.bfloat16 and b['targets'].uv.dtype==torch.float32
    assert b['targets'].uv[0,0,0].item()==pytest.approx(.1234567,abs=1e-8)
    assert b['targets'].point_valid.tolist()==[[True,False]]
    assert not SubtaskGoalDataset(path)[0]['targets'].point_valid.any()
    assert 'goal_image' not in b and 'parent_task' not in b


def test_manifest_rejects_episode_end_substitution_and_verified_claim(tmp_path):
    from openwam.dataloader.subtask_goal import load_subtask_manifest
    path,row=make_manifest(tmp_path)
    row['goal_source_frame']=19;path.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='subtask'):load_subtask_manifest(path)
    row['goal_source_frame']=7;row['success_verified']=True;path.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='success'):load_subtask_manifest(path)


def test_recorded_tcp_restoration_and_camera_qa_are_not_borrowed(tmp_path):
    from scripts.data.prepare_subtask_goals import recorded_tcp_transform, camera_eligibility
    t=np.eye(4);t[0,3]=-.15
    info={'starvla_metadata':{'rectification_history':[{'schema_version':'eef_tcp_to_wrist_0804_v1',
        'transforms':[{'arm':side,'T_source_wrist_row_major':t.ravel().tolist()} for side in ('left','right')]}]}}
    result=recorded_tcp_transform('robochallenge_v2',info)
    assert result['left'][0][3]==pytest.approx(.15)
    assert recorded_tcp_transform('robochallenge_v2',{}) is None
    assert camera_eligibility('abc_130k',tmp_path,3)[0] is False
    (tmp_path/'meta').mkdir();(tmp_path/'meta/camera_geometry_eligibility.jsonl').write_text(json.dumps({'episode_index':3,'broadcast_qa_status':'rejected'})+'\n')
    assert camera_eligibility('abc_130k',tmp_path,3)[0] is False


def test_split_leakage_and_duplicate_occurrence_rejected(tmp_path):
    from openwam.dataloader.subtask_goal import load_subtask_manifest
    path,row=make_manifest(tmp_path);second={**row,'id':'b','split':'val','segment_number':1}
    path.write_text(json.dumps(row)+'\n'+json.dumps(second)+'\n')
    with pytest.raises(ValueError,match='leakage'):load_subtask_manifest(path)
    second.update(split='diagnostic',segment_number=0)
    path.write_text(json.dumps(row)+'\n'+json.dumps(second)+'\n')
    with pytest.raises(ValueError,match='occurrence'):load_subtask_manifest(path)


def test_v1_manifest_requires_anchor_at_subtask_start(tmp_path):
    from openwam.dataloader.subtask_goal import load_subtask_manifest
    path,row=make_manifest(tmp_path);row['anchor_source_frame']=3
    path.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='subtask'):load_subtask_manifest(path)


def test_preview_verifier_rejects_frame_and_dtype_mismatch(tmp_path):
    from scripts.data.prepare_subtask_goals import validate_preview_batch
    from openwam.dataloader.subtask_goal import SubtaskGoalDataset
    path,row=make_manifest(tmp_path);b=SubtaskGoalDataset(path,allow_unreviewed_geometry=True)[0]
    row['provenance'].update(goal_video={'source_frame_index':7},observed_video={'source_frame_index':2})
    assert validate_preview_batch(row,b)['target_frame_matches']
    row['provenance']['goal_video']['source_frame_index']=8
    with pytest.raises(ValueError,match='frame'):validate_preview_batch(row,b)
    row['provenance']['goal_video']['source_frame_index']=7;b['observed']=b['observed'].float()
    with pytest.raises(ValueError,match='BF16'):validate_preview_batch(row,b)
