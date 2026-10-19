import copy
import importlib
import json
from pathlib import Path
import numpy as np
import pytest

from openwam.dataloader.utils.mora_relative import MoraHeadTCPRepresentation


def test_native_measured_gripper_uses_verified_state_and_keeps_pose_independent():
    info={'features':{'observation.states.eef':{'shape':[20]},'observation.state':{'shape':[2]}}}
    rep=MoraHeadTCPRepresentation(info,joint_state_key='absent', gripper_state_sources={
        'left':dict(key='observation.state',index=0,scale=.1,provenance='0..10 openness'),
    })
    cols={'observation.states.eef':np.ones((3,20)), 'observation.state':np.array([[2,9],[3,8],[20,7]])}
    measured=rep.measured_grippers(cols,3)
    np.testing.assert_allclose(measured[:,0],[.2,.3,1])
    assert np.isnan(measured[:,1]).all()
    action,mask,state,sm=rep.encode_window(cols)
    assert state[0,9]==pytest.approx(.2) and sm[0,9] and not sm[0,19]
    assert not mask[:,[9,19]].any(), 'measured state cannot stand in for controller command'


def test_native_sources_change_stats_fingerprint():
    info={'features':{'native':{'shape':[2]}}}
    a=MoraHeadTCPRepresentation(info,gripper_state_sources={})
    b=MoraHeadTCPRepresentation(info,gripper_state_sources={'left':{'key':'native','index':0}})
    assert a.contract['schema_fingerprint']!=b.contract['schema_fingerprint']


def robot_info(robot='ARX Lift-2', count=6):
    names=[*[f'left_joint_{i}' for i in range(count)],'left_gripper',*[f'right_joint_{i}' for i in range(count)],'right_gripper']
    return {'features':{'observation.states.joint':{'shape':[len(names)],'names':{'motors':names},'unit':'radian','gripper_unit':'openness_0_closed_1_open'},
        'states.left_tcp_to_robot_pose':{'shape':[7]},'states.robot_to_env_pose':{'shape':[7]},
        'observation.images.cam_high':{'dtype':'video','shape':[32,40,3]}},
        'starvla_metadata':{'robot_type':robot,'arm_joint_unit':'radian','gripper_normalization':{'convention':'0=closed, 1=open','range':[0,1]}}}


def test_profile_preserves_single_side_tcp_and_joint_without_camera_calibration():
    from openwam.dataloader.native_stage2_profiles import resolve_native_profile
    p=resolve_native_profile('interndata_a1/x',robot_info())
    assert p['representation']['tcp_pose_keys']=={'left':'states.left_tcp_to_robot_pose'}
    assert len(p['representation']['joint_indices']['right'])==6
    assert not p['representation']['head_camera_available']
    assert p['cameras']==['observation.images.cam_high',None,None]
    assert p['fk_supervision'] is None


def test_profile_rejects_unknown_joint_units_and_order():
    from openwam.dataloader.native_stage2_profiles import resolve_native_profile
    i=robot_info();i['features']['observation.states.joint']['unit']='degrees'
    with pytest.raises(ValueError,match='joint_unit'):resolve_native_profile('interndata_a1/x',i)
    i=robot_info();i['features']['observation.states.joint']['names']['motors'][0]='unknown'
    with pytest.raises(ValueError,match='joint_order'):resolve_native_profile('interndata_a1/x',i)


def test_fr3_unconfirmed_gripper_does_not_erase_seven_joints():
    from openwam.dataloader.native_stage2_profiles import resolve_native_profile
    i=robot_info('robomind_franka_fr3_dual',7)
    f=i['features'];f['observation.state']=f.pop('observation.states.joint')
    f['observation.state']['names']['motors']=[*[f'puppet_joint_position_{n}' for n in range(7)],'left_gripper',*[f'puppet_joint_position_{n}' for n in range(8,15)],'right_gripper']
    f['observation.images.cam_front']=f.pop('observation.images.cam_high')
    i['starvla_metadata']['gripper_normalization']={'status':'source_native_unconverted','convention':'source_native_unconfirmed'}
    p=resolve_native_profile('robomind_v1/x',i)
    assert len(p['representation']['joint_indices']['left'])==7
    assert p['representation']['gripper_state_sources']=={}
    assert 'unverified_gripper_state' in p['mask_reasons']


def test_language_ranges_repeat_ids_keep_disjoint_occurrences():
    from openwam.dataloader.curated_goal import language_intervals
    frames=dict(frame_index=list(range(8)),task_index=[0]*8,subtask_index=[1,1,2,2,1,1,-1,-1])
    tasks,subs=language_intervals(frames,{0:'task'},[{'subtask_index':1,'subtask':'one'},{'subtask_index':2,'subtask':'two'}],8)
    assert [(s['start'],s['end_exclusive']) for s in subs]==[(0,2),(2,4),(4,6)]
    with pytest.raises(ValueError,match='conflicting'):
        language_intervals(frames,{0:'task'},[{'subtask_index':1,'subtask':'one'},{'subtask_index':1,'subtask':'wrong'}],8)


def test_window_options_do_not_cross_gaps_or_terminal_anchor():
    from openwam.dataloader.native_stage2_io import window_options
    m=dict(length=20,tasks=[dict(start=0,end_exclusive=20,text='x')],subtasks=[])
    options=window_options(m,[(0,2,5),(0,11,14)],'task')
    assert [(o['start'],o['end_exclusive'],o['last_source']) for o in options]==[(2,4,4),(11,13,13)]


def test_source_masks_keep_independent_joints_and_mask_future_and_anchor_pose():
    from openwam.dataloader.native_stage2_io import merge_source_masks
    rep=MoraHeadTCPRepresentation({'features':{}})
    a=np.ones((2,52),bool);s=np.ones((1,52),bool)
    cols={'observation.states.eef.mask':np.ones((3,20)), 'canonical.frame_valid':np.array([1,1,0])}
    cols['observation.states.eef.mask'][0,:9]=0
    merge_source_masks(rep,cols,a,s)
    assert not a[:,:9].any() and a[0,20:34].all() and not a[1].any()


def test_byte_lru_enforces_bytes_and_entries():
    from openwam.dataloader.native_stage2_io import ByteLRU
    c=ByteLRU(2,100);c.put('a',{'x':np.zeros(8)},64);c.put('b',{'x':np.zeros(8)},64)
    assert 'a' not in c and len(c)==1 and c.bytes==64
    with pytest.raises(ValueError,match='cache_byte_budget'):c.put('c',{},101)


def make_native_source(tmp_path, *, subtasks=True):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import yaml
    root=tmp_path/'source';repo=root/'interndata_a1/unit';(repo/'meta/episodes/chunk-000').mkdir(parents=True);(repo/'data').mkdir()
    i=robot_info();i.update(fps=30,data_path='data/episode-{episode_index}.parquet',video_path='videos/{video_key}/{file_index}.mp4')
    i['features']['task_index']={'shape':[]};i['features']['frame_index']={'shape':[]}
    i['features']['master_actions.left_gripper.openness']={'shape':[]}
    (repo/'meta/info.json').write_text(json.dumps(i))
    ep=dict(episode_index=0,length=24,**{'data/chunk_index':0,'data/file_index':0})
    pq.write_table(pa.Table.from_pylist([ep]),repo/'meta/episodes/chunk-000/file-000.parquet')
    pq.write_table(pa.table({'task_index':[0],'task':['move']}),repo/'meta/tasks.parquet')
    data={'episode_index':[0]*24,'frame_index':list(range(24)),'task_index':[0]*24,
        'observation.states.joint':[[t*.01]*6+[.2]+[t*.02]*6+[.3] for t in range(24)],
        'states.left_tcp_to_robot_pose':[[t*.01,0,0,1,0,0,0] for t in range(24)],
        'states.robot_to_env_pose':[[0,0,0,1,0,0,0]]*24,
        'master_actions.left_gripper.openness':[.4]*24}
    if subtasks:
        data['subtask_index']=[1]*6+[2]*6+[1]*6+[-1]*6
        pq.write_table(pa.table({'episode_index':[0,0],'subtask_index':[1,2],'subtask':['pick','place']}),repo/'meta/sub_tasks.parquet')
    pq.write_table(pa.table(data),repo/'data/episode-0.parquet')
    selection=tmp_path/'selection.yaml';selection.write_text(yaml.safe_dump(dict(data_root_dir=str(root),resolver=dict(segments_by_repo={'interndata_a1/unit':[[0,2,5],[0,12,22]]},repo_weights={'interndata_a1/unit':1}))))
    return selection


def test_dataset_native_short_windows_clean_head_and_missing_views(tmp_path,monkeypatch):
    from openwam.dataloader.curated_stage2 import CuratedStage2Dataset
    from openwam.dataloader.native_stage2_stats import prepare_native_stats
    from PIL import Image
    selection=make_native_source(tmp_path)
    d=CuratedStage2Dataset(selection,seed=4)
    registry=prepare_native_stats(d,tmp_path/'stats',max_episodes_per_profile=1,max_windows_per_profile=16)
    d=CuratedStage2Dataset(selection,seed=4,normalization_registry=registry,observation_augmentation={'color_jitter':{'p':1,'brightness':.4}})
    seen=[]
    def decode(m,indices):
        seen.extend(indices);return [{'observation.images.cam_high':Image.new('RGB',(40,32),(100,120,140))} for _ in indices],{}
    monkeypatch.setattr(d,'_decode_views',decode)
    a=d.sample_at(3);b=d.sample_at(3)
    assert a['trace']['identity']==b['trace']['identity']
    assert a['goal_observed_head'].shape==(3,1,256,320) and a['goal_head_valid']
    assert a['action'].shape==(32,112) and a['video_view_mask'].shape==(9,3)
    assert not a['video_view_mask'][:,1:].any()
    last=a['trace']['last_source'];anchor=a['trace']['anchor']
    assert max(seen)<=last
    assert a['action_mask'][last-anchor:].sum()==0
    assert a['video_mask'].sum()==len(range(anchor,last+1,4))
    assert a['action_mask'][:last-anchor,80:86].all()
    assert not a['action_mask'][:,43].any(), 'missing right command stays masked'
    assert a['prompt']==a['instruction']
    assert a['report_target']['goal_source_frame']>=anchor


def test_dataset_selection_split_and_stats_provenance(tmp_path):
    from openwam.dataloader.curated_stage2 import CuratedStage2Dataset
    from openwam.dataloader.native_stage2_stats import prepare_native_stats
    selection=make_native_source(tmp_path,subtasks=False)
    with pytest.raises(ValueError,match='evaluation_source_yaml'):
        CuratedStage2Dataset.from_config({'source_yaml':str(selection)},split='val')
    d=CuratedStage2Dataset(selection)
    with pytest.raises(ValueError,match='stats_entry_missing'):d.sample_at(0)
    registry=prepare_native_stats(d,tmp_path/'stats',max_episodes_per_profile=1,max_windows_per_profile=4)
    selection.write_text(selection.read_text()+'\n# changed selection identity\n')
    with pytest.raises(ValueError,match='selection'):
        CuratedStage2Dataset(selection,normalization_registry=registry)


def test_stats_ignore_padding_and_reject_active_unseen_dimensions():
    from openwam.dataloader.native_stage2_stats import summarize, normalize
    x=np.zeros((3,52),np.float32);m=np.zeros_like(x,dtype=bool)
    x[:2,20]=[2,4];m[:2,20]=True
    stats=summarize([(x,m)])
    assert stats['count'][20]==2 and stats['min'][20]==2 and stats['max'][20]==4
    m[0,21]=True
    with pytest.raises(ValueError,match='insufficient_valid_population'):
        normalize(x,m,stats,source='repo:joint')


def test_absent_robot_joints_accept_empty_per_side_mapping():
    rep=MoraHeadTCPRepresentation({'features':{}},joint_indices={'left':[],'right':[]})
    action,mask,state,sm=rep.encode_window({'_rows':np.arange(3)})
    assert not mask.any() and action.shape==(2,52)


def test_video_offset_must_match_previous_episode_lengths():
    from openwam.dataloader.native_stage2_io import video_frame_base
    cam='head';rows=[{'episode_index':0,'length':10,'dataset_from_index':0},{'episode_index':1,'length':20,'dataset_from_index':10}]
    ep={**rows[1],'videos/head/from_timestamp':10/30}
    assert video_frame_base(ep,rows,cam,30)==10
    ep['videos/head/from_timestamp']=11/30
    with pytest.raises(ValueError,match='video_offset'):video_frame_base(ep,rows,cam,30)


def test_sampling_balances_language_and_excludes_spike_windows(tmp_path):
    from openwam.dataloader.curated_stage2 import CuratedStage2Dataset
    d=CuratedStage2Dataset(make_native_source(tmp_path));repo=d.repos[0];m=d._episode(repo,0)
    levels=[d._draw(i)[-1] for i in range(600)]
    assert .4<levels.count('subtask')/len(levels)<.6
    m['bad'][3]=True
    for i in range(40):
        _,_,_,o,a,_=d._draw(i)
        assert not (a<=3<=min(a+32,o['last_source']))
    d._cache.clear();d._cache.bytes=0
    assert d._draw(111)[4:]==d._draw(111)[4:]


def test_abc_head_pose_uses_per_episode_calibration_eligibility(tmp_path):
    from openwam.dataloader.native_stage2_io import calibration_eligibility
    root=tmp_path;'unused'
    (root/'meta').mkdir();p=root/'meta/camera_geometry_eligibility.jsonl'
    p.write_text(json.dumps({'episode_index':2,'broadcast_qa_status':'passed'})+'\n'+json.dumps({'episode_index':3,'broadcast_qa_status':'failed'})+'\n')
    assert calibration_eligibility(root,'abc_130k',2)[0]
    assert not calibration_eligibility(root,'abc_130k',3)[0]
    assert not calibration_eligibility(root,'abc_130k',4)[0]


def test_from_config_rejects_training_without_stats(tmp_path):
    from openwam.dataloader.curated_stage2 import CuratedStage2Dataset
    selection=make_native_source(tmp_path)
    with pytest.raises(ValueError,match='stats_entry_missing'):
        CuratedStage2Dataset.from_config({'source_yaml':str(selection)})


def test_preflight_composes_hydra_defaults(tmp_path):
    from scripts.data.preflight_stage2 import load_config
    root=tmp_path/'configs';(root/'goal').mkdir(parents=True)
    (root/'base.yaml').write_text('# @package _global_\ndataloader:\n  type: curated_stage2\n  seed: 9\n')
    (root/'goal/run.yaml').write_text('# @package _global_\ndefaults:\n  - /base\n  - _self_\ndataloader:\n  source_yaml: selected.yaml\n')
    cfg=load_config(root/'goal/run.yaml')
    assert cfg['dataloader']['type']=='curated_stage2' and cfg['dataloader']['source_yaml']=='selected.yaml'


def test_invalid_observation_and_no_valid_future_are_excluded(tmp_path):
    from openwam.dataloader.curated_stage2 import CuratedStage2Dataset
    d=CuratedStage2Dataset(make_native_source(tmp_path));repo=d.repos[0];m=d._episode(repo,0)
    flags=np.zeros(24);flags[[3,4,18,21]]=1;m['columns']['canonical.frame_valid']=flags
    m['columns']['training.valid']=np.ones(24)
    options=d._options(m,repo['ranges'],'task')
    assert [(o['start'],o['end_exclusive']) for o in options]==[(3,4),(18,19)]
    for draw in range(50):assert d._draw(draw)[4] in {3,18}


def test_future_validity_is_bounded_by_horizon_not_only_interval(tmp_path):
    from openwam.dataloader.curated_stage2 import CuratedStage2Dataset
    d=CuratedStage2Dataset(make_native_source(tmp_path))
    flags=np.zeros(100);flags[[0,80,81]]=1
    m=dict(length=100,tasks=[dict(start=0,end_exclusive=100,text='move')],subtasks=[],bad=np.zeros(100,bool),columns={'training.valid':flags})
    options=d._options(m,[(0,0,100)],'task')
    assert [(o['start'],o['end_exclusive']) for o in options]==[(80,81)]


def test_global_validity_rejects_nonfinite_flags_in_sampling_and_masks(tmp_path):
    from openwam.dataloader.curated_stage2 import CuratedStage2Dataset
    from openwam.dataloader.native_stage2_io import merge_source_masks
    d=CuratedStage2Dataset(make_native_source(tmp_path))
    flags=np.array([1,np.inf,1,np.nan,0])
    m=dict(length=5,tasks=[dict(start=0,end_exclusive=5,text='move')],subtasks=[],bad=np.zeros(5,bool),columns={'training.valid':flags})
    options=d._options(m,[(0,0,5)],'task')
    assert [(o['start'],o['end_exclusive']) for o in options]==[(0,1)]
    action=np.ones((4,52),bool);state=np.ones((1,52),bool)
    validity=merge_source_masks(MoraHeadTCPRepresentation({'features':{}}),m['columns'],action,state)
    assert validity.tolist()==[True,False,True,False,False]
    assert action.any(1).tolist()==[False,True,False,False]


def test_native_action_mask_inverts_through_actual_prepare_inputs_and_loss(tmp_path,monkeypatch):
    import torch
    from types import SimpleNamespace
    from PIL import Image
    from openwam.dataloader.curated_stage2 import CuratedStage2Dataset
    from openwam.dataloader.native_stage2_stats import prepare_native_stats
    from tests.goal.test_patch_stage2_memory import make_arch
    selection=make_native_source(tmp_path)
    prep=CuratedStage2Dataset(selection)
    registry=prepare_native_stats(prep,tmp_path/'stats',max_episodes_per_profile=1,max_windows_per_profile=8)
    dataset=CuratedStage2Dataset(selection,normalization_registry=registry)
    monkeypatch.setattr(dataset,'_decode_views',lambda m,indices:([{'observation.images.cam_high':Image.new('RGB',(40,32))} for _ in indices],{}))
    sample=dataset.sample_at(4)
    arch=make_arch();arch.set_training_runtime()
    arch._pipeline_transform_instance=SimpleNamespace(apply=lambda value:value)
    arch.preprocess=lambda **kwargs:dict(input_latents=torch.zeros(1,48,3,24,20),first_frame_latents=torch.zeros(1,48,1,24,20))
    inputs=arch.prepare_inputs([sample])
    assert torch.equal(inputs['action_is_pad'][0],~sample['action_mask'])
    assert inputs['action_is_pad'].shape==(1,32,112)
    pred=torch.ones(1,32,112,requires_grad=True)
    target=torch.zeros_like(pred);target[inputs['action_is_pad']]=1000
    scheduler=SimpleNamespace(training_weight=lambda ids:torch.ones(len(ids)))
    loss=arch._compute_action_loss(pred,target,torch.zeros(1,dtype=torch.long),scheduler,inputs,'cpu')
    torch.testing.assert_close(loss,torch.tensor(1.));loss.backward()
    assert pred.grad[inputs['action_is_pad']].count_nonzero()==0
    assert pred.grad[~inputs['action_is_pad']].abs().sum()>0


@pytest.mark.parametrize('invalid', [0., float('nan'), float('inf')])
def test_stored_eef_pose_masks_share_head_stats_and_loss(invalid):
    import torch
    from types import SimpleNamespace
    from openwam.dataloader.native_stage2_io import merge_source_masks
    from openwam.dataloader.native_stage2_stats import summarize
    from tests.goal.test_patch_stage2_memory import make_arch
    rep=MoraHeadTCPRepresentation({'features':{}})
    flags=np.ones((3,20));flags[0,5]=invalid;flags[1,13]=invalid
    mask=np.ones((2,52),bool);state=np.ones((1,52),bool)
    merge_source_masks(rep,{'actions.eef.mask':flags},mask,state)
    assert not mask[0,:9].any() and not mask[0,34:43].any()
    assert not mask[1,10:19].any() and not mask[1,43:52].any()
    assert mask[:,20:34].all() and mask[:,[9,19]].all() and state.all()
    assert mask[1,:9].all() and mask[0,43:52].all()
    counts=np.asarray(summarize([(np.ones((2,52)),mask)])['count'])
    assert (counts[34:52]==1).all() and (counts[20:34]==2).all()
    pred=torch.ones(1,2,52,requires_grad=True);pad=torch.from_numpy(~mask)[None]
    target=torch.zeros_like(pred);target[pad]=1000
    arch=make_arch();scheduler=SimpleNamespace(training_weight=lambda ids:torch.ones(len(ids)))
    loss=arch._compute_action_loss(pred,target,torch.zeros(1,dtype=torch.long),scheduler,{'action_is_pad':pad},'cpu')
    torch.testing.assert_close(loss,torch.tensor(1.));loss.backward()
    assert pred.grad[pad].count_nonzero()==0 and pred.grad[~pad].abs().sum()>0


@pytest.mark.parametrize('case,reason', [('missing','missing_episode_metadata'),('duplicate','nonunique_episode_metadata'),('bounds','invalid_segment_bounds'),('length','invalid_episode_length')])
def test_metadata_audit_checks_every_selected_episode_range(tmp_path,case,reason):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from openwam.dataloader.curated_stage2 import CuratedStage2Dataset
    from openwam.dataloader.native_stage2_stats import audit_selection,prepare_native_stats
    d=CuratedStage2Dataset(make_native_source(tmp_path));repo=d.repos[0]
    path=Path(repo['path'])/'meta/episodes/chunk-000/file-000.parquet'
    if case=='missing':repo['ranges']=[(999,0,1000000000)]
    elif case=='duplicate':pq.write_table(pq.read_table(path),path.with_name('file-001.parquet'))
    elif case=='bounds':repo['ranges']=[(0,2,5),(0,12,25)]
    else:
        row=pq.read_table(path).to_pylist()[0];row['length']=-1;pq.write_table(pa.Table.from_pylist([row]),path)
    audit=audit_selection(d)
    assert audit['errors'], case
    assert all(error['repo']==repo['repo'] for error in audit['errors'])
    assert any(reason in error['error'] and 'episode' in error and 'range' in error for error in audit['errors'])
    with pytest.raises(ValueError,match='metadata preflight failed'):
        prepare_native_stats(d,tmp_path/'stats',audit=audit)


def test_metadata_audit_streams_only_identity_length_and_counts_disjoint_ranges(tmp_path,monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from openwam.dataloader.curated_stage2 import CuratedStage2Dataset
    from openwam.dataloader.native_stage2_stats import audit_selection
    d=CuratedStage2Dataset(make_native_source(tmp_path));repo=d.repos[0]
    path=Path(repo['path'])/'meta/episodes/chunk-000/file-000.parquet'
    row=pq.read_table(path).to_pylist()[0];row['stats/unneeded']=[1.]*10000
    pq.write_table(pa.Table.from_pylist([row]),path)
    original=pq.ParquetFile;calls=[]
    class ProjectedFile:
        def __init__(self,*args,**kwargs):self.wrapped=original(*args,**kwargs)
        def __getattr__(self,key):return getattr(self.wrapped,key)
        def iter_batches(self,**kwargs):
            calls.append(kwargs);return self.wrapped.iter_batches(**kwargs)
    monkeypatch.setattr(pq,'ParquetFile',ProjectedFile)
    audit=audit_selection(d)
    assert not audit['errors'] and audit['selected_episodes_checked']==1 and audit['selected_ranges_checked']==2
    assert calls and all(set(c['columns'])=={'episode_index','length'} and c['batch_size']<=4096 for c in calls)
    metadata=audit['sources'][repo['repo']]['episode_metadata_audit']
    assert metadata['selected_episodes']==1 and metadata['selected_ranges']==2 and len(metadata['sha256'])==64
    row['length']=25;pq.write_table(pa.Table.from_pylist([row]),path)
    changed=audit_selection(d)
    assert changed['sources'][repo['repo']]['episode_metadata_audit']['sha256']!=metadata['sha256']


def test_previous_pose_mask_policy_registry_is_rejected(tmp_path):
    import hashlib
    from openwam.dataloader.curated_stage2 import CuratedStage2Dataset
    selection=make_native_source(tmp_path);dataset=CuratedStage2Dataset(selection)
    contract=dataset.stats_contract()
    assert contract.pop('source_pose_mask_policy')=='shared_local_head_all9_finite_gt_0.5_v1'
    raw=json.dumps({'contract':contract,'sources':{},'entries':{}}).encode()
    path=tmp_path/('registry-'+hashlib.sha256(raw).hexdigest()+'.json');path.write_bytes(raw)
    with pytest.raises(ValueError,match='selection/contract fingerprint_mismatch'):
        CuratedStage2Dataset(selection,normalization_registry=path)


def add_eef_native_fields(selection, *, mask_key, stored_width=10, invalid=0.):
    """Real projected parquet fixture: left EEF10 plus independently sourced right TCP/joints."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import yaml
    repo=Path(yaml.safe_load(selection.read_text())['data_root_dir'])/'interndata_a1/unit'
    info_path=repo/'meta/info.json';info=json.loads(info_path.read_text())
    info['features']['observation.states.eef']={'shape':[10],'gripper_unit':'openness_0_closed_1_open'}
    info['features']['states.right_tcp_to_robot_pose']={'shape':[7]}
    width=14 if mask_key=='actions.joint.mask' else stored_width
    info['features'][mask_key]={'shape':[width]}
    info_path.write_text(json.dumps(info))
    path=repo/'data/episode-0.parquet';data=pq.read_table(path).to_pydict()
    data['observation.states.eef']=[[0,0,0,1,0,0,0,1,0,.2]]*24
    data['states.right_tcp_to_robot_pose']=[[t*.02,0,0,1,0,0,0] for t in range(24)]
    flags=np.ones((24,width));flags[:,0]=invalid;data[mask_key]=flags.tolist()
    pq.write_table(pa.table(data),path)


@pytest.mark.parametrize('mask_key', ['observation.states.eef.mask','actions.eef.mask'])
def test_eef10_masks_through_native_stats_and_sample_preserve_other_arm(tmp_path,monkeypatch,mask_key):
    from PIL import Image
    from openwam.dataloader.curated_stage2 import CuratedStage2Dataset
    from openwam.dataloader.native_stage2_stats import prepare_native_stats
    selection=make_native_source(tmp_path);add_eef_native_fields(selection,mask_key=mask_key)
    dataset=CuratedStage2Dataset(selection)
    registry=prepare_native_stats(dataset,tmp_path/'stats',max_episodes_per_profile=1,max_windows_per_profile=8)
    entry=next(iter(json.loads(Path(registry).read_text())['entries'].values()))
    assert not any(entry['action']['count'][:9]) and all(entry['action']['count'][10:19])
    assert all(entry['action']['count'][20:26]) and all(entry['action']['count'][27:33])
    if mask_key.startswith('observation'):
        assert entry['state']['count'][0]==0 and all(entry['state']['count'][10:19])
    dataset=CuratedStage2Dataset(selection,normalization_registry=registry)
    monkeypatch.setattr(dataset,'_decode_views',lambda m,indices:([{'observation.images.cam_high':Image.new('RGB',(40,32))} for _ in indices],{}))
    item=dataset.sample_at(4);valid=item['action_mask']
    assert not valid[:,:9].any() and valid[:,34:43].any()
    assert valid[:,80:86].any() and valid[:,87:93].any()


@pytest.mark.parametrize('mask_key', ['observation.states.eef.mask','actions.eef.mask'])
@pytest.mark.parametrize('width',[9,20])
def test_eef10_declared_schema_rejects_malformed_stored_width(tmp_path,mask_key,width):
    from openwam.dataloader.curated_stage2 import CuratedStage2Dataset
    selection=make_native_source(tmp_path);add_eef_native_fields(selection,mask_key=mask_key,stored_width=width)
    d=CuratedStage2Dataset(selection);repo,ep,m,option,anchor,level=d._draw(0)
    with pytest.raises(ValueError,match=mask_key+r'.*EEF10.*width'):
        d.raw_window(m,option,anchor)


@pytest.mark.parametrize('invalid',[0.,float('nan'),float('inf')])
@pytest.mark.parametrize('key',['observation.states.eef.mask','actions.joint.mask','source.camera.world_to_camera_valid','observation.states.camera.valid'])
def test_all_stored_validity_flags_are_finite_thresholded(key,invalid):
    from openwam.dataloader.native_stage2_io import merge_source_masks
    from openwam.dataloader.native_stage2_stats import summarize
    rep=MoraHeadTCPRepresentation(robot_info())
    a=np.ones((2,52),bool);s=np.ones((1,52),bool)
    if key=='observation.states.eef.mask':
        flags=np.ones((3,20));flags[:,0]=invalid
    elif key=='actions.joint.mask':
        flags=np.ones((3,14));flags[:,0]=invalid
    else:flags=np.full(3,invalid)
    merge_source_masks(rep,{key:flags},a,s)
    counts=summarize([(np.ones((2,52)),a)])['count']
    if key=='observation.states.eef.mask':
        assert not a[:,:9].any() and not a[:,34:43].any() and not s[:,0].any()
        assert counts[0]==counts[34]==0 and a[:,10:34].all()
    elif key=='actions.joint.mask':
        assert not a[:,20].any() and counts[20]==0 and a[:,21:34].all() and s.all()
    else:
        assert not a[:,34:].any() and not s[:,34:].any() and a[:,:34].all()
        assert counts[34:]==[0]*18


def test_prefixed_mask_implementation_registry_requires_repreparation(tmp_path):
    from openwam.dataloader.curated_stage2 import CuratedStage2Dataset
    from openwam.dataloader.native_stage2_stats import prepare_native_stats
    import hashlib
    selection=make_native_source(tmp_path);d=CuratedStage2Dataset(selection)
    registry=prepare_native_stats(d,tmp_path/'stats',max_episodes_per_profile=1,max_windows_per_profile=4)
    payload=json.loads(Path(registry).read_text())
    assert payload['contract']['stored_mask_policy']=='declared_eef_blocks_all_flags_finite_v2'
    del payload['contract']['stored_mask_policy']
    raw=json.dumps(payload,sort_keys=True,separators=(',',':')).encode()
    old=tmp_path/('registry-'+hashlib.sha256(raw).hexdigest()+'.json');old.write_bytes(raw)
    with pytest.raises(ValueError,match='fingerprint_mismatch'):CuratedStage2Dataset(selection,normalization_registry=old)
