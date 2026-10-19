"""Native current-camera alignment and Stage1 dispatch; bounded CPU diagnostics."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch
import yaml

from openwam.dataloader.curated_goal import CuratedGoalDataset
from openwam.train.goal_curated import load_config, prepare_batch, train_curated
from openwam.train.goal_patch import make_smoke_data


def native_fixture(tmp_path, wrists=('left', 'right')):
    import pyarrow as pa
    import pyarrow.parquet as pq
    root = tmp_path/'LightWheel/fixture'
    (root/'meta/episodes/chunk-000').mkdir(parents=True)
    cameras = ['observation.images.head_left'] + ['observation.images.cam_'+side+'_wrist' for side in wrists]
    info = dict(fps=10, features={camera: dict(dtype='video', shape=[24,32,3]) for camera in cameras},
                data_path='data.parquet', video_path='shared.mp4')
    # Joint semantics are deliberately unavailable: camera reading must not
    # trigger Stage2 robot joint validation.
    info['features']['observation.states.joint'] = dict(dtype='float32', shape=[3], unit='unknown')
    (root/'meta/info.json').write_text(json.dumps(info))
    ep = dict(episode_index=0, length=10, **{'data/chunk_index':0, 'data/file_index':0})
    for camera, offset in zip(cameras, (.2,.4,.6)):
        ep.update({f'videos/{camera}/chunk_index':0, f'videos/{camera}/file_index':0,
                   f'videos/{camera}/from_timestamp':offset})
    pq.write_table(pa.Table.from_pylist([ep]),root/'meta/episodes/chunk-000/file-000.parquet')
    pq.write_table(pa.Table.from_pylist([dict(episode_index=0,frame_index=i,task_index=0 if i<8 else 1) for i in range(10)]), root/'data.parquet')
    pq.write_table(pa.Table.from_pylist([dict(task_index=0,task='first'),dict(task_index=1,task='second')]),root/'meta/tasks.parquet')
    config=tmp_path/'source.yaml'
    config.write_text(yaml.safe_dump(dict(data_root_dir=str(tmp_path), selection=dict(
        segments_by_repo={'LightWheel/fixture':[[0,1,2]]}, repo_weights={'LightWheel/fixture':1}))))
    av=pytest.importorskip('av')
    with av.open(str(root/'shared.mp4'),'w') as container:
        stream=container.add_stream('mpeg4',rate=10)
        stream.width=32; stream.height=24; stream.pix_fmt='yuv420p'
        for i in range(20):
            frame=av.VideoFrame.from_ndarray(np.full((24,32,3),i*10,np.uint8),format='rgb24')
            for packet in stream.encode(frame):container.mux(packet)
        for packet in stream.encode():container.mux(packet)
    return config,root


def test_native_multiview_shared_file_offsets_decode_only_current_wrists(tmp_path,monkeypatch):
    from scripts.data import prepare_subtask_goals as decoder
    config,_=native_fixture(tmp_path)
    calls=[]; decode=decoder.decode_frames
    def trace(path,offset,fps,indices,**kwargs):
        calls.append((offset,list(indices)))
        return decode(path,offset,fps,indices,**kwargs)
    monkeypatch.setattr(decoder,'decode_frames',trace)
    head=CuratedGoalDataset(config,tmp_path/'head').sample(0)
    calls.clear()
    record,batch=CuratedGoalDataset(config,tmp_path/'multi',observation_mode='multiview').sample(0)
    assert calls==[(.2,[1,7]),(.4,[1]),(.6,[1])]
    assert batch['observed'].shape==(1,3,1,384,320)
    assert batch['targets'].image.shape==(1,3,1,256,320)
    assert batch['observed_view_valid'].tolist()==[[True,True,True]]
    assert record['view_valid']==[True,True,True]
    for name,offset in zip(('head','left_wrist','right_wrist'),(2,4,6)):
        view=record['provenance']['observed_views'][name]
        assert view['alignment']['source_frame_index']==1
        assert view['alignment']['video_ordinal']==1+offset
        assert view['alignment']['offset_frames']==offset
        assert view['available'] and view['mask_reason'] is None
    assert record['goal_source_frame']==7
    assert record['provenance']['endpoint']==head[0]['provenance']['endpoint']
    assert record['provenance']['camera']==head[0]['provenance']['camera']
    assert torch.equal(batch['observed'][...,:256,:],head[1]['observed'])
    assert torch.equal(batch['targets'].image,head[1]['targets'].image)
    assert torch.equal(batch['targets'].uv,head[1]['targets'].uv)
    # Each wrist respects its own source offset in a shared video file.
    assert batch['observed'][...,256:,:160].float().mean()<batch['observed'][...,256:,160:].float().mean()


def test_missing_wrist_mask_and_actual_bf16_export(tmp_path):
    from openwam.train.goal_curated import preflight
    config,_=native_fixture(tmp_path,wrists=('left',))
    dataset=CuratedGoalDataset(config,tmp_path/'cache',observation_mode='multiview')
    record,batch=dataset.sample(0)
    assert record['view_valid']==[True,True,False]
    assert batch['observed_view_valid'].dtype==torch.bool
    assert (batch['observed'][...,256:,160:]==-1).all()
    assert record['provenance']['observed_views']['right_wrist']['mask_reason']=='missing_camera_feature'
    preflight(dataset,tmp_path/'audit')
    with np.load(tmp_path/'audit/inputs/case_0000.npz') as data:
        assert data['observed_view_valid'].tolist()==[[True,True,False]]
        assert data['observation_token_valid'].shape==(1,120)
        assert not data['observation_token_valid'][0].reshape(12,10)[8:,5:].any()
        saved=torch.from_numpy(data['observed_bfloat16_bits']).view(torch.bfloat16)
        assert saved.shape==(1,3,1,384,320)
        assert torch.equal(saved,dataset.evaluation_samples()[0][1]['observed'])
    html=(tmp_path/'audit/inputs/index.html').read_text()
    assert 'observed_view_valid' in html and 'missing_camera_feature' in html
    assert Image.open(tmp_path/'audit/inputs/case_0000_observed.png').size==(320,384)


@pytest.mark.parametrize('corruption',['dtype','offset_missing','offset_nan','offset_nonintegral','video_missing'])
def test_present_bad_wrist_fails_instead_of_becoming_missing(tmp_path,corruption):
    import pyarrow as pa
    import pyarrow.parquet as pq
    config,root=native_fixture(tmp_path)
    camera='observation.images.cam_left_wrist'
    info=json.loads((root/'meta/info.json').read_text())
    if corruption=='dtype':info['features'][camera]['dtype']='float32'
    elif corruption=='video_missing':info['video_path']='missing.mp4'
    else:
        path=root/'meta/episodes/chunk-000/file-000.parquet'
        ep=pq.read_table(path).to_pylist()[0]
        key=f'videos/{camera}/from_timestamp'
        if corruption=='offset_missing':del ep[key]
        else:ep[key]=float('nan') if corruption=='offset_nan' else .45
        pq.write_table(pa.Table.from_pylist([ep]),path)
    (root/'meta/info.json').write_text(json.dumps(info))
    with pytest.raises((ValueError,FileNotFoundError),match='camera|video|offset|timestamp'):
        CuratedGoalDataset(config,tmp_path/'cache',observation_mode='multiview').sample(0)


def test_presets_inherit_source_scope_and_enforce_mode_agreement(tmp_path):
    root=Path(__file__).resolve().parents[2]
    for suffix in ('sample','3000h'):
        multi=load_config(root/f'configs/goal/stage1_multiview_{suffix}.yaml')
        head=load_config(root/f'configs/goal/stage1_joint_{suffix}.yaml')
        assert multi['observation_mode']==multi['dataset']['observation_mode']=='multiview'
        assert multi['dataset']['source_yaml']==head['dataset']['source_yaml']
        assert multi['max_steps']==head['max_steps']
        assert multi['dataset']['cache_dir']!=head['dataset']['cache_dir']
    bad=tmp_path/'bad.yaml'
    bad.write_text('observation_mode: multiview\ndataset:\n  observation_mode: head\n')
    with pytest.raises(ValueError,match='observation_mode'):load_config(bad)


def test_view_mask_device_transfer_and_real_wan_streaming_export(tmp_path):
    expert,batches,records=make_smoke_data(observation_mode='multiview')
    batch=batches[0];record=records[0]
    moved=prepare_batch(record,batch,device='meta',dtype=torch.bfloat16)
    assert moved['observed_view_valid'].device.type=='meta'
    assert moved['observed_view_valid'].dtype==torch.bool
    assert batch['observed_view_valid'].device.type=='cpu'
    class Dataset:
        def sample(self,step):return record,batch
        def evaluation_samples(self):return [(record,batch)]
        def scope_summary(self):return dict(synthetic=True)
    before=next(expert.dit.parameters()).detach().clone()
    result=train_curated(expert,dataset=Dataset(),context_cache=None,output_dir=tmp_path/'run',
                         steps=1,eval_every=1,sampling_steps=1,learning_rate=.001,precomputed_text_fingerprint='a'*64)
    assert result['observation_mode']=='multiview'
    assert not torch.equal(before,next(expert.dit.parameters()))
    assert [x['step'] for x in result['evaluations']]==[0,1]
    for step in (0,1):
        metrics=json.loads((tmp_path/f'run/eval_step_{step:06d}/metrics.json').read_text())
        assert metrics['review'][0]['observed_view_valid']==[[True,True,False]]
        assert metrics['review'][0]['observed_shape']==[1,3,1,384,320]
    assert (tmp_path/'run/eval_step_000000/index.html').exists()


def test_checked_v1_initialization_preserves_source_and_actual_modes(tmp_path):
    from openwam.train.goal_curated import initialize_goal_checkpoint
    from openwam.train.goal_patch import save_patch_bundle
    from test_expert import TinyVAE
    from test_patch_expert import make_expert
    head=make_expert()
    path=tmp_path/'v1.safetensors'
    save_patch_bundle(head,path,precomputed_text_fingerprint='a'*64)
    kwargs=dict(vae=TinyVAE(),dit_config=head.dit_config,observation_mode='multiview',
                text_dim=16,precomputed_text_fingerprint='a'*64)
    initialized=initialize_goal_checkpoint(path,**kwargs)
    assert initialized.observation_mode=='multiview'
    assert initialized.bundle_metadata['format']=='openwam.goal.patch.v1'
    assert initialized.initialization_metadata['trained_multiview'] is False
    assert initialized.initialization_metadata['source_checkpoint_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    assert all(torch.equal(v,initialized.state_dict()[k]) for k,v in head.state_dict().items())
    assert initialized.use_gradient_checkpointing
    with pytest.raises(ValueError,match='text'):
        initialize_goal_checkpoint(path,**{**kwargs,'precomputed_text_fingerprint':'b'*64})
    with pytest.raises(ValueError,match='architecture'):
        initialize_goal_checkpoint(path,**{**kwargs,'dit_config':{**head.dit_config,'dim':96}})
    with pytest.raises(ValueError,match='multiview'):
        initialize_goal_checkpoint(path,**{**kwargs,'observation_mode':'head'})


def test_initialization_rebuilds_configured_training_grid_and_roundtrips(tmp_path,monkeypatch):
    from openwam.train import goal_patch
    from openwam.train.goal_curated import initialize_goal_checkpoint
    from test_expert import TinyVAE
    from test_patch_expert import make_expert
    from test_multiview_patch import multiview_batch
    head=make_expert(shift=5.)
    path=tmp_path/'v1.safetensors'
    goal_patch.save_patch_bundle(head,path,precomputed_text_fingerprint='a'*64)
    loaded=[]
    load=goal_patch.load_patch_bundle
    def capture(*args,**kwargs):
        expert=load(*args,**kwargs);loaded.append(expert);return expert
    monkeypatch.setattr(goal_patch,'load_patch_bundle',capture)
    initialized=initialize_goal_checkpoint(path,vae=TinyVAE(),dit_config=head.dit_config,
        observation_mode='multiview',text_dim=16,precomputed_text_fingerprint='a'*64,
        shift=8.,lambda_image=.25,lambda_tcp=2.,use_gradient_checkpointing=False)
    assert initialized.dit is loaded[0].dit and initialized.codec is loaded[0].codec
    assert initialized.bundle_metadata['shift']==5.
    assert initialized.initialization_metadata['source_format']=='openwam.goal.patch.v1'
    assert not initialized.training and not any(p.requires_grad for p in initialized.parameters())
    reference=make_expert(observation_mode='multiview',shift=8.,lambda_image=.25,lambda_tcp=2.)
    reference.load_state_dict(head.state_dict())
    for name in ('sigmas','timesteps','linear_timesteps_weights'):
        assert torch.equal(getattr(initialized.scheduler,name),getattr(reference.scheduler,name))
    assert initialized.scheduler.linear_timesteps_weights.mean().item()==pytest.approx(1.)
    obs,ctx,targets,valid=multiview_batch(1)
    kwargs=dict(observed_view_valid=valid,timestep_indices=torch.tensor([700]),
                noise=torch.ones(1,48,1,16,20),tcp_noise=torch.ones(1,192))
    initialized.train_stage_a();reference.train_stage_a()
    actual=initialized(obs,ctx,targets,**kwargs)
    expected=reference(obs,ctx,targets,**kwargs)
    assert torch.equal(actual['loss'],expected['loss'])
    assert torch.equal(actual['sigma'],reference.scheduler.sigmas[kwargs['timestep_indices']])
    actual['loss'].backward()
    assert torch.isfinite(initialized.dit.patch_embedding.weight.grad).all()
    initialized.eval();reference.eval()
    generated=initialized.generate(obs,ctx,observed_view_valid=valid,num_steps=2,seed=17)
    canonical=reference.generate(obs,ctx,observed_view_valid=valid,num_steps=2,seed=17)
    assert torch.equal(generated.latent,canonical.latent)
    assert torch.equal(generated.tcp_patch,canonical.tcp_patch)
    saved=tmp_path/'v2.safetensors'
    meta=goal_patch.save_patch_bundle(initialized,saved,precomputed_text_fingerprint='a'*64)
    assert (meta['shift'],meta['lambda_image'],meta['lambda_tcp'])==(8.,.25,2.)
    reloaded=load(saved,vae=TinyVAE(),text_dim=16,precomputed_text_fingerprint='a'*64)
    for name in ('sigmas','timesteps','linear_timesteps_weights'):
        assert torch.equal(getattr(initialized.scheduler,name),getattr(reloaded.scheduler,name))
    repeated=reloaded.generate(obs,ctx,observed_view_valid=valid,num_steps=2,seed=17)
    assert torch.equal(generated.latent,repeated.latent)
    assert torch.equal(generated.tcp_patch,repeated.tcp_patch)


def test_initialization_omitted_settings_preserve_nondefault_source(tmp_path):
    from openwam.train.goal_curated import initialize_goal_checkpoint
    from openwam.train.goal_patch import save_patch_bundle
    from test_expert import TinyVAE
    from test_patch_expert import make_expert
    source=make_expert(shift=3.,lambda_image=.75,lambda_tcp=2.)
    path=tmp_path/'v1.safetensors'
    save_patch_bundle(source,path,precomputed_text_fingerprint='a'*64)
    actual=initialize_goal_checkpoint(path,vae=TinyVAE(),dit_config=source.dit_config,
        observation_mode='multiview',text_dim=16,precomputed_text_fingerprint='a'*64)
    assert (actual.shift,actual.lambda_image,actual.lambda_tcp)==(3.,.75,2.)
    assert torch.equal(actual.scheduler.sigmas,source.scheduler.sigmas)
    assert torch.equal(actual.scheduler.linear_timesteps_weights,source.scheduler.linear_timesteps_weights)


@pytest.mark.parametrize('overrides',[
    {'shift':0.},{'shift':-1.},{'shift':float('nan')},{'shift':float('inf')},
    {'lambda_image':-1.},{'lambda_image':float('nan')},
    {'lambda_tcp':-1.},{'lambda_tcp':float('inf')},
])
def test_initialization_rejects_invalid_flow_settings(tmp_path,overrides):
    from openwam.train.goal_curated import initialize_goal_checkpoint
    from openwam.train.goal_patch import save_patch_bundle
    from test_expert import TinyVAE
    from test_patch_expert import make_expert
    source=make_expert()
    path=tmp_path/'v1.safetensors'
    save_patch_bundle(source,path,precomputed_text_fingerprint='a'*64)
    with pytest.raises(ValueError,match='shift/loss weights'):
        initialize_goal_checkpoint(path,vae=TinyVAE(),dit_config=source.dit_config,
            observation_mode='multiview',text_dim=16,precomputed_text_fingerprint='a'*64,**overrides)


def test_camera_resolver_matches_stage2_preference_without_joint_validation():
    from openwam.dataloader.native_cameras import resolve_native_cameras
    from openwam.dataloader.native_stage2_profiles import resolve_native_profile
    cameras=['observation.images.cam_high','observation.images.cam_left_wrist',
             'observation.images.camera_left','observation.images.cam_right']
    info=dict(features={c:dict(dtype='video',shape=[24,32,3]) for c in cameras})
    expected=[cameras[0],cameras[1],cameras[3]]
    assert resolve_native_cameras('Genrobot/example',info)==expected
    assert resolve_native_profile('Genrobot/example',info)['cameras']==expected
    info['features']['observation.states.joint']=dict(dtype='float32',shape=[3],unit='unknown')
    assert resolve_native_cameras('abc_130k/example',info)==expected


@pytest.mark.parametrize('configured_shift',[8.,None])
def test_cli_smoke_honors_multiview_preset_and_explicit_initialization(tmp_path,configured_shift):
    import subprocess
    import sys
    from openwam.train.goal_patch import save_patch_bundle
    expert,batches,_=make_smoke_data()
    if configured_shift is None:
        from openwam.model.goal.patch_expert import PatchGoalExpert
        expert=PatchGoalExpert(expert.dit,expert.codec,dit_config=expert.dit_config,shift=3.,lambda_tcp=2.)
    fingerprint=hashlib.sha256(batches[0]['context'].cpu().numpy().tobytes()).hexdigest()
    source=tmp_path/'head.safetensors'
    save_patch_bundle(expert,source,precomputed_text_fingerprint=fingerprint)
    output=tmp_path/'cli'
    root=Path(__file__).resolve().parents[2]
    config=tmp_path/'configured.yaml'
    settings=dict(base_config=str(root/'configs/goal/stage1_multiview_sample.yaml'))
    if configured_shift is not None:settings.update(shift=configured_shift,lambda_image=.25,lambda_tcp=2.)
    config.write_text(yaml.safe_dump(settings))
    command=[sys.executable,str(root/'scripts/train_goal_curated.py'),'--smoke','--device','cpu',
             '--config',str(config),
             '--steps','1','--output-dir',str(output),'--init-goal-checkpoint',str(source)]
    completed=subprocess.run(command,cwd=root,text=True,capture_output=True)
    assert completed.returncode==0, completed.stderr+'\n'+completed.stdout
    summary=json.loads((output/'summary.json').read_text())
    assert summary['observation_mode']=='multiview'
    assert summary['initialization']['source_format']=='openwam.goal.patch.v1'
    assert summary['initialization']['optimizer']=='fresh FP32-master AdamW'
    assert summary['real_wan_blocks'] and summary['synthetic_diagnostic']
    assert summary['completed_steps']==1
    meta=json.loads((output/'patch_goal.json').read_text())
    assert meta['format']=='openwam.goal.patch.v2'
    assert (meta['shift'],meta['lambda_image'],meta['lambda_tcp'])==((8.,.25,2.) if configured_shift is not None else (5.,.5,1.))
    assert meta['training']['observation_mode']=='multiview'
    assert meta['training']['completed_steps']==1
    assert meta['initialization']['trained_multiview'] is False
    assert (output/'eval_step_000000/index.html').exists()
    assert (output/'eval_step_000001/index.html').exists()
    with np.load(output/'inputs/case_0000.npz') as arrays:
        assert arrays['observed'].shape==(1,3,1,384,320)
        assert arrays['observed_view_valid'].tolist()==[[True,True,False]]


def test_present_malformed_wrist_shape_is_reported(tmp_path):
    config,root=native_fixture(tmp_path)
    info=json.loads((root/'meta/info.json').read_text())
    info['features']['observation.images.cam_left_wrist']['shape']=[24,32]
    (root/'meta/info.json').write_text(json.dumps(info))
    with pytest.raises(ValueError,match='camera.*shape'):
        CuratedGoalDataset(config,tmp_path/'cache',observation_mode='multiview').sample(0)


def test_preflight_visible_input_dimensions_and_view_validity(tmp_path):
    from openwam.train.goal_curated import preflight
    config,_=native_fixture(tmp_path,wrists=('left',))
    dataset=CuratedGoalDataset(config,tmp_path/'cache',observation_mode='multiview')
    preflight(dataset,tmp_path/'audit')
    for path in (tmp_path/'audit/index.html',tmp_path/'audit/inputs/index.html'):
        content=path.read_text()
        assert 'Current three-view input (H384 × W320)' in content
        assert 'Head-only endpoint GT (H256 × W320)' in content
        assert 'head: valid · left_wrist: valid · right_wrist: missing' in content
        assert 'Same anchor source frame: 1' in content
