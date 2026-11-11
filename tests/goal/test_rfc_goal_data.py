import json
import numpy as np
import pytest
import torch
from test_curated_goal import fixture
from openwam.dataloader.rfc_goal import RFCGoalDataset, collate_goal_memory, select_history_frames
from openwam.model.goal.memory_codec import GoalMemoryCodec

class VAE(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.weight=torch.nn.Parameter(torch.ones(())); self.calls=0
    def batch_encode(self, pixels, device=None):
        self.calls+=1
        return pixels.mean((1,2,3,4))[:,None,None,None,None].expand(-1,48,1,24,20)*self.weight

def source(tmp_path):
    import av
    config,root=fixture(tmp_path,annotations=True)
    with av.open(str(root/'video.mp4'),'w') as container:
        stream=container.add_stream('mpeg4',rate=10);stream.width=32;stream.height=24;stream.pix_fmt='yuv420p'
        for index in range(10):
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.full((24,32,3),index*20,dtype=np.uint8),format='rgb24')):container.mux(packet)
        for packet in stream.encode():container.mux(packet)
    return config,root

def test_grid():
    assert select_history_frames(100,30,2)==[0,60,90]
    assert select_history_frames(60,30,2)==[0,30]
    assert select_history_frames(0,30,2)==[]
    assert select_history_frames(5,.5,4)==[0,1,2,3,4]

def test_actual_reader_warm_cache_and_pairing(tmp_path,monkeypatch):
    config,root=source(tmp_path);vae=VAE()
    codec=GoalMemoryCodec(vae,tmp_path/'latents',vae_identity='test')
    dataset=RFCGoalDataset(config,tmp_path/'reader',codec=codec,epoch_size=10,require_cache=False)
    import scripts.data.prepare_subtask_goals as decoding
    original=decoding.decode_frames;calls=[]
    def count(*a,**k):calls.append(list(a[3]));return original(*a,**k)
    monkeypatch.setattr(decoding,'decode_frames',count)
    cold=dataset[0]; n=vae.calls
    assert n==1 and [0] in calls
    calls.clear();warm=dataset[0]
    assert vae.calls==n and [0] not in calls
    assert torch.equal(cold['memory_latents'],warm['memory_latents'])
    assert cold['target_image'].shape==(3,1,384,320)
    assert cold['target_view_valid'].tolist()==[True,False,False]
    assert cold['target_point_valid'].tolist()==[False,False]
    assert cold['record']['goal_source_frame']==cold['record']['instruction_end_exclusive']-1
    assert collate_goal_memory([cold,warm])['memory_latents'].shape==(2,3,48,1,24,20)
    with torch.no_grad():vae.weight.add_(1)
    with pytest.raises(ValueError,match='VAE.*changed'):dataset[0]

def test_finite_ratio_resume_and_source_change(tmp_path):
    config,root=source(tmp_path)
    dataset=RFCGoalDataset(config,tmp_path/'reader',epoch_size=10)
    levels=[dataset.selection_at(i)['level'] for i in range(10)]
    assert levels.count('subtask')==7
    assert [dataset.selection_at(i)['anchor'] for i in range(3,10)]==[dataset.selection_at(i)['anchor'] for i in range(10)][3:]
    dataset.set_epoch(1)
    assert sum(dataset.selection_at(i)['level']=='subtask' for i in range(10))==7
    (root/'meta/info.json').write_text((root/'meta/info.json').read_text()+' ')
    with pytest.raises(ValueError,match='source dependency changed'):dataset[0]

def test_required_cache_miss_before_history_decode(tmp_path):
    config,_=source(tmp_path)
    codec=GoalMemoryCodec(VAE(),tmp_path/'latents',vae_identity='test')
    dataset=RFCGoalDataset(config,tmp_path/'reader',codec=codec,epoch_size=10)
    with pytest.raises(FileNotFoundError,match='precompute'):dataset[0]
    assert codec.vae.calls==0

def test_precompute_all_epochs_and_coverage(tmp_path):
    from scripts.data.precompute_goal_memory import precompute_goal_memory
    config,_=source(tmp_path);vae=VAE()
    codec=GoalMemoryCodec(vae,tmp_path/'latents',vae_identity='test')
    dataset=RFCGoalDataset(config,tmp_path/'reader',codec=codec,epoch_size=10)
    before=precompute_goal_memory(dataset,batch_size=2,audit_only=True)
    assert before['missing']==1 and not before['complete'] and vae.calls==0
    report=precompute_goal_memory(dataset,batch_size=2)
    assert report['written']==1 and report['complete'] and vae.calls==1
    after=precompute_goal_memory(dataset,audit_only=True)
    assert after['hits']==1 and after['missing']==0
    for epoch in range(3):
        dataset.set_epoch(epoch)
        for index in range(10):dataset[index]
    assert vae.calls==1


def test_codec_batched_t1_and_identity(tmp_path):
    vae=VAE();codec=GoalMemoryCodec(vae,tmp_path/'cache',vae_identity='a',max_encode_batch_size=4)
    images=torch.stack([torch.full((3,1,384,320),v) for v in [-.5,.5]])
    output=codec.encode_images(images)
    assert output.shape==(2,48,1,24,20) and vae.calls==1
    assert output[0].mean()<0 and output[1].mean()>0
    codec.put_cached(source='source',episode=0,frame=0,latent=output[0])
    other=VAE()
    with torch.no_grad():other.weight.add_(1)
    changed=GoalMemoryCodec(other,tmp_path/'cache',vae_identity='a')
    assert changed.get_cached(source='source',episode=0,frame=0) is None

def test_three_view_terminal_pairing_and_batched_precompute(tmp_path,monkeypatch):
    from scripts.data.precompute_goal_memory import precompute_goal_memory
    import pyarrow as pa
    import pyarrow.parquet as pq
    config,root=source(tmp_path)
    info=json.loads((root/'meta/info.json').read_text());info['fps']=1
    cameras=['observation.images.egocentric','observation.images.left_wrist','observation.images.right_wrist']
    for camera in cameras:info['features'][camera]=dict(dtype='video',shape=[24,32,3])
    (root/'meta/info.json').write_text(json.dumps(info))
    ep_path=root/'meta/episodes/chunk-000/file-000.parquet';ep=pq.read_table(ep_path).to_pylist()[0]
    for camera in cameras:
        for key,value in [('chunk_index',0),('file_index',0),('from_timestamp',0.)]:ep[f'videos/{camera}/{key}']=value
    pq.write_table(pa.Table.from_pylist([ep]),ep_path)
    # The fixture decoder uses ordinal Mecka frame alignment, independent of FPS.
    vae=VAE();codec=GoalMemoryCodec(vae,tmp_path/'latents',vae_identity='test')
    dataset=RFCGoalDataset(config,tmp_path/'reader',codec=codec,epoch_size=10)
    report=precompute_goal_memory(dataset,batch_size=4)
    assert report['written']==3 and vae.calls==1
    import scripts.data.prepare_subtask_goals as decoding
    original=decoding.decode_frames;calls=[]
    def count(*a,**k):calls.append(list(a[3]));return original(*a,**k)
    monkeypatch.setattr(decoding,'decode_frames',count)
    for index in range(10):
        calls.clear();sample=dataset[index];record=sample['record']
        assert sample['target_view_valid'].all()
        assert calls==[[record['anchor_source_frame'],record['goal_source_frame']]]*3
        canvas=sample['target_image']
        assert torch.allclose(canvas[:,:,0:256,:].mean(),canvas[:,:,256:,:160].mean(),atol=.02)
        assert torch.allclose(canvas[:,:,0:256,:].mean(),canvas[:,:,256:,160:].mean(),atol=.02)
        assert sample['memory_ages'][sample['memory_valid']].min()>0


def test_invalid_target_views_and_codec_free_preflight(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    config,root=source(tmp_path)
    rows=pq.read_table(root/'data.parquet').to_pylist()
    for row in rows:row['canonical.frame_valid']=row['frame_index']!=7
    pq.write_table(pa.Table.from_pylist(rows),root/'data.parquet')
    dataset=RFCGoalDataset(config,tmp_path/'reader',epoch_size=10,subtask_probability=0)
    sample=dataset[0]
    assert not sample['target_view_valid'].any() and not sample['target_image_valid']
    assert not sample['target_point_valid'].any()
    assert torch.isfinite(sample['target_image']).all()
    assert 'memory_images' in sample


def test_missing_subtasks_rejected_instead_of_skewing_ratio(tmp_path):
    config,root=source(tmp_path);(root/'meta/sub_tasks.parquet').unlink()
    with pytest.raises(ValueError,match='no eligible subtask'):
        RFCGoalDataset(config,tmp_path/'reader',epoch_size=10)


def test_video_generation_changes_new_cache_identity(tmp_path):
    config,root=source(tmp_path)
    first=RFCGoalDataset(config,tmp_path/'reader',epoch_size=10)
    path=root/'video.mp4'
    with path.open('ab') as stream:stream.write(b'changed')
    with pytest.raises(ValueError,match='source dependency changed'):first.memory_requests(0)
    second=RFCGoalDataset(config,tmp_path/'reader',epoch_size=10)
    assert second.population_identity!=first.population_identity

@pytest.mark.parametrize('fps',[.3,.5,1.,1.5,2.5,29.97])
def test_compact_precompute_covers_exact_reachable_frames(fps):
    from types import SimpleNamespace
    for a,b in [(0,1),(0,5),(1,2),(6,13),(30,80)]:
        for recent in [0,1,2,4]:
            dataset=RFCGoalDataset.__new__(RFCGoalDataset)
            item=dict(source='s',repo={'repo':'r'},episode=0,start=a,stop=b)
            dataset.pools={'task':[item]};dataset.recent_frames=recent
            dataset._check_source=lambda item:None
            metadata=dict(info={'fps':fps},source_valid=np.ones(100,dtype=bool),cameras=['head',None,None])
            dataset.reader=SimpleNamespace(_episode=lambda *args:metadata)
            actual={r['frame'] for r in dataset.precompute_requests()}
            expected={f for anchor in range(a,b) for f in select_history_frames(anchor,fps,recent)}
            assert actual==expected

def test_metadata_checks_localized_and_frozen(tmp_path,monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import openwam.dataloader.rfc_goal as module
    config,root=source(tmp_path)
    folder=root/'meta/episodes/chunk-000'
    row=pq.read_table(folder/'file-000.parquet').to_pylist()[0]
    for index in range(1,61):
        pq.write_table(pa.Table.from_pylist([dict(row,episode_index=index)]),folder/f'file-{index:03}.parquet')
    codec=GoalMemoryCodec(VAE(),tmp_path/'latents',vae_identity='test')
    dataset=RFCGoalDataset(config,tmp_path/'reader',epoch_size=10,codec=codec,require_cache=False)
    dataset[0]  # Fill once, then measure the warm-cache sample.
    original=module._stamp;calls=[]
    def count(path):calls.append(str(path));return original(path)
    monkeypatch.setattr(module,'_stamp',count)
    sample=dataset[0]
    assert len(calls)==14  # Two checks of seven actual source files, independent of shard count.
    assert not any('file-060.parquet' in p for p in calls)
    # Evict reader metadata and ensure it still resolves the exact frozen row.
    dataset.reader._episodes.clear()
    import openwam.dataloader.native_stage2_io as io
    monkeypatch.setattr(io,'read_episode_metadata',lambda *a: (_ for _ in ()).throw(AssertionError('global reader used')))
    assert dataset[0]['record']==sample['record']
    path=folder/'file-000.parquet'
    pq.write_table(pa.Table.from_pylist([dict(row,length=11)]),path)
    with pytest.raises(ValueError,match='source dependency changed'):dataset[0]


def test_duplicate_metadata_rejected_and_epoch_snapshot_checked(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    config,root=source(tmp_path);folder=root/'meta/episodes/chunk-000'
    row=pq.read_table(folder/'file-000.parquet').to_pylist()[0]
    dataset=RFCGoalDataset(config,tmp_path/'reader',epoch_size=10)
    pq.write_table(pa.Table.from_pylist([row]),folder/'file-001.parquet')
    with pytest.raises(ValueError,match='metadata.*changed'):dataset.set_epoch(1)
    with pytest.raises(ValueError,match='nonunique episode metadata'):
        RFCGoalDataset(config,tmp_path/'reader2',epoch_size=10)


def test_resume_fresh_dataset_crosses_epoch_boundary(tmp_path):
    config,_=source(tmp_path)
    first=RFCGoalDataset(config,tmp_path/'reader',epoch_size=10)
    def identity(sample):
        r=sample['record'];return sample['granularity'],r['source_episode'],r['anchor_source_frame'],r['goal_source_frame']
    uninterrupted=[identity(first.sample(i)) for i in range(13)]
    resumed=RFCGoalDataset(config,tmp_path/'reader2',epoch_size=10)
    assert [identity(resumed.sample(i)) for i in range(8,13)]==uninterrupted[8:]

def test_config_precompute_loads_exact_baseline_vae(tmp_path):
    from safetensors.torch import save_file
    from scripts.data.precompute_goal_memory import load_configured_vae
    from openwam.model.goal.dependencies import module_identity
    checkpoint=tmp_path/'baseline.safetensors'
    save_file({'video_backbone.vae.weight':torch.tensor(3.),'unrelated.weight':torch.ones(2)},str(checkpoint))
    config=dict(baseline_checkpoint=str(checkpoint),components=dict(
        vae=dict(model_class='test_rfc_goal_data.VAE'),dit=dict(model_class='must.not.load'),
        text_encoder=dict(model_class='must.not.load')))
    loaded=load_configured_vae(config,torch.float32)
    expected=VAE();expected.load_state_dict({'weight':torch.tensor(3.)});expected.eval().requires_grad_(False)
    assert module_identity(loaded)==module_identity(expected)
    a=GoalMemoryCodec(loaded,tmp_path/'latents',vae_identity=module_identity(loaded)['fingerprint'])
    b=GoalMemoryCodec(expected,tmp_path/'latents',vae_identity=module_identity(expected)['fingerprint'])
    assert a.vae_identity==b.vae_identity

def test_repo_weights_preserve_marginals_with_unequal_eligible_lengths(tmp_path):
    import shutil
    import yaml
    config,root=source(tmp_path)
    shutil.copytree(root,root.parent/'fixture2')
    doc=yaml.safe_load(config.read_text())
    doc['selection']['segments_by_repo']={'mecka/fixture':[[0,1,2]],'mecka/fixture2':[[0,0,7]]}
    doc['selection']['repo_weights']={'mecka/fixture':1,'mecka/fixture2':1}
    config.write_text(yaml.safe_dump(doc))
    dataset=RFCGoalDataset(config,tmp_path/'reader',epoch_size=1000)
    for level,pool in dataset.pools.items():
        mass=sum(w for item,w in zip(pool,dataset.pool_weights[level]) if item['repo']['repo']=='mecka/fixture')
        assert mass==pytest.approx(.5)
    selections=[dataset.selection_at(i) for i in range(1000)]
    assert 430<sum(s['repo']['repo']=='mecka/fixture' for s in selections)<570

def test_precompute_cli_config_is_training_cache_compatible(tmp_path,monkeypatch):
    import sys
    import yaml
    from safetensors.torch import save_file
    from scripts.data.precompute_goal_memory import main,load_configured_vae
    from openwam.train.goal_memory import load_config
    from openwam.train.goal_memory_data import make_dataset
    from openwam.model.goal.dependencies import module_identity
    source_yaml,_=source(tmp_path)
    checkpoint=tmp_path/'baseline.safetensors'
    save_file({'video_backbone.vae.weight':torch.tensor(3.)},str(checkpoint))
    configuration=dict(representation='goal_memory',stage=1,max_updates=10,seed=42,
        baseline_checkpoint=str(checkpoint),components=dict(vae=dict(model_class='test_rfc_goal_data.VAE')),
        dataset=dict(source_yaml=str(source_yaml),cache_dir=str(tmp_path/'reader'),epoch_size=10),
        memory_cache=dict(directory=str(tmp_path/'latents')))
    config_path=tmp_path/'config.yaml';config_path.write_text(yaml.safe_dump(configuration))
    report_path=tmp_path/'coverage.json'
    monkeypatch.setattr(sys,'argv',['precompute_goal_memory.py','--config',str(config_path),
        '--dtype','float32','--device','cpu','--report',str(report_path)])
    assert main()==0
    report=json.loads(report_path.read_text());assert report['written']==1 and report['complete']
    config=load_config(config_path);vae=load_configured_vae(config,torch.float32)
    codec=GoalMemoryCodec(vae,config['memory_cache']['directory'],
        vae_identity=module_identity(vae)['fingerprint'],device='cpu',dtype=torch.float32)
    dataset=make_dataset(config,codec)
    dataset[0]
    assert vae.calls==0  # A fresh training provider finds the offline-produced cache.
    assert report['population_identity']==dataset.population_identity


def test_frozen_metadata_missing_selected_episode_rejected(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    config,root=source(tmp_path);path=root/'meta/episodes/chunk-000/file-000.parquet'
    row=pq.read_table(path).to_pylist()[0]
    pq.write_table(pa.Table.from_pylist([dict(row,episode_index=1)]),path)
    with pytest.raises(ValueError,match='missing episode metadata'):
        RFCGoalDataset(config,tmp_path/'reader',epoch_size=10)
