"""Portable streaming runner invariants; no checkpoint or source data required."""
import json
from pathlib import Path
import pytest
import torch
from openwam.train.goal_curated import load_config, TextContextCache, prepare_batch, preflight, train_curated
from openwam.train.goal_patch import make_smoke_data


def test_relative_base_and_override(tmp_path):
    (tmp_path/'base.yaml').write_text('dataset:\n  source_yaml: relative.yaml\n  max_cached_episodes: 2\nmax_steps: 7\n')
    (tmp_path/'child.yaml').write_text('base_config: base.yaml\ndataset:\n  subtask_probability: 0.5\n')
    config=load_config(tmp_path/'child.yaml', source_yaml='/tmp/explicit.yaml')
    assert config['max_steps']==7
    assert config['dataset']['max_cached_episodes']==2
    assert config['dataset']['source_yaml']=='/tmp/explicit.yaml'


def test_context_cache_bounded_cpu_and_no_cache():
    calls=[]
    def encode(prompt):
        calls.append(prompt); return torch.ones(1,2,3),torch.tensor([2])
    cache=TextContextCache(encode,max_entries=1)
    assert cache.get('a')[0].device.type=='cpu'
    cache.get('a'); cache.get('b'); cache.get('a')
    assert calls==['a','b','a'] and len(cache)==1
    disabled=TextContextCache(encode,max_entries=0)
    disabled.get('x');disabled.get('x');assert len(disabled)==0


class TinyDataset:
    def __init__(self):
        self.expert,self.batches,self.records=make_smoke_data()
        self.records[0].update(metric_key='synthetic:task:case0',granularity='task')
        self.requested=[]
    def sample(self,step):
        self.requested.append(step);return self.records[0],self.batches[0]
    def evaluation_samples(self):return list(zip(self.records,self.batches))
    def scope_summary(self):return {'families':['synthetic'],'synthetic':True}


def test_preflight_bf16_exports_without_model(tmp_path):
    dataset=TinyDataset()
    for b in dataset.batches:
        b['observed']=b['observed'].bfloat16();b['targets'].image=b['targets'].image.bfloat16()
    result=preflight(dataset,tmp_path/'preflight')
    assert result['model_loaded'] is False
    import numpy as np
    arrays=np.load(tmp_path/'preflight/inputs/case_0000.npz')
    assert 'observed_bfloat16_bits' in arrays
    assert (tmp_path/'preflight/index.html').is_file()
    assert not dataset.requested


def test_streaming_real_dit_checkpoint_and_history(tmp_path):
    torch.set_num_threads(2)
    dataset=TinyDataset()
    before=next(dataset.expert.dit.parameters()).detach().clone()
    result=train_curated(dataset.expert,dataset=dataset,context_cache=None,output_dir=tmp_path/'run',
        steps=2,eval_every=1,sampling_steps=1,learning_rate=.001,seed=19,
        precomputed_text_fingerprint='0'*64)
    assert dataset.requested==[0,1]
    assert not torch.equal(before,next(dataset.expert.dit.parameters()))
    assert [e['step'] for e in result['evaluations']]==[0,1,2]
    assert 'source:synthetic:task:case0' in result['evaluations'][0]['point_metrics']
    assert (tmp_path/'run/patch_goal.safetensors').is_file()
    assert (tmp_path/'run/timeline.html').is_file()
    assert result['completed_steps']==2


def test_reject_distributed_and_invalid_budgets(tmp_path,monkeypatch):
    from openwam.train.goal_curated import require_single_process
    monkeypatch.setenv('WORLD_SIZE','2')
    with pytest.raises(ValueError,match='one process'):require_single_process()
    monkeypatch.setenv('WORLD_SIZE','1')
    dataset=TinyDataset()
    with pytest.raises(ValueError,match='positive integer'):
        train_curated(dataset.expert,dataset=dataset,context_cache=None,output_dir=tmp_path/'bad',steps=0)
    assert not (tmp_path/'bad').exists()


def test_uv_keeps_float32_when_images_use_bfloat16():
    dataset=TinyDataset();record,batch=dataset.sample(0)
    result=prepare_batch(record,batch,device='cpu',dtype=torch.bfloat16)
    assert result['observed'].dtype==torch.bfloat16
    assert result['targets'].image.dtype==torch.bfloat16
    assert result['targets'].uv.dtype==torch.float32
    assert batch['observed'].dtype==torch.float32


def test_config_cycle_fails_and_data_path_uses_project_root(tmp_path):
    from openwam.train.goal_curated import ROOT
    config=tmp_path/'config.yaml'
    config.write_text('dataset:\n  source_yaml: configs/data.yaml\n')
    assert load_config(config)['dataset']['source_yaml']==str(ROOT/'configs/data.yaml')
    config.write_text('base_config: config.yaml\n')
    with pytest.raises(ValueError,match='cyclic'):load_config(config)
