import random
import numpy as np
import pytest
import torch


def update(model, optimizer):
    optimizer.zero_grad()
    x = torch.randn(2, 3) + np.random.rand() + random.random()
    model(x.to(next(model.parameters()).dtype)).float().square().mean().backward()
    optimizer.step()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_resume_preserves_fp32_master_moments_cursor_and_rng(tmp_path, dtype):
    from openwam.train.goal_memory_checkpoint import save_training_state, restore_training_state
    from openwam.train.goal_overfit import FP32MasterAdamW
    torch.manual_seed(9); random.seed(9); np.random.seed(9)
    model = torch.nn.Linear(3, 2).to(dtype)
    optimizer = FP32MasterAdamW([dict(params=list(model.parameters()), lr=.003)])
    update(model, optimizer)
    path = tmp_path / 'state.pt'
    contract = dict(stage=1, total_updates=5, subtask_probability=.5, population='abc')
    save_training_state(path, model, optimizer, update=1, contract=contract)
    update(model, optimizer)
    expected = {k: v.clone() for k, v in model.state_dict().items()}
    restored = torch.nn.Linear(3, 2).to(dtype)
    restored_opt = FP32MasterAdamW([dict(params=list(restored.parameters()), lr=.003)])
    assert restore_training_state(path, restored, restored_opt, contract=contract) == 1
    update(restored, restored_opt)
    for key, value in restored.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    for state in restored_opt.optimizer.state.values():
        assert state['exp_avg'].dtype == torch.float32


def test_resume_rejects_changed_data_ratio_schedule_or_model_before_loading(tmp_path):
    from openwam.train.goal_memory_checkpoint import save_training_state, restore_training_state
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    contract = dict(total_updates=10, subtask_probability=.5, population='abc')
    path = tmp_path / 'state.pt'
    save_training_state(path, model, optimizer, update=2, contract=contract)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    for change in (dict(total_updates=11), dict(subtask_probability=.3), dict(population='xyz')):
        with pytest.raises(ValueError, match='contract'):
            restore_training_state(path, model, optimizer, contract={**contract, **change})
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])
    with pytest.raises(ValueError, match='model'):
        restore_training_state(path, torch.nn.Linear(4, 2), optimizer, contract=contract)


def test_portable_bundle_roundtrip_and_dependency_gate(tmp_path):
    from openwam.train.goal_memory_checkpoint import save_memory_bundle, load_memory_bundle, memory_bundle_metadata
    from openwam.model.goal.goal_memory import GoalMemoryExpert, GoalMemoryConfig
    from openwam.model.video_backbone.wan.models.dit import WanModel
    from openwam.train.goal_trainer import TINY_DIT_CONFIG
    config = {**TINY_DIT_CONFIG, 'num_layers': 2}
    model = GoalMemoryExpert(WanModel(**config), GoalMemoryConfig(history_dropout=0)).eval()
    dep = dict(vae='weights-a', text='text-a', preprocess='1Hz-T1-multiview')
    path = tmp_path / 'model.safetensors'
    save_memory_bundle(model, path, dit_config=config, dependencies=dep)
    loaded = load_memory_bundle(path, dependencies=dep).eval()
    assert memory_bundle_metadata(path)['format'] == 'openwam.goal_memory.v1'
    assert not any('vae' in k for k in loaded.state_dict())
    for key, value in model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key])
    with pytest.raises(ValueError, match='dependencies'):
        load_memory_bundle(path, dependencies={**dep, 'vae': 'weights-b'})
    from safetensors.torch import save_file
    save_file({'x': torch.zeros(1)}, str(tmp_path/'old.safetensors'), metadata={'patch_goal_bundle':'{}'})
    with pytest.raises(ValueError, match='old C'):
        memory_bundle_metadata(tmp_path/'old.safetensors')

@pytest.mark.parametrize('mismatch', ['dtype', 'device', 'model_device', 'autocast'])
def test_resume_execution_mismatch_rejected_before_any_mutation(tmp_path, mismatch):
    from openwam.train.goal_memory_checkpoint import save_training_state, restore_training_state
    model = torch.nn.Linear(3, 2)
    from openwam.train.goal_overfit import FP32MasterAdamW
    optimizer = FP32MasterAdamW([dict(params=list(model.parameters()), lr=.003)])
    update(model, optimizer)
    path = tmp_path/'resume.pt'
    save_training_state(path, model, optimizer, update=1, contract={})
    target = torch.nn.Linear(3, 2).to(torch.bfloat16 if mismatch == 'dtype' else torch.float32)
    target_opt = FP32MasterAdamW([dict(params=list(target.parameters()), lr=.003)])
    update(target, target_opt)
    masters = [m.clone() for _,m in target_opt.parameter_pairs]
    moments = {p:{k:v.clone() if torch.is_tensor(v) else v for k,v in state.items()}
               for p,state in target_opt.optimizer.state.items()}
    if mismatch == 'device':
        payload = torch.load(path, weights_only=False)
        payload.setdefault('execution', {})['rng_devices'] = ['cuda:0']
        torch.save(payload, path)
    if mismatch == 'model_device':
        payload = torch.load(path, weights_only=False)
        payload['execution']['tensors']['weight']['device'] = 'cuda:0'
        torch.save(payload, path)
    before = {k:v.clone() for k,v in target.state_dict().items()}
    rng = torch.get_rng_state().clone()
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    with torch.autocast('cpu', enabled=mismatch == 'autocast', dtype=torch.bfloat16):
        with pytest.raises(ValueError, match='execution'):
            restore_training_state(path, target, target_opt, contract={})
    for (_,master), saved in zip(target_opt.parameter_pairs, masters):
        assert torch.equal(master, saved)
    for parameter,state in target_opt.optimizer.state.items():
        for key,value in state.items():
            torch.testing.assert_close(value, moments[parameter][key], rtol=0, atol=0)
    assert random.getstate() == python_rng
    after_numpy = np.random.get_state()
    assert numpy_rng[0] == after_numpy[0] and numpy_rng[2:] == after_numpy[2:]
    np.testing.assert_array_equal(numpy_rng[1], after_numpy[1])
    assert torch.equal(rng, torch.get_rng_state())
    for k,v in target.state_dict().items():
        assert torch.equal(v, before[k])
