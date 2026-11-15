import pytest
import torch

from test_goal_memory_core import fixture
from openwam.train.goal_image_metrics import goal_image_metrics


def test_kf1_endpoint_ignores_annotated_remaining_time_in_train_and_eval():
    model, args = fixture()
    model.config.goal_horizon_mode = 'kf1_endpoint'
    for training in (True, False):
        model.train(training)
        first = model.prepare_state(**args)
        second = model.prepare_state(**(args | {'goal_horizon_s': torch.tensor([0., 90.])}))
        torch.testing.assert_close(first.time, second.time, rtol=0, atol=0)
        assert model.config.goal_horizon_mode == 'kf1_endpoint'


def test_seconds_mode_retains_legacy_conditioning_and_tensor_schema():
    model, args = fixture()
    model.config.goal_horizon_mode = 'seconds'
    first = model.prepare_state(**args)
    second = model.prepare_state(**(args | {'goal_horizon_s': torch.zeros(2)}))
    assert not torch.equal(first.time[:, first.target_slice], second.time[:, second.target_slice])
    names = {k: (v.shape, v.dtype) for k, v in model.state_dict().items()}
    model.config.goal_horizon_mode = 'kf1_endpoint'
    assert names == {k: (v.shape, v.dtype) for k, v in model.state_dict().items()}


def test_configuration_rejects_unknown_time_semantics():
    from openwam.model.goal.goal_memory import GoalMemoryConfig
    with pytest.raises(ValueError, match='goal_horizon_mode'):
        GoalMemoryConfig(goal_horizon_mode='unknown')


def test_normal_training_loss_uses_fixed_condition():
    model, args = fixture()
    model.config.goal_horizon_mode = 'kf1_endpoint'
    model.train()
    target = args.pop('noisy_target_latents')
    args.pop('timestep')
    options = dict(target_latents=target, target_uv=torch.zeros(2, 2, 2),
                   target_image_valid=torch.ones(2, dtype=torch.bool),
                   target_point_valid=torch.ones(2, 2, dtype=torch.bool),
                   target_in_frame=torch.ones(2, 2, dtype=torch.bool),
                   sigma=torch.tensor([.2, .7]), noise=torch.ones_like(target))
    torch.manual_seed(10)
    first = model.training_loss(**args, **options)
    torch.manual_seed(10)
    second = model.training_loss(**(args | {'goal_horizon_s': torch.zeros(2)}), **options)
    torch.testing.assert_close(first['loss'], second['loss'], rtol=0, atol=0)


def test_bundle_retains_kf1_mode_and_legacy_bundle_defaults_to_seconds(tmp_path):
    import json
    from safetensors import safe_open
    from safetensors.torch import save_file
    from openwam.train.goal_memory_checkpoint import save_memory_bundle, load_memory_bundle
    model, _ = fixture()
    dit_config = dict(dim=24, in_dim=48, ffn_dim=48, out_dim=48, text_dim=16,
                      freq_dim=16, eps=1e-6, patch_size=(1, 2, 2), num_heads=2, num_layers=2,
                      has_image_input=False, require_vae_embedding=False, require_clip_embedding=False,
                      fuse_vae_embedding_in_latents=True)
    model.config.goal_horizon_mode = 'kf1_endpoint'
    path = tmp_path / 'model.safetensors'
    save_memory_bundle(model, path, dit_config=dit_config, dependencies={})
    restored = load_memory_bundle(path, dependencies={})
    assert restored.config.goal_horizon_mode == 'kf1_endpoint'
    with safe_open(str(path), framework='pt', device='cpu') as source:
        metadata = source.metadata()
        values = {key: source.get_tensor(key) for key in source.keys()}
    bundle = json.loads(metadata['goal_memory_bundle'])
    bundle['goal_memory'].pop('goal_horizon_mode')
    metadata['goal_memory_bundle'] = json.dumps(bundle)
    save_file(values, str(tmp_path / 'legacy.safetensors'), metadata=metadata)
    legacy = load_memory_bundle(tmp_path / 'legacy.safetensors', dependencies={})
    assert legacy.config.goal_horizon_mode == 'seconds'


def test_region_metrics_copy_baseline_and_valid_point_union():
    current = torch.zeros(3, 1, 4, 6, dtype=torch.uint8)
    target = current.clone()
    target[:, :, :, :3] = 255
    prediction = target.float() / 255
    metric = goal_image_metrics(prediction, current, target, image_valid=True,
                                target_uv=torch.tensor([[0., 0.], [1., 1.]]),
                                point_valid=torch.tensor([True, False]), tcp_radius_px=0)
    assert metric['regions']['full']['copy_mae'] == .5
    assert metric['regions']['changed']['copy_mae'] == 1.
    assert metric['regions']['changed']['model_mae'] == 0.
    assert metric['regions']['changed']['gain_over_copy'] == 1.
    assert metric['regions']['static']['copy_mae'] == 0.
    assert metric['regions']['tcp']['rgb_den'] == 3
    assert metric['regions']['full']['rgb_den'] == 72


def test_missing_image_empty_regions_and_nonfinite_prediction():
    raw = torch.zeros(3, 1, 4, 6, dtype=torch.uint8)
    result = goal_image_metrics(raw.float(), raw, raw, image_valid=False,
                               target_uv=torch.zeros(2, 2), point_valid=torch.zeros(2, dtype=torch.bool))
    assert all(r['rgb_den'] == 0 and r['model_mae'] is None for r in result['regions'].values())
    result = goal_image_metrics(raw.float(), raw, raw, image_valid=True,
                               target_uv=torch.zeros(2, 2), point_valid=torch.zeros(2, dtype=torch.bool))
    assert result['regions']['changed']['model_mae'] is None
    assert result['regions']['tcp']['model_mae'] is None
    with pytest.raises(ValueError, match='finite'):
        goal_image_metrics(raw.float() + float('nan'), raw, raw, image_valid=True,
                           target_uv=torch.zeros(2, 2), point_valid=torch.ones(2, dtype=torch.bool))
