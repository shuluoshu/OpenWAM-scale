"""The public Stage2 path reads both token groups without image/TCP decoding."""
from types import SimpleNamespace
import copy

import pytest
import torch
from omegaconf import OmegaConf

from test_query_goal import make_query, QUERY
from test_multiview_patch import multiview_batch
from test_patch_stage2_memory import make_arch, inputs


def config():
    return dict(enabled=True, format='openwam.goal.query.v1', representation='query_latent',
        query=QUERY, input_policy='clean_multiview', observation_mode='multiview',
        mode='predicted', conditioning='query_memory', num_steps=1,
        frozen_dependencies_eval=True, microbatch_size=1, cache_size=8, cache_max_bytes=1000000)


def attach():
    from openwam.model.goal.runtime import configure_goal_adapter, GoalPredictionCache
    a = make_arch()
    a.goal_expert = make_query().freeze()
    a.goal_config = config()
    configure_goal_adapter(a, config())
    a.goal_cache = GoalPredictionCache()
    a.goal_asset_identity = 'query-fixture'
    return a


def test_prepare_queries_never_calls_generate_and_missing_head_masks(monkeypatch):
    from openwam.model.goal.runtime import prepare_goal_inputs
    a = attach(); obs, ctx, _, views = multiview_batch()
    monkeypatch.setattr(a.goal_expert, 'generate', lambda *a, **kw: pytest.fail('decoder invoked'))
    views[1, 0] = False
    out = prepare_goal_inputs(a, None, ['a', 'b'], ctx, head_valid=[True, False],
        seeds=[1, 2], goal_observed_multiview=obs, goal_view_valid=views)
    assert out['goal_latent'].shape == (2, 10, 32)
    assert out['goal_token_valid'][0].all() and not out['goal_token_valid'][1].any()
    assert not out['goal_latent'].requires_grad
    assert a.goal_diagnostics['representation'] == 'query_latent'
    assert torch.isnan(a.goal_diagnostics['predicted_uv']).all()
    assert a.goal_diagnostics['goal_decode_available'] is False


def test_both_groups_affect_action_video_and_no_grad_enters_teacher():
    a = attach()
    x = inputs(); z = torch.randn(1, 10, 32, requires_grad=True)
    mask = torch.ones(1, 10, dtype=torch.bool)
    y = a(**x, goal_latent=z, goal_token_valid=mask)
    for start, end in ((0, 8), (8, 10)):
        changed = z.detach().clone(); changed[:, start:end] += 5.
        other = a(**x, goal_latent=changed, goal_token_valid=mask)
        assert not torch.allclose(y[1], other[1])
        assert not torch.allclose(y[0][:, :, 1:], other[0][:, :, 1:])
        torch.testing.assert_close(y[0][:, :, :1], other[0][:, :, :1])
    sum(v.square().mean() for v in y).backward()
    assert z.grad is None
    assert a.goal_adapter.projection.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in a.goal_expert.parameters())
    with pytest.raises(ValueError, match='mask|token'):
        a(**x, goal_latent=z, goal_token_valid=torch.ones(1, 9, dtype=torch.bool))


def test_dynamic_mask_retains_legacy_default_validation():
    from openwam.model.goal.conditioning import append_goal_mask
    base = torch.ones(4, 7, dtype=torch.bool)
    valid = torch.ones(1, 10, dtype=torch.bool)
    with pytest.raises(ValueError):
        append_goal_mask(base, valid, 2)
    result = append_goal_mask(base, valid, 2, num_tokens=10)
    assert result.shape == (1, 1, 4, 17) and not result[:, :, :2, 7:].any()


def test_model_bundle_contract_cannot_silently_switch():
    from openwam.model.goal.runtime import goal_observation_mode, goal_representation
    assert goal_observation_mode(config()) == 'multiview'
    assert goal_representation(config()) == 'query_latent'
    with pytest.raises(ValueError, match='representation|format'):
        goal_observation_mode({**config(), 'representation': 'patch_denoise'})
    from openwam.train.utils.stage2 import validate_stage2_config
    from hydra import compose, initialize_config_dir
    from pathlib import Path
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[2]/'configs'), version_base=None):
        cfg = compose(config_name='goal/stage2_query_sample')
    validate_stage2_config(cfg)
    cfg.model.goal_representation = 'patch_denoise'
    with pytest.raises(ValueError, match='representation|format'):
        validate_stage2_config(cfg)


@pytest.mark.parametrize('change', ['mode', 'width'])
def test_direct_checkpoint_loader_rejects_changed_representation_before_build(tmp_path, monkeypatch, change):
    from openwam.train.utils import ckpt_model_loader as loader
    import openwam.model as model
    from hydra import compose, initialize_config_dir
    from pathlib import Path
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[2]/'configs'), version_base=None):
        saved = compose(config_name='goal/stage2_query_sample')
        live = compose(config_name='goal/stage2_multiview_sample' if change=='mode' else 'goal/stage2_query_sample')
    if change == 'width':
        live.model.query_goal.dim = 256
    OmegaConf.save(saved, tmp_path/'config.yaml')
    monkeypatch.setattr(model, 'resolve_architecture_config', lambda *a: pytest.fail('reached model build with incompatible goal'))
    with pytest.raises(ValueError, match='representation|query'):
        loader.build_architecture_from_ckpt_dir(str(tmp_path), override_cfg=live, weights_required=False)


def test_configure_goal_rejects_model_selector_disagreement_before_asset_load():
    from openwam.model.goal.runtime import configure_goal
    cfg = OmegaConf.create(dict(model=dict(goal_representation='patch_denoise'), goal=config()))
    with pytest.raises(ValueError, match='representation'):
        configure_goal(make_arch(), cfg)


def test_query_stage2_metrics_do_not_decode_or_measure_unavailable_goal():
    from openwam.train.stage2_eval import prediction_metrics
    from test_stage2_history import sample
    import numpy as np
    s = sample()
    result = dict(normalized_actions=np.zeros((32,112)), video=s['video'])
    diagnostics = dict(image_token_valid=torch.ones(1, 10, dtype=torch.bool),
        predicted_uv=torch.full((1, 2, 2), float('nan')), goal_decode_available=False)
    metrics = prediction_metrics(result, diagnostics, None, s)
    assert metrics['goal_image_mae']['value'] is None
    assert metrics['tcp_pixel_error']['value'] is None
    assert metrics['future_video_mae']['value'] is not None


def test_real_encoder_load_portable_checkpoint_and_public_generate(tmp_path):
    from openwam.train.goal_query import save_query_bundle
    from openwam.model.goal import runtime
    from openwam.model.video_backbone.wan.shared.diffusion.flow_match import FlowMatchScheduler
    from safetensors.torch import load_file
    import shutil
    a = make_arch(); teacher = make_query()
    teacher.codec.vae = a.video_backbone.vae
    path = tmp_path/'original.safetensors'
    save_query_bundle(teacher, path, precomputed_text_fingerprint='a'*64)
    cfg = OmegaConf.create(dict(model=dict(goal_representation='query_latent', query_goal=QUERY),
        goal={**config(), 'checkpoint':str(path), 'precomputed_text_fingerprint':'a'*64}))
    runtime.configure_goal(a, cfg)
    assert not hasattr(a.goal_expert, 'decoder')
    assert not hasattr(a.goal_expert, 'readout')
    obs, ctx, _, views = multiview_batch(1)
    def prepare(arch):
        return runtime.prepare_goal_inputs(arch, None, ['move'], ctx, head_valid=[True],
            seeds=[17], goal_observed_multiview=obs, goal_view_valid=views)
    expected = prepare(a)
    saved = tmp_path/'saved'
    runtime.save_goal_assets(a, saved, cfg)
    a.save_checkpoint(str(saved/'checkpoint_step_1.safetensors'))
    weights = load_file(str(saved/'checkpoint_step_1.safetensors'))
    assert not any(k.startswith('goal_expert.') for k in weights)
    path.unlink()
    moved = tmp_path/'moved'; shutil.move(saved, moved)
    b = make_arch()
    runtime.configure_goal_adapter(b, OmegaConf.to_container(cfg.goal, resolve=True))
    b.load_checkpoint(str(moved/'checkpoint_step_1.safetensors'))
    runtime.configure_goal(b, cfg, checkpoint_dir=moved)
    actual = prepare(b)
    for k in expected:
        torch.testing.assert_close(actual[k], expected[k], rtol=0, atol=0)
    b.video_backbone._scheduler = FlowMatchScheduler(template='Wan')
    b.init_training_schedulers()
    x = inputs()
    b.video_backbone.preprocess_input_for_inference = lambda **kw: dict(context=ctx, latents=x['latents'].clone())
    result = b.generate([(1000.,1000.),(0.,0.)], 'move', num_frames=9, action_num_frames=33,
        goal_observed_multiview=obs, goal_view_valid=views, goal_seed=17, decode_video=False)
    assert result['normalized_actions'].shape == (32,112)
    assert b.goal_diagnostics['predicted_latent'].shape == (1,10,32)
    assert b.goal_diagnostics['goal_decode_available'] is False


def test_query_stage2_report_is_complete_without_goal_decoder(tmp_path, monkeypatch):
    from test_stage2_history import Tiny, sample, schedule
    from openwam.train import stage2_eval as evaluation
    import json
    class QueryTiny(Tiny):
        def generate(self, *args, **kwargs):
            result = super().generate(*args, **kwargs)
            self.goal_diagnostics.update(predicted_latent=torch.ones(1,10,32),
                predicted_uv=torch.full((1,2,2), float('nan')),
                image_token_valid=torch.ones(1,10,dtype=torch.bool),
                representation='query_latent', goal_decode_available=False)
            return result
    a = QueryTiny()
    a.goal_expert.representation = 'query_latent'
    a.goal_expert.codec.decode = lambda *args: pytest.fail('query tokens decoded as image')
    monkeypatch.setattr(evaluation, 'make_schedule', schedule)
    evaluation.evaluate_stage2(a, [sample()], tmp_path, step=0, seed=3, num_steps=1)
    folder = tmp_path/'eval_step_000000'
    report = json.loads((folder/'metrics.json').read_text())
    assert report['goal_representation'] == 'query_latent'
    assert report['aggregate']['goal_image_mae']['value'] is None
    assert not (folder/'case_000/goal_prediction.png').exists()
    assert (folder/'case_000/future_gt_prediction.mp4').is_file()


@pytest.mark.parametrize('scenario', ['full_state', 'public_loaders'])
def test_query_real_accelerate_resume_and_public_loaders(tmp_path, monkeypatch, scenario):
    """Reuse the full existing CPU resume/loader protocol with a query teacher."""
    import test_patch_stage2_checkpoint as protocol
    from openwam.model.goal import runtime
    from openwam.model.goal.query_expert import QueryGoalExpert
    from openwam.train.goal_patch import make_smoke_data
    from openwam.train.goal_query import save_query_bundle
    def configured(folder):
        arch = make_arch(); base, batches, _ = make_smoke_data()
        base.codec.vae = arch.video_backbone.vae
        teacher = QueryGoalExpert(base.dit, base.codec, dit_config=base.dit_config,
                                   query_config=QUERY, observation_mode='head')
        asset = folder/'original.safetensors'
        save_query_bundle(teacher, asset, precomputed_text_fingerprint='a'*64)
        cfg = OmegaConf.create({'goal': {**config(), 'input_policy':'clean_head',
            'observation_mode':'head', 'checkpoint':str(asset), 'precomputed_text_fingerprint':'a'*64}})
        runtime.configure_goal(arch, cfg)
        return arch, cfg, batches[0]
    monkeypatch.setattr(protocol, 'configured', configured)
    if scenario == 'full_state':
        protocol.test_full_state_real_accelerate_next_update_rng_scheduler_and_sampler(tmp_path, monkeypatch)
    else:
        protocol.test_train_and_deploy_loaders_restore_real_adapter_before_goal(tmp_path, monkeypatch)
