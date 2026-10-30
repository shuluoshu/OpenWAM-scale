"""Action-only Stage2 removes future tokens, rather than masking them."""
import copy
import pytest
import torch
from test_patch_stage2_memory import make_arch, inputs


def test_action_only_real_blocks_remove_future_tokens_and_video_head(monkeypatch):
    a = make_arch(); a.cfg['future_video_mode'] = 'none'
    x = inputs(); x['first_frame_latents'] = x['latents'][:, :, :1].clone()
    seen = []
    original = a.video_backbone.prepare
    def prepare(**kw):
        seen.append(kw['latents'].shape[2]); return original(**kw)
    monkeypatch.setattr(a.video_backbone, 'prepare', prepare)
    monkeypatch.setattr(a.video_backbone, 'finalize', lambda *args: pytest.fail('video head invoked'))
    y = a(**x)
    other = copy.copy(x); other['latents'] = x['latents'].clone(); other['latents'][:, :, 1:] += 100
    z = a(**other)
    assert y[0] is None and seen == [1, 1]
    torch.testing.assert_close(y[1], z[1])
    y[1].square().mean().backward()
    assert a.action_backbone.action_encoder.weight.grad.abs().sum() > 0


def test_future_video_mode_defaults_and_validation():
    a = make_arch(); assert a.future_video_mode == 'joint'
    a.cfg['future_video_mode'] = 'invalid'
    with pytest.raises(ValueError, match='future_video_mode'): _ = a.future_video_mode


def test_preprocess_drops_future_before_encoder(monkeypatch):
    a = make_arch(); a.cfg['future_video_mode'] = 'none'
    seen = []
    monkeypatch.setattr(a.video_backbone, 'preprocess_input_for_train', lambda **kw: seen.append(kw) or {})
    frames = [[object(), object(), object()]]
    a.preprocess(frames=frames, text=['move'], vace_videos=frames)
    assert seen[0]['frames'] == [frames[0][:1]]
    assert seen[0]['vace_videos'] is None
    a.cfg['future_video_mode'] = 'joint'
    a.preprocess(frames=frames, text=['move'])
    assert seen[1]['frames'] is frames


def test_action_only_loss_backward_skips_video_noise_loss_and_head(monkeypatch):
    a = make_arch(); a.cfg['future_video_mode'] = 'none'
    from openwam.model.video_backbone.wan.shared.diffusion.flow_match import FlowMatchScheduler
    a.video_backbone._scheduler = FlowMatchScheduler(template="Wan")
    a.init_training_schedulers(10)
    x = inputs(); current = x['latents'][:, :, :1].clone()
    monkeypatch.setattr(a, '_compute_video_loss', lambda *args: pytest.fail('video loss invoked'))
    monkeypatch.setattr(a.video_backbone, 'finalize', lambda *args: pytest.fail('video head invoked'))
    result = a.compute_loss(actions=x['noisy_actions'], input_latents=x['latents'],
        first_frame_latents=current, context=x['context'],
        action_is_pad=torch.zeros_like(x['noisy_actions'], dtype=torch.bool))
    assert result['loss_video'].item() == 0
    assert torch.isfinite(result['loss'])
    result['loss'].backward()
    assert a.action_backbone.action_encoder.weight.grad.abs().sum() > 0


def test_joint_baseline_keeps_future_and_video_prediction():
    a = make_arch(); x = inputs()
    video, action = a(**x)
    assert video.shape == x['latents'].shape and action.shape == x['noisy_actions'].shape


def test_action_only_generate_uses_current_only_and_no_decode(monkeypatch):
    from openwam.model.video_backbone.wan.shared.diffusion.flow_match import FlowMatchScheduler
    a = make_arch(); a.cfg['future_video_mode'] = 'none'
    a.video_backbone._scheduler = FlowMatchScheduler(template='Wan'); a.init_training_schedulers(10)
    x = inputs(); seen = []
    def preprocess(**kw):
        seen.append(kw['num_frames'])
        return dict(context=x['context'], latents=x['latents'][:, :, :1].clone(),
                    first_frame_latents=x['latents'][:, :, :1].clone())
    monkeypatch.setattr(a.video_backbone, 'preprocess_input_for_inference', preprocess)
    monkeypatch.setattr(a.video_backbone, 'decode_video', lambda *args, **kw: pytest.fail('decode invoked'))
    result = a.generate([(1000.,1000.), (0.,0.)], 'move', num_frames=9, action_num_frames=33)
    assert seen == [1] and result['video'] is None
    assert result['normalized_actions'].shape == (32,112)


def test_stage2_action_only_config_validation():
    from hydra import compose, initialize_config_dir
    from pathlib import Path
    from openwam.train.utils.stage2 import validate_stage2_config
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[2]/'configs'), version_base=None):
        cfg = compose(config_name='goal/stage2_query_sample', overrides=['model.architecture.future_video_mode=none'])
    validate_stage2_config(cfg)
    cfg.model.architecture.future_video_mode = 'typo'
    with pytest.raises(ValueError, match='future_video_mode'): validate_stage2_config(cfg)


def test_action_only_query_memory_affects_actions_with_frozen_teacher():
    from test_query_stage2 import attach
    a = attach(); a.cfg['future_video_mode'] = 'none'
    x = inputs(); x['first_frame_latents'] = x['latents'][:, :, :1].clone()
    z = torch.randn(1,10,32, requires_grad=True)
    mask = torch.ones(1,10,dtype=torch.bool)
    first = a(**x, goal_latent=z, goal_token_valid=mask)
    other = a(**x, goal_latent=z.detach()+5, goal_token_valid=mask)
    assert first[0] is None and not torch.allclose(first[1], other[1])
    first[1].square().mean().backward()
    assert a.goal_adapter.projection.weight.grad.abs().sum() > 0
    assert z.grad is None and all(p.grad is None for p in a.goal_expert.parameters())


def test_mode_switch_preserves_strict_checkpoint_topology():
    a = make_arch(); b = make_arch(); b.cfg['future_video_mode'] = 'none'
    b.load_state_dict(a.state_dict(), strict=True)
    assert {k: v.shape for k,v in a.state_dict().items()} == {k: v.shape for k,v in b.state_dict().items()}


def test_stage2_fixed_history_zero_jitter_is_valid():
    from hydra import compose, initialize_config_dir
    from pathlib import Path
    from openwam.train.utils.stage2 import validate_stage2_config
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[2]/'configs'), version_base=None):
        cfg = compose(config_name='goal/stage2_query_sample')
    from omegaconf import open_dict
    with open_dict(cfg):
        cfg.goal.history = dict(enabled=True)
        cfg.model.query_history = dict(enabled=True)
        cfg.dataloader.history = dict(enabled=True, jitter_fraction=0.)
    validate_stage2_config(cfg)


def test_action_only_rejects_backbones_without_ti2v_current_observation_contract():
    a = make_arch(); a.cfg['future_video_mode'] = 'none'
    a.video_backbone = torch.nn.Linear(2,2)
    with pytest.raises(ValueError, match='TI2V'): _ = a.future_video_mode


def test_native_evaluator_publishes_action_only_without_video_or_goal_decoder(tmp_path, monkeypatch):
    import json
    import numpy as np
    from test_stage2_history import Tiny, sample, schedule
    from openwam.train import stage2_eval as evaluation
    class ActionOnly(Tiny):
        future_video_mode = 'none'
        def generate(self, *args, **kwargs):
            result = super().generate(*args, **kwargs)
            result['video'] = None
            self.goal_diagnostics.update(goal_decode_available=False,
                predicted_latent=torch.zeros(1,10,32), predicted_uv=torch.full((1,2,2),float('nan')))
            return result
    a = ActionOnly(); a.goal_expert.representation = 'query_latent'
    a.goal_expert.codec.decode = lambda *args: pytest.fail('goal decoder invoked')
    monkeypatch.setattr(evaluation, 'make_schedule', schedule)
    evaluation.evaluate_stage2(a, [sample()], tmp_path, step=0, seed=3, num_steps=1)
    folder = tmp_path/'eval_step_000000'
    report = json.loads((folder/'metrics.json').read_text())
    assert report['future_video_mode'] == 'none'
    assert report['aggregate']['action_mae']['valid_count'] == 2
    assert report['cases'][0]['metrics']['future_video_mae']['status'] == 'not_applicable'
    for key in ('future_video_mae', 'goal_image_mae', 'tcp_pixel_error'):
        assert report['cases'][0]['metrics'][key]['value'] is None
    assert (folder/'case_000/action_curves.svg').exists()
    assert not (folder/'case_000/future_gt_prediction.mp4').exists()
    assert not (folder/'case_000/goal_prediction.png').exists()
    assert 'not applicable' in (folder/'index.html').read_text()
    assert '<video ' not in (folder/'index.html').read_text()


def test_action_curves_explain_colors_and_masked_channels(tmp_path):
    import numpy as np
    from test_stage2_history import sample
    from openwam.train.stage2_eval import _curves
    _curves(tmp_path, np.zeros((32,112)), sample())
    svg = (tmp_path/'action_curves.svg').read_text()
    assert 'Prediction (red)' in svg and 'Valid GT (teal)' in svg
    assert 'No valid GT: excluded from metrics' in svg
    assert 'channel 0 (no valid GT)' in svg
    assert 'height="2560"' in svg and 'translate(0,40)' in svg
