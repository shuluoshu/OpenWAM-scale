from types import SimpleNamespace

import pytest
import torch
from torch import nn

from openwam.model.action_backbone.scheduler import ActionScheduler
from openwam.model.architectures.base import BaseWAMArchitecture
from openwam.train.openwam_trainer import OpenWAMTrainer
from openwam.train.utils.training_utils import reduce_step_metrics
from tests.model.test_fk_supervision import chain as arm_chain_fixture
from tests.model.test_fk_supervision import payload

chain = arm_chain_fixture  # pytest fixture shared with analytic FK tests


class TinyFlow(nn.Module):
    compute_loss = BaseWAMArchitecture.compute_loss
    _compute_video_loss = BaseWAMArchitecture._compute_video_loss
    _compute_action_loss = BaseWAMArchitecture._compute_action_loss

    def __init__(self):
        super().__init__()
        self.clean = nn.Parameter(torch.zeros(1, 2, 112))
        scheduler = ActionScheduler()
        scheduler.timesteps = torch.tensor([500.0])
        scheduler.sigmas = torch.tensor([0.5])
        scheduler.linear_timesteps_weights = torch.ones(1)
        self.action_backbone = SimpleNamespace(scheduler=scheduler)
        self.video_backbone = SimpleNamespace(scheduler=scheduler)
        self.device = torch.device("cpu")
        self.dtype = torch.float32

    def forward(self, noisy, action_timestep, **inputs):
        assert "fk_supervision" not in inputs
        return torch.zeros_like(inputs["latents"]), (noisy - self.clean) / 0.5


def test_actual_compute_loss_includes_fk_and_backpropagates(chain):
    model = TinyFlow()
    result = model.compute_loss(
        actions=torch.zeros_like(model.clean),
        input_latents=torch.zeros(1, 1, 2, 1, 1),
        fk_supervision=[payload(chain)],
        lambda_fk_position=2.0,
        lambda_fk_rotation=3.0,
    )
    assert result["fk_valid_count"] == 2
    assert result["loss_fk_position"] > 0 and result["loss_fk_rotation"] > 0
    torch.testing.assert_close(
        result["loss"].detach(),
        result["loss_video"] + result["loss_action"] + result["loss_fk_position"] + result["loss_fk_rotation"],
    )
    result["loss"].backward()
    assert model.clean.grad[0, :, 80].abs().sum() > 0


def test_trainer_forwards_weights_and_keeps_metrics():
    trainer = OpenWAMTrainer.__new__(OpenWAMTrainer)
    trainer.lambda_video = trainer.lambda_action = 1.0
    trainer.lambda_fk_position = 2.0
    trainer.lambda_fk_rotation = 3.0

    def compute(**kw):
        assert kw["lambda_fk_position"] == 2 and kw["lambda_fk_rotation"] == 3
        return {
            "loss": torch.tensor(10.0),
            "loss_video": torch.tensor(1.0),
            "loss_action": torch.tensor(2.0),
            "loss_fk_position": torch.tensor(3.0),
            "loss_fk_rotation": torch.tensor(4.0),
            "fk_valid_count": torch.tensor(8.0),
        }

    trainer.architecture = SimpleNamespace(
        prepare_inputs=lambda batch: {"actions": torch.zeros(1)}, compute_loss=compute
    )
    losses = trainer.compute_loss([{}])
    metrics = reduce_step_metrics(None, losses, torch.tensor(0.0))
    assert metrics["loss_fk_position"] == 3 and metrics["loss_fk_rotation"] == 4 and metrics["fk_valid_count"] == 8


def test_fk_weights_without_action_fail_explicitly(chain):
    with pytest.raises(ValueError, match="lambda_action"):
        TinyFlow().compute_loss(
            actions=None,
            input_latents=torch.zeros(1, 1, 2, 1, 1),
            lambda_action=0,
            lambda_fk_position=1,
            fk_supervision=[payload(chain)],
        )


def test_prepare_inputs_preserves_mixed_source_payloads(chain):
    from tests.test_prepare_inputs_proprio_mask import _make_sample, _StubArch

    arch = _StubArch(state_dim=112)
    samples = [_make_sample(state_dim=112), _make_sample(state_dim=112)]
    data = payload(chain)
    samples[0]["fk_supervision"] = data
    inputs = arch.prepare_inputs(samples)
    assert inputs["fk_supervision"][0] is data and inputs["fk_supervision"][1] is None


def test_missing_profile_yields_zero_fk_but_keeps_other_losses():
    model = TinyFlow()
    result = model.compute_loss(
        actions=torch.zeros_like(model.clean),
        input_latents=torch.zeros(1, 1, 2, 1, 1),
        lambda_fk_position=1.0,
        lambda_fk_rotation=1.0,
    )
    assert result["fk_valid_count"] == 0
    assert result["loss_fk_position"] == 0 and result["loss_fk_rotation"] == 0
    torch.testing.assert_close(result["loss"].detach(), result["loss_video"] + result["loss_action"])


def test_enabled_fk_cannot_silently_accept_missing_action_prediction(chain):
    model = TinyFlow()
    model.forward = lambda *args, **kw: (torch.zeros_like(kw["latents"]), None)
    with pytest.raises(ValueError, match="action prediction"):
        model.compute_loss(
            actions=torch.zeros_like(model.clean),
            input_latents=torch.zeros(1, 1, 2, 1, 1),
            lambda_fk_position=1.0,
            fk_supervision=[payload(chain)],
        )


def test_idm_override_also_applies_fk_loss(chain):
    from tests.test_dual_system_idm import _CapturePrepareVideoBackbone, _make_idm_with_video

    vb = _CapturePrepareVideoBackbone()
    arch = _make_idm_with_video(vb, action_dim=112)
    arch.init_training_schedulers(10)
    result = arch.compute_loss(
        input_latents=torch.randn(1, 1, 1, 1, 1),
        context=torch.randn(1, 2, 8),
        context_mask=torch.ones(1, 2, dtype=torch.bool),
        actions=torch.zeros(1, 2, 112),
        fk_supervision=[payload(chain)],
        lambda_fk_position=1.0,
        lambda_fk_rotation=1.0,
    )
    assert result["fk_valid_count"] == 2
    assert torch.isfinite(result["loss_fk_position"] + result["loss_fk_rotation"])
    torch.testing.assert_close(
        result["loss"].detach(),
        result["loss_video"] + result["loss_action"] + result["loss_fk_position"] + result["loss_fk_rotation"],
    )
    result["loss"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in arch.action_backbone.parameters())
