"""The expansion flag belongs exclusively to explicit finetune loading."""
import sys
from types import ModuleType, SimpleNamespace

import pytest
from omegaconf import OmegaConf

from openwam.train.utils import ckpt_model_loader as cml


def _live_cfg(**patch):
    cfg = OmegaConf.create({'model': {
        'video_backbone': {'name': 'wan22_ti2v_5b'},
        'architecture': {'framework': 'dual_system', 'variant': 'joint_self_attn', 'action_dim': 80, 'state_dim': 80}
    }, 'training': {}})
    for key, value in patch.items():
        OmegaConf.update(cfg, 'model.' + key.replace('__', '.'), value)
    return cfg


def _write_ckpt_cfg(tmp_path):
    OmegaConf.save(_live_cfg(), tmp_path / 'config.yaml')
    return str(tmp_path)


@pytest.mark.parametrize('enabled', [None, False, True])
def test_only_explicit_finetune_flag_reaches_checkpoint_loader(tmp_path, monkeypatch, enabled):
    cfg = _live_cfg(architecture__action_dim=94, architecture__state_dim=94)
    if enabled is not None:
        OmegaConf.update(cfg, 'training.expand_action_dim_from_checkpoint', enabled, force_add=True)
    calls = []

    class Built:
        def load_checkpoint(self, path, **kwargs):
            calls.append((path, kwargs))

    model_module = ModuleType('openwam.model')
    model_module.build_architecture = lambda *a: Built()
    model_module.resolve_architecture_config = lambda cfg: SimpleNamespace(
        registry_name='dual_system', params={'video_backbone': {}}
    )
    monkeypatch.setitem(sys.modules, 'openwam.model', model_module)
    monkeypatch.setattr(cml, 'find_latest_weights', lambda _: 'weights.safetensors')
    cml.build_architecture_from_ckpt_dir(_write_ckpt_cfg(tmp_path), weights_required=True, override_cfg=cfg)
    assert calls == [('weights.safetensors', {'expand_action_dim_from_checkpoint': True} if enabled else {})]


def test_resume_never_loads_or_expands_safetensors(tmp_path, monkeypatch):
    cfg = _live_cfg()
    OmegaConf.update(cfg, 'training.expand_action_dim_from_checkpoint', True, force_add=True)

    class Built:
        def load_checkpoint(self, *args, **kwargs):
            pytest.fail('Resume must restore accelerator state, without expansion')

    model_module = ModuleType('openwam.model')
    model_module.build_architecture = lambda *a: Built()
    model_module.resolve_architecture_config = lambda cfg: SimpleNamespace(
        registry_name='dual_system', params={'video_backbone': {}}
    )
    monkeypatch.setitem(sys.modules, 'openwam.model', model_module)
    cml.build_architecture_from_ckpt_dir(_write_ckpt_cfg(tmp_path), weights_required=False, override_cfg=cfg)
