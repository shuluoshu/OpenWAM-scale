"""CPU-only checks for bounded action checkpoint warm starts."""
import ast
import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

# Load the pure helper and real checkpoint method without optional backbones.
MODEL_DIR = Path(__file__).parents[1] / 'openwam/model'
spec = importlib.util.spec_from_file_location('checkpoint_utils', MODEL_DIR / 'checkpoint_utils.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
expand_action_state_dict = helper.expand_action_state_dict
base_tree = ast.parse((MODEL_DIR / 'architectures/base.py').read_text())
base_class = next(n for n in base_tree.body if isinstance(n, ast.ClassDef) and n.name == 'BaseWAMArchitecture')
load_method = next(n for n in base_class.body if isinstance(n, ast.FunctionDef) and n.name == 'load_checkpoint')
namespace = {'VLM_STATE_DICT_PREFIX': 'vlm_backbone.'}
exec(compile(ast.Module(body=[load_method], type_ignores=[]), str(MODEL_DIR / 'architectures/base.py'), 'exec'), namespace)
load_checkpoint = namespace['load_checkpoint']

AXES = {
    'action_backbone.action_encoder.weight': 1,
    'action_backbone.action_decoder.weight': 0,
    'action_backbone.action_decoder.bias': 0,
    'proprio_encoder.weight': 1,
}


def projections(dim):
    model = nn.Module()
    model.action_backbone = nn.Module()
    model.action_backbone.action_encoder = nn.Linear(dim, 3)
    model.action_backbone.action_decoder = nn.Linear(3, dim)
    model.proprio_encoder = nn.Linear(dim, 3)
    return model


def expansion(state, target_shapes, **kwargs):
    return expand_action_state_dict(state, target_shapes, **kwargs)

@pytest.mark.parametrize("source_dim,target_dim", [(80, 94), (80, 112), (94, 112)])
def test_expansion_preserves_values_and_zeros_appended_coordinates(source_dim, target_dim):
    original = projections(source_dim).state_dict()
    target = projections(target_dim).state_dict()
    expanded = expansion(original, {k: v.shape for k, v in target.items()}, source_dim=source_dim, target_dim=target_dim)
    for key, value in original.items():
        if key not in AXES:
            assert torch.equal(expanded[key], value)
            continue
        axis = AXES[key]
        assert torch.equal(expanded[key].narrow(axis, 0, source_dim), value)
        assert torch.count_nonzero(expanded[key].narrow(axis, source_dim, target_dim - source_dim)) == 0
        assert expanded[key].dtype == value.dtype
        assert original[key].shape[axis] == source_dim


@pytest.mark.parametrize('dim', [79, 81, 95])
def test_other_migrations_rejected(dim):
    with pytest.raises(ValueError, match='80.*94'):
        expansion(projections(dim).state_dict(), {k: v.shape for k, v in projections(94).state_dict().items()})


def test_unrelated_shape_mismatch_rejected():
    source = projections(80).state_dict()
    source['other.weight'] = torch.zeros(2, 3)
    target = {k: v.shape for k, v in projections(94).state_dict().items()}
    target['other.weight'] = (3, 3)
    with pytest.raises(ValueError, match='other.weight'):
        expansion(source, target)


def test_non_action_axis_mismatch_rejected():
    source = projections(80).state_dict()
    source['action_backbone.action_encoder.weight'] = torch.zeros(4, 80)
    with pytest.raises(ValueError, match='action_encoder.weight'):
        expansion(source, {k: v.shape for k, v in projections(94).state_dict().items()})


def test_dimension_arguments_cannot_enable_other_migrations():
    with pytest.raises(ValueError, match='80.*94'):
        expansion({}, {}, source_dim=79, target_dim=94)


def test_load_remains_strict_without_opt_in(tmp_path):
    path = str(tmp_path / 'weights.safetensors')
    save_file(projections(80).state_dict(), path)
    with pytest.raises(RuntimeError, match='size mismatch'):
        load_checkpoint(projections(94), path)


def test_explicit_load_expands(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "openwam.model.checkpoint_utils", helper)
    source = projections(80).state_dict()
    path = str(tmp_path / 'weights.safetensors')
    save_file(source, path)
    target = projections(94)
    load_checkpoint(target, path, expand_action_dim_from_checkpoint=True)
    for key, axis in AXES.items():
        assert torch.equal(target.state_dict()[key].narrow(axis, 0, 80), source[key])
        assert torch.count_nonzero(target.state_dict()[key].narrow(axis, 80, 14)) == 0


@pytest.mark.parametrize("source_dim,target_dim", [(80, 112), (94, 112), (80, 80), (94, 94), (112, 112)])
def test_load_infers_projection_dimensions(tmp_path, monkeypatch, source_dim, target_dim):
    monkeypatch.setitem(sys.modules, "openwam.model.checkpoint_utils", helper)
    source = projections(source_dim).state_dict()
    path = str(tmp_path / 'weights.safetensors')
    save_file(source, path)
    target = projections(target_dim)
    load_checkpoint(target, path, expand_action_dim_from_checkpoint=True)
    for key, axis in AXES.items():
        actual = target.state_dict()[key]
        assert torch.equal(actual.narrow(axis, 0, source_dim), source[key])
        assert torch.count_nonzero(actual.narrow(axis, source_dim, target_dim - source_dim)) == 0


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("side", ["source", "target"])
def test_load_rejects_inconsistent_projection_dimensions(tmp_path, monkeypatch, strict, side):
    monkeypatch.setitem(sys.modules, "openwam.model.checkpoint_utils", helper)
    source = projections(80).state_dict()
    target = projections(94)
    if side == "source":
        source['proprio_encoder.weight'] = torch.zeros(3, 94)
    else:
        target.proprio_encoder = nn.Linear(80, 3)
    path = str(tmp_path / 'weights.safetensors')
    save_file(source, path)
    with pytest.raises(ValueError, match="[Ii]nconsistent"):
        load_checkpoint(target, path, strict=strict, expand_action_dim_from_checkpoint=True)


@pytest.mark.parametrize("source_dim,target_dim", [(112, 94), (94, 80), (79, 112), (80, 113)])
def test_load_rejects_unapproved_migrations(tmp_path, monkeypatch, source_dim, target_dim):
    monkeypatch.setitem(sys.modules, "openwam.model.checkpoint_utils", helper)
    path = str(tmp_path / 'weights.safetensors')
    save_file(projections(source_dim).state_dict(), path)
    with pytest.raises(ValueError, match="supports only"):
        load_checkpoint(projections(target_dim), path, strict=False, expand_action_dim_from_checkpoint=True)


@pytest.mark.parametrize("strict", [False, True])
def test_load_expansion_rejects_unrelated_shapes_even_non_strict(tmp_path, monkeypatch, strict):
    monkeypatch.setitem(sys.modules, "openwam.model.checkpoint_utils", helper)
    source = projections(94).state_dict()
    source['proprio_encoder.bias'] = torch.zeros(4)
    path = str(tmp_path / 'weights.safetensors')
    save_file(source, path)
    with pytest.raises(ValueError, match="proprio_encoder.bias"):
        load_checkpoint(projections(112), path, strict=strict, expand_action_dim_from_checkpoint=True)


@pytest.mark.parametrize("invalid_key", ["missing", "unexpected"])
def test_load_expansion_preserves_strict_key_validation(tmp_path, monkeypatch, invalid_key):
    monkeypatch.setitem(sys.modules, "openwam.model.checkpoint_utils", helper)
    source = projections(94).state_dict()
    if invalid_key == "missing":
        del source['proprio_encoder.weight']
    else:
        source['other.weight'] = torch.zeros(1)
    path = str(tmp_path / 'weights.safetensors')
    save_file(source, path)
    with pytest.raises(RuntimeError, match="Strict load failed"):
        load_checkpoint(projections(112), path, expand_action_dim_from_checkpoint=True)
