"""Narrow, explicit checkpoint migrations for training warm starts."""
from collections.abc import Mapping

import torch

_ACTION_AXES = {
    'action_backbone.action_encoder.weight': 1,
    'action_backbone.action_decoder.weight': 0,
    'action_backbone.action_decoder.bias': 0,
    'proprio_encoder.weight': 1,
}


def infer_action_dimensions(
    state_dict: Mapping[str, torch.Tensor],
    target_shapes: Mapping[str, tuple],
) -> tuple[int, int]:
    """Infer consistent source/target widths from the known projection axes."""
    def dimension(shapes, label):
        widths = set()
        for key, axis in _ACTION_AXES.items():
            if key not in shapes:
                continue
            shape = tuple(shapes[key])
            expected_rank = 1 if key.endswith('.bias') else 2
            if len(shape) != expected_rank:
                raise ValueError(f'Unsupported checkpoint projection shape for {key}: {shape}')
            widths.add(shape[axis])
        if len(widths) > 1:
            raise ValueError(f'Inconsistent {label} action/proprio projection dimensions: {sorted(widths)}')
        if not widths:
            raise ValueError(f'Cannot infer {label} action dimension: no known projections')
        return widths.pop()

    return dimension({key: value.shape for key, value in state_dict.items()}, 'source'), dimension(target_shapes, 'target')


def expand_action_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    target_shapes: Mapping[str, tuple],
    source_dim: int = 80,
    target_dim: int = 94,
) -> dict[str, torch.Tensor]:
    """Zero-pad known projections for 80 -> 94, 80 -> 112, or 94 -> 112.

    Matching tensors pass through unchanged. Missing/unexpected keys remain for
    the architecture's strict loader to validate. Every other shape mismatch
    raises, even when the caller requests a non-strict state dictionary load.
    The input dictionary and tensors are never modified.
    """
    if source_dim != target_dim and (source_dim, target_dim) not in {(80, 94), (80, 112), (94, 112)}:
        raise ValueError('Checkpoint action expansion supports only 80 -> 94, 80 -> 112, or 94 -> 112')
    expanded = dict(state_dict)
    for key, value in state_dict.items():
        if key not in target_shapes:
            continue
        target_shape = tuple(target_shapes[key])
        if tuple(value.shape) == target_shape:
            continue
        axis = _ACTION_AXES.get(key)
        expected_rank = 1 if key == 'action_backbone.action_decoder.bias' else 2
        valid = axis is not None and value.ndim == expected_rank and len(target_shape) == expected_rank
        if valid:
            valid = value.shape[axis] == source_dim and target_shape[axis] == target_dim
            valid = valid and all(value.shape[i] == target_shape[i] for i in range(value.ndim) if i != axis)
        if not valid:
            raise ValueError(
                f'Unsupported checkpoint shape mismatch for {key}: {tuple(value.shape)} -> '
                f'{target_shape}; only known action/proprio projections may expand '
                f'{source_dim} -> {target_dim} (supported: 80 -> 94, 80 -> 112, 94 -> 112)'
            )
        padded = value.new_zeros(target_shape)
        padded.narrow(axis, 0, source_dim).copy_(value)
        expanded[key] = padded
    return expanded
