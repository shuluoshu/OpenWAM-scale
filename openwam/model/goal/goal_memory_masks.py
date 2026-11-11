"""Explicit M/C/G/Y routing. Boolean True means an allowed attention edge."""
import torch


def view_token_mask(view_valid, rows, columns):
    """Three-view canvas: top 2/3 head, bottom left/right wrists; empty safe."""
    if view_valid.dtype != torch.bool or view_valid.shape[-1] != 3:
        raise ValueError('view validity must be boolean [...,3]')
    if rows % 3 or columns % 2:
        raise ValueError('three-view patch grid requires rows divisible by 3 and even columns')
    grid = torch.empty((*view_valid.shape[:-1], rows, columns), dtype=torch.bool, device=view_valid.device)
    boundary = rows * 2 // 3
    grid[..., :boundary, :] = view_valid[..., 0, None, None]
    grid[..., boundary:, :columns//2] = view_valid[..., 1, None, None]
    grid[..., boundary:, columns//2:] = view_valid[..., 2, None, None]
    return grid.flatten(-2)


def group_attention_mask(memory_slices, current_slice, goal_slice, target_slice, key_valid):
    n = key_valid.shape[1]
    allowed = torch.zeros(n, n, dtype=torch.bool, device=key_valid.device)
    for group in memory_slices:
        allowed[group, group] = True
    allowed[current_slice, current_slice] = True
    allowed[goal_slice, :goal_slice.stop] = True
    allowed[target_slice, current_slice.start:target_slice.stop] = True
    return allowed[None, None] & key_valid[:, None, None, :]
