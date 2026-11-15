"""Head RGB errors with a current-frame copy baseline and GT-defined regions."""
from __future__ import annotations

import math
import torch


def goal_image_metrics(prediction, current_rgb, target_rgb, *, image_valid,
                       target_uv, point_valid, change_threshold=.1, tcp_radius_px=24):
    """Return additive RGB sums/counts. Empty regions have null MAE.

    RGB inputs are uint8 [3,1,H,W], prediction is floating [3,1,H,W]
    in [0,1]. Changed pixels use mean-channel |current-GT| > threshold.
    TCP regions are the union of disks around valid GT points, with the
    normalized coordinates converted using W-1 and H-1.
    """
    prediction = torch.as_tensor(prediction).detach().cpu().double()
    current = torch.as_tensor(current_rgb).detach().cpu()
    target = torch.as_tensor(target_rgb).detach().cpu()
    if (prediction.ndim != 4 or prediction.shape[:2] != (3, 1)
            or current.shape != prediction.shape or target.shape != prediction.shape
            or current.dtype != torch.uint8 or target.dtype != torch.uint8):
        raise ValueError('expected equal [3,1,H,W] RGB with uint8 current and target')
    if not math.isfinite(change_threshold) or not 0 <= change_threshold <= 1:
        raise ValueError('change_threshold must lie in [0,1]')
    if not math.isfinite(tcp_radius_px) or tcp_radius_px < 0:
        raise ValueError('tcp_radius_px must be finite nonnegative')
    uv = torch.as_tensor(target_uv).detach().cpu().double()
    valid = torch.as_tensor(point_valid).detach().cpu()
    if uv.shape != (2, 2) or valid.shape != (2,) or valid.dtype != torch.bool:
        raise ValueError('expected two UV points and boolean point validity')
    if bool(image_valid) and not torch.isfinite(prediction).all():
        raise ValueError('prediction must be finite for a valid image')
    current, target = current.double() / 255, target.double() / 255
    height, width = prediction.shape[-2:]
    full = torch.full((height, width), bool(image_valid), dtype=torch.bool)
    copy_error = (current - target).abs()
    changed = (copy_error.mean((0, 1)) > change_threshold) & full
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
    tcp = torch.zeros_like(full)
    for point, enabled in zip(uv, valid):
        if enabled and torch.isfinite(point).all() and ((point >= 0) & (point <= 1)).all():
            x, y = point * torch.tensor([width - 1, height - 1])
            tcp |= (xx - x).square() + (yy - y).square() <= tcp_radius_px ** 2
    model_error = (prediction.clamp(0, 1) - target).abs()
    regions = {}
    for name, mask in dict(full=full, changed=changed, static=full & ~changed, tcp=tcp & full).items():
        denominator = int(mask.sum()) * 3
        model_sum = float(model_error[:, 0, mask].sum())
        copy_sum = float(copy_error[:, 0, mask].sum())
        regions[name] = dict(rgb_den=denominator, model_sum=model_sum, copy_sum=copy_sum,
                             model_mae=model_sum / denominator if denominator else None,
                             copy_mae=copy_sum / denominator if denominator else None,
                             gain_over_copy=(copy_sum - model_sum) / denominator if denominator else None)
    return dict(protocol='head-RGB0to1-current-copy-regions-v1',
                change_threshold=change_threshold, tcp_radius_px=tcp_radius_px, regions=regions)


def aggregate_goal_image_metrics(records):
    """Aggregate by RGB-value count, preserving region sums for later regrouping."""
    result = {}
    for name in ('full', 'changed', 'static', 'tcp'):
        model_sum = sum(r['regions'][name]['model_sum'] for r in records)
        copy_sum = sum(r['regions'][name]['copy_sum'] for r in records)
        denominator = sum(r['regions'][name]['rgb_den'] for r in records)
        result[name] = dict(rgb_den=denominator, model_sum=model_sum, copy_sum=copy_sum,
                            model_mae=model_sum / denominator if denominator else None,
                            copy_mae=copy_sum / denominator if denominator else None,
                            gain_over_copy=(copy_sum - model_sum) / denominator if denominator else None)
    return result
