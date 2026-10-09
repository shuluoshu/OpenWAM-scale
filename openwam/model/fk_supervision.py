"""Differentiable URDF FK supervision of denoised joint deltas against source GT."""

import math

import torch


def torch_fk(q, chain):
    """Batched FK in the declared base, including the fixed physical endpoint."""
    if q.shape[-1] != chain["dof"]:
        raise ValueError("FK joint width differs from declared chain")
    eye = torch.eye(4, device=q.device, dtype=q.dtype)
    transform = eye.expand(*q.shape[:-1], 4, 4)
    for joint in chain["joints"]:
        transform = transform @ torch.as_tensor(joint["fixed"], device=q.device, dtype=q.dtype)
        if joint["kind"] == "fixed":
            continue
        value = q[..., joint["index"]]
        axis = torch.as_tensor(joint["axis"], device=q.device, dtype=q.dtype)
        motion = eye.expand(*q.shape[:-1], 4, 4).clone()
        if joint["kind"] == "prismatic":
            motion[..., :3, 3] = value[..., None] * axis
        else:
            x, y, z = axis.unbind()
            zero = x * 0
            skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)
            motion[..., :3, :3] = (
                eye[:3, :3] + value.sin()[..., None, None] * skew + (1 - value.cos())[..., None, None] * (skew @ skew)
            )
        transform = transform @ motion
    return transform @ torch.as_tensor(chain["tool"], device=q.device, dtype=q.dtype)


def fk_gt_loss(clean_actions, payloads, action_is_pad=None):
    """Return raw position(m²), rotation chordal loss, and valid arm/time count.

    A sample contributes its mean over valid arm/time endpoints. Unsupported
    samples contribute zero to the batch mean, matching the action-loss policy.
    Only predicted joint slots receive gradients; GT and constants are detached.
    """
    if clean_actions.ndim != 3 or clean_actions.shape[-1] not in (94, 112):
        raise ValueError("FK supervision requires B,T,94/112 relative-joint actions")
    if payloads is None or len(payloads) != clean_actions.shape[0]:
        raise ValueError("FK supervision needs one payload (or None) per sample")
    if action_is_pad is not None and action_is_pad.shape != clean_actions.shape:
        raise ValueError("FK requires a per-time/per-channel action padding mask")
    # Empty differentiable sum cannot be contaminated by masked NaN predictions.
    zero = clean_actions[..., 80:94].reshape(-1)[:0].sum().float()
    positions, rotations, count = [], [], 0
    with torch.autocast(device_type=clean_actions.device.type, enabled=False):
        clean = clean_actions.float()
        for batch, payload in enumerate(payloads):
            p_sum, r_sum, n = zero, zero, 0
            for arm in (payload or {}).get("arms", []):

                def constant(key, dtype=torch.float32):
                    return torch.as_tensor(arm[key], device=clean.device, dtype=dtype).detach()

                slots = arm["slots"]
                if slots != list(range(slots[0], slots[0] + arm["chain"]["dof"])) or slots[0] not in (80, 87):
                    raise ValueError("FK slots must match a contiguous left/right joint block")
                valid = constant("valid", torch.bool)
                if valid.shape != (clean.shape[1],):
                    raise ValueError("FK mask horizon mismatch")
                if action_is_pad is not None:
                    valid = valid & ~action_is_pad[batch, :, slots].to(clean.device).any(-1)
                if not valid.any():
                    continue
                q0 = constant("q_anchor")
                delta = clean[batch, :, slots][valid] * constant("scale") + constant("offset")
                q = q0 + delta
                gt = constant("gt_relative")[valid]
                if not torch.isfinite(q).all() or not torch.isfinite(gt).all():
                    raise ValueError("nonfinite valid FK prediction/GT")
                t0 = torch_fk(q0, arm["chain"])
                future = torch_fk(q, arm["chain"])
                r0 = t0[:3, :3].T
                position = (r0 @ (future[..., :3, 3] - t0[:3, 3])[..., None])[..., 0]
                rotation = r0 @ future[..., :3, :3]
                p_sum = p_sum + (position - gt[..., :3, 3]).square().sum()
                r_sum = r_sum + 0.5 * (rotation - gt[..., :3, :3]).square().sum()
                n += int(valid.sum())
            positions.append(p_sum / max(n, 1))
            rotations.append(r_sum / max(n, 1))
            count += n
    return torch.stack(positions).mean(), torch.stack(rotations).mean(), count


def validate_fk_weights(position, rotation, lambda_action):
    if not all(math.isfinite(v) and v >= 0 for v in (position, rotation)):
        raise ValueError("FK loss weights must be finite and nonnegative")
    if (position or rotation) and lambda_action <= 0:
        raise ValueError("FK supervision requires lambda_action > 0")


def add_fk_losses(result, noisy, velocity, sigma, payloads, action_is_pad, position_weight, rotation_weight):
    """Add physical GT losses from x0_hat=x_sigma-sigma*v_hat, not from velocity."""
    if not (position_weight or rotation_weight):
        return result
    if noisy is None or velocity is None or sigma is None:
        raise ValueError("FK supervision requires action flow predictions")
    clean = noisy.float() - sigma.float() * velocity.float()
    pos, rot, count = fk_gt_loss(clean, payloads if payloads is not None else [None] * len(clean), action_is_pad)
    result["loss"] = result["loss"] + position_weight * pos + rotation_weight * rot
    result.update(
        loss_fk_position=(position_weight * pos).detach(),
        loss_fk_rotation=(rotation_weight * rot).detach(),
        fk_valid_count=torch.tensor(float(count), device=clean.device),
    )
    return result
