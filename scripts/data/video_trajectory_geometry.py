"""Pose representation roundtrips for GT videos, before any normalization.

Inputs are world poses (..., arms, 4, 4), anchor poses (arms, 4, 4), and one
world-to-head camera transform. Returned matrices encode delta rotations and
delta translations, not absolute transforms from the world origin. Callers
must carry source validity masks separately; these functions do not infer TCP.
"""
import numpy as np


def future_indices(current, count, horizon=32):
    """Anchor plus real future states; never pad a video with repeated targets."""
    if not 0 <= current < count or horizon < 1:
        raise ValueError('current must address a real row and horizon must be positive')
    return np.arange(current, min(count, current + horizon + 1))


def encode_relative(poses, anchor, world_to_camera, mode):
    """Encode current-endpoint-local or current-head-reference pose deltas."""
    poses, anchor = np.asarray(poses), np.asarray(anchor)
    encoded = np.broadcast_to(np.eye(4), poses.shape).copy()
    delta = poses[..., :3, 3] - anchor[..., :3, 3]
    anchor_rotation_inverse = np.swapaxes(anchor[..., :3, :3], -1, -2)
    if mode == 'local':
        encoded[..., :3, 3] = np.einsum('sij,...sj->...si', anchor_rotation_inverse, delta)
        encoded[..., :3, :3] = anchor_rotation_inverse @ poses[..., :3, :3]
    elif mode == 'head':
        rotation = np.asarray(world_to_camera)[:3, :3]
        encoded[..., :3, 3] = delta @ rotation.T
        encoded[..., :3, :3] = rotation @ poses[..., :3, :3] @ anchor_rotation_inverse @ rotation.T
    else:
        raise ValueError('mode must be local or head')
    return encoded


def decode_relative(encoded, anchor, world_to_camera, mode):
    """Restore absolute world poses, adding the endpoint origin exactly once."""
    encoded, anchor = np.asarray(encoded), np.asarray(anchor)
    restored = np.broadcast_to(np.eye(4), encoded.shape).copy()
    if mode == 'local':
        restored[..., :3, :3] = anchor[..., :3, :3] @ encoded[..., :3, :3]
        restored[..., :3, 3] = np.einsum('sij,...sj->...si', anchor[..., :3, :3], encoded[..., :3, 3])
    elif mode == 'head':
        rotation = np.asarray(world_to_camera)[:3, :3]
        restored[..., :3, :3] = rotation.T @ encoded[..., :3, :3] @ rotation @ anchor[..., :3, :3]
        restored[..., :3, 3] = encoded[..., :3, 3] @ rotation
    else:
        raise ValueError('mode must be local or head')
    restored[..., :3, 3] += anchor[..., :3, 3]
    return restored
