"""Architecture-side helpers shared across WAM frameworks.

Bridge-layer resolution, latent-mask downsampling, and the video
tokens-per-frame derivation used by the MoT drivers and the SingleSystem
mask builder.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import torch

if TYPE_CHECKING:
    from openwam.model.video_backbone.base import BlockLoopState

# Wan VAE encodes every 4 video frames into 1 latent time step.
VAE_TEMPORAL_FACTOR = 4


def compute_video_tokens_per_frame(vstate: "BlockLoopState", driver_name: str) -> int:
    """Derive video tokens-per-frame from the spatial dims populated on ``vstate``.

    Used by every MoT driver to build the v↔v block of the joint attention mask.
    The video backbone's ``prepare()`` must populate ``grid_height`` and
    ``grid_width`` on the ``BlockLoopState``; if not, raise a clear error
    attributable to the calling driver via ``driver_name``.
    """
    h = int(getattr(vstate, "grid_height", 0))
    w = int(getattr(vstate, "grid_width", 0))
    if h <= 0 or w <= 0:
        raise ValueError(
            f"{driver_name}: cannot derive video_tokens_per_frame from vstate "
            f"(grid_height={h}, grid_width={w}). The video backbone's prepare() must populate them."
        )
    return h * w


def downsample_video_mask_to_latent(
    video_is_pad: torch.Tensor,
    *,
    temporal_factor: int = VAE_TEMPORAL_FACTOR,
    skip_first: bool = True,
    strict: bool = False,
) -> torch.Tensor:
    """Downsample frame-level padding mask to VAE latent temporal dimension.

    Wan2pt1-family causal VAEs encode frame 0 into latent[0] alone, then
    group the remaining tail frames by ``temporal_factor``. A latent step is
    padded only if ALL frames in the group are padded by default. Stage2
    uses ``strict=True`` to exclude any group containing padded RGB targets.

    Modes:
      - ``skip_first=True`` (FastWAM / Wan TI2V): latent[0] is the conditioning
        frame and excluded from loss; the returned mask covers tail latent
        steps only (shape ``T_latent_tail = ceil((T_video - 1) / k)``). This
        matches the Wan loss path which trims pred/target via ``[:, :, 1:]``.
      - ``skip_first=False`` (Cosmos T2V, no first-frame conditioning): all
        latents — including latent[0] — are predicted, so the mask must
        include frame 0 too (shape ``T_latent = 1 + T_latent_tail``).

    Args:
        video_is_pad: (..., T_video) bool, True=padded.
        temporal_factor: VAE temporal compression factor for the tail.
        skip_first: whether latent[0] is excluded from the loss (default True
            for Wan compatibility).

    Returns:
        (..., T_latent_out) bool mask, where ``T_latent_out`` is either
        ``T_latent_tail`` (skip_first=True) or ``1 + T_latent_tail``.
    """
    T = video_is_pad.shape[-1]
    leading_shape = video_is_pad.shape[:-1]
    if T == 0:
        return video_is_pad

    first_latent_mask = video_is_pad[..., 0:1]  # (..., 1)

    if T == 1:
        if skip_first:
            return torch.zeros((*leading_shape, 0), dtype=torch.bool, device=video_is_pad.device)
        return first_latent_mask

    tail_is_pad = video_is_pad[..., 1:]
    T_tail = tail_is_pad.shape[-1]
    pad_len = (temporal_factor - T_tail % temporal_factor) % temporal_factor
    if pad_len > 0:
        pad_block = torch.ones((*leading_shape, pad_len), dtype=torch.bool, device=tail_is_pad.device)
        tail_is_pad = torch.cat([tail_is_pad, pad_block], dim=-1)
    grouped = tail_is_pad.view(*tail_is_pad.shape[:-1], -1, temporal_factor)
    tail_latent_mask = grouped.any(dim=-1) if strict else grouped.all(dim=-1)

    if skip_first:
        return tail_latent_mask
    return torch.cat([first_latent_mask, tail_latent_mask], dim=-1)


def stage2_latent_view_mask(video_valid, view_valid, latent_hw, *,
                            temporal_factor, skip_first):
    """Valid latent cells for the fixed head/left-wrist/right-wrist L canvas.

    Source camera availability is supervision metadata, separate from observation
    view-drop augmentation. The 384x320 canvas has head rows0:256, then two
    128x160 wrist views. Tail compression groups containing any unavailable RGB
    frame are excluded. This does not estimate the VAE's spatial receptive field.
    """
    if (video_valid.dtype != torch.bool or video_valid.ndim != 1
            or view_valid.dtype != torch.bool
            or view_valid.shape != (video_valid.shape[0], 3)):
        raise ValueError('Stage2 video/view validity must be bool [T] and [T,3]')
    height, width = latent_hw
    if height <= 0 or width <= 0 or height % 3 or width % 2:
        raise ValueError('Stage2 latent grid must align with the fixed L canvas')
    if temporal_factor < 1:
        raise ValueError('temporal_factor must be positive')
    source_valid = view_valid.to(video_valid.device) & video_valid[:, None]
    per_view = ~downsample_video_mask_to_latent(~source_valid.T,
        temporal_factor=temporal_factor, skip_first=skip_first, strict=True)
    mask = torch.zeros(per_view.shape[1], height, width,
                       dtype=torch.bool, device=video_valid.device)
    head_end, wrist_mid = height * 2 // 3, width // 2
    mask[:, :head_end] = per_view[0, :, None, None]
    mask[:, head_end:, :wrist_mid] = per_view[1, :, None, None]
    mask[:, head_end:, wrist_mid:] = per_view[2, :, None, None]
    return mask


def resolve_bridge_layers(cfg: Any, *, num_layers: Optional[int] = None) -> tuple:
    """Parse ``bridge_layers`` indices from an architecture config.

    Two input modes:
      - Explicit: ``cfg.bridge_layers`` is a list / tuple / comma-separated str.
      - Interval: ``cfg.bridge_layers`` is None and ``cfg.bridge_interval`` is set;
        the indices are ``range(0, num_layers, bridge_interval)``.

    ``num_layers`` should be passed by the caller (typically the video backbone's
    ``num_layers``). It falls back to ``cfg.num_dit_layers`` for backward
    compatibility.

    The output is always sorted with no duplicates.
    """
    bl_raw = cfg.get("bridge_layers", None) if isinstance(cfg, dict) else getattr(cfg, "bridge_layers", None)

    if bl_raw is None:
        interval_raw = (
            cfg.get("bridge_interval", None) if isinstance(cfg, dict) else getattr(cfg, "bridge_interval", None)
        )
        if interval_raw is None:
            raise ValueError("bridge_layers is null but bridge_interval is not set")

        if num_layers is None:
            num_layers = (
                cfg.get("num_dit_layers", None) if isinstance(cfg, dict) else getattr(cfg, "num_dit_layers", None)
            )
        if num_layers is None:
            raise ValueError("num_layers must be provided (or num_dit_layers set on cfg) when bridge_layers is null")
        num_layers = int(num_layers)

        interval = int(interval_raw)
        assert interval >= 1, f"bridge_interval must be >= 1, got {interval}"
        bl = tuple(range(0, num_layers, interval))
    elif isinstance(bl_raw, str):
        bl = tuple(int(x) for x in bl_raw.split(","))
    elif isinstance(bl_raw, tuple):
        bl = bl_raw
    else:
        bl = tuple(bl_raw)

    bl = tuple(sorted(bl))
    assert len(set(bl)) == len(bl), f"bridge_layers must be unique, got {bl}"
    return bl


__all__ = [
    "VAE_TEMPORAL_FACTOR",
    "compute_video_tokens_per_frame",
    "downsample_video_mask_to_latent",
    "resolve_bridge_layers",
]
