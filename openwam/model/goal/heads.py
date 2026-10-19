"""Ordered left/right heatmaps read from predicted clean goal latents."""
import torch
from torch import nn
import torch.nn.functional as F


class GoalPointHead(nn.Module):
    def __init__(self, hidden_dim=64, heatmap_size=(64,80)):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heatmap_size = tuple(heatmap_size)
        self.net = nn.Sequential(nn.Conv2d(48,hidden_dim,3,padding=1), nn.SiLU(), nn.Conv2d(hidden_dim,2,1))

    def forward(self, clean_latent):
        return F.interpolate(self.net(clean_latent[:,:,0]),size=self.heatmap_size,mode='bilinear',align_corners=False)


def point_mask(targets):
    return (targets.image_valid.bool()[:,None] & targets.point_valid.bool() & targets.in_frame.bool()
            & torch.isfinite(targets.uv).all(-1))


def target_uv_pixels(targets, valid=None):
    """Convert normalized GoalTargets UV to 320x256 pixels after safe masking."""
    valid = point_mask(targets) if valid is None else valid
    uv = torch.where(valid[..., None], targets.uv.float(), torch.zeros_like(targets.uv.float()))
    return uv * uv.new_tensor([319., 255.])


def heatmap_loss(logits, targets, sigma, gaussian_sigma_px=8.):
    """Spatial Gaussian cross entropy, safe even if every point is absent.

    L = sum_{b,h} m_bh (1-sigma_b)^2 CE(H_bh, P_bh) /
        max(1, sum_{b,h} m_bh). Visibility does not replace geometry validity.
    NaN coordinates are removed *before* Gaussian arithmetic so invalid labels
    cannot poison the loss or its gradient. Pixel centers follow resize math.
    """
    valid = point_mask(targets)
    height,width = logits.shape[-2:]
    uv = target_uv_pixels(targets, valid)
    yy=(torch.arange(height,device=logits.device).float()+.5)*(256/height)-.5
    xx=(torch.arange(width,device=logits.device).float()+.5)*(320/width)-.5
    dist=(xx[None,None,None,:]-uv[:,:,0,None,None]).square()+(yy[None,None,:,None]-uv[:,:,1,None,None]).square()
    # Softmax of log Gaussian avoids underflow for labels near image borders.
    label=(-dist/(2*gaussian_sigma_px**2)).flatten(2).softmax(-1)
    ce=-(label*logits.float().flatten(2).log_softmax(-1)).sum(-1)
    weights=valid.float()*(1-sigma.float()[:,None]).square()
    return (ce*weights).sum()/valid.sum().clamp_min(1), int(valid.sum().item())


def heatmap_uv(logits):
    height,width=logits.shape[-2:]
    index=logits.flatten(2).argmax(-1)
    return torch.stack(((index%width+.5)*(320/width)-.5,
                        (index//width+.5)*(256/height)-.5),dim=-1)


def overlay_points(image, logits):
    """PIL diagnostic only; ordered left=red/right=blue, never supervision."""
    from PIL import ImageDraw
    image=image.copy();draw=ImageDraw.Draw(image)
    for point,color in zip(heatmap_uv(logits[None])[0].detach().cpu().tolist(),('red','blue')):
        x,y=point;draw.ellipse((x-4,y-4,x+4,y+4),outline=color,width=2)
    return image
