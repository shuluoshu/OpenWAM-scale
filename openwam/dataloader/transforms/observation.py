"""Train-only observation changes after camera resize/crop and composition."""
from __future__ import annotations

import math
import random

import numpy as np
from PIL import Image
import torch

from openwam.dataloader.transforms.base import ModalityTransform
from openwam.dataloader.transforms.video import VideoColorJitter

TRACE_KEY = "_observation_augmentation"
CLEAN_STATE_KEY = "_augmentation_clean_state"


def view_boxes(width, height):
    """The same rounded L-layout boundaries as assemble_multiview_layout."""
    top, half = round(height * 2 / 3), width // 2
    return {0: (0, 0, width, top), 1: (0, top, half, height), 2: (half, top, width, height)}


def missing_view_mask(width, height, available):
    result = Image.new("L", (width, height), 0)
    for index, box in view_boxes(width, height).items():
        if not available[index]:
            result.paste(255, box)
    return result


def _range(value, name, maximum=None):
    try:
        lo, hi = map(float, value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be [minimum, maximum]") from None
    if not (math.isfinite(lo) and math.isfinite(hi) and 0 < lo <= hi
            and (maximum is None or hi <= maximum)):
        raise ValueError(f"invalid {name}: {value}")
    return lo, hi


def _draw(p):
    return p >= 1 or (p > 0 and random.random() < p)


class ObservationAugmentation(ModalityTransform):
    """One clip uses one set of factors/rectangles; supervision is immutable.

    Only gripper/pose/joint *inputs* may be removed by state_dropout; action
    labels, loss masks, source anchors and FK payloads remain intact.
    """

    def __init__(self, config=None, *, training=True):
        super().__init__(apply_to=["video", "proprio"], training=training)
        config = {} if config is None else dict(config)
        allowed = {"enabled", "color_jitter", "patch_mask", "view_dropout", "state_dropout"}
        if set(config) - allowed:
            raise ValueError(f"unknown observation augmentation keys: {set(config)-allowed}")
        if type(config.get("enabled", True)) is not bool:
            raise ValueError("augmentation enabled must be boolean")
        defaults = {
            "color_jitter": dict(p=.8, brightness=.2, contrast=.2, saturation=.2),
            "patch_mask": dict(p=.5, area_range=[.05,.2], aspect_range=[.5,2.]),
            "view_dropout": dict(p=.1, keep_one_view=True),
            "state_dropout": dict(p=.1),
        }
        self.config = {"enabled": config.get("enabled", True)}
        for name, default in defaults.items():
            node = config.get(name)
            if node is None or node is False:
                node = {"p": 0.}
            else:
                node = dict(node)
            if set(node) - set(default):
                raise ValueError(f"unknown {name} keys: {set(node)-set(default)}")
            node = {**default, **node}
            try:
                probability = float(node["p"])
            except (ValueError, TypeError):
                raise ValueError(f"{name}.p must be a probability") from None
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError(f"{name}.p must be in [0,1]")
            node["p"] = probability
            self.config[name] = node
        color = self.config["color_jitter"]
        for name in ("brightness", "contrast", "saturation"):
            if not math.isfinite(float(color[name])) or not 0 <= float(color[name]) <= 1:
                raise ValueError(f"color_jitter.{name} must be in [0,1]")
            color[name] = float(color[name])
        patch = self.config["patch_mask"]
        patch["area_range"] = _range(patch["area_range"], "patch area_range", 1)
        patch["aspect_range"] = _range(patch["aspect_range"], "patch aspect_range")
        if type(self.config["view_dropout"]["keep_one_view"]) is not bool:
            raise ValueError("view_dropout.keep_one_view must be boolean")
        self.enabled = self.config["enabled"] and any(self.config[k]["p"] > 0 for k in defaults)
        self.has_image_operations = self.enabled and any(self.config[k]["p"] > 0
            for k in ("color_jitter", "patch_mask", "view_dropout"))
        self._color = VideoColorJitter(**{k: color[k] for k in ("brightness", "contrast", "saturation")})

    def apply(self, sample, missing_masks=None):
        if not self.enabled or not self.training:
            return sample
        out = dict(sample)
        meta = dict(image_changed=False, state_dropped=False, patch=None, dropped_view=None,
                    color_jitter=False, view_boxes={})
        frames = sample.get("video")
        if self.has_image_operations and frames:
            width, height = frames[0].size
            boxes = view_boxes(width, height)
            meta["view_boxes"] = boxes
            if any(frame.size != (width, height) for frame in frames):
                raise ValueError("all clip frames must share the final canvas size")
            if missing_masks is not None and (len(missing_masks) != len(frames)
                    or any(m.size != (width, height) for m in missing_masks)):
                raise ValueError("missing masks must match the final video frames")
            available = [i for i, box in boxes.items() if missing_masks is None
                         or any(mask.crop(box).getextrema()[0] == 0 for mask in missing_masks)]
            meta["available_views"] = available
            surviving = list(available)
            drop = self.config["view_dropout"]
            if surviving and (len(surviving)>1 or not drop["keep_one_view"]) and _draw(drop["p"]):
                meta["dropped_view"] = random.choice(surviving)
                surviving.remove(meta["dropped_view"])
            if available and _draw(self.config["color_jitter"]["p"]):
                payload = {"video": frames}
                if missing_masks is not None:
                    payload["video_jitter_exclusion_masks"] = missing_masks
                frames = self._color.apply(payload)["video"]
                meta["color_jitter"] = meta["image_changed"] = True
            patch = self.config["patch_mask"]
            if surviving and _draw(patch["p"]):
                view = random.choice(surviving)
                x0,y0,x1,y1 = boxes[view]
                area = (x1-x0)*(y1-y0)*random.uniform(*patch["area_range"])
                aspect = math.exp(random.uniform(*map(math.log,patch["aspect_range"])))
                w = max(1,min(x1-x0,round(math.sqrt(area*aspect))))
                h = max(1,min(y1-y0,round(math.sqrt(area/aspect))))
                x,y = random.randint(x0,x1-w),random.randint(y0,y1-h)
                meta["patch"] = dict(view=view, box=(x,y,x+w,y+h))
            if meta["patch"] is not None or meta["dropped_view"] is not None:
                frames = [frame.copy() for frame in frames]
                for frame in frames:
                    if meta["patch"] is not None:
                        frame.paste((0,0,0),meta["patch"]["box"])
                    if meta["dropped_view"] is not None:
                        frame.paste((0,0,0),boxes[meta["dropped_view"]])
                meta["image_changed"] = True
            if meta["image_changed"]:
                out["video"] = frames
                out["first_frame_image"] = [frames[0]]
        if sample.get("proprio") is not None and _draw(self.config["state_dropout"]["p"]):
            state = sample["proprio"]
            mask = sample.get("proprio_mask")
            out[CLEAN_STATE_KEY] = (state, mask)
            out["proprio"] = torch.zeros_like(state) if torch.is_tensor(state) else np.zeros_like(state)
            if mask is None:
                out["proprio_mask"] = (torch.zeros_like(state,dtype=torch.bool) if torch.is_tensor(state)
                                       else np.zeros_like(state,dtype=bool))
            else:
                out["proprio_mask"] = torch.zeros_like(mask) if torch.is_tensor(mask) else np.zeros_like(mask)
            meta["state_dropped"] = True
        out[TRACE_KEY] = meta
        return out
