"""Read prepared curated fixtures with cached RGB and production target logic.

This adapter is for the curated smoke manifest produced by
scripts.data.prepare_curated_gripper_commands. It does not alter source labels.
"""
from collections import Counter
from contextlib import contextmanager
import json
from pathlib import Path
from types import MethodType

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import Dataset

from openwam.dataloader.registry import build_dataset
from openwam.dataloader.transforms.observation import missing_view_mask


def _cached_video(self, row, ep_local, offset, real_local_indices, idx, *, return_missing_masks=False):
    images = [Image.fromarray(np.array(self._curated_frames[int(offset+i)])) for i in real_local_indices]
    frames = images + [images[-1]] * (self._num_video_frames-len(images))
    if return_missing_masks:
        mask = missing_view_mask(*frames[0].size, self._curated_view_available)
        return frames, [mask] * len(frames)
    return frames


class CuratedFixtureDataset(Dataset):
    def __init__(self, manifest, *, augmentation=None, split="train"):
        manifest = Path(manifest).resolve()
        rows = json.loads(manifest.read_text())
        self.rows, self.readers, self.mapping, self.weights, self.skipped = [], [], [], [], []
        for row in rows:
            path = manifest.parent / "data" / row["id"]
            config = OmegaConf.load(path / "data.yaml")
            if augmentation is not None:
                config.observation_augmentation = augmentation
            reader = build_dataset(config, split=split)
            if not len(reader):
                self.skipped.append(row["id"])
                continue
            reader._curated_frames = np.load(path / "canvases.npy", mmap_mode="r")
            reader._curated_view_available = tuple(bool(row.get(k)) for k in ("head", "left", "right"))
            reader._decode_window_video = MethodType(_cached_video, reader)
            self.rows.append(row)
            self.readers.append(reader)
        if not self.readers:
            raise ValueError("All curated windows rejected by gripper state filtering")
        family_counts = Counter(row["family"] for row in self.rows)
        for i, (row, reader) in enumerate(zip(self.rows, self.readers)):
            self.mapping.extend((i, j) for j in range(len(reader)))
            self.weights.extend([1 / (family_counts[row["family"]] * len(reader))] * len(reader))
        self.action_dim = 112

    def __len__(self):
        return len(self.mapping)

    @contextmanager
    def unaugmented(self):
        """Fixed evaluation reads clean RGB/state and restores training modes."""
        transforms = [t for reader in self.readers for t in
                      (reader.observation_augmentation, reader._color_jitter) if t is not None]
        modes = [t.training for t in transforms]
        try:
            for transform in transforms:
                transform.eval()
            yield self
        finally:
            for transform, training in zip(transforms, modes):
                transform.training = training

    def __getitem__(self, index):
        i, j = self.mapping[int(index)]
        sample = self.readers[i][j]
        sample["_curated_index"], sample["_curated_id"] = int(index), self.rows[i]["id"]
        return sample

    def eval_indices(self):
        wanted = {"00_Genrobot", "04_LightWheel", "08_abc_130k", "12_egodex", "16_egoverse",
                  "20_interndata_a1", "22_interndata_a1", "23_interndata_a1", "24_mecka",
                  "26_morphirobot0930", "30_robochallenge_v1", "35_robochallenge_v2",
                  "38_robomind_v1", "42_robomind_v2"}
        return [k for k, (i, j) in enumerate(self.mapping) if j == 0 and self.rows[i]["id"] in wanted]
