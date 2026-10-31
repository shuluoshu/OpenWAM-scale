"""Scalar-only batched fixed-anchor targets for exhaustive native statistics.

The arithmetic and mask policy mirror ``CuratedStage2Dataset.raw_window``.
Absolute pose conversion happens once per episode; batches allocate only their
32 future rows. No image decoding, normalization, clipping, or anchor selection
is performed here (selection remains the dataset's ``_options`` policy).
"""
from __future__ import annotations

import numpy as np

from openwam.dataloader.native_stage2_io import (
    _eef_mask_flags, _valid_flags, source_frame_validity,
)
from openwam.dataloader.utils.gripper import ALIGNED_COMMAND, ALIGNED_VALID
from openwam.dataloader.utils.mora_relative import MoraHeadTCPRepresentation, _rotation, _rot6d


class RelativeEpisodeEncoder:
    """Cache one loaded native episode and encode arbitrary eligible anchors.

    ``encode(anchors, ends)`` returns ``(action, action_mask, state, state_mask)``
    with shapes ``[B,32,52]``, ``[B,32,52]``, ``[B,52]``, ``[B,52]``.
    Values are float32, masks bool, invalid values zero. Ends are exclusive
    selected/instruction source boundaries, capped at anchor+33 internally.
    Every anchor must have at least one future row. The metadata and its
    representation must remain unchanged after constructing the encoder.
    """

    horizon = 32
    raw_dim = 52

    def __init__(self, metadata):
        rep = metadata['representation']
        if type(rep) is not MoraHeadTCPRepresentation:
            raise ValueError('RelativeEpisodeEncoder requires native MoraHeadTCPRepresentation')
        cols = metadata['columns']
        self.length = int(metadata['length'])
        if self.length < 2 or any(len(v) != self.length for v in cols.values()):
            raise ValueError('episode columns must have the declared length >= 2')
        self.values, self.valid = rep._absolute_values(cols)
        self.rotations = [_rotation(self.values[:, start+3:start+9])[0] for start in (0, 10)]
        self.prealigned = ALIGNED_COMMAND in cols
        if self.prealigned:
            self.commands = np.asarray(cols[ALIGNED_COMMAND])
            self.command_valid = np.asarray(cols[ALIGNED_VALID], dtype=bool)
        else:
            self.commands, self.command_valid = rep.gripper_commands.align(cols, self.length)
        self.command_offsets = [rep.gripper_commands.sources.get(side, {}).get('row_offset', 0)
                                for side in ('left', 'right')]
        self._cache_camera(rep, cols, metadata['head_calibration_eligible'])
        self._cache_source_masks(rep, cols)
        self.state = np.zeros((self.length, self.raw_dim), np.float32)
        self.state_mask = np.zeros_like(self.state, bool)
        self.state[:, :34] = self.values
        self.state_mask[:, :34] = self.valid
        for side, start in enumerate((0, 10)):
            dst = 34+side*9
            self.state[:, dst:dst+3] = (self.rc @ self.values[:, start:start+3, None])[..., 0] + self.tc
            self.state[:, dst+3:dst+9] = _rot6d(self.rc @ self.rotations[side])
            self.state_mask[:, dst:dst+9] = (self.camera_valid & self.valid[:, start])[:, None]
        self.state_mask &= self.source_state_mask
        self.state[~self.state_mask] = 0

    def _cache_camera(self, rep, cols, eligible):
        n = self.length
        self.rc = np.tile(np.eye(3), (n, 1, 1))
        self.tc = np.zeros((n, 3), np.float64)
        self.camera_valid = np.zeros(n, bool)
        key = rep.head_calibration_key
        if not eligible or not rep.head_camera_available or key not in cols:
            return
        camera = np.asarray(cols[key], np.float64).reshape(n, 4, 4)
        finite = np.isfinite(camera).all(axis=(1, 2))
        # Invalid matrices cannot contribute; replacing them also prevents
        # overflow/invalid arithmetic warnings during the vectorized checks.
        safe = np.where(finite[:, None, None], camera, np.eye(4))
        rc = safe[:, :3, :3]
        good = finite & np.isclose(safe[:, 3], [0, 0, 0, 1], atol=1e-5).all(-1)
        good &= np.isclose(rc @ rc.transpose(0, 2, 1), np.eye(3), atol=1e-4).all(axis=(1, 2))
        good &= np.isclose(np.linalg.det(rc), 1, atol=1e-4)
        self.camera_valid = good
        self.rc[good] = rc[good]
        self.tc[good] = safe[good, :3, 3]

    def _cache_source_masks(self, rep, cols):
        """Factor merge_source_masks into anchor, future and command rows.

        Stored state grip flags apply only to proprio; source action flags are
        indexed at future-1. Camera flags apply only at the fixed anchor.
        """
        n = self.length
        global_valid = source_frame_validity(cols, n)
        eef = np.ones((n, 20), bool)
        if 'observation.states.eef.mask' in cols:
            eef &= _eef_mask_flags(rep, cols, 'observation.states.eef.mask', n)
        for key, threshold in [('observation.states.hand.valid', .5), ('observation.states.hand.confidence', .8)]:
            if key in cols:
                flags = _valid_flags(np.asarray(cols[key]).reshape(n, 2), threshold, inclusive='confidence' in key)
                for side in range(2):
                    eef[:, side*10:(side+1)*10] &= flags[:, side, None]
        key = 'observation.states.eef.confidence'
        if key in cols:
            flags = _valid_flags(np.asarray(cols[key]).reshape(n, 4), .8, inclusive=True)
            eef[:, 9] &= flags[:, 2]
            eef[:, 19] &= flags[:, 3]
        self.source_state_mask = np.broadcast_to(global_valid[:, None], (n, 52)).copy()
        self.source_pose_mask = np.broadcast_to(global_valid[:, None], (n, 52)).copy()
        self.source_command_mask = np.ones((n, 52), bool)
        for side, start in enumerate((0, 10)):
            head = slice(34+side*9, 43+side*9)
            pose = eef[:, start:start+9].all(-1)
            self.source_state_mask[:, start:start+10] &= eef[:, start:start+10]
            self.source_state_mask[:, head] &= pose[:, None]
            self.source_pose_mask[:, start:start+9] &= pose[:, None]
            self.source_pose_mask[:, head] &= pose[:, None]
        if 'actions.eef.mask' in cols:
            flags = _eef_mask_flags(rep, cols, 'actions.eef.mask', n)
            for side, start in enumerate((0, 10)):
                pose = flags[:, start:start+9].all(-1)
                self.source_command_mask[:, start:start+9] &= pose[:, None]
                self.source_command_mask[:, 34+side*9:43+side*9] &= pose[:, None]
                self.source_command_mask[:, start+9] &= flags[:, start+9]
        if 'actions.joint.mask' in cols:
            flags = _valid_flags(cols['actions.joint.mask'])
            for side, name in enumerate(('left', 'right')):
                indices = rep.joint_indices[name]
                self.source_command_mask[:, 20+side*7:20+side*7+len(indices)] &= flags[:, indices]
        for key in ('source.camera.world_to_camera_valid', 'observation.states.camera.valid'):
            if key in cols:
                self.camera_valid &= _valid_flags(np.asarray(cols[key]).reshape(n, -1)).all(-1)

    def encode(self, anchors, ends):
        anchors, ends = np.asarray(anchors), np.asarray(ends)
        if (anchors.ndim != 1 or ends.shape != anchors.shape or
                (anchors.size and (anchors.dtype.kind not in 'iu' or ends.dtype.kind not in 'iu'))):
            raise ValueError('anchors and ends must be equal-length one-dimensional integer arrays')
        if (np.any(anchors < 0) or np.any(ends > self.length) or
                np.any(anchors >= self.length-1) or np.any(ends <= anchors+1)):
            raise ValueError('bounds require 0 <= anchor < anchor+1 < end <= episode length')
        anchors, ends = anchors.astype(np.int64), ends.astype(np.int64)
        ends = np.minimum(ends, anchors+self.horizon+1)
        future = anchors[:, None] + np.arange(1, self.horizon+1)
        in_window = future < ends[:, None]
        future = np.minimum(future, self.length-1)
        previous = future-1
        a = np.zeros((len(anchors), self.horizon, self.raw_dim), np.float32)
        mask = np.zeros_like(a, bool)
        a[:, :, :34] = self.values[future]
        mask[:, :, :34] = self.valid[future]
        for side, start in enumerate((0, 10)):
            rotations = self.rotations[side]
            ra = rotations[anchors]
            rf = rotations[future]
            reference = ra.transpose(0, 2, 1)[:, None]
            delta = self.values[future, start:start+3] - self.values[anchors, None, start:start+3]
            a[:, :, start:start+3] = (reference @ delta[..., None])[..., 0]
            a[:, :, start+3:start+9] = _rot6d(reference @ rf)
            pose_valid = self.valid[anchors, None, start] & self.valid[future, start]
            mask[:, :, start:start+9] = pose_valid[..., None]
            dst = 34+side*9
            rc = self.rc[anchors, None]
            a[:, :, dst:dst+3] = (rc @ delta[..., None])[..., 0]
            a[:, :, dst+3:dst+9] = _rot6d(rc @ rf @ reference @ rc.swapaxes(-1, -2))
            mask[:, :, dst:dst+9] = (pose_valid & self.camera_valid[anchors, None])[..., None]
        a[:, :, 20:34] -= self.values[anchors, None, 20:34]
        mask[:, :, 20:34] &= self.valid[anchors, None, 20:34]
        a[:, :, [9, 19]] = self.commands[previous]
        mask[:, :, [9, 19]] = self.command_valid[previous]
        if not self.prealigned:
            # Native raw_window aligns within its slice only when no episode
            # alignment exists. Preserve that fallback boundary behavior too.
            for side, dim in enumerate((9, 19)):
                rows = previous+self.command_offsets[side]
                mask[:, :, dim] &= (rows >= anchors[:, None]) & (rows < ends[:, None])
        mask &= self.source_pose_mask[anchors, None] & self.source_pose_mask[future]
        mask &= self.source_command_mask[previous]
        mask &= in_window[..., None]
        a[~mask] = 0
        return a, mask, self.state[anchors].copy(), self.state_mask[anchors].copy()
