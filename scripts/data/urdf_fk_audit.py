"""Read-only URDF FK and pose utilities for reproducible data consistency audits."""
from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from openwam.dataloader.utils.lerobotv3 import compute_file_local_offsets, load_episodes_parquet
from scripts.data.prepare_openwam_eef_smoke import _read_table


def pose7_to_matrix(pose):
    """xyz + wxyz quaternion → (...,4,4), rejecting invalid poses."""
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape[-1] != 7 or not np.isfinite(pose).all():
        raise ValueError('expected finite xyz+wxyz pose')
    out = np.broadcast_to(np.eye(4), (*pose.shape[:-1], 4, 4)).copy()
    out[..., :3, 3] = pose[..., :3]
    out[..., :3, :3] = Rotation.from_quat(pose[..., [4, 5, 6, 3]].reshape(-1, 4)).as_matrix().reshape(*pose.shape[:-1], 3, 3)
    return out


def eef20_to_matrix(eef):
    """Canonical column-major rot6d EEF20 → (N,2,4,4)."""
    eef = np.asarray(eef, np.float64)
    result = np.broadcast_to(np.eye(4), (len(eef), 2, 4, 4)).copy()
    for side, start in enumerate((0, 10)):
        result[:, side, :3, 3] = eef[:, start:start+3]
        a = eef[:, start+3:start+6].copy()
        b = eef[:, start+6:start+9].copy()
        na = np.linalg.norm(a, axis=-1, keepdims=True)
        if np.any(na < 1e-8):
            raise ValueError('degenerate rot6d first column')
        a /= na
        b -= (a*b).sum(-1, keepdims=True)*a
        nb = np.linalg.norm(b, axis=-1, keepdims=True)
        if np.any(nb < 1e-8):
            raise ValueError('degenerate rot6d second column')
        b /= nb
        result[:, side, :3, :3] = np.stack((a, b, np.cross(a, b)), axis=-1)
    if not np.isfinite(result).all():
        raise ValueError('nonfinite EEF pose')
    return result


def pose_errors(label, prediction):
    """Return Euclidean translation error in mm and SO(3) geodesic degrees."""
    label, prediction = np.asarray(label), np.asarray(prediction)
    if label.shape != prediction.shape or label.shape[-2:] != (4, 4):
        raise ValueError('pose shapes must match (...,4,4)')
    distance = np.linalg.norm(label[..., :3, 3]-prediction[..., :3, 3], axis=-1)*1000
    relative = np.swapaxes(label[..., :3, :3], -1, -2) @ prediction[..., :3, :3]
    angle = np.rad2deg(Rotation.from_matrix(relative.reshape(-1, 3, 3)).magnitude()).reshape(distance.shape)
    return distance, angle


class URDFModel:
    """Evaluate explicit joint values; never silently assume missing joints zero.

    Meshes/dynamics are not needed for kinematics. URDF origin is parent-to-joint,
    followed by joint-axis motion. Only the ancestor chain to target is read.
    """
    def __init__(self, path):
        self.path = Path(path)
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.xml = ET.parse(self.path).getroot()
        self.links = {node.get('name') for node in self.xml.findall('link')}
        self.joints = {j.get('name'): j for j in self.xml.findall('joint')}
        self.by_child = {j.find('child').get('link'): j for j in self.joints.values()}
        roots = self.links - set(self.by_child)
        if len(roots) != 1:
            raise ValueError(f'expected one URDF root, got {roots}')
        self.root_link = next(iter(roots))

    def chain(self, target_link, base_link=None):
        base_link = base_link or self.root_link
        if target_link not in self.links or base_link not in self.links:
            raise ValueError(f'unknown link: {base_link} -> {target_link}')
        chain, seen = [], set()
        while target_link != base_link:
            if target_link in seen or target_link not in self.by_child:
                raise ValueError(f'{base_link} is not an ancestor of link {target_link}')
            seen.add(target_link)
            joint = self.by_child[target_link]
            chain.append(joint)
            target_link = joint.find('parent').get('link')
        return list(reversed(chain))

    def fk(self, joint_values, target_link, base_link=None):
        values = {k: np.atleast_1d(np.asarray(v, np.float64)) for k, v in joint_values.items()}
        n = max((len(v) for v in values.values()), default=1)
        if any(v.ndim != 1 or len(v) not in (1, n) or not np.isfinite(v).all() for v in values.values()):
            raise ValueError('joint values must be finite vectors of equal length (or scalars)')
        def joint_value(name, visiting=()):
            if name in values:
                return np.broadcast_to(values[name], (n,))
            mimic = self.joints[name].find('mimic')
            if mimic is None or name in visiting:
                raise ValueError(f'missing required joint {name}')
            return (joint_value(mimic.get('joint'), (*visiting, name))*float(mimic.get('multiplier', 1))
                    + float(mimic.get('offset', 0)))
        transform = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
        for joint in self.chain(target_link, base_link):
            origin = joint.find('origin')
            fixed = np.eye(4)
            if origin is not None:
                fixed[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
                fixed[:3, :3] = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
            transform = transform @ fixed
            kind = joint.get('type')
            if kind == 'fixed':
                continue
            axis_node = joint.find('axis')
            axis = np.fromstring(axis_node.get('xyz', '1 0 0') if axis_node is not None else '1 0 0', sep=' ')
            axis_norm = np.linalg.norm(axis)
            if axis_norm < 1e-12:
                raise ValueError(f'zero joint axis: {joint.get("name")}')
            axis /= axis_norm
            q = joint_value(joint.get('name'))
            motion = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
            if kind in ('revolute', 'continuous'):
                motion[:, :3, :3] = Rotation.from_rotvec(q[:, None]*axis).as_matrix()
            elif kind == 'prismatic':
                motion[:, :3, 3] = q[:, None]*axis
            else:
                raise ValueError(f'unsupported URDF joint type {kind}')
            transform = transform @ motion
        return transform


def load_episode(profile_path, episode_index=0, columns=None):
    """Return config, metadata, episode row, all original episode state rows."""
    config = OmegaConf.load(profile_path)
    bucket = Path(config.dataset_dir)
    info = json.loads((bucket/'meta/info.json').read_text())
    eps = load_episodes_parquet(bucket)
    eps['_offset'] = compute_file_local_offsets(eps, 'data/chunk_index', 'data/file_index')
    selected = eps[eps.episode_index == episode_index]
    if len(selected) != 1:
        raise ValueError(f'episode {episode_index} not unique/present')
    row = selected.iloc[0]
    path = bucket/info['data_path'].format(chunk_index=int(row['data/chunk_index']), file_index=int(row['data/file_index']))
    keys = None if columns is None else list(dict.fromkeys(['episode_index','frame_index','timestamp', *columns]))
    frame = _read_table(path, keys).slice(int(row['_offset']), int(row.length)).to_pandas()
    if len(frame) != int(row.length) or set(frame.episode_index) != {episode_index}:
        raise ValueError('episode physical row alignment failed')
    return config, info, row, frame


def write_result(output_dir, metadata, **arrays):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, array in arrays.items():
        if np.asarray(array).dtype.kind in 'fc' and not np.isfinite(array).all():
            raise ValueError(f'nonfinite result {key}')
    np.savez_compressed(output_dir/'arrays.npz', **arrays)
    (output_dir/'metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    return metadata
