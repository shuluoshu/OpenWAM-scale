"""Parse explicit URDF arm chains into immutable-by-convention numeric data."""

import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def rigid_matrix(value):
    matrix = np.asarray(value, dtype=np.float64)
    if (
        matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6)
        or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-5)
    ):
        raise ValueError("endpoint transform must be a finite rigid 4x4 matrix")
    return matrix


def compile_chain(path, base_link, target_link, joint_names, endpoint_transform=None):
    """No missing-joint defaults or implicit mimic expansion; fail on ambiguity."""
    path = Path(path)
    content = path.read_bytes()
    xml = ET.fromstring(content)
    links = {link.get("name") for link in xml.findall("link")}
    joints = xml.findall("joint")
    by_child = {j.find("child").get("link"): j for j in joints}
    if len(by_child) != len(joints):
        raise ValueError("duplicate URDF child links")
    if base_link not in links or target_link not in links:
        raise ValueError("unknown URDF base/target link")
    if not 1 <= len(joint_names) <= 7 or len(set(joint_names)) != len(joint_names):
        raise ValueError("expected one to seven distinct arm joint names")
    chain, visited, required = [], set(), set()
    child = target_link
    while child != base_link:
        if child in visited or child not in by_child:
            raise ValueError(f"{base_link} is not an ancestor of {target_link}")
        visited.add(child)
        j = by_child[child]
        kind, name = j.get("type"), j.get("name")
        if kind not in ("fixed", "revolute", "continuous", "prismatic"):
            raise ValueError(f"unsupported URDF joint type: {kind}")
        index = -1
        if kind != "fixed":
            if j.find("mimic") is not None or name not in joint_names:
                raise ValueError(f"required joint {name} must be supplied explicitly (no mimic)")
            index = joint_names.index(name)
            required.add(name)
        origin = j.find("origin")
        fixed = np.eye(4)
        if origin is not None:
            fixed[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            fixed[:3, :3] = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")).as_matrix()
        rigid_matrix(fixed)
        node = j.find("axis")
        axis = np.fromstring(node.get("xyz", "1 0 0") if node is not None else "1 0 0", sep=" ")
        if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-12:
            raise ValueError(f"invalid joint axis: {name}")
        chain.append(
            dict(name=name, kind=kind, index=index, fixed=fixed.tolist(), axis=(axis / np.linalg.norm(axis)).tolist())
        )
        child = j.find("parent").get("link")
    if required != set(joint_names):
        raise ValueError("configured joints must exactly cover moving joints in the arm chain")
    return dict(
        joints=chain[::-1],
        dof=len(joint_names),
        joint_names=list(joint_names),
        tool=rigid_matrix(np.eye(4) if endpoint_transform is None else endpoint_transform).tolist(),
        urdf_path=str(path.resolve()),
        urdf_sha256=hashlib.sha256(content).hexdigest(),
        base_link=base_link,
        target_link=target_link,
    )
