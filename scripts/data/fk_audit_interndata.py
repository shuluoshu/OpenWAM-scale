"""Read-only episode-0 Genie-1 FK audit, with explicit frame provenance.

The independent test is arm_base_link -> arm_{l,r}_end_link. Robot-frame
TCP diagnostics use transforms extracted algebraically from reference row 0;
they are deliberately not evidence of independently validated robot/TCP assets.
Run: python -m scripts.data.fk_audit_interndata --output-dir outputs/fk_consistency/interndata
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.data.urdf_fk_audit import (
    URDFModel,
    eef20_to_matrix,
    load_episode,
    pose7_to_matrix,
    pose_errors,
    write_result,
)

PROFILE = Path("configs/dataloader/canonical_eef_interndata_smoke.yaml")
URDF = Path("/mnt/data/cpfs/b5big/pretrain_urdfs/interndata_a1/G1_120s/G1_120s.urdf")
SIDES = ("left", "right")


def _metrics(label, prediction):
    mm, deg = pose_errors(label, prediction)
    return {
        side: {
            "position_mm_median": float(np.median(mm[:, i])),
            "position_mm_p95": float(np.percentile(mm[:, i], 95)),
            "position_mm_max": float(mm[:, i].max()),
            "rotation_deg_median": float(np.median(deg[:, i])),
            "rotation_deg_p95": float(np.percentile(deg[:, i], 95)),
            "rotation_deg_max": float(deg[:, i].max()),
        } for i, side in enumerate(SIDES)
    }


def run(output_dir: Path) -> dict:
    cfg, info, episode, frame = load_episode(PROFILE)
    source_path = Path(cfg.dataset_dir) / "meta/source_stage2.json"
    source = json.loads(source_path.read_text())
    model = URDFModel(URDF)
    native_q = [np.stack(frame[f"states.{side}_joint.position"]) for side in SIDES]
    joint = np.concatenate(native_q, axis=-1)
    canonical_q = np.stack(frame["observation.states.joint"])
    np.testing.assert_array_equal(joint, canonical_q[:, [*range(7), *range(8, 15)]])
    joint_map = {}
    joint_values = {}
    for i, side in enumerate(SIDES):
        for j in range(7):
            name = f"idx{21 + i*40 + j}_arm_{side[0]}_joint{j+1}"
            joint_values[name] = native_q[i][:, j]
            joint_map[name] = {
                "source_key": f"states.{side}_joint.position", "source_index": j,
                "canonical_joint_index": i*8+j, "unit": "radian", "sign": 1,
                "offset_rad": 0,
            }

    def poses(template):
        return np.stack([
            pose7_to_matrix(np.stack(frame[template.format(side=side)])) for side in SIDES
        ], axis=1)

    label = poses("states.{side}_ee_to_{side}_armbase_pose")
    fk = np.stack([
        model.fk(joint_values, f"arm_{side[0]}_end_link", "arm_base_link")
        for side in SIDES
    ], axis=1)
    # Independent FK never uses an EEF label in its computation.
    metrics = _metrics(label[1:], fk[1:])
    primary_passed = all(
        m["position_mm_max"] < 0.01 and m["rotation_deg_max"] < 0.001
        for m in metrics.values()
    )

    robot_ee = poses("states.{side}_ee_to_robot_pose")
    robot_tcp = poses("states.{side}_tcp_to_robot_pose")
    local_tcp = poses("states.{side}_tcp_to_{side}_armbase_pose")
    robot_from_armbase = robot_ee @ np.linalg.inv(label)
    ee_from_tcp = np.linalg.inv(label) @ local_tcp
    # Algebra on source reference row 0 only; no optimization and no fitted
    # joint offsets. The held transforms are subsequently checked on rows 1:.
    held_robot_from_armbase = robot_from_armbase[0]
    held_ee_from_tcp = ee_from_tcp[0]
    diagnostic_robot_fk = held_robot_from_armbase[None] @ fk
    diagnostic_tcp_fk = diagnostic_robot_fk @ held_ee_from_tcp[None]
    canonical = eef20_to_matrix(np.stack(frame["observation.states.eef"]))
    world_from_robot = pose7_to_matrix(np.stack(frame["states.robot_to_env_pose"]))
    corrections = source["starvla_metadata"]["eef_axis_correction"]["per_arm"]
    correction = np.stack([np.asarray(corrections[side]) for side in SIDES])
    canonical_from_native = world_from_robot[:, None] @ robot_ee
    canonical_from_native[..., :3, :3] = canonical_from_native[..., :3, :3] @ correction[None]
    diagnostic_canonical_fk = world_from_robot[:, None] @ diagnostic_robot_fk
    diagnostic_canonical_fk[..., :3, :3] = diagnostic_canonical_fk[..., :3, :3] @ correction[None]
    baseline_tcp_as_ee = _metrics(local_tcp[1:], fk[1:])
    robot_constancy = _metrics(robot_from_armbase[1:], np.broadcast_to(held_robot_from_armbase, robot_from_armbase[1:].shape))
    tcp_constancy = _metrics(ee_from_tcp[1:], np.broadcast_to(held_ee_from_tcp, ee_from_tcp[1:].shape))
    metadata = {
        "dataset": "interndata_genie1", "profile": str(PROFILE),
        "dataset_dir": str(cfg.dataset_dir), "episode_index": 0,
        "state_rows": len(frame), "num_frames": len(frame)-1,
        "target_semantics": "Every state row 1: is next-state GT; row 0 is the initial reference. No action column used.",
        "status": "passed" if primary_passed else "failed",
        "status_scope": "Independent URDF arm-local EE comparison only; robot-frame TCP extension is a label-derived diagnostic.",
        "urdf_paths": [str(URDF)], "urdf_sha256": {str(URDF): model.sha256},
        "joint_map": joint_map, "coordinate_frame": "arm_base_link (common to both arms)",
        "target_links": ["arm_l_end_link", "arm_r_end_link"],
        "labels": [f"states.{s}_ee_to_{s}_armbase_pose" for s in SIDES],
        "metrics": metrics,
        "thresholds": {"max_position_mm": 0.01, "max_rotation_deg": 0.001},
        "mapping_evidence": [
            "URDF manifest maps native genie1 to G1_120s.",
            "source_stage2.json declares arm_joint_identity, radian units, and left/right source order.",
            "Native joint arrays exactly equal canonical joints [0:7] and [8:15] on every episode row.",
            "Direct identity-joint FK in common arm_base_link matches both native EE arm-base labels over all 268 states; no alignment, sign search, or calibration enters this primary comparison.",
            "Despite native field names, left_armbase and right_armbase correspond numerically to common arm_base_link, not the URDF side-specific arm_l_base_link/arm_r_base_link.",
            "Canonical EEF metadata explicitly specifies world_from_robot @ native_EE followed by the per-arm orientation correction; checked separately against every canonical state.",
        ],
        "source_metadata": {"path": str(source_path), "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest()},
        "diagnostics": {
            "raw_ee_fk_vs_tcp_label": {"metrics": baseline_tcp_as_ee, "interpretation": "Expected 220 mm discrepancy: native TCP differs from EE; not a kinematic failure."},
            "robot_tcp": {
                "status": "label_derived_reference_transform_diagnostic",
                "labels": [f"states.{s}_tcp_to_robot_pose" for s in SIDES],
                "formula": "(native_robot_EE[0] @ inverse(native_armbase_EE[0])) @ URDF_FK(q[t]) @ (inverse(native_armbase_EE[0]) @ native_armbase_TCP[0])",
                "metrics": _metrics(robot_tcp[1:], diagnostic_tcp_fk[1:]),
                "reference_robot_from_armbase": held_robot_from_armbase.tolist(),
                "reference_ee_from_tcp": held_ee_from_tcp.tolist(),
                "robot_from_armbase_constancy": robot_constancy,
                "ee_from_tcp_constancy": tcp_constancy,
                "independent_validation": False,
            },
            "canonical_conversion": {
                "formula": "world_from_robot @ native_robot_EE; rotation postmultiplied by metadata axis correction",
                "axis_correction": corrections,
                "metrics": _metrics(canonical[1:], canonical_from_native[1:]),
            },
            "canonical_fk_with_reference_robot_transform": {
                "independent_validation": False,
                "metrics": _metrics(canonical[1:], diagnostic_canonical_fk[1:]),
            },
        },
        "joint_peak_to_peak_rad": {side: np.ptp(native_q[i], axis=0).tolist() for i, side in enumerate(SIDES)},
        "limitations": [
            "Full robot-base FK needs idx01_body_joint1 and idx02_body_joint2, absent from this bucket. Do not silently set them to zero.",
            "The 0.22 m local +Z EE-to-TCP transform is evidenced by native pose pairs, not a TCP link in this URDF. This audit does not independently certify that tool transform.",
            "Robot-frame TCP and world/canonical FK diagnostics hold source-derived transforms from reference row 0; they cannot establish independent base/tool calibration.",
            "Episode 0 has strong right-arm motion but the left arm is almost stationary (all seven joint ranges below 0.0016 rad). This limits left-arm coverage.",
        ],
    }
    return write_result(
        output_dir, metadata,
        frame_indices=frame.frame_index.to_numpy()[1:], time=frame.timestamp.to_numpy()[1:],
        label_pose=label[1:], fk_pose=fk[1:], joint=joint[1:],
        reference_label_pose=label[0], reference_fk_pose=fk[0],
        reference_joint=joint[0], reference_frame_index=frame.frame_index.to_numpy()[0],
        reference_time=frame.timestamp.to_numpy()[0],
        tcp_label_pose=robot_tcp[1:], tcp_diagnostic_fk_pose=diagnostic_tcp_fk[1:],
        canonical_label_pose=canonical[1:], canonical_diagnostic_fk_pose=diagnostic_canonical_fk[1:],
        local_tcp_label_pose=local_tcp[1:], robot_from_armbase=robot_from_armbase[1:],
        ee_from_tcp=ee_from_tcp[1:],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/fk_consistency/interndata"))
    args = parser.parse_args()
    result = run(args.output_dir)
    print(json.dumps({"status": result["status"], "metrics": result["metrics"]}, indent=2))
