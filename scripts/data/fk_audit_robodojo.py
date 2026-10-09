"""Read-only RoboDojo episode-0 URDF FK audit, including base-free motion checks.

Run with ``python -m scripts.data.fk_audit_robodojo --output-dir PATH``.
Absolute FK uses the local simulator's declared arm mounting transforms. The
first-state alignment is diagnostic only and never establishes absolute truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.data.urdf_fk_audit import (
    URDFModel,
    eef20_to_matrix,
    load_episode,
    pose7_to_matrix,
    pose_errors,
    write_result,
)

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "configs/dataloader/canonical_eef_robodojo_openloop.yaml"
URDF = Path("/mnt/data/cpfs/b5/assets/RoboDojo/Assets/Robots/x5/X5A.urdf")
SIMULATOR = Path("/mnt/workspace/suran/projects/RoboDojo")
BASE_CONFIG = SIMULATOR / "env_cfg/robot/dual_x5.yml"


def _summary(label, prediction):
    position, orientation = pose_errors(label, prediction)
    return {
        side: {
            "position_mm": {"mean": float(position[:, arm].mean()),
                            "p95": float(np.percentile(position[:, arm], 95)),
                            "max": float(position[:, arm].max())},
            "rotation_deg": {"mean": float(orientation[:, arm].mean()),
                             "p95": float(np.percentile(orientation[:, arm], 95)),
                             "max": float(orientation[:, arm].max())},
        }
        for arm, side in enumerate(("left", "right"))
    }


def run(output_dir: Path) -> dict:
    config, info, _, frame = load_episode(
        PROFILE, episode_index=0,
        columns=["observation.states.joint", "observation.states.eef"],
    )
    joint_source = np.stack(frame["observation.states.joint"]).astype(np.float64)
    label_all = eef20_to_matrix(np.stack(frame["observation.states.eef"]))
    if joint_source.shape != (len(frame), 14) or len(frame) < 2:
        raise ValueError("expected at least two states with 14 joints/grippers")
    if not np.array_equal(frame.frame_index.to_numpy(), np.arange(len(frame))):
        raise ValueError("episode frame indices must be contiguous from zero")
    model = URDFModel(URDF)
    arm_fk_all = np.stack([
        model.fk({f"joint{j+1}": joint_source[:, arm*7+j] for j in range(6)},
                 target_link="link6", base_link="base_link")
        for arm in range(2)
    ], axis=1)
    base_config = yaml.safe_load(BASE_CONFIG.read_text())
    base_pose7 = np.asarray([r["default_root_pos"] + r["default_root_rot"]
                            for r in base_config["robots"]], dtype=np.float64)
    base_transforms = pose7_to_matrix(base_pose7)
    fk_all = base_transforms[None] @ arm_fk_all
    # Canonical joint14 is two seven-DOF slots. Slot seven is unused for X5;
    # gripper openness in source column 6/13 is never a seventh arm angle.
    joint = joint_source[1:].copy()
    joint[:, [6, 13]] = 0
    label, fk = label_all[1:], fk_all[1:]
    relative_label = np.linalg.inv(label_all[0])[None] @ label
    relative_fk = np.linalg.inv(fk_all[0])[None] @ fk
    diagnostic_base = label_all[0] @ np.linalg.inv(arm_fk_all[0])
    diagnostic_fk = diagnostic_base[None] @ arm_fk_all[1:]
    relative_position, relative_rotation = pose_errors(relative_label, relative_fk)
    relative_pass = bool((relative_position <= 1.0).all() and
                         (relative_rotation <= 0.1).all())
    absolute_position, absolute_rotation = pose_errors(label, fk)
    absolute_pass = bool((absolute_position <= 1.0).all() and
                         (absolute_rotation <= 0.1).all())
    metadata = {
        "dataset": "RoboDojo sweep_blocks",
        "dataset_dir": str(config.dataset_dir), "profile": str(PROFILE),
        "episode_index": 0, "state_count": len(frame), "sample_count": len(label),
        "fps": info["fps"],
        "status": "source_config_absolute_and_relative_pass" if absolute_pass and relative_pass
                  else "source_config_mismatch",
        "urdf": str(URDF), "urdf_sha256": model.sha256,
        "urdf_config": str(URDF.with_name("robot_config.yml")),
        "base_config": str(BASE_CONFIG),
        "base_config_sha256": hashlib.sha256(BASE_CONFIG.read_bytes()).hexdigest(),
        "source_base_pose_xyz_wxyz": base_pose7.tolist(),
        "source_base_transforms": base_transforms.tolist(),
        "urdf_provenance": "Actual local RoboDojo assets; SHA256 also equals RoboDojo source-checkout Assets/Robots/x5/X5A.urdf and local RoboTwin X5A.urdf. Only RoboDojo task transforms are used.",
        "joint_mapping": {
            "left": {f"joint{j+1}": j for j in range(6)},
            "right": {f"joint{j+1}": 7+j for j in range(6)},
            "source_order": info["starvla_metadata"]["joint_names"],
            "unit": "radian", "sign": "+1", "offset": 0,
            "canonical_joint14_mask": [1,1,1,1,1,1,0,1,1,1,1,1,1,0],
            "gripper_source_columns_excluded_from_fk": [6, 13],
        },
        "reference_frames": {
            "label": "Stage3 robot_base_used_as_stage3_world_reference; source get_real_endpose is scene/env-origin relative, not individual-arm-base relative",
            "fk": "Source dual_x5.yml scene-to-base_link transform @ URDF base_link-to-link6 FK",
            "end_link": "link6 (RoboDojo robot_config.yml ee_link)",
            "label_source": "observation.endpose.left/right, xyz + wxyz converted to EEF20 xyz + first two rotation-matrix columns",
            "relative": "inv(T_state0) @ T_state_t independently for labels and FK; cancels constant left-multiplied base transform",
            "tcp": "Not audited: source TCP fields existed but stage3 did not emit them; gripper_bias=0.145 is not applied to link6/endpose",
        },
        "alignment": "Full episode original rows 1..N-1: joint state[t+1] and EEF state[t+1], not actions.* or model predictions",
        "metrics": {
            "source_config_absolute": _summary(label, fk),
            "raw_arm_base_unaligned": _summary(label, arm_fk_all[1:]),
            "relative_to_state0": _summary(relative_label, relative_fk),
            "first_state_base_alignment_diagnostic_only": _summary(label, diagnostic_fk),
        },
        "relative_acceptance_threshold": {"position_mm": 1.0, "rotation_deg": 0.1,
                                           "all_samples_pass": relative_pass},
        "absolute_acceptance_threshold": {"position_mm": 1.0, "rotation_deg": 0.1,
                                           "all_samples_pass": absolute_pass},
        "diagnostic_first_state_base_transforms": diagnostic_base.tolist(),
        "limitations": [
            "Source config quaternions [0.707,0,0,0.707] are normalized for rigid transforms. A constant ~0.0173 degree yaw residual remains; simulator handling of the nonunit quaternion was not reproduced or fitted away.",
            "First-state fitted base transforms are diagnostics, excluded from acceptance. Raw arm-base discrepancies compare distinct frames and do not establish dataset errors.",
            "Local simulator source supplies task/base/end-link conventions, but the exact data-generation commit is not recorded in dataset metadata; pass is a numerical consistency finding for episode 0, not a provenance certification or TCP audit.",
            "Audit examines 702 next-state targets in episode 0 only. Gripper actuation is not part of the six-joint arm FK chain.",
        ],
        "evidence": [str(URDF), str(URDF.with_name("robot_config.yml")),
                     str(BASE_CONFIG), str(SIMULATOR/"env_cfg/arx_x5.yml"),
                     str(SIMULATOR/"task/RoboDojo/config/_task.yml"),
                     str(SIMULATOR/"env/robot_manager/robot_manager.py"),
                     str(Path(config.dataset_dir)/"meta/info.json"),
                     str(Path(config.dataset_dir)/"meta/source_stage2.json")],
    }
    return write_result(
        output_dir, metadata,
        frame_indices=frame.frame_index.to_numpy()[1:],
        time=frame.timestamp.to_numpy()[1:], joint=joint,
        gripper=joint_source[1:, [6, 13]], label_pose=label, fk_pose=fk,
        raw_arm_fk_pose=arm_fk_all[1:],
        reference_label_pose=label_all[0], reference_fk_pose=fk_all[0],
        relative_label_pose=relative_label, relative_fk_pose=relative_fk,
        diagnostic_aligned_fk_pose=diagnostic_fk,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/fk_consistency/robodojo")
    args = parser.parse_args()
    result = run(args.output_dir)
    print(json.dumps({"status": result["status"], "sample_count": result["sample_count"],
                      "output_dir": str(args.output_dir)}, indent=2))
