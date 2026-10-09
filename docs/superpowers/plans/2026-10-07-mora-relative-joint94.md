# Mora relative TCP and joint94 implementation plan

**Goal:** Extend OpenWAM's Mora-data adapter with 94D action/proprio, current-TCP-relative next-state pose targets, and arm-joint deltas.

**Architecture:** Keep the established 80D slot positions and append left/right seven-joint blocks at 80:87 / 87:94. An explicit representation selected by the canonical reader config owns source mappings, pose references, masks and stats. Preserve the 80D path for existing checkpoint evaluation. Missing TCP never implies a fabricated supervised pose.

**Tech stack:** NumPy, PyArrow, PyTorch, existing LeRobot reader and pytest.

## Contracts

- User requested OpenWAM changes only; source Mora datasets and sibling repository remain read-only.
- Target time remains state[t+k+1], with every delta anchored at window-start state[t]. Joint delta is q[t+k+1]-q[t]. Grippers remain absolute openness in existing slots.
- Relative pose is R[t].T @ (p[target]-p[t]) and R[t].T @ R[target], with column-major rot6d.
- TCP sources are explicit per-arm pose columns (xyz + wxyz quaternion), or EEF poses with a declared fixed EEF-to-TCP transform. Explicit identity transforms permit the existing EEF reference when that convention is selected. With neither source, mask all nine pose channels on that side. Availability is per component, independently of joint/gripper supervision.
- Arm joints use explicit source indices per side. A six-joint arm leaves its seventh slot masked. Missing joints are masked. Retain all old 80D positions, without relabeling finger slots as arm joints.
- Proprio carries current absolute poses in the source reference frame plus current joint angles. Its availability mask is separate from action validity.
- Stats must be generated for the exact relative contract and horizon, across all valid anchor/target pairs in selected episodes; zero-filled missing channels do not enter statistics. Rotation normalization remains identity.
- New action/state widths are 94. Existing global 80D defaults remain unchanged. Old checkpoint expansion is explicit and only pads the known action input/output and proprio projections; unrelated shape mismatches still fail.
- No GPU training, source-data edits, commits or pushes are required for this task.

## Work

- [x] Add failing coordinate, mapping, unavailable-pose, six-joint, tail-mask and stale-stat tests, then implement a shared source/representation helper and the 94D reader mode.
- [x] Add failing masked relative-stat tests, then implement a stats entry point using the same representation and actual window horizon.
- [x] Add explicit 80→94 checkpoint projection expansion and test preserved weights, new dimensions and rejection of unrelated shapes.
- [x] Update canonical profiles, training dimensions, trace/report and open-loop shape contracts; keep legacy absolute80 behavior accessible explicitly.
- [x] Regenerate bounded real-data stats and reports, verify next-state timing and relative transforms directly from raw parquet, and run focused plus LeRobot regression suites.
- [x] Obtain independent read-only review, resolve findings, and document the tested behavior and remaining source-TCP limitations.

## Verification

- 123 focused reader/stats/checkpoint/report/legacy tests passed.
- 276 LeRobot, mask, mixture and normalization regression tests passed (overlaps some focused utility tests).
- Independent raw-parquet audit checked t=7→state8:40, all chunk joint deltas and InternData TCP relative transforms; all four real buckets passed and terminal windows supervise exactly one target. Evidence: `outputs/action_alignment_audit/tcp_relative_joint94_evidence.json`.
- Four profiles/stats/reports regenerated. Independent review found and verified a fixed stats/reader episode-selection mismatch; no remaining blockers.
- Agent validated checkpoint expansion in the full environment with existing finetune loader tests: 20 passed. No GPU training or inference run.
- Changed/new feature files pass Ruff; broad base-reader lint exposes a pre-existing unused `file_col` outside the edited lines.
