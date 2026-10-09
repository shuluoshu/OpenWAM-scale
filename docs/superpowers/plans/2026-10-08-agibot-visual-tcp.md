# AgiWorld visual TCP correction

Goal: Replace the default 100/128 mm origin-only comparison with a useful virtual fingertip-center preview in the existing videos, as requested by the user.

The user explicitly authorizes URDF inspection and manual visual adjustment. Source A2D/Beta assets are G1, not the example G2. Verify inv(T_base_flange) @ T_base_link against direct flange-chain FK at several arm configurations; read actual pad meshes through their visual transforms. Compare candidate lengths on the screenshot's task362/frame216 and independent task365/410 frames. Preserve source calibration and keep training configuration untouched.

Implementation: record the CAD evidence and report-only visual definition; use canonical-flange TransX(0.240m), identity rotation, explicitly selected for the known A2D samples. Re-render all four Agi report clips and snapshot data with this same definition. Make current-point markers visually distinct from rolling future paths; show old named points separately. Include code/formula audit, transform, adjustment basis and visual limitations in HTML.

Validation: policy tests must reject implicit promotion of named centers and invalid transforms; confirm local-axis multiplication and masks. Verify geometry chain invariance, per-clip TCP offsets, source-state/frame alignment, relative reconstruction, all regenerated MP4 frames, HTML media hashes, offline ZIP and served report. Inspect new video frames across three tasks and the screenshot timestamp. No independent metric calibration claim.

Completed: formula/CAD audit, explicit visual definition, 20 regenerated videos and static report, 58 tests, media/pose/HTML/HTTP checks. Visual review includes exact screenshot frame216 and three additional frames per task. Physical calibration remains unverified.
