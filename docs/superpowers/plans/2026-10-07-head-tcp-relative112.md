# Simultaneous head-reference and TCP-local EE targets

Preserve existing 94 slots; append head-left pose9 at 94:103 and head-right
pose9 at 103:112. Raw representation is TCP/gripper20 + joints14 + head18.
Targets use future states t+1 onward and a fixed window-start anchor.
Head delta rotation is R_camera R_future R_current^T R_camera^T; TCP-local
rotation is R_tcp_current^T R_tcp_future. Masks are per component and side.
No head calibration or image means no head pose supervision. No declared TCP
means no TCP-local supervision. Joint/gripper masks remain independent.
Keep legacy 80/94 readers working. No source dataset writes or GPU training.
User confirmed: both branches MUST use the identical declared TCP physical point.
Without TCP both pose branches mask. Convert native TCP to common world per row
when tcp_reference_to_world_key is configured (InternData robot_to_env_pose).
No flange/canonical EEF fallback. Missing head only masks head branch.

- [x] Extend explicit checkpoint padding to 80/94 -> 112; retain strict shape checks.
- [x] Implement representation and reader/config/statistics integration with geometry tests.
- [x] Generate four profiles' stats and verify real sample masks and frame math.
- [x] Update documentation and visualization consumers, review and run relevant tests.

Validation: 76 targeted CPU tests passed; Ruff and git diff --check passed.
Four real profiles verified at anchor7/targets8..39; independent TCP/head matrix
check max component error 1.44e-7. Mixture16 windows, model/reader112 dimensions
confirmed. Review /tmp/review-head112.md: no blockers. No GPU run.
