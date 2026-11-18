# Stage1 goal observation validity implementation plan

**Goal:** Recover valid endpoint observations without changing endpoint selection or Stage2 action validity; genuinely generate even when GT is invalid.

**Architecture:** Add a source-gated Stage1 validity policy to the curated reader and a target-independent G generation helper. Keep archived production helpers immutable; bind the maintained generation helper through the repository entrypoint and include the adapter in the runtime identity.

**Constraints:** No DLC/training launch; G5515/C*3321 weights and optimizer files stay unchanged; task/subtask 3:7, causal 1Hz history, KF1 fixed0 and end_exclusive-1 remain unchanged. Unknown source semantics retain the strict mask. Canonical-invalid targets remain invalid. RGB validity never invents TCP validity.

- [x] Add failing policy, reader and GT-independent generation regressions.
- [x] Implement policy, per-target camera masks, label identity and generation adapter; preserve history cache identity and archived helpers.
- [x] Run focused regressions and snapshot integrity checks.
- [x] Rebuild seven endpoint labels in a separate prepared generation; verify source frame, checkpoint and unchanged input identities.
- [x] Run bounded DSW inference for both models, preserving seeds and original diagnostics; publish a Chinese before/after HTML at the existing 10.x:8012 service.
- [ ] Review, verify remote tree, push the new branch and archive delivery evidence.

Files: `openwam/dataloader/goal_validity.py`, `curated_goal.py`, `rfc_goal.py`; `openwam/train/goal_generation.py`; `training/stage1_longtrain_16gpu/entrypoint.py`; targeted tests; a reproducible DSW validator and training migration notes. Historical `runtime/`, `reviewed/`, prepared cohorts, source data and checkpoint bundles remain byte-for-byte intact.
