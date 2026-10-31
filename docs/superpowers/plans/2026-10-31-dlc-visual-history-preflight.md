# Visual history DLC preparation implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development task by task; review each implementation before integrating.

**Goal:** Validate the current optimized Stage1 and frozen Stage2 on DSW before submitting a four-node priority-nine DLC job.

**Architecture:** Pin a clean remote revision and immutable offline relative112/horizon32 statistics. Prepare data and checkpoint adaptation before GPU allocation; missing supervision is masked per modality. Stage2 starts only after an explicitly defined Stage1 completion/convergence gate.

**Tech Stack:** PyTorch, safetensors, SQLite, OmegaConf/Hydra, torchrun, Alibaba PAI CLI.

## Global constraints

- Deployment checkout starts at remote dc695376dc829b99ee8f9e8e4863fa96bdb507e3; preserve active overfit checkout/processes.
- Four nodes, priority 9; verify selected quota and per-node shape through the live CLI where permissions allow.
- Reuse completed registry e534aa97c759551411d46c4827a9b242532a4bb42eff1b943d52a7584bae6668; fixed-anchor relative112, horizon32.
- Source selection is requested 20261030 YAML minus only the previously user-approved missing wipe_table_v1_test item. All 5702 remaining repository directories and necessary metadata passed current path checks.
- Both stages initialize from official OpenWAM foundation154000, with explicit four-projection width80-to112 migration for Stage2.
- No DLC submission before real single-GPU pipeline smoke passes. No claim of convergence based solely on update budget.
- User selected Stage1 minimum50,000 updates and evaluation every10,000 updates. Retain proposed maximum154,000, plateau patience5 and relative improvement1%; Stage2 budget154,000. No handoff if maximum reached without convergence. Fixed train-population diagnostics are not an independent held-out set; label evaluation scope honestly.

## Task 1: Fix missing-target mask boundary cases

**Files:** openwam/dataloader/curated_goal.py; scripts/data/trajectory_projection.py; openwam/model/architectures/base.py; focused tests under tests/goal and tests.

**Inputs:** Audit and exact reproducer at ../mask_audit.md and ../mask_audit_source_probe.py (relative to checkout root's parent). Source training.valid must mask future supervision separately from current head availability. Finite pose validity flags must be finite AND above threshold. Already masked NaN action targets must not poison flow-matching loss.

**Output contract:** Public data/model APIs stay unchanged; finite valid labels retain existing values; unavailable labels contribute zero loss and zero gradients, including entirely masked batches; genuine nonfinite predictions on valid labels remain detectable.

- [ ] Add regressions reproducing target training.valid=0 while current head/history remains valid; Inf validity flag affects only its arm; all-masked NaN action values have zero finite loss/gradient.
- [ ] Run regressions to show failure, implement minimal masking corrections, then rerun focused existing mask tests and two-rank CPU backward.
- [ ] Save report to ../task-1-report.md and commit only task-owned files.
- [ ] Independent spec and code review, followed by fixes if needed.

## Task 2: Official initialization and offline preparation

**Files:** scripts/pai/ official-checkpoint preparation utility and tests; production pipeline/stage configs; run artifacts on CPFS.

- [ ] Reuse existing strict action projection expansion, preserve all unaffected tensor values, explicitly record source and output hashes and four migrated shapes; do not reuse the local trained gripper checkpoint.
- [ ] Pin tokenizer/config beside derived checkpoint; baseline validators continue enforcing width112 after conversion.
- [ ] Prepare full Stage1 population and reusable Stage2 audit offline with exact source/config/statistics identity. New missing data requires user response.
- [ ] Review and test artifact corruption/identity rejection and idempotent reuse.

## Task 3: Bound multinode startup

**Files:** openwam/train/stage12.py, scripts/train_goal_epoch.py, openwam/dataloader/goal_epoch.py, openwam/train/utils/stage2.py, scripts/train.py, scripts/pai/stage12_entry.sh, focused tests.

- [ ] Consume an offline validated artifact bound to data/config/registry/implementation identity, eliminating 33 Stage2 metadata audits and duplicate Stage1 full scans during training startup.
- [ ] Keep strict legacy validation when artifact absent, reject stale/mismatched artifacts, and validate sampled source dependencies lazily.
- [ ] Bound per-node simultaneous checkpoint loads; make all ranks fail coherently on bootstrap errors.
- [ ] Require exact pinned clean revision including launcher scripts; run CPU multi-node orchestration regressions.

## Task 4: Stage boundary and live submission

- [ ] Resolve requested production budget and convergence semantics, implement deterministic fixed-evaluation gate and fail-closed maximum budget if requested.
- [ ] On free DSW GPU run native prepare/check/train Stage1 -> freeze -> Stage2, including missing supervision fixtures; inspect saved proof, finite losses, frozen teacher and action horizon32.
- [ ] Publish reviewed changes to current remote branch; compare remote tree; generate create-only request with four nodes and priority9; validate and CLI dry-run.
- [ ] Submit once, query returned JobId, and report observed queue state plus proof/config paths. Quota query currently denied pai:GetQuota; do not claim capacity known or change permissions/target.
