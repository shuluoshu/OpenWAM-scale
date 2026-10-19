# Multiview Observation / Head Goal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Track completed tasks in outputs/stage1_multiview_20261019/progress.json.

**Goal:** Add three-view current observation to Stage1 while preserving head-only image/TCP output and Stage2 goal memory.
**Architecture:** H384xW320 observation -> frozen VAE ->120 observation tokens; head goal noise ->80 tokens plus TCP noise ->1 token. Unequal observation/target grids, explicit missing-view mask, versioned checkpoint metadata.
**Tech Stack:** PyTorch/Wan, PIL, curated readers, existing Stage1 runners and Stage2 runtime, pytest/HTML.

## Global Constraints
Follow [the approved design](../specs/2026-10-19-multiview-head-goal-design.md). Preserve existing head-only behavior, original source workspace edits and running jobs. Use an isolated branch with guarded installation. No full training in this implementation task.

## Task 1: Core unequal-grid expert and versioned bundle
Files: openwam/model/goal/observation.py (new), patch_expert.py, openwam/train/goal_patch.py; tests/goal/test_multiview_patch.py.
Interfaces: observation_mode='head' (default) or 'multiview' on PatchGoalExpert; optional observed_view_valid passed to forward/generate/velocity. New observation helper owns clean_multiview(frames, view_valid) -> [3,1,384,320], sanitize_canvas + token mask; codec output spec stays head-only. v1 head and v2 multiview metadata strict. Export constants/helper used by later tasks. Optional explicit load_patch_bundle(... observation_mode='multiview', warm_start=True) or named helper for v1 warm-start; normal loader must never silently change mode.
- [x] Write tiny real-Wan tests for mixed masks/batch, target80+TCP1 output, valid-wrist sensitivity and missing-wrist invariance (including NaN missing pixels), prefix isolation, nonzero gradient, seed stability, v1/v2 roundtrip, explicit warm-start.
- [x] Run new tests red with /root/miniconda3/envs/openwam/bin/python -m pytest tests/goal/test_multiview_patch.py -q.
- [x] Implement separate patch grids, per-grid RoPE, mask and per-mode observation VAE shape handling; keep output head latent and old v1 numerical contract.
- [x] Run tests/goal/test_patch_expert.py tests/goal/test_goal_patch.py and new tests; write report with commands/results/owned files.
- [x] Root installs/commits; task review spec and quality gate.

## Task 2: Native three-view Stage1 data, runners, initialization and reports
Files: curated_goal.py, goal_curated.py, goal_overfit.py (only common forwarding/export where necessary), scripts/train_goal_curated.py, configs/goal/stage1_multiview_{sample,3000h}.yaml; necessary tests. Patch runner/evaluate_patch/train_patch forwarding in goal_patch.py must accept optional observed_view_valid (coordinate with prior task).
- [x] Test current same-anchor head/wrist decoding (shared-file timestamp offsets), missing wrist explicit masking, unchanged head endpoint and UV, sample/3000h preset inheritance, mask retained device transfer/export/forward/evaluation, explicit v1 initialization.
- [x] Implement observation_mode in CuratedGoalDataset; use canonical native family camera profile rather than guessed names. Decode only current wrist frames. Preserve raw source provenance and view_valid in record plus actual-model-input NPZ/HTML.
- [x] Add new config presets, --init-goal-checkpoint (fresh optimizer, dependency identity checks), synthetic multiview smoke honoring config, existing head-only behavior intact.
- [x] Run focused reader/runner/report regression and tiny synthetic training with immutable step reports. No full model training.
- [x] Root installs/commits; task review gate.

## Task 3: Stage2 runtime integration, strict config and final verification
Files: runtime.py, architectures/base.py, curated_stage2.py, stage2_eval.py, train/utils/stage2.py, openwam_trainer.py as needed, configs/goal/stage2_multiview_{sample,3000h}.yaml, tests, docs.
- [x] Test v1 vs v2 input dispatch, explicit clean multiview + view mask through training/generation/eval, cache mask identity, config/checkpoint mismatch rejection, serialization/resume policy preservation.
- [x] Implement 120 current observation token conditioning in frozen v2 expert while keeping adapter80, generation81 and head output unchanged. New args goal_observed_multiview / goal_view_valid; head args legacy intact. Use actual runtime bundle mode, strict configured policy agreement.
- [x] Update preflight supported formats/policies, saved-run goal configs and inference docs. Add stage2 presets expecting trained v2 bundle.
- [x] Run all goal/model + affected dataloader/checkpoint regression once changes settle; bounded sample real-data preflight and validate HTML, inspect actual input images; preserve all old reports.
- [x] Complete final review and fixes with regression evidence. Publish this verified tree to the same remote branch; release SHA and delivery checks are recorded in the HTML report below. No new 5k model training has been executed.

## Delivery evidence

Actual native input report: [DSW HTML](https://897628-proxy-18081.dsw-gateway-cn-shanghai.data.aliyuncs.com/outputs_all/stage1_multiview_20261019/index.html) (Alibaba Cloud login required). The report links execution logs, source-frame alignment results and release status. See [training and inference usage](../../multiview_goal_training.md).
