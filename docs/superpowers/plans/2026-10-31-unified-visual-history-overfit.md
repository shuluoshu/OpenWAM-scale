# Unified visual-history Stage1 / frozen Stage2 overfit

## Global constraints
- Preserve the dirty starting worktree and unrelated changes; no data/checkpoint overwrite.
- User authorized remote push to fork and training: Stage1 exactly 1000 optimizer updates, then frozen Stage1 and Stage2 joint / none exactly 500 updates each.
- Select three different datasets, one episode each. Report actual selections and limits; no claim that three episodes certify whole datasets.
- Missing head camera must mask corresponding TCP supervision and loss; keep valid independent gripper / local TCP supervision.
- Use optimized VisualHistoryGoalExpert and its frozen encoder in Stage2, not query-only fallback.
- Visualization follows openwam-visualization: complete subtask/task, inference every 10 source frames plus last, GT/pred two rows, current head/wrists plus Stage1 goal, honest projection coverage.

## Task 1: Unified Stage1 integration and head-mask correctness
Integrate visual_history into generic goal_curated / train_goal_epoch entry and checkpoint/resume/Stage12 handshake. Support bounded optimizer update overfit without falsely asserting complete finite-population epoch. Save portable visual_history bundle and expose exactly completed update count. Add reusable visual-history config. Audit missing-head masks through reader and model loss (NaNs in masked targets must not poison loss); add meaningful regression tests. Review Stage2 already-supported visual history encoder and ensure pipeline contracts accept it. Own openwam/train, curated_goal.py, relevant model loss if necessary, scripts/train_goal_epoch.py, configs, focused tests. Do not edit experiment scripts or curated_stage2.py without coordination. Do not commit/push independently: controller stages selected final files.

## Task 2: Reproducible three-episode run
Prepare explicit source/index/stats/caches and unified configs for three useful datasets. Reuse actual native reader mappings and evaluate projection prerequisites. Execute Task 1 unified Stage1 entry for 1000 updates, fingerprint teacher; execute same frozen teacher with Stage2 joint and none for 500 each. Preserve metrics, status, seeds and exact commands. Add reusable bounded orchestration if needed. Tests cover run contracts and exact update stopping.

## Task 3: Review, visualization, publication
Review integration and remedy substantive issues; run targeted tests. Generate fullspan six videos and coverage report via installed visualization skill; validate decode, frame anchors, HTTP Range. Publish HTML under Alibaba DSW proxy. Commit relevant implementation/config/tests/scripts and push fork branch without force; verify remote SHA. Final report includes code and HTML links, K/N, datasets and observed limitations.
