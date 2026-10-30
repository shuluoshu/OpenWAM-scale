# C prefix conditioning production integration

> Execute inline using executing-plans. User selected the existing, tested Wan prefix design; no further architecture choice is pending.

**Goal:** Expose C for sample and 3000h Stage1 training and the existing one-epoch Stage1 → frozen-query Stage2 pipeline.
**Architecture:** Reuse `prefix_self_attention` and strict bundle metadata. Stage2 consumes the same encoder Z; it does not load the auxiliary decoder. Preserve legacy defaults.
**Tech stack:** PyTorch, OmegaConf/Hydra, existing finite-epoch trainer and tests.

- [x] Add regression tests for disjoint trainable parameter groups with explicit `condition_learning_rate=1e-4`; preserve omitted-option behavior. Verify red.
- [x] Add the optional learning rate to shared grouping, curated and epoch trainers and their CLI wiring; validate invalid modes/options before model loading. Run targeted tests.
- [x] Add C sample/3000h/finite epoch presets and single-node sample / 4×8 3FPS posttrain pipelines. Keep history10s, 1Hz, dropout0.2, Stage2 encoder freezing, old configs and running tasks intact.
- [x] Validate composed Stage1/Stage2 query/history compatibility, real tiny Wan forward/backward and C bundle→encoder-only Stage2 loading. Include real C overfit bundle metadata audit.
- [x] Document architecture, launch commands, dataset/config switching, CPFS outputs, fresh-run requirement and previous overfit results. Run relevant regression suite and commit only this integration; keep unrelated report changes separate.

Validation: 125 focused regressions passed; actual CLI synthetic C/history run completed two updates; real ABC C checkpoint encoder-only extraction produced frozen finite [1,66,512] tokens from 10 history frames. Full Stage2 training and 3000h population preparation were not launched.
