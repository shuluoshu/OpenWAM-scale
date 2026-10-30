# History window and episode isolation implementation plan

User-approved requirements: use ten 1 Hz history slots within the preceding ten seconds, encode realized age relative to current frame, randomly mask history during training, clear feature caches on episode changes, mask all history at episode start, expose one history-enabled switch.

Implementation in the existing isolated worktree, without modifying live pinned snapshots:
- [x] Add regression tests for fixed sampling, dropout, episode-start padding, A/B/A cache switching, and the single config switch.
- [x] Extend HistorySamplingConfig with frame_dropout_prob (preset 0.2, evaluation 0); fixed sampling uses 1..10 second offsets. Preserve explicit legacy jitter configuration for old experiments only.
- [x] Carry a stable int64 history_episode_id derived from canonical source path and episode index through both training readers, batch transfer, encoder, Stage2 runtime/cache and inference entry point. Missing IDs disable cross-call feature reuse conservatively.
- [x] Clear the entire feature LRU at each row's episode change. Invalid slots remain zero. Retain prior explicit clearing for incomplete histories.
- [x] Use history_enabled in sample/3000h inherited configs for both model and reader; update Stage2 presets similarly. Document compatible-checkpoint requirements and current-run deployment limitation.
- [x] Run data, query, Stage2 and config tests, review changes, export a patch on CPFS. Do not stop/restart existing jobs without resolving the user's earlier instruction to preserve them.

Model architecture scope: explain, but do not silently remove, readout heads, constant prefix, or the learned-Z-to-text-adapter path. These are separate architecture changes with checkpoint compatibility implications.
