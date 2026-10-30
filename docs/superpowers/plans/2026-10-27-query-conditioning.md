# Query conditioning implementation plan

**Goal:** Compare three condition injection paths on a real subtask with the same Wan.
**Architecture:** See ../specs/2026-10-27-query-conditioning-design.md; user approved.
**Tech:** PyTorch, Wan, existing query bundle, CPFS, static HTML.

- [x] Test modes, old exact behavior, MLP bypass, prefix causal isolation, gradients,
  bundle roundtrip and Stage2 encoder compatibility.
- [x] Implement decoder context hook + prefix velocity in a focused module;
  support optional no-cross-attention Wan block without changing default calls.
- [x] Wire mode into constructors, presets and bundle metadata; freeze unused
  parameters and exclude them from optimizer groups.
- [x] Prepare one real subtask; audit original frames, processed target and TCP GT;
  cache frozen encodings only, preserve actual instruction and history metadata.
- [x] Run matched real-model training with fixed noise and non-overwriting reports;
  measure condition sensitivity, memory and speed separately from eval overhead.
- [x] Validate report serving/assets; document flags, comparison and limitations;
  commit code and export patch on CPFS.
