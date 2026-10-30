# Compressed head history implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development or superpowers:executing-plans task by task. Preserve the paused experiment; no long training launch.

**Goal:** Verify real subtask conditioning and add inexpensive timestamp-aware history to the current query-latent model.

**Architecture:** Ten strictly past head frames become sixteen learned spatial tokens each. Frozen single-frame VAE features are cached before a trainable compressor. The 160 tokens join current observation and text as read-only memory in the small query encoder. Its 64 image and 2 TCP output queries, auxiliary joint-flow decoder and Stage2 sequence lengths remain unchanged.

**Tech stack:** PyTorch, existing Wan VAE/T5, native curated readers, safetensors, pytest.

## Global constraints

- Existing history-disabled checkpoints/configurations must retain their original behavior. Default history disabled.
- Head only, 10 past slots, oldest first; current frame excluded. Missing/duplicate/invalid slots masked, never substituted with a valid current or future frame.
- Training samples cumulative uniform gaps in [0.5, 1.5] seconds (mean 1 second); evaluation uses fixed 1-second gaps. Encode realized physical ages, not slot numbers. History may cross a subtask boundary in the same episode, but stays within the selected contiguous curated range.
- Frame timestamps follow the video decoder's validated CFR ordinal/FPS contract. Report timestamp source explicitly; never claim ordinal positions are source sensor timestamps.
- Clean head resize matches the existing single-head preprocessing; no history augmentation. Target images/TCP transforms stay unchanged.
- Cache detached pre-compression VAE latents with bounded CPU storage and content identity. Trainable compressed outputs are never cached.
- Compressor uses FP32 parameters/compute, nonzero initialization, separate learning rate. Stage1 losses train it; Stage2 freezes it with the existing query encoder. No claim of action-loss training through the frozen encoder.
- Existing stopped training must not restart. Only bounded tests and diagnostics.

## Task 1: Native historical inputs and language evidence

Files: `openwam/dataloader/head_history.py`, `curated_goal.py`, `curated_stage2.py`, `tests/dataloader/test_head_history.py`.

Interface: `HistorySamplingConfig(enabled=False, num_frames=10, mean_interval_s=1.0, jitter_fraction=0.5)`. `select_history_frames(anchor, fps, start, config, rng, training)` returns fixed-length indices (-1 padding), ages in seconds (0 padding), bool valid mask. Selection is deterministic for supplied RNG, strictly past and distinct. `materialize_head_history(...)` uses the source head video offset, validated decoding and clean head resize, returns `history_images [10,3,1,256,320]`, `history_ages [10]` FP32, `history_valid [10]` bool, plus JSON trace.

- [x] Test sampling boundaries, duplicate suppression, deterministic seeds, nonuniform gaps and real ages; run failing tests.
- [x] Implement standalone sampling/materialization; attach batch singleton fields in Stage1 and unbatched sample fields in Stage2. Do not decode wrists for history.
- [x] Verify task/subtask selected instruction reaches the text cache unchanged with a capturing encoder; record code locations and an actual example.
- [x] Run new and existing reader tests.

## Task 2: Learnable history compression, cache and Stage1 plumbing

Files: `openwam/model/goal/history.py`, `query_expert.py`, `openwam/train/goal_query.py`, `goal_curated.py`, `goal_patch.py`, `scripts/train_goal_curated.py`, `tests/goal/test_query_history.py`.

Interface: `HistoryConfig(enabled=False, num_frames=10, tokens_per_frame=16, compressor_dim=128, cache_size=256)`, supplied as `history_config` to query encoder/expert. `extract_tokens`, `forward`, `generate` accept optional `history_images`, `history_latents`, `history_ages`, `history_valid`. Latents shape `[B,10,48,1,16,20]`; exactly one image/latent source when enabled. An explicit all-false history is valid; missing history for an enabled model is rejected.

- [x] Write failing tests for masked NaN safety, timestamp sensitivity, fixed query shape, frozen VAE/gradient boundary, cache/live equivalence and old/new bundle roundtrips.
- [x] Compress VAE grids with learned 4x4 queries attending projected spatial features, with learned spatial position. Use no pooling bypass. Add Fourier age encoding after compression; no historical token reads current observation, text or another historical frame.
- [x] Append compressed memory only to the lightweight transformer; keep the history-off branch identical. Cache pre-compressor latents using per-image content and VAE parameter identity; clear on model device/dtype changes, expose explicit reset.
- [x] Save/restore history configuration and encoder-owned compressor tensors. Wire Stage1 batch transfer, train/eval calls, FP32-master optimizer groups and separate history learning rate.
- [x] Test gradients and tiny real-Wan joint image/TCP smoke without full training.

## Task 3: Stage2, launch configs, diagnostics and review

Files: `openwam/model/goal/runtime.py`, `openwam/model/architectures/base.py`, `openwam/train/utils/stage2.py`, `openwam/train/stage2_eval.py`, `configs/goal/*query_history*.yaml`, `docs/query_history_training.md`, `tests/goal/test_query_history_stage2.py`.

- [x] Add explicit history arguments to public inference/training paths and include history pixels/latents, ages and masks in final-goal cache identity.
- [x] Validate model/dataset/checkpoint history agreement. Frozen Stage2 uses identical query extraction and no auxiliary decoder. Retain exact no-history behavior.
- [x] Provide smoke and 3000h Stage1/Stage2 configurations and commands, model diagram, sampling/cache contract and reset guidance.
- [x] Export a bounded real-data history/input audit with instruction, frame ordinal, actual age and masks; do not modify GT or resume training. Benchmark query-only overhead separately from cold/warm VAE cost.
- [x] Run relevant regression suites; perform independent review and fix concrete findings. Document evidence and limits.

## Reference

OpenDM `fe1ed79a22d81cc1bfbeb2b4418bbb79fc75fc50`, `opendm/data/transforms.py:233` (history selection) and `opendm/model/dm05/dm05_arch.py:1590` (pooling/injection). This implementation adapts that idea to OpenWAM VAE/query memory; it does not copy the VLM token-ID or prefix-cache layout.

## Completed verification

318 relevant tests pass. Four real task/subtask input cases redecoded exactly; two-step tiny Wan smoke completed. H20 benchmark includes cold, warm and external latent cache paths. Independent review findings (FP32 cast precision; source-invalid history) fixed and re-reviewed. No full training, no merge/push. Evidence: repository-root `outputs/head_history_20261023/`.
