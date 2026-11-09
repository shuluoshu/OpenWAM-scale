# History VAE Cache and Batching Implementation Plan

> Execution follows the approved scope from this chat. Use test-driven-development, verification-before-completion, and requesting-code-review. The two live DLC checkouts and configurations are immutable inputs.

**Goal:** Reuse frozen single-frame VAE latents across shuffled episodes and encode cache misses in bounded batches, retaining temporal PE, history masks and the 120x3072 C* interface.

**Architecture:** Keep history RGB on CPU during Stage1 preparation. HeadFeatureCache uses episode/content keys in a bounded multi-episode LRU, independent of age/dropout. GoalCodec caches each image separately and uses the native VAE batch_encode method in small chunks; legacy codecs retain a fallback. Frozen VAE changes invalidate caches. Conditional goal caches keep their separate episode policy.

**Tech Stack:** PyTorch 2.7.1, WanVideoVAE38, CPU LRU caches, pytest, safetensors; openwam conda environment on DSW.

## Constraints

- Base: 10dc74e. Work branch: codex/history-vae-cache-batching-20261109.
- Do not edit or restart either live DLC. Do not alter the active recovery checkout.
- Do not change time PE, dropout probabilities, sampling, loss or trainable architecture.
- Encode frames with temporal length one, using the batch dimension for parallelism.
- Cache frozen VAE outputs only. Invalidate on weight/device/dtype changes. Never encode invalid history slots.
- CPU tests precede implementation; CUDA measurements use identical inputs and synchronized boundaries, and report ambient contention and scope.

## Task 1: Codec and history cache

Files: openwam/model/goal/codec.py, history.py; tests/goal/test_history_cache_batching.py.

- [x] Add failing tests for multi-episode reuse, partial masks, per-image batch overlap, bounded batch_encode dispatch, frozen dependency invalidation and missing identity.
- [x] Confirm baseline failures are behavioral assertions.
- [x] Add GoalCodec(max_encode_batch_size=4), encode(image, device=None), per-image cache keys and native batch encoding with legacy fallback.
- [x] Change HeadFeatureCache to a bounded episode/content LRU; collect duplicate misses, batch valid frames, return to the explicit target device, and preserve invalid zero slots.
- [x] Verify cold/warm results, invalid inputs, dependency changes and cache limits.

## Task 2: Training and runtime integration

Files: goal_curated.py, visual_history.py, query_expert.py, patch_expert.py, runtime.py; existing history tests.

- [x] Add failing CPU-preparation test with a meta target device: history RGB remains on CPU while age/mask move.
- [x] Keep Stage1 history pixels on CPU; pass the model target device to history encoding.
- [x] Propagate bounded codec batch size to observation codecs.
- [x] Retain episode resets for conditional goal predictions, preserve raw VAE entries across known episodes, and update obsolete eviction assertions.
- [x] Run focused and broader goal/loader tests, including temporal PE and checkpoint reload.

## Task 3: DSW validation, review and delivery

Files: scripts/experiments/benchmark_history_vae.py; docs/experiments/2026-11-09-history-vae-cache.md.

- [x] Build a reproducible benchmark from exported real training RGB/history/masks and foundation VAE weights, with baseline and candidate imports in separate processes.
- [x] Measure cold shuffled batches and warm revisits, native single-frame latent numerical error, cache/miss calls, peak GPU memory, and synchronized per-batch times. Include a bounded Stage1 forward/backward measurement when feasible.
- [x] Record GPU background load, warmup/repetition count, model/input scope, and avoid extrapolating single-GPU results to 32-rank DLC throughput.
- [x] Obtain independent code review and resolve actionable findings, then run justified regressions.
- [x] Commit the independent branch; retain it for review, without merging or deploying.
- [x] Evaluate reusable benchmark workflow for skill creation/update, validate discovery if installed, and archive this chat with evidence and final delivery.
