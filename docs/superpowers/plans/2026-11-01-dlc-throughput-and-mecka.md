# DLC throughput and exact Mecka reads

> Execution: independent decoder, optimizer-checkpoint, and benchmark tasks use the dispatching-parallel-agents workflow; the primary agent owns deployment and integration.

**Goal:** Compare actual Stage1 training throughput on the two existing 4×8 GPU DLC reservations, fix repeated Mecka sequential decoding, and publish measured configuration choices.

**Architecture:** Keep the running production checkout immutable. Implement a standalone, finite benchmark using the production model, shuffled population, masking, optimizer and offline statistics. Admit its base configuration using a freshly generated runtime proof, then vary only explicitly allowed performance controls. Cache exact VFR ordinal/PTS/keyframe information outside raw data, with pixel-equivalent reference validation. Replace slow optimizer checkpoint byte conversion while retaining the existing checkpoint format.

## Constraints

- No DSW GPU capacity or training experiments; all GPU comparisons run in DLC.
- Retain the current source YAML after the previously approved missing-path exclusion, the offline relative112 statistics, 3 Hz anchor population, BF16 model, FP32 optimizer masters and original supervision.
- H100 reservation `dlc17vidcbxadyj2`: eventual Stage2 joint; H200 reservation `dlc18fhk3v9fifql`: eventual Stage2 none. Stage2 remains gated on Stage1 convergence and freezing.
- Preserve existing checkpoints/logs. Existing production processes have no on-demand checkpoint handler; stopping loses only unsaved validation updates, not their existing artifacts. Keep the allocation's `sleep 36000000` alive.
- Do not modify running deployment code, raw videos or original dataset metadata.
- Benchmark output is not a converged model, resumable production checkpoint, or Stage2 admission proof.

## Task 1: Exact indexed Mecka decoder

Files: `scripts/data/prepare_subtask_goals.py`, a dedicated module under `openwam/dataloader/utils`, matching tests and a selected-video index builder.

- [ ] Write tests for VFR presentation ordinals, nonzero starting PTS, B frames, random requested ordinals, stale/corrupt cache and concurrent publication.
- [ ] Implement source-bound index publication, nearest valid keyframe seeking, exact PTS/pixel verification and sequential fallback on ambiguous or unsupported sources.
- [ ] Expose `OPENWAM_MECKA_DECODE_MODE=sequential|indexed` and `OPENWAM_VIDEO_INDEX_CACHE`; report indexed-read and fallback evidence.
- [ ] Compare old/new RGB hashes and PTS on real selected Mecka videos, separately recording cold index build and warm read times.
- [ ] Inventory selected unique video paths and prepare benchmark source indices before the optimized comparison.

## Task 2: Fast optimizer checkpoint collection

Files: `openwam/train/goal_optimizer.py`, focused checkpoint tests.

- [ ] Reproduce the bytearray-to-ByteTensor slow path with a regression test.
- [ ] Implement efficient buffer-backed serialization transport, CPU state snapshots and destination-only deserialization, preserving `openwam.zero1_fp32.v1`.
- [ ] Validate two-rank save/restore equivalence for model weights, exact FP32 masters, Adam moments and steps with Gloo; run NCCL verification on DLC.
- [ ] Measure a real checkpoint separately from measured training steps.

## Task 3: Reusable real-data benchmark

Files: `scripts/benchmark_stage1.py`, `openwam/train/goal_benchmark.py`, focused tests.

- [ ] Test matrix parsing, semantic-override rejection, warmup exclusion and sample-rate calculations.
- [ ] Bootstrap the production model once, reset trainable state and optimizer between cases, and recreate readers/caches consistently.
- [ ] Run real forward/backward/clip/optimizer updates with the native masks and shuffled data. Measure per-rank loading, preparation, forward/backward, optimizer and total time; summarize maximum and median across ranks.
- [ ] Record finite loss/gradient checks, valid TCP counts, memory peaks, true global samples/s, input population identity and all performance controls.
- [ ] Bound distributed failure detection to at most five minutes and terminate a failed matrix without unsafe CUDA recovery.

## Task 4: DLC experiment and selection

- [ ] Record current live job/step/checkpoint status and retain original outputs.
- [ ] Commit integrated code, deploy a new immutable shared checkout and prepare matching runtime artifacts.
- [ ] Stop only the existing training child trees, verify GPU compute processes exited, keep sleep reservations running.
- [ ] Screen batch 32/64/128 with eight readers and sufficient singleton prefetch; optionally extend the larger GPU class only if headroom supports it.
- [ ] Sweep workers 4/8/16 and prefetch for the strongest batch candidates, one control at a time; include a same-config sequential-versus-indexed Mecka comparison.
- [ ] Recheck finalists with at least 20 measured updates and disclose cold/warm-cache conditions, variability, OOMs and sample windows.
- [ ] Publish measured tables and configuration files, push reviewed changes to the remote branch, and clearly distinguish tested recommendations from pending production rollout.

## Initial evidence

The prior `bef4521` run uses B32, eight readers and prefetch four on both GPU classes. H100 approximately 20 samples/s is below its previous B128 CPU-optimizer run's 34 samples/s. Native Torch 2.7.1 ZeRO checkpoint collection serializes each shard through `torch.ByteTensor(bytearray(...))`; observed initial consolidation is approximately 16 minutes. Mecka VFR decoding repeatedly traverses frames from the beginning; stack samples observed this path during initial diagnostics. These are measured targets, not assumed improvements.
