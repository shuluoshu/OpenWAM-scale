# DLC reserved launch and Stage1 throughput implementation plan

> Execution: superpowers:subagent-driven-development. The user has requested the implementation and the previously discussed batching/prefetch direction. Existing live jobs remain untouched while preparing and validating replacement artifacts.

Goal: use measured per-device memory/throughput to select real batches and data workers, and schedule DLC containers with `sleep 36000000` rather than automatic training.

Constraints:
- Work only in this isolated worktree; never edit the CPFS checkout used by active jobs.
- Preserve relative112/32 offline normalization, dataset selection, random finite order, missing-view/label masks, and Stage1 convergence before frozen Stage2.
- Batch size means samples processed together, not a renamed accumulation count. Global batch is local batch times world size.
- Workers load only CPU data using spawn, deterministic sample seeds, worker-private readers/caches, and bounded prefetch. Checkpoints count consumed samples, not prefetched ones.
- Training errors must reach peers; no deadlocks or silent dropped samples, including uneven population tails.
- GPU memory tuning includes forward/backward and initialized optimizer state, using real data and latency/throughput measurements. Do not call untested defaults optimized.
- Reservation UserCommand is exactly `sleep 36000000` (shell seconds). It must never auto-start training. Separate explicit per-node start scripts retain runtime hash, pinned code, preparation/config checks and prevent duplicate starts.

Tasks:
1. Real Stage1 batch collation and finite-rank prefetch iterator, integrated into train_epoch and CLI config. Tests: mixed masks/history/context lengths, deterministic worker delivery/resume, partial tails and 2-rank synchronized updates. Default batch 1/workers 0 preserves old callers.
2. Separate DLC reservation and explicit launch. Tests: default UserCommand sleep only, no runtime load at scheduling, explicit script retains code/config checks, shell quoting and duplicate-start guard.
3. Check batch-aware model operations and test real optimizer updates on DSW for candidate batch sizes 1/2/4/8 and worker settings 2/4/8. Record peak memory and samples/s; validate CPU/GPU memory budget classes separately.
4. Integrate reviewed changes, run focused regression checks, publish remote branch and immutable CPFS deployment, prepare per-memory configuration and reservation submissions. Report concrete paths and exact validation limits. Do not relaunch old unoptimized training automatically.
