# GPU optimizer and training input revision

User authorization: adjust the compared training configuration, publish the revised version, and push a remote branch. Existing pinned DLC jobs remain unchanged. This is a new-run recipe, not a checkpoint migration.

## Design

Keep Stage1 DDP gradient semantics, but use PyTorch ZeroRedundancyOptimizer (ZeRO-1) to partition GPU FP32 master weights and Adam moments. Each owner updates its parameters in FP32 then broadcasts BF16 model weights. ZeRO-2 for Stage1 would require replacing the trainer/checkpoint engine; ZeRO-1 removes redundant optimizer memory while preserving the existing verified loss/mask and handoff logic. Preserve the old unsharded optimizer for old recipes. Make beta/eps/weight-decay explicit, with production beta2=0.95.

Stage2 retains native DeepSpeed ZeRO-2 but disables optimizer CPU offload in new profiles. Native probes exposed foreach temporary-memory OOM and a confirmed Torch 2.7.1 fused large-flat-tensor overrun. Final profiles select single-tensor GPU AdamW (foreach=False, fused=False), with CUDA save/reload regression and legacy backend defaults preserved. Start with equal effective batch budgets across hardware: Stage1 batch32/card, Stage2 batch8/card; increase only if native capacity probes justify it. Readers remain bounded at Stage1 eight/card and Stage2 four/card. Enable pinned Stage1 inputs with explicit custom-target support. Remove Stage1 training PNG/NPZ roundtrips without changing exported evaluation inputs, RGB values or labels. Stage2 none reads the current image and past history only; joint still reads nine times. Preserve 32x112 relative actions and masks. Mecka's VFR exact-ordinal decoder remains unchanged until an independently validated seek index exists.

## Work / verification

- [x] Implement GPU sharded FP32 AdamW in a leaf module, explicit optimizer config, and collective checkpoint preparation. Compare multi-process Gloo updates/state to the old full FP32 reference, including missing gradients, sub-BF16 updates, multi-LR groups and save/resume. Keep old checkpoint tests passing.
- [x] Remove Stage1 training intermediate files, retain evaluation assets, support pinned target tensors. Add numerical-parity and missing-view/TCP tests.
- [x] Avoid future-video decoding for Stage2 none while preserving current/history, action/FK targets and augmentation; fail mismatched configuration. Add decoder-call and loss/shape tests.
- [x] Add separate versioned production recipes and hold-mode DLC templates, same data/statistics/foundation and convergence gate; global batches equal across hardware, explicit GPU optimizer and Adam hyperparameters. New output directories and prepared proofs required.
- [x] Run focused CPU/distributed tests and review changed files. Document the final before/after, then commit/push the codex branch. User explicitly stopped further DSW parameter/capacity validation; full Stage2 stability, capacity and throughput are deferred to DLC. Do not claim a passing single-GPU Stage2 capacity result.

No mutation to the active deployed checkout; no DLC job restart/submission in this request. No change to relative action statistics, mask validity, cadence, or convergence budget. Benchmark scripts and evidence go to outputs; reusable validation scripts may be committed.
