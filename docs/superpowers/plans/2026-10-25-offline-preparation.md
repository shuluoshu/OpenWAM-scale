# CPFS offline preparation and H200 resubmission

**Goal:** Complete reusable offline preparation and real-source verification before allocating a priority-9, four-node/eight-GPU-per-node H200 job.

**Constraints:** All prepared data, caches, diagnostics, configuration, logs and checkpoints reside under `/mnt/data/cpfs/`. Preserve all 5904 selected sources and task/subtask labels, 3 FPS population semantics, and completed index transactions. Do not silently discard decode failures. Do not allocate the new job until preparation and validation succeed.

## Work items

- [x] Stop and verify termination of `dlc1rdrggbuyd7kr`; retain its index.
- [x] Diagnose Mecka average-rate disagreement against real decoded ordinals and source metadata. Add a failing regression test, fix sequential decoding without weakening CFR alignment, and replay failed sources through the real model.
- [ ] Accelerate `openwam/dataloader/goal_epoch.py` metadata preparation with bounded parallelism and durable batched transactions; prove identical sample IDs, source validation and resume behavior against serial generation. Publish progress on CPFS.
- [ ] Complete Stage2 metadata audit and bounded normalization statistics on CPFS. Reuse prepared outputs across the new GPU job.
- [ ] Require completed offline artifacts before submission. Run relevant tests, review diffs, commit and publish the fixed version to the existing remote branch.
- [ ] Validate the exact four-by-eight H200 request, dry-run, submit at priority 9, query the result and attach the existing three-hour Feishu monitor to the new job.
- [ ] Deliver the DLC link, observed state and exact CPFS artifact paths.

Discovered during full-population preparation: Morphic episodes cross Parquet file boundaries. Repair Stage1/Stage2 and offline goal readers to assemble all contributing files, retain strict row validation, and record every source dependency. Real boundary episodes and all original 5904 sample selections are verified; existing Stage2 statistics regenerate byte-for-byte identically.
