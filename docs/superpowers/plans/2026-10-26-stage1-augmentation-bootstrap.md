# Stage1 observation augmentation and parallel bootstrap

> Execute in the existing isolated worktree. User requested implementation and publication to `shuluoshu/OpenWAM-scale:codex/query-latent-goal`.

**Goal:** Enable the previously discussed three-camera perturbations in Stage1 and remove globally serialized rank initialization while retaining the completed global shuffle fix.

**Architecture:** Reuse `ObservationAugmentation` on the current composed PIL image before tensor conversion. Seed augmentation separately using sample identity; never modify target images, TCP labels, instruction, or historical frames. Bootstrap ranks in bounded local-rank waves across all nodes, synchronize failures after each wave, and run the codec validation after moving the VAE to its GPU.

**Constraints:** No modification of the running job or its checkout. Data and verification artifacts remain on CPFS. The existing finite population/index remains reusable; training must use a new output directory because the configuration/checkpoint identity changes. CPU/Gloo tests cannot establish actual 32-GPU NCCL startup time.

## Task 1: Deterministic Stage1 observation perturbations

Files: `openwam/dataloader/curated_goal.py`, new `openwam/dataloader/transforms/goal_observation.py`, `configs/goal/stage1_multiview_3000h.yaml`, `configs/goal/stage1_multiview_sample.yaml`, `tests/goal/test_goal_observation_augmentation.py`.

- [ ] Write tests with decoded three-view fixture: deterministic current-image changes, missing-camera exclusion, untouched goal/TCP/history/instruction, clean evaluation, restoration of RNG state, and disabled backward compatibility.
- [ ] Run tests and confirm failure from the missing augmentation configuration/path.
- [ ] Add a strict image-only adapter using existing color/patch/view operations; reject state perturbation and image augmentation on the head-only layout. Use a private seeded Python RNG context restored on exit.
- [ ] Apply after resize/composition and before saving `observed.png`; record the transformation trace. Preserve structural view validity so a physically present but hidden view is distinguishable from a missing camera.
- [ ] Enable color `p=.8`, brightness/contrast/saturation `.2`, patch `p=.5`, area `.05–.2`, aspect `.5–2`, view dropout `p=.1`, keep-one-view, in multiview sample/3000h presets. Query/history/epoch presets inherit.
- [ ] Verify with actual tiny query model forward/backward and focused regression tests.

## Task 2: Bounded parallel rank startup

Files: new `openwam/train/bootstrap.py`, `scripts/train_goal_epoch.py`, `openwam/train/goal_trainer.py`, `configs/goal/stage1_query_history_epoch.yaml`, `tests/goal/test_parallel_bootstrap.py`.

- [ ] Write real Gloo multiprocess tests demonstrating concurrent execution across simulated nodes, per-node bounds, exactly-once callbacks, and all-rank error propagation. Test invalid/mismatched topology/configuration before callbacks.
- [ ] Confirm tests fail because the coordinator is absent.
- [ ] Implement homogeneous torchrun topology validation and waves based on `LOCAL_RANK // workers_per_node`, gathering errors between waves. Default two concurrent loaders per node; print flushed phase/timing events.
- [ ] Replace the global `for loading_rank in range(world)` loop with the coordinator. Keep weights construction/loading on CPU, then move to each rank's device. Defer the startup VAE encode to GPU without removing the check; other callers keep the old default check.
- [ ] Verify bounded scheduling plus launcher integration with injected small components, and run existing distributed training/resume/population tests.

## Task 3: Review and publication

- [ ] Document exact knobs, data/GT boundaries, bootstrap schedule, known verification limits, and fresh-run requirement.
- [ ] Run combined regression checks, review diff, and request independent code review.
- [ ] Commit locally. Publish a tree matching the tested commit on top of the verified remote branch head using a non-force expected-head update; verify the remote commit/tree. Do not restart or resubmit the existing DLC as part of this code publication.
