# Stage1 exhaustive epoch → frozen Stage2 implementation plan

> Use superpowers:subagent-driven-development for isolated implementation and independent review.

**Goal:** One launch command trains Stage1 exactly once over all eligible 15 FPS current-frame/task-subtask pairs, atomically publishes the final bundle, and then starts Stage2 with that encoder frozen.

**Architecture:** A compact persistent interval index defines the finite Stage1 population. Global sample cursor partitions into rank-local batch-one examples without replacement. DDP uses zero-loss padding only for the final incomplete global batch. Full optimizer/master-weight state and the committed cursor are saved together. A CPU orchestrator validates all inputs before model launch and accepts only a verified completed epoch as the handoff.

**Tech stack:** Existing PyTorch/Wan query-history code, torchrun DDP, SQLite interval index, YAML configuration, existing Stage2 trainer and frozen goal loading.

## Global constraints

- Work in the existing isolated checkout `outputs/query_latent_20261020/code`; preserve all pre-existing edits and running experiment snapshots.
- Source YAML: `/mnt/data/cpfs/b5/yeqingda/pretrain/launch/moravla/train_files/pretrain_v0_3_curated_3000h_posttrain_20261019.yaml`.
- Selected option 2: every eligible current frame at 15 FPS, task plus subtask where available. No weighted replacement sampling for Stage1. Goal is original language-interval last frame, current frame strictly earlier. Global native invalid flags exclude anchors with explicit counts; no unreadable-source fallback or silent repository drop.
- Query history remains enabled: 10 past head frames, 16 learned tokens/frame, actual ages, jitter, fixed 64+2 output tokens. Stage2 frozen encoder only; Stage1 auxiliary decoder discarded.
- Support 8 local GPUs with per-rank batch1; exact tail accounting and deterministic per-global-index RNG. Persist FP32 optimizer master parameters as well as Adam states.
- No full 3000h training launch in this task. CPU fixtures, distributed CPU smoke, and actual source preflight are authorized. Never modify source datasets or guess unresolved physical mappings.
- New posttrain repositories have native 29/34/64 layouts; reuse audited physical conventions, preserve all unsupported-geometry masks, and fail explicitly when a required mapping cannot be established. The missing wipe_table_v1_test path has been queried from user.

## Task 1 — Dataset adapters and finite population

Files owned by data implementer: new `openwam/dataloader/goal_epoch.py`, native posttrain adapter modules; necessary edits to `curated_goal.py`, `native_cameras.py`, `native_stage2_io.py`, `native_stage2_profiles.py`; corresponding dataloader/goal tests. Do not edit training/orchestrator scripts.

Interface: `build_epoch_index(dataset, index_path, *, sample_fps=15.0)` returns JSON-serializable summary with `total_samples`, `index_sha256`, `source_yaml_sha256`, counts and completion status. Index compact run records, not one record per frame. `FiniteGoalDataset(dataset, index_path)` exposes `__len__()`, `sample(global_index) -> (record,batch)`, `evaluation_samples()`, `scope_summary()`. It must validate source identity and index completeness. Prefer an explicit `validate_index` API that does not decode images.

Steps:
- [ ] Add failing fixture tests: overlapping task/subtask yields both pairs; finite global IDs cover exactly once; source flags exclude with counts; strict goal future; 30→15 grid; nondivisible source rate either handled explicitly or rejected; source identity mismatch rejected; memory bounded and index can resume interrupted building.
- [ ] Implement index and wrapper over existing materializer, deterministic history draws, explicit source references and narrow native hooks.
- [ ] Adapt absolute selected source keys and external subtask intervals from YAML metadata, including FPS/camera/native profile. Reuse verified source fields/FK from posttraining audit code; report missing geometry instead of fabricating GT.
- [ ] Verify actual metadata for representatives of each native layout and existing-family regressions. Audit 5905 selected repository headers and surface missing source files.

## Task 2 — Distributed finite Stage1 and atomic resume

Files: new `openwam/train/goal_epoch.py`, `scripts/train_goal_epoch.py`, tests `tests/goal/test_goal_epoch_training.py`.

Interface: Stage1 CLI takes resolved config, index path, output directory, optional `--resume`. It writes `epoch_complete.json` only after complete population coverage, final checkpoint plus dependencies and optimizer checkpoint are durable. Completion includes total/seen samples, index/source SHA256, checkpoint SHA256, global update count.

Steps:
- [ ] Test global rank partitioning with N=11/world=3 including final padded ranks, resume cursor equivalence, FP32 master and Adam restoration, and rejection of changed index/world contract.
- [ ] Build real components from existing config/bootstrap, retain learned compressor FP32, wrap complete expert in DDP, synchronize weighted loss and aggregate source counters. Keep VAE/text frozen and bounded caches.
- [ ] Save versioned checkpoint directories via temporary names then atomic publish; store model bundle, master weights, Adam states, cursor and config fingerprints. Failures never create completion.
- [ ] Evaluate fixed bounded examples every 500 updates on rank0, append timeline; all ranks synchronize and report failures. Final bundle survives optimizer cleanup for Stage2.

## Task 3 — Two-stage launcher, configuration, documentation and review

Files: new `scripts/train_stage12.py`, `configs/goal/stage12_query_history_posttrain.yaml`, associated launcher tests, `docs/stage12_one_pass.md`.

Steps:
- [ ] Test orchestrator with fake child executables: failure/partial epoch/bad hash never launches Stage2; complete epoch passes exact final checkpoint/source; resume never reruns completed Stage1; Stage2 failure preserves resumability.
- [ ] Support prepare/preflight/train modes. Preflight source and Stage2 normalization/profile prerequisites before expensive Stage1. Child commands use argument lists, same declared selection, and existing Stage2 launcher with query-history config/frozen encoder.
- [ ] Require immutable statistics registry; preserve action112, grip clip/augmentation/FK configuration and original Stage2 limits. Record phase state atomically; avoid arbitrary shell interpolation and duplicated launches.
- [ ] Document 8-GPU command, one-pass semantics, initialization vs resume, path remapping, masks, exact handoff, and actual outstanding data blockers. Include local code links and no claim of full training readiness if data prerequisites fail.
- [ ] Run focused tests plus independent final review; fix correctness findings and report exact checks. Do not launch production training.
