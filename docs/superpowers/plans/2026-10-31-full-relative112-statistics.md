# Offline selected-population relative112 statistics

> **For agentic workers:** Use superpowers:subagent-driven-development and review each bounded task.

**Goal:** Generate reusable CPU/offline normalization assets for the user's exact 20261030 exclusion YAML, OpenWAM 112-dimensional/32-step reader, using fixed-anchor relative poses, never consecutive-frame deltas.

**Architecture:** Preserve the native reader's representation, profile grouping, source selection and masks. A vectorized scalar-only encoder is checked against `CuratedStage2Dataset.raw_window`. Enumerate all eligible task/subtask anchors and future offsets 1..32 in every selected episode. Merge deterministic per-repository checkpoints into an immutable existing-format registry. Moments/extrema/counts cover the complete population; approximate quantiles have explicit deterministic histogram bounds. DLC validates and reuses the registry without recomputation. No GPU training or job submission.

**Tech Stack:** Python, NumPy/SciPy, PyArrow, current native reader; bounded CPU processes on CPFS.

## Global constraints

- Source: `/mnt/data/cpfs/b5/yeqingda/pretrain/launch/moravla/train_files/pretrain_v0_3_curated_3000h_posttrain_20261019_exclude_rcv1_rmv1_rmv2_20261030.yaml`.
- Current worktree only; preserve existing unrelated modifications. No source dataset edits.
- 32 future source rows, unified112/raw52 layout, fixed current TCP and fixed current head axes. Joint targets relative to anchor; gripper absolute with existing alignment.
- Native validity/calibration/gripper filtering must be identical to training. No image decode, no clipping while accumulating.
- Invalid/missing dimensions retain zero count and masks; never publish complete if any selected repository/episode failed.
- Registry hash binds selection and representation; resumable work binds source/code/policy identities. Quantile approximation and enumeration weighting must be explicit.

## Task 1: Batched fixed-anchor encoder

- Add `openwam/dataloader/native_relative_population.py` and focused tests.
- API: `RelativeEpisodeEncoder(metadata)` caches scalar-only absolute values, rotations and source masks. `encode(anchors, ends)` returns action `[B,32,52]`, mask, state `[B,52]`, state mask. End is exclusive source boundary; all unavailable tails invalid.
- Compare every output against repeated `dataset.raw_window(metadata, {'last_source':end-1}, anchor)` for synthetic rotating/translating poses, missing head, source flags, masks, shifted gripper commands, moving robot references and native fixtures. Assert second target is relative to row zero, not preceding row.
- Run red/green tests. Do not alter native training representation.

## Task 2: Exhaustive streaming statistics and offline runner

- Add bounded deterministic mergeable histogram/moment accumulator and tests for exact counts/extrema, merge equivalence and quantile error bounds including negative/zero values.
- Add CPU CLI to load full selection/audit, enumerate all selected episodes and reader `_options` for each available instruction level, encode in bounded blocks, accumulate without normalization, and save verified per-repository shards for resume.
- Quantile bins use signed logarithmic edges with explicit maximum error; retain exact moments/min/max. Publish raw52 stats and unified112 dimension mapping, policy/coverage records and native immutable registry only after all work succeeds.
- Keep local/head statistics separate and joints grouped by profile; export a separate informational cross-profile TCP summary, never mix different joint semantics for training.
- Benchmark representative real sources, then run full selected population with measured progress/ETA and bounded worker count.

## Task 3: DLC registry reuse

- Update `openwam/train/stage12.py --prepare` to honor an explicit precomputed `stats_registry`, validate it against the actual dataloader contract and source metadata before any expensive Stage1 preparation, and skip `prepare_native_stats` in this path.
- Keep existing behavior when no registry is supplied. Add tests for successful reuse, stale selection/schema rejection and missing files.
- Provide concrete CPFS registry path and minimal Stage2/Stage12 config overrides plus documentation. Do not submit DLC.

## Task 4: Review and delivery

- Independently review encoder equivalence, all-population scope, restart correctness, quantile bounds and fail-closed publication.
- Verify native dataset construction consumes final registry and representative unnormalized relative targets roundtrip according to the explicitly chosen normalization policy (statistics production does not change clipping).
- Report actual completed repositories/episodes/anchors/valid dimension counts, missing supervision and immutable artifact/config paths. Never label a running or partial scan complete.

## Authorized scope revision: remove unavailable wipe_table_v1_test

The user explicitly said “没有就不要了”. Generate a new YAML excluding exactly
that repository (55 episodes, 118,653 frames), preserving all other repository
records. Keep the original scan and checkpoints intact. Independently audit the
new selection, validate exact retained-repository and contract equivalence, and
reuse verified per-repository histograms with explicit migration provenance.
Publish the new immutable registry only after all 5,702 remaining repositories
finish and the consumer validation succeeds. Test rejection of changed retained
ranges/weights and source changes. Keep DLC documentation bound to the new YAML.
