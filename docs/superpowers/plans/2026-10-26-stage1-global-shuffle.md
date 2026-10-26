# Stage1 global shuffle repair

> Execute inline using systematic debugging, test-driven development, and verification-before-completion.

**Goal:** Mix the finite task/subtask population globally while preserving exactly-once coverage, distributed tail weighting, and deterministic resume.

**Architecture:** Keep SQLite sample IDs immutable. Allocate one seeded NumPy PCG64 permutation of those IDs per rank; the saved cursor addresses permutation positions. Record algorithm, NumPy version, seed, and population size in the checkpoint identity. Use existing training collectives to report batch composition and active TCP supervision.

**Constraints:** Do not modify the live job's checkout or source data. Store validation artifacts on CPFS. Do not restart the live DLC job as part of code validation. Reuse the prepared index; do not rebuild or silently change population weights. Global shuffle mixes according to actual sample counts, not equal dataset probabilities.

## 1. Sampler and resume contract

- [x] Reproduce the sequential-order bug in the actual tiny training loop, and test deterministic full coverage at world sizes 1, 8, and 32, including partial tails.
- [x] Add `openwam/train/epoch_order.py`: `sampling_contract(config, total)` and `EpochOrder(contract)` with a read-only uint32 permutation (uint64 for larger populations), indexed by global cursor position.
- [x] Include the sampling contract in `epoch_identity`; reject absent or changed contracts before training/resume. Check that all ranks constructed the same permutation before their first update.
- [x] Verify uninterrupted and resumed traversal return the same remaining sample IDs, including different task/subtask and data-source labels.

## 2. Training integration and visibility

- [x] Map `rank_sample` positions through `EpochOrder` in `train_epoch`, keeping per-sample RNG keyed by immutable dataset ID and keeping zero-weight padded tail entries out of coverage counts.
- [x] Extend existing preparation gathering with active sample family/granularity. Extend existing loss reduction with valid TCP-point and TCP-supervised sample counts. Record per-step family and task/subtask counts.
- [x] Run actual CPU tiny-model updates and two-rank Gloo synchronization with a partial tail; assert the trainer used the permutation and logged correct supervision counts.
- [x] Set production epoch presets to global shuffle and update docs explaining memory, source-count proportions, random-I/O costs, and the requirement for a new run when migrating from sequential order.

## 3. Production-population audit and delivery

- [x] Build the real 53,822,139-element permutation with seed 42. Measure memory/time, verify permutation coverage, and inspect the first 1,000 32-rank batches against the existing SQLite index.
- [x] Report actual dataset and task/subtask distribution; compare it with the old contiguous prefix. Store evidence under `outputs/shuffle_fix_20261026` on CPFS.
- [x] Run the affected epoch/pipeline tests, review the diff and create a concrete commit. Preserve the running job and its outputs; report deployment status separately from code verification.
