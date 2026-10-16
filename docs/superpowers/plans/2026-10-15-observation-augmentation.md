# Observation augmentation implementation plan

> Execute inline with TDD. The user explicitly requested color jitter, one variable-sized patch within one view after resize/crop, probabilistic single-view dropout, and probabilistic state zeroing. This is implementation authorization. Existing branch and uncommitted data adaptation work are retained; no training restart or commit is requested.

**Goal:** Add configurable, temporally consistent observation augmentation to the current Mora/curated training path without changing action GT, target masks or FK anchor data.

**Architecture:** A shared sample transform runs on the final resized/cropped multiview canvas and measured proprio. The base LeRobot reader invokes it after deterministic assembly; cached fixtures supply explicit missing-view masks. The curated trainer bypasses prepared-image caches for changed RGB, restores clean state only for cache construction, and applies state dropout to a copied prepared payload. Evaluation temporarily disables augmentation.

**Tech stack:** Python, PIL, NumPy/PyTorch, OmegaConf, pytest. No new dependencies.

## Semantics and configurable defaults

- color_jitter: p=0.8, brightness/contrast/saturation=0.2; reuse existing clip-consistent implementation, exclude structural missing slots.
- patch_mask: p=0.5; choose exactly one surviving available camera and one rectangle, area 5–20% of that slot, aspect 0.5–2; apply the same rectangle to every frame. Mask after final resize/crop. Fill black.
- view_dropout: p=0.1; drop exactly one available camera across the clip. keep_one_view=true; skip if only one source view exists. Select patch from surviving views.
- state_dropout: p=0.1; zero the complete input proprio and its input mask. Preserve original values privately for constructing a clean cache; never feed the private values into an augmented forward.
- Independently configurable probabilities, disabled by absent configuration or eval split/context. No source GT/statistics/filter contract changes; q[t] in FK payload stays intact.
- All video frames, including future training video targets, use the same photometric/occlusion transform. Action targets remain unchanged. No geometric crop/flip augmentation is added.

## Task 1: Shared transform and cache contract

Files: openwam/dataloader/transforms/observation.py, openwam/train/utils/observation_cache.py, tests/dataloader/test_observation_augmentation.py.

- [x] Write failing tests for per-view bounds, exactly one variable patch, time consistency, missing slots, keep-one-view, train/eval/p0 behavior, state zeros/input masks and immutable GT/FK.
- [x] Implement validated configuration, per-view boxes matching current L layout, source-missing masks, augmentation trace and clean-proprio preservation.
- [x] Write failing cache tests for changed RGB bypass and clean-state cache hit/miss behavior; implement cache helper without mutating stored tensors.
- [x] Verify seeding reproducibility and repeated draws vary; reject malformed probabilities/ranges rather than silently ignoring config.

## Task 2: Reader and curated training integration

Files: openwam/dataloader/bases/lerobot_v3_reader.py, openwam/dataloader/curated_fixture.py, configs/augmentation/multiview.yaml, tests/dataloader/test_mora_relative_reader.py, outputs/curated_gripper_human_20261014/{run,data}.py.

- [x] Add reader config key observation_augmentation and run after final canvas composition; avoid applying legacy and new color jitter twice.
- [x] Cached decoder returns structural view masks instead of rejecting jitter; CuratedFixtureDataset accepts an override config and supplies source camera availability.
- [x] Add an evaluation context that restores prior augmentation state even on error; disable augmentation for fixed-noise evaluation.
- [x] Add --augmentation-config to the curated training entrypoint and use augmentation-aware cache preparation. Default existing launch remains unaugmented unless explicitly configured; active Python process is unchanged.
- [x] Add regression tests proving source reader/curated paths use the transform and preserve GT, filtering, stats and FK payloads.

## Task 3: Verification and review artifacts

- [x] Run targeted augmentation/cache and reader/representation/FK regressions.
- [x] On real ABC, A1 and human fixtures, compare clean/augmented samples and save before/after images with actual augmentation trace. Verify input state zeros, source pose/joint/gripper GT and FK anchors unchanged.
- [x] Verify actual training entrypoint accepts configuration and clean evaluation/prepared caches cannot leak stochastic augmentation.
- [x] Save a reviewable HTML and implementation/usage results here. Do not restart the current experiment.

## Verification results (2026-10-15)

- 121 CPU regression tests passed across observation augmentation/cache, canonical/Mora reader, gripper targets, relative representations, FK payload/loss, action loss and worker seeding. CUDA availability probe and existing NumPy deprecation warnings were emitted; no GPU training was run.
- Real fixture audit: 46 clips, 6691 accepted windows. Compared the first accepted window in each clip; filtering indices match throughout each reader. The audit is not an exhaustive comparison of every window's values. Action GT/masks and FK payloads remain identical; dropped state is zero with a false input mask. Fixed evaluation samples match clean RGB/state and consume no augmentation RNG.
- 18 visual examples: LightWheel, abc_130k and Interndata A1, each with default probabilities, four forced individual operations, and forced combined operations. Offline report: outputs/observation_augmentation_20261015/review/index.html (images embedded; browser-openable as one file). Evidence: validation.json and delivery_check.json alongside it.
- Independent review found and fixed a legacy color-jitter cache edge case. Both legacy-only and legacy-plus-state dropout are regression-tested so augmented RGB cannot freeze in cache or leak into clean evaluation. Review follow-up found no further important issue.
- Actual Trainer.prepared was exercised on CPU with a fake encoder: state-drop first miss, clean cache reuse, changed-RGB bypass and mixed-batch state masks/GT/FK checks passed (review/trainer_cache_check.json). Actual curated trainer --help exposes --augmentation-config. Existing active experiment was not restarted. A new run name is required for the next experiment. Image changes bypass prepared-input caching and require fresh image encoding; the previous batch 320 throughput/memory has not been benchmarked with these augmentations.

## Usage

From the repository root, add this option to the existing curated run command:

```bash
/root/miniconda3/envs/openwam/bin/python outputs/curated_gripper_human_20261014/run.py \
  --name gripper_human_aug_lr0.0001_b320 \
  --batch 320 --steps 1000 --lr 0.0001 --eval-every 25 \
  --augmentation-config configs/augmentation/multiview.yaml
```

This is a launch example, not a validated batch-size recommendation. Without --augmentation-config the old curated launch remains unchanged. The individual data.yaml files are not rewritten. A production canonical/Mora reader can instead receive the same mapping under observation_augmentation in its dataset configuration. Image operations require the existing multiview L canvas. State-only dropout also works without multiview. Set individual p to 0, omit its block, or set enabled: false to disable operations. Evaluation split or CuratedFixtureDataset.unaugmented() disables augmentation.

Chain: source decode -> existing resize/crop -> L canvas -> shared clip color jitter -> exactly one patch in a surviving camera + optionally one camera zeroed -> state/input-mask dropout -> model prepare_inputs. View dropout selection happens before patch selection, so the patch is never wasted inside a dropped view. No additional crop/flip is introduced. All future video target frames share the image augmentation; action GT, gripper targets, loss masks and FK q_anchor remain unchanged. No new loss term is introduced.

To reproduce the CPU review with a fresh output directory:

```bash
/root/miniconda3/envs/openwam/bin/python -m scripts.visualize_observation_augmentation \
  --manifest outputs/curated_gripper_human_20261014/data_manifest.json \
  --output-dir outputs/observation_augmentation_new_review
```
