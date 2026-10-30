# Optimized Stage1 and frozen Stage2

The unified Stage1 entry accepts `model.goal_representation: visual_history_masked_sa`. It pools ten historical head frames into 32 tokens, conditions current multiview Wan tokens through masked self attention, and jointly denoises a target head image and two 2D TCP points. Stage2 loads the decoder-pruned, frozen encoder and uses its 120 current/history-conditioned tokens as visual memory. Stage2 does not consume the generated goal image or 2D TCP as an action target.

A missing head camera leaves available wrist inputs intact. The corresponding head image target, 2D TCP labels and head history are invalid. Masked NaNs cannot enter the image/TCP losses; independent native Stage2 local TCP, joints and valid gripper targets retain their own masks.

## Unified entries

- `configs/goal/stage1_visual_history_overfit.yaml`: bounded repeated-population training, 1,000 updates, one source frame/second; small-input caches are bounded.
- `configs/goal/stage2_visual_history_joint.yaml`: frozen encoder plus joint future-video/action denoising.
- `configs/goal/stage2_visual_history_none.yaml`: the same frozen encoder, with no future-video denoising target.
- `configs/goal/stage12_visual_history_overfit.yaml`: the native sequential Stage1→Stage2 launcher. Set `STAGE2_SOURCE_YAML`, `STAGE12_OUTPUT_DIR`, `OPENWAM_BASELINE_CHECKPOINT`, `STAGE2_BASELINE_CHECKPOINT`, and optionally `STAGE2_STATS_REGISTRY` to real assets before use.

```bash
python -m openwam.train.stage12 --config configs/goal/stage12_visual_history_overfit.yaml --prepare
python -m openwam.train.stage12 --config configs/goal/stage12_visual_history_overfit.yaml --train
```

Direct Stage1 entry, with an explicit resolved config and source selection:

```bash
python scripts/train_goal_epoch.py --prepare --config "$STAGE1_CONFIG" --index "$RUN/population.sqlite" --output-dir "$RUN/stage1"
python scripts/train_goal_epoch.py --train --config "$STAGE1_CONFIG" --index "$RUN/population.sqlite" --output-dir "$RUN/stage1"
```

Use `--resume` with the same config/index after interruption. `max_updates` and `repeat_population` belong to the hashed training identity. A bounded run publishes `training_complete.json`; it does not claim a complete finite-population epoch. `visual_history.safetensors`, `resolved_config.yaml`, `dependencies.json`, `text_contexts.pt`, and their artifact hashes support native reload and visualization. The completion proof names the durable checkpoint generation and its SHA256.

## Matched three-episode experiment

The source selection is `configs/experiments/three_episode_overfit_20261031.yaml`: ABC wood-block tower episode 0, Interndata toothbrush/toothpaste episode 472, and RoboChallenge v2 tray episode 0. Only these episodes enter the experiment. Interndata microwave episode 190 was rejected before training because the reader detected a 32-frame video-offset inconsistency.

Create a run directory containing `source.yaml`, a resolved `stage1_config.yaml`, and a resolved `comparison_config.yaml` with the same source YAML, actual initialization paths and native normalization registry. Statistics must be prepared from that source selection with `prepare_native_stats`; reusing a registry from another YAML is rejected. The supplied preparation entry materializes twelve evenly spaced valid windows per available task/subtask level (60 total in the recorded run):

```bash
python scripts/experiments/prepare_three_episode_overfit.py --config "$RUN/comparison_config.yaml" --output "$RUN/data"
python scripts/experiments/run_three_episode_stage12.py --run "$RUN" --stage1-updates 1000 --stage2-updates 500
```

The supervisor invokes unified Stage1 and the native Stage2 model/loss/optimizer groups. Both arms receive the identical frozen teacher and Stage2 initialization, fresh FP32-master AdamW, seed 42, 25-update LR warmup and exactly 500 updates. A wall-clock timeout is a failure, not successful completion. It checks teacher SHA256 around both arms; the native worker also checks teacher parameter versions after every update. `--wait-stage1-pid` attaches to an already-running unified Stage1 without starting another copy.

Stage1 trains the 1 FPS task/subtask population; Stage2 trains the explicit bounded windows. Intermediate Stage2 generation evaluates three recorded training anchors at updates 0, 250 and 500. This is overfit diagnostics, not held-out generalization. Both arms use the same per-update seed, but joint video sampling consumes RNG before action sampling, so their action-noise draws need not match.

## Visualization and limits

Use the `openwam-visualization` skill's `fullspan-samples`, `fullspan-inference`, `fullspan-video`, and `fullspan-validate` entries. A complete subtask is chosen for ABC and RoboChallenge v2; Interndata has a complete task. Every ten source frames plus the final frame gets fresh inference. The two-row videos show GT/predicted future 32-step TCP trajectories on the same current three camera views, and Stage1 goal RGB/TCP in the fourth column.

Generate the overview with `python scripts/experiments/report_three_episode_stage12.py --run "$RUN"` (requires Matplotlib). It reports actual optimizer counts and fixed-anchor generation metrics, flags regression against each arm's initialization, and requires all six videos to pass validation before displaying overall completion. Optional `code_publication.json` records the remote branch/commit and matching source tree.

RoboChallenge v2 wrists are retained as images with projection unavailable: their optical extrinsics remain manually selected candidates. Do not label them verified. These three native profiles have no matched FK supervision profile; direct TCP projection is not an URDF/FK verification. Run `projection_coverage.py` to report actual visible GT per camera and `validate_fullspan_video.py` for the video/overlay contract. The report server supports HTTP Range for seeking.
