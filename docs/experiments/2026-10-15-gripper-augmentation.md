# Gripper supervision and observation augmentation

This branch retains 112D state/action slots and 32 future steps. Pose rotations use relative matrix products, and joint targets remain q[t+k+1] - q[t]. The gripper input is measured state; robot gripper targets use aligned controller commands, and audited human data uses the stored openness targets. Gripper values are clipped to [0,1]. Missing targets are masked, and windows containing detected state spikes are excluded consistently from the reader and normalization population. Existing stats with the previous target contract must be regenerated using scripts/data/prepare_openwam_relative_stats.py.

The final resized/cropped L canvas supports clip-consistent brightness/contrast/saturation jitter, one variable-sized patch in one surviving view, and single-view dropout. Whole-state dropout also clears the input mask, while preserving action targets, target masks and the original FK anchor. The default augmentation profile is configs/augmentation/multiview.yaml (probabilities 0.8/0.5/0.1/0.1). Future video target frames share the augmentation. Missing cameras remain missing. Evaluation disables augmentation, and changed RGB bypasses the prepared-input cache.

## Current experiment

- Python: /root/miniconda3/envs/openwam/bin/python.
- Baseline: gripper_human_lr0.0001_b320, completed 1000 steps.
- Augmented run: gripper_human_aug_lr0.0001_b320, newly started from the same OpenWAM Alpha foundation checkpoint_step_154000; not resumed from baseline step 1000.
- Same 46 prepared clips / 6691 accepted windows; batch 320, 1000 optimizer steps, seed 42, bf16, video/action/projection learning rates 1e-5/1e-4/1e-3, clean fixed-noise evaluation every 25 steps.
- Saved configurations differ only in augmentation and output paths. All clean evaluation metrics at step 0 are exactly identical. Peak allocated memory observed at startup: 108.18 GiB.
- TensorBoard comparison: https://897628-proxy-18083.dsw-gateway-cn-shanghai.data.aliyuncs.com/#scalars ; broader historical comparison: port 18082. An authenticated Alibaba DSW browser session is required.
- New run launch/verification: outputs/curated_observation_aug_20261015/{launch,supervise,verify_restart}.py. The actual trainer is outputs/curated_gripper_human_20261014/run.py, accepting --augmentation-config configs/augmentation/multiview.yaml and a fresh --name. The supervisor snapshots source/config and historical event hashes and updates comparison.json with equal-step clean evaluations.

These exact experiment and historical visualization Python drivers are deliberately versioned under outputs/ so their existing paths remain valid. Generated Parquet, RGB caches, normalization files, events, checkpoints, videos and HTML remain local. Reproducing a driver requires preparing its fixture/manifest first; this is not a standalone public dataset bundle. scripts/data/prepare_curated_gripper_commands.py and verify_curated_gripper_commands.py prepare and validate command/human-target fixtures from an existing source clip manifest. The human-only audit also uses the previous robot-command fixture for comparison. Historical renderers retain their documented input/cache assumptions.

## Visualization and validation

The portable skill is skills/openwam-visualization/SKILL.md, with source-side export_current_inputs.py, validate_report.py and a tool catalog. scripts/skills/install_openwam_visualization.py installs it into the configured CODEX_HOME. The catalog's historical Python drivers and their local Python helpers are included; source data, URDF and Mora processor dependencies remain external.

scripts/visualize_observation_augmentation.py generates an offline HTML with actual clean/augmented inputs and verifies unchanged GT/FK. Local delivered example: outputs/observation_augmentation_20261015/review/index.html (18 comparisons across LightWheel, ABC and Interndata A1). The review sampled the first accepted window of all 46 clips, checked filtering-index equality and clean evaluation, and did not execute a model forward.

Before delivery, 726 CPU tests passed across dataloaders, affected FK/action model losses, worker seeding and skill helpers. The actual CLI configuration path was additionally exercised on CPU, catching and fixing Hydra struct-mode insertion. GPU startup verification checked batch 320, finite losses, identical initial evaluation and 11 unchanged historical event files. Compare clean eval_fixed_noise at equal steps; initial training losses do not establish a generalization or convergence benefit.
