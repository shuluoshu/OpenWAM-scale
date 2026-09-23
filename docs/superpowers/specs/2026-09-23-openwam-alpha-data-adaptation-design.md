# OpenWAM-α Canonical EEF Data Adaptation Design

## Objective

Build a minimal, reproducible end-to-end path that:

1. fine-tunes the downloaded OpenWAM-α foundation checkpoint for one or two real optimizer steps on a deterministic subset of `pretrain_datasetsV0_1`;
2. performs offline open-loop inference on one RoboDojo and one RobotWin trajectory;
3. consumes the EEF fields produced by the MoraVLA preprocessing pipeline without importing MoraVLA's model-specific 96D action layout; and
4. produces an offline report showing every important sample transformation immediately before model forward.

The implementation must not enumerate or train on the complete supplied dataset. It must not modify source data under `/mnt/data/cpfs`.

## Confirmed Inputs

| Purpose | Path |
| --- | --- |
| Training root | `/mnt/data/cpfs/b5big/pretrain_data_release/pretrain_datasetsV0_1` |
| RoboDojo trajectories | `/mnt/data/cpfs/b5/post_train_data/robodojo_sim` |
| RobotWin video/joint trajectories | `/mnt/data/cpfs/b5/post_train_data/robotwin_traj` |
| RobotWin EEF companion | `/mnt/data/cpfs/b5/post_train_data/robotwin_traj_fullfield` |
| MoraVLA reference | `/mnt/data/cpfs/b5/yeqingda/pretrain` |
| Original comparison page | `/mnt/data/cpfs/b5/yeqingda/pretrain_vs_openwam_comparison.html` |
| Foundation checkpoint | `assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model` |

RobotWin `robotwin_traj_fullfield` is an augmented view of `robotwin_traj`: its metadata/parquet files add EEF and camera calibration, while its `videos` entry points back to the original trajectory videos. The user approved using this companion.

## Selected Approach

Add a generic canonical-EEF LeRobot-v3 reader and keep the source-specific choices in YAML profiles. The reader consumes already materialized `observation.states.eef` and `actions.eef`, then applies OpenWAM's own normalization, unified-action mapping, temporal windowing, masks, and multiview rendering.

This is preferred over two alternatives:

- Adding independent LeRobot branches to the legacy HDF5 RoboDojo and RobotWin readers would duplicate window, video, prompt, normalization, and mask behavior.
- Porting MoraVLA's complete `VlaPipeline` would bring in a second dataset framework and its `dual_eef_local_96d` model contract. OpenWAM-α uses a different 80D action head, so that port would be larger and less reliable.

The design reuses MoraVLA's explicit field/frame semantics and processed EEF artifacts, not its Qwen tokenizer or model-specific action-space composer.

## Verified MoraVLA Launch Contract

The authoritative MoraVLA reference for this task is:

```text
/mnt/data/cpfs/b5/yeqingda/pretrain/launch/qwena1_nogen/pretrain.sh
  -> launch/qwena1_nogen/train_files/pretrain.yaml
```

The launch script selects the `qwena1_nogen` framework and overrides the YAML's nominal 30 FPS with `FRAME_SAMPLE_FPS=5` unless the caller provides another value. It enables external LeRobot stats from `${DATA_ROOT_DIR}/stats`. The loaded YAML uses a 36D state/action head, `action_mode: delta`, `eef_frame: head_camera`, a 0.5 probability of choosing a wrist EEF frame during training, and image offsets `[-15, 0, 15]`.

These are facts about the actual MoraVLA training recipe, superseding the earlier assumption that its current default recipe directly emitted 96D actions. MoraVLA still contains 96D layout implementations elsewhere, but they are not the launch contract used for this comparison. OpenWAM cannot directly consume either the verified 36D head-camera delta contract or the separate 96D layouts because its foundation checkpoint expects normalized 80D unified actions. The adaptation therefore borrows field selection, coordinate-frame handling, temporal sampling, delta construction order, and external-stats validation as reference behavior while retaining the approved OpenWAM absolute EEF20-to-80D model contract.

## Representation Contract

### Raw EEF20

Every selected sample presents:

```text
[left xyz(3), left rot6d(6), left gripper(1),
 right xyz(3), right rot6d(6), right gripper(1)]
```

Both state and action are absolute EEF poses already materialized in parquet. Rotation uses the first two rotation-matrix columns. The reader validates width, finite values, non-degenerate rot6d blocks, and gripper width.

### OpenWAM unified 80D

Normalization occurs in raw 20D space. The normalized result is scattered into:

```text
raw left  0:10 -> unified  0:10
raw right 10:20 -> unified 34:44
```

All other unified dimensions are zero with `False` action/proprio masks. The model therefore receives `action.shape == [32, 80]` and `proprio.shape == [1, 80]`.

### Why MoraVLA deltas are not copied

MoraVLA's verified `qwena1_nogen` recipe converts EEF poses into head-camera deltas and feeds a 36D action/state head; training may randomly use wrist-frame EEF with probability 0.5. The downloaded OpenWAM-α foundation checkpoint instead has an 80D action/state head and the existing OpenWAM data path normalizes absolute EEF before scattering it into that head. Feeding the 36D delta values directly would load valid numbers into the wrong shape and semantics.

The comparison documentation will show this boundary explicitly, including the 36D launch recipe and the distinct 96D layouts found elsewhere in MoraVLA. The new reader can expose raw calibration fields for inspection, but it will not introduce a second MoraVLA model path.

## Data Flow

```text
LeRobot bucket
  -> validate info.json + parquet schema
  -> choose deterministic episodes/windows
  -> read 33 row-aligned state/action steps
  -> read task text from tasks.parquet
  -> decode head/left-wrist/right-wrist frames at indices 0,4,...,32
  -> assemble 9 L-shaped 384x320 RGB frames
  -> take state at t0 and action targets t0..t31
  -> normalize source EEF20 with source-specific smoke stats
  -> scatter EEF20 into OpenWAM unified 80D
  -> construct temporal/dimension masks
  -> return the existing BaseDataset sample dictionary
  -> OpenWAM collate / architecture forward or generate
```

The emitted dictionary keeps the established keys:

```text
video, vace_video, first_frame_image,
action, action_mask, video_mask,
proprio, proprio_mask, prompt
```

It also carries an optional `trace` metadata dictionary only when `return_trace=true`. Training leaves tracing off. Visualization uses it to show source paths, episode/frame indices, raw EEF20, normalized EEF20, unified 80D, and each transformation label.

## Components

### Canonical EEF reader

`openwam/dataloader/canonical_eef_lerobot.py` will subclass `LeRobotV3Reader` and provide:

- configurable state/action/camera keys with strict defaults matching the supplied data;
- `meta/tasks.parquet` prompt resolution;
- source-specific `normalization_stats_path`;
- deterministic `episode_indices`, `max_episodes`, and `max_windows` limits;
- raw-to-unified normalization helpers that can also invert model outputs for reports;
- optional trace metadata; and
- root discovery through an explicit list of bucket directories, avoiding a full recursive scan during smoke runs.

The dataset registry name will be `canonical_eef_lerobot`. Legacy `robotwin` and `robodojo` types retain their existing HDF5 behavior.

### Stats preparation

`scripts/data/prepare_openwam_eef_smoke.py` will read only the configured episode subset and write one small `.npy` artifact per source. Each artifact contains separate `eef` action statistics and `eef_state` statistics with count, mean, standard deviation, min, max, q01, and q99 plus provenance:

- bucket path;
- episode indices;
- field names;
- frame count;
- generation seed; and
- source metadata fingerprint.

The script writes only below the repository output/example directory. It never writes into a mounted dataset.

### Smoke training configuration

`configs/openwam_alpha_v0_1_smoke.yaml` will be a complete Hydra config with two enabled mixture entries:

- one InternData-A1 Genie-1 bucket;
- one AgiBotWorld A2D gripper bucket.

Each source is limited to two episodes and at most eight windows. The training defaults are:

```text
foundation warm start: OpenWAM-Alpha-Pretrain-Foundation-Model
world size:             1
per-device batch:       1
optimizer steps:        2
gradient accumulation:  1
precision:              bf16
gradient checkpointing: on
checkpoint saving:      off
video/action loss:      both enabled
data workers:           0
seed:                   42
```

The command will use Hydra's `--config-name openwam_alpha_v0_1_smoke` and `NPROC_PER_NODE=1`. The YAML contains the actual absolute paths confirmed for this environment and documents CLI overrides for relocation.

### Offline open-loop inference

`scripts/open_loop_inference.py` will:

1. build one canonical reader from the RoboDojo profile or RobotWin fullfield profile;
2. load the self-contained foundation checkpoint;
3. pass the normalized unified proprio, first observation frame, prompt, and active 80D mask to `architecture.generate`;
4. use two denoising steps by default for smoke and allow ten for the normal setting;
5. disable video decode by default while retaining the action/world-model denoising path;
6. save predicted normalized 80D, gathered raw normalized EEF20, inverse-normalized raw EEF20, GT, masks, prompt, timing, seed, paths, and checkpoint identity; and
7. exit nonzero on a missing field, non-finite tensor, wrong shape, or failed model load.

The smoke acceptance shape is `[32, 80]` for model-space predictions and `[32, 20]` for gathered EEF predictions. This is an execution/contract test, not a success-rate claim.

### Visualization

`scripts/visualize_model_inputs.py` will support all four profiles:

- InternData-A1 smoke training sample;
- AgiBotWorld smoke training sample;
- RoboDojo open-loop sample;
- RobotWin fullfield open-loop sample.

For each profile it produces:

- a PNG contact sheet of the nine processed multiview frames;
- a JSON manifest with source, indices, shapes, prompt, stats provenance, masks, and scalar summaries;
- an NPZ containing raw EEF20, normalized EEF20, unified 80D, GT, and optional prediction; and
- one offline HTML report linking the images and plotting EEF/action traces with inline SVG/Canvas JavaScript, requiring no new Python dependency.

Small representative PNG/JSON examples will be kept under `assets/openwam_alpha_data_examples/`. Large decoded clips and generated model weights will remain ignored runtime outputs.

### Documentation

`assets/openwam_usage_docs/openwam-alpha-data-adaptation.md` will explain:

- every YAML field;
- exact stats, training, open-loop, and visualization commands;
- absolute EEF20 versus the verified MoraVLA 36D head/wrist-frame delta semantics;
- normalized versus physical-unit predictions; and
- how to add another compatible dataset profile.

`docs/pretrain_vs_openwam_data_pipeline.html` will be derived from the supplied comparison page and add a verified, task-specific section covering the implemented preprocessing, reader boundaries, 96D-versus-80D mapping, sample screenshots, commands, and evidence. The original HTML outside the repository will not be overwritten.

## Failure Handling

- A raw RobotWin directory without EEF fields fails with a message naming the approved `robotwin_traj_fullfield` companion; there is no silent joint fallback.
- Missing `info.json`, episodes parquet, tasks parquet, configured data fields, video files, or stats fail with the bucket and exact missing path/key.
- A stats artifact whose field names, population fingerprint, or width does not match the current bucket is rejected.
- Non-finite EEF values, invalid rot6d blocks, invalid masks, or wrong model dimensions fail before model forward.
- A corrupt auxiliary wrist video may produce a black wrist tile only when the base reader's existing tolerated-video policy applies; head video failure remains fatal.
- Visualization of one profile may report that profile as failed while continuing other profiles, but its process exits nonzero when any requested profile failed.
- Training and inference commands use explicit timeouts in verification so a stalled distributed/CUDA startup is visible rather than silently hanging.

## Testing and Acceptance

### CPU unit tests

Tests will cover:

- strict schema validation and helpful missing-EEF errors;
- episode/window limiting without enumerating the whole root;
- prompt lookup;
- 33-step window to 32-step action and 9-frame video alignment;
- time padding and 2D action/proprio masks;
- raw EEF20 normalization and inverse normalization;
- exact `0:10`/`34:44` scatter/gather behavior;
- stats provenance rejection;
- RobotWin fullfield companion resolution; and
- trace metadata being opt-in and absent from ordinary training samples.

### Real-data CPU checks

One sample from each of the four profiles must load and satisfy:

```text
len(video) == 9
action.shape == (32, 80)
action_mask.shape == (32, 80)
proprio.shape == (1, 80)
proprio_mask.shape == (1, 80)
prompt is non-empty
all active numeric values are finite
```

### GPU checks

On the authorized external GPU environment:

- the OpenWAM Python environment must import Torch and initialize CUDA;
- the two-step training command must complete both optimizer steps with finite video, action, and total losses;
- RoboDojo open-loop generation must complete and write both `[32,80]` and `[32,20]` outputs; and
- RobotWin open-loop generation must do the same.

The current environment probe found an H20-3e with approximately 144 GB memory, but the OpenWAM Python `import torch` probe timed out. This is an explicit prerequisite to diagnose during implementation, not evidence that GPU verification already passes.

### Documentation checks

- Both HTML files open without a web server and contain no broken local asset links.
- Every documented command matches a checked-in parser/config option.
- The session worklog contains the commands, outcomes, modified files, and remaining limitations.

## Non-goals

- Full-corpus training or statistics over the complete V0_1 root.
- Closed-loop RoboTwin/RoboDojo simulator evaluation.
- Success-rate or model-quality claims from the smoke run.
- Replacing existing HDF5 readers.
- Adding the MoraVLA model, Qwen processor, 36D/96D action heads, or VLA losses to OpenWAM.
- Modifying any supplied dataset in place.

## Deliverables

The completed change will include the canonical reader and registry entry, profile configs for the two training sources and two inference sources, one full smoke training YAML, stats preparation, open-loop inference, visualization, focused tests, representative sample artifacts, usage documentation, an updated comparison HTML, and the continuously maintained session worklog.
