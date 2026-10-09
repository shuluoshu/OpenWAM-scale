# AgiWorld Additional Video Samples Implementation Plan

**Goal:** Add three useful AgiWorld task clips to the existing portable video report, preserving endpoint uncertainty and source identity.

**Architecture:** Reuse the current robot extractor, renderer and AgiBot FK audit with explicit dataset/episode inputs. Store each new sample in an isolated report subfolder, then include its metrics and embedded videos in the main HTML. No source dataset or training configuration changes.

**Tech Stack:** Python, NumPy, PyArrow, PyAV, OpenCV, existing URDF FK utilities.

## Constraints and selection

- Existing user-approved video modes and playback controls remain the design.
- Survey six tasks at multiple timestamps and select three distinct scenes with visible grippers and manipulation; retain selection and exact source frame evidence.
- All selected sources use the existing A2D gripper embodiment and canonical wrist-axis convention.
- Read camera audit status from each selected episode; never copy episode0/task512's borderline status to other tasks.
- TCP remains unresolved. The 100/128 mm points are diagnostic origins, not verified contact TCP.
- Compute FK independently for each selected dataset and episode. Missing torso still masks world FK; display inter-arm comparison.

## Tasks

- [x] Survey and visually inspect contact sheets; record three chosen task/episode/frame windows.
- [x] Parameterize `video_projection_samples.extract_robot` and `fk_audit_agibot.run` with optional profile/dataset/episode; preserve defaults and enforce provenance.
- [x] Carry episode camera audit into endpoint policy and video labels; verify non-borderline and missing metadata behavior.
- [x] Add a reproducible additional-sample driver with a manifest, source decoding and per-sample FK/rendering.
- [x] Include manifest samples and task/episode labels in `build_report`, with an `agibot-examples` anchor.
- [x] Decode all new MP4s, inspect representative frames, verify exact video/pose/frame joins, FK metrics, HTML media hashes, offline ZIP and served latest content.

Execution is inline in the existing workspace. User has requested these additional examples within the existing visualization design; no additional design approval is needed.
