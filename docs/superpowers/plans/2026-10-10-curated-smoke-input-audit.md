# Curated smoke input and relative TCP audit plan

**Goal:** Run the current OpenWAM reader on all 46 explicitly selected source clips, record actual model-input processing and GT transforms, and deliver local/browser HTML with videos.

**Scope:** YAML `pretrain_v0_3_curated_smoke.yaml` lines 108–298, source root as configured (V0_2), exclusive segment ends. Current checkout 490a229 plus workspace changes; do not change model/reader semantics or unrelated files. 112D, 32 future source states, video stride 4. No adoption of Mora's 50-step / 15Hz settings. No training launch pending user's training-vs-inspection response.

**Architecture:** A report-only driver extracts bounded local fixtures from source Parquet and preserves exact source/video indexing. It configures and invokes the production relative reader and normalization utilities, saves input tensors, and renders directly from the same source poses and production relative outputs. Source TCP contracts must be evidenced; missing TCP or camera calibration remains masked/unavailable rather than fabricated. Source EEF diagnostics are labeled separately from supervised TCP.

## Tasks

- [x] 1. Inventory every YAML segment and source metadata. Save `outputs/curated_smoke_20261010/manifest.json` with original source paths, episode/frame intervals, camera keys, joint maps, TCP evidence and projection availability.
- [x] 2. Create bounded local fixtures/configurations and per-clip smoke statistics using `prepare_relative_stats`; invoke `build_dataset` on interior and tail anchors. Preserve source metadata/video offsets and map local frame 0 to the explicit source start.
- [x] 3. Verify tensors: shape 32×112, next-state t+1, current-anchor relative rotations by matrix multiplication, joint deltas vs current state, masks, padding and clip boundaries. Compare raw-relative decode back to source TCP positions and orientations.
- [x] 4. Capture reader image stages and actual input canvas. Audit all resizes, crop, jitter, missing-view padding, temporal padding, first-frame conditioning, normalization/scatter and the model preprocessing boundary against current code.
- [x] 5. Export each clip as head-camera projection + actual multiview canvas + current TCP-local pose trajectory video; display mask/no-calibration reasons where unavailable. Head-relative deltas must regain anchor camera position before projection; never project a zero-origin delta as an absolute camera point. Include pose axes, fixed/current viewpoint labels and GT identity.
- [x] 6. Build standalone local HTML, publish in the existing report service, validate video decode and links, and provide precise coverage/failures. Keep generated arrays/configs/statistics and evidence reproducible. Do not commit or push without a new request.

## Verification

Source table row IDs must equal requested half-open intervals. Actual reader video frames must match the reconstructed decode/resize/layout pipeline. Anchor and tail windows must not consume frames outside a selected segment. Pose roundtrips are checked in physical units, before clipped normalization. Source masks and unsupported source contracts must be reported accurately. HTML uses relative asset URLs and requires no remote scripts.

## Executed result (2026-10-10)

- Report: `outputs/curated_smoke_20261010/index.html`; served at `/curated_smoke_20261010/index.html` on the existing 18081 report service.
- All 46 source segments / 12 families read, 184 interior/tail windows checked; action32×112 and next-state +1, fixed-anchor joints, masks, quantile/scatter verified. Source FPS remains 30.
- TCP-local supervision: 11 clips (A1×4, ABC×4, DOS-W1×3). Head-relative supervision: 7 (A1×4, DOS-W1×3). Arm-joint supervision: 28. Missing physical TCP masks both pose branches; ABC strict camera eligibility fails and masks head.
- 46 videos: 75 display frames / 5 seconds each, head projection plus actual model canvas and TCP-local pose axes. Nine actual reader images per clip are pixel-identical to reconstructed decode/resize/layout. Raw trajectory rendering is explicitly before quantile clipping; actual normalized targets and clipping statistics are included separately.
- Production compatibility fix: task prompt loading now supports real explicit task columns plus legacy string indices, with validation. Current focused verification: 46 tests passed (prompt, relative reader, relative/head representations, canonical reader). Independent code/data review completed.
- Report-only fixture video-offset adapter restores original file-local ordinals from the full source episode manifest and cross-checks timestamps. `data.yaml` requires `build_clip_reader`, and is not a standalone training config.
- Source differences handled: ABC clip11 metadata dimensions differ from its 848×480 video; use decoded video dimensions for display. Mecka non-pinhole intrinsic vectors are not incorrectly interpreted as 3×3 K; no physical TCP means projection stays masked.
- Existing source validity/confidence fields are not newly wired into production masks; the limitation and sampled values are documented. Pose roundtrip validates coordinate arithmetic, not calibration accuracy.
- No training/model/VAE forward, no new checkpoint, no commit or push. Original pyproject and unrelated local changes preserved.
- Delivery verified: local HTML and MP4 HTTP 200, 1,641 HTML links resolve, no missing relative assets. Public gateway returns 302 to Alibaba DSW login; authenticated browser access required. Offline ZIP (109 MB) includes HTML, videos, input PNG/NPZ, configs and evidence; CRC passes.
