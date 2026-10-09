# Continuous trajectory video report

**Goal:** Replace snapshot-only delivery with actual continuous videos and clearly
separate coordinate representation, physical endpoint, camera projection and FK.

**Architecture:** Reuse seven verified source-calibration adapters. Extract up to
180 consecutive rows/images per episode beginning around row30. Render H.264
videos at source FPS with per-frame head-camera projection of a rolling 32-step
GT future window, and a fixed-initial-camera comparison. Add a synchronized 3D
FK/label trajectory panel for robot sources; never manufacture missing torso or
tool calibration. Embed videos into the existing offline coverage HTML.

**Constraints:** Source data/training code untouched; source GT only, not policy
inference. Current OpenWAM112 adaptation is relative; distinguish it from original
absolute representation. Mora source EEF and explicitly declared TCP are different
endpoint contracts. Relative targets must be reconstructed to world coordinates
before camera projection. FK comparisons preserve previous independent versus
label-derived diagnostic limitations. No extra user approval or commits required.

- [x] Audit original/current OpenWAM and MoraVLA representation/endpoint code.
- [x] Extract continuous seven-source pose/image windows, per-frame calibration,
      masks, ordinal/PTS provenance and robot-to-world transform where present.
- [x] Match existing full-episode joint/URDF FK outputs to exact video frames,
      masking unavailable world/TCP transforms and exposing local FK comparisons.
- [x] Implement/test absolute, head-relative and TCP/EEF-local encode/decode;
      test nonidentity anchor transforms and rolling horizon boundary/masks.
- [x] Render source-FPS continuous videos: current-head TCP/EEF/FK with source
      image, fixed-head anchor projection, source-model vs pinhole diagnostics,
      relative roundtrip overlays, and animated 3D label/FK paths.
- [x] Integrate playable videos, definitions, source links/provenance, legends,
      counts and numeric roundtrip/FK metrics into existing HTML and offline ZIP.
- [x] Verify focused tests, lint, video decode/frame count/duration, actual visual
      frames, embedded video bytes, player markup/JS, local HTTP and gateway.

Files: `scripts/data/video_projection_samples.py` (source extraction),
`scripts/data/video_fk.py` (audited FK tracks),
`scripts/data/video_trajectory_geometry.py` (representations and rolling windows),
`scripts/visualize_trajectory_videos.py` (MP4/report), associated focused tests.
Prior static projection and FK reports remain available under collapsed details.

## Completion evidence (2026-10-08 UTC)

Seven datasets each contain 180 consecutive states/images, frames30..209.
Six sources use30FPS/6seconds; RoboDojo uses25FPS/7.2seconds. All1260 JPEGs
decoded; source PTS and ordinal spacing checked. The initial33 pose/calibration/
mask rows match the previous samples within1e-12. Mecka read-only extraction
needed escalation for its mode000 source video; no source permissions changed.

Delivered25 H264/yuv420p faststart MP4s,1280x600: head, projection and relative
videos for allseven; additionally FK videos for fourrobots. All4500 encodedframes
fully decoded, durations/rates/counts checked. Relative head/local3D panels use
the same scale centered on current endpoint origin. FK3D is a growing trajectory;
source q is displayed. Head image comparisons show rolling targets t+1..t+32,
truncated at sample end. Absolute/relative comparisons use the same physical
endpoint and target row, without asserting native Mora actions.eef alignment.

Important audit correction: original OpenWAM EEF is absolute in source reference;
current112D adaptation is relative already. Mora96 defaults to source EEF rather
than eef_tcp. Mora has configurable random head/wrist selection, but current both
recipe does not invoke it. See outputs/fk_consistency/trajectory_videos/semantics_audit.txt.

49 focused geometry/FK/projection tests passed and allseven edited/new Python
modules/tests passed Ruff; git diff --check passed. Read-only review geometry and
FK scope passed. Fixed review findings: video switching preserves currentTime and
speed for both loaded/unloaded players; relative panes use a shared scale;
Genrobot no longer labeled human wrist and terminal header says nofuturetarget.

HTML verification:25 embedded videos byte-identical to MP4 files,7 cards,
no external rendering dependencies, JS syntax and loaded/unloaded/return-mode
time/speed logic passed; ZIP/mirror byte-identical and report rebuild idempotent.
HTML41,036,519 bytes; ZIP27,627,763 bytes. Local HTTP200 serves exactlatest HTML;
gateway302 redirects to PAI login as expected. No browser automation available;
actual encoded frames were visually inspected and all videos fully decoded.

Commands:

```bash
python3 -m scripts.data.video_projection_samples
python3 -m scripts.visualize_trajectory_videos
python3 -m scripts.visualize_trajectory_videos --report-only
python3 -m pytest tests/test_video_trajectory_geometry.py tests/test_video_fk.py tests/test_trajectory_projection.py tests/test_tcp_projection_report.py -q
```

URL: https://897628-proxy-18081.dsw-gateway-cn-shanghai.data.aliyuncs.com/tcp_coverage_v2_v3.html
