# Wrist-camera videos and AgiWorld TCP tuning

Goal: add physical wrist-camera TCP projections to the existing portable HTML and compare AgiWorld 240/245/250 mm offsets before selecting a slight increase.

Scope: existing seven source families and ten video samples. The optional scope question received no answer during implementation; completed the stated assumption of all existing samples. The 27-row recoverability audit is not claimed to have 27 new video samples.

Design: preserve current video modes. Add one synchronized left/right moving wrist-camera video and a fixed-first-wrist-camera comparison when calibrated source videos exist. Each frame projects world TCP[t:t+32] using its paired camera/image at t; use actual source K, lens model, image resolution, episode timestamps and masks. Raw views are shown with explicit masked-projection reason if calibration is incomplete. Sources with no wrist cameras receive a visible unavailable entry, not a substituted head view. Agi gets a separate 240/245/250 mm comparison video with enlarged end-effector crops; selected preview definition remains report-only.

Files: new scripts/data/wrist_projection_samples.py for extraction/provenance and validated camera arrays; new scripts/visualize_wrist_videos.py for rendering and mode-metrics integration; extend scripts/visualize_trajectory_videos.py report descriptions and coverage table. Keep the established head/static/FK geometry modules.

Tasks:
- [x] Audit camera fields and source frame conventions across all existing samples; inspect missing Genrobot optical chain.
- [x] Extract exact wrist image ordinals and corresponding per-row calibration, verify malformed/missing masks with focused tests.
- [x] Generate and inspect Agi offset comparisons; choose 245 or 250 mm using the same fixed transform across clips, record manual-preview limitations.
- [x] Refresh Agi main videos/static poses with selected offset; render wrist and length-comparison modes for all applicable samples.
- [x] Embed modes and all-source wrist availability/reason table in original HTML, preserve #previous-fk and existing anchors/offline ZIP.
- [x] Validate source joins, all new video decodes, projection constraints, mode hashes, test suite, selected screenshots and HTTP latest-report hash.

No source dataset, training reader or training masks are modified. Execute inline in the existing workspace; existing visualization design and the user's instruction authorize this extension.

Result: selected +245mm manual preview, 240/245/250 comparison in four Agi clips. Added 19 videos (60 total): 14 calibrated moving/fixed wrist composites, one Genrobot raw wrist composite with projection masked, four length comparisons. LightWheel/Mecka have no wrist camera in current samples. 61 tests pass; all 4,500 new video frames decoded, source joins/K/E verified, independent projection/FOV checks pass. HTML/ZIP/mirror/media hashes and JS wrist switching pass. Local HTTP 200 latest hash, gateway 302 to PAI login. No browser automation; decoded frames visually inspected.
