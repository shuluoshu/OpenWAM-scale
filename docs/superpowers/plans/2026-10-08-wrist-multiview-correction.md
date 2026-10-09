# Wrist projection correction, 2026-10-08

User request: re-evaluate Agi 245mm using left/right wrist videos, tune it and diagnose missing Genrobot wrist TCP tracks. Existing report/video design and manual endpoint adjustment are authorized. No training changes.

- [x] Reproduce Agi head/wrist discrepancy with same physical point and source camera matrices; inspect original calibration metadata.
- [x] Compare 160/190/200/210/245mm; keep a shorter 200mm preview with explicit unresolved cross-view residuals, not a claim of recovered physical calibration. Preserve identical TCP across cameras.
- [x] Trace Genrobot source conversion and Finger CAD, verify raw pose vs canonical EEF and optical rotation on real wrist frames; implement bounded CAD-derived chain with source validity and identity-calibration guard.
- [x] Focused regression tests (65 pass), source audit and projection screenshots.
- [x] Regenerate Agi main modes, add synchronized head/left/right candidate videos, regenerate Genrobot moving/fixed wrist tracks, update portable original HTML.
- [x] Decode new/changed media, validate source calibration/joins, independent URDF composition, HTML media hashes/ZIP/JS, HTTP latest hash; document result and remaining Agi residual.

Evidence: outputs/fk_consistency/tcp_origin_audit/agibotworld/multiview_offset_audit.json and outputs/fk_consistency/genrobot_wrist_audit/.

Validation complete: 65 tests; 24 supplementary modes/5760 decoded frames, three Agi clips ×9 modes ×300 frames plus base Agi main5×180; independent CAD/pinhole/DS projection checks, source joins, 65 embedded-media hashes, ZIP/mirror and JS checks pass. Local HTTP200 latest hash; gateway302 to PAI login. Agi cross-view alignment remains imperfect and explicitly documented.
