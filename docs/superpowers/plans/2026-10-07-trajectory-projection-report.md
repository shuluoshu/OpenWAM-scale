# Trajectory projections in the TCP coverage report

User authorized integrating previous trajectory visualizations into the existing
HTML and distinguishing projection methods. Keep source datasets and training
readers unchanged. Preserve the report URL on DSW port 18081.

Design: embed the existing four-source FK report; add actual-image trajectory
comparisons for those sources plus V0_3 Genrobot, LightWheel and Mecka. Separate
native calibrated camera projection from pinhole-only diagnostics; separate fixed
anchor camera from a later camera and its matching image. Never overlay points
expressed in different moving cameras onto a single anchor image as if aligned.
Distinguish source EEF/wrist, declared TCP, and explicitly labelled TCP candidates.
Show head-reference and TCP-local 3D coordinates separately from pixel projection.
Labels use state t+1 onward, anchored at t, with 32-step examples where available.
All image assets and existing interactive FK plots embedded for offline viewing.

- [x] Implement/test projection math with finite/domain/depth masks.
- [x] Extract V0_3 calibrated samples and provenance; verify actual video alignment.
- [x] Integrate prior four datasets, render endpoint/view/model comparisons and 3D frames.
- [x] Embed prior FK report, preserve coverage audit, update ZIP and verify HTML/HTTP.

Validation: independent projection reference tests, per-source numerical/visual
inspection, no external render dependencies, correct titles/units/masks, and
gateway URL checks. No claim of full calibration accuracy from a plausible overlay.

Completed validation: 34 focused tests passed; five new Python modules/tests passed
Ruff. Reviewed actual-image overlays for all seven samples, Genrobot native versus
pinhole, and Mecka matched moving snapshots. Independent review found two report
issues, both addressed: wholly off-image streams no longer appear in legends and
visible point-count tables describe omissions; all eight original FK NPZ/JSON
downloads are embedded as data URLs with filenames.

Final artifact validation: seven sample cards/selectors, 52 valid embedded images,
eight byte-identical embedded downloads, no external rendering dependencies,
JavaScript syntax and selector behavior passed, ZIP HTML and mirror byte-identical.
HTML 23,681,224 bytes; ZIP 14,788,972 bytes. Local server returned HTTP 200 with
the exact final HTML; DSW gateway returned the expected HTTP 302 to PAI login.
Browser automation and authenticated client-side gateway access were not tested.

Open: https://897628-proxy-18081.dsw-gateway-cn-shanghai.data.aliyuncs.com/tcp_coverage_v2_v3.html
Re-render: `MPLCONFIGDIR=/tmp/openwam-projection-mpl python3 -m scripts.visualize_tcp_projections --reuse-samples`.
