# URDF / FK consistency audit implementation plan

**Goal:** Audit one complete episode per currently configured dataset (InternData, AgiBotWorld, RoboDojo, RobotWin), render an offline HTML report with quantitative pose errors and visualizations.

**Architecture:** A tested NumPy/SciPy URDF chain evaluator plus independent dataset adapters produce auditable transforms, joint labels and metadata. A report script computes position/angular and current-frame-relative errors, embeds trajectory plots, sampled source frames, provenance and limitations into one standalone HTML.

**Scope:** Source datasets, URDFs and existing training reader remain read-only. CPU only; no commits or deployment. Episode 0, rows 1 onward are future-state joint/pose supervision, with row 0 kept as reference where available. Never use normalized joint deltas directly as absolute FK input. Preserve old workspace changes.

**Comparison contract:** Prefer documented joint/base/tool mappings; do not fit away error and label it verified. When a mapping is inferred or missing, retain a raw/assumption baseline and explicitly mark unverified. Any fitted diagnostic must be separate. Report maxima, medians, RMS and per-frame translation (mm) and geodesic rotation (deg). A 1 mm / 1 deg threshold is an audit criterion, not a dataset-provided guarantee.

- [x] Test and implement `scripts/data/urdf_fk_audit.py`: batched FK for fixed/revolute/continuous/prismatic joints, strict required joint values, origin-before-motion, arbitrary ancestor base, pose conversion/errors and episode loading.
- [x] Implement independently auditable adapters `fk_audit_{interndata,agibot,robodojo,robotwin}.py`; examine original assets/calibration and emit NPZ + JSON per source.
- [x] Implement `scripts/visualize_fk_consistency.py`: run/reuse adapters, render all four sources with trajectory/error plots and camera contact sheets, HTML inline images and optional offline trajectory interaction; JSON/NPZ downloads alongside.
- [x] Run toy-FK unit tests and real full-episode audits; inspect resulting figures/HTML and verify all assets and finite metrics.
- [x] Review report and coordinate assumptions, record explicit limitations, deliver HTML path and concise findings.

## Results and verification

- Output: `outputs/fk_consistency/index.html`, standalone HTML with 13 embedded figures/contact sheets, inline Plotly, per-dataset NPZ/JSON companions.
- 4,679 next-state targets in episode 0 across four datasets: 267 InternData, 3,237 AgiBot, 702 RoboDojo, 473 RobotWin.
- InternData independent common-arm-base EE max 0.0000292 mm / 0.00000445 deg; robot/TCP derived-transform diagnostic explicitly separate. Corrected prior wrong choice of per-arm base frames.
- AgiBot common-torso-cancelled inter-arm max 0.0169143 mm / 0.0009421 deg. Absolute FK remains unverified because release lacks torso joints.
- RoboDojo actual source dual_x5 configuration + X5A URDF absolute max 0.158871 mm / 0.0173484 deg; relative motion max 0.000615 mm. Base config found in actual simulator source; no label fit used for acceptance.
- RobotWin metadata world/base + same URDF EEF reconstruction max 0.00006128 mm / 0.00000210 deg; not independent physical calibration.
- Actual reader first 32 unnormalized joint deltas + current joints reconstruct audit q targets, max mismatch <= 1.1e-19 rad across four profiles.
- 17 focused tests passed (4 new FK/math tests plus existing relative reader tests). Changed audit files pass Ruff, git diff check clean. Full CLI regenerates all adapter outputs and HTML.
- Independently reviewed math, conventions, full NPZ metrics and source-code mounting definitions; no blockers. Matplotlib figures visually inspected; HTML images decoded, all artifact/anchor links checked, inline JavaScript syntax validated with node --check. Browser runtime not available in this environment.
