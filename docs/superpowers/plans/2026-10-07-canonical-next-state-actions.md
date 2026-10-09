# Canonical EEF next-state action targets

**Goal:** In OpenWAM's canonical reader, construct each raw action target from the next observed EEF state, independently of the source's precomputed action horizon.

**Architecture:** Read `state[t:t+H+1]`, use row zero as proprio and rows `1:H+1` as absolute EEF20 targets. Keep normalization, 80D scatter and video sampling. The reader remains local to OpenWAM; no MoraVLA code or source parquet is modified.

**Tech stack:** Python, NumPy, PyArrow, pytest, existing OpenWAM reader hooks.

## Constraints and decisions

- User explicitly selected OpenWAM's canonical reader, not the `pretrain` repository.
- Existing feature branch and user edits remain in place. No commit or push is required.
- A 33-row window produces 32 targets: `state[t+1:t+33]`.
- Require at least two real states. At episode boundaries, only `actual_raw_len - 1` targets are supervised; remaining positions are zero-padded and masked.
- Source `actions.eef` may encode +3 and is never used for target values. Keep the existing `action_key` argument as provenance identifying the superseded source field for compatibility with profiles; reading/validating values depends only on `state_key`.
- Stats action population becomes states `1:end` separately per episode, never the first row of the next episode. Keep proprio stats over observed states.
- Stats must declare `action_target_source: next_state` and `action_target_offset: 1`; reject old stats with a regeneration instruction.
- Trace/report fields identify target state key and frame indices; padded target indices are null.

## Implementation and checks

- [x] Add reader tests with deliberately different source-action values, changed/missing source action fields, complete windows, episode-tail masks and stale stats. Run them and observe failures before implementation.
- [x] Update canonical `_action_20d`, `_finalize_action`, `_train_min_window_len`, schema requirements and trace. The relevant operation is `raw = validate(win[state_key])[1:]`, followed by existing normalization; finalization receives `actual_raw_len - 1`.
- [x] Add stats tests proving action population equals the union of each selected episode's `states[1:]`, with count `sum(length-1)`, and rejecting episodes shorter than two rows. Observe failures, then implement and version the provenance.
- [x] Update report/open-loop manifests and usage/config comments to state the new temporal contract.
- [x] Run canonical/stats/config/open-loop/report tests, then LeRobot reader regression tests. Keep same-directory pytest inputs adjacent because this environment has a known interleaved-directory fixture collection issue.
- [x] Regenerate the four bounded stats and sample reports using only the existing episode 0/1 profiles. Preserve historical inference predictions; they are not current verification.
- [x] Validate actual reader samples for all four profiles: raw first GT exactly equals next-row EEF20, raw chunk equals `states[1:33]`, shapes/masks remain correct. Record examples and checks in the session worklog.
- [x] Review changes and whitespace; report validation scope without claiming a new GPU training run.
