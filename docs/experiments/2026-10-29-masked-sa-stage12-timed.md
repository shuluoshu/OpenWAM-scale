# Masked-SA Stage1 → frozen visual memory → two Stage2 arms

Requested outcome: automatically continue the latest verified Stage1, freeze one
checkpoint, adapt and train joint/none Stage2, and publish measured fitting and
prediction comparisons within eleven hours of 2026-10-28 16:38:21 UTC.

## Fixed contract

- Worktree starts at e182d1f; leave the dirty main checkout untouched.
- Stage1 starts at `outputs/stage1_masked_sa_20261028/run_v2/visual_history.safetensors`.
  Original Wan/VAE dependency is official OpenWAM checkpoint_step_154000.
- Stage1 uses the 92 anchor_0/anchor_1 smoke examples from 12 families, actual
  per-example instruction, history-only pooling (32 slots), visual fusion and
  native masked SA/text CA. LR 2e-5 Wan / 2e-4 history, warmup100, clip1.
- Alternate anchors belong to the same episodes. Four were used by earlier
  diagnostics. Reports must not claim unseen-episode generalization.
- Both Stage2 arms receive identical frozen 120×3072 fused current visual
  tokens, with the original per-view key-validity mask. Text remains native
  Stage2 conditioning. This is not the superseded 64+2 query encoder.
- Both Stage2 arms start from the same validated 112-action baseline and use
  the existing data_v4 21 windows, normalization, action masks and FK payloads.
  Joint trains video+action; none trains action with current observation only.
- Only case18 has matched, active FK supervision. Missing geometry stays N/A.

## Execution

1. Continue Stage1 for approximately90min, export a portable frozen bundle and
   evaluate fitted/alternate anchors with same-model single/history inputs.
2. Extend runtime with dedicated representation/adapter and pruned frozen
   encoder; test exact encoder equivalence, masks, freezing and relocation.
3. Run a short real Stage2 train+generation smoke for both arms. Reject failed
   smoke before launching long arms. CPU tests do not substitute for this.
4. Supervisor allocates equal remaining arm budgets, targeting3.5–4h each,
   preserving at least1h for final evaluation, rendering and failure recovery.
   LR multiplier1.5, warmup100. Record effective LRs, actual wall time, updates,
   GPU memory, init/config/data/teacher hashes and frozen-teacher assertions.
   Prefer single-GPU BF16 forward / FP32 master AdamW with sequential buffers;
   if its real smoke fails, revalidate BOTH arms with native CPU optimizer
   offload. Never compare arms trained with different optimizer backends.
5. Save immutable evaluations periodically and at end. Publish loss vs updates
   and wall time, predictions, direct TCP projections, joint→FK projections,
   source validity and projection eligibility. Never compare raw total losses
   as if the arms optimize the same objectives.
6. Check local HTTP200, assets and video decode. DSW proxy login302 is an
   authentication requirement, not proof of a publicly accessible page.

## Review/verification

- Focused CPU tests: old query/patch contracts unchanged, new visual encoder
  matches Stage1 condition_tokens, all-invalid views masked, teacher frozen.
- Real smoke: finite gradients/loss and predictions, actual optimizer update,
  nonzero valid FK loss where available, native generation for joint/none.
- Geometry: relative joints restore q_anchor+delta; current source registration
  held fixed across predicted horizon, explicitly labeled rather than treated
  as independent whole-body calibration. No future-GT-fitted transform.
- Final report accurately distinguishes running/failed/completed stages and
  train-set fitting from generalization. No unsupported success claim.

## Initial execution evidence

- Stage1 started at 2026-10-28 16:47:40 UTC, PID642210; approximately0.44s/update.
- 82 focused regression tests passed; supervisor sequence/fallback/error tests
  are separate. Actual full native Stage2 construction and visual encoding
  passed on CPU (12,698M total params, 6,117M trainable, frozen encoder195M;
  C* shape1×120×3072, layer KV30×1×120×3072, finite, no CUDA allocation).
- The Stage2 encoder intentionally retains only Stage1 current patch embedding,
  time modulation, visual history pooling and its visual adapter; the auxiliary
  image/TCP target decoder is omitted. It exports visual C*, not generated goals
  or the superseded text-fused 64+2 queries. Stage2 retains its own native text CA.
- Appended memory has no native RoPE in MoT, so the dedicated adapter adds
  learned12×10 row/column positions in its512-wide bottleneck; masked cameras
  remain unreadable. Both arms use the identical adapter contract.
- A read-only interim checkpoint replay on4 fitted anchors produced image MAE
  3.8–7.5/255 and valid TCP14.9–21.6px. This is interim, not final freeze quality.
- Offline renderer tested with immutable prior Stage2 predictions, separately
  under render_fixture; those are NOT published as new experiment results.
  Direct/FK metrics score finite out-of-frame predictions; rendering never
  clamps them to the image edge. Source-anchor camera mapping is audited.
