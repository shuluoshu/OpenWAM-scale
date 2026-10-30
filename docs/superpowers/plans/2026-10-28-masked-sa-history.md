# Masked SA visual-history Stage1 implementation plan

> Execute inline with the executing-plans workflow. User approved implementation and overfit validation on 2026-10-28.

**Goal:** Implement learned all-history pooling, visual-only history/current fusion, and native Wan masked-SA + text-CA decoding; demonstrate input-dependent scene reconstruction with real checkpoints.

**Architecture:** History-only N-query pooling outputs N Wan-width tokens. A copied Wan block without text CA fuses these with current native spatial tokens, with a zero-initialized residual gate. The existing PatchGoalExpert receives these current tokens and keeps native text CA, image/TCP targets, and current-prefix isolation.

**Tech stack:** PyTorch, existing Wan/VAE, pytest tiny real Wan fixtures, BF16 + FP32-master AdamW, existing cached real smoke samples, static HTML reports.

## Constraints

- Worktree: outputs/stage1_masked_sa_20261028/code, base 5cd534c, branch codex/stage1-masked-sa-history. No changes to previous snapshots or runs.
- Official checkpoint_step_154000.safetensors; reuse verified VAE/T5 identities and cached text features.
- N=32 total history tokens as an initial configurable engineering choice; not an empirically selected optimum. Pool hidden width 256, 8 heads.
- No text/current/future target inputs to history pool. No future targets in fusion. Existing observed-view and history-valid masks remain enforced.
- Mask: current rows read current only; image/TCP rows read current + image/TCP. Text remains native cross-attention. Targets remain 80 image + 1 TCP, not noised 66.
- Tests and trials must distinguish no-history fallback, zero gate, and learned nonzero history contributions.
- Do not claim generalization from train-set overfit; include wrong visual, blank visual, and history interventions.

## Task 1: tests before implementation

Files: tests/goal/test_visual_history.py.

- [x] Write tests importing VisualHistoryGoalExpert through a small fixture using TINY_DIT and TinyVAE.
- [x] Check gate-zero exact equivalence to native multiview PatchGoalExpert with identical state and noise.
- [x] Check empty/masked-NaN history is finite and exactly independent of padding; history features and time affect outputs when gate is open.
- [x] Check history pool emits a total N tokens, no text interface exists, prefix output is independent of target/noise/timestep, text padding is sanitized, valid text changes predictions.
- [x] Check gradients reach pool/adapter/native text, generation accepts history, and strict state-dict roundtrip reproduces predictions.
- [x] Run `python -m pytest tests/goal/test_visual_history.py -q`; confirm expected missing-feature failure.

## Task 2: visual history module and minimal decoder extension

Files: openwam/model/goal/visual_history.py, openwam/model/goal/patch_expert.py.

- [x] Add optional `observation_tokens=None` to PatchGoalExpert velocity/forward/generate. Validate [B,n_obs,dim], replace only current tokens, preserve default path and original masks/positions.
- [x] Sanitize masked text before native embedding to prevent masked NaNs reaching V values.
- [x] Implement `VisualHistoryGoalExpert.condition_tokens(observed, *, observed_view_valid, history_images=None, history_latents=None, history_ages=None, history_valid=None, history_episode_id=None) -> Tensor[B,M,D]`.
- [x] Implement learned spatial/age-aware attention pooling over all history tokens, using an empty-history-safe dummy key and zero outputs for empty samples.
- [x] Copy a Wan block for visual fusion, remove unused text modules, use clean time modulation, neutral rotary positions for abstract history slots and native current-grid rotary positions.
- [x] Fuse `C + tanh(gate)*delta_current`, and explicitly bypass empty-history samples.
- [x] Keep frozen-VAE feature caching only; never cache trainable pooling/patch embeddings. Forward/generate reuse existing joint image/TCP losses and scheduler.
- [x] Run new tests plus existing patch/multiview tests. Fix regressions before GPU training.

## Task 3: reproducible overfit and counterfactual validation

Files: scripts/experiments/overfit_visual_history.py; output ../protocol.json, metrics.json, train.jsonl, report/.

- [x] Load official Wan/VAE; record source/checkpoint identity, file hashes, seeds, optimizer groups, chosen source rows.
- [x] Baseline evaluation before updates; singleton ABC screenshot sample training, then two-scene same-text routing training; optionally widen to four scenes if runtime permits.
- [x] Use identical evaluation noise for real/no-history/wrong-history/wrong-current/wrong-all/blank conditions, report per-source target MAE and nearest-scene retrieval instead of only aggregate loss.
- [x] Save actual input canvas, ten history slots with validity/ages, GT, VAE reconstruction, final prediction, and denoising x0 trace at fixed iterations. TCP metrics only where original valid GT exists.
- [x] Confirm nonzero finite history gradients after gate opens and nonzero condition sensitivity. Compare generated scene with supplied input after swapping visual conditions.
- [x] Save complete model state + metadata and strict-reload verification; preserve logs and full run scripts.

## Task 4: review and delivery

- [x] Inspect rendered contact sheets; distinguish successful scene routing from fine-detail errors and future-motion correctness.
- [x] Run relevant tests, inspect git diff, record implementation status and limitations in report.
- [x] Serve report under existing report-only server, validate assets, localhost SHA256 and DSW login status.
- [x] Deliver branch/code/checkpoint/report locations with measured results and unresolved limits.
