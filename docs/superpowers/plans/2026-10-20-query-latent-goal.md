# Query latent goal implementation plan

**Goal:** Select the original patch-denoising Stage1/Stage2 or an independent,
single-forward query encoder through model configuration, for smoke/3000h data.

**Architecture:** Learnable image and TCP queries cross-attend current observation
patches and instruction features. Only the resulting tokens condition Stage2.
During Stage1 an auxiliary joint image/TCP flow decoder learns from these tokens;
it has no direct observation/language bypass. Stage2 loads only encoder weights.

**Tech stack:** PyTorch, existing frozen Wan VAE/T5, existing curated reader,
safetensors, Hydra Stage2 / OmegaConf Stage1, pytest.

## Constraints

- Base commit 2490537; isolated branch codex/query-latent-goal.
- Existing running experiments and source/calibration labels remain untouched.
- `model.goal_representation`: `patch_denoise` (legacy default), `query_latent`.
- New tokens are `[B,N1+N2,D]`; no conversion to fake image VAE latents.
- Stage2 freezes the independent query encoder. No target/decode/noise inputs
  enter its inference API. Auxiliary decoder is absent from Stage2 parameters.
- Keep current task/subtask sampling, missing-view masks, and invalid TCP masks.
- Preserve legacy bundles, adapter state keys, and default numerical behavior.
- Sample means curated smoke YAML; `--smoke` means synthetic computational check.
- No large training launch is required; deliver executable commands and bounded tests.

## Tasks

### 1. Encoder, auxiliary training, serialization

Files: `openwam/model/goal/query_expert.py`, `openwam/train/goal_query.py`,
`tests/goal/test_query_goal.py`.

- [x] Test input-dependent learned queries, padded text / missing-view invariance,
  target independence, encoder gradients through auxiliary flow, decoder-free load.
- [x] Implement single forward transformer query encoder; configurable N1/N2,
  width/depth/heads. Default 64 image, 2 TCP, D512, 4 layers, 8 heads.
- [x] Auxiliary decoder uses existing joint flow with tokens as sole condition.
  Separate image/TCP readout losses bind groups to their names.
- [x] Save explicit query bundle contract, dependency identities, model dimensions;
  loader selectively reads encoder tensors for Stage2 and checks identities.

### 2. Stage1 entry and presets

Files: `scripts/train_goal_curated.py`, `openwam/train/goal_curated.py`,
`configs/goal/stage1_query_{sample,3000h}.yaml`.

- [x] Test config selection and bad model values; retain legacy default.
- [x] Dispatch construction/save; preserve existing immutable evaluation timeline.
- [x] Expose CLI model/query settings; disallow legacy checkpoint initialization
  into incompatible query topology.
- [x] Run synthetic training including checkpoint generation, image/TCP metrics.

### 3. Stage2 runtime and memory

Files: `openwam/model/goal/{runtime,conditioning}.py`,
`openwam/train/utils/stage2.py`, loader call sites, Stage2 evaluation.

- [x] Test direct token memory, both groups receiving action gradients in adapter,
  frozen encoder, dynamic attention masks and legacy regressions.
- [x] Implement query runtime/cache and adapter; no auxiliary decode at inference.
- [x] Enforce model/bundle agreement at launch, load, warm-start and resume.
- [x] Evaluation represents absent goal image/UV diagnostics as unavailable,
  continues actual action/video evaluation; never decodes query tokens as images.
- [x] Add query sample/3000h presets and launcher model selection.

### 4. Operation manual and verification

File: `docs/query_latent_training.md`.

- [x] Document both model diagrams, config table, real smoke/3000h data overrides,
  baseline/dependency prerequisites, Stage1 -> Stage2 commands and resume.
- [x] Verify command parsing/config composition, rejection of incompatible
  checkpoints, end-to-end tiny CPU/GPU paths as available.
- [x] Run affected legacy and new tests and inspect final diff.
- [x] Commit isolated implementation and report measured validation limits.
