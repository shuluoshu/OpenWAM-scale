# Visual-history masked-SA Stage1

This branch adds `VisualHistoryGoalExpert` on top of `PatchGoalExpert`.

- Current three-view canvas: VAE -> native Wan patch embedding -> 120 spatial tokens.
- Ten strictly past head-frame slots: VAE per frame -> native patch embedding -> history-only learned pooling, **32 tokens total**, width 3072. Pool hidden width 256 with 8 heads. Token count is configurable.
- Pure visual fusion: a copy of pretrained Wan block 0, without its text cross-attention, operates on history/current. Abstract history slots use identity RoPE; current tokens retain their native grid RoPE. Read out the current-token delta and add `tanh(gate) * delta`, with gate initialized to zero. Empty history returns original current tokens exactly.
- Main Wan: `[current 120 | noisy goal 80 | noisy TCP 1]`; current reads current only, target rows read all valid current/target tokens. Current diffusion time is zero. Native text embedding and text cross-attention remain active.
- History receives neither text nor future GT. Frozen VAE features can be cached; trainable tokens are never cached. This replaces the earlier 66-query image/text bottleneck rather than adding noise to those queries.

## Usage

Load pretrained Wan weights **before constructing** the expert, so the visual adapter copies pretrained block weights:

```python
from openwam.model.goal.visual_history import VisualHistoryGoalExpert

expert = VisualHistoryGoalExpert(
    pretrained_dit, frozen_codec, dit_config=dit_config,
    num_history_tokens=32, pool_dim=256, pool_heads=8,
    adapter_block=0, use_gradient_checkpointing=True,
).to(device='cuda', dtype=torch.bfloat16)
expert.train_stage_a()
result = expert(
    observed_canvas, text_context, goal_targets,
    context_lens=text_lengths, observed_view_valid=view_mask,
    history_images=history_rgb, history_ages=ages_seconds,
    history_valid=history_mask, history_episode_id=episode_ids,
)
result['loss'].backward()
```

The RGB history shape is `[B,10,3,1,256,320]`. Alternatively provide `history_latents=[B,10,48,1,16,20]` and omit `history_images`. Ages/validity are `[B,10]`, oldest-to-newest with strictly positive decreasing valid ages. Episode IDs are optional int64 `[B]`; provide them for cache isolation. To compare single-frame input with the same model, mask all history slots; do not instantiate another model.

The main conditional interface remains 120 spatial tokens, so this is not compatible with the old Stage2 66x512 Z input contract. Stage2 adaptation is outside this change.

## Validation and artifacts

Run from this worktree using the existing `openwam` environment:

```bash
/root/miniconda3/envs/openwam/bin/python -m pytest \
  tests/goal/test_visual_history.py tests/goal/test_patch_expert.py \
  tests/goal/test_multiview_patch.py tests/goal/test_query_goal.py \
  tests/goal/test_query_history.py tests/goal/test_query_conditioning.py -q

/root/miniconda3/envs/openwam/bin/python -u scripts/experiments/overfit_visual_history.py \
  --root /mnt/data/cpfs/b5/yeqingda/openwam-0911/OpenWAM \
  --output /mnt/data/cpfs/b5/yeqingda/openwam-0911/OpenWAM/outputs/stage1_masked_sa_20261028/reproduction
```

The experiment loads official `checkpoint_step_154000.safetensors`, uses the real smoke caches, and runs 200 singleton + 600 two-scene + 800 four-scene updates. Four scenes deliberately share one cached instruction; this tests visual routing, not semantic instruction following. Original valid TCP labels are retained. Frozen VAE/T5 match the earlier audited dependencies. BF16 forward uses FP32 master AdamW, Wan learning rate 1e-5, new history parameters 1e-4, clip norm 1, history dropout 0.2, and a fixed evaluation seed.

Final comparisons include real/no-history/wrong-history/wrong-current/wrong-all/blank visual inputs, held anchors in the same episodes, and true intermediate denoising x0 images. These are overfit checks and do not demonstrate cross-episode generalization or a history advantage. Joint flow losses include valid TCP; no Stage2 or FK is run.

The authoritative corrected run is `outputs/stage1_masked_sa_20261028/run_v2`. The initial `run` was interrupted after review found a raw-RGB episode-ID conversion bug; it is retained only for audit. New tests reproduce the failure and cover raw RGB against cached latents, mixed empty history batches, checkpoint gradients, native-prefix equivalence, target isolation, text masks, and serialization.

Use `openwam.train.goal_visual_history.load_visual_history_bundle` to reconstruct the portable checkpoint with an explicit frozen VAE, text width, and matching text fingerprint. `scripts/experiments/reload_visual_history.py` verifies fresh-process reconstruction on actual RGB inputs. `scripts/experiments/report_visual_history.py` renders only saved real predictions and inputs; it requires an environment with NumPy, Pillow, and matplotlib.

## Corrected run results

After 1600 optimizer steps (about 908 seconds including initialization, evaluation, and save/reload), fixed-noise image MAE on the four training anchors:

| Scene | Before training | Copy current | No history | With history | Valid TCP px |
|---|---:|---:|---:|---:|---:|
| ABC | 35.58 | 16.94 | 3.32 | 3.30 | unavailable |
| EgoDex | 16.90 | 5.12 | 3.11 | 3.18 | 4.95 |
| LightWheel | 48.65 | 44.90 | 6.98 | 6.99 | 3.22 |
| Morphirobot | 16.85 | 7.02 | 3.67 | 3.66 | 6.03 |

All four predictions retrieve their own target scene by pixel MAE. Swapping all visual inputs produces the other scene's normal prediction pixel-for-pixel under identical text/noise. Swapping only the current frame also changes scene identity; swapping only history does not. This demonstrates current-image conditioning on trained scenes, not meaningful history improvement. The learned gate is approximately -0.00098; history changes the condition features by small nonzero amounts. Three TCP-valid scenes use original validity masks; no Stage2/FK metric is involved.

Held anchors in the same episodes have image MAE 15.94 / 4.31 / 13.95 / 5.14 and retain the correct scene. Those are not held episodes, and they do not establish generalization.

Fresh reconstruction checks all saved tensors exactly and validates frozen VAE/text identities. Changing parameter `requires_grad` state between train-time evaluation and frozen inference causes small BF16 numerical differences (mean pixel difference 0.16–0.41 on the 0–255 scale); this is reported separately from weight equality and raw-RGB/cache equivalence. The reload diagnostic repeats evaluation with the original trainability state to test strict pixel reproducibility.
