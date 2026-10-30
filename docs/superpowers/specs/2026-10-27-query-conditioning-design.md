# Stage1 condition injection comparison

Base: 02edd62, with direct readout removed and ten-second episode-isolated history.
User approved independent normalization + linear cross-attention and a controlled
Wan-based PatchWAM-style prefix arm (not full FLUX PatchWAM).

Three modes, selected by model.query_conditioning:
- legacy_text: identical existing Linear -> Wan text MLP -> cross-attention;
  80 zero observation slots, 80 goal patches and 1 TCP patch.
- normalized_cross_attention: LayerNorm(Z) -> Linear(query_dim, Wan dim), bypass
  text MLP; preserve legacy 80 zero slots to isolate the context interface.
- prefix_self_attention: identical new adapter, inject Z as clean prefix replacing
  zero slots; prefix reads prefix only, noisy image/TCP read all tokens. Disable
  cross-attention and freeze its unused weights. This is the conditioning/mask
  mechanism transplanted into Wan, not a reproduction of FLUX's dual-stream blocks.

Default stays legacy_text for checkpoint/config compatibility. New explicit
presets select normalized_cross_attention. Bundle records mode; old bundles default
to legacy. No automatic cross-mode checkpoint conversion or optimizer migration.
Stage2 still consumes encoder.* only.

Experiment: one verified subtask instruction and matching subtask-terminal head
image/TCP. Choose a nontrivial earlier current frame (not goal minus one). All arms
share source observation, target, history, frozen VAE/T5 encodings, initial Wan and
query weights, loss weights, optimizer, step/noise schedule, and eval seeds. Fixed
history ages in seconds, 0.2 per-frame training dropout, no spatial augmentation.
500 updates per arm initially, evaluate 0/50/100/250/500; extend all equally only
if this budget does not establish the overfit trend. Record RGB MAE, pixel TCP
Euclidean error, fixed-noise validation losses, condition interventions, gradients,
GPU peak memory and training step time. Single-sample fitting cannot establish
instruction grounding or generalization.

All outputs and model snapshots on CPFS; HTML append-only evaluations. Preserve
existing DLC jobs and pinned code. Check input overlays before expensive training.
