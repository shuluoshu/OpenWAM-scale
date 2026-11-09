# Language-conditioned C* implementation

Approved scope: selected task/subtask text enters C* construction; preserve the
120 × 3072 production interface, ten-frame history, real-age PE, history masks,
and selected-segment terminal image/TCP supervision. Do not modify either running
DLC or its checkout. Work starts from cache optimization commit 58fd1e5.

## Design

Add an explicit `language_conditioning` option, default false for legacy models.
The enabled adapter retains native Wan text cross-attention and owns a trainable
copy of the native text projection. Pool history exactly as before, fuse pooled
history and current spatial tokens with text in the native adapter, then return
only the 120 current tokens. Initialize the residual fusion gate to 0.1 so text
has a path immediately. With absent or entirely masked history, keep the text
path active. Mask padded text before projection and preserve lengths.

Keep visual-only v1 bundles unchanged. Language-conditioned bundles use v2 and
record their mode in metadata. Reject mode mismatches in configuration, bundle
loading, Stage1→Stage2 preparation, and Stage2 resume. The frozen Stage2 encoder
keeps the text projection and cross-attention while pruning the target decoder.
Do not change dataset sampling or add a language dropout objective.

## Execution and validation

1. Add failing tests for text sensitivity, absent/masked history, padding,
   gradients from terminal losses, strict export/load, runtime cache behavior,
   and incompatible assets/configurations.
2. Implement adapter, frozen encoder, bundle format and Stage2 runtime plumbing.
3. Add separate Stage1 and Stage2 presets; keep existing launch presets intact.
4. Run relevant legacy and new CPU tests, native CUDA smoke validation, and
   independent code review. Record evidence, commit, and archive dialogue.

Structural text sensitivity is a wiring check; trained instruction adherence
still requires a new Stage1 training run and downstream evaluation.
