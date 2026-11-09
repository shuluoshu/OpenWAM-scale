# Language-conditioned C*

Branch: `codex/cstar-text-conditioning-20261109`, based on cache optimization
`58fd1e5`. Both active DLC jobs and their checkout remain unchanged. The live
checkout is still clean at `10dc74e`.

## Behavior and configuration

Enable `model.visual_history.language_conditioning: true` in Stage1. History
pooling retains the same ten-frame input, real-age PE and masks. The adapter now
retains native Wan text cross-attention and a dedicated trainable copy of the
native text projection. It fuses pooled history and current spatial tokens with
the selected instruction, then returns only the current 120 tokens. Production
width remains 3072. The residual gate starts at 0.1 so text affects C* from the
first update, including when all history is masked or history is absent.

The target decoder still receives text and the same terminal image/TCP labels.
No future target enters C*. Dataset sampling is unchanged: the selected task or
subtask supplies its text and its own segment end for both image and TCP. The
finite shuffled population mixes both granularities without enforcing 50/50.
History dropout remains configurable through `dataset.history.frame_dropout_prob`
(and Stage2 `dataloader.history`); these presets inherit the existing value 0.
This change does not introduce text dropout or an additional semantic objective.

New configurations:

- `configs/goal/stage1_text_visual_history_relative112_production.yaml`
- `configs/goal/stage1_text_visual_history_relative112_gpu_zero1.yaml`
- `configs/goal/stage2_text_visual_history_joint_gpu.yaml`
- `configs/goal/stage2_text_visual_history_none_gpu.yaml`

Visual-only defaults and v1 bundles remain supported. Text-conditioned bundles
use `openwam.visual_history.v2` and require `goal.language_conditioning: true` in
Stage2. Loading, pipeline preparation and resume reject mode mismatches. The
frozen Stage2 encoder keeps the C* text projection/cross-attention and removes
the target decoder. Runtime forwards the selected contexts and their lengths;
goal prediction cache keys already include the text. Frozen VAE caches stay
text independent and retain the earlier performance fixes.

Start a new Stage1 run with a new output directory using a text configuration;
the production preset initializes the new path from the foundation weights.
An old visual-only full optimizer checkpoint cannot directly resume into this
topology. Train and validate Stage1, export its v2 bundle, then use the matching
Stage2 joint or none configuration. No DLC launch was performed in this task.

## Verification

The new tests failed before implementation because the constructor did not
support language conditioning and Stage2 did not accept its format. Final
focused regressions: **81 passed**, covering text sensitivity, no/empty history,
padding NaNs/lengths, separate terminal-loss gradients, checkpointing, strict
bundle loading, Stage2 runtime cache behavior, asset relocation, exact interrupted
resume, presets, and retained history/cache behavior. Static independent review
found no actionable correctness issue. `git diff --check` passed.

Command (from this checkout, Python environment `openwam`):

```bash
python -m pytest -q tests/goal/test_text_visual_history.py \
  tests/goal/test_visual_history.py tests/goal/test_visual_history_stage2.py \
  tests/goal/test_visual_history_unified.py tests/goal/test_history_cache_batching.py \
  tests/goal/test_visual_history_presets.py tests/goal/test_stage12_pipeline.py \
  tests/goal/test_stage12_config_validation.py --tb=short
```

The wider `tests/goal` run reported 857 passed, 9 failed, 1 skipped (before the
final additional language-mode cases were added). Eight deterministic failures
were reproduced unchanged on baseline `58fd1e5`: CPFS path requirements in
`test_dlc_offline_ready`, removed `target_path` fixture assumptions, an empty
history augmentation fixture, and stricter Genrobot TCP prerequisites. A ninth
distributed test timed out during concurrent heavy model bootstrap; it passed
in 9.17 s after bootstrap ended, and the baseline passed in 8.99 s. These wider
issues were recorded rather than changing unrelated dataset/admission behavior.

## DSW native BF16 validation

Artifact root:
`/mnt/data/cpfs/b5/yeqingda/openwam-0911/OpenWAM/outputs/cstar_text_validation_20261109/`.
`validate_real.py` is the exact bounded experiment script; `result.json` and
`run_with_tcp.log` hold the accepted run. `regression.log`,
`baseline_failures.log`, `baseline_distributed.log`, and
`distributed_idle_recheck.log` record CPU checks and baseline comparisons.

NVIDIA H20-3e, full 30-layer foundation DiT, real frozen VAE and T5, one actual
LightWheel observation/terminal image with two valid terminal TCP labels, and ten
real history frames. No optimizer update was performed. The first diagnostic
case had no TCP labels and is retained as `result_image_only.json`; it is not
used as evidence of TCP supervision.

Accepted output: `[1,120,3072]`. Image flow loss 0.61796546; TCP loss 1.09725034;
total weighted loss 1.40623307. The text projection's separate gradient L1 sums
were 60.60079 from image loss and 206.99315 from TCP loss. The total-loss gradients
also reached text cross-attention and history pooling and were finite/nonzero.

Replacing the actual instruction while holding images/history fixed changed
C*: RMS difference 0.00218342 with history and 0.00217009 with history fully
masked or absent. Masked padding NaNs had no effect. Frozen Stage2 encoder output
matched Stage1 C* exactly, and all encoder parameters were frozen. The
HistoryTokenPool AST, including time PE, matched `58fd1e5` exactly.

One forward/total-loss backward took 0.6998 s; peak tensor allocation was
20.9935 GiB. The script additionally checked the two losses with separate
backwards. This is a bounded wiring check, not a steady-state throughput or
32-rank measurement. Text sensitivity and nonzero gradients do not establish
learned semantic correctness; a new training run and held-out evaluation remain
necessary.
