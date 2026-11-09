# Causal language-conditioned C* and explicit architecture continuation

The new `causal_history` option changes both history pooling and native fusion.
The external C* remains 120 × 3072, history remains ten strictly past T=1 frames,
and selected task/subtask terminal image/TCP supervision and text conditioning
are retained. Visual-only v1 and language-conditioned v2 keep their old behavior.

## Causal semantics

The 32 history queries now have chronological cutoffs over the ten frame slots:
`floor(((query_index + 1) * 10 - 1) / 32)`. Each query reads only valid frames
at/before its cutoff. All frames retain their real-age PE. The prefix before
the first available frame outputs zero and is masked as an attention key.
Missing intermediate frames leave earlier valid memory available to later
prefixes; the last query always includes the latest available frame.

History query rows in the fusion adapter read only history keys whose cutoff
is no later than their own. They cannot read current image tokens. Current
tokens read valid history and the same-time current spatial tokens. The fixed
instruction is available through native text cross-attention. Neither future
target RGB nor TCP enters this path. Empty-prefix dummy keys prevent NaNs but
do not expose those prefix rows to current tokens.

The total history budget stays 32, rather than multiplying 32 by ten. Changing
the attention masks adds no parameters. Language conditioning adds 16 parameter
tensors relative to the original visual-only model.

## Formats and presets

Causal bundles use `openwam.visual_history.v3`; Stage2 requires the matching
`goal.causal_history: true`. Bundle loading, preparation, Stage1→Stage2 contracts
and Stage2 resume reject inconsistent modes. Four presets inherit the validated
language/cache recipes:

- `stage1_causal_text_visual_history_relative112_production.yaml`
- `stage1_causal_text_visual_history_relative112_gpu_zero1.yaml`
- `stage2_causal_text_visual_history_joint_gpu.yaml`
- `stage2_causal_text_visual_history_none_gpu.yaml`

History frame dropout remains the inherited configured value, 0. This change
does not enable random frame dropout or change the dataset population/sampler.

## Architecture continuation

`openwam/train/visual_history_migration.py` is deliberately separate from subset
migration. It requires an immutable source manifest and matching dependencies,
reconstructs named optimizer group orders from both real topologies, and verifies
every FP32 master rounds to its named saved model tensor. This catches repeated
transformer parameters with identical shapes that would otherwise be silently
misassigned by positional loading.

All existing model parameters, including history pool/gate, and their FP32
masters/moments/steps are retained exactly. The 16 added text projection,
cross-attention and norm tensors copy the trained main DiT's corresponding
weights and FP32 masters, with independent storage, zero moments and step zero.
The optimizer schema remains `openwam.zero1_fp32.v1` but its indices are rebuilt
for the new 875-parameter ordering (source: 859). This is an explicitly recorded
new architecture continuation, not an unchanged resume.

Finite population, source/index identity, world size, seed, sampling/loading,
cursor and update must remain unchanged. The helper supports the tested finite
non-convergence ZeRO-1 continuation; it does not silently reset an unsupported
optimizer or a bounded/convergence run. Publish to a separate output using
`save_epoch_checkpoint`, regenerate runtime preparation with the standard entry
point, and verify the serialized named state before stopping any running child.

## Verification

New causal tests failed before implementation because the options did not exist.
CPU tests cover later-frame/current-image perturbations, all-empty and sparse
prefixes, language gradients, checkpointing, v3 strict bundles/runtime, presets,
asset relocation and exact interrupted resume. Independent review also checked
history budgets 1/3/5/10/20/32, sparse masks, latest-frame dependency, finite
gradients and zero masked-input gradients, finding no causal-mask defect.
The final integrated focused suite passed **116 tests** in 22.51 s. This includes
the existing language/history/cache/Stage2/pipeline suites and migration/optimizer
regressions. `git diff --check` and final independent review passed.

Migration regressions cover v1/v2→v3, reordered named groups/parameters, scalar
gate state, wrong same-shaped mappings, source tampering, independent donor
storage, actual v3 safetensors export and a real singleton-Gloo ZeRO restore and
next update matching FP32 AdamW. The focused migration/optimizer suite passed
19 tests before integration.
Review additionally reproduced acceptance of nonfinite moments, negative second
moments and fractional Adam steps; four failing regressions preceded the fix.
Migration now rejects those numerical states before publication, and all 17
migration tests passed in the reviewer's final run.

Native DSW check: H20-3e, BF16, full 30-layer foundation DiT, real VAE/T5, one
actual LightWheel example with ten history frames and two valid terminal TCP
points. C* was `[1,120,3072]`; image/TCP losses were 0.61616880/1.10231507 and both
produced finite nonzero text-projection gradients. Earlier adapted history was
bitwise invariant to later-history and current-image changes. Removing the
target decoder preserved C* exactly. One forward/total-loss backward took
0.7028 s with 20.9939 GiB peak tensor allocation. No optimizer update was made.
This is a wiring check, not a learned-quality or distributed-throughput claim.

Run artifacts, final test log, timing/progress snapshots and switch preparation
are stored separately at:
`/mnt/data/cpfs/b5/yeqingda/openwam-0911/OpenWAM/outputs/causal_cstar_h100_switch_20261109/`.

## Live experiment boundary

H100-labelled job `dlc17vidcbxadyj2` was observed running the old `10dc74e` code
at update 5575, cursor 32,836,866 / 48,547,815, recent median 50.75 s/update. Its
latest durable checkpoint was update 5000, cursor 29,304,066. The four nodes
reported `NVIDIA L20Z` despite the H100 experiment name. Existing code has no
immediate checkpoint signal handler; an immediate switch must explicitly report
replaying the updates since the last durable save. Waiting for update 6000
preserves more work but takes several hours at the observed rate.

The user was asked to choose this timing tradeoff while implementation and
preparation continued. Switching is dependent on that answer. H200 job
`dlc18fhk3v9fifql` was already StoppedByUser before this work and is not changed.
The operation artifact directory and final conversation record, rather than this
preparation-time report, are authoritative for subsequent launch status.
