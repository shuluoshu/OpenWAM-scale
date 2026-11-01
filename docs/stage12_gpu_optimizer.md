# GPU optimizer training revision (2026-11-01)

This is a new-run configuration for four nodes with eight GPUs each. It keeps
the relative 112-dimensional, 32-step action contract, the previously computed
statistics, OpenWAM foundation initialization, missing-view/label masks, and
the Stage1 convergence gate before freezing Stage1 and starting Stage2.

The running reservations use a separate pinned checkout. Publishing these
files does not change their configuration or restart them. Start this recipe
in a new output directory; old full/CPU-optimizer checkpoints are not a
transparent resume into the new optimizer or batch identity.

## Revised parameters

| Setting | Previous 80 GiB / 140 GiB recipes | New recipes, both hardware classes |
|---|---|---|
| Stage1 optimizer | Full FP32 AdamW per rank; CPU / GPU | GPU FP32 master AdamW, ZeRO-1 owner partition |
| Stage1 gradient synchronization | DDP | DDP, unchanged |
| Stage1 Adam betas | `(0.9, 0.999)` | `(0.9, 0.95)` |
| Stage1 local / global batch | 128 / 4096; 64 / 2048 | 32 / 1024 |
| Stage1 CPU readers per rank | 8 | 8 |
| Stage1 prefetch per reader | 16 / 8 individual samples | 4 individual samples |
| Stage1 pinned CPU inputs | Disabled | Enabled, including label tensors |
| Stage1 training intermediates | Write/read PNG and compressed NPZ | Direct in-memory RGB/label conversion |
| Stage2 optimizer | DeepSpeed ZeRO-2 with CPU offload | DeepSpeed ZeRO-2 on GPU, single-tensor AdamW, no optimizer offload |
| Stage2 local / global batch | joint 8 / 256; none 16 / 512 | joint and none 8 / 256 |
| Stage2 CPU readers per rank | 4 | 4 |
| Stage2 prefetch per reader | 1 batch | 1 batch |
| Stage2 Adam betas | `(0.9, 0.95)` | `(0.9, 0.95)` |
| Stage2 none video reads | Current + eight future times, plus history | Current + history only |
| Gradient accumulation | 1 | 1 |
| DLC priority / startup | 9 / hold | 9 / hold (`sleep 36000000`) |

Stage1 epsilon is `1e-8`, weight decay `0.01`. The GPU owner maintains FP32
master weights and Adam moments; BF16 model weights are broadcast after each
update. The CPU scalar Adam step counter is not CPU optimizer offload.
Stage2 uses native DeepSpeed with `training.adam_foreach: false` and existing
gradient checkpointing. This explicitly selects `foreach=False, fused=False`
in AdamW. Omitting the option preserves the legacy backend. Masters and moments
remain on GPU; a CPU scalar step counter does not constitute optimizer offload.
The Stage1 ZeRO-1 checkpoint format is `openwam.zero1_fp32.v1`; consolidation
is collective, and reload preserves FP32 state, including scalar trainable gates.

Learning rates remain unchanged: Stage1 base `1e-5`, history `1e-4`; Stage2
video `1e-5`, action `1e-4`, projection `1e-3`, goal adapter `1e-4`. No new LR
schedule or warmup is introduced. This separates the optimizer/input change
from a learning-rate experiment.

The reader counts were retained, not increased blindly: an eight-GPU node has
64 Stage1 readers or 32 Stage2 readers, plus training processes. Stage1 has at
most `8 * 4 = 32` outstanding singleton samples per rank, in addition to the
current batch. Stage2 has four prefetched batches per rank. The new none
reader preserves the nine-slot sample format with invalid future placeholders;
the actual 32-step action supervision, FK targets and history remain intact.
Evaluation/export still writes persistent Stage1 RGB/label assets.

## Data and training budget

Use the approved YAML with only the missing wipe-table directory additionally
excluded, and the existing relative112 statistics registry. Both pipeline
files contain their absolute paths. No new statistics are estimated at launch.
Stage1 retains its random order over the previously configured 3 Hz anchor
population. Stage2 retains random anchors in selected native rows and contiguous
source-row action windows; the YAML's sampling hint does not rescale its action
indices. The action chunk retains each source's actual time alignment.

Stage1 trains at least 50,000 updates, evaluates every 10,000, and requires
the existing image/TCP convergence gate (five plateau evaluations with 1%
relative tolerance). The ceiling is 154,000 updates. Exhausting the ceiling
without accepted convergence refuses the Stage2 handoff. Stage2 remains
154,000 updates with Stage1 frozen.

Equal global batch across the two GPU classes makes sample exposure comparable.
It also changes exposure versus the old larger batches: 50,000 Stage1 updates
now process about 51.2 million samples, close to the 51,514,188-pair population,
instead of approximately two/four passes with the old batches. This is a
training-budget change even though the update count and LR are unchanged.

## Files and launch boundary

- `configs/goal/stage1_visual_history_relative112_gpu_zero1.yaml`
- `configs/goal/stage2_visual_history_joint_gpu.yaml`
- `configs/goal/stage2_visual_history_none_gpu.yaml`
- `configs/goal/stage12_gpu_zero1_80g_joint_4x8gpu.yaml`
- `configs/goal/stage12_gpu_zero1_140g_none_4x8gpu.yaml`
- `configs/pai/stage12_gpu_zero1_80g.site.json`
- `configs/pai/stage12_gpu_zero1_140g.site.json`

The site templates target the existing H100/H200 resource pools; the installed
GPU name and memory must still be checked at explicit launch. Templates require
75/130 GiB minimum memory respectively. They point to a new
`outputs/stage12_gpu_optimizer_20261101/code` checkout and separate run outputs.
Materialize the pushed commit there and create fresh offline preparation proofs
before launching training. `commit: auto` is resolved by the request renderer;
it is not permission to run mutable code. See `docs/dlc_reserved_training.md`
for reservation, explicit per-pod launch and runtime verification.

## Verification and limits

The distributed optimizer regression compares two Gloo ranks against full
FP32 AdamW across updates, missing gradients, multiple LR groups and checkpoint
resume. The actual trainer test covers partial-population tails and collective
checkpoint/handoff. Input tests cover exact FP32/BF16 conversion parity, missing
head/wrist supervision, recursive target pinning and spawned CPU readers.
Stage2 tests compare decoder calls and seeded none-mode loss/action masks.

Final combined targeted run: 217 passed, with the CUDA scalar regression run
separately and one existing Genrobot fixture excluded (its missing TCP metadata
fails identically against HEAD). The standalone optimizer run, including CUDA
scalar-state reload, passed all three tests. Independent read-only review found
no actionable correctness issue. These are targeted checks, not a claim that
the entire repository test suite passes.

The first native Stage2 single-GPU trial with default foreach AdamW completed
forward/backward but exhausted the 132 GiB allocator cap at a further 18.63 GiB
optimizer temporary. A fused diagnostic removed that allocation but produced
NaN action loss on update two. An isolated padded-buffer test confirmed the
installed Torch 2.7.1 fused kernel writes 15,168 elements past a tensor of
`2**31 + 50368` elements. The identical single-tensor AdamW control made no
out-of-bounds writes. This matters because the single-GPU video/proprio ZeRO
group has 5,000,250,560 elements. The final recipe explicitly disables both
fused and foreach, retains GPU optimizer placement, and awaits validation of
the 32-rank native training path on DLC. Failed diagnostic results are retained;
they are not reported as passing capacity checks. The same canary also exposed
the foreach multi-tensor path, so both accelerated backends remain disabled.

The GPU backend regression also checks FP32 updates after optimizer save/reload.
The final Stage2 contract/trainer suite passed 88 tests with the selected
single-tensor backend; GPU backend/profile checks passed separately. Test counts
are not additive because the suites overlap.

On one H20-3e DSW GPU, the full Stage1 model completed three batch-32 updates
with GPU FP32 AdamW. The last two optimizer updates took 0.540 and 0.537 seconds;
full model steps took 8.28 and 8.33 seconds with predecoded real mixed-source
inputs. Peak reserved memory was 94.23 GiB because world size one stores every
optimizer shard. This is neither an 80 GiB fit result nor a 32-GPU throughput
measurement. Two-rank correctness tests establish partition behavior; actual
32-GPU NCCL, checkpoint peaks and filesystem contention still require a cluster
run. Batch sizes are starting settings, not a claim of maximum throughput.

The remaining exact-ordinal sequential video decoder, notably Mecka VFR inputs,
is unchanged. Slow decoding can still dominate loader waits after optimizer
offload is removed; an indexed seek implementation needs separate frame-level
alignment validation.

## Publication status

At the user's explicit request, DSW capacity/training experiments were stopped
before publication. No further single-GPU optimizer adaptation is included.
The committed Stage2 recipe targets 32-rank ZeRO-2, not the unpartitioned
single-GPU optimizer-state footprint. Multi-step stability, actual per-rank
memory peaks and throughput must be validated on DLC next. The attempted
full-model single-GPU Stage2 capacity tests did not establish a passing result.
Existing running jobs and their pinned deployments were not changed.
