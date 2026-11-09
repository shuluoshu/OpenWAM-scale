# History VAE cache and bounded batch encoding

The Stage1 history path previously evicted frozen frame features whenever a shuffled sample changed episode or a partial history appeared. RGB was transferred to CUDA before being copied back for hashing, and cache misses used per-frame VAE calls. This change keeps Stage1 history RGB on CPU, retains bounded episode/content entries, caches codec outputs per image, and encodes misses as independent T=1 frames in microbatches of four.

Branch: `codex/history-vae-cache-batching-20261109`, based on `10dc74e`. The active DLC checkout, source configuration, sampler, optimizer and jobs were not changed. This is a bounded memory-cache optimization; no full-dataset latent precomputation was performed.

## Behavior

- Raw VAE features are detached CPU tensors. Episode/content keys isolate sources while allowing previously visited episodes to remain in the LRU. Masks and relative age do not define a frame's encoded content.
- Invalid history slots are not encoded and remain zero. Known episode switches and random/partial masks do not flush other entries. Unknown episode IDs keep the conservative no-persistence behavior of the outer cache.
- The generic codec caches individual images, enabling overlap between batches. It invalidates on normal parameter/buffer version, dtype or device changes. The current Wan normalization attributes are fixed; callers that change ordinary `mean/std/scale` attributes must explicitly clear caches.
- Native `batch_encode` keeps temporal length one; a legacy `encode` fallback remains. The default upper bound is four images per VAE call.
- Conditional goal predictions retain their separate episode/mask reset policy. Learned history pooling, text handling and C* are never cached as frozen features.
- HistoryTokenPool, including its time PE, is AST-identical to the baseline. C* remains 120x3072 and history/dropout settings are unchanged.

## DSW validation

Accepted measurement: paired single-process comparison on NVIDIA H20-3e, BF16, eight real shuffled examples from distinct episodes with 80 valid history frames. Inputs cover ABC, LightWheel, Morphi and Interndata. A real 4000-step 30-layer Stage1 bundle and VAE weights are shared by object identity. Text is a fixed synthetic 512-token context; no language-quality claim is made. Both variants stay resident, order alternates each pair, one warmup pair is excluded and five pairs are measured with CUDA synchronization.

| Timed path | Baseline median | Candidate median | Ratio of medians | Median paired ratio |
|---|---:|---:|---:|---:|
| Cold history | 1.081441 s | 0.768109 s | 1.4079x | 1.4004x |
| Warm history | 0.259725 s | 0.034688 s | 7.4874x | 7.4007x |
| Full 30-layer Stage1 forward/backward, cold | 2.025796 s | 1.700836 s | 1.1911x | 1.1911x |

The measured forward/backward rate is **3.9491 -> 4.7036 samples/s (+19.1%)**, with 16.0% lower latency. Peak tensor allocation is 21.037 -> 21.043 GiB, approximately unchanged (+6 MiB). Cold-history peak increases about 274 MiB because the VAE processes four images together, while the full forward/backward peak remains dominated by the model.

Warm measurements revisit the same eight examples and are not a predicted hit rate for the 48.5M shuffled population. The timed full-model interval excludes optimizer, T5 inference, DataLoader and distributed communication. It is not an end-to-end DLC throughput measurement. Shared-GPU activity was present during the task; paired alternation and common resident memory reduce, but do not eliminate, ambient noise.

| Numerical check | Result |
|---|---:|
| History VAE relative L2 | 0.497512% |
| History maximum / mean absolute error | 0.0625 / 0.00133335 |
| C* relative L2 | 0.404202% |
| C* maximum / mean absolute error | 0.5 / 0.00077229 |
| Stage1 loss | 0.05472602 -> 0.05474304 |

Both relative-L2 checks pass the predeclared 1% BF16 tolerance. Batched convolution is not bitwise equivalent to singleton BF16 convolution; downstream task quality was not measured.

## Evidence and reproduction

Artifacts: `/mnt/data/cpfs/b5/yeqingda/openwam-0911/OpenWAM/outputs/history_vae_cache_benchmark_20261109/`.

- `fixture.safetensors` / `fixture.json`: exact normalized input bits, masks, ages, source records. SHA256 `a293c75583ce82d60835f092cb037722b0ed7bfb1ab2ac06e0bfdf098ed41d26`.
- `paired.json`: accepted measurements, all samples, hashes, shared-parameter/native-backend assertions and numerical errors.
- Bundle SHA256: `53ddccdfa32c194bde470f9f1a4ae800454d66ecf06acf44b2f7ae48cbddccd9`.
- Implementation diff SHA256 against `10dc74e` for `openwam/`: `3f75e12fd76137456335e4fe3258d0588f1fbb4b81978cf2a74cc0cbba3ca7c2`.

The benchmark recorded a documentation-only HEAD plus the working-tree implementation hash. The final committed implementation is verified against that same diff hash in `delivery_verification.json`.

Run the paired driver with the actual baseline checkout, Stage1 config, saved bundle and fixture:

```bash
/root/miniconda3/envs/openwam/bin/python \
  scripts/experiments/benchmark_history_vae_paired.py \
  --baseline-repo /mnt/data/cpfs/b5/yeqingda/CodeX/worktrees/openwam-dlc-recovery-20261109 \
  --config /mnt/data/cpfs/b5/yeqingda/openwam-0911/OpenWAM/outputs/stage12_subset_continuation_20261109/h100_joint_stage1.yaml \
  --fixture /mnt/data/cpfs/b5/yeqingda/openwam-0911/OpenWAM/outputs/history_vae_cache_benchmark_20261109/fixture.safetensors \
  --bundle /mnt/data/cpfs/b5/yeqingda/openwam-0911/OpenWAM/outputs/stage12_subset_continuation_20261109/h100_joint/stage1/checkpoints/update_000004000_migrated_e7cfde19/visual_history.safetensors \
  --output /path/to/new-paired-result.json \
  --count 8 --encode-batch-size 4 --warmup 1 --repeats 5
```

`benchmark_history_vae.py --prepare` builds the real fixture from the immutable finite index and actual reader. Separate-process modes are also available for non-shared backends. Earlier `baseline.json`, `candidate_b4.json` and `baseline_idle.json` saw differing ambient GPU load; their ratios are diagnostic only. An initial paired run failed because a historical absolute GoalCodec import escaped its package alias. Explicit codec binding, type assertions and `test_history_benchmark_isolation.py` now cover that failure. Failed runs produced no accepted result JSON.

## Checks

- 93 focused cache/history/pooling/runtime/loader tests passed.
- 55 additional patch, query, epoch and Stage2 tests passed.
- The baseline-import isolation regression passed; total **149 distinct CPU tests**.
- Independent production-code review ran 48 overlapping tests, then nine final cache/isolation tests; no outstanding actionable findings.
- `git diff --check` and Python compilation passed.
- The reusable workflow is installed at `/mnt/data/cpfs/b5/yeqingda/CodeX/skills/openwam-history-vae-performance/`; its schema validation and independent read-only forward test passed.
