# C* auxiliary video and exact action KV cache

**Goal:** Preserve the frozen visual-history Stage1 encoder and relative112 semantics while training video as an auxiliary task and caching fixed visual/C* conditions during action-only inference.
**Baseline:** dc1c586. Reference: /mnt/data/cpfs/b5/yeqingda/openwam/model/architectures/dual_system/{joint_self_attn,mot_driver}.py. Do not import LAV or change Stage1.
**Environment:** DSW /root/miniconda3/envs/openwam/bin/python, CPU and available H20 CUDA. Tiny native Wan/ActionDiT only; no DLC mutation or full training.

- [x] Add failing native-block tests: isolated action independent of future video; C* and first frame remain influential; auxiliary video loss produces gradients, encoder frozen; config validation.
- [x] Add opt-in `goal/stage2_visual_history_aux_gpu`: joint video targets + isolated cross-modal mask + first_frame_causal; keep joint/none defaults. Update driver/runtime/Stage2 validation. Permit only explicit mutual-to-isolated warm-start; never relax resume architecture checks.
- [x] Add failing cache tests: CPU/CUDA multiple steps, partial/all-invalid C*, padded text, state/condition refresh; same seeded generate outputs and inactive action dims. Measure visual layer and adapter call counts.
- [x] Implement request-scoped cache with per-layer first-frame K/V and C* K/V. Keys remain ordered [video,action,goal]. First-frame prefill never reads C*. Preserve text/proprio context and masks. No reuse across generate calls, no training cache, no approximate velocity cache.
- [x] Add `use_action_kv_cache` generate override and `model.architecture.action_kv_cache` default false; auxiliary preset opts in. Cache eligibility: Wan22 TI2V, first_frame_causal, isolated+joint or mutual/isolated+none, cfg_scale=1, no requested future video output. Unsupported requests use existing joint path. Public cached generate must still compute C* via frozen encoder, never accept caller-supplied goal tensors.
- [x] Run CPU regressions and GPU FP32/BF16 native-block checks; record numerical errors, finite losses/gradients, peak allocated memory and invocation counts. No throughput/quality claim from a tiny model.
- [x] Review changes, document use, limitations and evidence; commit isolated branch and deliver paths.
