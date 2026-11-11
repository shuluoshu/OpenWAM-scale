# RFC goal-memory WAM, 1Hz stage A

Approved task: implement RFC revision71 from current3c34210 on a new branch. User overrides: no KF2 +3s; no gist / long windows; history remains1s/frame; configurable mixed task/subtask training. Preserve previously requested label pairing (task→task end, subtask→subtask end). User confirmed task:subtask = 3:7 (subtask_probability=0.7). No running DLC/process/config changes.

Architecture: one native Wan VideoDiT30 shared by memory/current/G/keyframe groups. G32×3072 learned queries. M block-diagonal by frame, C self-only, G reads M/C/G and text; Y reads C/G/Y with text only during explicit warmup. Clean groups t0, Y own diffusion timestep plus goal-horizon embedding. TCP small MLP directly from G, masked normalizedUV regression. Stage1 multiview target120tokens. Stage2 native ActionDiT32×112 and shared Stage1-initialized VideoDiT; isolated default, action_sees_video ablation; keyframe auxiliary0.1 when unfrozen, no target/noisy-video leakage into G. Frozen ablation explicit. No independent teacher Cstar/512 adapter in this new representation.

History override: each memory unit is a T=1 three-view 384×320 frame at1Hz, frozenVAE latent48×1×24×20. No stride4 temporal compression or streaming conv state; explicitly record this difference from RFC. Bounded stageA sink+2recent, configurable short window up to4recent; no gist. Offline cache is persistent, keyed by source/episode/frame/view/preprocess/VAE identity, atomic writes; strict dependency checks. Real relative time metadata retained; RoPE window-relative; per-frame normalized pre-RoPE KV cached across chunks, re-rotated on access. Episode changes clear runtime caches.

Loss / augmentation: imageFM, G→UV masked regression; history latent Gaussian mix, whole-history dropout0.15 configurable; G dropout ablation. Warmup Y text probability0.5→0 over first20% configuredupdates, strict I1 after warmup. Context padding and invalid views/memory/target safe against NaN.

Global constraints
- Current branch starts3c342105196f5dcf6d578577171a5d6dfd1a76e5; old paths/configs remain compatible.
- New representation `goal_memory`, bundle `openwam.goal_memory.v1`; never reinterpret visual_history.v3.
- Shared core interfaces: GoalMemoryConfig, GoalMemoryState, GoalMemoryExpert in openwam/model/goal/goal_memory.py.
- Core is latent-in: current_latents[B,48,1,24,20], memory_latents[B,M,48,1,24,20], target_latents[B,48,1,24,20], text[B,L,text_dim]. `prepare_state`, `pre_attn_at_layer`, `post_attn_at_layer`, `run_state`, `outputs` reusable by Stage2.
- pre_attn returns rotatedQ, rotatedK,V,post; optional raw normalized K cache; text-query gating must zero CA after output projection including bias.
- Batches use observed,observed_view_valid,memory_latents,memory_valid,memory_view_valid,memory_ages,memory_frame_ids,target_image,target_uv,target_image_valid,target_view_valid,target_point_valid,target_in_frame,goal_horizon_s,instruction,granularity. Single-record samples may be unbatched; collate explicit.
- Cross-attention only G (and warmup Y) for Stage1; memory/current cannot acquire text through CA output bias. Future/target perturbations cannot alter G.
- Existing native Wan weights reused; no surrogate transformer in tests. Tiny tests + real native GPU check when feasible; no paid job submission.

Minimum experimental comparisons: currentCstar vs newG; no/history vs real/shuffled history; noG/realG/swappedG; isolated vs action_sees_video sameStage1; frozen vs unfrozenVideo with/withoutKFaux; cacheoff/on equivalence+latency; task/subtask stratified evaluation. Automated invariants separately gate causality, bottleneck and cache consistency.
