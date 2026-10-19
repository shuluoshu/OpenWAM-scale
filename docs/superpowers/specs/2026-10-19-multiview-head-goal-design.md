# Three-view observation, head-only goal design

User approved: Stage1 sees current head/left-wrist/right-wrist views, still generates only the head goal image and ordered 4D head-camera TCP UV. Stage2 uses the same 80 image goal memory tokens. No long training requested this turn.

Current head-only v1 checkpoints and presets remain usable and unchanged. New presets explicitly select multiview and v2 metadata; strict runtime mismatch errors. An explicit warm-start copies the existing v1 Stage1 DiT into the multiview expert; this is initialization, not evidence the new view conditioning is trained.

Observation canvas H384 W320 matches Stage2 layout: head H256 W320, wrists H128 W160 each. All current views use the exact same anchor source frame, native timestamps/offsets, BILINEAR resize. Missing wrist regions are black, carry bool view_valid[B,3] and token-key masks. Sanitize missing pixels before VAE to prevent arbitrary missing-slot values leaking through convolution. No terminal wrist video decoding required. Targets unchanged H256 W320, latent[48,1,16,20], 80 image +1 TCP generated tokens. Observation latent[48,1,24,20], 120 patches. Shared VAE weights, separate observation/target calls; no promise to deduplicate Stage2 augmented-input encoding.

Transformer separately patchifies observation and goal. RoPE: obs time0 (12x10), goal time1 (8x10), TCP time2. Observation queries cannot read generated tokens; generated image/TCP mutually read each other and valid observation keys. Head-only path must retain old numerical behavior. Missing-view mask changes cannot share prediction cache entries. Output head coordinates remain normalized to W320/H256; no canvas-height normalization for TCP.

Stage2 takes clean current multiview canvas before augmentation, explicit current view mask; same generator path in training/public inference/fixed evaluation. Goal labels and future images never enter condition or cache. Adapter and video/action tensor contracts unchanged. Existing single-head input API kept for v1; new multiview explicit args required for v2.

Deliver source/config/tests/docs to same isolated branch shuluoshu/OpenWAM-scale codex/stage1-joint-task-subtask. Run meaningful tiny real-Wan gradient/generation/checkpoint/cache tests, old regressions and bounded native sample preflight with HTML. No stopping existing processes, overwriting old reports, multi-GPU launch or full training. Explicitly report that existing 5k bundle is v1 until a multiview Stage1 run trains and saves v2.
