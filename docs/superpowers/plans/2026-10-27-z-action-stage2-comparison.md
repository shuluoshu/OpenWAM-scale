# Z-conditioned action-only Stage2 and matched overfit comparison

User approved the architecture in the preceding conversation and explicitly requested implementation and two approximately five-hour runs. Execute without further design approval.

Goal: optional Stage2 route uses current observed video features + frozen predicted query Z + language/state to denoise actions, with no future-video tokens in training or inference. Preserve existing mutual baseline.

Architecture: keep backbone/checkpoint shapes compatible. Add an explicit future-video mode and dispatch action-only preparation/loss/generation. Retain current observation feature extraction; do not masquerade future-noise-only or masked future tokens as removing them. No future image VAE encoding or future-video DiT output/loss in the new route. Current features can be computed by the video backbone. Stage1 remains frozen and identical across arms.

Global constraints:
- Base commit 7dd1e67, worktree /root/.codex-chatgpt/worktrees/stage1-epoch-shuffle/OpenWAM. Preserve two existing unrelated report edits.
- All experiment inputs/cache/outputs/checkpoints on CPFS; real original model sizes and baseline initialization.
- Cover every dataset family in smoke YAML (expected 14; audit authoritative YAML). At least one valid episode each. Include corresponding subtask instructions where available and task examples otherwise. Retain explicit invalid target masks.
- Same fixed examples, seed, action horizon32 (native reader), normalization, optimizer settings and frozen Stage1 asset for both arms. Repeat shuffled balanced examples. Report both equal-time and matched-update results.
- Two sequential runs, each ~18000 seconds training/evaluation budget, plus preparation. Evaluate at start, every 500 updates and final; preserve all versions, never overwrite past predictions.
- Report real action metrics and trajectories. New architecture has no predicted video: show absent/not applicable, never fabricate a goal image or substitute Stage1 output as Stage2 video.
- Monitor wall time, step time, peak VRAM, losses, per-dataset normalized and valid physical-unit action errors. HTML contains status, config/source identities and conclusions scoped to overfit data.

Tasks:
1. Core architecture and tests: prove future frames cannot influence action-only output; no future slots/loss; production training/inference configuration and checkpoint compatibility. Keep baseline tests passing.
2. Dataset audit/preparation: deterministic manifest covering all families, actual decoded windows and labels, shared normalization and target validity; select current frames with legal within-episode history.
3. Real-model runner: cached valid inputs, full-dimensional model, optimizer ownership, time-limited sequential arms, robust final save and status, frozen Stage1 checks, append-only evaluation and HTML.
4. CPU tests + actual GPU forward/backward smoke both arms; inspect GT overlays before launching; measure throughput to choose sample count (start with 1 episode / family and 1–2 fixed windows / instruction level).
5. Launch durable runner, verify first losses and GPU memory, record PID/config and queue second arm automatically. Monitor completion and publish final HTML conclusions; no existing unrelated training is stopped.

Verification commands to adapt after file ownership is assigned:
- /root/miniconda3/envs/openwam/bin/python -m pytest tests/goal/test_z_action_stage2.py -q
- Run matched runner --smoke with both modes, then --seconds-per-arm 18000.
