# RFC goal-memory 1Hz implementation plan

> Execute with subagent-driven-development. Spec: docs/superpowers/specs/2026-11-11-rfc-goal-memory-1hz.md.

Goal: runnable Stage1 / Stage2 / offline memory / inference implementation, separate from existing Cstar. Python/PyTorch nativeWan + ActionDiT, existing source readers and optimizer helpers.

## Task1 — shared VideoDiT model and cache
Create goal_memory.py, goal_memory_masks.py, goal_memory_cache.py and tests. Implement native prepare/pre/post/run/outputs, masked text routing, clean timestep groups, G queries, keyframe velocity/TCP head, horizon embedding, FM+UV losses/warmup/augment. Implement pre-RoPE frame KV and window cache. No model copied30layer decoder. Test I1 fixedGKV/text/history intervention; I2 noisyY/t changes; I6 frame alone/window/cached; finiteempty/padding, gradients, native-equivalence mode. Report all public signatures for Stage2.

## Task2 — 1Hz multiview persistent cache and proportional dataset
Create dataloader/rfc_goal.py, model/goal/memory_codec.py, scripts/data/precompute_goal_memory.py, tests. Use source-native curated metadata/decoding/projection. Add opt-in required materialization support to existing curated_goal.py only as necessary, preserving old fields and outputs. Select episode sink and recent1Hz frames strictly beforeanchor; no duplicateframes; perview masks; no futureselection. Mixed finite samples must use configured ratio, not merely store unused option. Full multiview terminal image and UV from same selectedinstructionend. Persistent atomicVAE cache with completeprovenance; precomputeCLI and coverage audit. Tests actualreader fixtures, cachecold/warm/sourceVAEmutation, ratio determinism/resume and masks. Own these files exclusively.

## Task3 — Stage2 native shared-policy and inference
Create architectures/dual_system/goal_memory.py (+ registration config as needed), tests. GoalMemoryPolicy uses same coreVideo expert, nativeActionDiT pre/post, projectionproprio matchingbaseline. Stage1 weight initialization, isolated/default and action_sees_video explicit masks, optional futurevideoFM/KFaux0.1, freeze and auxiliaryzero ablations, Gdropout. Perchunk cleanprefill then actiononlydenoise, memory cacheoff/on matching. New Stage2training/inference CLI/pipeline integration runnable with actualbaseline and same data contract. Own architecture/newStage2entry files; coordinate coreAPIs, avoid editing parent-owned stage1/config/bundle files. Test deletingY/V invariance underisolated, video neverreadsaction, action usesG/history, cache equivalence gradients and baselineload contracts.

## Task4 — controller integration / packaging / validation
Add train/goal_memory.py, scripts/train_goal_memory.py, bundle/checkpoint loader, Stage1/Stage2 config presets and tests; wire actualcomponents, data, sampler, collate, optimizer, resume/checkpoint, offlinecache CLI, Stage1→Stage2 handoff. Keep old Cstar paths unchanged. Baselinefocusedtests before edits, taskTDD, integrationruns. Independent task and wholebranch reviews; fix findings. Realnative GPU smoke ifavailable and modest. Write ablationmatrix and invocationdocs. Commit new branch; do not touch liveDLC. Record unverifiablefulltraining limits honestly.
