# Curated OpenWAM training execution plan

Goal: Train current OpenWAM 32-step / 112D on exactly the 46 curated 150-frame segments, maximize stable GPU microbatch, publish TensorBoard and preserve comparable fixed-noise losses.

Environment: /root/miniconda3/envs/openwam; one H20 ~140 GiB. Initialize from OpenWAM Alpha foundation checkpoint, not prior single-sample fitted weights. Preserve video/action/FK objectives and modality masks.

- [x] Build bounded training fixtures from audited segments. Cache actual deterministic decoded/resized RGB canvases losslessly; verify against prior reader outputs. Keep every valid anchor (149/clip). Add verified Genie-1 URDF-base TCP GT for FK; do not silently train the known mismatched ARX FK or guess other chains.
- [x] Build reproducible source-balanced loader, preserve source video offsets and native GT transforms. CPU-check samples, masks, next-state indexing and FK GT floor.
- [x] Pause only the identified burn_gpu.py process while training, automatically resume it on completion/failure. Launch detached GPU job and TensorBoard.
- [x] Probe real forward/backward microbatch sizes with fixed checkpoint and no weight updates, reserve memory for ZeRO/optimizer setup. Confirm actual training steps; fall back on OOM with logged reasons.
- [ ] Compare two learning-rate settings from the same checkpoint/seed and fixed evaluation samples/noise. Keep loss weights fixed; do not rank by stochastic total loss alone. Continue the lower fixed-noise clean-action error configuration for 1000 steps, track joint/TCP/head and valid FK metrics separately.
- [x] Verify live event files, finite optimization metrics and HTTP TensorBoard endpoint. Report actual batch, checkpoint, loss scope and pilot status without claiming convergence early.

Execution evidence (2026-10-12; output directory retains its initial 20261011 name):

- Experiment: `outputs/curated_train_20261011`; launch/coordinate/run/data scripts and per-run configs are saved there. Python is `/root/miniconda3/envs/openwam/bin/python`.
- 46 clips / 12 source families / 6854 window anchors; all deterministic image checks passed. Statistics regenerated for copied dataset provenance, with exact equality to prior action/state statistics.
- Two verified Genie1 clips' GT-through-FK floors: position 3.83e-15 and 4.09e-15 m²; rotation 2.73e-14 and 2.07e-14; 55 and 23 valid arm/time endpoints at the fixed evaluation anchors.
- Foundation checkpoint step154000; 6.021B trainable parameters. New encoder input columns 80:112 initialized with seed20261009; old80 columns unchanged. Video LR1e-5, action LR5e-5 or1e-4, action input/output projection LR10x action LR. All four loss weights remain1.
- Initial batch64 pilots completed30 optimizer steps each. Fixed training-window/noise clean action MSE0.364289 vs0.243136; joint0.752804 vs0.453446; FK position0.075362 vs0.032272 m². LR1e-4 performed better in this bounded comparison; not a held-out generalization result.
- Batch64 raw forward/backward peak47.11GiB leaves ample room. Initial64 evidence preserved as `initial64_*.json`; larger-batch search and matched LR pilots follow. The initial64 main run was intentionally stopped for this extension; keep its TensorBoard trace as historical, not the active run.
- TensorBoard port18082, event files confirmed readable and scalar HTTP plugin enabled. URL: https://897628-proxy-18082.dsw-gateway-cn-shanghai.data.aliyuncs.com/#scalars . Main run follows the two larger-batch pilots automatically for1000steps.
- Production reader/model/loss code is unchanged by these experiment drivers. Frozen-input CPU caching is bounded at16GiB for larger-batch runs; only deterministic frozen VAE/text outputs and GT are cached, not trainable model outputs or noise.
- Extended probe: B128=60.37GiB, B256=91.79GiB, B320=108.84GiB, B384=125.90GiB. SelectB320 with22% probe headroom; actual ZeRO2 optimizer training confirmed at least5 finite updates, peak108.18GiB allocated /110.96GiB reserved, gradient accumulation1. Effective optimizer groups confirmed [1e-5,5e-5,5e-4] for first pilot. Larger-batch pilots/main remain running under the detached coordinator.
- Public gateway returns302 to Alibaba DSW login; local TensorBoard serves HTTP200. `delivery_verification.json` records latest process/event/scalar verification. Login with access to this DSW instance is required.
