# Observation augmentation experiment restart

User authorized restarting the experiment and comparing losses with the previous run. Execute inline in the current workspace; retain all historical runs and event files. No model/training behavior changes beyond the already implemented augmentation.

- [x] Confirm baseline gripper_human_lr0.0001_b320 completed 1000 steps and TensorBoards 18082/18083 still serve historical data.
- [x] Confirm effective video/action/projection LRs are 1e-5/1e-4/1e-3; same foundation checkpoint, batch 320, 1000 steps, 25-step evaluation, seed 42 and filtered 46-clip fixture.
- [x] Snapshot augmentation configuration, current source and historical event hashes. Create a distinct run gripper_human_aug_lr0.0001_b320 and add it to both existing TensorBoard directories.
- [x] Pause only this repository's identified burn_gpu.py process, launch the actual trainer in /root/miniconda3/envs/openwam, and resume the burn process when training ends. The baseline is complete, so no training process needs stopping.
- [x] Verify initial clean fixed-noise evaluation matches the baseline at step 0, actual optimizer steps log finite losses at batch 320, saved config enables all four operations, and both TensorBoards show new/old scalars together.
- [x] Save launch/status/preservation verification and a continuously refreshed comparison.json with matched-step evaluations and baseline final metrics.

The main comparison is clean eval_fixed_noise at the same optimizer step. Training RGB and future video targets are augmented, so raw training loss is not the same measurement as the baseline. Do not infer convergence improvements from initial steps or compare new step 1 directly with baseline step 1000. Batch 320 must be confirmed in the actual run; do not silently change the comparison's effective batch size if OOM occurs.

Artifacts: outputs/curated_observation_aug_20261015/{launch,supervise,verify_restart}.py; actual run under outputs/curated_gripper_human_20261014/gripper_human_aug_lr0.0001_b320; comparison TensorBoard 18083 and historical comparison TensorBoard 18082. The new run starts from the same foundation initialization, not from the old step-1000 checkpoint.

## Launch progress

- First attempt exited before model construction because Hydra struct mode rejected adding a new dataloader observation_augmentation key. Fixed the real entrypoint using OmegaConf.update(..., force_add=True). A CPU test executed actual main() with only model/trainer creation replaced, reproduced the failure and then passed after the one-line fix. Evidence: outputs/curated_observation_aug_20261015/cli_config_validation.json. Failed attempt was moved to failed_config_attempt; no training scalar was written and historical files were untouched.
- Second launch: supervisor 2144899, torchrun 2144918. Same branch's real reader/trainer code and augmentation config snapshotted. Four probabilities: 0.8 color, 0.5 patch, 0.1 view drop, 0.1 whole-state drop.
- Initial fixed-noise evaluation is exactly equal for every metric to the baseline at step 0 (including loss 1.2209681538598878, clean_joint_mse 1.2002480212185118, clean_gripper_mse 0.1419402887199381). Complete saved model/training/data configuration comparison passes after excluding augmentation and output directories. Gripper/joint label contract and effective LRs match.
- 44 augmentation/reader CPU tests pass after the CLI fix. Both existing TensorBoard servers are reused; no old events are overwritten. Main two-run comparison: https://897628-proxy-18083.dsw-gateway-cn-shanghai.data.aliyuncs.com/#scalars ; broader history: port 18082.

## Verified handoff

verify_restart.py passed against running processes and both TensorBoard HTTP APIs. Confirmed through optimizer step 5 with batch 320; latest train total/action/video = 0.858170/0.654186/0.191249. Peak allocated GPU memory 108.18 GiB. All 11 historical event files retain their exact size and SHA-256, the new and baseline runs appear on both 18083/18082, and source snapshots match running launch inputs. Evidence: restart_validation.json. Comparison.json updates every 10 seconds with matched clean evaluations. Only initial evaluation (step 0) is available at handoff, so augmentation improvement has not yet been established.

Launch: /root/miniconda3/envs/openwam/bin/python outputs/curated_observation_aug_20261015/launch.py

Verification: /root/miniconda3/envs/openwam/bin/python outputs/curated_observation_aug_20261015/verify_restart.py
