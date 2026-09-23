# OpenWAM-α canonical EEF 数据适配与 smoke 使用说明

本方案只使用 `pretrain_datasetsV0_1` 的两个真实 bucket 做 smoke，不枚举全量训练集。统一数据链为：LeRobot parquet 中的 absolute EEF20 → 数据源小样本 stats 归一化 → scatter 到 OpenWAM 80D → model forward。

## 数据与动作契约

EEF20 排列是左右臂各 10 维：`xyz(3) + rot6d(6) + gripper(1)`。归一化发生在 20D 空间，随后左臂 `0:10` 写入 OpenWAM `0:10`，右臂 `10:20` 写入 `34:44`，其余 60 维为零且 mask=false。一个样本读取 33 个 raw steps，GT action 为 `(32,80)`，按 stride 4 取 `[0,4,...,32]` 九帧，三相机组成 384×320 画布。

MoraVLA 的实际入口 `pretrain/launch/qwena1_nogen/pretrain.sh` 使用 36D action/state、`action_mode: delta`、`eef_frame: head_camera`，并可能随机选择 wrist frame。这里参考它已物化的 EEF 字段、帧选择和 external stats 思路，不把 36D delta（也不把仓库中其他 96D layout）直接送进 OpenWAM 的 80D head。

## 1. 生成小样本 stats

四条命令都只读取 episode 0/1，输出写在当前仓库，不修改源数据。公共参数 `--bucket` 是一个明确的 LeRobot-v3 bucket，`--episode-index` 可重复，`--output` 是生成的 `.npy`。

```bash
python scripts/data/prepare_openwam_eef_smoke.py \
  --bucket /mnt/data/cpfs/b5big/pretrain_data_release/pretrain_datasetsV0_1/interndata_a1/pick_and_place_tasks/pick_and_place_part2/omniobject3d-lotus_root/genie1__parallel_gripper \
  --output assets/openwam_alpha_data_examples/stats/interndata_genie1.npy \
  --episode-index 0 --episode-index 1

python scripts/data/prepare_openwam_eef_smoke.py \
  --bucket /mnt/data/cpfs/b5big/pretrain_data_release/pretrain_datasetsV0_1/agibotworld/task_512/a2d_gripper__dual_gripper__base__unmodified \
  --output assets/openwam_alpha_data_examples/stats/agibotworld_a2d.npy \
  --episode-index 0 --episode-index 1

python scripts/data/prepare_openwam_eef_smoke.py \
  --bucket /mnt/data/cpfs/b5/post_train_data/robodojo_sim/sweep_blocks \
  --output assets/openwam_alpha_data_examples/stats/robodojo_sweep_blocks.npy \
  --episode-index 0 --episode-index 1

python scripts/data/prepare_openwam_eef_smoke.py \
  --bucket /mnt/data/cpfs/b5/post_train_data/robotwin_traj_fullfield/blocks_ranking_size/demo_clean \
  --output assets/openwam_alpha_data_examples/stats/robotwin_blocks_ranking.npy \
  --episode-index 0 --episode-index 1
```

stats 包含 action `eef`、state `eef_state` 的 count/mean/std/min/max/q01/q99，以及 bucket、字段、episode、schema SHA256。rot6d 维固定为 identity normalization。reader 会逐项核对 provenance；换 bucket、字段或 episode 后必须重新生成。

## 2. 两步 smoke 训练

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
NPROC_PER_NODE=1 torchrun --standalone --nproc_per_node=1 \
  scripts/train.py --config-name openwam_alpha_v0_1_smoke
```

`configs/openwam_alpha_v0_1_smoke.yaml` 的关键字段：

- `defaults.model=dual_system`：构建与 foundation checkpoint 匹配的 80D dual-system。
- `training.finetune_ckpt_path`：从已下载的 OpenWAM-α foundation 权重 warm start，不恢复旧 optimizer step。
- `dataloader.type=mixture`：混合 InternData-A1 Genie-1 与 AgiBotWorld A2D 两个明确 bucket。
- `episode_indices=[0,1]`、`max_episodes=2`、`max_windows=8`：硬限制每个源的 smoke 数据量。
- `num_frames=33`、`video_stride=4`：32-step GT 与九帧视频。
- `normalize_mode=quantile`：使用 q01/q99 映射到 `[-1,1]`；rot6d 不缩放。
- `unify_action=true`、`unify_action_map=["0-9","34-43"]`：20D scatter 到 80D。
- `batch_size=1`、`max_steps=2`、`gradient_accumulation_steps=1`：恰好两个 optimizer step。
- `mixed_precision=bf16`、`use_gradient_checkpointing=true`：单张 H20 的 smoke 内存设置。
- `zero_stage=2`、`offload_optimizer_device=cpu`：将 Adam state/step 放到 CPU，避开 12.4B 模型在单卡 optimizer step 的显存峰值。
- `dataset_num_workers=0`：问题可直接定位到主进程。
- `lambda_video=1`、`lambda_action=1`：两个 loss 都参与。
- `save_steps=0`：profiling/smoke 不写约 25 GB 的最终 checkpoint。
- `project.seed=42`：dataset、noise 和 timestep 可复现。

路径迁移时可直接覆盖，例如 `training.finetune_ckpt_path=/new/checkpoint`；stats 的 `dataset_dir` provenance 必须与新 profile 一致。

## 3. Offline open-loop 推理

RoboDojo：

```bash
python scripts/open_loop_inference.py \
  --profile configs/dataloader/canonical_eef_robodojo_openloop.yaml \
  --checkpoint-dir assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model \
  --output-dir outputs/openwam_alpha_open_loop/robodojo \
  --device cuda:0 --denoise-steps 2 --seed 42
```

RobotWin：

```bash
python scripts/open_loop_inference.py \
  --profile configs/dataloader/canonical_eef_robotwin_openloop.yaml \
  --checkpoint-dir assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model \
  --output-dir outputs/openwam_alpha_open_loop/robotwin \
  --device cuda:0 --denoise-steps 2 --seed 42
```

这是 open-loop contract smoke，不启动仿真器，也不报告成功率。输出 `prediction.npz` 同时保存 normalized 80D、gather 后 normalized EEF20、反归一化 physical EEF20、GT 和 mask；`manifest.json` 保存指令、episode、checkpoint、耗时和 shape。正常评估可把 `--denoise-steps` 从 2 调到 10。

RobotWin profile 的 `dataset_dir` 指向 `robotwin_traj_fullfield` 以读取 EEF/标定，`video_root` 明确指向用户提供的 `robotwin_traj`；这是为了绕开 companion 中容器内失效的 `/mnt/cpfs/...` 软链接。

## 4. Forward 前可视化

```bash
python scripts/visualize_model_inputs.py \
  --profile configs/dataloader/canonical_eef_interndata_smoke.yaml \
  --profile configs/dataloader/canonical_eef_agibotworld_smoke.yaml \
  --profile configs/dataloader/canonical_eef_robodojo_openloop.yaml \
  --profile configs/dataloader/canonical_eef_robotwin_openloop.yaml \
  --output-dir assets/openwam_alpha_data_examples --sample-index 0
```

打开 `assets/openwam_alpha_data_examples/index.html` 即可离线查看。每个源有九帧 PNG、包含指令/索引/shape/mask/stats provenance 的 JSON，以及 raw EEF20、normalized EEF20、unified 80D、GT 的 NPZ。

## 5. 新增数据源

新 bucket 必须具有 `meta/info.json`、episodes/tasks parquet、三相机视频，以及宽度 20 的 `observation.states.eef` 和 `actions.eef`。复制一个 `configs/dataloader/canonical_eef_*.yaml`，改 bucket/camera/video_root，先用明确 episode 生成 stats，再运行可视化。reader 对缺失 EEF、非有限数、退化 rot6d、stats provenance 或 shape 不一致会立即报错，不会静默回退到 joint 数据。
