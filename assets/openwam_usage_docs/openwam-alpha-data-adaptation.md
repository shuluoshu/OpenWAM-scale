# OpenWAM-α Mora 数据适配：头相机 relative TCP + TCP-local + joint112

2026-10-07：四个 canonical profile 已切换到 `action_representation: head_tcp_relative_joint112`。只读源 parquet，不改 MoraVLA 或预处理数据。两源 smoke 使用 InternData / AgiBotWorld 的 episode 0、1；统计不是全量训练统计。

## 数据与动作契约

保留原 80D 槽位，追加 joint14 和头相机 pose18。两套 pose 必须使用同一个明确的 TCP 物理点：

| 槽位（Python 半开区间） | 内容 |
|---|---|
| `0:9` / `34:43` | 左 / 右 TCP 相对位置 3D + 相对旋转 rot6d |
| `9` / `43` | 左 / 右目标 gripper，绝对开合值 |
| `80:87` / `87:94` | 左 / 右机械臂 joint delta，各 7D |
| `94:103` / `103:112` | 左 / 右 TCP 在当前头相机坐标轴下的 relative pose，各 9D |
| 原有其余槽位 | 保留位置，本 reader 不提供其监督 |

读取窗口 `[t, ..., t+32]`，输出 action `(32,112)`、proprio `(1,112)`；GT 第 k 项始终来自 **state[t+k+1]**，不读 `actions.eef`。所有未来目标共享当前 t 的参考系：

```text
relative_xyz[k] = R_tcp[t].T @ (p_tcp[t+k+1] - p_tcp[t])
relative_R[k]   = R_tcp[t].T @ R_tcp[t+k+1]
head_xyz[k]     = R_world_to_head[t] @ (p_tcp[t+k+1] - p_tcp[t])
head_R[k]       = R_world_to_head[t] @ R_tcp[t+k+1] @ R_tcp[t].T @ R_world_to_head[t].T
joint_delta[k]  = q[t+k+1] - q[t]
```

旋转取矩阵前两列，以列优先方式展开为 6D。上式 TCP 位姿均在共同世界坐标系中；整个 chunk 使用当前 t 的头相机和 TCP 参考，不使用未来相机参考，也不逐步累加。proprio 保存当前世界 TCP 位姿、当前关节角及当前头相机中的 TCP 位姿。33 行中按 stride 4 取九帧视频；要改 chunk 长度须同时改 `num_frames` 并重建统计。

`eef_frame: both` 是 MoraVLA 的选项，表示同时构造两套 EEF。其当前训练配置配合 `dual_eef_local_96d` 使用 head-reference + local，`random_wrist_prob` 不作用于 both。OpenWAM 本 reader 使用显式 `action_representation: head_tcp_relative_joint112`，不依赖 `eef_frame`，也不随机选 EE 坐标系。

`head_world_to_camera_key` 指定当前头相机的 world→camera 4×4 行优先矩阵。没有头相机或有效标定只屏蔽头相机 pose。没有头相机时设置 `head_camera: null`，用 `primary_camera` 显式指定可用的条件视频；该视频不会被当成头相机监督。

`tcp_reference_to_world_key` 可指定每帧 source→world 变换（xyz+wxyz 或 4×4 行优先）；缺省表示 TCP 源已在世界参考系。InternData 显式使用 `states.robot_to_env_pose`，先把两个 TCP 从 robot 转到 world，再生成两套标签，也因此计入机器人基座运动。指定但缺失/无效的变换会屏蔽相关 pose。


TCP 必须逐侧明确指定：`tcp_pose_keys.left/right` 读取 `xyz + quaternion(wxyz)` 的 7D 列；或 `tcp_from_eef.left/right` 指定 4×4 固定变换，满足 `T_reference_tcp = T_reference_eef @ T_eef_tcp`。只有确认 EEF 原点就是所需 TCP 时才能显式使用单位矩阵。两种来源不可在同一侧同时指定。

| 配置 | 当前 TCP 来源 | 监督 |
|---|---|---|
| InternData | `states.left_tcp_to_robot_pose`、`states.right_tcp_to_robot_pose` | 双侧两套 pose、gripper、14 joints |
| AgiBotWorld | 未声明 TCP；现有 EEF 原点为 flange，未假设 TCP 偏移 | gripper、14 joints；两套 pose 均屏蔽 |
| RoboDojo / RobotWin | 未声明 TCP 来源 | gripper、12 joints；两套 pose 与两侧第 7 joint 屏蔽 |

无 TCP 的一侧将两套共 18 个 pose 通道置零且 mask=false；不回退到 canonical EEF / flange；不会取消该侧有效 joint/gripper。ego 等没有 joint 的样本也可将 joint 全屏蔽，保留 video 监督。缺失数据不进入统计，归一化后再次清零。不存在的 wrist camera 显式设 null；EEF / joint 列可缺失。该模式尚不读取 finger pose 监督。

episode 末尾只监督实际存在的 `N-1` 个下一状态，补齐位置 mask=false，不跨 episode。trace 的 `action_target_frame_indices` 对填充项记录 null。

## 1. 生成与 reader 一致的 masked relative stats

```bash
python scripts/data/prepare_openwam_relative_stats.py --profile configs/dataloader/canonical_eef_interndata_smoke.yaml
python scripts/data/prepare_openwam_relative_stats.py --profile configs/dataloader/canonical_eef_agibotworld_smoke.yaml
python scripts/data/prepare_openwam_relative_stats.py --profile configs/dataloader/canonical_eef_robodojo_openloop.yaml
python scripts/data/prepare_openwam_relative_stats.py --profile configs/dataloader/canonical_eef_robotwin_openloop.yaml
```

输出为对应 `*_relative112.npy`，写在当前仓库。统计内部为 TCP/gripper20 + joint14 + head18 的 raw52，归一化后 scatter 到 112D。统计覆盖所选 episodes 的所有有效窗口及完整 chunk delta，忽略展示用的 `max_windows`；按维度 counts 排除 masked label，rot6d 固定 identity normalization。proprio 统计来自这些窗口的起点。

脚本复用 reader 的 split、排除列表、episode 限制、hour sampling 和窗口索引；`--split val` 用于验证集。metadata 绑定 TCP / joint 映射、schema、选中 episodes、split、horizon 与 stride。旧 absolute stats 不能用于新表示。

旧 94D 路径仍可通过 `tcp_relative_joint94` 与旧 relative94 统计使用（其 TCP proprio 保留原始源参考系）；当前 112D 路径统一到 world。

旧 80D 路径仍可通过 `action_representation: absolute_eef80` 和旧 20D stats、`unify_action_map: ["0-9", "34-43"]` 显式使用；其统计入口是 `scripts/data/prepare_openwam_eef_smoke.py`。

## 2. 两步 smoke 训练

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
NPROC_PER_NODE=1 torchrun --standalone --nproc_per_node=1 \
  scripts/train.py --config-name openwam_alpha_v0_1_smoke
```

`configs/openwam_alpha_v0_1_smoke.yaml` 的关键字段：

- `defaults.model=dual_system`：构建 dual-system；本配置覆盖 `model.architecture.action_dim=112` 和 `state_dim=112`。
- `training.expand_action_dim_from_checkpoint=true`：仅在 finetune warm start 时允许 80→94、80→112 或 94→112，保留旧投影权重，新维度补零；不代表旧 EEF 权重已经学会新相对语义，仍需训练。其他形状不匹配仍报错，resume / 部署不自动扩维。
- `training.finetune_ckpt_path`：从已下载的 OpenWAM-α foundation 权重 warm start，不恢复旧 optimizer step。
- `dataloader.type=mixture`：混合 InternData-A1 Genie-1 与 AgiBotWorld A2D 两个明确 bucket。
- `episode_indices=[0,1]`、`max_episodes=2`、`max_windows=8`：硬限制每个源的 smoke 数据量。
- `num_frames=33`、`video_stride=4`：32-step GT 与九帧视频。
- `normalize_mode=quantile`：使用 q01/q99 映射到 `[-1,1]`；rot6d 不缩放。
- `unify_action=true`、`unify_action_map=["0-9","34-43","80-93","94-111"]`：raw52 scatter 到 112D。
- `batch_size=1`、`max_steps=2`、`gradient_accumulation_steps=1`：恰好两个 optimizer step。
- `mixed_precision=bf16`、`use_gradient_checkpointing=true`：单张 H20 的 smoke 内存设置。
- `zero_stage=2`、`offload_optimizer_device=cpu`：将 Adam state/step 放到 CPU，避开 12.4B 模型在单卡 optimizer step 的显存峰值。
- `dataset_num_workers=0`：问题可直接定位到主进程。
- `lambda_video=1`、`lambda_action=1`：视频与动作 flow-matching loss 均参与。
- `lambda_fk_position=1`、`lambda_fk_rotation=1`：2026-10-09 新增可微 FK 对 GT 的辅助监督；当前仅配置了 InternData，详见下节。
- `save_steps=0`：profiling/smoke 不写约 25 GB 的最终 checkpoint。
- `project.seed=42`：dataset、noise 和 timestep 可复现。

路径迁移时可直接覆盖，例如 `training.finetune_ckpt_path=/new/checkpoint`；stats 的 `dataset_dir` provenance 必须与新 profile 一致。

## 2026-10-09：relative joint → 可微 FK → GT relative EEF 辅助监督

用户选择方案1：与GT比较，不与模型预测的另一套pose比较。当前环境 `/root/miniconda3/envs/openwam`，保持32步。新增 `training.lambda_fk_position` / `training.lambda_fk_rotation`，通用默认值均为0（保留旧训练行为），当前两源smoke均设1.0。权重只是可运行初值，未做训练调参。

训练路径：

```text
x0_hat = noisy_action - sigma * predicted_flow_velocity
predicted_delta_q = inverse_joint_normalization(x0_hat[..., 80:94])
q_future_hat = raw_q_anchor + predicted_delta_q
T_pred_relative = inverse(FK(q_anchor) @ T_end_tcp) @ (FK(q_future_hat) @ T_end_tcp)
T_gt_relative = inverse(T_gt_anchor) @ T_gt_future
L_fk_position = mean_valid ||p_pred_relative - p_gt_relative||²
L_fk_rotation = mean_valid 0.5 * ||R_pred_relative - R_gt_relative||²_F
L_total = lambda_video*L_video + lambda_action*L_action
        + lambda_fk_position*L_fk_position + lambda_fk_rotation*L_fk_rotation
```

位置项单位m²；旋转项为矩阵chordal距离（小角度近似rad²），不是直接减rot6d，也不是精确角度平方。每个样本先对有效的“手臂×未来时间步”平均，再对batch平均；无可用FK样本贡献零。物理FK loss不额外乘scheduler时间步权重。FK运算强制FP32并保持joint梯度，GT、当前真实joint、标定和URDF常量detach。现有视频/action flow loss未修改。

数据源以 `fk_supervision.arms.left/right` 显式声明：`urdf_path`、`base_link`、`target_link`、`joint_names`（与reader joint索引顺序一致）、`endpoint_transform`、`endpoint`、`gt_pose_key`、`gt_frame: urdf_base`。读取源GT pose7（xyz+wxyz），绝不把GT joint的FK结果作为GT。解析只接受完整的fixed/revolute/continuous/prismatic链；未提供的活动joint或mimic直接报错，不默认为零。读取时记录URDF路径/hash。

**当前启用范围：**InternData Genie1双臂；匹配 `G1_120s.urdf` 的 `arm_base_link → arm_l/r_end_link`，end→TCP采用源标签一致性核验支持的局部+Z0.22m。GT来自 `states.left_tcp_to_left_armbase_pose` / `states.right_tcp_to_right_armbase_pose`，这些标签对应共同URDF arm_base_link，而非各侧安装link。该固定工具关系属于源标签支持的关系，并非独立实物标定。

**坐标系边界：**辅助loss比较URDF臂基座系中的TCP运动，再表达在当前TCP轴下。原112D action的local/head目标仍包含其原有world/reference语义，没有改动；若基座运动，辅助loss仅约束机械臂joint引起的局部运动，不将不可由这些joint解释的世界运动强加给机械臂。当前仅提供这种显式base-local GT接口，不自动从head/world标签推断安装外参。

AgiBot仍未启用：缺躯干关节，已有跨臂同一时刻检查无法证明跨时间世界系TCP轨迹正确。RoboDojo/RobotWin等未配置此训练辅助项，报告里的人工TCP候选不会自动成为监督。

mask包括GT当前/未来pose、当前/未来所需joint、动作时间/维度mask及episode尾部。quantile/min-max截断的GT joint无法由现有FM目标无损还原；只对该手臂/时间步屏蔽FK loss，避免目标冲突，原FM监督保留。缺GTpose仅关闭FK，不关闭有效joint/gripper监督。

训练日志新增 `loss_fk_position`、`loss_fk_rotation`（均为加权后的总loss贡献），以及进度条/W&B的 `fk_valid_count`（当前batch有效手臂×时间步数；多卡日志为每rank计数均值）。Debug CSV跟随loss labels记录两个FK项。要禁用辅助监督，将两个lambda设0；移除数据源的 `fk_supervision` 可同时停止读取额外GT列。

复现真实数据CPU/CUDA数值与梯度核验（不加载foundation模型）：

```bash
/root/miniconda3/envs/openwam/bin/python -m scripts.verify_fk_training_loss --device cpu
/root/miniconda3/envs/openwam/bin/python -m scripts.verify_fk_training_loss --device cuda
```

输出 `outputs/fk_training_loss/verification_cpu.json` / `verification_cuda.json`。检查episode0/1共5个窗口（包含尾部）：GT joint预测的FK误差接近浮点精度，给joint加0.05rad后两项loss变大且梯度有限非零。该检查不是完整112D模型训练或效果评估。

## 3. Offline open-loop 推理

必须使用按此 relative112 表示训练并保存的 checkpoint。现有 80D foundation 不能直接评估这些 112D profile；下方路径是待替换占位符。本次未运行 GPU 训练或模型推理。

RoboDojo：

```bash
python scripts/open_loop_inference.py \
  --profile configs/dataloader/canonical_eef_robodojo_openloop.yaml \
  --checkpoint-dir /path/to/trained-relative112-checkpoint \
  --output-dir outputs/openwam_alpha_open_loop/robodojo \
  --device cuda:0 --denoise-steps 2 --seed 42
```

RobotWin：

```bash
python scripts/open_loop_inference.py \
  --profile configs/dataloader/canonical_eef_robotwin_openloop.yaml \
  --checkpoint-dir /path/to/trained-relative112-checkpoint \
  --output-dir outputs/openwam_alpha_open_loop/robotwin \
  --device cuda:0 --denoise-steps 2 --seed 42
```

这是 open-loop contract smoke，不启动仿真器，也不报告成功率。输出 `prediction.npz` 同时保存 normalized 112D、gather 后 normalized EEF20、反归一化 physical EEF20/head18、GT 和 mask；`manifest.json` 保存指令、episode、checkpoint、耗时、shape 和 action 目标帧索引。正常评估可把 `--denoise-steps` 从 2 调到 10。2026-09-23 的历史 GPU 训练/推理记录属于旧数据语义，不作为本次 next-state 修改后的验证结果；历史预测不能直接和重建后的 stats 混用。

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

打开 `assets/openwam_alpha_data_examples/index.html` 即可离线查看。每个源有九帧 PNG、包含指令/索引/shape/mask/stats provenance 的 JSON，以及 raw EEF20、normalized EEF20、unified 112D + raw joint14/head18、GT 的 NPZ。

## 5. 新增数据源

新 bucket 需有 LeRobot v3 的 info、episodes/tasks parquet 和至少一个可用条件视频。复制 profile 后显式设置 TCP 来源、joint 列及左右 joint 索引。joint metadata 有完整 left_/right_ 名称时可自动识别并排除 gripper；无名称则必须配置 `joint_indices`。每侧最多 7 个，不足的维度屏蔽。只有可确认的 TCP 来源才启用 pose 监督；新增或修正来源后重建 stats。
