# OpenWAM-α 数据适配会话记录

- 日期：2026-09-23（UTC）
- 仓库：`/mnt/data/cpfs/b5/yeqingda/openwam-0911/OpenWAM`
- 状态：设计与环境盘点中
- 说明：本文件持续记录本次任务中用户可见的对话、执行的命令/文件操作、关键输出、设计决定和验证结果；不记录隐藏推理、凭据或无关敏感信息。

## 目标

1. 从指定预训练数据中选取小规模样本，提供可运行的 OpenWAM-α smoke 训练 YAML，并解释配置。
2. 使用指定 RoboDojo/RobotWin 数据与现有 checkpoint 跑通推理脚本。
3. 参考 MoraVLA 的 EEF 预处理/读取链路，将已转换的 LeRobot 数据适配到 OpenWAM，并更新对比 HTML。
4. 提供逐数据集的 model forward 前输入可视化脚本与样例，包括处理链、图像、指令和 GT。

## 已确认输入

- 训练数据：`/mnt/data/cpfs/b5big/pretrain_data_release/pretrain_datasetsV0_1`
- RoboDojo 数据：`/mnt/data/cpfs/b5/post_train_data/robodojo_sim`
- RobotWin 数据：`/mnt/data/cpfs/b5/post_train_data/robotwin_traj`
- MoraVLA：`/mnt/data/cpfs/b5/yeqingda/pretrain`
- 原对比文档：`/mnt/data/cpfs/b5/yeqingda/pretrain_vs_openwam_comparison.html`
- 用户授权：GPU 相关检查和 smoke 运行可在容器外执行。

## 对话记录

### 用户请求（2026-09-23）

用户要求完成四项工作：OpenWAM-α 小数据训练 YAML、推理跑通、参考 MoraVLA 适配 EEF 数据预处理与读取并整理 HTML、提供每个数据集在 model forward 前的可视化脚本和样例数据。

### 助手阶段性回复

- 说明将先读取项目规则与工作流，盘点 OpenWAM、MoraVLA、checkpoint 和数据目录，再形成可验证计划。
- 报告发现当前仓库已有 RoboDojo/RobotWin dataloader 测试骨架，但指定的 `pretrain_datasetsV0_3` 不存在，询问替代路径。

### 用户确认（2026-09-23）

用户确认先使用 `pretrain_datasetsV0_1`；说明容器内看不到 GPU，授权容器外执行；要求把所有操作和对话过程持续写入当前会话维护的 Markdown 文件。

用户随后确认推理验收先采用 open-loop，不要求启动闭环仿真 benchmark。

用户确认允许 RobotWin 适配器读取 `robotwin_traj_fullfield` companion 目录作为 EEF/标定来源；原 `robotwin_traj` 仍是视频来源。

用户批准推荐的总体架构：新增通用 canonical EEF LeRobot-v3 reader；V0_1 选 InternData-A1 与 AgiBotWorld 小规模数据；统一使用 absolute EEF20 → source normalization → OpenWAM 80D；RobotWin 使用 fullfield companion；foundation checkpoint 做 open-loop；输出 HTML/PNG/NPZ 可视化；新版对比 HTML 放在当前仓库内且不覆盖原文件。

用户随后批准验收与失败策略：缺失必要字段/stats 时 fail-fast；CPU contract、四类真实数据、两步 GPU 训练、两类 open-loop、离线 HTML 均作为验收项。

## 已比较的实现方案

1. **通用 canonical EEF LeRobot-v3 reader（已批准）**：消费已物化的 `observation.states.eef/actions.eef`，复用 OpenWAM 的视频、窗口、mask、normalize、unify 机制。
2. 分别扩展旧 RoboDojo/RobotWin/A1/AgiBot reader：数据源语义明确，但会引入多份相似 LeRobot 分支。
3. 整体移植 MoraVLA VlaPipeline：功能最全，但 local-delta 96D 输出与 OpenWAM-α absolute-EEF 80D contract 不兼容，范围和风险过大。

选择方案 1；借鉴 MoraVLA 的字段/坐标语义与显式处理阶段，但不移植其模型专用 96D compose/tokenizer 链。

### 助手阶段性回复（续）

- 确认后续使用 `pretrain_datasetsV0_1`，会持续维护本文件；GPU 检查与真实 smoke 将在容器外进行。
- 报告已定位 checkpoint，并开始核对 GPU、Python CUDA 初始化与代表性 EEF 数据 schema。

## 操作与发现

### 规则与工作流

- 完整读取 `/mnt/data/cpfs/b5/yeqingda/software/AGENTS_RULES.md`。
- 读取并启用 `superpowers:using-superpowers`、`superpowers:brainstorming` 和 `superpowers:writing-plans` 的说明。
- 当前遵循 brainstorming 设计门禁：业务代码、配置和脚本将在设计获用户确认后实施。

### 仓库状态

- 当前分支：`main`，HEAD `4e47093`。
- 用户已有未跟踪文件：`burn_gpu.py`、`nohup.out`、`openwam_codebase_guide.html`；保持不动。
- 仓库已有 OpenWAM dataloader、训练/部署脚本、RoboTwin benchmark 适配器与较完整测试集。

### 数据与 schema 初查

- 两个推理/后训练目录均为 LeRobot v3 风格：`meta/info.json`、`meta/episodes/*.parquet`、`data/*.parquet`、多相机 MP4。
- 当前 `openwam/dataloader/robotwin.py` 与 `robodojo.py` 主要读取原始 HDF5 布局，不能直接消费上述 LeRobot v3 路径。
- OpenWAM 已有 `LeRobotV3Reader` 和多 bucket 基类，适合复用并新增面向已预处理 EEF schema 的薄适配器。
- MoraVLA 的核心链路包括 EEF frame selection、world-to-camera EEF、delta action、图像键映射、action-space compose/mask、归一化和模型输入整理。
- OpenWAM 默认统一动作布局为 80D；MoraVLA 默认 `dual_eef_local_96d` 为 96D，不能直接照搬布局，需要明确映射语义。

### 路径与 checkpoint 初查

- 原指定 `.../pretrain_datasetsV0_3` 不存在。
- 已确认改用存在的 `.../pretrain_datasetsV0_1`。
- checkpoint 已定位：
  - `assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model/checkpoint_step_154000.safetensors`（约 24.8 GB，80D unified action）；
  - `assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-LIBERO/checkpoint_step_10690.safetensors`；
  - `assets/openwam_ckpt/openwam_study/video_backbone/robotwin_dual_system_joint_self_attention_wan21_vace_1_3b/checkpoint_step_9890.safetensors`（20D raw EEF）。

### GPU 与运行环境初查

- 容器外 `nvidia-smi`：1 张 NVIDIA H20-3e，显存总计 143771 MiB，检查时空闲 142646 MiB，compute capability 9.0。
- `/mnt/data/cpfs/b5/yeqingda/envs/openwam/bin/python` 在 `import torch`/CUDA 探测前后 30 秒内无输出并超时（退出码 124）。这不是“无 GPU”，后续需按系统化调试流程区分 NFS 环境启动慢、Python 包导入卡住或 CUDA 初始化问题。

### `pretrain_datasetsV0_1` 初查

- 顶层包含 `interndata_a1`、`agibotworld`、`robotwin`、`robochallenge_v2`、`egodex_wrist`、`egoverse_wrist`、`abc_130k`、`robomind_v1/v2` 等。
- 当前 OpenWAM 已有成熟的 `interndata_a1` 与 `agibotworld` LeRobot v3 reader，并且 foundation checkpoint 的保存配置本身也使用这两个源；因此它们是 smoke 训练的首选代表数据集。
- 候选代表 bucket：
  - InternData-A1：`interndata_a1/pick_and_place_tasks/pick_and_place_part2/omniobject3d-lotus_root/genie1__parallel_gripper`；
  - AgiBotWorld：`agibotworld/task_512/a2d_gripper__dual_gripper__base__unmodified`。
- foundation checkpoint 的 action/state contract 是统一 80D、33 个原始时刻、32-step GT action、视频 stride 4 后 9 帧、384×320 多视图。

### Open-loop 数据 contract 初查

- RoboDojo `robodojo_sim/sweep_blocks`：100 episodes / 60770 frames / 25 FPS；已有 20D `observation.states.eef` 和 `actions.eef`，也有 head/左右腕相机的 world-to-camera 标定。
- 用户指定的 RobotWin `robotwin_traj/blocks_ranking_size/demo_clean`：50 episodes / 23085 frames / 30 FPS；只有 14D `observation.state/action`，不含 EEF 或相机外参。
- 相邻的 `robotwin_traj_fullfield/blocks_ranking_size/demo_clean` 与原目录帧数一致，额外包含 20D `observation.states.eef/actions.eef` 和三相机 world-to-camera；其 `videos` 是指向原 `robotwin_traj/.../videos` 的软链接。因此 fullfield 是 MoraVLA 实际使用的“在原轨迹上补齐 EEF/标定字段”版本，而不是另一套视频数据。
- RobotWin 示例任务指令来自 `tasks.parquet`，例如 “Put large block, medium block, and small block in the center by size.”；RoboDojo 示例为 “Pick up the broom, hand it over to the right hand, then use the dustpan to sweep the blocks.”。
- 初步数据链路应保留绝对 EEF20 数据语义，在 OpenWAM reader 内按源 stats 归一化，然后 scatter 到统一 80D 的左臂 `0:10`、右臂 `34:44`；不会直接复用 MoraVLA 的 96D 输出布局。
- V0_1 的代表 InternData-A1/AgiBot bucket 没有当前 `meta/stats.json`（仅保留修复前备份或使用 MoraVLA external stats）；MoraVLA external stats针对 local/head/wrist delta 与 96D recipe，不能不加区分地用于 OpenWAM-α 的绝对 EEF20/80D contract。smoke 适配需要从选定 episode 生成独立、可复现的小规模 raw EEF state/action stats。
- Foundation checkpoint 可通过 self-contained component specs 构建，action/state 宽度为 80D。其 checkpoint 目录没有单一 `normalization_stats.npy`，符合多数据源模型的事实：离线脚本应使用当前样本所属数据源的 normalizer，将 raw EEF20 归一化并 scatter 为 80D；预测先保留 normalized 80D，同时按该源 stats 反变换为 raw EEF20 用于报告。

### 已执行的只读检查（摘要）

- `rg --files`、`find`：盘点仓库、MoraVLA、数据目录与 checkpoint。
- `git status --short`、`git log -5 --oneline --decorate`：确认工作树和近期提交。
- `sed`/`rg`：阅读训练/部署 YAML、OpenWAM dataloader、测试、MoraVLA pipeline/transform、benchmark 文档和对比 HTML。
- `find`：抽查 RoboDojo 与 RobotWin LeRobot 目录结构。
- `python3` + `json`：列出四类代表 bucket 的 `info.json` feature contract。
- `python3` + `pyarrow.parquet`：抽查 RoboDojo/RobotWin fullfield 的 parquet 实值、任务指令及相机标定字段。
- 容器外 `nvidia-smi --query-gpu=...`：确认 H20-3e GPU 与显存。
- 容器外 `timeout 30s .../envs/openwam/bin/python -c 'import torch; ...'`：Python/Torch 初始化超时，未得到版本或 CUDA API 输出。
- 未执行破坏性操作，未修改数据源。

## 待确认与后续

- open-loop 验收已明确：RoboDojo 与 RobotWin 各取一个真实轨迹样本，运行到 checkpoint 的 `generate`，保存预测、GT、mask 和可视化；不启动仿真器。
- 对候选 InternData-A1/AgiBotWorld bucket 做完整 schema 与 reader contract 检查。
- 选择数据适配实现方案并确认设计。
- 正式 design spec 已写入 `docs/superpowers/specs/2026-09-23-openwam-alpha-data-adaptation-design.md` 并完成占位符、内部一致性、范围和歧义自审；等待用户复核。
- 系统化定位 OpenWAM Python/Torch 初始化超时原因。
- 给出 2–3 种数据适配方案与推荐设计，等待用户批准。
- 设计批准后编写 spec、计划、测试与实现，并记录全部验证命令和结果。
