# OpenWAM-α 数据适配会话记录

- 日期：2026-09-23（UTC）
- 仓库：`/mnt/data/cpfs/b5/yeqingda/openwam-0911/OpenWAM`
- 状态：设计已确认，正在编写实施计划
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

用户补充 MoraVLA/pretrain 的实际启动入口为 `pretrain/launch/qwena1_nogen/pretrain.sh`，要求参考该入口确认配置。

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

### MoraVLA 实际启动配置复核（2026-09-23）

- 完整读取 `/mnt/data/cpfs/b5/yeqingda/pretrain/launch/qwena1_nogen/pretrain.sh`。
- 完整读取其默认加载的 `launch/qwena1_nogen/train_files/pretrain.yaml`。
- 已核实 framework 为 `qwena1_nogen`，model state/action width 为 36D，训练策略为 `action_mode: delta`，EEF 默认在 `head_camera` 坐标系，训练时以 0.5 概率随机使用 wrist frame。
- YAML 的图像相对索引为 `[-15, 0, 15]`；启动脚本默认通过 CLI 将 `frame_sample_fps` 覆盖为 5。
- 启动脚本设置 `LEROBOT_STATS_ROOT=${DATA_ROOT_DIR}/stats`，配置启用 external stats。
- 这修正了先前将“当前默认 MoraVLA recipe”描述为 96D 的不准确假设：MoraVLA 代码中仍有 96D layout，但用户给出的实际启动入口使用 36D contract。
- 已据此修订设计文档。OpenWAM 适配仍保持已批准的 absolute EEF20 → source stats normalization → unified 80D；只参考 MoraVLA 的 EEF 字段、坐标帧选择、delta/normalize 次序和外部 stats 校验，不直接复用 36D delta 输出。
- 检查仓库根目录时未发现本地 `AGENTS.md`；本轮重新完整读取并遵循 `/mnt/data/cpfs/b5/yeqingda/software/AGENTS_RULES.md`。

## 实现执行记录（2026-09-23）

### 计划与流程

- 用户明确要求“开始执行”并多次要求继续。
- 创建实施计划 `docs/superpowers/plans/2026-09-23-openwam-alpha-data-adaptation.md`，完成占位符扫描和 `git diff --check`。
- 启用 `superpowers:executing-plans`、`superpowers:test-driven-development`；遇到导入超时时启用 `superpowers:systematic-debugging`。
- 当前检出是 `main` 的普通工作目录而非 linked worktree。用户已明确授权在当前仓库开始执行；保持用户原有 `burn_gpu.py`、`nohup.out`、`openwam_codebase_guide.html` 不动。
- 基线：`tests/dataloader/test_lerobotv3_utils.py tests/dataloader/test_mixture_from_config.py` 共 37 项通过。

### Canonical EEF reader

- 先创建 `tests/dataloader/test_canonical_eef_lerobot.py`，确认因模块不存在而红灯。
- 新增 `openwam/dataloader/canonical_eef_lerobot.py` 并注册 `canonical_eef_lerobot`。
- contract：absolute EEF20，action/state 分离 stats，raw normalization 后 scatter 到 80D，episode/window 硬限制，trace opt-in，missing EEF/non-finite/degenerate rot6d/provenance mismatch fail-fast。
- 有意覆盖 `__getitem__` 不使用基类的“失败后换窗口重试”，避免坏 EEF 被静默跳过。
- reader 5 项、reader+mixture 14 项测试通过。

### Stats、YAML、open-loop 与可视化

- 新增 bounded stats 脚本和测试；相关 8 项测试通过。
- 新增两训练 profile、两 open-loop profile 和 `openwam_alpha_v0_1_smoke.yaml`；Hydra 组合 2 项测试通过。
- 新增 `scripts/open_loop_inference.py`；2 项 orchestration/shape/non-finite 测试通过。
- 系统化定位 open-loop 单测 180 秒无输出：根因是导入 `openwam.deploy.*` 时 package `__init__` eager-import 完整 deploy/model 栈，在容器内触发 CUDA 304 warning 与大量共享文件系统 module stat。将 checkpoint loader 延迟到真实运行时导入，并在脚本内按公开 sync schedule contract 构造 schedule；测试恢复正常。
- 新增 `scripts/visualize_model_inputs.py`，2 项 PNG/JSON/NPZ/offline HTML 测试通过。

### 四类真实数据 CPU 验收

- 仅扫描 episode 0/1 生成 stats：InternData-A1 558 行、AgiBotWorld 6016 行、RoboDojo 1689 行、RobotWin 927 行。
- InternData-A1 指令：`Pick up the lotus root with the right arm and place it in the basket`。
- AgiBotWorld 指令：`Wipe the whiteboard`。
- RoboDojo 指令：`Pick up the broom, hand it over to the right hand, then use the dustpan to sweep the blocks.`。
- 首轮 RobotWin 可视化失败，证据为 fullfield 下 `videos` 软链接指向容器内不存在的 `/mnt/cpfs/post_train_data/...`；实际视频位于用户提供的 `/mnt/data/cpfs/b5/post_train_data/robotwin_traj/...`。
- 先增加 failing test，再在 LeRobot 基类增加可覆盖的 `_resolve_video_file` hook，canonical reader/profile 增加显式 `video_root`。metadata/EEF 继续来自 fullfield，视频来自原轨迹。canonical + InternData 回归 109 项通过。
- 重新执行后四个 profile 均产出 PNG/JSON/NPZ，`assets/openwam_alpha_data_examples/index.html` 离线报告生成成功；人工查看 RobotWin contact sheet，九帧三视角内容正常。

### GPU smoke 首轮结果

- 容器外 H20-3e：143771 MiB total / 142646 MiB free；Torch 2.7.1+cu128，实测 `import torch` 24.84 秒，CUDA available，GPU tensor 加和成功。此前 30 秒探测超时不能证明环境不可用。
- 首次两步训练已成功完成数据构建、24.8 GB foundation 权重加载和第一个 forward/backward；在第一个 Adam optimizer step 的 `torch._foreach_sqrt` OOM。
- OOM 证据：GPU 139.81 GiB，PyTorch allocated 112.91 GiB、reserved-unallocated 11.18 GiB、free 13.83 GiB，临时申请 22.43 GiB。
- 根因是 12.4B 总参数/6.0B trainable 参数在单卡上的 Adam step 峰值与显存碎片，不是数据或模型 shape 错误。
- 为使 smoke YAML 在单卡稳定可训，按配置原生能力将 `zero_stage` 改为 2、`offload_optimizer_device` 改为 `cpu`，并在文档命令加入 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`；随后重跑两步。

### GPU smoke 重跑结果与本轮续接（2026-09-23）

- 用户询问“现在呢”，助手说明会接管既有 GPU session，不重复启动训练，并继续维护本日志。
- 重新完整读取 `/mnt/data/cpfs/b5/yeqingda/software/AGENTS_RULES.md`；仓库根目录仍无独立 `AGENTS.md`。复核实施计划、当前分支和工作树：仍在用户已经授权执行的 `main` 普通工作目录中，保留三个用户文件不动。
- 通过既有 terminal session `26883` 取得修正后训练的最终输出。DeepSpeed ZeRO-2 + CPU optimizer offload 初始化成功；两步训练均完成且 loss 有限：step 1 `loss=0.1251`（video `0.1199`、action `0.0052`），step 2 `loss=0.3834`（video `0.3832`、action `0.0002`）。总训练迭代约 51 秒，进程退出码为 0。
- 下一项验收是使用 foundation checkpoint、CUDA、2 denoise steps 对 RoboDojo 与 RobotWin 各运行一个真实 open-loop 样例，并检查保存数组的 shape 和 finite 性。

### 真实 CUDA open-loop 验收（2026-09-23）

- 使用 `/mnt/data/cpfs/b5/yeqingda/envs/openwam/bin/python`、foundation checkpoint、`cuda:0`、seed 42 和 2 denoise steps 运行 `scripts/open_loop_inference.py`。
- RoboDojo profile：checkpoint `generate` 成功，joint denoising 约 24.25 秒；指令为 `Pick up the broom, hand it over to the right hand, then use the dustpan to sweep the blocks.`；保存 `outputs/openwam_alpha_open_loop/robodojo/{prediction.npz,manifest.json}`。
- RobotWin profile：checkpoint `generate` 成功，joint denoising 约 5.14 秒；指令为 `Put large block, medium block, and small block in the center by size.`；保存 `outputs/openwam_alpha_open_loop/robotwin/{prediction.npz,manifest.json}`。
- 两次运行都返回 `prediction_normalized80 (32,80)`、gather 后的 normalized/physical EEF `(32,20)`、GT `(32,80)`，active indices 精确为 `0..9,34..43`。
- 单独加载两个 NPZ 检查：所有浮点数组均 finite，两个输出目录都仅包含预期的 `prediction.npz` 与 `manifest.json`。
- RoboDojo 首次启动在共享存储导入/加载阶段长时间无 stdout；只读检查显示进程处于 disk-sleep、GPU 尚空闲。尝试用 `py-spy dump` 采样调用栈时，采样工具因目标进程已经结束而返回 `No such file or directory`；随后读取原 terminal session，确认推理本身已正常退出（code 0）。未因此修改代码或重启成功任务。

### 最终验证记录（2026-09-23）

- 首次将 focused 测试文件以“`tests/dataloader` → `tests/` → `tests/dataloader`”交错顺序传给 pytest 时，18 项通过、9 项 `test_mixture_from_config.py` 在 setup 阶段找不到 `fake_dataset_factory`。该 fixture 位于 `tests/dataloader/conftest.py`，不是产品代码报错。
- 系统化最小复现：`canonical + mixture` 为 15 passed，`prepare + mixture` 为 12 passed，`canonical + prepare + mixture` 的交错参数顺序复现 fixture error；将两个 `tests/dataloader/*` 参数相邻排列后，同样 18 项全部通过。曾尝试把测试 bucket helper 移出 test module，但不能改变该行为，已完整撤回该无效修改。结论是 pytest 对交错目录参数的 conftest scope/collection 行为；最终命令将同目录文件连续排列。
- Fresh focused suite：27 passed（canonical、mixture from config、stats、open-loop、visualization、configs、docs）。
- Fresh LeRobot 回归 suite：146 passed（LeRobot-v3 utils、InternData-A1、canonical、mixture from config）。
- `git diff --check` 通过；相关源码、配置、测试与文档的尾随空白扫描无结果。
- 离线链接检查：`assets/openwam_alpha_data_examples/index.html` 的 12 个本地链接、`docs/pretrain_vs_openwam_data_pipeline.html` 的 5 个本地链接全部存在。
- 资产清点：四个 profile 各有 PNG/JSON/NPZ，另有四份 bounded stats；两个 open-loop 输出各有 manifest 与 prediction NPZ。
- 最终自审未发现 TODO/FIXME/未实现分支。stats 脚本第 86 行的 `pass` 仅是 `Path.relative_to` 的正常 `try/except` 控制流，用于允许输出位于数据源目录之外。
- 撤回无效测试-helper 尝试后，在最终文件状态重新运行相同的正确排序 focused suite：27 passed in 2.55s；再次运行 `git diff --check` 通过。
- 最终 `git status --short` 确认 `burn_gpu.py`、`nohup.out`、`openwam_codebase_guide.html` 仍仅为用户原有 untracked 文件；未加入、未覆盖、未删除。当前仍是普通 `main` 工作目录，未自行 commit、push 或创建 PR。

### 集成选择（2026-09-23）

- 用户选择选项 2：创建功能分支、提交、推送并创建 Pull Request。
- 远端为 `https://github.com/OpenWAM-Official/OpenWAM.git`；当前环境未安装 `gh` CLI。将先创建功能分支并精确暂存本任务文件（包括被 `.gitignore` 忽略、但属于交付要求的四套样例与 stats），明确排除三个用户文件和运行期 `outputs/`，再尝试通过现有 Git 凭据推送。PR 创建将依据推送结果和可用认证能力继续。
- 已创建分支 `feat/openwam-alpha-v01-data-adaptation`。首次 cached diff 检查发现 5 个新文件有多余 EOF 空行，提交被检查阻止；删除多余空行、重新暂存后 `git diff --cached --check` 通过。
- 已创建功能提交 `c243611`（`feat: add OpenWAM alpha EEF data pipeline`），包含 39 个任务文件、四套样例及 bounded stats；未包含 `outputs/`、checkpoint 或三个用户文件。
- 助手最初按选项 2 尝试向官方 `origin` 推送；用户立即要求暂停并明确目标应为其 fork `https://github.com/shuluoshu/OpenWAM-scale.git`。助手向进程发送 Ctrl-C，push 以 exit 130 中断；随后通过 `git ls-remote --heads` 只读确认官方仓库没有生成 `feat/openwam-alpha-v01-data-adaptation` 分支。后续保持官方 `origin` 不变，只向独立的 `fork` remote 推送。
- 添加 `fork` remote：`https://github.com/shuluoshu/OpenWAM-scale.git`，并提交本条纠正记录（`96d39ae`）。`git push -u fork feat/openwam-alpha-v01-data-adaptation` 成功，远端创建同名分支并设为本地 upstream。GitHub 返回 PR 创建入口：`https://github.com/shuluoshu/OpenWAM-scale/pull/new/feat/openwam-alpha-v01-data-adaptation`；当前环境没有 `gh` CLI，因此没有自动提交 PR 表单。
