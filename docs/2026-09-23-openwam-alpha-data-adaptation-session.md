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

### 局域网可视化服务（2026-09-23）

- 用户要求提供同网络可访问的端口网址。为避免暴露整个仓库、`.git` 和 checkpoint，只读发布 `assets/openwam_alpha_data_examples`。
- 沙箱内首次绑定端口因权限限制失败；按授权在宿主网络启动 `python3 -m http.server 18080 --bind 0.0.0.0 --directory assets/openwam_alpha_data_examples`，terminal session 为 `75353`。
- 宿主 IP 为 `10.1.128.132`、`22.52.149.254`；宿主侧请求 `http://127.0.0.1:18080/index.html` 返回 HTTP 200、Content-Length 5192，确认服务正常。

### Canonical reader 改用下一帧 state（2026-10-07）

- 用户明确选择修改 OpenWAM 的 canonical reader。未修改上层 `pretrain` 的 MoraVLA 代码或任何源 parquet。
- 新契约为 `action[k] = state[t+k+1]`、`proprio = state[t]`，均指归一化前的绝对 EEF20。源 `actions.eef` 不参与目标构造，允许其物理列和 schema 均不存在；配置中的 `action_key` 仅保留来源记录。
- 当前 profile 的 33 个原始 state 生成 32 个 action，随后仍归一化并映射到 `(32,80)`；9 帧视频采样不变。最后一个 state 不再作为起点，episode 尾部只监督真实存在的下一帧，剩余 action 补零且 mask=false，不跨 episode。
- Stats 的 action 总体改为各 episode 的 `states[1:]`；proprio stats 仍包含全部 state。新增 `action_target_source=next_state`、`action_target_offset=1`，旧 stats 会报错并提示使用 `scripts/data/prepare_openwam_eef_smoke.py` 重建。Trace 与报告增加目标帧索引，补齐位置为 null。
- 已重建四个现有 profile 的 episode 0/1 stats 与输入报告。action/state 行数分别为：InternData 556/558、AgiBotWorld 6014/6016、RoboDojo 1687/1689、RobotWin 925/927。这些是限定样例统计，不是全量训练数据统计。
- 先用源 action 故意等于 `state[t+3]` 的 fixture 验证失败，再实现修改。覆盖完整窗口、无源 action 列、尾部边界、旧 stats 拒绝、stats 总体及报告字段。Focused suite 最终 33 passed；LeRobot reader 回归 150 passed；Ruff 检查通过。独立只读代码审查未发现需修复问题。
- 四类真实 reader 均在 episode 0、起点 `t=7` 验证：全部 20 维的 32 个原始 GT 精确等于 `states[8:40]`，proprio 精确等于 `state[7]`，shape 为 `(32,80)`，有效通道为 `0..9,34..43`。每类还验证了 episode 0 最后一个样本只有一个真实目标，其余全零且 mask=false。
- 真实首个 GT 的右臂 XYZ（`state[8]`）：InternData `[0.3163852692,-0.2354988456,0.9350468516]`；AgiBotWorld `[0.5302587152,-0.3021565378,0.9371637106]`；RoboDojo `[0.3004709482,-0.3522998691,0.9215000272]`；RobotWin `[0.3015777171,-0.3078349531,0.9758213162]`。InternData 旧源 `actions.eef[7]` 为 `[0.3142823577,-0.2335686684,0.9335929751]`，不再作为 GT。
- 完整 20 维实测证据保存在 `outputs/action_alignment_audit/next_state_evidence.json`（运行产物，不纳入版本控制）。本次未重跑 GPU 训练或模型推理；2026-09-23 的 GPU 结果仍是历史记录，不能作为本次目标变更的验证。未提交或推送本次修改。


## 2026-10-07 Mora TCP-relative + joint94

- 用户要求扩展 OpenWAM Mora reader：保留原80槽位，左7 joint位于80:87，右7位于87:94，不重复gripper。canonical profiles切换`tcp_relative_joint94`；旧absolute80入口保留。
- 首个目标仍为state[t+1]；整个chunk的TCP pose与joint delta均相对窗口起点t。gripper保持绝对值，proprio使用当前绝对pose/joints。
- TCP来源逐侧显式声明（xyz+wxyz列或固定EEF→TCP变换）；缺失侧的9D pose置零、mask=false。InternData已接入两个TCP列；Agi/RoboDojo/RobotWin未声明TCP，暂只学有效joint/gripper。6轴臂第7joint屏蔽。
- 新masked raw34 stats覆盖完整chunk deltas，复用reader split/exclusion/hour/window选择，绑定94D表示与horizon。四套`*_relative94.npy`及94D样例报告已重建。
- smoke模型action_dim/state_dim=94，训练warm start显式允许四个投影80→94补零；部署/恢复不自动扩维。旧权重必须再训练才能用于新语义。
- 验证：123项focused测试通过；276项reader/mask/mixture/normalization回归通过（部分工具测试重叠）。独立raw-parquet审计四个bucket的t7→state8:40及末尾mask通过；独立代码复审无遗留阻塞。checkpoint子任务full-env含原finetune回归20通过。本次无GPU训练/模型推理、无源数据修改或commit。
- 详细用法：`assets/openwam_usage_docs/openwam-alpha-data-adaptation.md`。真实数值审计：`outputs/action_alignment_audit/tcp_relative_joint94_evidence.json`。


## 2026-10-07 joint→URDF/FK 抽样一致性报告

用户要求每个已接入数据源抽一条轨迹、joint标签经URDF/FK对照EEF并输出HTML。已生成 `outputs/fk_consistency/index.html`，含四源episode0全轨迹（共4,679个t+1目标）、源视频三帧、交互3D轨迹、位置/旋转误差、URDF与joint映射证据及逐帧NPZ。复现：`MPLCONFIGDIR=/tmp/openwam-fk-matplotlib python3 -m scripts.visualize_fk_consistency`。

结果必须按范围解释：InternData公共arm_base_link下原始EE一致，上一轮naive FK误用了左右独立臂基座；TCP +0.22m工具偏移及robot-frame扩展仅有首帧源标签派生诊断。AgiBot缺躯干state，只有消去公共躯干的双臂相对检查独立通过。RoboDojo找到实际源码dual_x5安装配置，无拟合绝对误差最大0.159mm/0.01735°。RobotWin同源URDF重建最大0.0000613mm/0.0000021°，验证的是转换，不是独立标定。reader前32个joint监督加回proprio与审计绝对joint一致。未修改reader训练语义或源数据。17项focused检查通过，独立审查无阻塞；HTML离线资产及JS语法检查通过，图像已目视检查。


## 2026-10-07：同一 TCP 的 head-reference + TCP-local 双分支（112D）

用户明确两套必须使用同一 TCP 物理点，缺 TCP 两套 pose 均 mask；未采用 canonical flange fallback。新增 `head_tcp_relative_joint112`：原94槽位保留，左右 head pose9 追加到94:103/103:112，raw52；head使用当前窗口头相机reference差分，local使用当前TCP自身差分。InternData明确读取robot_to_env_pose逐帧变换TCP到world，计入基座运动；缺头相机/标定仅屏蔽head，joint/gripper独立。

四个canonical配置与两源smoke模型已切112，四份relative112统计已重建，统计仍只覆盖profile所选episodes。旧80/94路径和统计保留。显式warm-start支持80→94、80→112、94→112，旧维度保留、新维补零。Open-loop导出head18反归一化输出；四套forward-input示例已更新。

独立真实样本核验：episode0、anchor7、targets8..39，InternData两臂的local/head矩阵复算最大分量误差1.44e-7；其余三个当前profile无已声明TCP，两套pose的mask均为false，joint维度为Agi14、Dojo12、RobotWin12。证据：outputs/action_alignment_audit/head_tcp_relative112_evidence.json。审查无阻塞项；未运行GPU训练/模型推理。

最终验证：76项相关CPU测试通过，新增/修改模块Ruff与git diff --check通过；两源mixture共16窗口，action(32,112)/proprio(1,112)与模型配置匹配。

## 2026-10-07：coverage HTML 合并轨迹与三类投影对照

用户确认同时展示相机模型、固定/随动视角、wrist/TCP 端点。现有 TCP coverage HTML 已加入原四数据源 FK 完整交互报告，以及 InternData、AgiBot、RoboDojo、posttrain RobotWin、V0_3 Genrobot、LightWheel、Mecka 七组真实画面轨迹。每组取 episode0 的 anchor30，目标 state31..62；原始数组、视频时间和标定来源保存在 `outputs/fk_consistency/projection_samples/`。

固定视角将整段世界轨迹投影到当前相机；随动对照选择 t/t+16/t+32，相机参数与背景图像同步切换。Genrobot double-sphere、Mecka Brown-Conrady 各自与忽略畸变的 pinhole 对照；只有 K 的来源不虚构畸变。头相机参考与 TCP-local 相对 XYZ 另画米制 3D 图，不直接对 relative delta 做透视除法。AgiBot/Genrobot TCP 候选有明确标注，Dojo/posttrain RobotWin 未声明 TCP 则只展示 EEF 诊断；没有修改训练监督或源数据。

独立审查发现的离线下载链接和画面外图例问题均已修正：八个旧 FK NPZ/JSON 文件嵌入 HTML；每个视角显示左右端点画面内点数，画面外整段不列图例。34 项 focused 测试和五个模块/测试 Ruff 通过，七组叠加图已目视检查。HTML 验证 52 张内嵌图、7 个选择控件、8 个下载文件、无外部渲染依赖，JS 语法/切换逻辑与 ZIP 内容一致性通过。没有运行浏览器自动化。

页面 `outputs/fk_consistency/tcp_coverage_v2_v3.html`（23.68 MB），同名 ZIP（14.79 MB）可下载解压离线打开。服务 HTTP 200 内容与本地完全相同；网关返回 PAI 登录跳转 302。浏览器地址：`https://897628-proxy-18081.dsw-gateway-cn-shanghai.data.aliyuncs.com/tcp_coverage_v2_v3.html`。复现：`MPLCONFIGDIR=/tmp/openwam-projection-mpl python3 -m scripts.visualize_tcp_projections --reuse-samples`。

## 2026-10-08：连续视频与表示/端点/FK 细化

用户明确要视频而非单张图片，要求头相机 TCP 轨迹、joint→URDF/FK 轨迹，以及 absolute/relative、wrist/TCP 的区分。已用七组各180个真实连续源帧（frame30..209）生成25段 H264 视频，全部嵌入同一个 coverage HTML。六组30FPS/6秒，RoboDojo25FPS/7.2秒；每帧推进背景和相机标定并显示真实未来最多32步。原静态报告折叠保留。每组可切换头相机端点/还原对照、固定/随动/模型投影、米制 head/local 相对轨迹；四机器人另有 FK 头相机投影与3D随时间轨迹，显示输入 q。网页支持暂停、进度、0.5×/0.25×、全屏，切换内容保留当前时间。

再次源码核验修正笼统说法：OpenWAM原始EEF路径为源参考系absolute，当前112D适配已是relative；Mora96默认源EEF（可能腕部/法兰），不自动取eef_tcp。Mora的random头/腕选择为可选模式，当前both recipe不执行随机。relative需加回端点原点并恢复到world后再投影；同端点同目标时刻的absolute/relative应重合。视频是源state目标的几何/转换核对，不是模型推理，也不是原生Mora actions.eef时序对齐验证。

FK严格使用已有全episode审计结果按整数frame_index匹配。InternData arm-base独立检查，世界/TCP扩展仍是源row0变换派生诊断；AgiBot缺躯干state，world FK屏蔽，展示左右腕相对FK；RoboDojo和posttrain RobotWin只验证已知EEF/法兰，不伪造TCP；后三组缺匹配joint+URDF，不提供FK。候选TCP及human virtual pinch均清楚标注。

验证：49项focused测试、Ruff、diff检查通过；1260源JPEG及25段视频的4500帧完整解码，FPS/时长/faststart/H264核对；HTML25个内嵌MP4与文件逐字节一致，7张卡，JS切换时间与速度逻辑通过，ZIP/镜像相同、重复构建幂等。没有浏览器自动化，已目视检查多源视频实际解码帧。HTML41.04MB、ZIP27.63MB，本机HTTP200返回最新内容，DSW网关302至PAI登录。复现：`python3 -m scripts.data.video_projection_samples`，`python3 -m scripts.visualize_trajectory_videos`；若仅更新HTML，使用后者`--report-only`。

## 2026-10-08：AgiBot 原点错误与无字段 TCP 可恢复性排查

用户观察 AgiBot 投影不在 TCP，要求排查并核查 RoboDojo/RobotWin 及表格中无 TCP 字段来源是否可从 CAD/config 恢复。已按端点、FK、相机分别取证，报告与视频已更新。

- AgiBot 源 EEF 是 wrist flange，跨臂 FK 对照支持该定义；旧 +100mm 变换计算本身匹配 audit URDF，但 named gripper_center 物理上靠近壳体，不能作为接触 TCP。完整视觉 URDF 又有约28mm安装件和不同右侧工具朝向，不能盲目替换。当前 release episode0 的 head audit 明确 borderline/borderline，未修；Stage2畸变系数未被Stage3保留，具体系数不可恢复。源/发布视频相同640×480、三个对应帧约44dB PSNR，没有几何缩放错误证据。无法可靠映射旧原始episode编号到release0，故未把其它episode的相机误差当作本例。尚未得到可信接触TCP或精确相机修正。
- InternData 用原生 TCP pose 并逐帧 robot_to_env，无需借用具名 URDF center，保持原有语义。
- 当前 posttrain RobotWin 匹配 URDF/config 与 SAPIEN3.0.0b1：joint frame Rx(pi) 与 config global Rx(pi)抵消，故 link6→仿真夹持中心为 TransX(.12m)、rotation I。已补全报告 TCP/FK→TCP；180帧两臂最大位置差约0.0000594mm，这是同源运动学重建而非独立实物标定。配置的视频原生320×240，640×480是报告缩放显示。历史采集runtime未独立固定。
- RoboDojo 原 observation.tcp 字段在Stage3未输出；config gripper_bias=.145m 和CAD支持局部+X原点候选，点位在指尖后约12.57mm。但可用源码只加载该参数而没有消费，无法裁决任务TCP的I/Rx(pi)朝向；完整pose继续mask，新增原点候选视频，不冒充原生TCP。

新增 `scripts/data/tcp_endpoint_policy.py` 供报告/样例提取统一使用：撤下AgiBot旧TCP标签、RobotWin派生完整TCP、Dojo只展示候选原点。旧转换JSON保留原矩阵并标记retired_as_contact_tcp。训练reader/profile/stats均未改。两个新增origin视频解释具名点/配置候选，当前共27段视频。覆盖V0_2/V0_3全部来源和当前posttrain的27行可恢复性表在 `outputs/fk_consistency/tcp_origin_audit/recoverability.json`；字段存在与完整变换可恢复性分离，包含型号/版本/安装/坐标链限制。源码、CAD、相机审计证据以JSON/TXT保存在同目录并嵌入HTML。

验证：53项focused测试通过，相关Ruff/diff通过，重新完整解码18段受影响机器人视频共3240帧，TCP masks和RobotWin FK→TCP数值核验通过；全部27个HTML内嵌MP4与磁盘一致，ZIP/镜像一致，播放器切换检查通过。HTML42,737,286 bytes，ZIP28,875,164 bytes。本机HTTP200内容hash匹配，DSW网关302至PAI登录。入口同原URL，新增 `#tcp-origin-audit` 区域；没有把AgiBot仍未解决的物理标定问题声称为已修复。

## 2026-10-08：用户指定 RoboDojo / RobotWin +120mm 验收版

用户要求两组先统一沿+X固定偏移0.12m画出，由用户验收。已明确按腕部/法兰局部+X→TCP实现：`T_world_tcp = T_world_wrist @ TransX(.12)`，旋转不变，左右各自使用当前腕部局部轴；源EEF与joint→URDF/FK均用同一变换。报告状态为`user_defined_tcp_preview`、`pending_user_review`，不视为已验收的标定，训练reader/config/stats未改。

RoboDojo旧145mm候选不再出现在当前视频选择器；完整报告更新为26段。新增轻量验收页`outputs/fk_consistency/tcp_120mm_review.html`（3,511,872字节），内嵌两组各一段头相机EEF/TCP对照和一段FK→TCP视频，可离线播放。54项focused测试和Ruff通过；重新解码两组8段更新视频共1440帧，4个验收页MP4与源文件字节一致，完整HTML/ZIP/镜像一致。两组实际pose反算局部偏移与[.12,0,0]最大误差≤8.33e-17m。验收页HTTP200、text/html、内容hash匹配。

验收链接：`https://897628-proxy-18081.dsw-gateway-cn-shanghai.data.aliyuncs.com/tcp_120mm_review.html`。

## 2026-10-08：验收后统一新版视频入口

用户视觉确认 RoboDojo / RobotWin 局部+X 120mm 点位，报告 acceptance_status 更新为 user_visual_accepted，仍不将目视确认表述为独立物理标定。用户指定的 #previous-fk 原先指向折叠区内旧FK iframe，现改为可见的新版连续视频总入口；历史报告保留在 #archived-fk，hash导航会自动展开目标的折叠父节点。长审计说明折叠，方便直接找到七组视频。AgiBot/AgiWorld 默认显示最新 origin 候选点对照，真实接触TCP仍 unresolved_contact_tcp；未修改其几何或训练监督。

主HTML含26段视频，全部内嵌内容与当前metrics引用文件逐字节一致，七组各一个默认显示模式；ZIP和镜像一致。锚点唯一且不在折叠区，Node验证自动展开与滚动逻辑通过，3项endpoint policy测试及相关Ruff通过。本机HTTP200，41,880,162字节内容与最新版hash一致；DSW网关302至PAI登录。未进行浏览器自动化。入口：https://897628-proxy-18081.dsw-gateway-cn-shanghai.data.aliyuncs.com/tcp_coverage_v2_v3.html#previous-fk 。

## 2026-10-08：新增三个 AgiWorld 连续视频样例

用户认为原擦白板样例不适合观察，要求追加多个 AgiWorld 视频。检查 task327/356/362/365/385/410 各episode0的六个时间点后，选取 task365（整理日用品）、task362（双臂叠短裤）、task410（倒水）。三者均取frame180..479，300连续源帧、30FPS、10秒；每组五种模式：origin、head、projection、relative、FK，共新增15段。头相机源审计均 normal/acceptable，repair not_required；已修正报告policy和视频标签，读取各自episode审计，避免沿用task512的borderline。

沿用同一 A2D canonical wrist-flange 轴定义，100/128mm只作为具名原点候选，真实contact TCP仍mask。每组重新读取对应数据集joint并运行URDF FK，没有复用旧episode的FK；缺躯干仍不提供world FK，跨臂最大位置差依次0.021104/0.012552/0.014753mm。候选点比腕部靠前但仍不能冒充已标定TCP。训练reader/profile未改。

独立读取源parquet检查三个样例的EEF/K/W2C/joint与frame_index完全对齐；视频ordinal与PTS、300帧、30FPS、H264/faststart核验通过。15段4500帧全部解码，三个样例各四个origin时刻及FK帧已目视检查。22项相关测试与Ruff通过，HTML现有10张样例卡、41段内嵌视频，媒体hash与文件一致、ZIP/镜像一致、锚点展开逻辑通过；没有浏览器自动化。HTTP200内容hash匹配，主HTML59,667,153字节；DSW网关302至PAI登录。

原入口#previous-fk保留，新样例置于旧擦白板之前，直达：https://897628-proxy-18081.dsw-gateway-cn-shanghai.data.aliyuncs.com/tcp_coverage_v2_v3.html#agibot-examples 。清单、独立验证、取样联系表和数据位于outputs/fk_consistency/agibot_examples/；复现：python3 -m scripts.visualize_agibot_examples。

## 2026-10-08：按用户截图与授权调整 AgiWorld 虚拟TCP

已读取用户截图 `/mnt/workspace/TEMP-FILE-STATION/screenshot-20261008-173222.png`，对应task362/ep0/frame216。用户明确授权先核对URDF、必要时人肉调整到夹爪末端虚拟中心。检查确认公式 inv(T_base_flange) @ T_base_target 正确；本机没有urdfpy，查阅其官方API并用仓库URDFModel实际执行全链相对公式与法兰直接链比较，两个G1模型/左右臂/各5组关节配置最大分量差3.0554e-16。数据匹配A2D/Beta G1资产，不是示例G2；本地G2模型也无对应gripper中心/接触垫links。

审计URDF的具名center在canonical X +100mm，视觉URDF加安装件约128mm；二者均不是手指接触中心。读取完整G1视觉URDF及真实OBJ，显式zero关节配置下，两侧pad包围盒中心的平均点在canonical flange约[0.2375526255,-0.0000221679,-0.0001647529]m。手指开合链不是固定变换，不能将cfg={}当作数据中真实开度。以此CAD参考并比较207/229/257/280mm投影，在三个任务12帧目视调整选择固定虚拟点TransX(.240m)，旋转沿用canonical腕部I；又检查三个任务各frame250/360/460。属于用户授权的CAD参考＋人工目视预览，不是恢复出的官方TCP标定。

报告policy只对当前四个A2D task362/365/410/512启用visual_tcp_preview，status pending_user_visual_review，tcp_orientation_verified=false，tcp_training_supervision_enabled=false；其他embodiment继续mask。所有四个Agi样例的TCP/head-relative/local-relative采用同一物理预览点，FK跨臂仍是腕部比较且world FK继续mask（缺躯干）。源数据、训练reader/profile/stats未改。主默认origin视频左旧100/128点，右新240点，当前t使用圈十字并标L/R TCP t，未来路径另画。静态报告同步更新，历史证据保留。

新增复现脚本scripts/data/agibot_tcp_geometry.py和scripts/update_agibot_visual_tcp.py；证据及原/新取样对照在outputs/fk_consistency/tcp_origin_audit/agibotworld/visual_tcp_geometry.json与visual_review/。7项endpoint policy测试包含先失败后通过的local轴/validity/预览性质检查，58项相关测试及Ruff通过。20段更新视频5400帧完整解码，TCP=eef@TransX(.24)、源frame/pose/joint/K/W2C和PTS一致；head还原最大位置差约1.91e-8m，local约1.24e-16m。41个HTML内嵌MP4hash匹配、ZIP/镜像一致、JS锚点通过，截图相同时刻新视频已目视检查，无浏览器自动化。HTTP200、60,116,925字节hash与本地一致，DSW网关正常302至登录。

直达截图对应样例：https://897628-proxy-18081.dsw-gateway-cn-shanghai.data.aliyuncs.com/tcp_coverage_v2_v3.html#video-agibot-task362-ep0 。


## 2026-10-08：AgiWorld 245mm微调与真实腕部相机视频

用户要求在240mm上继续微调比较，并给所有数据集增加腕部视角。按已说明的范围补齐原页面已有7类来源、10个样例；没有将27行可恢复性审计表声称为27组新视频。四个A2D样例统一使用canonical腕部局部+X 245mm，比上版增加5mm；新增240/245/250mm三列动态放大视频，每帧各列共用同一个裁剪区域，只标当前t点，左右臂分上下两行。检查task362截图对应216帧、330帧及365/410等样例；选择245作为待用户验收的人工预览，未宣称毫米级实物标定或以延长TCP修正相机误差。原20段Agi视频及静态缓存/端点说明同步更新；训练reader/config/stats未修改。

新增scripts/data/wrist_projection_samples.py、scripts/visualize_wrist_videos.py。InternData、AgiWorld四个样例、RoboDojo、RobotWin共7个样例，各读取左右真实腕部视频、自己的逐帧K与world/base→camera矩阵，按episode/frame_index及视频from_timestamp匹配。每组新增随动腕部视角和固定首帧腕部视角，左右相机并排，两臂TCP使用青/橙色。随动使用当前相机和当前背景投影世界轨迹t..t+32；固定模式冻结首帧相机与背景，仍滚动显示目标。这是光学相机像素投影，区别于TCP-local相对位姿坐标图。RobotWin使用各侧intrinsic_cv，禁止用头相机K回填；原生320×240。部分时刻当前TCP出视野，例如frame120左腕像素约(320.814,323.019)，严格mask，因此固定首帧左腕在本窗口可能全空。

Genrobot左右真实腕部视频来自robot1/2.sensor.camera0，K/D是double-sphere，180行T_b_c均为单位变换；没有完整、可验证的canonical finger-device EEF→腕部光学坐标链，不能把单位T_b_c当作EEF=相机。新增真实左右腕部视频，明确标注TCP投影mask。LightWheel当前样例只有head_left/head_right、Mecka只有ego视频，列表明确无腕部视频，不伪造投影。16路真实腕部流共3600个源JPEG，其中14路使用源声明标定，2路Genrobot投影屏蔽；未声称独立标定这些相机。

新增19段视频（14腕部随动/固定、1 Genrobot原画面、4长度对比），主页面共60段内嵌视频、10张样例卡。新增#wrist-views覆盖表，点击有腕部视频的条目会自动切到该模式并保留时间与倍速；各卡下拉也可选择固定视图和长度对比。#previous-fk及旧报告锚点保持。HTML 83,608,177字节，离线ZIP 58,863,556字节，镜像和ZIP内容相同。

验证：61项相关测试通过，新增模块Ruff与git diff --check通过。独立重读所有真实腕部流源parquet，核对K/E、同episode/frame、视频偏移及PTS；独立矩阵计算投影和视野mask，逐帧结果与视频渲染记录一致；19段新增H264/faststart视频4500帧完整解码。三组新增Agi的当前8模式各300帧亦全解码，并重查TCP=eef@TransX(.245)、FK关节源匹配和relative还原。HTML60个媒体hash全部匹配，JS锚点及腕部切换运行检查通过；已检查实际解码帧，无浏览器自动化。验证脚本和JSON位于outputs/fk_consistency/wrist_samples/及agibot_examples/。HTTP200返回内容hash与当前HTML相同；DSW网关302至PAI登录。

入口：https://897628-proxy-18081.dsw-gateway-cn-shanghai.data.aliyuncs.com/tcp_coverage_v2_v3.html#wrist-views 。复现顺序：需要重建Agi主视频时先运行python3 -m scripts.update_agibot_visual_tcp，随后运行python3 -m scripts.visualize_wrist_videos --reuse-samples（缓存缺失则去掉reuse-samples）；仅改页面用python3 -m scripts.visualize_trajectory_videos --report-only。主视频生成器会重建该样例metrics，故腕部补充生成器必须在其后运行。


## 2026-10-08：Agi多视角回调与Genrobot腕部轨迹恢复

用户反馈Agi245mm在左右腕画面中过长，并要求排查Genrobot无轨迹。已按同一物理点跨相机比较，保留原内外参。Agi三任务的人工腕部夹爪末端像素参考显示：左腕拟合长度约192–199mm、右腕约155–167mm；task362/frame216以手工点(336,243)/(337,239)为参考，245mm误差约49.63/67.36px，200mm约9.60/35.79px。因此先缩短为200mm预览，明确右腕仍有残差、头部也不保证最优；不能声称完成物理TCP标定。原CAD零关节几何237.55mm保留为不同证据，不能把它与视觉拟合混为一谈。源meta确认腕部外参经过retained-episode robust fixed EEF_T_camera拟合，非独立实物标定；这本身不证明拟合导致了全部误差。没有按相机分别修改TCP或添加像素补偿。

四个Agi样例主模式和静态缓存统一更新200mm。新增tcp_multiview视频四列160/190/200/245mm、三行头部/左腕/右腕，当前t点同步，245保留作旧版对照；默认打开此模式。另有190/200/210头部放大比较。manual landmarks、参数扫描、原图对照和可复现脚本在outputs/fk_consistency/tcp_origin_audit/agibotworld/。人工像素近似及遮挡/夹爪开合限制明确记录，200mm仍pending_user_visual_review，训练监督未启用。

Genrobot原因已定位：当前converter仅生成camera2头部逐帧W2C，腕部只有原始静态T_b_c；上一版未接通finger base→光学轴，W2C为NaN全部mask。单位T_b_c不等于EEF轴=相机光学轴。找到/mnt/data/cpfs/b5/libinjie/Finger-urdf/urdf/Finger-urdf.urdf:87–99固定cam关节，base_link→cam_Link零平移、rpy=(-π/2,0,-π/2)，对应T_base_cam旋转[[0,0,1],[-1,0,0],[0,-1,0]]。实际raw xyz+xyzw pose与canonical EEF在左右180帧最大分量差<5.4e-8。新helper计算inverse(T_world_eef @ T_base_cam)，保留每侧K/D double-sphere和canonical/pose/video三种有效性，非单位T_b_c严格mask。现有TCP候选局部[.1015,-.0003,-.0398]在相机坐标为[.0003,.0398,.1015]，640宽图左右当前点约(312.60,331.67)/(312.03,346.71)，抽帧落在夹爪中心附近。头部Monitor轴套到腕部会z=-.1015全无效，历史FRD链会把点翻到图像上方，均未采用。

Genrobot新增/恢复左右腕随动与固定首帧两段轨迹视频。标记cad_derived_preview：CAD与采集设备版本/独立光学校准仍未确认，TCP保持origin candidate。源数据、训练reader/config/stats未修改。坐标链证据、URDF hash、独立矩阵验证、4时刻原视频联系表在outputs/fk_consistency/genrobot_wrist_audit/；新增scripts/data/genrobot_wrist_geometry.py和4项测试。

验证：65项相关测试通过，Ruff和diff检查通过。24个腕部/长度/多视角补充模式5760帧全解码；独立源K/E、frame/episode/PTS、针孔与DS公式、URDF解析矩阵核对通过。三个新增Agi各9模式×300帧全解码，原Agi5主模式×180亦检查；HTML共65个媒体hash、镜像/ZIP、JS模式切换和锚点均通过。已目视检查新版Genrobot及Agi实际视频帧，无浏览器自动化。HTML92,077,999字节；HTTP200返回最新hash，DSW代理302至PAI登录。原#wrist-views/#previous-fk均保持。

浏览入口：https://897628-proxy-18081.dsw-gateway-cn-shanghai.data.aliyuncs.com/tcp_coverage_v2_v3.html#agibot-examples ，Genrobot在#wrist-views点击对应行可自动切换腕部模式。复现时先scripts.update_agibot_visual_tcp，再重新extract_wrist_sample(genrobot)以更新旧mask缓存，最后scripts.visualize_wrist_videos --reuse-samples。

## 2026-10-08：补齐 a1 / RoboMind v1、v2 / RoboChallenge v2 / ABC130k

按用户指定读取 /mnt/data/cpfs/b5/yeqingda/pretrain/docs/v0_3_training_data_audit.md，并实际检查现有release、URDF/MJCF、夹指mesh和转换记录。新增5类来源9种embodiment：a1 Genie1/ARX Lift2/Split Aloha，RoboMind v1 AgileX/双FR3，RoboChallenge v2 ALOHA/DOS-W1，RoboMind v2 AgileX，ABC130k YAM。每种episode0取180连续帧（a1 120–299，其余30–209），均30FPS/6秒。各有头部、左右来源相机视频，按条件提供头部端点、相机模型、relative还原、FK/真实joint动画、随动与固定腕部视图；共51段新视频。原报告当前116段、19个样例卡，#expanded-datasets为新增证据表，旧#previous-fk/#wrist-views保留。训练reader/config/stats及源数据未变。

前版遗漏不等于没有TCP。a1三种原生TCP可直接读；对应raw EE→TCP局部平移Genie1 +Z220mm、ARX +X161.57mm、Split +Z135mm，报告使用原生TCP而非URDF同名center。世界FK安装/工具链部分依赖源标签，标为诊断；独立arm-base FK保留ARX约54.6–56.6mm不匹配，另两种近数值误差。a1来自sim_updated_lerobotv30，未据路径断言图像实拍/仿真。

RoboChallenge当前release曾0804将TCP转腕部，不能套用旧审计端点。DOS-W1逆完整转换恢复TCP，当前EEF局部L约[150.413,-.050,.108]mm，R约[150.305,.092,-.057]mm，保留完整旋转；ALOHA匹配Piper100 URDF实际检查joint7/8与STL，135.03mm只是指框原点，没有确认的接触TCP，继续mask（未套用不同ARX模型120mm）。Challenge右臂缺独立episode基座标定，world FK mask，双臂局部FK保留。ABC YAM逆0804与0826处理，EEF→TCP=[129.833,0,-44]mm、R=I，并与匹配MJCF tcp/grasp site核对；V0_3目录为同源V0_2链接。ABC D405保留inverse_brown_conrady源系数，当前仅pinhole近似预览，未声称像素级校准；腕部安装链属于派生候选。

RoboMind三个样例缺可验证相机外参及世界/臂基座链，实际左右视频展示但投影mask。v1 FR3/v2原字段只叫左右相机，v1 AgileX有wrist命名也缺安装证据，均标“来源左右相机，腕部安装未确认”。AgileX读取ARX5 URDF及link7/8真实STL，末端几何候选约[155.570,-.0015,0]mm，仅局部FK黄色候选轨迹，不提升为已校准TCP。FR3本地缺匹配模型，解析Panda后因型号不符拒用，显示实际14joint时序动画。没有独立GT时不报告FK自比较为零误差。

新增提取脚本scripts/data/expanded_{a1,robomind,challenge_abc}_samples.py；通用FK bundle严格校验dataset/episode/帧顺序、shape、SE3和mask，新增4项回归测试。scripts/visualize_expanded_sources.py集成9个样例；共享腕部生成器保留其他生成器已有覆盖条目，避免重建旧样例时丢失新增覆盖表。

验证：73项相关测试通过；本轮修改脚本Ruff通过，git diff --check通过。逐一核对9个样例源parquet关节值与FK输入完全一致；27路源相机4860张JPEG可读，同episode/frame/视频ordinal/PTS一致。51个新H264/faststart视频共9180帧全部解码，尺寸1280×600/30FPS/6秒；relative还原最大元素误差<1e-5，缺失TCP/相机/右臂world FK的mask符合逐源证据。已目视检查9个样例实际视频帧，复查最终RoboMind相机角色标注和ABC候选标定说明，无浏览器自动化。116个HTML内嵌媒体hash全匹配，离线ZIP/镜像一致，JS锚点和保留时间/倍速的腕部切换通过。HTML121,995,558字节，HTTP200返回最新内容hash；DSW入口302至PAI登录。验证脚本/JSON在outputs/fk_consistency/expanded_sources/，全报告验证在agibot_examples/verify_report.py。

入口：https://897628-proxy-18081.dsw-gateway-cn-shanghai.data.aliyuncs.com/tcp_coverage_v2_v3.html#expanded-datasets 。重建新增视频：python3 -m scripts.visualize_expanded_sources；仅重新汇总HTML：python3 -m scripts.visualize_expanded_sources --report-only。源缓存需要先分别用三个expanded提取脚本生成；输出目录内保存具体URDF路径/hash、TCP恢复矩阵和缺失原因。


## 2026-10-09：relative joint 经可微 URDF FK 与 GT TCP 比较的训练辅助 loss

用户选择方案1“跟GT对比”，已接入实际训练 loss，保持32步、112D和原video/action flow监督。由预测flow恢复clean action，反归一化80:94 joint delta，加当前真实关节得到未来joint；通过可微Torch URDF FK得到当前/未来TCP，比较 inverse(T_anchor) @ T_future 与直接读取的原生GT relative TCP。GT不由GT joint FK生成，也不与另一套预测pose互相约束。位置项为欧氏距离平方(m²)，旋转项为0.5倍矩阵Frobenius距离平方；两项独立权重，通用默认0，当前smoke各1.0（尚未调参）。旋转项不是rot6d直接相减，也不是精确测地角平方。

当前InternData Genie1双臂启用：G1_120s URDF、arm_base_link→arm_l/r_end_link、各7joint，末端局部+Z220mm为TCP。GT读取 states.left_tcp_to_left_armbase_pose / states.right_tcp_to_right_armbase_pose；固定工具变换由源标签支持，未声称独立物理标定。辅助GT在匹配URDF臂基座下定义，再转当前TCP坐标，排除移动基座/躯干运动，避免强迫arm joint解释基座运动；现有112D action世界参考下的relative目标未变。AgiBot缺完整躯干及可验证时序基座/TCP链，继续关闭FK辅助；其他未配置来源贡献零，报告中的人工TCP预览未自动加入训练。

新增 openwam/kinematics.py、model/fk_supervision.py、dataloader/utils/fk_supervision.py；reader输出原生GT、raw anchor joint、归一化逆变换、URDF链和mask。base及IDM的compute_loss均接通，FK强制FP32、GT/标定detach、保留预测joint梯度。mask包含当前/未来GT与joint有效性、episode padding、action维度；quantile/min-max截断的目标只屏蔽FK，原FM监督保留。未声明的活动关节/mimic/非法链显式报错，不用零关节补齐。启用FK却没有action预测时显式报错。训练日志增加加权loss_fk_position、loss_fk_rotation及fk_valid_count。

验证使用 /root/miniconda3/envs/openwam：139项相关测试通过，Ruff与git diff --check通过；独立只读复核未发现实质问题，另运行18项新测试通过。真实InternData两episode五个窗口（含两处尾部）分别在CPU及CUDA执行，valid arm-step为64/60/2/59/2，截断屏蔽0/4/0/5/0。GT joint输入的CUDA位置平方误差最大3.81e-15m²、旋转chordal最大3.67e-14；加入0.05rad关节扰动后两项loss均增大，梯度有限非零。未运行完整基础模型训练或评估最终任务效果。

复现：/root/miniconda3/envs/openwam/bin/python -m scripts.verify_fk_training_loss --device cpu（或cuda）。结果位于 outputs/fk_training_loss/verification_cpu.json 与 verification_cuda.json；完整公式、配置、范围及限制见 assets/openwam_usage_docs/openwam-alpha-data-adaptation.md 的2026-10-09节。原HTML可视化未重建，源数据未改，未提交或推送。


## 2026-10-09：整理并提交当前适配分支至 fork

用户明确要求将当前代码改动提交并推送至 fork（https://github.com/shuluoshu/OpenWAM-scale.git）的 feat/openwam-alpha-v01-data-adaptation 分支。本轮整理 next-state reader、94/112D relative TCP/head/joint、checkpoint维度扩展、FK训练辅助loss、URDF审计与多视角视频脚本、配置、测试、归一化小文件和相关文档。临时 burn_gpu.py、nohup.out 以及另一个代码解读任务的 openwam_codebase_guide.html 保留本地，不纳入本次提交；outputs/内大型生成视频和HTML保留本地，提交复现脚本和已有文档记录。

推送前核对远程同名分支仍为 ef3f745，与本地基点一致；使用普通推送，不强推、不合并主分支。训练/数据与其他相关回归在 /root/miniconda3/envs/openwam 下220项通过。该环境缺matplotlib，两个依赖绘图的测试文件改用已有绘图依赖的系统Python单独验证，4项通过；共224项通过。未修改训练环境。
