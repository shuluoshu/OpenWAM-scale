# OpenWAM 源码解读 HTML 报告设计

## 目标

生成一份面向源码阅读者的单文件中文 HTML 报告，系统解释当前 OpenWAM 仓库的代码逻辑、执行入口、benchmark 支持情况和环境配置。报告应既能作为架构导览，也能直接指导读者完成环境选择、训练、部署与评测。

## 读者与边界

- 主要读者：准备理解、训练或评测 OpenWAM 的研发人员。
- 重点：以实际源码和配置为依据说明当前仓库，而非复述论文概念。
- 环境判断：检查 `/root/miniconda3/envs/` 下的已有环境，将其包版本与项目约束对照，并给出复用建议。
- 不执行模型训练、权重下载或 benchmark 仿真；这些操作成本高且不属于代码梳理范围。
- 不修改 OpenWAM 的运行逻辑和配置。

## 输出形式

- 输出一个自包含 HTML 文件，CSS 和必要的交互脚本内嵌，不依赖网络资源。
- 使用中文叙述，代码、类名、配置键和命令保留原文。
- 页面适配桌面与窄屏，支持目录导航、章节折叠和关键词过滤。
- 所有结论注明对应仓库文件；命令明确区分“仓库内可直接运行”和“需先安装外部 benchmark 环境”。

## 信息架构

1. **结论摘要**：项目定位、主入口、推荐环境、已支持 benchmark 数量与关键注意事项。
2. **仓库地图**：解释 `openwam/`、`configs/`、`scripts/`、`benchmarks/`、`tests/`、`third_party/` 的职责。
3. **三条核心调用链**：
   - 训练：`scripts/train.sh` → `torchrun` → `scripts/train.py` → Hydra 配置 → dataset registry → `OpenWAMTrainer` → architecture registry → 优化与 checkpoint。
   - 部署：`scripts/deploy.sh` / `openwam-serve` → `openwam.deploy.server` → checkpoint/config loader → inference engine → executor/policy → WebSocket。
   - 评测：benchmark 环境 → benchmark adapter/client → WebSocket 协议 → policy server → 动作转换 → 仿真环境。
4. **模块深读**：数据加载、模型架构、视频/动作/VLM backbone、训练器、部署层、配置系统及它们的接口边界。
5. **入口清单**：资产下载、训练、部署、推理测试、各 benchmark 评测与统计脚本。
6. **Benchmark 矩阵**：区分训练数据适配、仓库内评测客户端、外部评测，并记录相机、状态/动作契约和启动位置。
7. **环境配置**：项目 Python/CUDA/PyTorch 约束、核心依赖、可选 Cosmos 依赖、系统前置条件，以及现有 Conda 环境兼容矩阵。
8. **推荐操作路径**：给出最小安装、环境验证、数据准备、训练、部署、评测命令。
9. **风险与易踩坑**：Hydra 覆盖语法、checkpoint 自包含配置、prompt 模板、episode reset、归一化、显存和 benchmark 外部依赖。

## 展示设计

- 顶部为项目标题、扫描时间和源码提交号。
- 左侧为固定章节目录，正文使用卡片和紧凑表格。
- 调用链使用纯 HTML/CSS 流程图，保证离线可查看和可复制。
- 状态统一标记为“内置支持”“训练支持/外部评测”“计划中”“环境兼容”“需调整”“不推荐”。
- 环境推荐必须同时展示“版本约束判断”和“实测导入结果”，避免仅凭环境名称推断。

## 数据与验证方法

- 入口结论来自 shell/Python 启动器、`pyproject.toml` console script 和各 benchmark README/脚本。
- 模块调用关系来自注册表、训练器、部署 server、模型 loader 与 benchmark client 的静态追踪。
- benchmark 支持状态同时对照根 README、`benchmarks/` 实际目录及 `configs/dataloader/`，不把仅有 dataloader 的数据源误标为内置评测。
- Conda 环境通过对应解释器查询 Python 与关键包版本，并运行轻量导入检查；不安装或升级包。
- 生成后检查 HTML 结构、内部锚点、脚本语法、表格内容和关键源码引用，并以浏览器无关的静态检查作为最低验证。

## 验收标准

- 训练、部署、评测三条主链路均能追到具体文件和关键函数/类。
- 完整列出当前仓库实际存在的 benchmark 适配器，并清楚标注 RoboDojo 的外部评测性质。
- 完整列出 dataloader registry 中的数据集类型，且与 benchmark 支持表分开。
- 对当前每个 Conda 环境给出基于版本和导入检查的结论，提供一个明确的推荐方案。
- HTML 单文件、离线可打开、中文可读、目录和过滤功能可用。
- 不包含占位符、未经证实的运行成功声明或要求用户执行后才能成立的结论。
