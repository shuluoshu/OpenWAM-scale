# Stage1 随机图像扰动与并行启动

本版本叠加全局 shuffle 修复，并沿用原 task/subtask 指令、最终目标帧、TCP 几何和历史采样规则。代码更新不会自动切换已运行的 DLC 进程。

## 随机扰动作用范围

`stage1_multiview_sample.yaml` 和 `stage1_multiview_3000h.yaml` 开启以下设置；query、query history、完整 epoch 和 3 FPS preset 继承它们：

```yaml
dataset:
  observation_augmentation:
    enabled: true
    color_jitter: {p: 0.8, brightness: 0.2, contrast: 0.2, saturation: 0.2}
    patch_mask: {p: 0.5, area_range: [0.05, 0.2], aspect_range: [0.5, 2.0]}
    view_dropout: {p: 0.1, keep_one_view: true}
```

处理顺序是原始相机帧 → 原有 resize/三视角拼接 → 随机扰动 → tensor/冻结 VAE → query encoder。只处理当前观测，分别按样本抽取颜色、单个矩形遮挡和单路视角丢弃。矩形面积以选中相机的区域为分母，限制在该视角内部；这是像素遮挡，不是删除 ViT tokens。

- 结构性缺失相机一直保持黑色，不参与随机视角选择；整路 dropout 至少保留一路可用相机。
- `observed_view_valid` 保持原始相机可用性：随机遮住的相机仍是存在的观测槽位，query 读取其黑色图像特征；结构性缺失相机继续走原有效性 mask。
- goal 图像、TCP UV/有效性、task/subtask 指令和历史 10 帧不受扰动。没有 crop/flip，不改变投影坐标。
- 历史顺序、随机跳帧与时间间隔编码保持原逻辑。增强使用独立的样本标识 seed，并恢复调用方 RNG；不会改变 flow noise。
- 训练导出的 `observed.png` 是实际送入模型的增强输入；`provenance.observation_augmentation` 记录区域、选中相机和 seed。
- `evaluation_samples()` 使用干净观测，确保跨 step 诊断可比。没有把历史图像或干净目标写进增强分支。

若要关闭，在派生 YAML 覆盖 `dataset.observation_augmentation.enabled: false`。不传增强配置的旧数据类调用仍保持原行为。head-only preset 未启用本功能；对其强行启用会报错，以免把头部图像误当三视角布局。Stage1 不支持 `state_dropout`，Stage2 既有 state 增强保持原设置。

## 启动与恢复并行度

完整 epoch preset 新增：

```yaml
bootstrap_workers_per_node: 2
sample_order: global_shuffle
```

原启动是全局 rank 0–31 逐个加载。新启动按 `LOCAL_RANK` 分波次，各节点同时参与；4 节点 × 8 GPU、每节点并发 2 时：

| 波次 | 每个节点加载的 local ranks | 全局同时加载进程数 |
|---|---|---:|
| 0 | 0、1 | 8 |
| 1 | 2、3 | 8 |
| 2 | 4、5 | 8 |
| 3 | 6、7 | 8 |

每个 rank 仍按 DiT、VAE、T5 顺序加载，限制瞬时主存；VAE 编码检查移到对应 GPU 执行。设置为 1 时仍可在 4 个节点并行，设为 8 时同节点所有 GPU 并行加载，需自行确认主存/存储压力；超过本节点进程数会被 launcher 截到该数。显式配置必须为正整数。

启动打印 `bootstrap_plan`、`bootstrap_started`、`bootstrap_finished` / `bootstrap_failed`，附 rank、波次与耗时。各波次汇总异常后同步退出；不能捕获的进程崩溃、强制终止仍依赖 torchrun/process-group 超时处理。要求同构节点、连续 torchrun rank 分配，并验证各 rank 的调度配置一致。

新版本的 checkpoint 恢复也采用相同的节点内并发上限。初始化权重 seed 保持一致；DDP 数据分配、损失归约和目标没有更改。

## 数据与启动配置

仍使用原 CPFS `source.yaml`、3 FPS `population.sqlite` 和 Stage2 statistics registry，无需重新解码或重建总体。随机扰动发生在在线样本读取时，shuffle 发生在样本 ID 层。

现有操作入口和 smoke/3000h 选择见 [Stage1→Stage2 操作手册](stage12_one_pass.md)。完整 3 FPS 训练继续选择 `configs/goal/stage1_query_history_epoch_3fps.yaml`。DLC 的代码 checkout、输出目录和离线资产都必须是 DSW/DLC 可见的 CPFS 路径。

更新代码与配置后必须建立新的训练输出，按原离线准备/`--check` 流程登记已有索引和统计。不要直接复用旧任务的输出或 `--resume`：旧顺序与新 shuffle、增强/代码身份不同，检查会拒绝旧游标恢复。发布远程分支与提交/切换 DLC 是两个独立操作。

## 验证范围

新增测试覆盖真实解码样例的输入/GT/历史边界、断点重放、缺失相机、干净评估、query 前向反向、入口接线和真实 Gloo 多进程并发/故障传播。实际 32 卡 NCCL 启动耗时仍需新 DLC 测量，不能把 32 次串行变为 4 波次直接宣称为实测 8 倍加速。

2026-10-26 验证：193 项相关回归测试通过，独立审查未发现需修正的问题；另在单张 NVIDIA H20-3e 上加载完整 baseline，使用真实 LightWheel 样本完成一次前向、反向及 FP32-master AdamW 更新。该次触发颜色扰动和头部视角 dropout，8 帧有效历史、2 个有效 TCP 点，loss/梯度有限；加载约 217.19 秒、更新约 3.78 秒，峰值已分配显存 96.63 GiB。这是单样本功能验证，不是收敛或多节点性能结论。记录位于 CPFS `outputs/shuffle_fix_20261026/augmented_gpu_result.json`。
