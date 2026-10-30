# Stage1 / Stage2 历史输入开关与缓存隔离

本说明适用于本次 history-window 修改后的代码，不适用于正在运行的旧代码快照。

## 固定 10 秒窗口

默认使用当前帧前约 10、9、…、1 秒的 10 个头部相机槽位，最旧到最新排列。按数据集 FPS 映射到源帧，取整后的 `history_ages=(anchor-index)/fps`（秒）送入 Time PE，不把槽位编号当时间。分数 FPS 下取不早于窗口起点的帧；当前帧、重复帧、选定片段之前的帧均不进入历史。

目前时间依据是数据集帧索引/FPS，不应称为已验证的传感器硬件时间戳。原有 mecka VFR 路径仍使用源行/FPS 的时间轴，并在 trace 中显式标注；未改写原始数据的时间对齐。

训练时每个有效历史槽位独立以 `frame_dropout_prob` 概率被 mask，默认 0.2。被 mask 的像素为零，索引为 -1，age 为零，attention mask 为 false。不会重新填充被丢弃的槽位，也不强制保留一帧。评估/推理不随机 mask。`jitter_fraction: 0.0` 是新的固定槽位配置；0.5 仅为旧实验显式复现保留，不用于这次要求的固定十秒实验。

## 只调整一个开关

Stage1 预设：
- `configs/goal/stage1_query_history_sample.yaml`：smoke/sample 数据。
- `configs/goal/stage1_query_history_3000h.yaml`：3000h 数据。
- `configs/goal/stage1_query_history_epoch_3fps.yaml`：全量来源、当前帧 3 FPS 的一轮遍历；继承 3000h 配置。

顶层开关：

```yaml
history_enabled: true   # false：不读取历史，也不构建历史压缩器
```

模型的 `model.query_history.enabled` 与数据的 `dataset.history.enabled` 都引用它。关闭历史不会关闭当前三视角或指令输入。

可以在 configs/goal 下新建自己的派生 YAML（base_config 相对该文件解析）：

```yaml
base_config: stage1_query_history_epoch_3fps.yaml
history_enabled: true
dataset:
  history:
    jitter_fraction: 0.0
    frame_dropout_prob: 0.2
```

`frame_dropout_prob: 0.0` 保留所有可用历史；`1.0` 在训练时 mask 全部历史，适用于保留相同模型结构的消融。若要验证时也完全屏蔽历史，应显式传入全 false 的 history_valid；dropout 概率不会改变验证策略。

Stage1 命令形状（CONFIG/INDEX/OUTPUT 需要指向你的实际配置、已有兼容索引、新输出目录）：

```bash
python scripts/train_goal_epoch.py --train --config "$CONFIG" --index "$INDEX" --output-dir "$OUTPUT"
```

32 卡仍由已有的 4×8 DLC/torchrun 启动器运行同一个入口。sample 单进程训练使用 `scripts/train_goal_curated.py --train --config "$CONFIG" --output-dir "$OUTPUT"`。

Stage2 使用 `goal/stage2_query_history_sample` 或 `goal/stage2_query_history_3000h` 配置，Hydra override `history_enabled=false` 同时控制模型、goal contract 和 dataloader；不要通过 `+goal=...` 把这个完整配置叠加到 train 上。正常使用 `scripts/train.py --config-name goal/stage2_query_history_3000h history_enabled=true`，并保留原来的 checkpoint、数据、统计量等必需 override。

开关改变了历史压缩器是否存在，因此有历史与无历史 checkpoint 不可混用。Stage2 开关须与加载的 Stage1 checkpoint 匹配；严格兼容检查会拒绝不匹配权重。

## Episode 隔离

Reader 根据解析后的数据源目录和 episode index 生成稳定 int64 `history_episode_id`，随 history_images / ages / valid 传给模型。不同数据源的相同 episode index 不相同。

历史特征 LRU 逐 batch 行检查身份；A→B、B→A 均整库清空，即使图像像素完全相同也重新编码。episode 首帧所有历史 mask=false；不足十帧也先清空缓存，只编码当前样本的有效历史。Stage2 goal 结果缓存同样按 episode 身份隔离并清空。VAE 每次独立编码继续清理自身时序状态。

旧 fixture 或调用方没有 episode ID 时仍可运行，但禁用跨次调用复用；不能静默假设它与上一次属于同一个 episode。部署调用方必须在新 episode / 环境 reset 时更新身份并只传当前 episode 的历史；代码无法从伪造的输入身份识别边界。

## 对正在运行实验的影响

修改文件不会热更新已经启动的 Python/DLC 进程。当前旧任务未停止，固定 code snapshot 未改写。本次更改需要新进程使用新代码和解析后的新配置才会生效。

旧 finite-epoch run 的 config/code identity 不同，不能直接对旧输出目录执行 `--resume`，也不能绕过一致性校验。改变历史采样策略后的续训需要单独准备有明确来源记录的迁移/重启配置；本次没有迁移优化器或重排 DLC。

## 当前三个结构选择的来源

- 直接 readout 分支及图像 latent / TCP 回归损失已删除。Stage1 只有联合 flow decoder 生成目标图像与 TCP，通过这两个去噪损失训练 Z。Stage2 仍只加载 encoder。
- QueryAuxiliaryDecoder 的 observation prefix 是全零 latent，占据原解码器的观测槽位；不是黑图经过 VAE。它保留旧 patch/位置/attention 布局，而将实际观测全部经 Z 输入。零 latent 经带 bias 的 patch embedding 后不一定是零 hidden tokens；该前缀仍消耗计算。
- Z 经 condition_projection(512→4096) 后进入原 Wan text_embedding(4096→3072→3072)。这是本仓库 query-latent 改动 c11ecd8 复用解码器接口的实现，不是 latent 在语义上变成文本。Wan 官方代码中的这个模块实际接收 text embeddings：https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/model.py 。没有证据将“64+2 queries 接入原文本 MLP”归因于 PatchWAM。

历史输入与缓存隔离修复后，已按用户要求移除 readout。零前缀与条件适配层保持原实现；readout 删除本身不能证明已解决条件分布不匹配和负向饱和。

“兼容旧编码器”的说法不准确：兼容的是本地 `PatchGoalExpert` 联合去噪解码器的观测 prefix / patch 位置 / attention 布局，不是某个旧视觉 encoder。

PatchWAM 论文使用 FLUX.2 [klein]，动作以固定重复和补零编码成 visual-width patch；当前图像、历史图像、含噪目标图像及动作进入共享视觉通路。公开 `awomo_05i/model.py` 将语言/多模态条件与线性投影的 state 放入 `encoder_hidden_states`，由 FLUX 的 context projection 和联合 attention 融合。这不等于把本项目 Z 经 Wan 文本 MLP。

参考：[PatchWAM §3](https://arxiv.org/html/2609.25961v1)、[Awomo 0.5I ActionPatchModel](https://github.com/Awomo-WestlakeDI/Awomo-0.5/blob/main/awomo/awomo_05i/model.py)。


后续新增了可切换的独立归一化 cross-attention 与 Wan prefix 条件入口，见 [条件入口对照手册](query-conditioning-comparison.md)。默认 legacy_text 仍保留此处描述的旧映射；新实验必须显式选择模式。
