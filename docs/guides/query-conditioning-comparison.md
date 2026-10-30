# Stage1 条件入口切换与单 subtask 对照

已选定 C 用于新实验。正式配置、自动 Stage1→Stage2 和启动命令见 [C 方案操作手册](query-prefix-training.md)。

基线提交为 `02edd62`：直接 readout 已移除，只有图像/TCP 联合 flow loss。

## 模型配置

```yaml
model:
  goal_representation: query_latent
  query_conditioning: normalized_cross_attention
```

可选值：

| 值 | Z 输入路径 | 去噪 self-attention 序列 | 训练的条件路径 |
|---|---|---|---|
| `legacy_text`（默认，旧配置兼容） | Linear(512→4096) → Wan text MLP | 80 零观测 + 80 目标 + 1 TCP | 原 text MLP、cross-attention |
| `normalized_cross_attention` | LayerNorm(512) → Linear(512→3072) | 同上 161 tokens | 新 adapter、原 cross-attention；text MLP 冻结且不执行 |
| `prefix_self_attention` | 与上行相同的新 adapter | 66 Z + 80 目标 + 1 TCP | 新 adapter、self-attention；text MLP、cross-attention 和 norm3 冻结且不执行 |

第三组是 Wan 中移植 PatchWAM 的 clean-prefix / noisy-suffix attention 组织，不是 FLUX 的双流/单流 Transformer 复现，也不是完整 PatchWAM 实验。全部条件先经过同一个 query encoder 得到 Z，第三组也没有加入图像或语言旁路。

Prefix 的 query 只读取 prefix；目标图像/TCP 可读取 prefix 及彼此。Z 不加扩散噪声、时间步为0；目标和TCP共享扩散时间步，但初始噪声独立。Prefix 按 query 索引分配 RoPE，目标/点沿用原位置。normalized cross 组保留零 prefix，是为了单独比较条件映射；不是认定该零前缀最优。

新快捷配置（继承既有历史、数据与训练配置）：
- smoke：`configs/goal/stage1_query_normalized_sample.yaml`
- 3000h：`configs/goal/stage1_query_normalized_3000h.yaml`

从隔离 worktree 或 DLC snapshot 启动时，请显式指定 `--source-yaml /mnt/data/cpfs/b5/yeqingda/pretrain/launch/moravla/train_files/pretrain_v0_3_curated_smoke.yaml`（3000h 替换为对应 YAML）；避免依赖 checkout 相对路径。

使用原 Stage1 启动器 `scripts/train_goal_curated.py --train --config <上述配置> --output-dir <CPFS新目录>` 或原有限遍历 launcher 中的 Stage1 config 字段。`sample` 指数据配置；`--smoke` 则是小模型合成单元检查，不能当成真实模型训练。

要做 prefix 对照，在继承配置中将 `model.query_conditioning` 改成 `prefix_self_attention`。不改此字段时保持 legacy。模型 bundle 保存 `conditioning_mode`，旧 bundle 缺失字段时默认为 legacy；完整加载按其保存的模式构建并严格校验权重。模式之间不自动迁移 optimizer/adapter。Stage2 只加载 encoder.*，不增加 Stage2 token 数，也不需要新的 Stage2 条件入口。

## 本次过拟合协议

- 单条 EgoDex subtask “Return stamp to box”，episode1904，当前帧300，目标帧458；注释区间为[246,459)。GT为子任务末帧，未认证成功。
- 同一源视频重新解码目标帧，与既有目标图像逐像素一致；使用此前审核的人手拇指尖/食指尖中点 TCP。左右点均有有效GT；头部视角；不存在腕相机，按缺失mask。
- 采用当前历史边界策略：当前采样片段从246开始，所以10个历史槽位中只有270帧有效，age=1秒，其余9槽mask；训练对有效帧再独立dropout0.2。没有跨episode缓存。
- 三组相同预训练Wan、相同随机query encoder（记录hash）、相同训练噪声/时间步/dropout；B/C新adapter初始权重相同（记录hash）。不是从已发生条件衰减的15k模型续训。
- 每组500次batch1更新；主网络lr1e-5，历史压缩器和条件adapter均1e-4，AdamW weight_decay=.01，clip_norm=1。BF16前向、FP32 master权重及优化器。全部组都无direct readout。
- 0/50/100/250/500步保留图像、TCP预测，20步Euler去噪、固定seed100042。最后增加100043/100044两个seed。另用相同5组time/noise测量flow loss，避免把训练随机时间步造成的loss抖动误认为回退。
- 记录纯训练步骤用时、峰值已分配显存；B 期间其他 GPU 进程开始共用设备，实测速度不适合用于独占吞吐结论。排除前10步、首次冻结特征编码及评估/保存。只缓存冻结VAE/T5结果，训练query/history compressor照常反传。
- 条件干预指标：相同含噪输入下更换指令或清空视觉，比较速度场输出。它能检测路径敏感性，不能证明语言语义正确；单样本实验本身可以靠记忆拟合。

运行：
```bash
/root/miniconda3/envs/openwam/bin/python scripts/compare_query_conditioning.py \
  --output /mnt/data/cpfs/b5/yeqingda/openwam-0911/OpenWAM/outputs/query_conditioning_subtask_20261027 \
  --config /mnt/data/cpfs/b5/yeqingda/openwam-0911/OpenWAM/outputs/head_history_overfit_20261024/config.yaml \
  --steps 500
```
输出目录必须先包含审计后的 `record.json`、`input_audit.json`、`batch.pt` 和 `report/inputs`。实验脚本拒绝覆盖已经存在的组；要重复实验，复制输入到新CPFS目录。这个实验配置只借用基线模型定义和权重；历史旧config中的readout字段、旧8500步checkpoint和原5k预算均不用于本次实验，实际设置记录在新 `protocol.json`。

结果入口：实验目录 `report/index.html`。它保留各步预测，附输入图/GT、VAE重构、曲线、左右TCP误差、配置及敏感性诊断。有限单样本结果不能替代多任务泛化、held-out语义条件或Stage2控制评估。


## 本次 500 步结果

下表为第500步、3个生成噪声种子的均值；训练只重复了1个初始化种子。

| 条件入口 | 图像 MAE /255 | TCP 平均欧氏误差 px | 训练峰值已分配显存 GiB |
|---|---:|---:|---:|
| 原 Wan 文本映射 | 3.339 | 5.227 | 86.01 |
| 独立 LN + Linear / cross-attention | 8.783 | 2.037 | 85.66 |
| Wan + PatchWAM 式 prefix | 5.274 | 3.877 | 68.79 |

原映射在这个样本的末步图像重构最好；独立归一化映射的TCP误差最低；prefix 在更早的100步图像拟合更快且显存更低，但其末步图像仍不优于原映射。不能宣称新入口全面胜出。直接复制当前图像 MAE=6.004；目标 VAE 重构 MAE=0.898（参考值，不是严格下界）。B 在这个预算下的图像误差仍高于复制当前帧。

旧入口输出的条件 RMS 从1.502降到0.00209；B从0.571变为0.1557，绕过了旧文本GELU。B仍对更换指令/清空视觉有响应，但这不是语义正确的证明。应通过同一观测、不同subtask目标的多样本控制检验条件利用，不能以单样本记忆能力替代。

输出全部保存在CPFS `outputs/query_conditioning_subtask_20261027/`：`comparison.json`、`protocol.json`、`input_audit.json`、`code_snapshot.json` 与 `source_used/`，每组 `query_goal.safetensors`、`train.jsonl`、逐步 eval 目录。`source_used/` 保留实际实验期间的相关源文件；随后修正了通用 curated trainer 在构建优化器前的 thaw 顺序（实验runner原本已经按此顺序执行，未改变本次三组训练），并完善报告打包。
