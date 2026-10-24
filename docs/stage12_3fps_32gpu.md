# 3 FPS / 单机 32 卡版本

新增 preset `configs/goal/stage12_query_history_posttrain_3fps_32gpu.yaml`，保留原 15 FPS / 8 卡版本。源目录和 episode 选择相同：仍使用 `outputs/stage12_posttrain_selection/source.yaml`，仅排除之前获授权的 `wipe_table_v1_test` 55 段。

| 项目 | 原版本 | 新版本 |
|---|---:|---:|
| Stage1 当前帧采样率 | 15 FPS | **3 FPS** |
| 30 FPS 源帧网格间隔 | 2 帧 | **10 帧** |
| 来源与 episode 选择 | 全部保留项 | 相同 |
| 单机 GPU 数 | 8 | **32** |
| 每卡 batch / 累积次数 | 1 / 1 | 1 / 1 |
| 全局 batch | 8 | **32** |
| 预测可视化 | 每 500 次更新追加 | 相同 |
| Stage2 最大更新数 | 100000 | **100000，未缩减** |

Stage1 一轮覆盖 3 FPS 网格上的全部有效 task/subtask 配对，然后保存检查点、冻结 query/history encoder，自动启动 Stage2。3 FPS 只控制当前帧 anchor：task/subtask 原文、原始区间末帧目标、目标头部 TCP、10 帧历史头部缓存与实际时间间隔编码不变。Stage2 仍预测原 32 步 action chunk 和每 4 帧视频，不把动作轨迹降为 3 FPS。

网格按 episode 起点对齐。特别短的 subtask 区间可能没有任何严格早于目标帧的 3 FPS anchor；本版本不额外补点。`population.json` 的精确 task/subtask 数量才是一轮预算。保留数据来源不等于保证每个极短指令区间都有样本。

## 启动

```bash
cd /mnt/data/cpfs/b5/yeqingda/openwam-0911/OpenWAM/outputs/query_latent_20261020/code
export PYTHON=/root/miniconda3/envs/openwam/bin/python
export STAGE12_CONFIG=configs/goal/stage12_query_history_posttrain_3fps_32gpu.yaml

# 如果当前 checkout 尚无派生 source.yaml，先按原手册生成排除选择。
# CPU 准备 3 FPS 索引与匹配统计；不要求准备机器有 32 卡。
"$PYTHON" scripts/train_stage12.py --config "$STAGE12_CONFIG" --prepare
"$PYTHON" scripts/train_stage12.py --config "$STAGE12_CONFIG" --check

# 在拥有 32 个可见 CUDA 设备的单机上执行。
"$PYTHON" scripts/train_stage12.py --config "$STAGE12_CONFIG" --train

# Stage1 中断恢复；卡数、采样率和数据不能更改。
"$PYTHON" scripts/train_stage12.py --config "$STAGE12_CONFIG" --train --resume
```

两个阶段均使用 `torch.distributed.run --standalone --nproc_per_node=32`。入口在启动模型前检查可见卡数；不足 32 卡就报错，不会在 8 张卡上挤入 32 个 worker。每卡依然持有完整 Stage1 模型/优化器；更多卡不会降低这部分单卡显存需求。32 个 rank 的 bootstrap/恢复串行进行以控制瞬时主机内存，正常训练仍为 DDP 同步。

新输出目录是 `outputs/stage12_posttrain_3fps_32gpu`，与旧实验分开。可视化仍在 `stage1/timeline.html` 追加。新 Stage1 配置 `stage1_query_history_epoch_3fps.yaml` 继承原 epoch 配置，仅添加 `sample_fps: 3.0`；学习率保持原值，未按卡数自动放大。新旧索引混用、修改采样率后继续旧 checkpoint 都会被拒绝。

Stage2 恢复、数据选择重建、baseline 与统计注册表说明见 [完整操作手册](stage12_one_pass.md)。切换配置请从新的一轮开始；不要将 15 FPS / 8 卡检查点视为这轮 3 FPS / 32 卡的覆盖记录。

## 验证与规模

同一组 16 个真实代表 episode 的 CPU 准备与检查结果：

| 指标 | 15 FPS / 8 卡 | 3 FPS / 32 卡 |
|---|---:|---:|
| task 配对 | 12639 | 2533 |
| subtask 配对 | 6692 | 1344 |
| 总配对 | 19331 | **3877** |
| 一轮全局更新数 | 2417 | **122** |

两份 probe 的 source SHA 相同，3 FPS 样本量为原来的约 20.06%。这是代表数据上的计数，未替代全量索引计数。按规模估算，Stage1 总样本约缩为 1/5、更新数约缩为 1/20；实际吞吐受通信、解码、缓存和显存影响，不能据此声称实际提速 20 倍。

Stage2 预算保持 100000 次更新，因此 global batch 从 8 到 32 后，其采样曝光量变为原来的 4 倍；这部分不属于本次稀疏采样的缩减范围。

已验证配置传递、30→3 FPS 的准确 anchor/goal、错误 FPS 拒绝、32-rank 尾批覆盖和既有双进程 Gloo 更新。**未启动生产训练，也未做 32 卡大模型显存或速度实测。**
