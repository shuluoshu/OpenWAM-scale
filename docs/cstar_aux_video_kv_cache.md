# C* 辅助视频训练与 action-only K/V 缓存

实现基于 DLC 固定版本 `dc1c586`，参考 `/mnt/data/cpfs/b5/yeqingda/openwam/model/architectures/dual_system/{joint_self_attn,mot_driver}.py` 的首帧 prefill 和 action cached loop。

## 保留的方案与两个 P0

Stage1 不变：当前三视角与 head history 经冻结的 visual-history encoder 产生 C*；正式配置仍为 120×3072 memory。Stage2 仍通过可训练 memory adapter 读取 C*，动作仍由独立 ActionDiT 预测，维度 112、horizon 32。不引入 LAV，也不把动作改成伪视频 patch。

1. 新增 `configs/goal/stage2_visual_history_aux_gpu.yaml`：`future_video_mode=joint`、`attention_mask_mode=isolated`、`video_attention_mask_mode=first_frame_causal`，video/action loss 权重均为 1。动作读取首帧、C* 和动作自身，不能读取未来视频；首帧也不能读取未来视频、动作或 C*。未来视频分支保留训练监督。视频目标可通过共享的视频参数和 C* adapter 影响训练，但效果收益需要实测。
2. action-only 推理先计算一次首帧各层 K/V，并计算一次 C* adapter 的各层 K/V；每个去噪步只运行 ActionDiT，键序保持 `[video, action, C*]`。缓存仅属于本次 `generate`，下一请求重新计算。保留 text padding、proprio mask、C* mask、动作 scheduler 和 inactive-action 归零规则。

这属于依赖隔离后的数学等价计算重排；FP32/BF16 仍会有浮点运算误差。不能给旧 mutual+joint checkpoint 直接删除未来视频依赖并声称等价。

## 使用

沿用现有 Stage2 训练命令的 checkpoint、数据、统计量和输出目录参数，将 config-name 改为 `goal/stage2_visual_history_aux_gpu`。新配置默认启用 `model.architecture.action_kv_cache=true`。

新运行可从 mutual baseline 初始化权重：preset 显式设置 `training.allow_attention_mask_warmstart=true`，仅允许 visual-history + joint_self_attn + joint 的 mutual→isolated 转换；需要新 optimizer。完整训练状态 resume 仍要求 mask 一致。Stage1 资产依赖与归一化校验保持原有要求。该 preset 继承的是 sample/GPU recipe，正式 DLC 的数据规模、batch size、步数等仍需沿用实际实验参数。

推理在原有输入（包括历史和当前观察）上设置：

```python
result = architecture.generate(
    **existing_generation_inputs,
    decode_video=False,
    use_action_kv_cache=True,
)
```

`use_action_kv_cache=False` 可用于同 checkpoint、同输入、同 seed 的数值对照。C* 仍由冻结 encoder 产生，公开 generate 入口拒绝外部 `goal_latent`/`goal_token_valid`。

缓存适用于 Wan22 TI2V、first_frame_causal、CFG=1，以及 isolated+joint 的 action-only 请求；mutual/isolated+none 也支持。joint 请求解码视频、CFG>1、不支持的 mask/backbone、非原生 timestep 设置、缺少或覆盖为多帧的 clean prefix 时自动回到原路径。`none` 本来就没有未来视频，仍可解码当前帧。训练不使用此缓存。原有配置默认不开启。

目前 preprocessing 保持原输入契约，可能仍分配 joint future latent 初始噪声；缓存优化的是 DiT 去噪计算，没有宣称移除了所有视频相关预处理。

## DSW 最小验证（2026-11-03）

环境：`/root/miniconda3/envs/openwam/bin/python`，PyTorch 2.7.1+cu128，NVIDIA H20-3e。

| 检查 | 结果 |
|---|---|
| CPU 扩展回归 | 201 passed，1 skipped（GPU optimizer 测试） |
| CUDA FP32 新增功能 + optimizer | 17 passed，含 GPU optimizer resume |
| CUDA BF16 新增功能 | 13 passed |
| CUDA FP32/BF16 单步训练 | video/action/adapter 梯度有限且非零，权重有更新；冻结 C* encoder 不变 |
| 5 步 FP32 缓存对照 | 最大 velocity 绝对误差 4.77e-7 |
| 5 步 BF16 缓存对照 | 最大 velocity 绝对误差 0.0078125 |
| 2 层模型 × 5 步调用次数 | visual layer 10→2，C* adapter 5→1 |

测试覆盖未来视频扰动/梯度隔离、首帧/C* 有效性、部分及全部无效 C*、padding/proprio、梯度 checkpointing、真实 frozen encoder 资产加载、seeded generate 对照、请求间条件更新、inactive action、warm-start/resume 限制与前缀回退。另修正基线 Cosmos 测试夹具缺失 `future_video_mode` 字段；生产 Cosmos 行为未改。

最小模型采用原生 Wan/ActionDiT/visual-history 模块，缩至 2 层、hidden=48，C* 为 120×48，VAE/T5 使用测试夹具；输入为合成数据。没有加载正式大模型 checkpoint 或真实 dataloader，没有启动完整训练或改动 DLC。上述调用次数不能当作端到端加速倍数，也未证明任务成功率与 joint 对齐。

运行可复现的 GPU 单步训练及 5 步缓存检查：

```bash
conda activate openwam
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 python scripts/verify_cstar_aux_cache.py --device cuda --dtype float32 --output outputs/cstar_aux_cache_validation/cuda_fp32.json
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 python scripts/verify_cstar_aux_cache.py --device cuda --dtype bfloat16 --output outputs/cstar_aux_cache_validation/cuda_bfloat16.json
CUDA_VISIBLE_DEVICES=0 OPENWAM_TEST_DEVICE=cuda OMP_NUM_THREADS=2 python -m pytest tests/goal/test_visual_history_aux_cache.py tests/goal/test_stage2_gpu_optimizer.py -q
CUDA_VISIBLE_DEVICES=0 OPENWAM_TEST_DEVICE=cuda OPENWAM_TEST_DTYPE=bfloat16 OMP_NUM_THREADS=2 python -m pytest tests/goal/test_visual_history_aux_cache.py -q
```

脚本支持 CPU float32；测试夹具的 avg_pool3d 不支持 CPU BF16。下一阶段应在正式 checkpoint 上测完整请求延迟/显存，再固定数据预算比较原 mutual+joint、isolated+joint 辅助视频、none 三组任务表现。
