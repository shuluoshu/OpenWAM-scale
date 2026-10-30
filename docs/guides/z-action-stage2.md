# Stage2：用目标 Z 替代显式未来视频

## 配置

保持 Stage1 query encoder 与 Z 的定义不变。新方案在进入 VAE 前仅保留当前图像，MoT 中没有未来视频 token；当前观测仍通过 Wan 分支提供视觉特征，action 读取当前观测、只读 Z KV、文本和 state。Stage1 encoder 冻结。只去掉未来视频预测，不是移除整个当前视觉编码器，也不代表已实现跨去噪步的当前视觉 KV 缓存。

| 方案 | `model.architecture.future_video_mode` | `training.lambda_video` |
|---|---|---|
| 原版：future video + Z + action | `joint`（默认） | `1.0` |
| 新版：current observation + Z → action | `none` | `0.0` |

新方案 preset：`goal/stage2_z_action_sample` 或 `goal/stage2_z_action_3000h`。两者支持 `history_enabled=false` 关闭历史，但必须选择与此开关一致的 Stage1 checkpoint。默认历史开启、10 秒 / 1 Hz，仅头部视角，TimePE 编码真实时间差；当前输入三视角不变。当前支持 Wan22 TI2V、query_latent、dual_system/joint_self_attn。

训练必须同时切换配置，不能只在旧模型推理时删除未来 token。旧 checkpoint 可用于同拓扑 warm-start；两种模式不要共用 optimizer/resume 状态。

```bash
python -m torch.distributed.run --standalone --nproc_per_node=1 scripts/train.py \
  --config-name goal/stage2_z_action_sample \
  goal.checkpoint=/absolute/cpfs/stage1/query_goal.safetensors \
  training.finetune_ckpt_path=/absolute/cpfs/baseline_checkpoint_directory \
  dataloader.source_yaml=/absolute/cpfs/source.yaml \
  dataloader.normalization_registry=/absolute/cpfs/registry.json \
  project.output_dir=/absolute/cpfs/new_stage2_run
```

preset 默认 CPU-offload / bf16 / batch1 / 梯度检查点参数由 preset 继承；按硬件再调整。把 config-name 改成 `goal/stage2_z_action_3000h` 可切换全量默认 YAML。必要时显式覆盖数据路径，避免 worktree 下相对路径指向错误目录。

## 本次两组约 5 小时过拟合

实验根目录：`/mnt/data/cpfs/b5/yeqingda/openwam-0911/OpenWAM/outputs/z_action_stage2_comparison_20261027`。

- 配置：`comparison_config.yaml`，固定样例：`data_v4/samples.pt`，清单：`data_v4/manifest.json`。
- authoritative smoke YAML 为 12 类、46 个源目录。抽取覆盖 15 种平台的 21 个 task/subtask 窗口，所有窗口都有有效动作目标；不是声称每个源目录都已训练。
- 动作沿用当前实际 reader 的 32 步 × 112 维；baseline 视频是 9 帧、每 4 个源帧采样一次。未改成历史讨论中的 50 步。
- 两组从同一原版动作模型初始化，冻结同一广覆盖 Stage1 20k query encoder。它来自既有全量运行，不是本次新训的 C 单样本 decoder；部署只使用其 query encoder。
- 在 H20 上 GPU optimizer 的临时梯度/状态张量曾超过可用显存；fused 也曾分配失败；开启 expandable allocator 后第二例出现非有限 loss，未通过连续更新。因此正式两组统一采用 原生 AdamW + `training.offload_optimizer_device=cpu`，保留原模型大小与学习率。GPU 模式失败记录保留在 smoke 子目录，不计入正式对照。
- 在有限样例上按类别均衡、每轮 shuffle。固定输入、关闭随机图像和历史增强以隔离架构差异。生产 preset 的增强设置未被此实验覆盖。
- batch1、CPU optimizer offload / DeepSpeed ZeRO2、bf16、原模型大小、相同学习率分组；不存在小模型替代。每组预算 18000 秒（包括周期评估），模型初始化额外计算；最后保存/评估可能略超时。
- 开始 / 每 500 optimizer updates / 最终评估，20 步去噪；预测结果不可覆盖。阶段报告与 action loss、MAE、显存、速度持续追加。
- 同 seed 不等于两路的 action noise 逐元素相同，因为联合分支还消耗 video noise RNG。比较固定评估 seed 的结果，同时提供相同步数和近似等时对照。
- 新方案没有 Stage2 视频预测，其图像生成栏明确 N/A；动作曲线/物理误差才是两组共有的评价。
- 多个人手数据的当前 native contract 仅有夹爪监督；缺失的 TCP 动作维度继续 mask。Genrobot 使用已审核的 finger-device TCP offset 接回原生 Stage2，未推断不存在的控制器夹爪 command。清单逐例列出有效维度和 mask 原因。

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false \
python scripts/compare_z_action_stage2.py \
  --config /mnt/data/cpfs/b5/yeqingda/openwam-0911/OpenWAM/outputs/z_action_stage2_comparison_20261027/comparison_config.yaml \
  --samples /mnt/data/cpfs/b5/yeqingda/openwam-0911/OpenWAM/outputs/z_action_stage2_comparison_20261027/data_v4/samples.pt \
  --output /mnt/data/cpfs/b5/yeqingda/openwam-0911/OpenWAM/outputs/z_action_stage2_comparison_20261027/run_5h_each \
  --seconds-per-arm 18000 --eval-every 500 --denoise-steps 20
```

Supervisor 顺序启动 joint、none 两个独立进程，失败立即写入 status 并停止，不把失败当完成。查看 `supervisor.json`、各组 `status.json` / `updates.jsonl`、`index.html`。两组结束后 HTML 才生成基于实测的结论；过拟合结果不等于未见数据泛化或真机成功率。

如接入既有 Stage1→Stage2 自动流水线，把流水线的 `stage2_config` 改为 `goal/stage2_z_action_3000h`；Stage1 的一轮覆盖与冻结规则不变。已有 Stage1 权重可直接启动上述 Stage2 命令，无需重新运行 Stage1。
