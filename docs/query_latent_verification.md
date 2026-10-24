# Query latent 实现验证记录

日期：2026-10-20（Asia/Shanghai）。基线 `2490537`，分支 `codex/query-latent-goal`。

## 已执行

- 旧 multiview/memory 基线：30 tests passed。
- 新 query 专项：26 tests passed。覆盖 query 对当前图像/语言的依赖、
  invalid-view/text padding 不变性、目标信息隔离、两组 query 梯度、
  invalid TCP NaN masking、只加载 encoder、完整辅助预测保存重载。
- 新 Stage2：真实 tiny Wan/action forward/backward、两组 memory 对 action/video
  的影响、当前观测隔离、decoder-free runtime、公共 generate、可搬迁资产、
  严格 train/deploy loader、配置不匹配拒绝、没有 decoder 的评估报告。
- CPU 真实 Accelerate full-state：query 模式保存/搬迁/重建后，下一 batch、
  loss、参数、optimizer、scheduler 与 RNG 精确匹配连续执行。
- 最终集成：**380 passed, 2 skipped, 4 warnings，37.07秒**。
  两项跳过为既有、需要 `OPENWAM_RUN_TRAIN_DEPLOY_GPU=1` 的 GPU 集成变体；
  warnings 为受限测试环境的 CUDA/NVML 初始化警告。
- 合成 Stage1 CPU smoke：真实小型 Wan blocks，2次 optimizer updates，
  step0/step2 图像与 TCP 联合生成、评估和 query bundle 保存成功。
- 同一小型结构的 **CUDA BF16 smoke：2次 updates 成功**，含 step0/step2
  辅助生成与 checkpoint 保存；没有停止任何已有进程。
- `git diff --check`、启动脚本 `bash -n` 和新增入口 Python 编译检查通过。
- 独立代码审查完成，最终无剩余 actionable findings。

最终集成命令：

```bash
/root/miniconda3/envs/openwam/bin/python -m pytest \
  tests/goal tests/dataloader/test_curated_stage2.py \
  tests/model/test_stage2_video_masks.py \
  tests/test_train_deploy_consistency.py tests/test_deployment_changes.py \
  tests/test_save_deploy_assets_hook.py tests/test_checkpointing.py -q
```

GPU smoke 的模型为 `N1=8,N2=2,D=32,heads=4,layers=2,ffn=64`，合成数据、
TinyVAE、2步辅助采样；不是正式配置 `N1=64,N2=2,D=512` 或完整 Wan。
该 smoke 的数秒运行时间不能用于推断正式训练速度。

## 产物

原始工作区的 `outputs/query_latent_20261020/`：

- `final_regression.log`：最终集成测试输出。
- `smoke_cpu/`：CPU 合成训练 summary、progress、checkpoint 和评估。
- `smoke_cuda_bf16/`：GPU BF16 合成训练同类记录。
- `code/`：完整隔离 checkout、配置、测试和操作手册。

## 尚未执行

- 未启动真实数据5000步或3000h全量训练，也未测量其收敛、成功率、显存峰值或吞吐。
- 未执行新 query 模式的大模型 Stage2 DeepSpeed / 多GPU完整恢复；CPU
  tiny 恢复证据不能替代该检查。
- 没有重做3000h全量数据审计；reader、投影转换、目标选例和标签 mask 未改。
- 默认 token 数量与分组辅助 loss 权重尚需消融。

操作命令、两种结构图、模型超参与数据切换方式见 [操作手册](query_latent_training.md)。
