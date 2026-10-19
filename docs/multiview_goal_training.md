# Stage1 三视角观测，头部 goal / TCP 输出

本版本只扩大 Stage1 的当前观测信息：头部、左腕、右腕同一源帧组成 H384×W320 输入；仍联合去噪生成 H256×W320 的头部目标图像和有序 `[uL,vL,uR,vR]` TCP 坐标。目标坐标始终属于头部目标帧，不是当前帧，也不是三视角画布。

## 模型与输入合同

- 共用冻结 Wan VAE 权重。观测 latent `[B,48,1,24,20]` 分成120个 patch；目标 latent `[B,48,1,16,20]` 分成80个 patch，TCP占1个 patch。观测和目标分别 patchify，使用明确的空间位置和 observation/goal/TCP 时间索引。
- 三视角输入对应 head H256×W320、左右腕各 H128×W160。缺相机不推断内容：缺失槽在 VAE 前置为黑色，`observed_view_valid` 是 bool `[B,3]`，缺失视角的 token 不作为 attention key。当前观测不能读取目标噪声 token；目标图像与 TCP 互相读取。
- Stage2 接收增强前的同源三视角观测供冻结 goal expert 使用；预测出的80个头部图像 memory token继续通过 GoalMemoryAdapter 接入 future video/action。TCP仍是 Stage1 辅助预测和诊断，未额外注入 Stage2 memory。
- 共享 VAE 权重不等于复用同一次编码。Stage1 读取干净观测，Stage2 的观测可以增强，因此本版仍分别编码。没有为节约编码次数改变增强行为。
- 只有头部的数据仍可用，两个腕部 mask 为 false；缺失或无效 TCP 标签的 loss mask 逻辑保持原样。相机存在但视频/时间元数据损坏会报错，不静默当作缺视角。

## 旧 checkpoint 与新配置

旧 head-only 配置及 `openwam.goal.patch.v1` bundle 保持兼容。新 multiview bundle 使用 `openwam.goal.patch.v2`，记录输入布局和 mask 合同；输出 codec 不变。普通加载严格遵从 bundle，不能只通过更改配置把旧 checkpoint 当作已训练的三视角模型。

`--init-goal-checkpoint` 支持显式从已有 v1 Stage1 权重初始化三视角 Stage1，校验冻结 VAE/T5/tokenizer 依赖，使用新的 optimizer。已有的真实5000步 bundle 仍是 v1 head-only，只能作为显式初始化来源；三视角条件是否有效需要新的训练和对照验证。本次未执行新的5000步训练。

在仓库根目录先检查 sample 的实际输入：

```bash
/root/miniconda3/envs/openwam/bin/python scripts/train_goal_curated.py \
  --preflight --config configs/goal/stage1_multiview_sample.yaml \
  --source-yaml /absolute/path/pretrain_v0_3_curated_smoke.yaml \
  --output-dir /absolute/path/new_multiview_preflight
```

显式启动新的 Stage1 训练：

```bash
/root/miniconda3/envs/openwam/bin/python scripts/train_goal_curated.py \
  --train --config configs/goal/stage1_multiview_sample.yaml \
  --source-yaml /absolute/path/pretrain_v0_3_curated_smoke.yaml \
  --init-goal-checkpoint /absolute/path/old_stage1/patch_goal.safetensors \
  --output-dir /absolute/path/new_multiview_stage1 --device cuda
```

依赖仍由配置中的 baseline/components/tokenizer 提供，需设置正确的 `OPENWAM_BASELINE_CHECKPOINT` 或配置路径。3000h使用 `configs/goal/stage1_multiview_3000h.yaml` 和对应 source YAML。task/subtask 混合策略不变，默认每500步追加评估，保留历史。该入口仍是单进程流式训练。

Stage1 新训练产出 v2 bundle 后，Stage2使用 `configs/goal/stage2_multiview_sample.yaml` 或 `stage2_multiview_3000h.yaml`。启动器设置 `STAGE2_GOAL_INPUT=multiview`；默认 head 模式继续服务旧 checkpoint。其余 Stage2 baseline、normalization registry、source YAML、输出目录和 PYTHON 设置见 [Stage2 启动说明](stage2_joint_training.md)。

## Stage2 输入、启动与恢复

三视角 Stage2 必须使用实际训练后保存的 v2 bundle，并同时设置 `goal.format=openwam.goal.patch.v2`、`goal.input_policy=clean_multiview`；新 preset 还显式记录 `goal.observation_mode=multiview`。运行时会核对实际加载 expert 的模式，CPU 预检也核对 bundle 元数据。旧 v1 / `clean_head` 配置继续兼容。恢复或从已有 Stage2 checkpoint 开始时，所选模式必须与已保存配置一致；不能在恢复时切换观测模式。

先设置新路径和已存在的统计 registry，然后执行预检。以下路径是需要替换的占位符：

```bash
export PYTHON=/root/miniconda3/envs/openwam/bin/python
export STAGE2_GOAL_INPUT=multiview
export STAGE1_GOAL_CHECKPOINT=/absolute/path/trained_multiview_stage1/patch_goal.safetensors
export STAGE2_BASELINE_CHECKPOINT=/absolute/path/baseline
export STAGE2_SOURCE_YAML=/absolute/path/pretrain_v0_3_curated_smoke.yaml
export STAGE2_STATS_REGISTRY=/absolute/path/native_statistics_registry.json
export STAGE2_OUTPUT_DIR=/absolute/path/new_multiview_stage2
bash scripts/launch_stage2.sh sample --preflight
```

确认输入和资源预算后，用 `bash scripts/launch_stage2.sh sample --train` 显式启动；3000h 将 `sample` 换为 `3000h`，并设置相应 source YAML。启动器默认 `STAGE2_GOAL_INPUT=head`，仅接受 `head` 或 `multiview`。这些命令为后续训练用法，本次没有启动完整训练。

原生 Stage2 reader 总是提供两个独立的干净输入：旧的 `goal_observed_head` 与新的 `goal_observed_multiview`，后者是增强前 `[3,1,384,320]` 画布，配套 bool `[3]` `goal_view_valid`。训练根据实际 expert 模式选择。`first_frame_image` 可以增强，不能从它裁剪或反推出 goal expert 的干净输入。缺头部行仍跳过 goal expert，全部80个 memory token无效；即使腕部存在也遵循该兼容策略。mask 的 head 位必须与 `goal_head_valid` 一致。

公开生成接口的 v2 调用示意：

```python
result = architecture.generate(
    schedule, prompt,
    goal_observed_multiview=clean_canvas,  # [B,3,1,384,320], RGB [-1,1]
    goal_view_valid=view_valid,           # bool [B,3], head/left/right
    goal_head_valid=True,
    goal_seed=17,
    # 其余 video/action/proprio 条件与既有调用相同
)
```

当前公开生成仍是单条样本接口，传入画布批次 B=1。v1 继续使用 `goal_observed_head`；v2 不回退到旧头部输入。缺失视角在 VAE 前置黑，其 mask 和预处理版本参与缓存身份；未来帧、目标图像和目标 UV 不进入该路径。

固定评估使用相同生成接口。报告记录实际 mode、mask、输入 shape/dtype；`fixed_inputs/case_*/supervision.npz` 保存清洗且转为实际模型 dtype 后的 `goal_observed_multiview`、`goal_view_valid` 和120个 observation token有效位。BF16 输入同时保存可读 float32 数值和 `goal_observed_multiview_bfloat16_bits` 的 uint16 原始位值。旧版 head 评估的条件 SHA 不受未使用的新数据字段影响。新版报告绑定实际 mode 和输入 mask，不能复用不同模式的旧评估结果。

## 验证范围

新增功能的测试、真实输入预检与运行证据随交付报告记录。合成小模型验证只证明前后向、mask和接口链路，不证明大模型三视角条件的学习效果。新配置的默认迭代次数不是已经完成的训练次数。

截至本次实现验证：隔离工作树集成回归485项通过（2条既有 NumPy 弃用警告）。真实 sample 预检覆盖12类数据、18个 task/subtask 样例；独立重解码 LightWheel、ABC130k、interndata_a1、RoboMIND v1 四类的当前源帧，Stage1 画布与 Stage2 增强前画布在 BF16 下逐值一致。这是输入对齐验证，不代表重新审核了全部 TCP 几何标定，也不是完整3000小时扫描。

最终审查修复了显式初始化的 flow shift 覆盖：新配置通过规范构造器同时重建训练 sigma/timestep 网格与归一化 loss 权重，生成和保存元数据使用相同设置；初始化仍复用已加载权重且保留来源记录。该修复在实际工作树的32项受影响测试中通过，包含不同 source/config shift 的前后向、生成和保存重载一致性。
