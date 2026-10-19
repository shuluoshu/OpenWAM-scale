# Stage 2 Patch Goal Adaptation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在已发布的联合去噪 Stage 1 分支上，接通冻结预测 goal 条件与原 action/video mutual 训练，支持 sample 和 3000h 的 task/subtask 混合数据，并交付经过验证的启动入口。

**Architecture:** 用独立冻结的 PatchGoalExpert 从当前 head 图像和当前语言生成目标图像 latent；80 个图像 memory token 通过可训练 GoalMemoryAdapter 提供给 future video/action。第一版 TCP patch 保留为 Stage 1 辅助输出和诊断，不增加到 Stage 2 memory；Stage 2 的视频 DiT 与冻结 goal DiT 必须是不同参数对象。

**Tech Stack:** PyTorch、Wan2.2、Hydra/OmegaConf、safetensors、PyArrow/PyAV、现有 torchrun + Accelerate/DeepSpeed、pytest、静态 HTML。

## Global Constraints

- 基线：远程 `fork/codex/stage1-joint-task-subtask`，commit `1aaab2a0d83bcfda51626c6681ffcb1bb6ad5484`。实施前核对远程新提交，禁止覆盖其他人工作。
- 本文件是实施计划；没有执行下述代码修改，也没有启动训练。
- 在隔离 worktree 中逐项移植本地 Stage B 原型，不复制整个脏工作区，不把未验证原型标成 ready。
- 不停止现有训练，不把旧 heatmap checkpoint 当作 patch checkpoint，不伪造 point_logits 兼容旧接口。
- 保留 gripper clip `[0,1]`、机器人 command/人类 openness 监督、异常状态过滤、观测增强、相对 joint、FK position/rotation loss。
- 沿用当前张量合同：action/state 112 维，32 个 action steps，33 个源帧，video stride=4，9 个视频帧，画布 384×320，其中 head 为 256×320。
- Stage 1 expert eval + requires_grad=False + no_grad；Stage 2 视频 DiT、action expert、proprio encoder、goal adapter 按参数组训练；VAE/T5 冻结。
- Stage 2 始终以预测 goal 为条件；未来图像、目标 TCP、成功标签和区间终点只用于监督/诊断，不能进入条件张量或缓存键。
- 有可用 subtask 的 episode 按 task/subtask=1:1 采样，否则 task-only；语言粒度须同时作用于 goal 分支和 Stage 2 prompt。
- 观测 anchor 限制在 YAML 选定区间。固定 action/video 输出形状不变，超出选定连续区间、语言区间或 episode 边界的监督逐时间步 mask；禁止跨越选定区间间隙。
- 每 500 个 optimizer updates 追加一次固定样例评估；保留 step 0 和结束时评估。不得覆盖之前的预测图、视频、点和误差。
- sample/3000h 均用按需读取、有界缓存；不能把所有帧、全部 episode reader 或所有目标 latent 一次性加载到内存/显存。
- full-state resume、checkpoint 搬迁、冻结依赖身份校验通过后才声明完整训练可用；只有 warm-start 时必须明确标为 warm-start。

## 已确认的现状与技术选择

| 项目 | 已确认状态 | 本计划选择 |
| --- | --- | --- |
| Stage 1 保存 | `patch_goal_bundle` / `openwam.goal.patch.v1` | 用 `load_patch_bundle`，按 metadata 校验 |
| 本地旧 Stage B loader | `load_goal_bundle`，只支持 `openwam.goal.v2` | 显式替换，不靠重命名文件 |
| 旧预测对象 | `point_logits` + 80 token mask | 新 runtime 返回 latent、UV、80 image mask |
| 新预测对象 | latent、UV、tcp_patch、81 token mask | 校验长度后取 image mask `[:, :80]` |
| GoalMemoryAdapter | 80 个空间图像 token | 复用该设计，TCP token 注入列为后续消融 |
| 旧数据配置 | 本地实验的固定 manifest | 新 Stage 2 reader 直接解析 curated YAML |
| Stage 1 数据合同 | 整个原始语言连续区间末帧为 goal | 继续沿用；不把区间末帧自动当成成功 |

选择 image-only memory 的原因：它与最初“最终态图像特征作为 condition”的设计一致，能单独验证联合训练是否改善 goal 表征；增加第 81 个 TCP memory token 会引入新的 adapter、mask 和对照变量，单独做消融更清楚。冻结 goal + 在线预测是第一版；离线预生成缓存可优化固定样例，但不替代在线部署路径；联合微调 goal 不属于本轮。

**图像约定：** 第一版 goal 分支读取同一 anchor 的原始 head，经 Stage 1 同款 BILINEAR resize 至 320×256，并在 Stage 2 观测增强前保留。Stage 2 的 L 型画布继续使用既有增强。这个双分支输入策略必须在配置、报告和推理代码中显式标记为 `goal.input_policy: clean_head`，不得悄悄从被 dropout 的画布恢复输入。需要增强 goal 分支时另设实验。GT 比较使用同一几何坐标系的未增强目标图像，不能把预测目标点叠到当前观测帧上。

**权重约定：** `training.finetune_ckpt_path` 是原 112 维 action/video baseline；`goal.checkpoint` 是训练完成的 Stage 1 patch bundle，两者不是同一个文件。不默认用 Stage 1 goal DiT 覆盖 Stage 2 视频 DiT。T5/VAE/tokenizer 的指纹必须与 Stage 1 产物一致，否则启动报错。

## Task 1: Patch checkpoint、预测接口与缓存

**Files**
- 移植并修改：`openwam/model/goal/runtime.py`
- 修改：`openwam/model/goal/types.py`
- 复用：`openwam/train/goal_patch.py`、`openwam/model/goal/patch_expert.py`
- 测试：`tests/goal/test_patch_stage2_runtime.py`

**Interfaces**
- `load_stage2_goal(path, *, vae, text_encoder, tokenizer, text_dim, device, dtype) -> PatchGoalExpert`：只接受 patch.v1 并验证冻结依赖。
- `GoalCondition(latent, image_token_valid, uv)`：latent `[B,48,1,16,20]`；image_token_valid bool `[B,80]`；uv FP32 `[B,2,2]`。
- `to_goal_condition(pred: PatchGoalPrediction) -> GoalCondition`：cache 和 runtime 仅依赖这个明确合同。

- [ ] 先写回归测试：真实 tiny patch bundle 保存/加载往返；旧 bundle 报错；VAE/T5 指纹不匹配报错；81→80 mask；FP32 UV；逐行 seed 在重排 batch 后结果一致。
- [ ] 测试中的 mask/类型断言：

```python
pred = PatchGoalPrediction(
    latent=torch.zeros(2,48,1,16,20),
    uv=torch.zeros(2,2,2,dtype=torch.float32),
    tcp_patch=torch.zeros(2,192),
    token_valid=torch.ones(2,81,dtype=torch.bool),
)
condition = to_goal_condition(pred)
assert condition.image_token_valid.shape == (2,80)
assert condition.uv.dtype == torch.float32
assert not condition.latent.requires_grad
```

- [ ] 运行 `python -m pytest tests/goal/test_patch_stage2_runtime.py -q`，确认旧 runtime 不满足新合同，再实现 loader、cache 新格式与转换。
- [ ] runtime 转换的核心代码：

```python
if pred.token_valid.shape != (pred.latent.shape[0],81):
    raise ValueError('patch goal requires 80 image tokens plus one TCP token')
return GoalCondition(pred.latent.detach(), pred.token_valid[:,:80].detach(),
                     pred.uv.detach().float())
```

- [ ] 缓存 key 包含实际输入 RGB、prompt/text context、seed、采样步数、shift、Stage 1 asset SHA、预处理版本；不包含未来标签。值为 detached CPU tensors，设置 entry/byte 上限；缺失 head 产生全 False image mask 并报告原因。
- [ ] 测试通过后提交 `feat: load frozen patch goals for Stage 2`。

## Task 2: Mutual goal memory 与参数冻结

**Files**
- 移植并修改：`openwam/model/goal/conditioning.py`、`openwam/train/goal_contract.py`
- 按最小 diff 移植：`openwam/model/architectures/base.py`、`openwam/model/architectures/dual_system/joint_self_attn.py`、`openwam/model/architectures/dual_system/mot_driver.py`、`openwam/model/video_backbone/wan_backbone.py`
- 测试：`tests/goal/test_patch_stage2_memory.py`

**Interfaces**
- `prepare_goal_inputs(...)` 只向 joint loop 提供 `goal_latent`、`goal_token_valid`（80）；UV 留给独立 diagnostics。
- `GoalMemoryAdapter.forward(latent, backbone) -> (keys, values)`；每层各追加 80 个 KV token。
- `build_g1_parameter_groups(architecture)`：每个 trainable 参数恰好一个 owner；goal expert 永远不进 optimizer。

- [ ] 写 tiny 双 expert 测试：future video/action queries 可读 goal KV，observed queries 不读；base mutual 区块逐元素保持一致；全部 goal mask=False 等价于关闭 goal memory。
- [ ] 移植旧 adapter 与 joint loop；保留空间位置编码和 latent.detach；不要把 goal 作为需要共同去噪的 future stream。
- [ ] 核心 mask 合同：

```python
assert goal_token_valid.dtype == torch.bool
assert goal_token_valid.shape[1] == 80
# existing observed-query rows are blocked only in appended goal columns
extended_mask = append_goal_mask(base_mask, goal_token_valid, observed_queries)
assert torch.equal(extended_mask[..., :base_mask.shape[-1]],
                   base_mask[None,None].expand_as(extended_mask[..., :base_mask.shape[-1]]))
```

- [ ] 真实 forward/backward + optimizer step 前后校验：goal 参数字节不变、无 grad；视频/action/adapter 指定参数有有限梯度且更新；goal DiT 与 video DiT 不共享 Parameter。
- [ ] 调用整模型 `.train()` 后再次检验 goal、VAE、T5 仍为 eval；冻结并不只是从 optimizer 排除。
- [ ] 运行 `python -m pytest tests/goal/test_patch_stage2_memory.py tests/model/test_fk_loss_integration.py -q`，通过后提交 `feat: condition mutual training on frozen goal memory`。

## Task 3: sample / 3000h task-subtask 窗口与 mask

**Files**
- 新建：`openwam/dataloader/curated_stage2.py`
- 提取共享的选择/语言元数据接口：`openwam/dataloader/curated_goal.py`
- 注册：`openwam/dataloader/registry.py`
- 最小修改：`openwam/dataloader/mora_relative_lerobot.py`
- 新建：`scripts/data/preflight_stage2.py`
- 测试：`tests/dataloader/test_curated_stage2.py`

**Interfaces**
- `CuratedStage2Dataset.from_config(config, split='train')`，只接受训练明确选择或独立评估选择。
- `sample_at(draw_id: int) -> dict`：由 `seed/draw_id` 确定 repo、episode、粒度、anchor；不生成全帧索引表。
- sample 提供原 action/video reader 合同，再附加 `goal_observed_head`、`granularity`、`instruction_start/end_exclusive`、源身份和 mask 原因；未来 goal GT 单独保存为 report target。
- 共享语言解析返回连续区间 `{start,end_exclusive,text}`；不得调用 Stage 1 的目标图像 `_materialize` 来生成每个 Stage 2 样本。

- [ ] 写合成源数据测试：两个 disjoint curated ranges、重复 subtask ID、缺失 subtask、冲突标注、末端短窗口、只有单侧 joint/EEF、相机内外参缺失、夹爪 command 缺失。
- [ ] 保留原 repo_weights、episode 选定时长权重、每 episode 1:1 粒度；没有 subtask 的只走 task。以 `(repo,episode,anchor,granularity,interval)` 为身份，允许同一 anchor 有两个语言例子。
- [ ] reader 仅加载当前 episode；读取既有字段映射、gripper command、stats 和 FK 配置。所需 joint 统计缺失则 preflight 给出 repo/字段并失败；不能把未归一化 joint 当作已归一化目标。未匹配 URDF 的样本 mask FK，不能借用其他机型链。
- [ ] 禁止旧 reader 的随机 retry 换 episode 后保留旧 prompt/goal；读取失败保留精确源身份并报错。
- [ ] 以源帧实际索引生成时间 mask，合并原字段 mask，不覆盖原失效状态。核心公式：

```python
last_source = min(selected_end_exclusive, instruction_end_exclusive, episode_length) - 1
# action[k] depends on next source state at anchor+k+1
valid_action_time = anchor + torch.arange(32) + 1 <= last_source
valid_video_time = anchor + torch.arange(9) * 4 <= last_source
sample['action_mask'] &= valid_action_time[:,None]
sample['video_mask'] &= valid_video_time
for arm in (sample.get('fk_supervision') or {}).get('arms',[]):
    arm['valid'] &= valid_action_time.cpu().numpy()
```

- [ ] 对视频 VAE 的时间压缩显式追踪 source-frame→latent-frame 有效性；不得把 9 帧 RGB mask 直接广播到不同长度 latent。增加边界测试，确认补齐帧对应的无效目标不会进入 video flow loss，且有效计数/归一化一致。
- [ ] 分别审计 `action_mask`（True=有效）、collate 后 `action_is_pad`（True=忽略）及实际 FK payload，测试补齐帧不会进入 loss。没有剩余有效 future step 的 anchor 从可选集合排除。
- [ ] `goal_observed_head` 从当前源 head 同帧、增强前 320×256 resize 取得；Stage 2 画布继续增强。导出原图/goal input/增强画布、帧号、尺寸变换、全部 GT 和 mask；保证无目标帧泄露。
- [ ] `preflight_stage2.py` 提供 `--config`、`--scope-only`、`--output-dir`；sample 完成每类至少一例固定导出；3000h 先做全选择 metadata 审计，再有界分层抽样，不宣称审计全部视频。
- [ ] 运行 `python -m pytest tests/dataloader/test_curated_stage2.py tests/dataloader/test_gripper_commands.py tests/dataloader/test_observation_augmentation.py -q`，通过后提交 `feat: stream curated task and subtask action windows`。

## Task 4: 配置、采样、优化器与启动入口

**Files**
- 新建：`configs/goal/stage2_joint_base.yaml`、`configs/goal/stage2_joint_sample.yaml`、`configs/goal/stage2_joint_3000h.yaml`
- 修改：`openwam/train/openwam_trainer.py`、`scripts/train.py`、`openwam/train/utils/ckpt_model_loader.py`
- 新建：`scripts/launch_stage2.sh`
- 测试：`tests/goal/test_stage2_config.py`、`tests/goal/test_stage2_sampler.py`

**Interfaces**
- Hydra entry 保持 `scripts/train.py --config-name goal/stage2_joint_sample`。
- `scripts/launch_stage2.sh sample|3000h --preflight|--train`，单机 `NPROC_PER_NODE`，GPU 数量不硬编码。
- 环境变量：`STAGE1_GOAL_CHECKPOINT`、`STAGE2_BASELINE_CHECKPOINT`、`STAGE2_SOURCE_YAML`、`STAGE2_OUTPUT_DIR`；缺任一必要路径则启动前报错。源码提供 sample/3000h YAML 默认值，部署新目录可覆写。

- [ ] 配置继承明确区分当前 Stage 1 config parser 与 Hydra；新增 Stage 2 配置使用 Hydra defaults，不直接复用 `base_config` 字段。
- [ ] 配置显式包含：`goal.enabled: true`、`goal.format: openwam.goal.patch.v1`、`goal.conditioning: image_memory`、`goal.input_policy: clean_head`、`num_steps: 20`、有界 cache。
- [ ] 保留 mutual / first_frame_causal、112 维合同、clip/command/filter/augmentation/FK；把历史 baseline_contract.json 的本机路径锁定替换成结构合同校验，仍禁止 action projection 暗中扩维/重置。
- [ ] 默认学习率沿用原分组：video/proprio `1e-5`、action `1e-4`、action encoder/decoder `1e-3`、goal adapter `1e-4`；配置与实际 optimizer groups 必须一致，不能只在 YAML 写值。
- [ ] 从 batch_size=1 的 smoke 开始测量显存；global batch 按 per-device × world-size × accumulation 计算，正式规模来自测量后的配置，不沿用旧 b320 名称猜测。
- [ ] 采样用全局 draw_id 而非各 rank 各自同 seed：先测试 2-rank CPU index 分配无同一 draw_id，完整样本自然重复属于加权有放回采样。epoch/offset/seed 可恢复；epoch 长度覆盖 `max_steps × accumulation` 所需 draws。
- [ ] DataLoader workers 不持有 CUDA，reader/cache 在 worker 内初始化；缓存预算乘 worker/rank 后仍有明确上限。训练循环不额外做第二次 rank shard。
- [ ] launcher 先做路径、bundle metadata、source YAML 和合同检查，再用 torchrun；单机 shell 核心形态如下，实际参数由选定 preset 生成：

```bash
exec torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-1}" \
  scripts/train.py --config-name "goal/stage2_joint_${preset}" \
  "goal.checkpoint=${STAGE1_GOAL_CHECKPOINT}" \
  "training.finetune_ckpt_path=${STAGE2_BASELINE_CHECKPOINT}" \
  "dataloader.source_yaml=${STAGE2_SOURCE_YAML}" \
  "project.output_dir=${STAGE2_OUTPUT_DIR}"
```

- [ ] `--preflight` 只运行 CPU 数据/合同检查；`--train` 才加载完整模型。配置解析测试不触发 GPU 初始化或训练。
- [ ] 运行 `python -m pytest tests/goal/test_stage2_config.py tests/goal/test_stage2_sampler.py -q`，通过后提交 `feat: configure sample and 3000h Stage 2 launches`。

## Task 5: 保存、恢复和推理一致性

**Files**
- 修改：`openwam/model/goal/runtime.py`、`openwam/model/checkpoint_utils.py`、`openwam/train/utils/checkpointing.py`、`openwam/deploy/model_loader.py`
- 测试：`tests/goal/test_patch_stage2_checkpoint.py`

- [ ] 将冻结 Stage 1 bundle 作为独立 `goal/expert.safetensors` 资产保存并写 SHA；Stage 2 unified weights 保存 baseline/adapter，冻结 goal 参数不重复写入 unified 权重。
- [ ] 用实际 `state_dict` key 集合测试旧 prototype 的注册行为；loader 与 saver 对 prefix 排除策略成对实现，不靠 `strict=False` 吞掉缺失 adapter。
- [ ] resume 同时恢复 Stage 2 模型、adapter、optimizer、scheduler（若启用）、RNG、sampler draw offset 与 global step；未保存 full states 的 checkpoint 只能 warm-start。
- [ ] 测试删除原 Stage 1 路径后，将 checkpoint 目录搬迁到 tmp 新目录仍能加载；损坏 bundle、缺 adapter、冻结依赖不一致明确失败。
- [ ] 固定 tiny inputs/seed，检查保存前后 prediction 和下一步更新等价；比较 optimizer state 和 sampler 下一批，而不是只检查文件存在。
- [ ] 部署复用相同 clean-head 预处理、PatchGoalExpert loader、80-token adapter 与预测 seed 策略；部署不读取训练 GT。
- [ ] 运行 `python -m pytest tests/goal/test_patch_stage2_checkpoint.py -q`，通过后提交 `feat: persist and resume patch-conditioned Stage 2`。

## Task 6: 每 500 步追加评估和可视化

**Files**
- 新建：`openwam/train/stage2_eval.py`、`scripts/eval_stage2.py`
- 修改：`openwam/train/openwam_trainer.py`
- 测试：`tests/goal/test_stage2_history.py`

**Interfaces**
- `evaluate_stage2(architecture, cases, output_dir, *, step, seed)`：所有 rank 在约定边界同步；按实际 DeepSpeed 包装方式处理 generation 所需 collectives，不能让 rank0 单独调用必须全 rank 参与的算子。
- `render_stage2_history(run_dir)`：仅消费现存 step 目录中的固定 schema metrics/assets，不重算预测。

- [ ] 固定每类至少一个 task，存在 subtask 的再固定一个；限制总量，冻结 anchor/prompt/seed。评估关闭观测增强，输入/GT 保存不可变资产。
- [ ] step 0、每 500 optimizer updates 和最终步写独立 `eval_step_XXXXXX/`；完整写完后原子发布 manifest，rank0 写 HTML，失败目录不能冒充完成报告。
- [ ] 每例展示：实际 Stage 1 head input、Stage 2 观测画布、末态 GT、预测 goal + UV、未来 video GT/预测并排视频、action GT/预测轨迹、各通道有效 mask。
- [ ] 数值包含 goal image MAE、TCP 平均欧氏像素误差、future video MAE（有效帧/视角）、归一化 action MAE、物理 joint/gripper 误差、FK position/rotation 误差；分别报告 sample count/valid count、family 和 granularity，缺标签用 unavailable，不记零。
- [ ] 测试连续写 step 0/500/1000，原 step 500 图片哈希不变，所有 step 链接可访问；随机种子固定且只统计有效目标。
- [ ] 日志输出 `report_ready` 携带 global step 和 HTML 路径，供已有监控消费。日志事件不等于已安装桌面通知自动化，不新增未请求的定时任务。
- [ ] 运行 `python -m pytest tests/goal/test_stage2_history.py -q`，通过后提交 `feat: append Stage 2 prediction and error reports`。

## Task 7: 分级验收、文档与发布

**Files**
- 新建：`docs/stage2_joint_training.md`、`scripts/verify_stage2.py`
- 测试：上述新测试 + 现有 `tests/model/test_fk_supervision.py`、`tests/model/test_fk_loss_integration.py`、`tests/dataloader/test_mora_relative_reader.py`、Stage 1 全部 focused tests。

- [ ] CPU：checkpoint 接口、80-token mask、无 future leakage、梯度所有权、时间 mask、2-rank draw 分配、保存恢复等测试全部通过；Stage 1 回归无退化。
- [ ] 真实数据：sample 全选择元数据检查，每类固定样例人工可检查的输入/GT HTML；3000h 全选择 metadata + 有界真实读取。分别列出缺标注、缺几何、缺 stats、缺 URDF，而非统一归为“缺 TCP”。
- [ ] 单 GPU：真实 Stage 1 bundle + baseline + sample 进行 2–5 optimizer steps，验证所有训练模块更新、goal 参数不变、FK valid count 合理、loss/梯度有限；记录 peak memory 和每步时间。
- [ ] 双 GPU：短 smoke 检查采样、冻结 expert 放置、DeepSpeed optimizer、评估 barrier、保存/恢复无 hang；只有通过后才声明分布式入口 ready。
- [ ] 固定每类片段的 500-update 诊断：验证 step 0/500 追加 HTML，观察 action/video loss 与轨迹。单个 case loss 非严格单调不能单独判失败，检查监督有效数量和可解释学习趋势。
- [ ] 启动 500 步或更长验证前，按当时资源与用户授权决定执行预算；本计划本身不启动这些任务。未执行的 GPU 或规模验证在文档中明确列出。
- [ ] 文档写明两个 checkpoint 的用途、两个 YAML、启动与 resume 命令、global batch 算法、增强策略、mask、评估范围和已完成的验证等级。
- [ ] 在隔离分支 review 全部 diff，提交至 `shuluoshu/OpenWAM-scale` 的同一功能分支，核对 remote SHA/tree；Stage 2 尚未到达验收等级时不把原 Stage 1 发布标成 Stage 2 ready。

## 最终启动接口（计划交付，当前尚不存在）

```bash
export STAGE1_GOAL_CHECKPOINT=/absolute/path/to/stage1/patch_goal.safetensors
export STAGE2_BASELINE_CHECKPOINT=/absolute/path/to/baseline/checkpoint.safetensors
export STAGE2_SOURCE_YAML=/absolute/path/to/pretrain_v0_3_curated_smoke.yaml
export STAGE2_OUTPUT_DIR=/absolute/path/to/new_stage2_run

bash scripts/launch_stage2.sh sample --preflight
NPROC_PER_NODE=1 bash scripts/launch_stage2.sh sample --train
```

3000h 切换 `STAGE2_SOURCE_YAML` 到 `pretrain_v0_3_curated_3000h.yaml` 并使用 `3000h` preset。上述路径是用户提供训练资产的位置参数，不是仓库自带权重；入口交付前不能执行这些命令。

## Ready 定义

只有“当前 patch checkpoint 可加载 → 独立冻结 goal → mutual 前后向 → 同语言 task/subtask 时序监督 → 合法 mask/FK → 真实 GPU 更新 → 保存恢复 → 500-step 历史报告”整条链完成，才称 Stage 2 ready。单元测试、CPU smoke、GPU smoke、500-step 诊断、3000h 正式运行分别记录，不相互替代。
