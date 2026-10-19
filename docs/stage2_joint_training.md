# Stage 2 联合训练：接口、验证等级与短 smoke

截至 2026-10-19，Stage 2 代码集成完成，最终回归 **450 passed**（2 个已有 NumPy deprecation warnings）。验证范围如下；这不是大规模训练收敛声明。

| 验证项 | 实际结果 | 边界 |
| --- | --- | --- |
| 最终 v4 原生数据与启动预检 | sample / 3000h 均通过；20 + 83 个归一化合同已重新准备 | 有界数组统计，不是全量视频逐帧审计 |
| 最终 v4 单 GPU / DeepSpeed fresh | 两次真实 optimizer updates、完整 state 1/2 保存、按计划退出；checks passed | 未执行最终 v4 的完整 GPU resume |
| 最终 v4 checkpoint 独立推理 | 独立进程 exit 0，step 2 complete manifest 与 26 个本地引用通过检查；两段 H.264 视频各 9 帧可解码 | 仅两个 diagnostic cases，不代表全部数据集质量 |
| 历史 v3 独立进程 resume | 实际从 update 2 恢复至 3，完成最终评估与完整保存 | 旧探针因合法 state dropout 后 proprio 零梯度报错；原始失败 JSON 保留，另附重放判定，不作为 v4 resume 验收 |
| CPU 完整恢复与迁移 | next-batch / next-update 精确等价，资产与监督合同校验通过 | 不证明 GPU bitwise 等价 |
| 多 GPU / 500-update 学习趋势 | 未执行 | 分布式与规模训练仍待验收 |

最终 v4 fresh 的 total loss 为 **0.639277 → 0.354449**，第二步 video/action loss 分别为 **0.351003 / 0.003446**。两个不同样本的 loss 不能作为学习趋势。第一步实际 FK valid count 为 18。video patch、action transformer、proprio、goal adapter 的代表参数均记录有限非零梯度和参数更新；全部约 5B goal 参数冻结、无梯度、版本不变且不在 optimizer 内。CUDA 峰值 allocated / reserved 约 **34.56 / 36.55 GiB**，训练计算分别约 **32.25 / 27.00 秒**；含模型加载、评估及完整状态保存的 fresh 总墙钟约 **19.4 分钟**。

证据在原始工作区 `outputs/stage2_implementation_20261019/`：`final_regression.log`、`final_review.md`、`gpu_smoke_v4/fresh.json`、`gpu_smoke_v4_eval.log`、`resume_proprio_adjudication.json` 和汇总 `index.html`。最终 v4 run 为 `gpu_smoke_v4/checkpoints/2026-10-19_21-19-13/`；历史 v3 失败探针记录保持原样。

## 两种 checkpoint 和两份配置

- Stage 1 `patch_goal.safetensors` 是独立的 `openwam.goal.patch.v1` 冻结目标专家，含真实图像 patch 与连续 TCP patch 的联合生成权重和冻结依赖身份。它不是旧 heatmap checkpoint；不能伪造 `point_logits` 接口。`goal.checkpoint` 指向此 bundle。
- Stage 2 baseline/run 提供 video/action/proprio 初始权重与模型重建资产。第一次用 `training.finetune_ckpt_path` 是 **warm-start**，optimizer/RNG/计数从零开始。后续 Stage 2 自包含 run 的 `checkpoint_step_N.safetensors` 是推理/评估权重，`accel_state_step_N/` 是与该权重 SHA 配对的完整训练状态。完整 resume 需要两者以及完成标记、原生 normalization/selection 资产、打包的独立 goal bundle。
- `configs/goal/stage2_joint_sample.yaml` 默认 5000 optimizer updates，使用 sample selection；`configs/goal/stage2_joint_3000h.yaml` 默认 100000 updates，使用 3000h selection。它们共享 `stage2_joint_base.yaml` 的合同；默认步数不是执行授权。先做 CPU preflight 和有预算的短 smoke。

`max_steps` 始终是**成功 optimizer updates**。Global batch = 每 GPU `training.batch_size` × world size × `gradient_accumulation_steps`。例如单 GPU batch1/accum1 的 global batch=1，双 GPU batch1/accum2 的 global batch=4。全局 draw 流只由 Accelerate 分片一次；checkpoint 记录实际消费的 draw，而非 DataLoader 预取游标。完整 resume 必须保持 max_steps/draw horizon、拓扑、seed、优化器/调度器和监督合同不变。

## 数据、冻结与监督合同

固定 action/state=112、action steps=32、源帧=33、video stride=4、video frames=9，画布 384×320，head 256×320。机器人 gripper 使用 command，人类使用 openness，归一化 clip `[0,1]`，保留异常状态过滤、相对 joint 和真实 FK position/rotation 监督。缺标注、缺几何、缺 stats、缺 URDF 分开记录，不能统称缺 TCP。

有 subtask 的 episode 按 task/subtask=1:1 抽样，否则 task-only；同一语言粒度同时供 goal 分支与 Stage 2 prompt。Anchor 受 YAML 选中连续区间约束。输出形状不变，越过连续区间、语言区间、episode 边界的 action/video 监督逐时间步 mask，不跨选中区间间隙。采样与真实读取使用有界 episode/byte LRU，不能把全部 reader、帧或目标 latent 一次性常驻。

Stage 1 expert 保持 eval、`requires_grad=False`、no_grad；VAE/T5 冻结。训练所有权组为 video（含 proprio encoder）、action transformer、action projection 和 goal adapter，学习率分别为 1e-5、1e-4、1e-3、1e-4。只读取预测 goal 的 80-token image memory；adapter 不注册冻结 backbone 的别名。首帧观测 query 不读 goal memory；video 使用 first-frame-causal、video/action 使用 mutual mask。

Stage 2 永远以**预测 goal**为条件。未来像素、GT action/TCP、成功标签、区间终点只用于监督或报告，不进入条件张量/goal 缓存键。Stage 1 使用增强前 clean head；Stage 2 observation 支持 color jitter、patch mask、view dropout（保留至少一视角）、state dropout。固定评估使用独立、关闭增强的数据集，不能改变训练增强配置。

## 原生数据审计边界

原始工作区 `outputs/stage2_implementation_20261019/` 保留证据：

- `preflight_sample_v4/summary.json`：46 repos / 46 selected episodes 的完整选择 metadata 检查，20 normalization contracts，6900 selected frames；每类/每合同有界真实读取得到可人工检查的输入/GT HTML。`all_videos_audited=false`，不声称逐帧全审计。
- `preflight_3000h_v4/summary.json`：5750 repos / 168467 episodes、323988246 selected frames 的完整选择 metadata 检查，83 normalization contracts；没有导出全部视频。有界统计读取 165 个 episode / 1320 个 window，不能据 metadata 全通过推断全部视频可解码。
- 最终 `stored_mask_policy=declared_eef_blocks_all_flags_finite_v2` 对应 v4 registry；sample 有界读取 32 个 episode / 256 个 window。`statistics_v3_v4_comparison.json` 记录同一 population 的全部 20 + 83 个统计 entry（包括 counts/quantiles）完全一致。这只证明这些有界样本未受该修正影响；不能据此复用旧 contract。旧 registry 和旧 run 的 normalization/resume contract 会被严格拒绝。
- CPU full-state 测试包括真实 CPU Accelerator/optimizer/scheduler/RNG/DataLoader 的 next-update 精确等价与 checkpoint 搬迁、冻结依赖身份校验。它们不证明 GPU bitwise 等价或 DeepSpeed shard 恢复。

## 普通启动与完整恢复

设置真实路径后，从隔离 checkout 启动。`launch_stage2.sh` 先跑 CPU metadata/资产 preflight，再按指定模式决定是否训练：

```bash
export STAGE1_GOAL_CHECKPOINT=/absolute/path/patch_goal.safetensors
export STAGE2_BASELINE_CHECKPOINT=/absolute/path/baseline/checkpoint_step_1000.safetensors
export STAGE2_STATS_REGISTRY=/absolute/path/sample/statistics/registry-SHA.json
export STAGE2_OUTPUT_DIR=/absolute/path/new_stage2_run
export PYTHON=/root/miniconda3/envs/openwam/bin/python
bash scripts/launch_stage2.sh sample --preflight
# 仅在当前预算允许后使用 --train；3000h 换 preset 和对应 registry。
bash scripts/launch_stage2.sh sample --train training.max_steps=3
```

完整 resume 直接使用同一 YAML 与原 run，明确清空 warm-start：

```bash
/root/miniconda3/envs/openwam/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=1 scripts/train.py \
  --config-name goal/stage2_joint_sample \
  training.finetune_ckpt_path=null training.resume_ckpt_path=/absolute/path/RUN \
  training.max_steps=3
```

实际恢复需保留原训练的全部有效 override（seed、batch、保存设置、评估范围等），下述 resolver 会给短 smoke 的完整列表。恢复时从打包资产解析 goal/normalization/selection，严格校验 scheduler、优化器状态和成对权重。缺完整状态只能 warm-start，不能称 full resume。原训练 data_root 身份需可用；checkpoint 搬迁不自动授权改变训练数据。

## 有界真实 GPU 验证：update 2 完整保存后退出，fresh process 恢复到 3

`scripts/verify_stage2.py` 包装真实 `scripts/train.py` Hydra → Accelerate/DeepSpeed → `OpenWAMTrainer.train_stage2`。它不实现替代训练循环、不替换 optimizer、不改监督或 sampler。固定 max_steps=3、seed=1、batch1、accum1、save1/keep3，评估只选显式 diagnostic 的两例、20 joint denoise steps。wrapper 在模型构造前拒绝改变这些预算合同；不能通过它启动 500 updates。

先只生成命令计划（不加载模型）：

```bash
/root/miniconda3/envs/openwam/bin/python scripts/verify_stage2.py \
  --resolve-assets /mnt/data/cpfs/b5/yeqingda/openwam-0911/OpenWAM/outputs/stage2_implementation_20261019 \
  --output-dir /mnt/data/cpfs/b5/yeqingda/openwam-0911/OpenWAM/outputs/stage2_implementation_20261019/gpu_smoke_v4 \
  > /tmp/stage2-smoke-plan.json
```

计划读取 `stage1_asset_contract.json` 和 `preflight_sample_v4/summary.json`（缺失 v4 时直接失败，不回退到已失效的 v3 registry），输出 `preflight_command`、`fresh_command`、`resume_command_template` 的精确 argv，避免手工拼错 registry/权重。依次在 checkout 工作目录执行：

```bash
# CPU preflight，确认 global_batch=1；不构造完整模型。
/root/miniconda3/envs/openwam/bin/python -c 'import json,subprocess; p=json.load(open("/tmp/stage2-smoke-plan.json")); subprocess.run(p["preflight_command"],check=True)'
# 真实 torchrun/DeepSpeed：update1/2，原始 save_full_state 成功返回后有意退出。
/root/miniconda3/envs/openwam/bin/python -c 'import json,subprocess; p=json.load(open("/tmp/stage2-smoke-plan.json")); subprocess.run(p["fresh_command"],check=True)'
# 独立进程：读取 fresh.json 的真实 run_dir，严格恢复 update2，执行 update3/最终评估/保存。
/root/miniconda3/envs/openwam/bin/python -c 'import json,subprocess; p=json.load(open("/tmp/stage2-smoke-plan.json")); f=json.load(open(p["fresh_command"][p["fresh_command"].index("--report")+1])); assert f["status"]=="checkpoint_stop" and f["checks"]["passed"]; c=[x.replace("training.resume_ckpt_path=RUN_DIR","training.resume_ckpt_path="+f["run_dir"]) for x in p["resume_command_template"]]; subprocess.run(c,check=True)'
```

fresh 与 resume 的 `max_steps` 都是 3。首进程只在原完整保存返回、single-rank barrier 返回、paired weights/完成标记存在后抛内部 intentional stop，JSON `status=checkpoint_stop`；这不是训练自然完成。第二进程 `status=completed` 才表示这段实际恢复完成。不会发 SIGKILL，也不会停止任何已有训练。

seed=1 是显式诊断选择，新 smoke 走 sample YAML/v4 registry 的原生 weighted sampler。此前 v3 记录的 draw0 为 Genie facial_cream episode8 anchor225（18 个 valid FK arm/time pairs），draw1 为 egodex throw_collect_objects episode1891 anchor471，draw2 为 A1Lift2 toothbrush episode472 anchor228。实际训练 JSON 的 `batches.sources` 与 `fk_valid_count` 才是运行证据；没有非零实际 valid count 就不能声称测试了 FK 梯度分支。

JSON 每 update 原子更新，正常/异常退出都会写：实际 backend/device、DeepSpeed flatten 前参数所有权、video patch embedding / action transformer matrix / proprio / goal adapter 整张代表 tensor 的 hash/变更数量、backward hook 有限非零梯度和 norm、全部 teacher 的 requires_grad/gradNone/版本计数、代表 teacher hash、实际 source trace/video 路径、FK valid count、loss、峰值 CUDA allocated/reserved、host peak RSS 与墙钟。`training_update_wall_seconds` 从 compute_loss 前计到 optimizer/zero_grad 后；`interval_wall_seconds` 还可能包含前一次保存/评估开销。hook 同步与 CPU snapshots 会影响速度，报告不等同未加探针的吞吐基准。

GPU JSON 会区分 failure 和完成，Python 异常保留 traceback；无法在进程被 kill/OOM-killer/主机丢失后保证最终 JSON。没有与未运行的连续 GPU baseline 做 bitwise 比较。恢复 snapshot 在 load_full_state 后重新采集，避免把参数恢复写入误算为更新。GPU 探针覆盖四个代表模块（video patch embedding、action transformer、proprio、goal adapter），其中 video/proprio 同属 video optimizer group；独立 action encoder/decoder projection group 没有实际 GPU 梯度/更新探针。参数所有权与 tiny 实际 transformer 测试支持 projection 梯度/更新，不能代替该 group 的真实 GPU 证据。这里只验证代表参数和全部 teacher 冻结条件。新版探针记录每例实际 observed proprio mask 的有效维数；仅全部明确为 inactive 时将 proprio 非零梯度/更新检查标为 N/A，并继续检查参数/梯度有限。active 或 mask 缺失/形状不符时仍要求有限非零梯度和整张代表 tensor 更新，augmentation trace 本身不构成豁免依据。统一权重约 **23.286 GiB**，包含冻结 teacher 的完整训练状态约 **122.159 GiB**，保存/恢复的存储与 I/O 成本较高。

## 固定评估、历史和剩余验收

普通评估保留 step0、每500成功 updates 与最终 step；resume 已存在的匹配 step 校验后复用。每个 `eval_step_XXXXXX/` 不可变，含输入/GT、预测图/视频、112 通道曲线、goal TCP、物理 joint/gripper/FK 误差与实际 valid count；`stage2_history.html` 链接历次完整报告。缺标签显示 unavailable/count0，不制造零误差。task 报告 GT 为真实 episode-end，subtask 为 instruction-end；这些终点只用于报告。

默认固定每个选中 family 的 task 及可找到的 subtask；cap 不够时正常模式拒绝静默截断。两例 smoke 必须标 diagnostic，不能当作每类全面覆盖。生成使用与训练一致的 predicted goal，固定 seed；保留/恢复 RNG、模块 train/eval 状态和调度器状态。

```bash
# 已保存权重的独立评估，不训练。
/root/miniconda3/envs/openwam/bin/python scripts/eval_stage2.py --run-dir RUN --steps 500 1000 --device cuda
# 仅刷新已有报告历史，不加载模型或读取源视频。
/root/miniconda3/envs/openwam/bin/python scripts/eval_stage2.py --run-dir RUN --history-only
```

当前剩余：最终 v4 完整 GPU resume、双 GPU 采样/teacher placement/optimizer/评估 barrier/保存恢复防 hang，以及有明确预算后的固定每类 500-update 趋势验证。历史 v3 已完成 backend continuation，不能替代最终策略的完整恢复验收。单 case loss 不严格单调不足以判失败，应结合有效监督数量与动作/视频趋势判断。分布式入口、规模训练和学习质量保持未验收状态。
