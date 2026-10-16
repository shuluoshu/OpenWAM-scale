# Human gripper supervision implementation plan

> Execute inline. The user approved adding the audited LightWheel, Mecka, EgoDex and EgoVerse openness labels to gripper supervision. This amends the October 13 controller-only rule; another design approval is unnecessary.

**Goal:** Restore the existing human openness targets to the gripper loss without changing measured state inputs, robot commands, joint/TCP targets, or state-spike rejection.

**Architecture:** Extend the existing provenance-backed gripper source mapping with an explicit `human_openness_target` kind. Recognize the four audited human schemas and read the gripper components of stored `actions.eef` at the current action row; retain the source's future-target timing. Robot controller commands take priority. Human target validity enables loss but is excluded from controller-command evidence used by the spike filter. Rebuild fixtures/statistics in a new directory because old explicit empty mappings and cached model inputs cannot hot-reload this change.

**Tech stack:** Python, NumPy, PyArrow, PyTorch, pytest, existing Mora relative94/112 reader.

## Constraints

- Gripper input remains measured state[t], clipped to [0,1]; target is absolute openness in [0,1].
- Human target uses actions.eef[t+k,9/19], without an additional shift. Audited LightWheel/Mecka store future state+1; EgoDex/EgoVerse store future state+3. Do not infer timing from generic action because LightWheel differs.
- Missing/nonfinite targets stay masked. Explicit empty mappings continue to disable supervision. Declared action.eef_target_valid, when available, must also be honored.
- Unknown sources and robot state-derived actions remain unrecognized; pose masks, joint state+1, 32-step horizon, and FK stay unchanged.
- Preserve spike rejection by distinguishing human proxy labels from independent controller commands.
- Preserve old fixtures, statistics, TensorBoard curves and running experiment. This request changes supervision; it does not explicitly request another training restart.

## Task 1: Source selection and representation

Files: openwam/dataloader/utils/gripper.py, openwam/dataloader/utils/mora_relative.py, tests/dataloader/test_gripper_commands.py.

- [x] Add failing regression tests for all four audited human schemas, source target vs input values, lack of TCP, robot priority, unsupported metadata, explicit disable, nonfinite/declared invalid targets, and source timing.
- [x] Extend mappings with `kind`, `valid_key`, `valid_index`; preserve legacy controller mappings without changing their serialized contract. Add `controller_valid(valid)` to remove human labels only from filter evidence.
- [x] Human recognition requires matching source, human embodiment, normalized [0,1] openness metadata, known distance formula, and a 20D actions.eef field. Use indices 9/19 and row_offset=0.
- [x] Record a human-specific target contract only when human labels are selected, avoiding unnecessary invalidation of historical robot statistics.
- [x] Run the targeted pytest file and existing relative representation/reader regression tests.

## Task 2: Reader, data preparation and end-to-end verification

Files: openwam/dataloader/mora_relative_lerobot.py, scripts/data/prepare_curated_gripper_commands.py, scripts/data/verify_curated_gripper_commands.py, tests/dataloader/test_mora_relative_reader.py.

- [x] Reproduce the filter failure where a proxy echo of a measured spike would exempt the spike. Use controller_valid in the reader and preserve kind through curated fixture mappings.
- [x] Build new fixtures with: /root/miniconda3/envs/openwam/bin/python -m scripts.data.prepare_curated_gripper_commands --manifest outputs/curated_train_20261011/data_manifest.json --output-root outputs/curated_gripper_human_20261014.
- [x] Verify all accepted windows with the existing verifier and separately compare all 14 human clips' prepared targets against original actions.eef. Assert robot targets, accepted window indices and non-gripper tensors match the previous cmd fixtures.
- [x] Confirm gripper mask enables an actual gradient for all four human families; preserve no-TCP pose masks and tail padding.
- [x] Save source provenance, coverage and verification results, prepare a distinct training entrypoint, and record that the current old process still uses old cached labels.

No commit/push or training restart is part of this change. No new dependencies are needed.

## 实现与验证结果

修正原因：此前把“gripper GT使用cmd”落实为只接受已确认的机器人控制器命令，导致无机器人cmd的人手开合度目标也被关闭。本次用户明确要求恢复四类人手标签，使用已有目标字段，保留其距离定义、归一化和时间对齐。

| 来源 | 片段数 | 保留窗口 | 恢复的gripper监督槽位数 |
|---|---:|---:|---:|
| LightWheel |4|562|32000|
| EgoDex |4|596|34176|
| EgoVerse |4|548|31104|
| Mecka |2|298|17088|

槽位数按重叠训练窗口累计，不是独立源帧数量。总计恢复14条片段、114368个有效目标槽位。整个46片段混合仍为6854个原始窗口、过滤163个、保留6691个；所有窗口索引与上一版一致。

数据契约：

- state输入：当前观测开合度，clip[0,1]。
- 机器人GT：继续使用已验证的cmd，无可信目标仍mask。
- 人手GT：使用actions.eef的第9/19维（0-based），经同一gripper通道写入unified第9/43维，clip[0,1]。来源标记为human_openness_target。
- 不额外移动人手action时间轴。本次LightWheel/Mecka存储的是state+1，EgoDex/EgoVerse是state+3。机器人joint仍为q[t+k+1]-q[t]。
- NaN/Inf、已有action.eef_target_valid=false、chunk padding保持mask。没有TCP继续只mask对应pose，不能因此mask有效gripper。
- 人手proxy不作为尖峰过滤器的独立cmd证据；已确认的短尖峰窗口继续剔除。

验证：先运行新增回归测试复现4类人手全部mask的问题（7项失败），修正后通过。另一个reader测试复现“未来测量proxy含相同尖峰，导致被当作cmd变化而放行”的问题，修正后通过。共85项相关CPU测试通过（62项reader/representation/canonical/gripper测试，23项FK/模型loss测试）。有既存NumPy弃用警告和沙箱CUDA探测警告；测试不依赖GPU。

生产reader逐一检查所有6691个窗口，包含state输入、目标值、mask、32步padding、joint相对值、图像首尾对齐及FK。另一次新旧reader逐窗口对比确认：输入和所有非gripper目标/掩码完全一致，全部机器人gripper值和mask也一致。14条人手片段的目标直接对比原始Parquet，最大绝对误差为0。

对每条人手片段首个有效窗口，用真实生产_compute_action_loss函数及其真实mask，在CPU上给预测值人为增加0.25，得到loss0.0625；64个有效gripper槽位全部产生非零梯度，其余槽位梯度为0。这验证loss/mask接入，不是预训练模型的GPU forward或新训练收敛结果。

工件：

- 新数据与统计：outputs/curated_gripper_human_20261014/data_manifest.json及data/。
- 全窗口验证：outputs/curated_gripper_human_20261014/validation.json。
- 源数据对照和梯度验证：outputs/curated_gripper_human_20261014/human_validation.json；复现脚本同目录verify_human.py。
- 新训练入口：outputs/curated_gripper_human_20261014/run.py；默认run名gripper_human_lr0.0001_b320，batch320、1000步、action LR1e-4、每25步eval。
- 默认TensorBoard目录复用outputs/curated_train_20261011/tensorboard；新run使用独立名字，不混入旧标签曲线。

当前状态：代码、数据缓存、统计量和训练入口已就绪；未停止或重启现有gripper_cmd_lr0.0001_b320。旧进程保留旧reader和prepared-input缓存，不能将这次CPU验证称为正在训练的run已经恢复人手监督。

## 用户授权重启（2026-10-14）

用户随后明确要求使用最新cmd GT、clip[0,1]和ego监督重启实验并生成新的TensorBoard。此节更新前文“未重启”的历史状态。

- 旧gripper_cmd_lr0.0001_b320在step856停止；日志和检查点保留，旧TensorBoard event前缀记录SHA256用于检查。
- 新run：gripper_human_lr0.0001_b320，输出目录outputs/curated_gripper_human_20261014/gripper_human_lr0.0001_b320。
- 初始化仍为OpenWAM-Alpha-Pretrain-Foundation-Model/checkpoint_step_154000.safetensors，不沿用旧cmd实验权重。
- batch320、32步112D目标、1000次迭代、每25步固定噪声评估。video/action/projection LR分别为1e-5/1e-4/1e-3。
- 独立TensorBoard端口18083，日志目录outputs/curated_gripper_human_20261014/tensorboard。旧18082对比服务通过新run的目录链接也可看到新曲线。
- 新网页：https://897628-proxy-18083.dsw-gateway-cn-shanghai.data.aliyuncs.com/#scalars。
- 启动入口：outputs/curated_gripper_human_20261014/launch.py；监督进程1510725、torchrun1511136、TensorBoard1510726。
- 当前运行验证结果保存在同目录restart_validation.json，源代码文件hash在source_snapshot.json。首次训练与ego评估指标写入后再确认运行验证通过。

重启后验证已通过：新TensorBoard本机HTTP200；云平台代理地址返回302到阿里云DSW登录网关。最近核验训练已到step43/1000，train/loss_total=0.3355925、action=0.1586328、video=0.1756386。batch320的首步峰值分配显存108.18GiB，loss_total=1.0995423；所有检查指标为有限数值。

四类人手gripper评估曲线均已出现并更新至step25：LightWheel0.01799843、EgoDex0.0000127872、EgoVerse0.00569999、Mecka0.00224658。固定噪声汇总clean_gripper_mse=0.02238596、clean_joint_mse=0.29649922、loss_fk_position=0.02252794。此处均是固定训练窗口的一次去噪诊断，不是独立测试集或完整多步推理评估。

旧TensorBoard event文件前缀hash全部一致，原有历史run仍可读取；18082对比服务也已显示gripper_human_lr0.0001_b320。独立18083服务只显示本次新run。当前新实验在后台持续训练。
