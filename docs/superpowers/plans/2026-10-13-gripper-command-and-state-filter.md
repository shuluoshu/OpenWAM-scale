# Gripper command supervision and state spike filtering

> 2026-10-14更新：用户明确要求将LightWheel、Mecka、EgoDex、EgoVerse的距离开合度目标加入gripper监督。下文的“human proxy不作为GT”是旧实验策略，已由[人手gripper监督修正记录](2026-10-14-human-gripper-supervision.md)取代。新代码和新fixture已验证；旧运行进程未自动切换。

> Execute inline with test-driven development. The user explicitly selected state input, command GT, unchanged next-state motor targets, and skipping noisy samples. No new approval is needed for this design.

**Goal:** Relative94/112 readers retain measured gripper state at the anchor, supervise grippers with declared controller commands, and exclude training windows containing gripper-state spikes.

**Architecture:** Separate command provenance/alignment and state quality detection into a gripper utility. Representation overrides only raw slots9/19 (unified9/43). Reader constructs a deterministic index of accepted windows before sampling; filtering never returns None or relies on bounded exception retries. Statistics and training share the same accepted index and command contract.

**Constraints:** 32 targets; joint delta q[t+k+1]-q[t], TCP/head targets, FK definition and masks retain their existing semantics. Gripper values remain continuous clip[0,1]. No command means masked gripper GT, not an implicit state fallback. Preserve source data and historical experiment artifacts.

## 1. Command contract and tests

- [x] Add tests in tests/dataloader/test_gripper_commands.py for scalar/vector commands, independent state and GT, missing/derived commands, nonfinite masks, and source columns shifted by one frame.
- [x] Implement openwam/dataloader/utils/gripper.py with explicit per-side key/index/row_offset/scale/bias/provenance specs. row_offset=-1 restores cmd[t] from a source column[t-1] that already stores cmd[t]. Unavailable episode-boundary commands remain masked.
- [x] Use metadata-proven commands only: A1 master openness, ABC same-frame controller command, Morphi preserved action.position, RoboMind documented master command sources. Never auto-promote human proxies or state-derived RoboChallenge source_action.
- [x] Wire command sources into MoraRelativeRepresentation; encode_window consumes aligned command arrays when supplied by the reader. Gripper state remains measured state. Add contract and schema fingerprint fields.

## 2. State quality and deterministic skipping

- [x] Test command-stable state jumps, short isolated returns, legitimate command-driven opening/closing, missing command, episode boundaries, tail padding, complete rejection, and index/video alignment.
- [x] Add configurable gripper_state_filter (default enabled): max_step=0.1, command_tolerance=0.01, settling_frames=15, reversal_window=3, return_tolerance=0.02. A jump after stable commands or an isolated jump-and-return without a matching command transition flags affected frames; normal sustained state transitions are retained.
- [x] Scan each selected episode without decoding video, align commands before slicing windows, and reject any 33-row training window containing flagged state frames. Cache only bounded per-episode diagnostic arrays; retain a compact array of accepted original sample indices.
- [x] Expose accepted/rejected counts and original anchor indices in reader diagnostics/trace. max_windows limits accepted windows. Zero accepted windows is explicit and is omitted by the curated mixture.

## 3. Statistics and fixtures

- [x] Update prepare_openwam_relative_stats.py to use the production reader's accepted indices and aligned command arrays. New stats provenance includes the command and filter contracts; stale label-contract statistics fail clearly.
- [x] Extend curated data preparation into a separate experiment directory, adding necessary original command columns and metadata from the original Parquet. Keep historical fixtures/statistics and RGB caches unchanged. Preserve family weighting over accepted samples.
- [x] Validate all accepted samples: measured state input, exact aligned command GT, other raw channels and FK payload semantics unchanged, and no rejected source windows enter training. Print real A1/ABC examples and per-source filter/missing-command counts.

## 4. Verification and training handoff

- [x] Run /root/miniconda3/envs/openwam/bin/python -m pytest on gripper, relative reader/representation, canonical, and FK tests.
- [x] Check git diff --check; inspect real-data filtering around A1 frames130/131 and legitimate ABC/A1 close/open transitions.
- [x] Persist the new label contract, source mappings and filter report. Any new training run must use a distinct run name and prepared-input cache; a running Python process does not hot-reload reader changes.

No commit or push is included in this task. Existing unrelated changes are preserved.

## 实现与验证结果（2026-10-13）

本次范围：OpenWAM 的 Mora relative94/112 reader。当前 32 步不变。

| 内容 | 窗口 anchor=t 时的值 |
|---|---|
| 夹爪 state 输入 | 测量 state[t]，clip 到 [0,1] |
| 第 k 个夹爪 GT（k=0..31） | 对齐后的 cmd[t+k]，clip 到 [0,1] |
| 第 k 个 joint GT | q[t+k+1] - q[t] |
| TCP/head/FK | 仍使用 state[t+k+1] 对应的 GT 与当前 anchor；旋转使用矩阵乘法 |
| 没有可信 cmd | 只 mask 夹爪 GT，保留其他有效监督；不退回 future state |

过滤只检查夹爪测量 state 的尖峰，采用明确、可配置的启发式规则，不声称识别所有传感器异常：

- 相邻帧幅值变化 >0.1，且此前至少15帧（30Hz下0.5秒）的 cmd 变化范围 <=0.01，标记跳变相邻帧。
- 或变化 >0.1 后在3帧内回到原值附近（误差 <=0.02），且没有附近的 cmd 变化，标记短峰区间。
- 有对应 cmd 变化的正常开合保留；NaN/Inf 延续原有 mask 行为。
- 一个33帧窗口只要包含标记帧，就从训练索引中整条移除。不是只 mask 尖峰，也不通过随机重试跳过。
- 默认开启，阈值单位是 [0,1] openness；settling_frames/reversal_window 是帧数，不假定所有原始数据都是30Hz。

### 真实样例

46个源片段，原始6854个训练窗口，过滤163个，保留6691个：

| 片段 | 原始窗口 | 保留 | 过滤 |
|---|---:|---:|---:|
| 23_interndata_a1 |149|68|81|
| 06_LightWheel |149|115|34|
| 17_egoverse |149|101|48|
| 其余43个片段合计 |6407|6407|0|

A1异常样例的源帧130/131（裁剪内41/42）被标记，覆盖该异常的所有训练窗口均已剔除。其余三个A1及四个ABC样例全部保留。

18个片段具有经过来源确认的 cmd：4个ABC、4个A1、2个Morphi、4个RoboMind v1、4个RoboMind v2。其余28个片段的夹爪GT关闭：Genrobot4、human类14、RoboChallenge8、Morphi2。Morphi28没有独立cmd；Morphi29虽然存在 action.position，但缺少前两例的保留来源/单位元信息，本次保守mask，可通过显式 gripper_command_sources 经审核后开启。

RoboMind v1 的 source_action 已保存 cmd[t+1]，reader 使用 row_offset=-1 恢复 cmd[t]，在整条episode上对齐后再切窗口，episode首行缺少前文时mask。新裁剪fixture从原始Parquet额外读前一行上下文，因此不损失裁剪首行cmd。RoboMind v2直接读取元信息声明的 raw.master scalar，按声明的0..10范围乘0.1；不更改该数据集历史标定。

验证环境 /root/miniconda3/envs/openwam，70项相关CPU回归测试通过。真实数据逐一检查6691个保留窗口：夹爪state输入、cmd数值/mask、32步padding、固定anchor的joint差值、窗口索引；每片段首尾窗口检查实际RGB缓存对齐。两个启用URDF/FK的A1片段共217个窗口全部检查，GT joint重建后FK position loss最大1.50e-13 m²，rotation loss最大1.10e-12。

### 文件与复现

核心实现：

- openwam/dataloader/utils/gripper.py：cmd来源/时间对齐、尖峰检测。
- openwam/dataloader/utils/mora_relative.py：只替换raw9/19（unified9/43）的GT。
- openwam/dataloader/mora_relative_lerobot.py：过滤索引、诊断、trace、episode上下文。
- scripts/data/prepare_openwam_relative_stats.py：只统计保留窗口，校验cmd/过滤契约。
- openwam/dataloader/curated_fixture.py：从过滤后的reader构建混合样本与权重，空片段跳过。

新数据：outputs/curated_gripper_cmd_20261013/data_manifest.json

过滤报告：outputs/curated_gripper_cmd_20261013/gripper_filter_report.json

全量验证：outputs/curated_gripper_cmd_20261013/validation.json

新训练入口：outputs/curated_gripper_cmd_20261013/run.py，默认batch320、1000步、action LR1e-4、每25步评估；同名已有metrics会拒绝启动，避免混入旧标签曲线。

准备到新的、不存在的目录：

```bash
/root/miniconda3/envs/openwam/bin/python -m scripts.data.prepare_curated_gripper_commands --manifest outputs/curated_train_20261011/data_manifest.json --output-root outputs/curated_gripper_cmd_next
/root/miniconda3/envs/openwam/bin/python -m scripts.data.verify_curated_gripper_commands --root outputs/curated_gripper_cmd_next
```

旧的running process不会热加载代码或prepared-input缓存。本次没有停止或重启旧run，也没有声称新规则已经产生训练loss。新入口、数据、统计量已准备并完成CPU数据/FK验证；当前GPU仍由旧run占用。旧源码数据、旧fixture、旧TensorBoard曲线均保留。新统计量在过滤后的样本上重算，因此其他通道的原始GT定义不变，但少数片段的归一化统计数值可能改变。

## 用户授权重启记录（2026-10-13）

随后用户明确要求重启训练并将新旧loss曲线放在同一个TensorBoard。本节更新前述“未重启”状态：

- 旧run `gripper01_lr0.0001_b320` 已在step157停止，全部event前缀SHA256校验保持一致；之前step1000的 `main_lr0.0001_b320` 同样保留。
- 新run `gripper_cmd_lr0.0001_b320` 已启动，batch320，计划1000步，每25步固定噪声评估，使用同一个foundation预训练权重重新初始化，不沿用旧标签训练得到的权重。
- supervisor PID1169705；torchrun PID1169747；启动脚本 `outputs/curated_gripper_cmd_20261013/launch.py`。
- 共用TensorBoard事件根目录 `outputs/curated_train_20261011/tensorboard`，同名的 `train/loss_total`、`train/loss_action`、`train/loss_video`、`eval_fixed_noise/clean_joint_mse`、`eval_fixed_noise/clean_gripper_mse`、`eval_fixed_noise/loss_fk_position` 可以按step叠加新旧run。
- 地址：https://897628-proxy-18082.dsw-gateway-cn-shanghai.data.aliyuncs.com/#scalars
- 重启验证 `outputs/curated_gripper_cmd_20261013/restart_validation.json`：本机HTTP200，新run已有step1/2/3，loss_total=1.028756/0.837071/0.833696。旧run有157个训练点、此前完成的main run有1000个训练点。新run初始eval(step0)已写入，下一次eval在step25。
- 标签/有效mask/部分样例归一化统计已改变，因此叠图用于观察各自收敛过程，夹爪loss不能据此作为相同GT条件的严格A/B比较。

重启流程只停止了已核对命令行的旧smoke进程；其日志和检查点不删除，TensorBoard服务继续复用。新的supervisor继承GPU保活程序的暂停/恢复责任。

## 2026-10-14 训练进度与run差异复核

2026-10-14 05:47 UTC的日志快照显示：`gripper_cmd_lr0.0001_b320` 已完成720/1000步，未停止；最近25步约81秒/步，剩余约6.3小时（不含额外评估与保存开销）。GPU查询利用率92%。旧 `gripper01` 于157步按用户授权切换停止，`main`已完成1000步。

| 比较项 | main_lr0.0001_b320 | gripper01_lr0.0001_b320 | gripper_cmd_lr0.0001_b320 |
|---|---|---|---|
| 夹爪输入来源 | 测量state[t] | 测量state[t] | 测量state[t] |
| 夹爪GT来源 | state[t+1:t+33] | state[t+1:t+33] | 对齐cmd[t:t+32] |
| 夹爪模型数值 | 按各片段state/action分位数映射到[-1,1] | 保留物理开合度并clip[0,1] | 保留物理开合度并clip[0,1] |
| 尖峰过滤 | 无新增过滤 | 无新增过滤 | 剔除包含夹爪state尖峰的完整窗口 |
| 窗口数 | 6854 | 6854 | 6691（排除163） |
| 缺少可信cmd | 仍可监督future state | 仍可监督future state | 只mask夹爪GT |
| 其他通道统计量 | 旧片段统计量 | 与main相同 | 对保留窗口重新计算 |

已逐字段比较3份保存的config.yaml：model完全相同；training除输出路径之外所有字段相同。均为112D、32步、batch320、同一foundation初始化，video/action/projection实际LR为1e-5/1e-4/1e-3，video/action/FK-position/FK-rotation loss权重均1，bf16、ZeRO2、CPU optimizer offload。关节均使用q[t+k+1]-q[t]；TCP/head/FK GT定义相同。

main到gripper01的变化不等于改用cmd。旧reader快照、gripper01 label_contract和6854窗口校验共同证明，该次只调整夹爪归一化及其反归一化，其他原始数值/监督mask不变。例如20_interndata_a1右夹爪state原始开合度0.03，在main的该片段分位数下映射为-0.61531，在gripper01中保持0.03。这个例子说明它不是简单地将负值截断而保留原来的量纲。

新run step719附近最近25步均值：train total0.10356、video0.09248、action0.01098。最近固定噪声eval step700：clean_joint_mse0.01292546、clean_gripper_mse0.00214376、loss_fk_position0.00481136。不同run的GT数值、mask和样本分布有变化，不能直接把loss更低等同于同条件下模型更好。

“gripper_cmd只显示开头”的复核：本机TensorBoard /data/plugin/scalars/scalars 接口实际返回719个train/loss_total点，末步719；固定噪声eval返回29点（step0到700，间隔25），event文件最后修改仅约7秒前，TensorBoard日志无报错。训练进程及TensorBoard数据读取没有复现中断。已向用户询问其页面URL、tag和最后显示步数；尚不能证实浏览器端具体是未刷新、筛选、坐标轴/缩放，还是查看了历史快照。没有因未经证实的显示端猜测去重启正在正常运行的训练或TensorBoard。

可独立查看的曲线快照：outputs/curated_gripper_cmd_20261013/progress_20261014/curves.png（截至step720）；原始审计数据progress.json包含各run全部metrics和TensorBoard接口结果。

## TensorBoard三个指标组与LightWheel/EgoDex曲线说明（2026-10-14）

- train：每步随机混合batch320，训练时抽取不同噪声/timestep，video/action flow-matching loss加FK辅助loss，参与反向传播。日志数值来自该步更新前forward。
- eval_dataset：启动step0、每25步、最后一步，对14个固定训练窗口逐一用eval/no_grad计算；每个窗口固定随机种子17001+n。按clip id分别记录clean_action/tcp/joint/head/gripper MSE，只写valid mask非空的项。并非遍历整个命名数据集。
- eval_fixed_noise：同一次评估的汇总，包含loss和clean MSE。clean_gripper_mse仅对具有有效gripper GT的窗口求等权平均；没有GT则不计入该clean指标分母。注意loss_action按模型返回值逐样例平均，无action监督的样例返回0，这与clean_action_mse跳过无GT窗口的聚合方式不同。
- clean MSE来自一次加噪forward的clean估计：a_hat=noisy_action-sigma*predicted_velocity，与模型输入尺度的GT比较。不是从纯噪声执行完整多步action采样的最终误差。两类eval均使用训练集合中的固定样例，用于过拟合诊断，非独立验证集。

直接读取TensorBoard event文件证实：

| run | LightWheel clean_gripper_mse | EgoDex clean_gripper_mse |
|---|---|---|
| main_lr0.0001_b320 | 41点，step0到1000 | 41点，step0到1000 |
| gripper01_lr0.0001_b320 | 7点，step0到150 | 7点，step0到150 |
| gripper_cmd_lr0.0001_b320 | 无此tag，从第0步即无GT | 无此tag，从第0步即无GT |

旧gripper01于157步切换停止，每25步eval一次，故最后eval点为150；不是训练在200步前异常中断。新cmd版本缺少的是这两个数据集的有效cmd监督。LightWheel metadata声明human_bimanual，其gripper由拇指到其他指尖距离计算；EgoDex声明human_hands，由拇指/食指pinch距离及closedness翻转得到openness。它们是观测到的人手开合代理量，不是实际控制器下发的cmd。按本次cmd-only GT约定，不回退使用future state作为cmd。

实际reader复核04_LightWheel和12_egodex固定评估窗口：左右gripper state输入有效，gripper action mask为0；这两者当前也没有有效TCP/robot joint监督，因此总action mask为0，仍参与video监督。eval_dataset当前只写action相关clean MSE，故新run没有这两个clip的该组卡片，但不代表源视频被剔除。08_abc_130k对照有真实cmd，其新run该指标已记录30点至step725。

原始tag核验保存在 outputs/curated_gripper_cmd_20261013/progress_20261014/gripper_eval_coverage.json。此前在用户未指明数据集/tag时只确认了全局loss持续更新；本次补充的具体缺失原因是监督mask加旧run停止，与那条全局结论相容，不应再笼统归为页面刷新问题。

## 关于“cmd + clip是否更好学”的限定结论

按同一step150比较，main/gripper01/cmd：eval clean_gripper_mse=0.030798/0.004068/0.006194，clean_joint_mse=0.054616/0.042185/0.052308，FK position loss=0.007433/0.009207/0.011464；train step126..150平均action loss=0.100814/0.044205/0.036451。训练action loss降低不等于所有评估误差同步改善。

main到gripper01：相同数据、raw目标及mask，变化为夹爪输入/目标不再按片段分位数拉伸，保持物理[0,1]。MSE换尺度后不能直接比较绝对值。原来窄幅state抖动被quantile放大；目前片段的13800个有限state gripper标签均在[0,1]，截断越界值本身不是这里已验证的主要收益。joint目标尺度未变，step150 joint MSE下降约22.8%，是改善联合优化的积极迹象，但只是一组smoke实验，FK并未同时改善。

gripper01到cmd同时改变：标签来源、有效监督mask、尖峰过滤、样本分布和部分统计量。无action GT样例仍在模型batch平均中返回0，因此train/action loss变低包含监督构成变化的影响。即便只对共同7个机器人评估样例汇总，step150 gripper MSE也为state-GT0.004199、cmd-GT0.006194；不能据此宣称cmd更容易，也不能反过来说更难，因为GT本身不同，A1过滤后固定评估窗口亦改变。

支持的结论：保留物理[0,1]避免不必要的噪声放大有依据；cmd目标与实际控制指令语义匹配有依据；当前整套cmd+mask+过滤流程能在样例数据上收敛。尚未证明cmd这个单一因素提高学习速度或泛化/控制表现。严格比较需同一具备state/cmd的源窗口、相同mask/过滤/统计/预算，分开消融GT来源和过滤开关，并在统一物理单位及固定任务指标下评估；当前仅为固定训练窗口的单次去噪诊断。
