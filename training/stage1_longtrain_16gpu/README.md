# 新 G：当前16卡训练代码、配置与复现说明

本目录封存2026-11-18停止前DLC长训使用的外置代码和启动记录。原始文件保持字节一致，见 `snapshot.json`。当前训练已按用户要求停止；目标有效性修复仅在DSW验证，未重启DLC。下文生产配置和启动链描述的是封存时状态。

## Stage1目标观察有效性修复

后续维护入口加入了已记录来源语义的Stage1目标mask分离与不依赖GT的G生成。历史runtime快照仍保持原文；新旧runner/core身份不同，不能直接混用旧续训合同。本次DSW七条验证、复现命令、缓存边界与未来完整64条迁移要求见[GOAL_OBSERVATION_FIX.md](GOAL_OBSERVATION_FIX.md)。

## 版本与目录

归档模型核心来自远程分支 `codex/kf1-fixed-time-region-eval-20261115` 的 `b054fd820b328b48500f271cb58d266942c1625c`。当时DLC固定代码目录使用 `221464dc92141f5d0c185f548651e3c452d3f8c8`，这两个提交的Git源码树均为 `932ced238a22de78c37e415154c0aa90718a5904`。本目录最初是在远程版本上补交的运行文件，不应据此声称旧本地提交221464d已进入远程祖先链。当前修复分支在此基础上修改core与维护入口；历史snapshot中的core hash仍指归档基线。

| 目录或文件 | 内容 |
| --- | --- |
| `runtime/` | 生产trainer、16卡DDP与原生检查点、全量数据provider、metadata view、诊断、报告、G/C*配置及数据准备工具原文 |
| `reviewed/` | 动态加载的RFC构建/collate、旧C*适配、图像/TCP评估、中文图表代码及其间接依赖原文 |
| `launch/` | 2026-11-18实际guarded launcher、lifecycle、DLC terminal transport、四个原Pod的spec与映射；属于历史记录 |
| `provenance/` | G实际运行contract、采集时检查点指针；不含权重或优化器张量 |
| `snapshot.json` | 36个原文件的来源路径、字节数、SHA256，以及与运行contract/spec核对结果 |
| `external_assets.json` | 权重、数据YAML、SQLite、metadata、原相机审计、诊断manifest的实际路径与可获取的身份信息 |
| `environment.json` / `requirements-runtime.txt` | 完整包版本记录与排除无关冲突包后的安装清单；不包含操作系统、驱动或镜像本体 |
| `entrypoint.py` | 执行原文trainer，把reviewed helper定位到本目录；父进程与spawn worker显式绑定所选代码路径，保留严格校验；兼容audit |
| `prepare_config.py` | 在新目录生成G配置，显式绑定代码与两份诊断数据；拒绝覆盖已有目录 |
| `launch_g_node.sh` | 每节点8卡的新启动入口，可只打印命令；节点和输出由调用者指定 |
| `verify_snapshot.py` | 默认只读核对包内原文件；可额外核对原CPFS文件 |

这里的runtime指训练执行代码。Python环境、基础模型、数据和检查点仍须按后文准备。`runtime/snapshot.json` 是早期prod_v2来源记录；当前文件的权威清单是本目录顶层 `snapshot.json`。

## 当前实际训练配置

| 项目 | 实际值 |
| --- | --- |
| 作业/节点 | `dlc17vidcbxadyj2`；G使用master-0与worker-0，C*使用worker-1与worker-2 |
| G规模 | 2节点×8卡，world_size=16；每卡microbatch80，梯度累积2次，全局batch2560 |
| 初始化与续训 | 正式实验同一foundation、seed42起训；2026-11-18这次恢复来自G原生update_5000，保留已有权重与优化器状态 |
| task∶subtask | 精确3∶7；每次更新768条task、1792条subtask；RGB/TCP标签均为同粒度区间末帧 |
| 数据采样 | 全量保留数据，3fps合法anchor，有放回连续绝对draw游标；不是256条prepared训练池 |
| 历史 | episode首帧加最近2帧，1Hz，因果可见，保留真实时间PE；history dropout0.15、noise mix0.1 |
| G与DiT | 32个G tokens；DiT30层、dim3072、24heads、latent48；语言进入G；KF1 endpoint固定时间条件0 |
| 损失 | G的 `memory.lambda_image=1`、`memory.lambda_uv=1`；有效元素按整个global update归一化 |
| 训练精度/优化器 | bf16，gradient checkpointing；原生ShardedFP32MasterAdamW，GPU FP32 master/moments；betas0.9/0.999、eps1e-8、weight_decay0.01 |
| 学习率/梯度 | backbone1e-5、新参数1e-4；clip1；每个micro同步DDP，bucket view开启 |
| Y文本warmup | reference50000 updates，前20%内Y直接文本概率从0.5线性降到0；此reference不是停止步数 |
| VAE/文本 | 冻结；独立T=1编码并合并cache miss；batch16/4；每rank8 workers、prefetch2 |
| 缓存 | rank私有帧缓存128GiB，G文本缓存64GiB；完整配置见原始G.json |
| 保存/报告 | 每1000步保存与固定64条拟合诊断，20步生成；检查点保留2代；默认持续训练 |

优先级：命令行决定microbatch、累积、输出和resume；`G.json`决定实际数据provider与model_config；`G_model.json`的 `components` 和 `memory` 决定G结构。继承字段 `batch_size=1`、`max_steps=5000`、旧cohort路径、`model.query_goal` 和顶层 `lambda_image=0.5` 均不代表本生产入口的上述实际值。完整原始配置已收录，未为美化而重写。

## 必须准备的外部资产

本包不把权重、优化器分片、数据或诊断张量写入Git。具体值、大小和哈希出处见 `external_assets.json`。

1. 当前foundation为 `checkpoint_step_154000.safetensors`，包含DiT、VAE、text encoder；tokenizer使用同一foundation目录下 `tokenizer/google/umt5-xxl`。不能只准备DiT权重。
2. 保留的全量源YAML为 `source_exclude_missing_133.yaml`，约13.8MB；它引用的数据目录、视频、状态、标定和外部subtask标注都须可读。不能换成小cohort或因缺文件静默重抽样。
3. `metadata_generation_v2/` 中的SQLite、audit、manifest、path_aliases及各个冻结view，必须成套准备。manifest又引用原始相机信息审计及外部annotation。只复制SQLite不够。
4. G的prepared目录和C*的prepared目录都需要。即使只训练G，当前contract也会读取两份manifest。两份目录内的train256、val64、text/context文件及manifest引用的external_files都要保留；C*部分context引用G目录的绝对路径。
5. `OPENWAM_VIDEO_INDEX_CACHE` 指向已准备的只读视频索引。新机器若没有对应索引，需先按仓库的数据准备流程生成，不能把read改成静默容错。
6. 原生续训还需要 `checkpoint.json` 指向的完整generation：权重bundle、manifest、training state、16个optimizer shard、各rank RNG。`provenance/G.checkpoint-at-capture.json` 仅用于说明来源，不是可独立恢复的checkpoint。

`external_assets.json` 中foundation哈希继承已验证的运行contract，本次没有重新读完整大权重。小文件哈希在采集时重算；SQLite记录原生audit所用size/mtime与逻辑身份。复制数据需要保留原生guard要求的时间戳/身份；换路径或重建索引时重新审计，不能编辑stamp绕过校验。缓存不属于学习到的状态，可在容量足够的新盘重建。

## 环境

采集到的共享训练Python是 `/tmp/openwam-runtime-20261031/env/bin/python`：Python3.10.20、torch2.7.1+cu128、torchvision0.22.1+cu128、CUDA build12.8。完整Python包版本见 `environment.json`。该记录来自当前共享训练环境；没有把它当成16个Pod的驱动/镜像逐项审计。

优先复用原已验证环境。该环境还安装了无关的FastWAM editable包与lerobot工具包，其声明依赖与实际版本存在冲突。完整库存仍保存在environment.json；安装清单排除了这两项及需要单独安装的OpenWAM，当前原生Stage1路径不导入它们。保留项的已安装版本满足相互间的基础依赖metadata，但尚未实际重建全新环境或验证所有optional extras。

另一环境可从该清单安装，再校验PyTorch/CUDA与依赖；例如在已经准备好的干净Python3.10环境内：

```bash
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
  -r training/stage1_longtrain_16gpu/requirements-runtime.txt
python -m pip install --no-deps -e .
```

版本清单不能代替容器、系统库、NCCL跨节点网络、CPFS挂载和GPU驱动。新环境的GPU训练仍须单独验证；本次提交只执行CPU检查，没有在正在训练的节点另起进程。每卡80是当前资源调优结果，不能推定所有GPU型号均能容纳。

## 克隆与只读检查

```bash
git clone --branch codex/kf1-fixed-time-region-eval-20261115 \
  https://github.com/shuluoshu/OpenWAM-scale.git OpenWAM
cd OpenWAM
python training/stage1_longtrain_16gpu/verify_snapshot.py

# 仅在原DSW环境需要逐项比对原文件时使用。
python training/stage1_longtrain_16gpu/verify_snapshot.py --check-sources
```

后一命令包含检查点指针等采集时的可变记录。若后续训练保存了新检查点，对应原文件变化应作为版本差异核对，不代表旧快照损坏，也不是训练故障。

```bash
/tmp/openwam-runtime-20261031/env/bin/python -m pytest \
  tests/goal/test_longtrain_source_snapshot.py \
  tests/goal/test_dlc_runtime_archive.py \
  tests/goal/test_kf1_contract_migration.py \
  tests/goal/test_kf1_condition_and_regions.py -q
```

## 在已挂载同一资产的环境新开一组训练

以下步骤用于新的输出目录。它们不提交新DLC作业，也不替换原作业的进程。先在已经分配好的两个空闲节点确认GPU、CPU内存、缓存盘、检查点盘和网络资源。初次换环境应先做完整输入审计与有界profile。

在仓库根目录生成新配置，输出到新的位置：

```bash
python training/stage1_longtrain_16gpu/prepare_config.py \
  --output-dir /tmp/openwam-g-repro-config
```

生成器默认把 `code_dir` 指向本次克隆，显式绑定原G/C*诊断路径，保留模型、数据、batch和调度语义。需要移动诊断数据时可传 `--prepared-g` 和 `--prepared-cstar`；生成器不复制数据。查看生成的G.json/G_model.json，按实际挂载修改数据、foundation、tokenizer或缓存路径；数据/依赖身份也必须随实际准备结果重新验证。只改文件名不能保持原run contract。

源或索引发生变化时，对实际配置的source、index和frozen metadata执行原生全量审计。示例使用当前资产路径；audit输出写入新配置目录，然后将新G.json的 `data_pipeline.audit_file` 指向该输出。

```bash
REPO=$(pwd)
TRAIN_PYTHON=/tmp/openwam-runtime-20261031/env/bin/python
PYTHONPATH="$REPO" "$TRAIN_PYTHON" \
  training/stage1_longtrain_16gpu/entrypoint.py audit \
  --code-dir "$REPO" \
  --source-yaml /mnt/data/cpfs/b5/yeqingda/openwam-0911/OpenWAM/outputs/stage12_subset_continuation_20261109/source_exclude_missing_133.yaml \
  --index-path /mnt/data/cpfs/b5big/yeqingda/CodeX/longtrain_16gpu_20261112/metadata_generation_v2/population.sqlite \
  --metadata-view-manifest /mnt/data/cpfs/b5big/yeqingda/CodeX/longtrain_16gpu_20261112/metadata_generation_v2/manifest.json \
  --output /tmp/openwam-g-repro-config/data-audit.json
```

该审计校验全量原生metadata/source身份，不代表全量视频内容已哈希或逐帧解码；实际读取仍保留episode digest、物理文件generation与相机策略检查。

两个节点执行同一仓库版本，读取同一配置和共享输出目录，选择两个节点互通且空闲的master端口。先只打印命令，例如：

```bash
# MASTER_ADDR填写所分配node_rank=0的跨节点可达IP；不能填写DSW网页服务IP。
MASTER_ADDR=22.52.151.91
export MASTER_PORT=30681
export TRAIN_PYTHON=/tmp/openwam-runtime-20261031/env/bin/python

# 先打印、核对；此命令不会启动训练。新输出目录必须与现有runs/G分开。
bash training/stage1_longtrain_16gpu/launch_g_node.sh --print-command \
  0 "$MASTER_ADDR" /tmp/openwam-g-repro-config/G.json \
  /mnt/data/cpfs/b5big/yeqingda/CodeX/repro_new_g/runs/G
```

`22.52.151.91` 仅是2026-11-18原G master的记录；新运行要换成实际master地址。实际启动时移除 `--print-command`；在node_rank0和node_rank1分别运行：

```bash
# node_rank=0；在第一个已分配且空闲的训练节点执行。
bash training/stage1_longtrain_16gpu/launch_g_node.sh \
  0 "$MASTER_ADDR" /tmp/openwam-g-repro-config/G.json \
  /mnt/data/cpfs/b5big/yeqingda/CodeX/repro_new_g/runs/G

# node_rank=1；在第二个节点执行，同样设置MASTER_ADDR、MASTER_PORT、TRAIN_PYTHON。
bash training/stage1_longtrain_16gpu/launch_g_node.sh \
  1 "$MASTER_ADDR" /tmp/openwam-g-repro-config/G.json \
  /mnt/data/cpfs/b5big/yeqingda/CodeX/repro_new_g/runs/G
```

默认batch仍是16×80×2。新launcher不是原生产guarded launcher，没有替调用者检查DLC作业身份、GPU占用或receipt；这些资源和防重复启动核对由调用者完成。`launch/`保留了当前生产guard的全部原文；没有在仓库入口中绕过或修改trainer原有的科学/检查点contract检查。

需要有界全量profile时，在两个空闲节点用相同torchrun参数运行 `entrypoint.py profile --arch G --config ... --local-batch 80 --accumulation 2 --output <独立profile目录> --profile-full-source --profile-warmup 1 --profile-steps 3`。这里应使用实际全量provider，不能把小fixture profile结果当成正式训练吞吐；此命令只作说明，本次未执行。

## 续训与当前原始启动链

当前生产链为：DSW `dlc_remote_verified.py` → 原Pod `launch_original_pod.py` → Pod对应spec → `lifecycle.py` → torchrun → 原文 `runtime/trainer.py`。G两个spec保存了实际启动命令、环境、代码commit、文件哈希、原检查点manifest哈希、log与receipt路径；第四节点的C*记录也保留用于说明共享launcher映射。

原G实际使用rank0/1、master `22.52.151.91:30577`，每节点8卡、local-batch80、accumulation2，`--resume` 指向原 `runs/G/checkpoint.json`。2026-11-18的恢复来源为update_5000，manifest SHA256为 `b11c041840a13b15a3119237e5c15e12cc0bea7fc1d96d3bd094d472ddab0529`。它是本次恢复的来源，不能当成之后最新检查点的声明。

已有检查点严格续训时：

1. 核对最新指针、完整manifest、权重和全部16个optimizer shard的哈希及rank ownership。保留FP32 master、Adam moments/step、RNG与absolute draw cursor，不重新初始化优化器。
2. 保持world_size、microbatch、accumulation、模型、数据、源代码、依赖与路径身份符合原contract。源码相同不代表移动代码目录后的完整contract相同。
3. 只有匹配同一个run contract时，才在两个节点加入同一 `--resume /absolute/checkpoint.json`，输出也必须与对应run一致。原生restore逐项校验；不能删除contract、改hash或跳过strict来强行加载。
4. 需要迁移代码/数据路径或其他contract字段时，单独准备并验证迁移记录；本次收录不自动迁移，也没有在GPU上验证换路径续训。新开配置不是旧run的续训配置。
5. 历史 `launch/*.spec.json` 绑定了已经存在的receipt、5000/3000检查点和旧代码commit。不能原样重放；它们用于追溯当前启动。下一次生产恢复需要对应最新已验证checkpoint与空闲Pod的新spec、端口、日志和回执。

停止由SIGINT/SIGTERM或run目录内 `STOP_REQUESTED` 请求触发，按优化器边界原生保存后退出；`--stop-after N` 指绝对更新步。默认无stop-after，50000只是warmup参考值。

## 千步输出与指标

| 产物 | 含义 |
| --- | --- |
| `train.jsonl` | 绝对update、原生loss、梯度、draw cursor、实际seconds与samples_per_second |
| `health/rank_XX.json` | 各rank阶段和实际步数；只看DLC Running状态不能证明在训练 |
| `checkpoint.json`、`checkpoints/step_*/manifest.json` | 原子发布的完整检查点与原生优化器分片 |
| `validation/step_*/manifest.json` | 固定64条、20步生成的拟合诊断与实际诊断checkpoint步数 |
| `report/index.html` | 中文loss与拟合报告及图像/TCP可视化；目录由实际output决定 |

报告分别标记实际训练步数和诊断步数。固定64条可能与全量训练重叠，只称拟合诊断。head RGB MAE用0–1值，TCP用平均欧氏像素误差；区域指标包含复制当前帧基线、变化区域和GT TCP邻域及相对复制改进。G和旧C*原生loss定义不同，不能用绝对loss高低判断架构优劣。

trainer负责每1000步保存、诊断和本地HTML，`runtime/report_publish.py`负责已有本地报告发布；向原Codex会话发送通知的monitor/队列是另外的运行服务，本包没有注册新的通知、账号或定时任务，故新训练不会自动向本会话发消息。当前既有监控继续由原服务处理。

当前已验证内网报告为 [中文长训报告](http://10.1.128.54:8012/kf1_longtrain_20261115/report/index.html)。该地址是当前DSW发布服务，不是新克隆仓库自动生成的服务。

## 本次验证范围

原始36文件字节、7个runner和6个reviewed helper的contract哈希、本次spec所绑定文件均已比对。CPU回归覆盖篡改/缺文件检出、配置语义与拒绝覆盖、包内动态import、runner/helper/source身份、metadata路径绑定、spawn反序列化、原CLI，以及启动命令参数的安全引用。没有修改 `openwam/` 模型或训练计算代码，没有重启DLC、加载大权重或另占训练GPU；新的完整GPU训练与跨机器恢复不属于已验证结论。
