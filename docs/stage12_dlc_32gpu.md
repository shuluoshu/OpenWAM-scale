# Stage1 → Stage2：3 FPS / DLC 4 节点 × 8 卡

用户已确认将原来的单机 32 卡改为 **4 节点 × 8 卡**。本版本保留全部已选数据来源，Stage1 按 3 FPS 遍历有效 task/subtask 配对一轮，随后自动冻结 query/history encoder，开始 Stage2 的 action/video 训练。每卡 batch=1，全局 batch=32；Stage2 仍为 100000 次更新。Stage1 每 500 步追加可视化。

## 文件与参数

| 文件 | 用途 |
|---|---|
| `configs/goal/stage12_query_history_posttrain_3fps_4x8gpu.yaml` | 4 节点训练 preset，与旧单机版输出分开 |
| `configs/pai/stage12_3fps_32gpu.site.json` | DLC 工作空间、资源组、镜像、每 Pod 资源、挂载与 VPC |
| `scripts/pai/stage12_entry.sh` | DLC 容器入口；支持 prepare/check/train |
| `scripts/pai/prepare_stage12_dlc.py` | 生成 CreateJob JSON、dry-run 与提交脚本；自身不会提交任务 |
| `openwam/train/stage12_distributed.py` | 主节点协调、分布式进程启动、失败传播和进程回收 |

部署参数来自 2026-10-24 查询到的成功作业 `dlc6s0umjw7jois6`，其原配置为 1 个 8 卡 Worker；本次将 Worker 数改为 4：

| 参数 | 值 |
|---|---|
| Region / 工作空间 | `cn-shanghai` / `1140257` |
| 专有资源组 | `quota9lheyy49shl` |
| Worker Pod 数 / 每 Pod GPU | **4 / 8** |
| 每 Pod CPU / 内存 / 共享内存 | 168 / 1600 GiB / 64 GiB |
| 总申请 | 32 GPU、672 CPU、6400 GiB 内存 |
| 镜像 | `moqi-sh-arc-registry-vpc.cn-shanghai.cr.aliyuncs.com/moqi-sh/mq_sanyi_train:sanyi-dsw-0515` |
| Python | `/mnt/data/cpfs/b5/yeqingda/envs/openwam/bin/python` |
| RDMA | 启用，沿用参考作业设置 |
| 输出 | `outputs/stage12_posttrain_3fps_4x8gpu` |

用户明确要求不查询空闲容量，直接按排队方式准备任务；因此不查询 quota、不安装 paistudio 插件，也不宣称资源当前空闲。最终调度由 DLC 决定。当前 DSW 的工作空间 `1025431` 与此次 DLC 的 `1140257` 不同，这是已有成功训练作业所用的目标工作空间。

## 1. 先完成 CPU 数据准备

```bash
cd /mnt/data/cpfs/b5/yeqingda/openwam-0911/OpenWAM/outputs/query_latent_20261020/code
export OPENWAM_CODE_DIR="$PWD"
export OPENWAM_PYTHON=/mnt/data/cpfs/b5/yeqingda/envs/openwam/bin/python
export OPENWAM_EXPECTED_COMMIT="$(git rev-parse HEAD)"
export STAGE12_CONFIG=configs/goal/stage12_query_history_posttrain_3fps_4x8gpu.yaml

bash scripts/pai/stage12_entry.sh prepare
bash scripts/pai/stage12_entry.sh check
```

准备会在派生 source.yaml 不存在时生成选择，只移除之前获授权的 `wipe_table_v1_test`，然后生成全量索引、元数据审计和统计。已有 16-episode probe 不能代替生产准备。训练入口发现缺少 `prepared.json` 会退出，不占用 32 GPU 做长时间索引构建。

## 2. 生成任务与 dry-run

```bash
python scripts/pai/prepare_stage12_dlc.py \
  --site configs/pai/stage12_3fps_32gpu.site.json \
  --out outputs/dlc_stage12_3fps_32gpu --dry-run
```

产物：

- `request.json`：create-only 请求，不包含响应状态、Pod 信息或密钥。
- `dry_run.sh`：CLI 序列化检查。
- `submit.sh`：执行一次真实 CreateJob，进入 DLC 排队。

`commit: auto` 在生成请求时解析为实际 Git SHA，写入任务环境变量，不在容器启动时动态取最新分支。训练目录不得有未提交改动；运行期间不要修改这个共享 checkout。新实验建议独立 checkout 与输出目录。挂载目录中的代码不会固化到镜像里。

生成器使用已安装 pai skill 的 validator；默认路径 `/root/.codex-chatgpt/skills/pai/scripts/validate_submission.py`，其他环境用 `--validator` 指定。CLI dry-run 只证明字段格式和序列化正确，不证明镜像、驱动、RAM、挂载、跨节点网络或运行时兼容。

## 3. 提交排队

完成 CPU 准备并检查请求后执行：

```bash
bash outputs/dlc_stage12_3fps_32gpu/submit.sh
aliyun pai-dlc get-job --region cn-shanghai --job-id RETURNED_JOB_ID
```

本次生成脚本本身不会创建 DLC 作业。提交后用实际返回的 JobId 替换示例值。

## 多机启动与阶段切换

DLC 给每个 Pod 注入的 `RANK/WORLD_SIZE` 分别是节点序号和节点数（0–3 / 4）。每节点由 torchrun 启动 8 个 GPU 进程，训练子进程的 `WORLD_SIZE=32`、全局 `RANK=0–31`。环境约定参照 [阿里云 DLC 分布式训练文档](https://help.aliyun.com/en/pai/product-overview/distributed-training-dlc-quickstart)。

- 每个 Pod 有一个 CPU 协调进程，使用 Gloo 和 DLC 注入的 `MASTER_ADDR:MASTER_PORT`。
- GPU torchrun 使用 `MASTER_ADDR:(MASTER_PORT+1)`，两个端口都需节点间可达。初始节点 rendezvous 最长等待 5 分钟；入口导入/挂载检查失败时由 DLC 结束对应 Pod，其他节点不会无限等待加入。不要额外包一层 torchrun，也不要覆写 DLC 的节点 rank。
- 只有节点 0 执行元数据检查、pipeline 文件锁、阶段状态写入、Stage1 完整覆盖证明验证及 Stage2 单进程 preflight。
- 主节点向各节点发送 GPU 训练命令；各节点只替换自己的 node_rank，不广播本地凭证环境。两个阶段均组成一个 32-rank 训练组。
- 任一节点的训练子进程失败会通知其他节点并终止剩余进程；Linux subreaper 接管异常退出的 torchrun 遗留子进程，回收独立 session 的 GPU workers，避免只结束 launcher。
- 单机 preset 保持原行为。新的多机 preset 使用独立输出目录，不能把 PodCount 单独改回 1 后继续使用此入口。

## 环境、挂载与恢复

当前共享 Python 在 DSW 可启动；已检查 torch 2.7.1+cu128、transformers 5.17.0、diffusers 0.40.0、deepspeed 0.18.5、av、pyarrow、Hydra、OmegaConf、psutil 等包版本。DLC 中仍需验证其与目标镜像/驱动的兼容性；入口会执行依赖导入、baseline/标定路径和每节点 8 GPU 检查。本次未做 4 台真实 GPU 机器上的 NCCL/RDMA 性能验证。

模板挂载 b5、b5big、B/F NAS。训练源路径位于 b5/b5big，标定还需要 `/mnt/data/nas/B_zone_nas/jeffry/calibration_results`；共享环境、baseline 与 tokenizer 位于 b5。Stage1 每 GPU 仍保存完整模型与优化器，更多卡不会降低单卡显存需求。

中断后在 site JSON 设置 `resume: true` 并改 display_name，再生成请求；如果已经进入 Stage2，还需把 `stage2_resume` 设为完整训练状态 checkpoint 路径。pipeline 会验证覆盖记录、索引、配置和 Stage2 身份；不接受更改采样率后继续旧覆盖记录。

## 验证范围

验证包含单机回归、4×8 拓扑与错误环境拒绝、两个节点连续启动 Stage1/Stage2 的真实 CPU torchrun、leader-only 协调、失败阻止阶段提升、独立 session 子进程回收、JSON schema 和 CLI dry-run。CPU 小模型测试证明启动与协调逻辑，不代表已经完成全量准备、32 卡大模型训练或镜像兼容性验证。

## 提交排队与飞书定时通知

用户已授权提交此任务，以及每 3 小时向“通知我！”群发送 DLC 链接、名称、loss 图、进度和 ETA；不再查询 quota 空闲容量。

必须先在 DSW 完成 CPU 全量准备与验证，再提交 GPU 任务。site JSON 必须指定 CPFS 上的 `prepared_marker`；生成器、`submit.sh` 和 GPU 入口都会即时检查完成索引、源 YAML 校验和、采样率与统计 registry。未准备完成直接拒绝提交或启动，不再等待后台准备进程。`priority` 支持 1–9，H200 本次重提设为 9。

2026-10-25 离线产物集中放在 `/mnt/data/cpfs/b5/yeqingda/openwam-0911/OpenWAM/outputs/offline_stage12_3fps_20261025`：`population.sqlite` 为可续建的 3 FPS 样本索引，`population.progress.json` 为持久进度，`stage2_metadata_audit.json` 为全来源审计，`statistics/registry-*.json` 为归一化统计，`validation/` 为 Mecka 修复后逐源真实 forward，`pipeline.yaml` 与 `dlc_site.json` 为本次配置。`prepared.json` 仅在全部准备及验证通过后写入。

索引准备 `scripts/train_goal_epoch.py --prepare` 默认使用 `--index-workers 32 --index-max-pending 64 --index-commit-every 256`，保留 v3 索引已有完整事务。并发只改变读取和提交效率，样本顺序与 ID 不变。设置 `TMPDIR`、`SQLITE_TMPDIR` 和缓存目录至 CPFS；不向 `/root/train` 写入准备数据。恢复前不要覆盖已有索引或修改源标签。

准备和训练共用完整 episode 读取器：episode 跨 Parquet 边界时，根据真实 episode_index 汇集各分片，并验证本地帧号及全局行号；末帧 TCP 标签也从完整分片集合读取。Mecka 使用 source_frame_index 指向原视频解码序号，覆盖当前、目标与历史帧。缺失或重复行会报出来源和 episode，不静默剔除。

每任务的 `scripts/pai/monitor_stage12.py` 从 DLC 状态与共享 Stage1/Stage2 `progress.jsonl` 取数据，复用已安装的 `dlc-loss-monitor` 飞书发送适配器。发送前校验接收群名称为“通知我！”。不扫描或推送其他任务，不复制凭据到仓库。

Stage1 每步记录归约后的 loss；Stage2 在成功 optimizer update 的第 1 步、每 10 步和最后一步记录归约后的 total/video/action/FK 等指标。两阶段曲线分别绘制。ETA 使用最近窗口中的真实耗时；断点恢复前后的 elapsed 不混用，超过 1 小时无更新则暂停外推。Stage1 期间只能估算 Stage1 的完成时间，Stage2 建立吞吐后才估算整场结束时间。

监控 daemon 首次立即推送，之后间隔 10800 秒；每分钟检查任务是否结束，终态推送成功后退出。它在 DSW 后台运行，不依赖聊天继续打开，但 DSW 停止/重启会中断它。PID、日志、发送回执和下次推送依据保存在该 JobId 对应的 monitor 输出目录；重启同一个监控可读取 state.json 延续推送时刻，文件锁防止重复实例。
