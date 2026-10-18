# Frozen T5 instruction feature cache

训练在 `WanBase.preprocess_input_for_train → encode_text` 中缓存冻结 T5 的
`context`（含置零的 padding）与对应 `seq_lens`。缓存位于可训练文本投影、
state/proprio token 之前；这些投影仍逐步计算并更新梯度。图像增强和 VAE 编码
仍按原流程执行，不会因为文本缓存而复用旧图像。

## 配置

### 先离线生成，再启动训练

准备 `instructions.json`（字符串数组，例如 `["pick up the cup", "place the cup"]`），
也支持 TXT 每行一条或 JSONL 每行 `{"prompt": "..."}`。重复指令自动去重。
离线入口只加载 checkpoint 中的 T5 和 tokenizer，不加载 DiT/VAE，不读取图像：

```bash
python scripts/cache_t5_text_features.py \
  --checkpoint /path/to/OpenWAM-checkpoint \
  --prompts-file instructions.json \
  --cache-dir /path/to/t5_cache \
  --dtype bfloat16 --device cuda \
  --report outputs/t5-precache.json
```

`--checkpoint` 支持自包含 checkpoint 目录（自动选最新权重）或具体
`checkpoint_step_*.safetensors` 文件；同目录需有 `config.yaml` 和 tokenizer。
离线步骤可重复执行：已有记录直接复用，缺失记录才编码；报告包含命中/未命中计数。

启动训练时设置 `training.text_feature_cache.enabled=true`，并把
`training.text_feature_cache.directory` 指向**同一个绝对路径**。
训练 T5 权重、dtype 和运行精度需匹配；不匹配时自动使用新命名空间重新生成，
不会把不兼容特征作为命中结果。训练遇到离线列表未覆盖的指令时会自动补齐。

### 训练配置

当前 `openwam_alpha_v0_1_smoke` 配置默认开启；`train` 通用配置默认关闭。
从仓库根目录启动，例如：

```bash
/root/miniconda3/envs/openwam/bin/torchrun --standalone --nproc_per_node=1 \
  scripts/train.py --config-name=openwam_alpha_v0_1_smoke \
  training.text_feature_cache.enabled=true \
  training.text_feature_cache.directory=outputs/cache/t5 \
  training.text_feature_cache.max_memory_mb=512
```

现有 `outputs/curated_gripper_human_20261014/run.py` 也 compose 此 smoke 配置，
因此下次启动直接生效。只修改通用 `train.yaml` 不会覆盖这个独立配置。

通过 `training.text_feature_cache.enabled=false` 关闭。使用其他自定义配置时，
如该节点不存在，Hydra 命令行使用 `+training.text_feature_cache.enabled=true`。
缓存路径相对启动工作目录解析；跨实验/多卡复用应指向同一绝对目录。

## 行为与边界

- 启用后要求 T5 全部参数冻结，否则报错；T5 保持 eval，关闭文本 dropout。
  因而与此前误处于 train 模式的冻结 T5 存在明确的训练行为区别。
- 首次遇到指令时编码一次并原子写入磁盘；再次使用命中 CPU LRU。
  每个 rank 的 LRU 上限为 `max_memory_mb`；超限后仍可从磁盘读取。
- 重启不需要重新编码已缓存指令。首次使用时仍会读取实际 encoder 参数，
  计算一次 SHA256 指纹；这属于启动成本，不是每个 step 的耗时。
- 命名空间包含实际权重、encoder 实现文件、dtype 和运行精度设置；每条记录
  以 token IDs 和 attention mask 为键。换权重、dtype、tokenizer/清洗结果后
  自动使用对应缓存。恢复权重在第一次编码前完成，指纹不依赖旧 checkpoint 路径。
- 未命中时逐条编码，保持当前 smoke 单条预处理的数值路径；原生批量 T5
  编码与逐条编码可能有浮点舍入差异。顺序与重复指令均保持不变。
- 缓存不写入模型 state_dict，也不增加优化器参数。损坏记录会记录 warning
  并重建。多 rank 可以重复计算首个 miss，但原子替换保证读取完整文件。
- 缓存磁盘空间随不同指令增长，不自动清理；磁盘只存有效 token，读取时补零。
  完整单条 bf16 `512×4096` 约 4 MiB，有效 32 token 的记录约 256 KiB。
  `max_memory_mb` 只限制常驻 CPU 缓存，不限制当前 batch 输出 tensor。

缓存对象在 `architecture.video_backbone._text_feature_cache`，其 `stats` 包含
`memory_hits / disk_hits / misses / memory_bytes`，可用于检查是否命中。

## 验证

```bash
/root/miniconda3/envs/openwam/bin/python -m pytest tests/test_text_feature_cache.py -q
```

覆盖重复指令/顺序、跨实例磁盘复用、权重和 tokenizer 变化、dtype、LRU、
损坏记录、冻结检查、T5 eval、下游投影梯度和图像增强后独立重算 VAE。

### 本机真实权重验证（2026-10-18）

使用 foundation `checkpoint_step_154000`、bf16 T5、320 条真实指令输入
（45 条不同指令），逐条预处理，与当前 smoke 路径一致。仅计文本部分：

| 场景 | 耗时 |
| --- | ---: |
| 不缓存，逐条编码 | 19.96 s |
| 首次填充缓存 | 9.37 s |
| 内存命中 | 1.72 s |
| 新缓存实例从磁盘复用 | 2.63 s |

首次填充/磁盘复用的上述耗时不含每次启动首次使用的权重指纹计算（约 9 s）。
三种缓存路径的 context 和 seq_lens 均与不缓存的 T5 eval 逐元素一致；
新实例复用时 encoder 调用为 0。45 个磁盘记录共 6,761,397 bytes，
CPU LRU 中补齐后的特征占 180 MiB。这组数值仅测文本编码，不能直接等同整步提速。

原始结果：`outputs/t5_feature_cache_20261018/verification.json`。
