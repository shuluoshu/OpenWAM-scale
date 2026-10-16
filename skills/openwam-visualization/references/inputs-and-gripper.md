# 当前输入、同帧对比和gripper

## 当前fixture走生产adapter

export_current_inputs.py调用当前CuratedFixtureDataset(manifest)，保留RGB缓存、原视频时刻、过滤后窗口索引、归一化、GT和mask。不是重解码原始视频。输出reader_frames.png、真实bf16 pre-VAE第0帧的显示还原、model_inputs.npz、trace.json（语言/源帧/contract/源码hash）和index.html。它是输入/数值曲线报告，不是轨迹视频或完整模型forward。

已有监督示例manifest：outputs/curated_gripper_human_20261014/data_manifest.json。“最新”先查运行config/label_contract/status，不从日期猜。sample-index是该clip的过滤后索引，trace记录原始anchor。

```bash
/root/miniconda3/envs/openwam/bin/python "$SKILL_DIR/scripts/export_current_inputs.py" \
  --manifest outputs/curated_gripper_human_20261014/data_manifest.json \
  --clip 08_abc_130k --output-dir outputs/visualization_review/abc
```

原始source profile可用scripts.visualize_model_inputs --profile ... --output-dir ...。裁剪fixture不能裸build_dataset后假设原视频shard偏移正确；使用CuratedFixtureDataset或已验证build_clip_reader。通用renderer历史pipeline文案也不能代替当前contract。

本机reader/像素处理用/root/miniconda3/envs/openwam/bin/python。绘图用系统python3，其有matplotlib，训练环境没有；避免为展示修改训练环境。

## 同一源帧OpenWAM/Mora

历史驱动/产物在outputs/curated_smoke_20261010/comparison/；Mora源码默认../../pretrain，需要本地Qwen processor、源视频/episode、stats和缓存。

```bash
python3 outputs/curated_smoke_20261010/comparison/prepare_images.py --ids 08_abc_130k
python3 outputs/curated_smoke_20261010/comparison/prepare_gt.py --ids 8
python3 outputs/curated_smoke_20261010/comparison/render_pairs.py
python3 outputs/curated_smoke_20261010/comparison/render_forward_inputs.py
```

prepare_gt的ids是整数前缀，prepare_images是完整id。后两个无CLI，处理既有manifest；forward图依赖9张model_frame PNG、image_metrics和Mora图，旧缓存不代表最新reader。原始图像解码后再走两边真实变换，显示JPEG不作模型输入。记录hash、resize/crop/pad、K变化、dtype、归一化和缺相机处理。

**当前exporter不能直接串接历史forward renderer。** 前者是当前fixture的一条样例，后者的目录/manifest schema固定。要求“最新同帧对比”时，在新目录适配driver：从trace取得过滤后真实源anchor（含原始row.start偏移）和相机对应frame_index，重新解码双方同一源帧；保存当前OpenWAM各时间帧、重新运行Mora真实resize/processor得到image_metrics；把forward renderer的输入目录/manifest读取改为新产物，并核对源图hash和patch反变换。不要只把最新pre-VAE PNG贴到旧Mora图旁边。此skill提供入口和边界，不承诺跨版本一键重建整个Mora流水线。

OpenWAM当前示例空间拼成320×384画布，9帧形成[1,3,9,384,320] pre-VAE tensor，偏移[0,4,...,32]，以profile为准。Mora相机独立处理，拼patch序列而非RGB画布；报告并排布局不属于模型输入。展示pixel_values、image_grid_thw和相机边界。pre-VAE/processor还原不是完整forward。

Mora读存储actions.eef，OpenWAM motor目标取next state，端点和未来offset可能不同。先记录字段和目标索引，再做同端点/同目标时刻的表示对照，不能用absolute/relative差异掩盖时间不一致。

## Gripper同步视频

历史gripper_audit.py→render_gripper_audit.py→analyze_gripper_targets.py无CLI，固定缓存，含reader_before_gripper01.py和state+1旧假设。复用其视频/曲线/跳变分析前，接入当前run manifest和生产contract，输出新报告目录。不能直接重跑历史驱动并宣称GT已更新。

当前权威是`openwam/dataloader/utils/gripper.py`、`openwam/dataloader/utils/mora_relative.py`、`openwam/dataloader/mora_relative_lerobot.py`及实际gripper_command_sources：

- 输入measured state[t]；机器人目标是有来源、已对齐cmd[t+k]；开合度clip[0,1]，0闭/1开，不套片段quantile。
- LightWheel/Mecka取拇指尖到其余四指尖距离最小值归一化，EgoDex/EgoVerse取拇指-食指pinch距离。是有效人手GT，不因没有硬件cmd而mask。
- 已审核实现读actions.eef第9/19维，保留已存时序；抽样LightWheel/Mecka为state+1，EgoDex/EgoVerse为state+3，不再shift，不推广全repo永久相同offset。
- 人手proxy不作为独立cmd证据豁免尖峰。无有效目标、NaN/Inf、target_valid=false和padding仍mask；缺TCP不影响有效gripper。
- 图中同时展示state、原始候选cmd、实际模型GT、mask、源帧和对应视频；未确认来源明确标候选。过滤索引映回源帧。
- 历史测试-0.2/1.4是合成边界值，不能冒充真实数据越界证据。

旧报告46片段/57视频的断言仅适用当时完整产物。新子集按实际manifest组装，不靠修改断言制造通过。
