# Stage1目标观察有效性最小修复

新分支：`codex/stage1-goal-mask-fix-20261118`。基于归档runtime的`1fece35`，不改历史`runtime/`、`reviewed/`与`provenance/`文件，不改源数据、检查点或优化器。本次只做DSW七条固定样本验证，不启动DLC。

## 规则

- 目标仍为选中task/subtask的`end_exclusive-1`；task:subtask=3:7、anchor选择、1Hz严格因果历史、时间PE、KF1固定0和模型参数结构保持原样。
- `native_stage2_io.source_frame_validity`及Stage2下一步动作规则保持原样。新的`goal_validity.stage1_goal_validity`只服务Stage1目标观察。
- 仅当family为Genrobot、info中的`starvla_metadata.source=genrobot`，且action明确为`absolute_next_step`、`frame_offset=1`、`last_frame_policy=copy_last_available_masked`、canonical字段存在时，Stage1目标观察不再依赖`training.valid`。canonical与source.frame_validity仍须有限且大于0.5；已知相机的video_alignment_valid继续约束目标视图。精确解码失败仍报错，不复制当前帧。
- 不满足显式来源合同的来源继续沿用严格旧规则。TCP仍独立核验每只手EEF、标定、投影与画内有效性；RGB恢复不会强制恢复TCP。`success_verified=False`保持原值。
- `goal_generation.generate_goal_latents`只接收当前帧、文本和因果历史，输出布局从当前`observed_view_valid`决定，目标GT张量和mask不进入生成。现有监督loss的GT mask保留。

## 代码、缓存与合同

`entrypoint.py`显式替换归档helper的G生成入口，不改归档helper字节。runner identity额外绑定entrypoint及维护的generation helper的SHA；core fingerprint也会变化。旧run必须被严格续训检查拒绝，不能改合同hash来掩盖差异。

`stage1-goal-observation-v2`写入目标标签身份、reader provenance、evaluation资产目录和RFC总体身份。历史帧source/cache key不变；原帧像素、VAE与预处理一致时历史latent仍可复用。生产frame cache的目标key已经包含views，原来invalid的目标没有有效target slot；新有效目标按新视图编码。G为三视图T=1、C*为head T=1，必须独立编码，不能裁剪G latent替代C*目标。

七条prepared及评分另存，不替换历史prepared、报告与分数。原检查点只严格加载权重，未加载优化器；检查点SHA、原source合同与新source差异写入推理证据。这是明确的推理代码迁移，不是绕过训练resume合同。

## DSW复现

使用已安装运行环境的Python（本机为`/tmp/openwam-runtime-20261031/env/bin/python`），选择**不存在的新输出目录**。以下参数均为已有只读资产；环境移植时按实际挂载替换。

```bash
python scripts/evaluation/validate_goal_observation_fix.py prepare \
  --prepared /mnt/data/cpfs/b5/yeqingda/CodeX/experiments/rfc_stage1_compare_20261111/data \
  --prepared-cstar /mnt/data/cpfs/b5/yeqingda/CodeX/experiments/rfc_stage1_compare_8gpu_20261111/baseline_cstar/data \
  --runs-root /mnt/data/cpfs/b5big/yeqingda/CodeX/storage/longtrain_16gpu_20261112_20261115/runs \
  --output /path/to/new/seven-case-generation

python scripts/evaluation/validate_goal_observation_fix.py infer \
  --prepared /mnt/data/cpfs/b5/yeqingda/CodeX/experiments/rfc_stage1_compare_20261111/data \
  --prepared-cstar /mnt/data/cpfs/b5/yeqingda/CodeX/experiments/rfc_stage1_compare_8gpu_20261111/baseline_cstar/data \
  --runs-root /mnt/data/cpfs/b5big/yeqingda/CodeX/storage/longtrain_16gpu_20261112_20261115/runs \
  --output /path/to/new/seven-case-generation --device cuda:0

# 在同一DSW回放修复前的归档生成逻辑，避免跨设备旧PNG形成混杂。
python scripts/evaluation/validate_goal_observation_fix.py baseline \
  --prepared /mnt/data/cpfs/b5/yeqingda/CodeX/experiments/rfc_stage1_compare_20261111/data \
  --prepared-cstar /mnt/data/cpfs/b5/yeqingda/CodeX/experiments/rfc_stage1_compare_8gpu_20261111/baseline_cstar/data \
  --runs-root /mnt/data/cpfs/b5big/yeqingda/CodeX/storage/longtrain_16gpu_20261112_20261115/runs \
  --output /path/to/new/seven-case-generation --device cuda:0

python scripts/evaluation/render_goal_observation_fix.py \
  --prepared /mnt/data/cpfs/b5/yeqingda/CodeX/experiments/rfc_stage1_compare_20261111/data \
  --runs-root /mnt/data/cpfs/b5big/yeqingda/CodeX/storage/longtrain_16gpu_20261112_20261115/runs \
  --output /path/to/new/seven-case-generation
```

固定val索引8、13、14、20、34、57、60。5条Wipe来自同一episode193，目标3726；另2条episode3目标4814，canonical无效，应维持无效。全部是task样本，不冒充subtask整体评估。两套检查点分别5515与3321步，不同训练步数下不能把原生loss直接横比。

验收：原manifest/context SHA、目标源图SHA、当前源图/实际预处理像素、原输入latent与history字段、指令、granularity和目标帧都一致；恢复5张RGB/10个TCP点；另2条指标仍null但G/C*都真实生成；记录复制基线、变化区域、TCP邻域误差。TCP单位是resize后320×256头图像素，`norm([319Δu,255Δv])`。

本次DSW实测结果（2026-11-18，两组都未更新权重）：

| 模型 | 有效图像/点 | Head RGB MAE | TCP平均欧氏像素误差 | 复制当前帧全图MAE | 变化区域改进 | TCP邻域改进 |
| --- | --- | --- | --- | --- | --- | --- |
| G5515 | 5 / 10 | 0.186325 | 27.9405 | 0.186025 | +0.006574 | -0.029801 |
| C*3321 | 5 / 10 | 0.193715 | 35.2856 | 0.186025 | +0.026174 | -0.085079 |

“改进”是复制MAE减预测MAE（RGB范围0–1）。两者变化区域稍优于复制，TCP邻域仍差于复制；标签修复通过，但现有权重的任务预测效果仍有限。同一DSW的前后回放中C*7条图像浮点输出和TCP逐值相同，G7条TCP逐值相同；G图像由全零latent占位变为真实生成。7条旧协议都没有有效GT，因此不将原null分数称作被改进的旧分数。

完整证据：`/mnt/data/cpfs/b5/yeqingda/CodeX/docs/2026-11-18/goal_mask_fix_dsw/`。中文页：<http://10.1.128.54:8012/goal_mask_fix_20261118/index.html>。浏览器验收包含63张嵌入图片、GT/pred叠加开关、5/2筛选和1440/390宽度无页面溢出。

```bash
python -m pytest tests/goal/test_goal_observation_validity.py \
  tests/goal/test_goal_mask_validation_publication.py \
  tests/goal/test_curated_goal.py tests/goal/test_rfc_goal_data.py \
  tests/goal/test_longtrain_source_snapshot.py tests/dataloader/test_head_history.py \
  tests/goal/test_kf1_condition_and_regions.py -q
python training/stage1_longtrain_16gpu/verify_snapshot.py
```

## 当前边界与未来续训

七条manifest是本次DSW诊断输入，**不能传给要求64条的生产validation16**。归档默认配置仍指向历史prepared集，保留旧分数可复现；此次没有把历史fixed64标签就地更新，也没有自动恢复训练。

后续用户授权续训后，需要另建完整诊断prepared generation（包括64条验证及所需profile记录），重建所有受影响目标标签/编码与manifest；通过显式新配置绑定。再逐项审计采样集合/顺序与依赖身份，生成新旧合同迁移记录，并严格保留权重、优化器和步数。不能只改code_dir或把本次七条目录当作完整训练输入。标签缺失修复不会让现有权重补学历史漏掉的监督，效果改善需后续训练验证。
