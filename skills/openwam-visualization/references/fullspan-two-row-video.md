# 完整指令、每10帧、GT/预测两排视频

用户已经选定布局和采样间隔，不再反复确认。默认每个数据集1个episode即可满足1–2个；明确列数据集范围。使用smoke清单定位原始episode，不能把smoke截取片段冒充完整指令。优先完整subtask；该来源无subtask才使用完整task。

## 时间与推理

从源指令标签读取[start,end_exclusive)，anchors=`range(start,end_exclusive,10)`并去重补上`end_exclusive−1`。每个anchor真实读取当前三视图、同episode历史、实际指令与proprio，分别运行冻结Stage1和所比较的Stage2；禁止只预测一次再滚动背景。每个结果绑定anchor、episode、输入hash和权重hash，原子保存，恢复运行时核验身份。

当前frame与未来action chunk的+1…+32步分别记录。越过指令/episode边界的GT尾部保持mask，不把padding当真实未来。Stage1目标固定为该完整指令的最后一帧；Stage2的名义horizon和最后有效动作帧不同。最后不足10帧的间隔需记录，不能声称是10帧。

源视频逐帧解码证据与训练监督有效性不同。若照片实际存在而canonical.frame_valid/training.valid仅因追踪/状态失效为false，不应在展示中把照片变黑。用于推理的视觉有效性必须有源frame/PTS/尺寸/hash证据；可以使用独立的报告推理适配器，保留action/proprio/FK监督mask不变并记录差异。真正缺图则留黑并标无法预测，不能强制打开mask。

## 四列、两排

| | 当前头相机 | 当前左腕相机 | 当前右腕相机 | Stage1关键帧 |
|---|---|---|---|---|
| GT | 当前真实图＋未来32步GT轨迹 | 同时刻真实图＋本相机投影 | 同时刻真实图＋本相机投影 | 指令结束帧真实图＋GT TCP |
| 预测 | 同一当前真实图＋未来32步预测轨迹 | 同一当前图＋本相机预测轨迹 | 同一当前图＋本相机预测轨迹 | 新预测图＋预测二维TCP |

整段视频上持续显示选中task/subtask文本、当前源帧、关键帧目标源帧、方法及有效action步数。左右手以L/R标记；GT绿色、预测红色，无GT的Stage1二维输出橙色。避免把完整配置、长审计表或状态术语塞进主视频。缺相机黑图；有图无标定保留图，用短文字说明不能投影。

未来32点全部用每个**当前anchor相机**的K/E/畸变投影到当前图。这里不同于未来视频逐帧叠对应时刻TCP：不能调用后者后不加区别地复用其相机。腕图独立查相机名、内外参、native尺寸和crop/resize；拒绝generic-head参数回退。图像缩放后，点使用相同320×256tile坐标；在拼入四列两排前画点，或者显式应用拼图偏移。

GT来源可以是原生动作监督或独立源端点标注，必须记录来源。预测原生TCP通道不可用时不应让有效源GT一起消失；也不能把源GT复制成预测。FK在用户需要且映射可信时单独导出同布局视频，明确标FK，不替换direct含义。源TCP是否对应指垫中心按trajectories.md核验，未知时不宣称接触标定已通过。

## 实现入口

在包含下列脚本的活动实验checkout执行，使用项目已有Python环境：

1. `scripts/experiments/fullspan_samples.py`：完整区间选择、每anchor独立样本、源解码和输入身份。
2. `scripts/experiments/fullspan_inference.py`：每anchor新Stage1/joint/none推理；读取已冻结checkpoint，不训练。
3. `scripts/experiments/fullspan_projection.py`（内部模块，无独立CLI）：当前三相机几何、32步世界轨迹、独立源GT。
4. `scripts/experiments/fullspan_video.py --root RUN --watch`：完整序列结果就绪后编码两排视频，并更新简单HTML列表。

先跑一个真实anchor核验模型输入/输出与时间，再启动全量；大任务报告实际完成数和实测ETA，不能把正在生成写成已完成。避免重新构造训练器/优化器。可缓存同文件stat未变时已核验的权重hash，不可用未知旧hash跳过首次核验。

## 最小交付验收

- manifest覆盖每个选中dataset/episode完整指令，优先subtask的选择有依据；视频源帧序列必须与完整anchors精确相等。
- 每个输出视频帧都对应同一anchor下新的Stage1/Stage2结果；GT和预测两排当前图、指令、相机和目标身份一致。
- 检查缺腕图为黑格、有图无标定仍有图；source GT存在但prediction不可用的情况不混淆。
- 解码全部视频核对帧数、首/中/末帧、最后目标帧、MP4/H264/yuv420p；检查实际叠点像素，不能仅验证JSON坐标存在。
- 原生video controls支持播放、暂停和拖动。展示采样FPS和源FPS，保持端点完整覆盖；没有浏览器实播时如实说明。
- 对较长MP4检查服务的Range请求：`Range: bytes=0-1023`应返回206、正确Content-Range和对应字节，而不是整文件200。当前checkout的`serve_reports.py`使用已有aiohttp FileResponse支持分段读取；替换服务前先用备用端口验证原报告路径和Range，再保留原公开端口，避免破坏既有链接。HTTP/编码验证与浏览器实播分别记录。
- 主页面每episode只放joint/none两段主视频和指令/范围；详细坐标、缺失原因、physical endpoint及hash存侧车JSON。交付网页URL与本地视频路径。

## 可重复执行命令

`tools.py` 目录已注册 `fullspan-samples`、`fullspan-inference`、`fullspan-video`、`fullspan-validate`、`report-server`。明确指定实际含这些脚本的 checkout；主仓库与实验 worktree 不一定同步。先用 `plan` 查看命令及依赖，再用 `run` 执行。`--python` 指向项目环境，避免系统 Python 缺少 torch/av。

```bash
python3 "$SKILL_DIR/scripts/tools.py" --repo "$CHECKOUT" --python "$PYTHON" run fullspan-samples -- --config "$CONFIG" --output "$RUN/data"
python3 "$SKILL_DIR/scripts/tools.py" --repo "$CHECKOUT" --python "$PYTHON" run fullspan-inference -- --run "$TRAINING_RUN" --manifest "$RUN/data/manifest.json" --output "$RUN" --anchor-major
python3 "$SKILL_DIR/scripts/tools.py" --repo "$CHECKOUT" --python "$PYTHON" run fullspan-video -- --root "$RUN"
python3 "$SKILL_DIR/scripts/tools.py" --repo "$CHECKOUT" --python "$PYTHON" run fullspan-validate -- --root "$RUN"
```

`--anchor-major` 同时保留三个模型，先实测显存；资源不足时省略它按阶段推理。当前 selection 默认每来源一个完整范围，修改选样必须保留完整区间和真实输入身份。

## 缺点和跳动的诊断证据

对用户指出的坏例子，输出一条明确的诊断记录：`source_repo / episode / prompt / anchor / checkpoint`、Stage1 TCP有效数、Stage2 TCP/joint/gripper/FK各通道有效数、训练采样证据、归一化裁剪比例、原始世界轨迹误差及相邻步距离、各相机有效/出画计数。日志证明的优化器采样、训练清单中可用、仅参与统计这三种“见过”分别说明。

- 有源GT但模型缺原生TCP映射：保留上排GT，下排标不可用。检查训练profile与统计有效计数，不能靠开启renderer mask补出预测。
- 有预测但明显跳动：先比较原始normalized action、反归一化世界坐标和像素坐标；检查quantile clipping与坐标回代误差。世界空间已经跳动时，不能用图像平滑宣称修复。
- Stage1无有效TCP监督：该二维输出不计算TCP误差、不标为可信物理投影；图上保留橙色和原因。
- 腕部存在K/E还不够：`visual_candidate_manual_selected`、CAD预览或未审核外参不能进入可信对比。保留腕图，标标定未验证，来源与拒绝原因写JSON。
- 过拟合需要用户的训练授权；固定样本、指令、checkpoint、采样noise与steps作前后比较。分别报告训练loss、真实采样误差、GT裁剪引入的误差下限；单样本成功不等于整段或全数据集通过。

运行注意：现有 `RUN/data/manifest.json` 会被复用，换配置或重选episode时使用新RUN；当前选样按family各一个，不等于所有source_repo各一个，也没有任意stride/episode选择CLI。`--first-anchor-gate` 仅在 `--anchor-major` 生效。未完成推理时，非watch renderer可以正常退出但只产生部分视频；交付必须核对coverage与validation状态及总数。

## GT 投影覆盖验收

逐视频解码、坐标复算和像素颜色检查只能证明已有点被按当前公式画出，不能将空 mask 当成 GT 投影通过，更不能认证物理标定。每次交付运行 `python -m scripts.experiments.projection_coverage --root "$RUN"`，提供 `projection_coverage.html/json`：按数据集、相机分别统计有图、相机可用、GT世界点、画面内/外点及原因。候选标定和可信标定分开标识，须在真实源帧检查 TCP 对齐；对于全部未画点的视图须主动披露。
