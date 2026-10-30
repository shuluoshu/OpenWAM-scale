# 轨迹、投影、URDF/FK

## 入口和现有报告

基础`visualize_trajectory_videos.py`支持interndata/agibotworld/robodojo/robotwin/genrobot/lightwheel/mecka。ABC等新增来源在`visualize_expanded_sources.py`，其samples为a1_genie1、a1_arx_lift2、a1_split_aloha、robomind_v1_agilex、robomind_v1_fr3、robochallenge_v2_aloha、robochallenge_v2_dosw1、robomind_v2_agilex、abc_130k_yam。

汇总`outputs/fk_consistency/tcp_coverage_v2_v3.html`保留`#previous-fk`；四源原始FK检查是同目录index.html。视频、标定数组和metrics在video_samples/wrist_samples/trajectory_videos/expanded_sources中。不要根据旧文档“四个来源”推断当前覆盖。

按需执行（仓库根目录；绘图用有matplotlib的系统Python）：

```bash
python3 -m scripts.visualize_fk_consistency --output-dir outputs/fk_review
python3 -m scripts.data.video_projection_samples --names interndata --anchor 30 --count 180
python3 -m scripts.visualize_trajectory_videos --samples interndata --no-report
python3 -m scripts.visualize_expanded_sources --samples abc_130k_yam --no-report
```

基础渲染器固定读取outputs/fk_consistency；只给extraction传--output-root不会改变后续renderer目录。--no-report仍会写视频/metrics；--report-only需要所有卡片依赖齐全，不能对空目录执行。腕部工具只有--reuse-samples，依赖agibot_examples/manifest.json，不能编造--samples/--output-dir。详见tools.json。

## 四种视图

1. 真移动头/腕相机：时刻t使用实际图像和相机t，投影未来点。
2. 固定首帧相机：相机、背景固定，仅轨迹窗口滚动，是对照模式。
3. 当前TCP-local 3D：以当前TCP为原点的米制轨迹，不是腕相机像素图。
4. FK：joint→匹配URDF→相同端点的pose，可画3D/投影；无匹配URDF只画原生joint曲线。

```text
local: ΔT(t,k) = inv(T_W_TCP(t)) @ T_W_TCP(k)
Δp_local = R_t.T @ (p_k-p_t); ΔR_local = R_t.T @ R_k
restore: T_W_TCP(k) = T_W_TCP(t) @ ΔT(t,k)
head: Δp = R_CW(t) @ (p_k-p_t)
ΔR_head = R_CW(t) @ R_k @ R_t.T @ R_CW(t).T
pixel = project_model(K_t, distortion_t, T_camera_world(t) @ p_world(k))
```

当前local位置为0，旋转为单位阵（6D不全为0）。回代只证明表示自洽。源参考系未必是world，先核对基座、单位和完整变换链。相机模型按源证据选pinhole/Brown/fisheye，不能编造畸变；crop/resize/pad同步变换K或像素坐标。无外参时保留视频和有效local曲线，投影mask，不用单位矩阵代替。

## 已知端点与FK边界

- 权威入口scripts/data/tcp_endpoint_policy.py。RoboDojo/RobotWin的局部+X 0.12m是用户目视验收定义，完整朝向未独立标定。Agi部分样例当前+X 0.200m是report-only preview；240/245mm是旧调试值，不写成官方TCP或训练标定。
- InternData原生TCP应用每行robot_to_env；Agi缺躯干joint时，双臂相对检验不能证明robot-base绝对FK正确。
- 区分独立基座映射、共享变换消去、由标签推导的诊断；不能从GT拟合每帧基座再报告零误差。
- URDF零位inv(T_base_flange)@T_base_tcp只能在相对链固定时作常量；穿过夹爪可动关节要逐cfg FK。具名gripper_center未必是接触TCP。核对joint顺序、mimic和末端轴。
- 第k项（k从0开始）joint relative为q[t+k+1]-q[t]。FK前按该stats反归一化，加q[t]，再转真实基座/工具链，对同端点GT报告位置m/mm和旋转角度。

## TCP 偏离夹爪中心时

先判断点的来源，不能见到偏移就加工具长度：

| 点的来源 | 可核验路径 | 偏离时应检查 |
|---|---|---|
| Stage1 预测 | `decode_tcp_patch` → 归一化 UV → `stage1_tcp_report.py`，当前320×256用 `[319,255]` | 是学到的二维输出。GT0时没有可信误差基准，离夹爪很远可能是模型预测错误；这里没有可“修正”的相机外参。保留预测原值。 |
| GT 端点 | `scripts/data/trajectory_projection.py::goal_source_tcp` → `project_goal_tcp`；源字段、mask、相机及端点变换 | 确认 source EEF 是腕/法兰/工具/TCP，接触中心是否需要特定变换；检查目标帧、K/E方向、畸变、pixel-center resize/crop。 |
| Stage2 direct | 原生表示和反归一化 → 观测锚点 → `stage12_projection.py` | 相对坐标基准、源 mask、动作步对齐、每帧相机、TCP变换是否一致；未训练/无统计的通道不能当物理预测。 |
| Stage2 FK | `native_stage2_profiles.py` + 匹配URDF/关节顺序/工具变换 → 相机 | 不能把末端link原点直接称为指尖接触中心；检查夹爪可动/mimic链及工具坐标轴。 |

每例列出实际 `definition_id`、`used_fields`、`transforms`、missing reasons 和相机/标定状态。人手 pinch midpoint、手持采集设备TCP、机器人法兰、夹爪接触中点不是一个概念；名称不能全部改成“夹爪中心”。如果用户要求机器人指垫间接触中点，需证明当前定义确实对应它，否则标未验证。

**旧修正的适用范围**：先读当前 `scripts/data/tcp_endpoint_policy.py`。RoboDojo/RobotWin 的+X0.12m是指定源的目视接受定义；部分 Agi +X0.20m 是限定任务/embodiment 的报告预览，不能迁移到 RoboMind、RoboChallenge 或其他型号，也不能反向加到 Stage1 二维预测。Genrobot 的设备端点变换等训练/原生链以 `goal_source_tcp`、`recorded_tcp_transform` 和 `native_stage2_profiles.py` 的实际当前配置为准。核对来源、版本、frame、单位、offset是否已经应用，防止重复平移。

验收分三层：①独立数学实现（如OpenCV或单独相机模型公式）对照原始字段，定位计算错误；②在同源、同帧、同尺寸真实图上单独显示GT，核对端点定义和接触位置；③模型预测与这个GT比较。第一层相等只说明数值自洽，不代表第二层通过。source-anchor拟合、复用同一投影helper重算、marker像素存在，都不能单独证明物理标定正确。

修正经证实的来源/变换/resize错误后，重建受影响报告全部案例和快照；如果标签或训练输入也受影响，明确列出缓存/checkpoint影响并按已授权范围处理。展示修复不能掩盖旧模型学到了错误标签。未经证实的像素平移不作为“夹爪中心修正”。

## ABC真移动腕视频

完整历史驱动在outputs/curated_smoke_20261010/comparison/：

```bash
python3 outputs/curated_smoke_20261010/comparison/prepare_abc_wrist_motion.py \
  --ids 08_abc_130k 09_abc_130k 10_abc_130k 11_abc_130k
python3 outputs/curated_smoke_20261010/comparison/render_abc_wrist_motion.py
```

依赖manifest、fixture、源视频、image/wrist eligibility，以及已有`image_source_left.png`/`image_source_right.png`。渲染器无CLI，处理manifest中全部ABC。历史输出75帧/15fps/5秒（150源帧每2帧展示一次），有MP4、首/中/尾contact sheet、源帧hash和metrics。展示FPS不等于GT采样频率。

`comparison/abc_wrist.html`是待嵌入片段，不是可直接打开的独立页面；它的资源路径以**curated报告根目录**为基准。直接打开会造成comparison/comparison路径重复。可由完整build_report.py嵌入，或在报告根目录创建独立页（文件名选择新名字）：

```python
from pathlib import Path
root = Path('outputs/curated_smoke_20261010')
page = root / 'abc_wrist_review.html'
assert not page.exists(), 'Use a new report name'
fragment = (root / 'comparison/abc_wrist.html').read_text()
page.write_text('<!doctype html><html lang="zh"><meta charset="utf-8">'
               '<title>ABC wrist TCP review</title><style>body{font:16px system-ui;'
               'max-width:1280px;margin:auto}video,img{max-width:100%}</style>'
               + fragment + '</html>')
```

验证时仍以curated报告根目录为--root，交付该页面及其实际引用的comparison子目录；不要只发送片段。

08/10/11腕标定QA-rejected，只能标诊断投影；09外参NaN，只播放视频和有效local曲线。四例head eligibility不通过不能打开训练mask。背景运动需源帧hash变化并逐帧匹配；腕相机与TCP刚性连接时，当前TCP像素近乎不动可能正常。
