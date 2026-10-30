# Stage1 / Stage2 视频对比合同

适用于模型预测报告。只检查输入、不运行模型的报告继续使用 inputs-and-gripper.md，勿强行增加模型预测。已有训练授权不受此 skill 的“可视化不启动训练”范围描述撤销；可按任务运行推理，保留训练资产。

## 逐案例页面

每个案例展开后，在顶部完整呈现当前三视图拼图、task/subtask、模型实际 conditioning text、数据源/episode/anchor、checkpoint/step；不存在的 subtask 标“未提供”，不要把 task 自动填入 subtask。

随后依次展示 Stage1、Stage2 direct、额外生成视频、FK。各栏可标不可用，但须写原因；不能只在折叠末尾给 N/A。有效投影视频默认可见，原始无投影视频可作为附加查看。joint / none 等方法使用同一案例，不按“同家族”“相近图像”拼接；多个快照以同一语义合同逐一验收。

**图内帧号**：当前输入、原始预测展示副本、预测叠点图、真实GT叠点图均直接标 `INPUT f… → task/subtask TARGET f…`；同时标选定指令区间 `[start,end_exclusive)` 和结束边界。标注放图下方新增区域，保留原始像素、UV原点、模型输出和源图片；不要在添加标题后忘记图像偏移。裸GT、生成图和叠点图使用同一个实际目标tile尺寸；不能一幅保持原始长宽比，另一幅按训练尺寸拉伸后直接比较坐标。

`frame_contract` 必含 `input_frame`、`instruction_kind`、`instruction_start_frame`、`instruction_end_exclusive`、`target_frame`、`stage2_horizon_last_frame`。input等于真实anchor，target严格等于选中指令的end_exclusive−1；Stage1也携带相同frame_contract和goal_frame。Stage2另列最后有效action/video帧及padding mask；名义horizon超出指令范围时不能标成真实观测。帧号来自源episode索引，不是切片数组下标或播放器序号。

Stage1 预测图与 TCP 需要真实 forward。存储输入身份 `(dataset, episode, anchor_frame, conditioning_text)`、输入/历史 tensor hash、权重 hash、预测坐标、目标时刻和有效 mask。Stage1 的 instruction-end 目标与 Stage2 的 action-horizon 末帧通常不同，分别标注帧号。不能把 Stage2 joint 生成末帧当作 Stage1 输出。

同模型单帧/历史对比使用同 checkpoint、同案例、同噪声，只切换历史输入；不自动承诺两个独立模型。历史槽是否参与 attention、各帧来源、无效视图可展开查看。

多个 Stage2 快照共享同一个冻结 Stage1 时，案例顶部可共享一套 Stage1 预测，但须记录 `stage1.checkpoint_sha256`，每个变体的 `stage1_checkpoint_sha256` 必须一致。若 Stage1 权重不同，分别创建带权重身份的案例/报告并实际推理，不能复用旧 Stage1 图。

## 轨迹与相机

- direct：恢复 normalization、动作单位和表示；delta 相对真实观测锚点；从已验证物理端点/坐标系转换到每个真实帧的相机。关节曲线不是 TCP 投影。
- FK：匹配机器人 URDF、关节顺序/单位、基座变换、工具端点及相机；FK 后再次投影，独立标记。source-anchor registration 不能写成独立物理标定。
- 真视频对齐每一帧的 K、E、畸变、resize/crop/拼图偏移和源 frame_index。头部相机的投影不可平移到腕图冒充腕相机标定；没有腕图标定时标记仅头视图投影。
- 使用真实背景逐帧投影点，可叠短尾轨迹；移动相机的历史轨迹若画在当前图，必须把历史世界点用当前相机重新投影。固定当前背景曲线只能作为辅助。
- GT-valid、prediction-exists、finite、projectable、in-frame、metric-eligible 是不同状态。预测不能只因 GT 无效而消失；可以投影但无 GT 的预测明确标未监督/未验证，不计误差。缺物理映射或掩蔽未训练的通道不叫有效预测。
- Stage1 `coordinate_output_kind=stage1_2d_model_output`，GT0时 `prediction_status=unverified_no_gt`；`ready`只表达文件已生成。无GT栏目不能命名“GT+预测TCP”，明确只显示二维预测。每例呈现物理端点定义和来源；用户质疑夹爪位置时按 trajectories.md 的 TCP 排查顺序检查，禁止把未验证UV误报为标定投影。
- 出画点保持原值/误差和出画计数，不夹到画面边界；NaN/相机后方单独报告，不画伪点。首观测帧没有未来预测时只画观测 GT。
- 统一使用 GT 绿方框、预测红圆圈及 L/R 标签；需要改变颜色时全报告同图例。无 GT 的未验证二维预测可用橙色，明确不代表已校准 3D 轨迹。

## 视频可用性与 RGB

所有变体都保留真实源视频，即使 none 无生成视频或当前案例无法投影。使用 MP4/H.264/yuv420p、HTML video controls；提供帧步进/共享时间轴以比较同案例多个视频。短 horizon 的 4fps 是预览速度，需列实际采样帧、源 FPS/时间，不能宣称原视频速度或连续全帧。

RGB 一致性覆盖当前/历史/目标/训练张量/生成图保存链路。检查原始源、解码结果、pre-VAE 实际输入和网页；预测输出不做展示专用 R/B 交换来掩盖训练错误。

## 完整性清单与自动检查

当前实现可在包含这些脚本的实验 checkout 中复用（检查活动 worktree，不能误跑旧主目录脚本）：

- `scripts/experiments/infer_stage1_samecase.py --run RUN`：从实际 `data/samples.pt` 的当前图、历史与指令运行冻结 Stage1；目标图/动作不进入输入 allowlist。已有输出只在输入和权重 hash 匹配时复用。
- `scripts/experiments/audit_stage1_samecase.py --run RUN`：从源数据核对当前拼图、instruction-end 目标与可用 TCP 重投影。这证明计算链一致，不证明独立物理标定正确。
- `scripts/experiments/report_stage12_contract.py --run RUN --validator "$SKILL_DIR/scripts/validate_stage12_contract.py" --decode-videos`：组合同案例结果与初始/最终快照，生成 `RUN/report/stage12_comparison.html` 和合同清单。它依赖已有 `report/assets` 的 Stage2 几何/视频；缺输出先补生成，不从其他案例借图。
- `scripts/experiments/stage12_projection.py`、`stage12_video.py`：逐帧相机投影及 GT/预测叠点实现。修改后要重新生成受影响的所有案例/快照，不能只改一张图。

写 `stage12_contract.json`，以实际评估 manifest 的案例列表填 expected_cases；所有变体/快照都列入 expected_variants，例如 joint_step0、joint_step530、none_step0、none_step602。不能从成功输出反推 expected 列表。schema：

```json
{
  "schema_version": 1, "page": "comparison.html",
  "expected_cases": ["000", "014"],
  "expected_variants": ["joint_step530", "none_step602"],
  "cases": [{
    "case_id": "000",
    "identity": {"dataset": "source/repo", "episode": "1", "anchor_frame": 100, "conditioning_text": "pick"},
    "task": "pick", "subtask": "", "granularity": "task",
    "frame_contract": {"input_frame": 100, "instruction_kind": "task", "instruction_start_frame": 0, "instruction_end_exclusive": 200, "target_frame": 199, "stage2_horizon_last_frame": 132},
    "current_image": "case000/current.png",
    "source_video": {"status": "ready", "assets": ["case000/real.mp4"]},
    "stage1": {"status": "ready", "identity": {}, "frame_contract": {}, "goal_frame": 199, "coordinate_output_kind": "stage1_2d_model_output", "valid_gt_count": 0, "prediction_status": "unverified_no_gt", "assets": ["case000/stage1.png", "case000/tcp.png"], "evidence": ["case000/stage1.json"]},
    "variants": {}
  }]
}
```

示例中的 identity、frame_contract 和 variants 需填实际内容：stage1.identity 与 stage1.frame_contract 必须分别与案例的同名字段相同，另加真实 `checkpoint_sha256`；每个 variant 含 `step`、`stage1_checkpoint_sha256`、`direct`、`generated`、`fk` 三栏。ready 栏包含实际相对资产路径，轨迹栏加 `evidence` 指向坐标/相机 JSON。unavailable 含具体 `reason` 和已有证据文件；not_applicable 仅用于结构无生成视频或无关节/FK分支，需 reason。漏做 inference/漏渲染写 missing，检查返回失败。仅有关节但暂无 FK 映射写 unavailable，不是架构无 FK。

源GT的valid mask与物理接触中点认证分别报告：`physical_contact_review` 记录具体审核证据；数值复算通过但图像上仍偏离夹爪时标 `visual_suspect_unresolved`，在该案例Stage1和Stage2区域都显示原因。其他没有独立接触位置证据的案例标 `not_independently_certified`，不因 GT2/2 或模型误差小就自动认证。可增加GT-only总览，先检查源端点，再评价模型；不能仅以预测散点猜测几何错在哪里。

```bash
python "$SKILL_DIR/scripts/validate_stage12_contract.py" /path/report/stage12_contract.json \
  --decode-videos --output-json /path/report/stage12_contract_check.json
python "$SKILL_DIR/scripts/validate_report.py" /path/report/comparison.html \
  --require-video --require-offline --output-json /path/report/assets_check.json
```

第一项检查案例/变体全覆盖、同锚点、指令边界/目标帧、二维输出类型与GT状态、必需栏、资产、视频 controls，并可完整解码。检查通过且有 unavailable 意味着**完整呈现了局限**，不等于所有投影已补齐。它不执行 JS，不证明实际叠点、真实背景、标定或同步正确。第二项检查全部 HTML 相对引用。图内文字还需检查实际显示副本，并比对保留的原始像素区域；JSON里有帧号不等于图上已标。

还必须检查实际视频首/中/末帧与坐标 JSON：有 valid in-frame 预测时像素里确实存在对应标记（不能只测 renderer 输入），暂停/拖动位置正确、同步后帧索引相等。覆盖每个有投影的案例/变体；其他案例验证不可用原因。抽查不能代替全清单完整性。错误坐标系和错误 RGB 需要回溯源数据/输入，不是多画几个点。

交付 skill/报告路径、全量案例覆盖、真实限制、验证结果。发布网页遵循 delivery.md；localhost200+hash 与 DSW302登录是不同结果。无需重训或篡改 mask 来让可视化“通过”。
