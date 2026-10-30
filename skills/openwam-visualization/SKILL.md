---
name: openwam-visualization
description: Use when OpenWAM or MoraVLA needs Stage1/Stage2 prediction comparisons, TCP/action/FK projection videos, same-frame model inputs, gripper supervision visualization, or HTML experiment reports. 适用于预测视频漏投影、训练结果对比与模型输入检查。
---

# OpenWAM 预测与轨迹可视化

复用已验证的渲染器，将坐标系、端点、目标时刻和有效mask一起交付。需要OpenWAM checkout和相应数据/缓存；跨仓库对比还需要Mora源码和本地processor。只封装可视化，不启动训练。

## 默认交付：两排完整指令视频

当前用户已指定以下布局与采样规则；按此生成，不再默认交付逐案例多栏目大页面。详细执行与验收见 [fullspan-two-row-video.md](references/fullspan-two-row-video.md)。

- 每个数据集抽1–2个episode，优先完整subtask，没有subtask才用完整task。每隔**10个源帧**重新推理，并补区间最后一帧；视频覆盖整段指令。
- **上排GT**：当前头图、当前左腕图、当前右腕图，分别叠加当前时刻未来32步action chunk的GT轨迹；第四格为Stage1目标关键帧及GT TCP。
- **下排预测**：相同的当前三视图，分别叠加本时刻新预测的未来32步轨迹；第四格为Stage1新预测关键帧及二维TCP。
- 图内显示实际task/subtask指令、当前源帧号、关键帧目标帧号。目标为选中指令`end_exclusive−1`，不混用Stage2的32步末帧。
- 缺失腕图保留黑格；有图但无可信标定则显示原图并标无法投影。每个相机使用自身**当前时刻**K/E/畸变，把32个未来世界点投影到该当前图；头相机坐标不能复制到腕图。
- 产物是可以播放、暂停和拖动的MP4；HTML只是视频列表。两种Stage2各产出同样的两排视频。额外生成视频或FK只作单独附加视频，不改变主布局。

GT可用性和模型预测可投影性分开判断。源端点标注有效时，即使模型该通道无有效映射，上排GT仍可显示；不能伪造下排预测。Stage1橙点是无有效GT的二维模型输出，不叫物理标定投影。缺项在对应格里简短标明，细节放坐标/来源JSON。

用户指出TCP不在夹爪末端中间时，先读 [已有端点修正与排查顺序](references/trajectories.md#tcp-偏离夹爪中心时)。模型误差、源端点定义和相机投影错误分别核验；数值复算一致不证明物理接触位置正确，不能挪预测点迎合图像。

旧的逐案例审计页和校验器仍可用于诊断：[stage12-video-contract.md](references/stage12-video-contract.md)，不代替本节要求的整段两排视频。

## 按任务选择

| 任务 | 参考 | 入口 |
|---|---|---|
| 整段Stage1/Stage2 GT/预测两排视频 | [fullspan-two-row-video.md](references/fullspan-two-row-video.md) | `fullspan-samples` / `fullspan-inference` / `fullspan-video` / `fullspan-validate` |
| 头/腕相机、TCP-local视频、URDF/FK | [trajectories.md](references/trajectories.md) | `trajectory` / `wrist` / `expanded` / `fk-audit` |
| 当前reader拼图、gripper GT/mask | [inputs-and-gripper.md](references/inputs-and-gripper.md) | `export_current_inputs.py` |
| 同源帧OpenWAM/Mora图像/轨迹、最终pre-VAE/patch组织 | 同上 | `mora-images` / `mora-gt` / `forward-inputs` |
| gripper state/cmd同步视频、跳变分析 | 同上 | `gripper-audit` / `gripper-video` / `gripper-compare` |
| 本地HTML、MP4检查、离线交付和服务器链接 | [delivery.md](references/delivery.md) | `validate_report.py` |

## 使用入口

将SKILL_DIR指向本skill目录，在仓库根执行；可用`--repo`或`OPENWAM_REPO`指定checkout。安装版用下方路径；仅检出仓库时改为`SKILL_DIR="$PWD/skills/openwam-visualization"`。

```bash
SKILL_DIR="${CODEX_HOME:-$HOME/.codex}/skills/openwam-visualization"
python3 "$SKILL_DIR/scripts/tools.py" list --check
python3 "$SKILL_DIR/scripts/tools.py" plan trajectory -- --samples interndata --no-report
```

`list`/`plan`只检查文件和打印命令，不导入/执行渲染器。把`plan`换成`run`才执行。部分旧脚本没有CLI，不要用它们的`--help`探测。完整目录和依赖、参数差异、输出位置见[tools.json](references/tools.json)。

导出当前训练fixture真实输入：

```bash
/root/miniconda3/envs/openwam/bin/python "$SKILL_DIR/scripts/export_current_inputs.py" \
  --manifest outputs/curated_gripper_human_20261014/data_manifest.json \
  --clip 04_LightWheel --output-dir outputs/visualization_review/lightwheel
```

上例是已有样例路径；“最新”必须核对当前manifest/config/contract和源码hash。输出实际reader帧、bf16 pre-VAE像素显示、GT曲线、mask、NPZ和HTML；没有执行VAE/模型forward。

```bash
python3 "$SKILL_DIR/scripts/validate_report.py" /path/report/index.html \
  --require-video --require-offline --decode-videos 2 --output-json /path/report/check.json
```

纯图像输入报告不传`--require-video`。检查器不证明CSS/JS动态资源或源背景真实运动；编码帧变化不能代替源帧hash验证。

## 保留这些语义

- 真移动视频同步更新背景、TCP anchor和相机；固定背景对照明确标注。保留frame_index、视频shard偏移、时间戳、实际FPS。
- TCP、法兰、腕部不能混称；两套训练pose用同一物理TCP。缺TCP两套pose mask，缺头相机仅head mask；有效gripper独立监督。
- 当前gripper输入用measured state，机器人GT用来源确认的cmd，人手可用已审核距离开合度目标，均clip[0,1]。不能把“没有机器人cmd”推广为所有ego gripper mask。读取当前gripper.py和实际contract。
- relative旋转用矩阵乘法；joint delta相对固定当前步。表示回代自洽不能认证相机/TCP物理标定。
- 旧驱动固定manifest、缓存、总量断言或历史GT假设，不能直接称为最新输入。只要求可视化时，不改源数据、训练标定/监督mask或正在运行的实验。

交付绝对本地路径、网页链接（如有服务）、源样例范围和实际验证。单条样例通过不代表整个数据集。
