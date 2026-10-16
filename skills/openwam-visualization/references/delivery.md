# 本地HTML、视频检查和网页链接

validate_report.py检查HTML的src/href/poster/data：文件存在、无绝对源路径、无逃出交付root。require-offline拒绝外部运行时资源，外部资料超链接不等于运行时依赖；require-video防止静态图页面冒充视频交付。

decode-videos N用OpenCV完整解码前N个不同视频（含base64内嵌），记录帧数、FPS、时长、fourcc和编码帧多样性；默认0只查引用。H264/avc1适合浏览器，还要核对yuv420p、faststart和实际播放，任意可解码编码不一定兼容浏览器。

该工具不执行浏览器、锚点、CSS/JS动态资源或源背景运动检查。需另核对相同frame_index的原始背景hash、相机/pose时间戳、首/中/尾contact sheet，并查看代表视频。叠加线变化也能使编码帧变化。

```bash
python3 "$SKILL_DIR/scripts/validate_report.py" outputs/curated_smoke_20261010/index.html \
  --require-video --require-offline --decode-videos 2 \
  --output-json outputs/visualization_review/delivery_check.json
```

纯输入图报告不传require-video。跨兄弟目录引用需选择包含全部资产的共同--root并完整交付，不能只扩大检查root却只发一张HTML。

本地交付可选单HTML内嵌JS/图片/MP4（体积较大），或HTML配相对资产完整文件夹/ZIP（解压打开index.html）。无需CDN。新子集直接压缩自包含报告目录；旧package_report.py硬编码57视频，只适合完整历史报告。ZIP检查CRC和相对路径，不意外打包源数据、训练权重或另一个ZIP。

## 服务器链接

用户需要网页时先检查既有服务root/端口，验证返回的是本次HTML hash，可复用，不随意重启。

```bash
python3 -m http.server 18081 --bind 0.0.0.0 --directory outputs/fk_consistency
```

确认端口可用，后台运行记录PID/日志/root。云平台VS Code浏览器通常无法直连服务器10.x私网IP，应使用该实例的端口转发，例如：

```text
https://<当前实例>-proxy-<端口>.dsw-gateway-<区域>.data.aliyuncs.com/<相对路径>
```

从环境或用户当前有效链接取得实例/区域，不照抄旧897628/877303等实例号。验证本机HTML/媒体HTTP200及字节hash；网关302到PAI登录表示需要认证，不是HTML不存在，也不证明匿名可访问。交付实际验证状态和绝对本地路径。

报告附样例范围、坐标系/端点图例、GT或模型预测阶段、标定诊断限制及证据链接。可视化任务不改现有训练/标定/mask。
