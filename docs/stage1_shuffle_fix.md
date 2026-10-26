# Stage1 全局 shuffle 修复（2026-10-26）

旧 finite 训练循环把 `cursor + rank` 直接传给 SQLite 数据集，源索引又按目录和 episode 连续存放。此前为了读取局部性保留顺序遍历，结果前 32,000 个样本全是 Genrobot/task；覆盖一次和可恢复并不要求这样顺序训练。

现在默认 `sample_order: global_shuffle`，固定 seed 42 构造全部样本的无放回随机排列，再按 rank 分配排列位置。原始 SQLite ID、指令文本、goal 端点、mask、task/subtask 总量均不变。恢复 cursor 记录排列位置；算法、NumPy 版本、seed、总体大小进入 checkpoint 身份。每个 rank 首次训练前核对排列 SHA256。

实际 3 FPS population 的检查结果：

| 项目 | 结果 |
| --- | --- |
| 总样本数 | 53,822,139 |
| 全排列覆盖 | 每个 ID 恰好一次，无缺失、无重复 |
| 前 1000 个 32 卡 batch | 32,000 个不同样本，覆盖全部 13 类来源 |
| task / subtask | 22,188 / 9,812；1000 个 batch 均同时含两种粒度 |
| 每个 batch 的来源种类数 | 最少 5、平均 8.418、最多 13 |
| 每 rank 排列内存 | 215,288,556 bytes，约 205 MiB 主机内存，零显存 |
| DSW 单进程排列构建和指纹耗时 | 2.04 秒 |
| 排列 SHA256 | `19ec8b2d3ec05258e6959b9cfda38dce2f8e8a5c90e7a60b098f9f0028013934` |

每步新增 `family_counts`、`granularity_counts`、`valid_tcp_points`、`samples_with_tcp`。零权重尾批占位不计入数量；TCP loss 为零时可据有效监督数量区分无标签与预测误差。

验证：91 项 epoch/索引/多机编排相关回归通过，随后补强测试的 14 项定向检查通过。真实小模型训练验证 shuffle 生效、缺失 TCP mask 和精确恢复；独立双进程 Gloo 检查验证中断前后样本顺序与最终权重逐位一致，错误排列身份和旧版无 sampling 身份均在所有 rank 上拒绝。

审计文件：`/mnt/data/cpfs/b5/yeqingda/openwam-0911/OpenWAM/outputs/shuffle_fix_20261026/shuffle_audit.json`。这次检查未执行真实 32 卡 NCCL 或完整随机视频读取测速，不能沿用旧顺序读取的秒/步估算新吞吐。

旧顺序运行不能直接切换模式后 `--resume`。启动修复版必须使用新输出目录；可复用 CPFS 上已经准备好的 source YAML、population.sqlite 和统计 registry。该代码修复及审计本身不修改已运行的 DLC 进程或其历史输出。
