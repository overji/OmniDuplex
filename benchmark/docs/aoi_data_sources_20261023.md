# AOI 数据源切换核查（2026-10-23）

本机实际目录是 `/remote-home/yushengliao/speech/jimodels/datasets/`（复数）。
`/remote-home/yushengliao/speech/jimodels/dataset/` 不存在，本次配置使用现有目录。

| 数据源 | QA | 视频 | 视频引用检查 | Agent 负样本 |
|---|---:|---:|---|---:|
| AOI-Dataset | 5,000 | 1,268 | 全部存在 | 0 |
| AOI-Bench | 2,107 | 852 | 经别名解析后全部存在 | 0 |

## 内容一致性

- 训练：与 `legacy/20261023T120552Z_train_data/train_data/5kqa_1023_all_qa.zip` 的全部 5,000 条记录匹配。对比时仅忽略 `video`、`video_folder`、`dataset_source` 来源字段，其余字段完整比较，包括问题、回复、工具参数和时间窗口。
- 评测：与 `benchmark/results/full_omni_duplex_plus_ab_gemini37_20261021/inputs/all_qa.json` 的 2,110 条记录按同一口径比较，保留 2,107 条，旧记录独有 3 条，新记录独有 0 条。剔除的是缺失素材 `video_103.mp4` 对应 QA。
- 使用 `.env` 中的 ModelScope 认证，只下载了 `overji/AOI-Dataset` 和 `overji/AOI-Bench` 的标注、metadata 和说明文件。两份远程 `all_qa.json`、训练 `video_metadata.json`、评测 `video_manifest.json` / `manifest.json` 与本地逐字节一致；没有重新下载全部视频，也没有修改远程仓库。

| 文件 | SHA-256 |
|---|---|
| AOI-Dataset/all_qa.json | `3809db9b78f1f3cadee16cafc13b7fd0133ee17ee84e451428ac87f376014d30` |
| AOI-Bench/all_qa.json | `a9382251c48c6fa791cd0d24a5314fb8715056b861895919347061ac0457f537` |

## 路径配置

训练 QA 中 `video` 为 `videos/<UUID>.mp4`，训练及 JEV 配置中的 `videos_dir` 因此指向 AOI-Dataset 根目录，`data` 指向该目录下的 `all_qa.json`。

评测视频平铺在 AOI-Bench/videos，但 QA 保留 571 个 `A/videos/`、131 个 `B/videos/` 和 150 个平铺引用。准备脚本根据发布的 `video_manifest.json` 建立 `benchmark/cache/aoi_bench_videos` 软链接目录，全部 852 个引用解析成功；不改写原始 QA 或视频。

重新下载或迁移数据后，在仓库根目录运行：

```bash
python benchmark/scripts/prepare_aoi_data.py
```

自定义目录使用 `--dataset-dir`、`--bench-dir`、`--output-dir`，并同步修改训练配置和评测命令参数。完整下载流程见仓库 README。

## 实际验证

- benchmark 路径默认值及别名测试：8 项通过。
- train 数据处理测试：6 项通过。
- JEV 核心测试：10 项通过。
- 使用真实 AOI-Dataset、默认训练配置、不传 `--data`，CPU prepare 成功选取 12 条训练、2 条验证记录；2,810 条因原有 32 秒前缀上限被排除，未调整该上限。
- 正在运行的 Agent 重评计划中的所有协议文件哈希保持一致；未启动新 GPU 训练或评测。

本次媒体核查验证文件路径存在和解析关系，未逐个重新解码视频或核验全部媒体内容哈希。

## 原始导入文件归档

原 `train/data` 的四个文件已移动至 `legacy/20261023T120552Z_train_data/train_data`，由根目录 `/legacy/` 规则排除 Git 管理。四个文件移动前后 SHA-256 与大小逐一一致；明细在归档目录的 `manifest.json`。当前训练直接读取 AOI-Dataset，不依赖该归档。
