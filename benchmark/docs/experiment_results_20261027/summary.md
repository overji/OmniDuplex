# Benchmark实验结果汇总

快照：2026-10-27T11:36:39.168807+00:00。

可独立编译：`cd benchmark/docs/experiment_results_20261027 && pdflatex main.tex`。
主表、消融附表及正文分析均在`experiments.tex`内，可直接复制或通过\input引入论文，无需其他表格文件。
LaTeX正文聚焦一张分组主表和一张消融附表；其余覆盖率、细粒度任务和内容judge统计保留在本Markdown及CSV中。
主表逐组数据见`grouped_main_results.csv`，分母及未舍入分数见`grouped_main_results.json`。短/中/长视频按≤120秒、120–300秒（不含120）、>300秒分组。总体为micro汇总；最优值按模式、指标、分组加粗，并列最优同样加粗。

## 数据对应和比较范围

- 远端模型名由用户确认：Gemini-3.7-flash、Qwen3.8Omni；原始文本不修改，pair ID按各自导出数据校验后统一索引。
- 当前主表范围：merged full sources。
- 500条QA中494条有合法Probe配对，共1154对。
- 导出集和旧2110条运行数据的sample_index有479处不同，已按整条QA指纹建立唯一映射；原始文件没有修改。
- 五模型主表仅比较当前主表范围内共同完成两侧的配对；前500条表保留为历史子集诊断，不与全量分母混用。
- 远端模型的API部署ID、prompt、媒体采样和token预算未随文件提供，比较仍受运行配置差异限制。
- 前500条不是随机样本；原始预测的缺失不能直接认定为API失败。
- 以下均沿用官方scorer，没有修复/恢复模型答案；解析决策与保存值不一致数：{"minicpmo": 0, "qwen3_omni": 0, "qwen25_omni": 0, "remote_gemini": 0, "remote_qwen": 0}

## 主要发现

- 当前主表五模型共同完成4163对、8326次请求（1953条QA）；Gemini-3.7-flash与Qwen3.8Omni的成对准确率分别为42.95%、27.38%。
- Probe只判断是否应开口，不判断回复内容正确性；各分组使用共同完成配对，总体按计数汇总。
- Realtime共同2008条QA上，MiniCPM/Gander/JoyAI的时间F1为25.31%/18.47%/14.86%。Gander更保守、精确率更高；JoyAI输出更多，召回较高而精确率偏低。JoyAI仅输入视觉。
- MiniCPM在2107条QA上，音视频/仅视觉/仅音频的时间F1为25.24%/20.63%/9.21%；Agent子集并未呈现相同的优势顺序。
- 既有内容judge中，两种模式都回复的90个reference response上，Realtime/Probe回复均分为1.317/1.244（满分2）。QA聚类bootstrap差值区间跨0，不能据此确认总体优劣；Agent调用评分只有7对。

## 实验表格

### 远端500+1607合并覆盖

输入共2107条公开QA，合法Probe为9188次请求。

| 模型 | 前500请求 | 后1607请求 | 合计成功请求 | 完整配对 | 覆盖QA |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gemini-3.7-flash | 2273 | 6486 | 8759 | 4378 | 2002 |
| Qwen3.8Omni | 2261 | 6466 | 8727 | 4350 | 2001 |

缺失请求保持缺失，不按静默或错误补齐。完整来源、哈希和索引映射保存在 `benchmark/results/from_remote_1027/merged_full/`。

### 分组主表：probe

| Model | Metric | RTD | PR | Agent | Perception | Comprehension | Reasoning | Short | Medium | Long | Overall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MiniCPM-o-4.5 | Pre | 2.38 | 17.67 | 30.89 | 10.86 | 13.88 | 6.07 | 15.52 | 9.17 | 5.86 | 11.15 |
| MiniCPM-o-4.5 | Post | 98.08 | 94.45 | 82.09 | 94.71 | 92.81 | **97.17** | 93.84 | 94.44 | 95.88 | 94.43 |
| MiniCPM-o-4.5 | Pair | 1.88 | 14.47 | 19.18 | 7.83 | 9.87 | 4.72 | 11.02 | 6.60 | 4.75 | 8.05 |
| Qwen3-Omni-30B-A3B | Pre | 1.63 | 11.56 | 64.46 | 14.91 | 16.74 | 10.80 | 15.95 | 14.15 | 14.26 | 14.87 |
| Qwen3-Omni-30B-A3B | Post | **99.50** | **97.65** | 57.40 | 92.00 | 91.53 | 92.17 | 91.93 | 92.09 | 90.97 | 91.86 |
| Qwen3-Omni-30B-A3B | Pair | 1.30 | 9.68 | 25.95 | 7.73 | 9.10 | 4.32 | 9.11 | 7.13 | 5.39 | 7.64 |
| Qwen2.5-Omni-7B | Pre | **93.01** | **99.72** | 39.49 | **84.28** | **87.71** | **84.48** | **86.45** | **85.17** | **84.79** | **85.61** |
| Qwen2.5-Omni-7B | Post | 8.03 | 1.69 | 0.00 | 7.62 | 1.65 | 5.80 | 5.48 | 5.03 | 3.96 | 5.04 |
| Qwen2.5-Omni-7B | Pair | 2.68 | 1.50 | 0.00 | 2.81 | 0.76 | 2.16 | 2.65 | 1.68 | 0.79 | 1.92 |
| Gemini-3.7-flash | Pre | 30.59 | 66.73 | **65.59** | 44.79 | 52.77 | 33.47 | 50.25 | 45.70 | 34.55 | 45.78 |
| Gemini-3.7-flash | Post | 94.77 | 96.52 | **97.60** | **95.73** | **96.63** | 93.66 | **94.46** | **96.23** | **97.31** | **95.70** |
| Gemini-3.7-flash | Pair | **27.45** | **64.00** | **63.61** | **42.36** | **50.35** | **28.74** | **46.74** | **42.98** | **33.12** | **42.95** |
| Qwen3.8Omni | Pre | 20.42 | 46.62 | 39.92 | 27.98 | 37.75 | 21.05 | 31.47 | 31.71 | 23.93 | 30.43 |
| Qwen3.8Omni | Post | 94.94 | 93.23 | 97.46 | 94.71 | 94.72 | 95.95 | 94.21 | 95.18 | 96.04 | 94.93 |
| Qwen3.8Omni | Pair | 17.82 | 41.64 | 38.22 | 25.01 | 34.31 | 18.62 | 27.83 | 28.77 | 22.03 | 27.38 |

### 分组主表：realtime

| Model | Metric | RTD | PR | Agent | Perception | Comprehension | Reasoning | Short | Medium | Long | Overall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MiniCPM-o-4.5 | Recall | 21.98 | **33.86** | **12.95** | 22.00 | 26.89 | **19.21** | **28.67** | 24.83 | 11.05 | 23.43 |
| MiniCPM-o-4.5 | Precision | 35.79 | 29.89 | 10.01 | 26.13 | 31.50 | 22.07 | 32.63 | 27.56 | **15.88** | 27.52 |
| MiniCPM-o-4.5 | F1 | **27.23** | **31.75** | 11.29 | **23.89** | **29.01** | **20.54** | **30.52** | **26.12** | **13.03** | **25.31** |
| Gander | Recall | 9.44 | 19.63 | 10.44 | 13.12 | 11.97 | 9.85 | 15.55 | 12.93 | 4.14 | 12.10 |
| Gander | Precision | **36.15** | **40.61** | **45.66** | **37.73** | **47.71** | **28.07** | **47.80** | **42.81** | 13.73 | **38.99** |
| Gander | F1 | 14.97 | 26.47 | **16.99** | 19.47 | 19.13 | 14.59 | 23.47 | 19.86 | 6.37 | 18.47 |
| JoyAI-VL-Interaction | Recall | **34.56** | 20.42 | 6.34 | **26.10** | **30.71** | 18.10 | 26.22 | **28.73** | **22.21** | **26.52** |
| JoyAI-VL-Interaction | Precision | 9.94 | 16.07 | 5.22 | 9.68 | 11.72 | 8.49 | 14.01 | 11.29 | 5.76 | 10.32 |
| JoyAI-VL-Interaction | F1 | 15.44 | 17.99 | 5.72 | 14.13 | 16.96 | 11.56 | 18.26 | 16.21 | 9.14 | 14.86 |

### Probe comparison on the first-500 subset

| Model | Positive Acc. | Negative Acc. | Balanced Acc. | Paired Acc. | Format Valid | Agent Call Valid |
| --- | --- | --- | --- | --- | --- | --- |
| MiniCPM-o-4.5 | 95.10 | 9.79 | 52.45 | 6.80 | 79.24 | 69.53 |
| Qwen3-Omni-30B-A3B | 91.57 | 13.69 | 52.63 | 6.26 | 99.59 | 95.86 |
| Qwen2.5-Omni-7B | 5.89 | 84.68 | 45.29 | 1.81 | 88.53 | -- |
| Gemini-3.7-flash | 95.29 | 45.97 | 70.63 | 42.79 | 96.37 | 90.79 |
| Qwen3.8Omni | 93.29 | 31.10 | 62.19 | 26.75 | 99.95 | 99.65 |

Probe comparison on the first-500 subset. All values are percentages. Models share 1103 fully completed pre/post pairs (2206 requests from 483 QA). Agent call validity is conditional on an Agent reply, not semantic correctness. Remote model names are operator-confirmed; exact inference settings were not included with predictions.

LaTeX：`probe_first500_common.tex`；CSV：`probe_first500_common.csv`。

### First-500 prediction coverage

| Model | Requests | Expected | Complete Pairs | Expected Pairs | QA |
| --- | --- | --- | --- | --- | --- |
| MiniCPM-o-4.5 | 2232 | 2308 | 1115 | 1154 | 486 |
| Qwen3-Omni-30B-A3B | 2308 | 2308 | 1154 | 1154 | 494 |
| Qwen2.5-Omni-7B | 2308 | 2308 | 1154 | 1154 | 494 |
| Gemini-3.7-flash | 2273 | 2308 | 1136 | 1154 | 493 |
| Qwen3.8Omni | 2261 | 2308 | 1124 | 1154 | 493 |

Coverage of provided predictions on the first-500 QA export. Missing rows are not assumed to be model errors or correct silence.

LaTeX：`probe_first500_coverage.tex`；CSV：`probe_first500_coverage.csv`。

### Remote first-500 results (historical subset)

| Model | Requests | Complete Pairs | Positive Acc. | Negative Acc. | Balanced Acc. | Paired Acc. | Format Valid | Agent Call Valid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gemini-3.7-flash | 2273 | 1136 | 95.33 | 46.09 | 70.71 | 43.05 | 96.35 | 90.91 |
| Qwen3.8Omni | 2261 | 1124 | 93.22 | 31.11 | 62.17 | 26.87 | 99.96 | 99.65 |

Scores from the original first-500 remote files only, using each model's own coverage. Accuracy denominators contain completed requests; paired accuracy uses only completed pre/post pairs. Values are percentages. Use the shared-subset table for comparisons with local models.

LaTeX：`probe_remote_supplied.tex`；CSV：`probe_remote_supplied.csv`。

### Probe paired accuracy by interaction

| Model | RTD | PR | Agent |
| --- | --- | --- | --- |
| MiniCPM-o-4.5 | 1.88 | 14.47 | 19.18 |
| Qwen3-Omni-30B-A3B | 1.30 | 9.68 | 25.95 |
| Qwen2.5-Omni-7B | 2.68 | 1.50 | 0.00 |
| Gemini-3.7-flash | 27.45 | 64.00 | 63.61 |
| Qwen3.8Omni | 17.82 | 41.64 | 38.22 |

Paired accuracy by interaction on the same primary shared subset. Values are percentages; both pre and post decisions must be correct.

LaTeX：`probe_interaction.tex`；CSV：`probe_interaction.csv`。

### Probe comparison after merging first500 and remaining1607

| Model | Positive Acc. | Negative Acc. | Balanced Acc. | Paired Acc. | Format Valid | Agent Call Valid |
| --- | --- | --- | --- | --- | --- | --- |
| MiniCPM-o-4.5 | 94.43 | 11.15 | 52.79 | 8.05 | 81.11 | 76.89 |
| Qwen3-Omni-30B-A3B | 91.86 | 14.87 | 53.36 | 7.64 | 99.60 | 97.22 |
| Qwen2.5-Omni-7B | 5.04 | 85.61 | 45.33 | 1.92 | 89.26 | -- |
| Gemini-3.7-flash | 95.70 | 45.78 | 70.74 | 42.95 | 96.96 | 90.81 |
| Qwen3.8Omni | 94.93 | 30.43 | 62.68 | 27.38 | 99.93 | 99.46 |

Probe comparison after merging first500 and remaining1607. All values are percentages. Models share 4163 fully completed pre/post pairs (8326 requests from 1953 QA). Agent call validity is conditional on an Agent reply, not semantic correctness. Raw source text is unchanged. Split IDs are normalized through full-QA fingerprints and verified pair IDs.

LaTeX：`probe_full_common.tex`；CSV：`probe_full_common.csv`。

### Full-source Probe coverage and descriptive scores

| Model | Requests | Expected | Complete Pairs | Expected Pairs | QA | Pre Acc. | Post Acc. | Pair Acc. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MiniCPM-o-4.5 | 8650 | 9188 | 4319 | 4594 | 1994 | 10.99 | 94.51 | 7.96 |
| Qwen3-Omni-30B-A3B | 9187 | 9188 | 4593 | 4594 | 2047 | 13.97 | 92.38 | 7.23 |
| Qwen2.5-Omni-7B | 9187 | 9188 | 4593 | 4594 | 2047 | 86.09 | 5.18 | 1.89 |
| Gemini-3.7-flash | 8759 | 9188 | 4378 | 4594 | 2002 | 45.45 | 95.59 | 42.55 |
| Qwen3.8Omni | 8727 | 9188 | 4350 | 4594 | 2001 | 30.35 | 94.74 | 27.17 |

Each model uses its own successful coverage here. Missing requests remain missing; use the shared-pair table for model comparisons.

LaTeX：`probe_full_coverage.tex`；CSV：`probe_full_coverage.csv`。

### Local Probe comparison on the full shared subset

| Model | Positive Acc. | Negative Acc. | Balanced Acc. | Paired Acc. | Format Valid | Agent Call Valid |
| --- | --- | --- | --- | --- | --- | --- |
| MiniCPM-o-4.5 | 94.51 | 11.02 | 52.77 | 7.97 | 80.78 | 76.72 |
| Qwen3-Omni-30B-A3B | 91.92 | 14.78 | 53.35 | 7.62 | 99.61 | 97.16 |
| Qwen2.5-Omni-7B | 5.42 | 85.41 | 45.41 | 1.97 | 89.43 | -- |

Local Probe comparison on the full shared subset. All values are percentages. Models share 4318 fully completed pre/post pairs (8636 requests from 1991 QA). Agent call validity is conditional on an Agent reply, not semantic correctness. 

LaTeX：`probe_local_common.tex`；CSV：`probe_local_common.csv`。

### Local full-run snapshots

| Model | Completed | Expected | Positive Acc. | Negative Acc. | Paired Acc. |
| --- | --- | --- | --- | --- | --- |
| MiniCPM-o-4.5 | 8650 | 9188 | 94.51 | 10.99 | 7.96 |
| Qwen3-Omni-30B-A3B | 9187 | 9188 | 92.38 | 13.97 | 7.23 |
| Qwen2.5-Omni-7B | 9187 | 9188 | 5.18 | 86.09 | 1.89 |

Local full-run snapshots with each model's own completion coverage. These rows do not share an identical denominator and are descriptive only.

LaTeX：`probe_local_coverage.tex`；CSV：`probe_local_coverage.csv`。

### Qwen2.5-Omni Probe modality ablations

| Model | Positive Acc. | Negative Acc. | Balanced Acc. | Paired Acc. | Format Valid | Agent Call Valid |
| --- | --- | --- | --- | --- | --- | --- |
| Audio + video | 5.18 | 86.09 | 45.63 | 1.89 | 89.96 | -- |
| Video only | 8.80 | 86.78 | 47.79 | 3.20 | 93.96 | 100.00 |
| Audio only | 1.98 | 94.51 | 48.25 | 0.85 | 96.20 | -- |

Qwen2.5-Omni Probe modality ablations. All values are percentages. Models share 4593 fully completed pre/post pairs (9186 requests from 2047 QA). Agent call validity is conditional on an Agent reply, not semantic correctness. 

LaTeX：`qwen25_probe_ablations.tex`；CSV：`qwen25_probe_ablations.csv`。

### Realtime timing comparison

| Model | QA | Windows | Outputs | Hits | Recall | Precision | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MiniCPM-o-4.5 | 2008 | 4669 | 3976 | 1094 | 23.43 | 27.52 | 25.31 |
| Gander | 2008 | 4669 | 1449 | 565 | 12.10 | 38.99 | 18.47 |
| JoyAI-VL-Interaction | 2008 | 4669 | 11991 | 1238 | 26.52 | 10.32 | 14.86 |

Realtime timing on 2008 shared completed QA. MiniCPM and Gander receive audio-video; JoyAI is vision-only. Agent uses the task-text rerun; RTD/PR reuse previous runs. No content correctness is required for a timing hit.

LaTeX：`realtime_main.tex`；CSV：`realtime_main.csv`。

### Realtime timing F1 by interaction

| Model | RTD | PR | Agent |
| --- | --- | --- | --- |
| MiniCPM-o-4.5 | 27.23 | 31.75 | 11.29 |
| Gander | 14.97 | 26.47 | 16.99 |
| JoyAI-VL-Interaction | 15.44 | 17.99 | 5.72 |

Timing F1 by interaction on the same shared QA as the realtime main table. Values are percentages.

LaTeX：`realtime_interaction.tex`；CSV：`realtime_interaction.csv`。

### MiniCPM realtime modality ablations

| Input | QA | Recall | Precision | F1 | RTD F1 | PR F1 | Agent F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Audio + video | 2107 | 23.30 | 27.53 | 25.24 | 26.96 | 31.41 | 11.29 |
| Video only | 2107 | 19.47 | 21.93 | 20.63 | 20.97 | 24.15 | 13.80 |
| Audio only | 2107 | 6.79 | 14.34 | 9.21 | 9.21 | 12.17 | 4.05 |

MiniCPM modality ablations on 2107 shared QA. Historical baseline reuse, DP differences and single-seed sampling limit causal interpretation.

LaTeX：`minicpmo_realtime_ablations.tex`；CSV：`minicpmo_realtime_ablations.csv`。

### Probe paired accuracy across twelve abilities

| Ability | Pairs | MiniCPM-o-4.5 | Qwen3-Omni-30B-A3B | Qwen2.5-Omni-7B | Gemini-3.7-flash | Qwen3.8Omni |
| --- | --- | --- | --- | --- | --- | --- |
| Attribute Perception | 316 | 11.39 | 9.49 | 3.80 | 46.20 | 28.80 |
| Temporal Grounding | 401 | 6.73 | 6.98 | 1.25 | 40.90 | 26.43 |
| Action Perception | 633 | 7.58 | 5.85 | 4.11 | 41.07 | 22.27 |
| Object Perception | 291 | 8.59 | 13.40 | 2.06 | 56.70 | 34.36 |
| Emotion Recognition | 210 | 4.29 | 4.29 | 1.43 | 23.33 | 11.90 |
| Event Understanding | 540 | 6.11 | 5.93 | 1.48 | 30.19 | 17.04 |
| Text-Rich Understanding | 609 | 14.45 | 14.45 | 0.49 | 62.23 | 44.83 |
| Object Counting | 170 | 8.82 | 7.65 | 0.00 | 56.47 | 38.82 |
| Action Counting | 252 | 7.54 | 3.97 | 0.40 | 60.71 | 42.86 |
| Spatial Understanding | 295 | 4.75 | 5.08 | 4.07 | 39.66 | 29.49 |
| Prospective Reasoning | 184 | 4.89 | 4.89 | 0.54 | 21.20 | 8.70 |
| Causal Reasoning | 262 | 4.58 | 3.05 | 1.15 | 21.76 | 13.36 |

Paired decision accuracy by ability on the primary shared subset. Percentages; no response-content judge is applied.

LaTeX：`probe_abilities.tex`；CSV：`probe_abilities.csv`。

### Probe paired accuracy by cognitive level

| Level | Pairs | MiniCPM-o-4.5 | Qwen3-Omni-30B-A3B | Qwen2.5-Omni-7B | Gemini-3.7-flash | Qwen3.8Omni |
| --- | --- | --- | --- | --- | --- | --- |
| Perception | 1851 | 7.83 | 7.73 | 2.81 | 42.36 | 25.01 |
| Comprehension | 1571 | 9.87 | 9.10 | 0.76 | 50.35 | 34.31 |
| Reasoning | 741 | 4.72 | 4.32 | 2.16 | 28.74 | 18.62 |

Cognitive levels follow the specified taxonomy: T1--T5 perception, T6--T9 comprehension, and T10--T12 reasoning. Counts are pooled before computing accuracy.

LaTeX：`probe_levels.tex`；CSV：`probe_levels.csv`。

### MiniCPM content judging on balanced 200 QA

| Mode | Task | Replies | Response | Instruction | Task Score |
| --- | --- | --- | --- | --- | --- |
| realtime | RTD | 47 | 1.181 | -- | 1.181 |
| realtime | PR | 42 | 1.500 | -- | 1.500 |
| realtime | AGENT | 7 | 0.857 | 0.000 | 0.429 |
| probe | RTD | 233 | 0.768 | -- | 0.768 |
| probe | PR | 108 | 1.259 | -- | 1.259 |
| probe | AGENT | 57 | 0.754 | 1.193 | 0.974 |

Existing DeepSeek Flash judging on the same balanced 200 QA. Scores range from 0 to 2; Agent averages response and instruction scores. Realtime uses timing-matched replies; Probe uses positive-prefix true positives. These are different conditional subsets.

LaTeX：`content_judge.tex`；CSV：`content_judge.csv`。

### Content judging on matched reference responses

| Task | Pairs | Realtime Response | Probe Response | Realtime Instruction | Probe Instruction |
| --- | --- | --- | --- | --- | --- |
| OVERALL | 90 | 1.317 | 1.244 | 0.000 | 1.857 |
| RTD | 42 | 1.214 | 1.012 | -- | -- |
| PR | 41 | 1.500 | 1.500 | -- | -- |
| AGENT | 7 | 0.857 | 1.143 | 0.000 | 1.857 |

Reuse existing independent judgments for the same QA and reference response where both modes replied. The 90 reference responses come from 65 QA; the Agent subgroup has only seven pairs.

LaTeX：`content_paired.tex`；CSV：`content_paired.csv`。

## 解释边界

- Probe正侧准确率不是内容正确率；成对准确率要求前后都判对。Realtime时间F1也不验证语义。
- 只有MiniCPM已有平衡200条QA的DeepSeek Flash内容评分；没有对远端模型补调用内容judge。
- 内容均分按不同条件选中的回复计算，需结合90个共同reference response的配对表，不能直接用不同回复集合的均分排名。
- MiniCPM模态消融复用历史音视频基线，未做多随机种子复验；不能据此作严格单因素因果结论。
- LiveStar及尚未完成的其他模型不以旧协议或不完整子集填入主表；本报告仅记录已明确来源的结果。
- 已撤销的GT历史负侧实验完全排除。没有执行新推理或付费API。

## 复现

`python benchmark/scripts/merge_remote_probe_results.py`
`python benchmark/scripts/build_experiment_report.py --remote-merged-dir benchmark/results/from_remote_1027/merged_full --output-dir benchmark/docs/experiment_results_20261027`

可传`--remote-gemini-name`和`--remote-qwen-name`修改展示标签；不改变原始分数。
`metrics.json`保存统计、来源路径及读取快照哈希；`first500_index_mapping.csv`保存索引映射。
LaTeX在当前机器尚未编译（未发现pdflatex/tectonic）；已检查生成表格和输入引用。
