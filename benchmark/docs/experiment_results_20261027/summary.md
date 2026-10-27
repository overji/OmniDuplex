# Benchmark实验结果汇总

快照：2026-10-27T01:08:35.893253+00:00。

可独立编译：`cd benchmark/docs/experiment_results_20261027 && pdflatex main.tex`。
主表、消融附表及正文分析均在`experiments.tex`内，可直接复制或通过\input引入论文，无需其他表格文件。
LaTeX正文聚焦一张分组主表和一张消融附表；其余覆盖率、细粒度任务和内容judge统计保留在本Markdown及CSV中。
主表逐组数据见`grouped_main_results.csv`，分母及未舍入分数见`grouped_main_results.json`。短/中/长视频按≤120秒、120–300秒（不含120）、>300秒分组。总体为micro汇总；最优值按模式、指标、分组加粗，并列最优同样加粗。

## 数据对应和比较范围

- 两份远端JSONL全部通过前500条导出文件的pair ID校验；模型名由用户确认：Gemini-3.7-flash、Qwen3.8Omni。
- 500条QA中494条有合法Probe配对，共1154对。
- 导出集和旧2110条运行数据的sample_index有479处不同，已按整条QA指纹建立唯一映射；原始文件没有修改。
- 五模型主表仅比较共同完成两侧的配对；另列各模型原始覆盖与本地全量交集，不把这些分母混用。
- 远端模型的API部署ID、prompt、媒体采样和token预算未随文件提供，比较仍受运行配置差异限制。
- 前500条不是随机样本；原始预测的缺失不能直接认定为API失败。
- 以下均沿用官方scorer，没有修复/恢复模型答案；解析决策与保存值不一致数：{"minicpmo": 0, "qwen3_omni": 0, "qwen25_omni": 0, "remote_gemini": 0, "remote_qwen": 0}

## 主要发现

- 五模型共同完成1103对、2206次请求；Gemini-3.7-flash与Qwen3.8Omni的成对准确率分别为42.79%、26.75%。
- MiniCPM和Qwen3的正侧准确率超过90%，负侧不足15%，主要问题是过早开口；Qwen2.5表现为相反的沉默倾向。这里只判断是否开口，不判断内容正确性。
- Realtime共同2008条QA上，MiniCPM/Gander/JoyAI的时间F1为25.31%/18.47%/14.86%。Gander更保守、精确率更高；JoyAI输出更多，召回较高而精确率偏低。JoyAI仅输入视觉。
- MiniCPM在2107条QA上，音视频/仅视觉/仅音频的时间F1为25.24%/20.63%/9.21%；Agent子集并未呈现相同的优势顺序。
- 既有内容judge中，两种模式都回复的90个reference response上，Realtime/Probe回复均分为1.317/1.244（满分2）。QA聚类bootstrap差值区间跨0，不能据此确认总体优劣；Agent调用评分只有7对。

## 实验表格

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

### Remote results on all supplied predictions

| Model | Requests | Complete Pairs | Positive Acc. | Negative Acc. | Balanced Acc. | Paired Acc. | Format Valid | Agent Call Valid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gemini-3.7-flash | 2273 | 1136 | 95.33 | 46.09 | 70.71 | 43.05 | 96.35 | 90.91 |
| Qwen3.8Omni | 2261 | 1124 | 93.22 | 31.11 | 62.17 | 26.87 | 99.96 | 99.65 |

Scores from all supplied remote predictions, using each model's own coverage. Accuracy denominators contain completed requests; paired accuracy uses only completed pre/post pairs. Values are percentages. Use the shared-subset table for comparisons with local models.

LaTeX：`probe_remote_supplied.tex`；CSV：`probe_remote_supplied.csv`。

### Probe paired accuracy by interaction

| Model | RTD | PR | Agent |
| --- | --- | --- | --- |
| MiniCPM-o-4.5 | 1.36 | 13.23 | 17.20 |
| Qwen3-Omni-30B-A3B | 1.52 | 7.39 | 21.51 |
| Qwen2.5-Omni-7B | 2.12 | 2.33 | 0.00 |
| Gemini-3.7-flash | 27.42 | 64.98 | 66.67 |
| Qwen3.8Omni | 18.18 | 38.52 | 40.86 |

Paired accuracy by interaction on the same shared first-500 subset. Values are percentages; both pre and post decisions must be correct.

LaTeX：`probe_interaction.tex`；CSV：`probe_interaction.csv`。

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
| Attribute Perception | 79 | 13.92 | 8.86 | 2.53 | 45.57 | 27.85 |
| Temporal Grounding | 129 | 3.10 | 6.98 | 0.00 | 33.33 | 23.26 |
| Action Perception | 172 | 6.40 | 6.40 | 4.65 | 39.53 | 18.02 |
| Object Perception | 69 | 5.80 | 11.59 | 1.45 | 53.62 | 33.33 |
| Emotion Recognition | 42 | 7.14 | 4.76 | 2.38 | 30.95 | 19.05 |
| Event Understanding | 146 | 4.79 | 4.11 | 1.37 | 26.03 | 17.12 |
| Text-Rich Understanding | 149 | 10.07 | 10.07 | 0.00 | 67.11 | 48.32 |
| Object Counting | 43 | 13.95 | 6.98 | 0.00 | 53.49 | 25.58 |
| Action Counting | 72 | 8.33 | 1.39 | 1.39 | 68.06 | 51.39 |
| Spatial Understanding | 81 | 3.70 | 4.94 | 4.94 | 48.15 | 28.40 |
| Prospective Reasoning | 60 | 1.67 | 3.33 | 0.00 | 16.67 | 6.67 |
| Causal Reasoning | 61 | 6.56 | 1.64 | 1.64 | 26.23 | 14.75 |

Paired decision accuracy by ability on the shared first-500 subset. Percentages; no response-content judge is applied.

LaTeX：`probe_abilities.tex`；CSV：`probe_abilities.csv`。

### Probe paired accuracy by cognitive level

| Level | Pairs | MiniCPM-o-4.5 | Qwen3-Omni-30B-A3B | Qwen2.5-Omni-7B | Gemini-3.7-flash | Qwen3.8Omni |
| --- | --- | --- | --- | --- | --- | --- |
| Perception | 491 | 6.72 | 7.54 | 2.44 | 40.12 | 23.22 |
| Comprehension | 410 | 8.29 | 6.10 | 0.73 | 51.22 | 35.37 |
| Reasoning | 202 | 3.96 | 3.47 | 2.48 | 32.18 | 17.82 |

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

`python benchmark/scripts/build_experiment_report.py --output-dir benchmark/docs/experiment_results_20261027`

可传`--remote-gemini-name`和`--remote-qwen-name`修改展示标签；不改变原始分数。
`metrics.json`保存统计、来源路径及读取快照哈希；`first500_index_mapping.csv`保存索引映射。
LaTeX在当前机器尚未编译（未发现pdflatex/tectonic）；已检查生成表格和输入引用。
