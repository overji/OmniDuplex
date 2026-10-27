# MiniCPM-o-4.5 realtime：完整输入与模态消融

统计快照：2026-10-27T00:52:36.536971+00:00。三条件共同完成2107条QA，按相同sample_index重新计算时间指标。

| 输入 | QA | 窗口 | 输出事件 | 命中 | Recall | Precision | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 完整音视频 | 2107 | 4935 | 4178 | 1150 | 23.30% | 27.53% | 25.24% |
| 仅视觉（no audio） | 2107 | 4935 | 4383 | 961 | 19.47% | 21.93% | 20.63% |
| 仅音频（no vision） | 2107 | 4935 | 2336 | 335 | 6.79% | 14.34% | 9.21% |

## 交互类型

| 交互 | QA | 输入 | Recall | Precision | F1 |
| --- | ---: | --- | ---: | ---: | ---: |
| rtd | 706 | 完整音视频 | 21.77% | 35.42% | 26.96% |
| rtd | 706 | 仅视觉（no audio） | 18.30% | 24.56% | 20.97% |
| rtd | 706 | 仅音频（no vision） | 6.03% | 19.52% | 9.21% |
| pr | 678 | 完整音视频 | 33.55% | 29.53% | 31.41% |
| pr | 678 | 仅视觉（no audio） | 25.54% | 22.91% | 24.15% |
| pr | 678 | 仅音频（no vision） | 10.83% | 13.91% | 12.17% |
| agent | 723 | 完整音视频 | 12.95% | 10.01% | 11.29% |
| agent | 723 | 仅视觉（no audio） | 14.40% | 13.24% | 13.80% |
| agent | 723 | 仅音频（no vision） | 3.30% | 5.24% | 4.05% |

## 十二任务时间F1

| 任务 | QA | 完整音视频 | 仅视觉 | 仅音频 |
| --- | ---: | ---: | ---: | ---: |
| 属性感知 | 163 | 22.25% | 24.47% | 4.85% |
| 时间定位 | 198 | 27.05% | 15.59% | 16.59% |
| 动作感知 | 289 | 19.89% | 18.14% | 7.01% |
| 物体感知 | 138 | 32.74% | 33.10% | 1.80% |
| 情绪识别 | 120 | 18.75% | 12.10% | 11.26% |
| 事件理解 | 274 | 23.63% | 14.39% | 13.23% |
| 文本内容理解 | 298 | 34.68% | 34.44% | 9.65% |
| 物体计数 | 92 | 21.08% | 20.28% | 4.73% |
| 动作计数 | 116 | 31.40% | 13.69% | 15.68% |
| 空间理解 | 137 | 27.43% | 27.38% | 5.96% |
| 前瞻推理 | 110 | 14.08% | 7.18% | 7.74% |
| 因果推理 | 172 | 17.93% | 10.00% | 9.50% |

## 按音频依赖标签分组

| 音频依赖 | QA | 完整音视频F1 | 仅视觉F1 | 仅音频F1 |
| --- | ---: | ---: | ---: | ---: |
| required | 600 | 20.90% | 8.94% | 14.55% |
| helpful | 905 | 25.99% | 21.43% | 9.65% |
| none | 602 | 28.05% | 29.69% | 3.68% |

## 比较边界

时间分数仅看回复起始点与参考窗口的一对一匹配，不检查回复和委托instruction的语义正确性。没有重新调用内容judge。

完整音视频基线复用历史RTD/PR结果，Agent使用task-text重评测；两组消融均为task-text运行。三条件模型权重和数据来源相同，记录的seed=42、sampling、max_speak_tokens=20、sliding_window_mode=off相同。历史运行/实现指纹和DP配置并非完全相同，也没有多个随机种子复验，因此不是严格只改变模态的受控因果实验。

逐QA的system_prompt哈希与完整音视频不一致数量（按mode）：{"video_only": {}, "audio_only": {}}。本次逐QA哈希均一致，说明三条件任务prompt文本相同；历史实现及DP差异仍需作为限制，不以单个消融分数证明某种能力。

仅视觉通过prefill省略audio_waveform，仅音频通过prefill省略frame_list，不补入丢弃模态的字幕或caption。

## 来源

- 完整音视频: `/root/jimodels/OmniDuplex/benchmark/results/full_2110_dp6_20261021/realtime/minicpmo`, `/root/jimodels/OmniDuplex/benchmark/results/task_text_agent100_20261024_v2/continuation/realtime/minicpmo_task_text_agent_full`
- 仅视觉（no audio）: `/root/jimodels/OmniDuplex/benchmark/results/task_text_agent100_20261024_v2/task_specific_probe_512_v3/suite/realtime/minicpmo_no_audio_reply_task_text`
- 仅音频（no vision）: `/root/jimodels/OmniDuplex/benchmark/results/task_text_agent100_20261024_v2/task_specific_probe_512_v3/suite/realtime/minicpmo_no_vision_reply_task_text`
