# SFT 后的多次 rollout 诊断

`train/scripts/sft_then_split_bench.py` 的可选 `rollout_configs` 配置接入两个独立诊断阶段。
它等待训练完整结束、验证最终 checkpoint 并导出，然后同时运行 AOI-Bench（0–3 卡）
和 OmniPro（4–7 卡）。两套诊断均成功后，才开始后续全量评测及原有内容 Judge。
失败或不完整的轨迹使队列停止，不以沉默替代失败，也不自动改变采样参数。

当前每套均匀随机抽取 50 条 QA（不放回，selection_seed=20261110），固定抽样索引，
不重排原始 QA。每条按 seed 42–49 进行 8 次完整视频 realtime 推理；实际样本种子为
`seed + sample_index`，与正式 MiniCPM runner 一致。每个 seed 内四实例数据并行。
除种子不同，沿用正式评测的提示词、媒体输入、解码和窗口。保留自然生成的重复回复。

各结果目录的 `diagnostic_rollouts/` 包含：

- `manifest.json`：原始索引、QA 摘要、输入摘要、种子、正式结果采用的固定 rollout。
- `sample_indices.json`、`commands.json`：可复现的抽样和推理命令。
- `rollout_00/` 至 `rollout_07/`：预测、时间分数，以及 worker 内逐秒 chunks 和完整提示词。
- `rollouts.jsonl`：800 条中的本数据集 400 条预测和各自时间分数，可直接用于离线分析。
- `complete.json`、`status.json`：完成数与当前阶段。

正式评测固定复用 `rollout_00`，与质量分数无关。复用前验证完整轨迹、依赖版本、
推理配置、数据、协议和提示词；只允许 GPU 位置和输出路径不同。原始 chunks、提示词
及预测写入正式分片，原 runner 跳过已完成的 50 条，仅运行剩余数据。
`rollout_reuse.json` 保存来源。其余七次不计入正式指标，不做 best-of-eight。

配置示意（`full_command` 使用现有全量 parallel 命令，包含实际路径）：

```json
{
  "output": "/path/to/results/diagnostic_rollouts",
  "selection_seed": 20261110,
  "sample_count": 50,
  "rollout_seeds": [42, 43, 44, 45, 46, 47, 48, 49],
  "full_command": ["/path/to/python", "-m", "benchmark", "parallel", "minicpmo", "..."]
}
```

只校验并冻结抽样、分片计划，不加载权重：

```bash
benchmark/.venv/bin/python benchmark/scripts/rollout_before_full.py \
  --config train/outputs/full18315_final_split_bench_20261110/rollout_aoi.json \
  --prepare-only
```

移除 `--prepare-only` 执行诊断；中断后以相同配置重启，已完成 QA 不重生成。
已有 partial/error 行不会被当成成功，必须检查后使用独立新目录重跑失败阶段。
不能修改已有 manifest 或 parallel 配置来混用结果。当前队列已经自动接入，无需手动启动。

该阶段不更新权重，不保存 GRPO 所需的可反向图或完整策略 log-prob；它用于在训练前
观察生成方差、FP/FN、重复开口和后续回复缺失。AOI-Bench 和 OmniPro 的这些评测 QA
不会自动加入 GRPO 训练集。诊断阶段保存时间指标；原有 LLM Judge 仍在正式评测后执行。
