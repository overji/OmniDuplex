# 把已有 benchmark 结果交给别人做 LLM Judge

发送方提供一个输入 JSON；评分方拉取代码、配置 DeepSeek key 后运行，最后只回传一个结果 JSON。整个评分过程不需要 GPU、视频、模型权重、原始 QA 或预测目录。

## 本次输入包

文件：`benchmark/exports/llm_judge_all_except_livestar_20261028.json`。

| 模式 | 纳入结果 |
| --- | --- |
| Realtime（5组） | MiniCPM-o-4.5、Gander、JoyAI，以及 MiniCPM no-audio/no-vision |
| Probe（7组） | MiniCPM-o-4.5、Qwen3-Omni、Qwen2.5-Omni、Qwen2.5 no-audio/no-vision、Gemini-3.7-flash、Qwen3.8Omni |

不包含 LiveStar、Ming、Qwen3-VL、Baichuan、HumanOmniV2、OmniVinci、取消的历史实验或训练中模型。全部重新评分，不混用历史200条评分。

输入文件 SHA256：`6fc98c8360139ee810021ede4ad15bda6b0b3bcbdbd85f7d8ef291911439740c`（约127 MiB）。

当前包有 **41797条记录**：20128条按规则直接记分，21669条需要API评分，完全相同的评分输入去重后为 **20910次独立请求**。请求失败重试会增加实际调用次数，`api_usage`记录实际尝试与用量。每条模型结果仍有独立任务ID，去重不会丢掉模型之间的映射。

输入内嵌评分标准、当前参考及证据文本、候选回复/调用、模型/模式/消融标识和来源哈希，不包含密钥和媒体文件。历史运行使用2110条QA的索引，公开数据为2107条；远端500+1607的预测已经校验并统一到历史索引。覆盖率按各模型实际成功输出记录，不把缺失推理当作0分。JoyAI只输入视觉，远端结果的完整推理设置没有随预测提供。

## 评分方：更新代码和安装最小环境

以下为Linux/macOS Bash命令，在仓库根目录执行：

```bash
# 已有仓库
git pull --ff-only

# 没有仓库时先执行：
# git clone https://github.com/overji/OmniDuplex.git
# cd OmniDuplex

python3 -m venv benchmark/.venv-judge
benchmark/.venv-judge/bin/pip install -r benchmark/requirements-judge.txt
```

使用Python 3.10或以上版本即可。不需要安装完整 `benchmark/requirements.txt`、PyTorch、CUDA、FFmpeg或下载数据集。将发送方的输入JSON放到 `benchmark/exports/`，不需要运行 `judge-export`。

```bash
INPUT="$PWD/benchmark/exports/llm_judge_all_except_livestar_20261028.json"
OUTPUT="$PWD/benchmark/results/portable_judge/judge_results_all_except_livestar_20261028.json"

# 先检查输入；不需要key，不调用API、不写评分文件。
benchmark/.venv-judge/bin/python -m benchmark judge-bundle \
  --input "$INPUT" --output "$OUTPUT" --dry-run
```

## 配置 DeepSeek Flash 并评分

在本机 `.env` 中填写：

```dotenv
OPENAI_API_KEY=YOUR_DEEPSEEK_API_KEY
```

需要DeepSeek的key；`OPENAI_API_KEY`只是OpenAI兼容接口的变量名。不要将真实key提交到Git或发回给数据发送方。

```bash
env -u OPENAI_API_KEY benchmark/.venv-judge/bin/python -m benchmark judge-bundle \
  --input "$INPUT" --output "$OUTPUT" \
  --provider openai --model deepseek-flash \
  --base-url https://api.deepseek.com --env-file "$PWD/.env" \
  --thinking disabled --max-tokens 1024 \
  --concurrency 4 --timeout 120 --retries 2
```

`env -u OPENAI_API_KEY`只移除本次子进程中可能遗留的同名key，使其读取 `.env`。如果已经正确 `export OPENAI_API_KEY=...`，可去掉它和 `--env-file`。key不进入输入、输出、缓存或评分配置。

模型名与地址见[DeepSeek官方Chat Completions接口](https://api-docs.deepseek.com/api/create-chat-completion/)。本命令显式关闭thinking、设置温度0和1024输出token预算；不修改旧 `judge` / `probe-judge` 命令的默认设置。模型别名实际后端可能更新，输出保存请求配置、时间以及服务返回的模型标识/指纹（若提供）。

## 中断、失败与恢复

- 正常重复执行同一命令，会复用已完成评分；不会重新请求已完成项。并发数可以调整，输入、模型、endpoint和其他评分配置必须相同，否则使用新输出文件。
- 每条请求初次加最多2次重试。失败、截断、非法JSON不会变成0分；结果保留 `failed` 和空分数，程序退出码为2。
- 失败项默认不会在下次运行自动反复调用。需要补评分时，在原命令后添加 `--retry-failed`；完成项仍复用。
- 运行时使用 `OUTPUT.progress.jsonl` 增量保存。Ctrl+C会停止排队请求，等待已发出的请求返回并保存当前结果；异常中止后再次执行会恢复完整日志记录。
- 最后一个未写完的日志行可以恢复；内部损坏或身份变化会报错，不拼接到另一个数据包。只保留最终结果JSON也能在之后恢复评分。

未完成时分组正式均值为null，`scored_only_*`仅描述已完成子集，不能冒充全量分数。

## 最后回传什么

**只需回传 `judge_results_all_except_livestar_20261028.json`**，即上面 `$OUTPUT` 指向的文件。它包含所有任务ID、分数和简短理由、来源包身份、模型/模式/消融信息、各分组汇总、失败状态和API用量。

不必回传 `.env`、输入JSON、视频、模型权重、`.lock`或 `.progress.jsonl`。即使还有失败项，也可回传结果JSON；请保留原始失败/空分数，不手动改成0。发送方用 `bundle_id` 和 `task_id` 对应原始数据。

## 评分口径

- Realtime：仅时间匹配回复调用Judge；未匹配窗口和空输出按现有规则计0。输出同时报告包含这些0分的均值与时间命中条件下的回复均值。
- Probe：只评正侧TP，其他决策只保留在覆盖率/分类统计中；TP条件内容均分不代表整体决策性能。
- 五档分数为0、0.5、1、1.5、2。RTD/PR任务分等于回复分，Agent任务分为回复分与调用分的平均值，缺失/非法调用为0。
- 使用当前参考、证据和上下文，不发送后续工具结果，不执行工具，不补写模型输出。文字参考评分不能独立核实视频里未提供的事实。
- 模型之间的成功覆盖率和TP集合可能不同。对比时应使用任务ID/QA身份取交集，不能只按不同覆盖集合的均值排序。

## 发送方：重新导出

只有持有本机原始实验结果的发送方需要这一步：

```bash
python -m benchmark judge-export \
  --sources benchmark/configs/judge_export_20261028.json \
  --output benchmark/exports/llm_judge_all_except_livestar_20261028.json
```

来源清单明确列出12组和Realtime的来源优先级，不自动扫描历史实验。数据和Probe清单哈希不符时拒绝导出。相同快照重复导出保留现有文件；来源变化需要新文件名。大型输入/输出JSON通过文件传输单独交付，Git仅提供代码、来源清单与文档。
