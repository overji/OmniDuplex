# GRPO pilot and independent plain-CE SFT

**Superseded:** the GRPO profile failed again on its CPU activation budget. The user cancelled GRPO and requested immediate eight-GPU ordinary-CE SFT; see [the new schedule](plain_ce_dp8_20261101.md). The old follow-up scheduler is stopped.

The current pilot runs up to **20 optimizer updates**, followed by a new ordinary-CE SFT from the original MiniCPM-o-4.5 base. It does not train SFT on the GRPO weights. GPU 0–3 train; GPU 7 hosts the GRPO judge; benchmark GPU 4–6 remain separate.

## Fixed timing check

`train/outputs/sft_bench_grpo_fullvideo_20261031/evaluation_selection.json` fixes 24 QA from the prepared validation split: 8 PR, 8 RTD, 8 Agent, with unique videos and short/medium/long coverage. Video splits are unchanged. References/windows are used only for scoring after autonomous full-video generation; prior reference responses are never fed into rollout history.

Evaluate the starting SFT policy at step 0 and the GRPO policy at steps 10 and 20, with the same per-QA seeds and sampling policy. Reports are written under the external GRPO output:

```
/remote-home/yushengliao/speech/jimodels/training_outputs/OmniDuplex/grpo_fullvideo_confusion_dp4_20261031/timing_evaluation/report.md
```

The report includes paired QA count, FP count, FP/QA, precision, recall, F1 and FP change. A failed prediction is excluded from both sides of a paired comparison, not treated as zero FP. Native empty/control-only response attempts are recorded separately in JSON; the report's parsed-event FP matches the benchmark timing convention, while the GRPO reward also penalizes those unparsed attempts. Content quality is not judged by this timing check. A 24-QA pilot is exploratory; lower FP with collapsed recall is not success.

Training-time `reward_fp` averages over different sampled QA at each step and is not a controlled before/after evaluation.

## Plain CE

Reusable config: `train/configs/minicpmo_full_sft_plain_ce_dp.json`.

`decision_loss_mode="plain_ce"` averages CE across all supervised decision, response, and instruction tokens **within each QA**, then retains the existing equal-QA batch reduction. Unit component weights are required. It retains causal shifting, masked media/input positions, all response targets, full-video coverage and post-question silence supervision. Diagnostics remain available. The older `token_mean` mode separately averages each component and sums them; it is preserved for historical compatibility and is not the ordinary-CE mode.

The queued SFT matches the latest SFT's 3996/1004 prepared train/validation split, 65536-token cap, effective batch size 16, learning rate 1e-6, two epochs, ZeRO-2 CPU optimizer and save interval 50. All checkpoints remain on external storage. TensorBoard is enabled; W&B is disabled.

Operational settings and state:

- GRPO controller: `train/outputs/sft_bench_grpo_fullvideo_20261031/chain.json`, `status.json`.
- Follow-up SFT: `train/outputs/plain_ce_after_grpo_20261101/after_grpo.json`, `sft_after_grpo_status.json`.
- Plain-CE output: `/remote-home/yushengliao/speech/jimodels/training_outputs/OmniDuplex/full_sft_plain_ce_dp4_bs16_fullvideo_65k_20261101`.
- Place `STOP_SFT` in the follow-up directory to cancel its wait.

The follow-up scheduler requires successful GRPO status, finite final reports for all four ranks with exactly 20 updates, complete checkpoint shards and free training GPUs. A GRPO failure blocks automatic SFT.

## Replay memory fix

The original full-video GRPO profile exceeded its 128 GiB/rank CPU activation budget on its first backward pass. `grpo_replay_checkpoint_segments=128` adds activation recomputation across groups of native decoder feeds. KV boundaries are represented as immutable tuples, each invocation creates its own DynamicCache wrapper, and gradients remain connected across boundaries. No video cropping, KV detachment, reward change or probability-tolerance relaxation is applied. CPU value/gradient tests use a real tiny causal decoder; the four-GPU profile remains the required acceptance gate.
