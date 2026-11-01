# LLM-judged TP/FP/FN/TN trajectory reward

Opt in with `train/configs/minicpmo_full_grpo_confusion_judge.json`.
The existing `minicpmo_full_grpo_judge.json` retains its historical reward.
No judge service or training process is started by loading either configuration.

## Definitions and formula

Reference windows and generated text onsets use the benchmark's inclusive,
one-to-one matcher. Every reference window can earn at most one TP; additional
responses, including extra onsets within that window, are FP. Unmatched windows
are FN. Native fresh response attempts that produce no parsed event contribute
additional FP (`max(0, native_starts - parsed_events)`), so empty attempts cannot
avoid the count penalty. Continuations are not new onsets.

Only matched TP content is judged, using the existing reference-based prompt and
0, 0.5, 1, 1.5, 2 score contract. RTD/PR quality is response_score/2. Agent quality
is (response_score + call_score)/4. Empty response content is forced to zero;
absent, incomplete, or schema-invalid delegation always receives call quality
zero regardless of the supplied judge score. A valid delegation has exactly one
`task_start` call with a valid instruction. Failed or invalid judge answers reject
the whole QA candidate group through the existing retry/exclusion path, rather
than being converted to a zero-quality training example.

```
N = number of reference windows
TP contribution = sum(2 * quality - 1) / N
FN contribution = -FN / N
FP contribution = -0.5 * FP / N
TN contribution = 0.1 * (2 * TN / D - 1)  # zero when D == 0
invalid-call contribution = -0.1 * invalid_calls / N
R = sum of the five contributions
```

All five coefficients are explicit `reward_weights` keys: `tp`, `fn`, `fp`,
`tn`, `invalid_call_penalty`. Missing scored windows reject the QA. Rewards are
not clipped: every additional FP retains its incremental penalty.

D is the number of post-question integer input seconds outside all reference
windows, with the same `ceil(question_time)+1` start convention as training.
TN requires a genuinely idle native listen action: the turn was closed before
generation, native `is_listen` is true, and no new response started. Continuation
seconds stay in the fixed D denominator but do not earn TN. New `idle_wait`
telemetry is recorded in rollouts; older trajectories without this field are
not silently treated as valid new-mode input. The pre-question interval earns
no credit. TP/FN are event counts; TN is clock-normalized and never added as a
raw count to event counts. Thus long silent videos do not drown out FN penalties.

## Running

Use the existing judged-GRPO setup in the training README: serve the configured
local Qwen judge, choose a compatible exported SFT model in `model_path`, then
prepare a fresh dataset and run the normal `grpo` command with the new config.
The sample model_path is inherited from the earlier judged-GRPO example; replace
it with the actual ordinary-text-protocol SFT export intended for this experiment.
Use a fresh external checkpoint output directory. Configuration and reward-code
fingerprints reject incompatible optimizer resumes.

The new mode explicitly requires `reward_backend=llm`, `training_method=full_grpo`
and `grpo_horizon=full_video`. The runner reads all complete video seconds,
including the period after the last reference window. The old 300-second prefix
cap does not filter late reference responses in this mode. Total sequence budget
is 65536; overflow or incomplete reference coverage excludes the trajectory/QA
group rather than truncating video and rewarding partial coverage. Actual media
lengths and GPU memory have not been profiled by the offline reward tests.

JSON rollout reward records contain TP/FP/FN/TN, reference_windows,
negative_seconds, unparsed_response_starts, invalid_calls, and all five signed
contributions. Update logs and W&B contain QA-candidate averaged `reward_*`
counts and contributions; `reward_tp` is a count, not semantic quality. Existing
`reward_timing/content/call/silence` diagnostics remain available. Compare timing
precision/recall/F1, response/call correctness, all-silent and zero-variance-group
rates, and worst-QA FP concentration. A reward increase alone is not acceptance.

## Limits

The reference-only judge cannot prove that an unannotated visual event is a true
trigger. No window widening or judge rescue of FP is performed. Extra-response
penalties target fresh onsets; a single indefinitely open repeating utterance
can have only one onset. It loses TN credit during continuation, but this reward
does not yet include a dedicated token-length or intra-response repetition
penalty. Those require separately validated detectors. Current GRPO uses one
trajectory-level advantage, not event-local credit assignment.

## Four-GPU candidate-parallel execution

`train/configs/minicpmo_full_grpo_confusion_judge_dp4.json` explicitly opts into
`grpo_candidate_parallel=true`, ZeRO-2, `grpo_judge_pipeline_depth=2`, and
`grpo_precheck_replay=false`. Existing serial/ZeRO-3 configs keep their defaults.
The model path is still an example and must be replaced with the intended SFT
export. The active post-SFT chain already selects its own external final export.

The four ranks process the same QA, with rank r sampling candidate r under the
same pre-update policy. World size must equal group_size (four in this config).
There is no per-token broadcast. Only events and per-second new_response and
idle_wait telemetry are gathered to rank zero for judging; token histories and
media stay local. Every rank receives the four rewards, computes the same group
advantages, and replays only its own candidate. ZeRO averages the four gradients.
Per-QA normalization and partial GAS batches preserve the serial group's mean
objective; GAS=4 still means four QA and sixteen candidate trajectories per full
optimizer update, not sixteen QA. Per-rank trajectory logs now contain that
rank's candidate, identified by its global candidate index. Global loss, ratio,
replay error and candidate counts are reduced for reporting.

Independent candidate lengths cannot use ZeRO-3's forward parameter gathers;
the configuration and runtime reject this combination. ZeRO-2 keeps full model
parameters on each GPU and increases residency, so a real model profile is
required before accepting the full run. The existing ZeRO-2 CPU accumulation
fix is also applied to this execution mode.

One CPU judge group runs asynchronously while GPUs sample the next QA. The
queue depth is bounded, and scored plus pending groups never exceed GAS.
Everything is drained before an optimizer update, checkpoint, or epoch change:
no rollout generated before an update is reused after that update. Judge errors
exclude whole groups; zero-variance groups retain the historical skip behavior.
There is no frozen encoder cache or change to video coverage, rewards, candidate
seeds or action distribution.

The extra no-grad replay is optional. Normal optimized training skips it, but
EVERY gradient replay still checks finite loss and sampling/replay probability
error before backward; a mismatch aborts training. The automatic profile enables
the extra check and includes Agent/PR/RTD videos of approximately 50, 144, 270,
and 299 seconds. An OOM or failed check stops the chain; no silent fallback changes
the training configuration.

Timing diagnostics in JSON and W&B include rollout_wall_seconds,
precheck_wall_seconds, gather_wall_seconds and judge_wait_wall_seconds. Judge
wait measures the residual foreground wait after overlap, not total service
work; phase maxima across ranks need not add to the exact step duration. Compare
real step_wall_seconds after warm-up for throughput/ETA. Four-rank CPU gradient
and pipeline tests do not establish real GPU memory requirements or speedup.
