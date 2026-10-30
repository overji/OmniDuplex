# SFT loss and diagnostics

## Balanced fresh-onset / silence objective

The original full-SFT configurations explicitly use `decision_loss_mode:
"balanced_gate"` for RTD, PR and Agent. Configuration files without this field
retain the historical `token_mean` decision objective. GRPO is unchanged.

Each QA contributes:

```
L_QA = decision_weight * L_decision
     + content_weight * meanCE(response text)
     + call_weight * meanCE(Agent instruction text)
L_decision = 0.8 * L_gate + 0.2 * meanCE(structure)
L_gate = 0.5 * meanCE(idle listen) + 0.5 * meanCE(fresh onset speak)
```

The five configurable fields are `decision_loss_mode`, `gate_weight` (0.8),
`structure_weight` (0.2), `silent_weight` (0.5), and `onset_weight` (0.5).
Weights are normalized within their level over nonempty, positive-weight
classes. For example, a QA without listen positions uses onset alone for its
gate. A QA without instruction has no instruction term. With all decision
classes present and the three existing outer weights equal to 1:

```
L_QA = 0.4 * L_silent + 0.4 * L_onset + 0.2 * L_structure
     + L_response + L_instruction  # last term only when present
```

`decision_roles` is target-aligned replay metadata: 0 = no decision target,
1 = supervised idle listen, 2 = first speak of each reference response,
3 = structure. Continuation-chunk speak, turn_eos and all ordinary tokens spelling task_start are structure;
chunk_eos remains unsupervised. Every
reference response has one fresh onset, even in a multi-response QA. The loss
applies the usual causal one-token shift to labels and roles together.

With `supervise_silence=true`, every idle second after the question in the
replayed trajectory is supervised as listen, including seconds inside reference
windows before or after the scheduled response. Scheduled response chunks always
take precedence. Pre-question positions remain masked; false disables idle loss.
The `all_post_question_idle_v1` checkpoint identity prevents resuming an optimizer
from the previous window-masked objective. No media input, response, window,
optimizer setting or sampling behavior changes. Valid QA objectives retain the
existing batch averaging. The full vocabulary projection remains chunked.

## Metrics

Local JSON and W&B train/validation metrics retain silence, speak, control,
response, agent_instruction and decision unweighted token-average CE and counts.
Additional components are onset, continuation_speak and structure. `speak`
includes both onset and continuation; `structure` includes continuation speak
and other controls. Components overlap and must not be summed.

Missing classes have null CE and zero counts; padding and excluded QA contribute
nothing. `local_*` fields describe that rank alone. Validation pools CE sums and
token counts over all valid QA and ranks, while objective contributions retain
per-QA normalization.

The actual objective is reconstructed by:

```
loss_decision_contribution + loss_response_contribution
+ loss_agent_instruction_contribution
```

For balanced mode, `loss_gate_contribution + loss_structure_contribution` also
reconstructs `loss_decision_contribution`. These include the outer decision
weight. The original `loss_decision` remains raw token-mean CE for diagnosis;
it is not the new weighted decision objective.

## Running and comparing

Existing running processes are not restarted. Use a fresh experiment for the
balanced objective; strict optimizer resume rejects changed configurations.
Historical configs without the mode field keep their exact old objective.
The new weights are saved in run/checkpoint configs and W&B hyperparameters.
A change in total loss scale does not establish improvement: compare first and
later window recall, window-external FP, timing F1, response correctness and
Agent valid-call rate on the same held-out inputs.

## Event-balanced objective (opt-in)

Use `train/configs/minicpmo_full_sft_event_balanced_dp.json` with the existing
training CLI's `--config` argument, a fresh output directory, and no optimizer
resume from an earlier objective. Existing configurations and queued jobs are
not switched automatically. All other settings are copied from the standard
DP configuration; inspect paths and resources before launching.

```
L_onset = mean(nonempty first_onset, later_onset)
L_silent = meanCE(all supervised idle tokens)
L_gate = 0.5 * L_silent + 0.5 * L_onset
L_structure = mean(nonempty continue, end, marker)
L_QA = 0.8 * L_gate + 0.2 * L_structure + L_response + L_instruction
```

Gate leaves are token-average CE. Every structure leaf, response text and Agent
instruction first averages tokens within a reference response, then averages
responses containing that category. Long utterances therefore do not dominate
short updates. All branch averages renormalize over present, positive-weight
branches, using the same existing gate/structure/silent/onset configuration
weights; absent content/instruction contributes zero.

`silence_before` precedes the first scheduled onset. `silence_after` covers idle
seconds after the first response completes, including gaps and the existing
tail. These two roles are retained only for diagnostics and historical telemetry
compatibility: their tokens share one pooled silence mean, not equal branch
weights. Relabeling an idle token as before/after cannot change the loss or
gradients. Active response chunks are never silence. First/later refers to scheduled
chronological order, not source indices. The separator is ordinary text, with
all its tokens assigned to `marker` and the same event even across chunks.
`event_roles` uses 0 for non-decision, then 1..7 for before, after, first onset,
later onset, continue, end, marker. `event_ids` is zero-based chronological
response ID; idle and masked positions are -1. Both are target-aligned, shifted
with labels for causal CE. No horizon, media, window, or target schedule changes.

The seven fine-role names above are logged as `loss_<name>`, `<name>_tokens`,
`<name>_qa`, and `loss_<name>_contribution` in JSON and W&B. Raw CE remains pooled
per-token diagnostics, whereas contributions reflect hierarchical per-QA
weighting. The seven fine contributions sum to decision contribution. Historical
telemetry slots are unchanged; new slots are appended. An `event_balanced_v2`
checkpoint fingerprint and strict config checks prevent incompatible resumes,
including optimizer resumes from the v1 objective that balanced before/after
silence separately.

This is teacher-forced supervision, not a penalty computed from generated reply
counts. Raising incorrect speak probability on a listen target raises CE, but
averaging prevents long videos or many legitimate replies from receiving an
unbounded count penalty. Free-running repetition may still require generated-
history correction or later RL. Compare omissions, first/later recall, per-QA
FP concentration, precision/F1 and valid Agent calls, not only total loss.

## Full-video horizon

Opt in using `sft_horizon=full_video` (full SFT and silence supervision required;
fixed `post_response_silence_seconds` must be 0). The dedicated full-video DP
config uses 65536 total tokens. The legacy prefix cap does not reject late
responses in this mode. Before media duration is known, trajectory horizon is
None, requesting Clip to read every complete second. Once known, horizon is
floor(duration); reference chunks extending beyond it still reject the QA.
Token overflow rejects the entire QA, including overflow after the last response:
full-video supervision cannot silently fall back to a short tail. Old modes
retain their previous horizon/optional-tail behavior. A dedicated checkpoint
fingerprint prevents optimizer resumes across these coverage changes. The
subsecond final interval remains outside the existing integer-second schedule.
