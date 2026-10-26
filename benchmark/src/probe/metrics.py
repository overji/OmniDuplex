"""Decision and paired accuracy, separate from online timing metrics."""
from collections import defaultdict
import re

from benchmark.data.files import atomic_write_text, write_json
from benchmark.probe.output import PROTOCOL, parse_reply


def record_output(record, row):
    if record.get('output_protocol') == PROTOCOL:
        return parse_reply(record['raw_response'], row['mode'],
                           complete=record.get('task_text_generation_complete'))
    if record.get('output_protocol') not in (None, 'yes_no_v1'):
        raise ValueError('Unknown probe output protocol')
    return {'decision': parse_decision(record['raw_response'])}


def parse_decision(text):
    if not isinstance(text, str):
        return None
    # No substring recovery from reasoning, markdown, or repeated YES/NO choices.
    match = re.fullmatch(r'\s*(?:(YES)(?:\s*:\s*\S[\s\S]*)?|(NO)[.!]?)\s*', text, re.I)
    return ('YES' if match.group(1) else 'NO') if match else None


def score_group(rows, predictions):
    complete = [row for row in rows if row['probe_id'] in predictions]
    confusion = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
    correct, invalid, pre_correct, post_correct, pre_total, post_total = 0, 0, 0, 0, 0, 0
    pairs = defaultdict(list)
    for row in rows:
        pairs[row['pair_id']].append(row)
    correctness = {}
    for row in complete:
        decision = record_output(predictions[row['probe_id']], row)['decision']
        right = decision == row['expected']
        correctness[row['probe_id']] = right
        correct += int(right)
        if row['side'] == 'pre':
            pre_total += 1
            pre_correct += int(right)
        else:
            post_total += 1
            post_correct += int(right)
        if decision is None:
            invalid += 1
        else:
            confusion[('tp' if row['expected'] == 'YES' else 'fp') if decision == 'YES'
                      else ('tn' if row['expected'] == 'NO' else 'fn')] += 1
    paired = [pair for pair in pairs.values() if len(pair) == 2
              and all(row['probe_id'] in predictions for row in pair)]
    paired_correct = sum(all(correctness[row['probe_id']] for row in pair) for pair in paired)
    tp, fp, fn = confusion['tp'], confusion['fp'], confusion['fn']
    def ratio(a, b):
        return a / b if b else None
    pre_acc, post_acc = ratio(pre_correct, pre_total), ratio(post_correct, post_total)
    result = {'eligible_probes': len(rows), 'completed_probes': len(complete),
        'probe_coverage': ratio(len(complete), len(rows)), 'accuracy': ratio(correct, len(complete)),
        'negative_accuracy': pre_acc, 'positive_accuracy': post_acc,
        'balanced_accuracy': (pre_acc + post_acc) / 2 if pre_acc is not None and post_acc is not None else None,
        'eligible_pairs': len(pairs), 'complete_pairs': len(paired),
        'pair_coverage': ratio(len(paired), len(pairs)), 'paired_accuracy': ratio(paired_correct, len(paired)),
        'invalid_outputs': invalid, 'invalid_rate': ratio(invalid, len(complete)),
        'parse_coverage': ratio(len(complete) - invalid, len(complete)),
        'valid_only_confusion': confusion, 'valid_only_precision': ratio(tp, tp + fp),
        'valid_only_recall': ratio(tp, tp + fn), 'valid_only_f1': ratio(2 * tp, 2 * tp + fp + fn)}
    generated = [(row, record_output(predictions[row['probe_id']], row)) for row in complete
                 if predictions[row['probe_id']].get('output_protocol') == PROTOCOL]
    if generated:
        agent_replies = [parsed for row, parsed in generated if row['mode'] == 'agent' and parsed['decision'] == 'YES']
        valid_agent = sum(bool(p['tool_calls']) for p in agent_replies)
        result.update(output_format_valid=sum(p['output_format_valid'] for _, p in generated),
            output_format_total=len(generated),
            output_format_valid_rate=ratio(sum(p['output_format_valid'] for _, p in generated), len(generated)),
            agent_reply_outputs=len(agent_replies), agent_valid_calls=valid_agent,
            agent_valid_call_rate=ratio(valid_agent, len(agent_replies)),
            truncated_outputs=sum(p['output_status'] == 'truncated' for _, p in generated),
            unknown_completion_outputs=sum(p['task_text_generation_complete'] is None for _, p in generated))
    return result


def summarize(rows, records, errors=()):
    protocols = {r.get('output_protocol', 'yes_no_v1') for r in records}
    if len(protocols) > 1:
        raise ValueError('Mixed probe output protocols; use separate result directories')
    known = {row['probe_id'] for row in rows}
    predictions = {}
    for record in records:
        identity = record['probe_id']
        if identity not in known or identity in predictions or record.get('status') != 'complete':
            raise ValueError('Unknown, duplicate or incomplete probe prediction')
        if not isinstance(record.get('raw_response'), str):
            raise ValueError('Probe prediction must preserve raw_response')
        predictions[identity] = record
    groups = defaultdict(list)
    for row in rows:
        for name in ('overall', 'mode:' + row['mode'], 'ability:' + row['ability']):
            groups[name].append(row)
    failures = {error['probe_id'] for error in errors if error.get('probe_id') in known} - predictions.keys()
    return {'evaluation': 'independent_paired_prefix_reply_decision_v1',
        'output_protocol': next(iter(protocols), None),
        'invalid_policy': 'incorrect_for_accuracy_and_paired_accuracy',
        'precision_recall_f1_scope': 'parseable_completed_probes_only',
        'failed_probes': len(failures),
        'error_attempts': sum(error.get('probe_id') in known for error in errors),
        'groups': {name: score_group(group, predictions) for name, group in groups.items()}}


def write_report(rows, records, output_dir, errors=()):
    metrics = summarize(rows, records, errors)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / 'probe_metrics.json', metrics)
    report = ['# Probe decision benchmark', '',
        'Accuracy includes invalid outputs as incorrect. Inference failures are missing coverage.',
        'Paired accuracy requires both sides correct and is computed over complete pairs.',
        'Precision/recall/F1 use parseable outputs only; always inspect parse coverage.', '',
        f"Failed probes awaiting retry: {metrics['failed_probes']}", '',
        '| Group | Completed / Eligible | Accuracy | Pre accuracy | Post accuracy | Complete pairs / Eligible | Paired accuracy | Parse coverage |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    def percent(value):
        return f'{value:.2%}' if value is not None else 'N/A'
    for name, value in metrics['groups'].items():
        report.append(f"| {name} | {value['completed_probes']}/{value['eligible_probes']} | "
            f"{percent(value['accuracy'])} | {percent(value['negative_accuracy'])} | "
            f"{percent(value['positive_accuracy'])} | {value['complete_pairs']}/{value['eligible_pairs']} | "
            f"{percent(value['paired_accuracy'])} | {percent(value['parse_coverage'])} |")
    if metrics['output_protocol'] == PROTOCOL:
        report += ['', '## Response / instruction format', '',
                   'Decision accuracy is independent of Agent payload validity. A spoken reply without an instruction is still a positive decision.',
                   'Unknown completion is reported separately; it is not a verified natural stop.', '',
                   '| Group | Valid format | Agent replies | Valid calls | Truncated | Unknown stop |',
                   '| --- | ---: | ---: | ---: | ---: | ---: |']
        for name, value in metrics['groups'].items():
            report.append(f"| {name} | {value.get('output_format_valid', 0)}/{value.get('output_format_total', 0)} | "
                f"{value.get('agent_reply_outputs', 0)} | {value.get('agent_valid_calls', 0)} | "
                f"{value.get('truncated_outputs', 0)} | {value.get('unknown_completion_outputs', 0)} |")
    atomic_write_text(output_dir / 'probe_report.md', '\n'.join(report) + '\n')
    return metrics
