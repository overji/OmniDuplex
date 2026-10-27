"""Reproduce the AOI training-data Markdown, JSON, CSV and chart report (CPU only)."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmark.data.timestamps import seconds
from benchmark.evaluation.timing import reference_windows
from train.src.domain.records import first_record
from train.src.infrastructure.config import load_config


def distribution(values):
    values = sorted(values)
    if not values:
        return {'count': 0}
    def percentile(p):
        position = (len(values) - 1) * p
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        return values[lower] + (values[upper] - values[lower]) * (position - lower)
    return {'count': len(values), 'mean': statistics.mean(values), 'median': statistics.median(values),
            'min': values[0], 'p25': percentile(.25), 'p75': percentile(.75),
            'p90': percentile(.9), 'p95': percentile(.95), 'max': values[-1]}


def modality_label(parts):
    order = ("visual", "speech", "sound")
    return "+".join([x for x in order if x in parts] + sorted(set(parts) - set(order)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'train/configs/minicpmo_full_sft.json')
    parser.add_argument('--dataset-dir', type=Path)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'train/docs')
    parser.add_argument('--plots', action='store_true', help='Requires existing matplotlib; no model loading')
    args = parser.parse_args()
    config = load_config(args.config)
    root = args.dataset_dir or Path(config['data']).parent
    source = root / 'all_qa.json'
    metadata_path = root / 'video_metadata.json'
    samples = json.loads(source.read_text())
    metadata = json.loads(metadata_path.read_text())
    records = metadata['videos']
    indexed = {row['file']: row for row in records}
    if len(indexed) != len(records):
        raise ValueError('Duplicate metadata file references')
    videos = Counter(s['video'] for s in samples)
    missing_metadata = sorted(set(videos) - indexed.keys())
    if missing_metadata:
        raise ValueError(f'Missing metadata: {missing_metadata[:5]}')
    missing_files = sorted(v for v in videos if not (root / v).is_file())
    extra_metadata = sorted(indexed.keys() - videos.keys())
    mode = Counter(s['mode'] for s in samples)
    ability = Counter(s['ability'] for s in samples)
    tasks = Counter(s['task'] for s in samples)
    matrix = {a: {m: sum(s['ability'] == a and s['mode'] == m for s in samples)
                  for m in ['rtd', 'pr', 'agent']} for a in sorted(ability)}
    refs = [(i, j, r) for i, s in enumerate(samples) for j, r in reference_windows(s)]
    all_responses = [r for s in samples for r in s['responses']]
    modalities = Counter(modality_label(set(r.get('trigger_type', 'unlabeled').split('+'))) for _, _, r in refs)
    qa_modalities = Counter()
    for s in samples:
        parts = set()
        for _, r in reference_windows(s):
            parts.update(r.get('trigger_type', 'unlabeled').split('+'))
        qa_modalities[modality_label(parts) or 'unlabeled'] += 1
    by_qa = defaultdict(list)
    for i, _, r in refs:
        by_qa[i].append(r)
    triggers = [seconds(r['trigger_time']) for _, _, r in refs]
    first_triggers = [min(seconds(r['trigger_time']) for r in rows) for rows in by_qa.values()]
    last_triggers = [max(seconds(r['trigger_time']) for r in rows) for rows in by_qa.values()]
    first_delays = [min(seconds(r['trigger_time']) for r in rows) - seconds(samples[i]['question_time'])
                    for i, rows in by_qa.items()]
    response_delays = [seconds(r['trigger_time']) - seconds(samples[i]['question_time']) for i, _, r in refs]
    window_widths = [seconds(r['window_end']) - seconds(r['window_start']) for _, _, r in refs]
    durations = [float(indexed[v]['duration_seconds']) for v in videos]
    questions = [seconds(s['question_time']) for s in samples]
    abilities_per_video = defaultdict(set)
    for s in samples:
        abilities_per_video[s['video']].add(s['ability'])
    negatives = [i for i, s in enumerate(samples) if s['mode'] == 'agent' and
                 s.get('sample_type', s.get('generation_context', {}).get('sample_type')) in
                 {'negative', 'missing_tool', 'no_context'}]
    bad_windows = [(i, j) for i, j, r in refs if not
                   0 <= seconds(r['window_start']) <= seconds(r['trigger_time']) <= seconds(r['window_end'])
                   <= float(indexed[samples[i]['video']]['duration_seconds']) + .001]
    window_details = []
    for i, j in bad_windows:
        sample, response = samples[i], samples[i]['responses'][j]
        start, trigger, end = [seconds(response[k]) for k in ['window_start', 'trigger_time', 'window_end']]
        duration = float(indexed[sample['video']]['duration_seconds'])
        window_details.append({'sample_index': i, 'response_index': j, 'video': sample['video'],
            'duration_seconds': duration, 'window_start_seconds': start, 'trigger_time_seconds': trigger,
            'window_end_seconds': end, 'end_excess_seconds': end - duration,
            'trigger_inside_video': 0 <= trigger <= duration,
            'window_order_valid': 0 <= start <= trigger <= end})
    pre_question = [(i, j) for i, j, r in refs if seconds(r['trigger_time']) < seconds(samples[i]['question_time'])]
    mismatched_counts = [v for v, n in videos.items() if indexed[v].get('qa_count') != n]
    conflicting_duration = [i for i, s in enumerate(samples) if
        abs(float(s['generation_context']['duration_seconds']) - float(indexed[s['video']]['duration_seconds'])) > .001]
    candidate_counts = {}
    candidate_modes = {}
    candidate_videos = {}
    excluded_counts = {}
    for limit in [32, 64, 128, 256, 512]:
        test_config = {**config, 'videos_dir': str(root), 'max_prefix_seconds': limit}
        eligible, excluded = [], Counter()
        for i, s in enumerate(samples):
            row, reason = first_record(s, i, test_config)
            if reason:
                excluded[reason] += 1
            else:
                eligible.append(row)
        candidate_counts[str(limit)] = len(eligible)
        candidate_modes[str(limit)] = dict(Counter(r['sample']['mode'] for r in eligible))
        candidate_videos[str(limit)] = len({r['sample']['video'] for r in eligible})
        excluded_counts[str(limit)] = dict(excluded)
    required_tools = Counter(t for s in samples if s['mode'] == 'agent' for t in s.get('required_tools', []))
    checks = {'metadata_video_count_matches': metadata.get('video_count') == len(records),
              'metadata_qa_count_matches': metadata.get('qa_record_count') == len(samples),
              'missing_video_files': missing_files, 'extra_metadata': extra_metadata,
              'metadata_qa_count_mismatches': mismatched_counts, 'duration_mismatches': conflicting_duration,
              'invalid_scored_windows': bad_windows, 'triggers_before_question': pre_question,
              'agent_negative_indices': negatives,
              'out_of_video_triggers': [(i,j) for i,j,r in refs if seconds(r['trigger_time']) > float(indexed[samples[i]['video']]['duration_seconds']) + .001]}
    stats = {'created_utc': datetime.now(timezone.utc).isoformat(), 'dataset_root': str(root.resolve()),
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'metadata_sha256': hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        'qa_count': len(samples), 'video_count': len(videos), 'response_record_count': len(all_responses),
        'scored_response_count': len(refs), 'excluded_response_count': len(all_responses) - len(refs),
        'roles': dict(Counter(r.get('role', 'unspecified') for r in all_responses)),
        'mode': dict(mode), 'ability': dict(ability), 'task': dict(tasks), 'ability_mode_matrix': matrix,
        'response_trigger_modality': dict(modalities), 'qa_trigger_modality_union': dict(qa_modalities),
        'preferred_modality': dict(Counter(s['generation_context'].get('preferred_modality', 'missing') for s in samples)),
        'vision_dependency': dict(Counter(s.get('vision_dependency', 'missing') for s in samples)),
        'audio_dependency': dict(Counter(s.get('audio_dependency', 'missing') for s in samples)),
        'dataset_source': dict(Counter(s.get('dataset_source', 'missing') for s in samples)),
        'question_position': dict(Counter(s['generation_context'].get('question_position', 'missing') for s in samples)),
        'qa_per_video': distribution(list(videos.values())),
        'qa_per_video_counts': dict(sorted(Counter(videos.values()).items())),
        'abilities_per_video': distribution([len(a) for a in abilities_per_video.values()]),
        'scored_responses_per_qa': distribution([len(by_qa[i]) for i in range(len(samples))]),
        'video_duration_seconds': distribution(durations), 'total_unique_video_hours': sum(durations) / 3600,
        'video_bytes': sum((root / v).stat().st_size for v in videos if (root / v).is_file()),
        'question_time_seconds': distribution(questions),
        'first_trigger_time_seconds': distribution(first_triggers), 'last_trigger_time_seconds': distribution(last_triggers),
        'all_trigger_time_seconds': distribution(triggers), 'first_response_delay_seconds': distribution(first_delays),
        'all_response_delay_seconds': distribution(response_delays), 'window_width_seconds': distribution(window_widths),
        'required_tools': dict(required_tools), 'window_boundary_details': window_details,
        'training_time_eligible': candidate_counts, 'training_time_eligible_by_mode': candidate_modes,
        'training_time_eligible_videos': candidate_videos, 'training_time_excluded': excluded_counts, 'checks': checks}
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / 'aoi_dataset_statistics.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2) + '\n')
    import csv
    with (output / 'aoi_dataset_video_statistics.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['video', 'original_file', 'duration_seconds', 'qa_count', 'ability_count', 'title'])
        for v in sorted(videos):
            r = indexed[v]
            writer.writerow([v, r.get('original_file'), r['duration_seconds'], videos[v], len(abilities_per_video[v]), r.get('title')])
    with (output / 'aoi_dataset_window_boundary_issues.csv').open('w', newline='') as f:
        fields = ['sample_index', 'response_index', 'video', 'duration_seconds', 'window_start_seconds',
                  'trigger_time_seconds', 'window_end_seconds', 'end_excess_seconds',
                  'trigger_inside_video', 'window_order_valid']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(window_details)
    check_summary = {k: len(v) if isinstance(v, list) else v for k, v in checks.items()}
    lines = ['# AOI-Dataset 训练数据统计', '', f'统计时间：{stats["created_utc"]}。', '',
             '数据集：[overji/AOI-Dataset](https://modelscope.cn/datasets/overji/AOI-Dataset)，视频来源全部为 FineVideo。',
             f'本次本地来源：`{root.resolve()}`。统计过程只读，不改写数据、不加载模型、不启动训练。', '',
             '## 总览', '', '| 指标 | 数值 |', '|---|---:|',
             f'| QA | {len(samples):,} |', f'| 唯一视频 | {len(videos):,} |',
             f'| 平均 QA / 视频 | {len(samples) / len(videos):.2f} |',
             f'| 平均不同任务类别 / 视频 | {stats["abilities_per_video"]["mean"]:.2f} |',
             f'| 原始 response 记录 | {len(all_responses):,} |', f'| 可评分 response | {len(refs):,} |',
             f'| 唯一视频总时长 | {sum(durations)/3600:.2f} 小时 |',
             f'| 视频文件总大小 | {stats["video_bytes"]/2**30:.2f} GiB |',
             f'| Agent 负样本 | {len(negatives)} |', '',
             '视频总时长按唯一视频计数；QA 与 response 均不是独立视频数。原始 response 包含工具结果及后续完成回复。',
             f'按当前 benchmark 的 reference_windows 口径排除 {len(all_responses)-len(refs):,} 条非触发记录后，剩余 {len(refs):,} 条用于后续 response 模态、触发时间、窗口及延时统计。',
             'Agent 只计含 agent_call/task_start 的 assistant 触发回复，不将模拟工具结果或工具返回后的完成回复算作触发。', '',
             '## 交互类型', '', '| 交互类型 | QA | 占比 |', '|---|---:|---:|']
    for label in ['rtd', 'pr', 'agent']:
        lines.append(f'| {label.upper()} | {mode[label]:,} | {mode[label]/len(samples):.2%} |')
    lines += ['', '## 任务类别 × 交互类型', '', '| 任务类别 | RTD | PR | Agent | 总计 | 占比 |', '|---|---:|---:|---:|---:|---:|']
    for a, row in matrix.items():
        total = sum(row.values())
        lines.append(f'| {a} | {row["rtd"]} | {row["pr"]} | {row["agent"]} | {total} | {total/len(samples):.2%} |')
    def counter_table(title, counts, denominator):
        lines.extend(['', '## ' + title, '', '| 类别 | 数量 | 占比 |', '|---|---:|---:|'])
        for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0]))):
            lines.append(f'| {k} | {v:,} | {v/denominator:.2%} |')
    counter_table('触发模态：按可评分 response', modalities, len(refs))
    lines += ['', '`speech` 指语言语音，`sound` 指非语言声音；组合标签表示同一触发依赖的证据模态。']
    counter_table('QA 模态：该 QA 全部触发模态的并集', qa_modalities, len(samples))
    lines += ['', 'QA 采用互斥的模态并集统计；某个 QA 的不同回复可分别为 visual 与 speech，其 QA 并集会计为 visual+speech。']
    counter_table('生成采样偏好 preferred_modality', Counter(stats['preferred_modality']), len(samples))
    lines += ['', 'preferred_modality 是生成时的采样偏好，不等于实际 trigger_type，不能将 audio 偏好占比当成纯音频 QA 占比。']
    counter_table('视觉依赖 vision_dependency', Counter(stats['vision_dependency']), len(samples))
    counter_table('音频依赖 audio_dependency', Counter(stats['audio_dependency']), len(samples))
    lines += ['', '## 视频时长、触发时间与交互延时', '', '| 指标（秒） | N | 均值 | 中位数 | P90 | P95 | 最小 | 最大 |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for label, key in [('唯一视频时长', 'video_duration_seconds'), ('问题输入时间', 'question_time_seconds'),
                       ('首次 trigger（距视频起点）', 'first_trigger_time_seconds'),
                       ('末次 trigger（距视频起点）', 'last_trigger_time_seconds'), ('全部 trigger（距视频起点）', 'all_trigger_time_seconds'),
                       ('question → 首次 trigger', 'first_response_delay_seconds'),
                       ('question → 每个 trigger', 'all_response_delay_seconds'), ('回复窗口宽度', 'window_width_seconds')]:
        d = stats[key]
        lines.append(f'| {label} | {d["count"]:,} | {d["mean"]:.2f} | {d["median"]:.2f} | {d["p90"]:.2f} | {d["p95"]:.2f} | {d["min"]:.2f} | {d["max"]:.2f} |')
    lines += ['', '延时指参考标注的 `trigger_time - question_time`，不是模型推理 wall time，也不是工具执行耗时；窗口宽度为 window_end − window_start。分位数采用线性插值。']
    counter_table('每视频 QA 数量分布（视频为分母）', Counter(videos.values()), len(videos))
    counter_table('Agent 必需工具', required_tools, mode['agent'])
    lines += ['', '同一条 Agent QA 可要求多个工具，因此必需工具占比之和可能超过 100%。']
    lines += ['', '## 当前训练配置下的时间条件覆盖', '',
              '每条 QA 取最早一个可评分 response，保留从视频第 0 秒开始的完整前缀；目标发声秒同时受问题输入、证据 available_time、trigger confirmed_time、窗口和整数秒因果约束。',
              '下表调用当前 train.first_record 逻辑，仅在内存中改变候选前缀上限做统计，没有修改训练配置。', '',
              '| 前缀上限（秒） | 合格 QA | 占全部 QA | 覆盖视频 | RTD | PR | Agent |', '|---|---:|---:|---:|---:|---:|---:|']
    for limit, n in candidate_counts.items():
        c = candidate_modes[limit]
        lines.append(f'| {limit} | {n:,} | {n/len(samples):.2%} | {candidate_videos[limit]} | {c.get("rtd",0)} | {c.get("pr",0)} | {c.get("agent",0)} |')
    lines += ['', '**这些是时间与媒体存在性条件下的候选数，不是编码成功数。** 尚未逐样本编码检查 `max_sequence_tokens=8192`，更长前缀仍可能超 token 上限或显存。',
              f'默认 32 秒排除原因：`{json.dumps(excluded_counts["32"], ensure_ascii=False)}`。默认 prepare 的 --max-samples 仍为 12，不会自动使用全部 5,000 条。', '',
              '## 数据完整性检查', '', '```json', json.dumps(check_summary, ensure_ascii=False, indent=2), '```', '',
              '上表的列表型检查项以问题记录数量展示，完整索引在 JSON 中。窗口边界问题详见 [CSV](aoi_dataset_window_boundary_issues.csv)。',
              (f'本次有 {len(window_details)} 条可评分窗口未通过严格边界检查；其中仅窗口末尾越过视频时长且其他时序合法的有 '
               f'{sum(r["window_order_valid"] and r["trigger_inside_video"] and r["end_excess_seconds"] > 0 for r in window_details)} 条，'
               f'最大越界 {max([r["end_excess_seconds"] for r in window_details] or [0]):.6f} 秒。未修改或裁剪原始窗口。'), '',
              '已检查全部引用文件存在、metadata 计数、每视频 QA 数、时长一致性、可评分窗口边界及问题/触发时序。未逐个重新解码视频，未校验媒体内容哈希；未执行语义正确性或感知证据审核。',
              '当前发布 metadata 没有视频体裁分类字段，报告不根据标题推测体裁。metadata 的 source_qa 和生成机路径属于来源记录，不是当前训练所依赖的可用路径。', '',
              '## 版本与复现', '', f'- all_qa.json SHA-256：`{stats["source_sha256"]}`',
              f'- video_metadata.json SHA-256：`{stats["metadata_sha256"]}`',
              '- 完整数值：[aoi_dataset_statistics.json](aoi_dataset_statistics.json)',
              '- 逐视频统计：[aoi_dataset_video_statistics.csv](aoi_dataset_video_statistics.csv)', '',
              '在仓库根目录执行；--plots 仅需要现有 matplotlib，不加载模型：', '', '```bash',
              'python train/scripts/dataset_statistics.py --plots',
              '# 自定义下载目录：',
              'python train/scripts/dataset_statistics.py --dataset-dir /path/to/AOI-Dataset --output-dir train/docs --plots', '```']
    if args.plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                             'axes.spines.right': False, 'axes.titleweight': 'bold', 'axes.titlepad': 12})
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), layout='constrained')
        blue, teal, orange = '#376795', '#398A80', '#D58C4E'
        a = axes[0, 0]
        labels, values = zip(*ability.most_common())
        a.barh([x.replace('_', ' ') for x in labels][::-1], list(values)[::-1], color=blue)
        a.set_title('Task categories (QA)');a.set_xlabel('QA count')
        a = axes[0, 1]
        labels = ['RTD', 'PR', 'Agent'];values = [mode[x.lower()] for x in labels]
        a.pie(values, labels=labels, autopct='%1.1f%%', startangle=90,
              colors=[blue, teal, orange], wedgeprops={'edgecolor': 'white', 'linewidth': 2})
        a.set_title('Interaction types (QA)')
        a = axes[0, 2]
        labels, values = zip(*modalities.most_common())
        a.barh(list(labels)[::-1], list(values)[::-1], color=teal)
        a.set_title('Trigger modalities (scored responses)');a.set_xlabel('Response count')
        for a, values, title, label in [(axes[1,0], durations, 'Unique-video durations', 'Seconds'),
                                         (axes[1,1], triggers, 'Trigger times from video start', 'Seconds'),
                                         (axes[1,2], first_delays, 'Question to first trigger delay', 'Seconds')]:
            a.hist(values, bins=30, color=blue, edgecolor='white', linewidth=.6)
            a.axvline(statistics.median(values), color=orange, linestyle='--', linewidth=2,
                      label=f'Median {statistics.median(values):.1f}s')
            a.set_title(title);a.set_xlabel(label);a.set_ylabel('Count');a.legend(frameon=False)
        fig.suptitle(f'AOI-Dataset  |  {len(samples):,} QA · {len(videos):,} videos · {len(refs):,} scored responses', fontsize=18, fontweight='bold')
        for suffix in ['png', 'svg']:
            fig.savefig(output / f'aoi_dataset_overview.{suffix}', dpi=180, bbox_inches='tight')
        plt.close(fig)
        lines[lines.index('## 交互类型'):lines.index('## 交互类型')] = [
            '![AOI-Dataset statistics](aoi_dataset_overview.png)', '', '[SVG 图表](aoi_dataset_overview.svg)', '']
    (output / 'aoi_dataset_statistics.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({k: stats[k] for k in ['qa_count', 'video_count', 'response_record_count', 'scored_response_count',
                      'mode', 'video_duration_seconds', 'training_time_eligible']} | {'checks': check_summary}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
