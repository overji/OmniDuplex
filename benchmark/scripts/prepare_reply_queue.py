#!/usr/bin/env python3
"""Prepare the response/silence continuation from existing model commands."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from benchmark.data.files import write_json
from benchmark.scripts.run_prepared_suite import option


def prepare_commands(source, config, reuse_livestar=None, agent_indices=None):
    def find(mode, adapter, variant=None):
        matches = [c for c in source if (c['mode'], c['adapter'], c.get('variant')) == (mode, adapter, variant)]
        if len(matches) != 1:
            raise ValueError(f'Expected one source command for {mode}/{adapter}/{variant}')
        return deepcopy(matches[0])
    ordered = [find('probe', name) for name in config['probe_first']]
    ordered += [find(**entry) for entry in config['ablations']]
    ordered += [find('realtime', name) for name in config['realtime_after_ablations']]
    ordered += [find('probe', name) for name in config['probe_last']]
    for sequence, entry in enumerate(ordered):
        entry['sequence'] = sequence
        entry.pop('shell', None)
        entry.pop('skip_smoke', None)
        entry['variant'] = '_'.join(filter(None, [entry.get('variant'), 'reply_task_text']))
        argv = entry['argv']
        if entry['mode'] == 'probe':
            argv = option(argv, '--output-protocol', config['probe_protocol'])
            if config.get('probe_prompt_layout'):
                argv = option(argv, '--prompt-layout', config['probe_prompt_layout'])
            if config.get('probe_max_new_tokens') is not None:
                if type(config['probe_max_new_tokens']) is not int or config['probe_max_new_tokens'] < 1:
                    raise ValueError('probe_max_new_tokens must be a positive integer')
                argv = option(argv, '--max-new-tokens', config['probe_max_new_tokens'])
            prompt = Path(config['probe_prompt'])
            argv = option(argv, '--system-prompt', prompt if prompt.is_absolute() else REPO / prompt)
            entry['smoke_agent'] = True
        else:
            argv = option(argv, '--agent-output-protocol', config['realtime_protocol'])
            argv = option(argv, '--agent-call-schema', 'task-start')
            if entry['adapter'] == 'livestar':
                if reuse_livestar:
                    argv = option(argv, '--reuse-non-agent', reuse_livestar)
                if agent_indices:
                    entry['smoke_sample_indices'] = str(agent_indices)
        entry['argv'] = argv
    return ordered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-commands', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=REPO/'benchmark/configs/suites/reply_task_text.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reuse-livestar', type=Path)
    parser.add_argument('--agent-indices', type=Path)
    args = parser.parse_args()
    commands = prepare_commands(json.loads(args.source_commands.read_text()), json.loads(args.config.read_text()),
        args.reuse_livestar.resolve() if args.reuse_livestar else None,
        args.agent_indices.resolve() if args.agent_indices else None)
    if args.output.exists():
        raise ValueError('Use a fresh queue command file')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, commands)
    print(f'Prepared {len(commands)} continuation jobs; no inference started')


if __name__ == '__main__':
    main()
