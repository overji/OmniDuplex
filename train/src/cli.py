import argparse
import json
from pathlib import Path

from train.src.data import ROOT, load_config


def main():
    parser = argparse.ArgumentParser(description='MiniCPM-o native-duplex SFT and full-parameter GRPO')
    parser.add_argument('--config', type=Path, default=ROOT / 'train/configs/minicpmo_lora.json')
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare')
    prepare.add_argument('--data', type=Path, required=True)
    prepare.add_argument('--output', type=Path, required=True)
    prepare.add_argument('--max-samples', type=int, default=12)
    prepare.add_argument('--validation-fraction', type=float, default=.2)
    estimate = commands.add_parser('estimate')
    estimate.add_argument('--lengths', type=int, nargs='+', default=[2048, 4096, 8192])
    export = commands.add_parser('export')
    export.add_argument('--checkpoint', type=Path, required=True)
    export.add_argument('--output', type=Path, required=True)
    for name in ('run', 'profile', 'grpo'):
        sub = commands.add_parser(name)
        sub.add_argument('--prepared', type=Path, required=True)
        sub.add_argument('--output', type=Path, required=True)
        sub.add_argument('--device', default='cuda:0')
        sub.add_argument('--max-steps', type=int, default=0)
        sub.add_argument('--sample-index', type=int, help='Profile a selected original sample index')
        sub.add_argument('--resume', type=Path, help='Full-training checkpoint directory')
        sub.add_argument('--local_rank', '--local-rank', type=int, default=-1)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == 'prepare':
        from train.src.data import prepare
        prepare(args.data, args.output, config, args.max_samples, args.validation_fraction)
    elif args.command == 'estimate':
        from train.src.estimate import estimate
        print(json.dumps(estimate(config, args.lengths), indent=2))
    elif args.command == 'export':
        from train.src.full import export_checkpoint
        export_checkpoint(args.checkpoint, args.output)
    else:
        if args.max_steps < 0:
            parser.error('max-steps must be nonnegative')
        args.profile = args.command == 'profile'
        method = config['training_method']
        if method != 'lora_sft' and args.device != 'cuda:0':
            parser.error('Full training selects GPUs through CUDA_VISIBLE_DEVICES and LOCAL_RANK; do not override --device')
        if args.command == 'grpo' and method != 'full_grpo':
            parser.error('grpo requires training_method=full_grpo')
        if method == 'full_grpo' and args.command not in ('grpo', 'profile'):
            parser.error('Use the grpo command with full_grpo config')
        if method == 'lora_sft':
            if args.resume:
                parser.error('Optimizer resume is supported by full training only')
            from train.src.runner import run
        else:
            from train.src.full import run
        run(args, config)
