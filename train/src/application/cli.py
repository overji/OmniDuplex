"""公共命令入口。所有模型/训练后端都在命令分支内延迟导入。"""

import argparse
import json
from pathlib import Path

from train.src.infrastructure.config import ROOT, load_config


def main():
    """解析公开CLI并按需分发。help/prepare/estimate路径不得因导入训练循环而初始化CUDA或DeepSpeed。"""
    parser = argparse.ArgumentParser(description='MiniCPM-o native-duplex full-parameter SFT and GRPO')
    parser.add_argument('--config', type=Path, default=ROOT / 'train/configs/minicpmo_full_sft.json')
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare')
    prepare.add_argument('--data', type=Path, help='QA JSON; defaults to config.data')
    prepare.add_argument('--output', type=Path, required=True)
    prepare.add_argument('--max-samples', type=int, default=12)
    prepare.add_argument('--validation-fraction', type=float, default=.2)
    audit = commands.add_parser('audit', help='CPU-only all-response scheduling and video-duration audit')
    audit.add_argument('--data', type=Path, help='QA JSON; defaults to config.data')
    audit.add_argument('--metadata', type=Path, help='Video metadata; defaults to videos_dir/video_metadata.json')
    audit.add_argument('--output', type=Path, required=True)
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
        from train.src.application.prepare import prepare
        if args.data is None:
            if not config.get('data'):
                parser.error('prepare requires --data or config.data')
            args.data = Path(config['data'])
        prepare(args.data, args.output, config, args.max_samples, args.validation_fraction)
    elif args.command == 'audit':
        from train.src.application.audit import audit
        audit(args.data or Path(config['data']), args.output, config, args.metadata)
    elif args.command == 'estimate':
        from train.src.application.estimate import estimate
        print(json.dumps(estimate(config, args.lengths), indent=2))
    elif args.command == 'export':
        from train.src.infrastructure.checkpoints import export_checkpoint
        export_checkpoint(args.checkpoint, args.output)
    else:
        if args.max_steps < 0:
            parser.error('max-steps must be nonnegative')
        args.profile = args.command == 'profile'
        method = config['training_method']
        if args.command == 'grpo' and method != 'full_grpo':
            parser.error('grpo requires training_method=full_grpo')
        if method == 'full_grpo' and args.command not in ('grpo', 'profile'):
            parser.error('Use the grpo command with full_grpo config')
        if args.device != 'cuda:0':
            parser.error('Select GPUs through CUDA_VISIBLE_DEVICES and LOCAL_RANK; do not override --device')
        from train.src.training.full import run
        run(args, config)
