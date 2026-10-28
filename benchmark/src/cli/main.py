"""Unified commands; optional model dependencies are loaded only for execution."""
import argparse
import importlib

from benchmark.cli import options


MODELS = ("minicpmo", "gander", "mmduet2", "livestar", "roma", "joyai")
REMOTE_MODELS = ("qwen_realtime", "gemini_live")
COMMANDS = {
    'prepare': 'benchmark.runtime.preparation',
    "probe": "benchmark.probe.runner",
    "judge": "benchmark.evaluation.judge",
    "probe-judge": "benchmark.evaluation.probe_judge",
    "judge-export": "benchmark.evaluation.judge_export",
    "judge-bundle": "benchmark.evaluation.judge_bundle",
    "collect": "benchmark.evaluation.collection",
    "compare": "benchmark.evaluation.comparison",
    "download": "benchmark.data.download",
    "queue": "benchmark.runtime.queue",
    "migrate-agent-windows": "benchmark.data.agent_windows",
}


def build_parser():
    parser = argparse.ArgumentParser(description="Realtime timing benchmark")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "parallel"):
        run = commands.add_parser(command, help="Run native streaming " + command)
        models = run.add_subparsers(dest="adapter_name", required=True)
        for name in MODELS + (REMOTE_MODELS if command == 'run' else ()):
            model_parser = models.add_parser(name)
            getattr(options, f"configure_{name}")(model_parser)
            if name == 'minicpmo':
                model_parser.add_argument('--input-modality', choices=('audio-video', 'video-only', 'audio-only'),
                                          default=argparse.SUPPRESS, help='Explicit media ablation; omitted keeps the baseline')
            if command == "parallel":
                options.configure_parallel(model_parser, joyai=name == "joyai")
            else:
                model_parser.add_argument("--internal-shard-plan", type=options.Path,
                                          help=argparse.SUPPRESS)
                model_parser.add_argument("--internal-rank", type=int, help=argparse.SUPPRESS)
    for name in COMMANDS:
        getattr(options, f"configure_{name.replace('-', '_')}")(commands.add_parser(name))
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    command = vars(args).pop("command")
    if command == "parallel":
        from benchmark.runtime.parallel import run_parallel
        model = vars(args).pop("adapter_name")
        controls = {key: vars(args).pop(key) for key in (
            "gpus", "worker_python", "poll_seconds", "sample_indices")}
        services = None
        if model == "joyai":
            services = vars(args).pop("joyai_services")
            if services is not None:
                import json
                services = json.loads(services.read_text())
        raise SystemExit(run_parallel(args, model, **controls, services=services))
    if command == "run":
        adapter_name = vars(args).pop("adapter_name")
        plan_path = vars(args).pop("internal_shard_plan")
        rank = vars(args).pop("internal_rank")
        assignment = None
        if plan_path is not None:
            from benchmark.data.agent_inputs import input_protocol
            import hashlib
            import json
            plan = json.loads(plan_path.read_text())
            if (rank is None or not 0 <= rank < len(plan["assignments"]) or
                    plan.get("agent_input_protocol") != input_protocol(args) or
                    plan["model"] != adapter_name or
                    plan["data_sha256"] != hashlib.sha256(args.data.read_bytes()).hexdigest() or
                    plan["prompt_sha256"] != hashlib.sha256(args.system_prompt.read_bytes()).hexdigest()):
                raise ValueError("Invalid worker assignment or changed dataset/prompt")
            assignment = plan["assignments"][rank]
        elif rank is not None:
            raise ValueError("internal-rank requires internal-shard-plan")
        module = ('minicpmo_ablation' if adapter_name == 'minicpmo'
                  and getattr(args, 'input_modality', 'audio-video') != 'audio-video' else adapter_name)
        adapter = importlib.import_module(f"benchmark.adapters.{module}")
        from benchmark.runtime.runner import run
        run(args, adapter, assignment=assignment)
    else:
        importlib.import_module(COMMANDS[command]).execute(args)
