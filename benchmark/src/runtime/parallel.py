"""Fixed data shards, isolated GPU processes and live aggregate artifacts."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from benchmark.data.files import read_jsonl, write_json
from benchmark.data.agent_inputs import AGENT_INPUT_PROTOCOL
from benchmark.data.paths import ROOT
from benchmark.evaluation.collection import collect


def assignments(samples, args, count, sample_indices=None):
    if count < 1 or args.max_samples < 0:
        raise ValueError("Worker count must be positive and max-samples nonnegative")
    start, step = getattr(args, "sample_start", 0), getattr(args, "sample_step", 1)
    if start < 0 or step < 1:
        raise ValueError("sample-start must be nonnegative and sample-step positive")
    if sample_indices is not None and (
            not isinstance(sample_indices, list) or
            any(type(index) is not int or not 0 <= index < len(samples) for index in sample_indices) or
            len(set(sample_indices)) != len(sample_indices)):
        raise ValueError("sample-indices must contain unique valid original indices")
    indices = (list(range(len(samples))) if sample_indices is None else sample_indices)[start::step]
    if args.max_samples:
        indices = indices[:args.max_samples]
    return [indices[rank::count] for rank in range(count)]


def prepare_plan(args, model, gpus, worker_python, services=None, sample_indices=None):
    if not gpus or len(set(gpus)) != len(gpus) or any(not str(g).isdigit() for g in gpus):
        raise ValueError("Specify distinct numeric GPU indices")
    if hasattr(args, "device") and args.device not in ("cuda", "cuda:0"):
        raise ValueError("Parallel workers use local cuda:0; select physical GPUs with --gpus")
    samples = json.loads(args.data.read_text())
    shards = assignments(samples, args, len(gpus), sample_indices)
    config = {
        "agent_input_protocol": AGENT_INPUT_PROTOCOL,
        "model": model, "gpus": [str(g) for g in gpus],
        "worker_python": str(Path(worker_python).absolute()),
        "arguments": {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "prompt_sha256": hashlib.sha256(args.system_prompt.read_bytes()).hexdigest(),
        "assignments": shards,
        "services": services,
    }
    if sample_indices is not None:
        config["sample_indices"] = sample_indices
    plan_path = args.output_dir / "parallel_plan.json"
    if not plan_path.exists() and (
            (args.output_dir / "predictions.jsonl").exists() or
            (args.output_dir / "run_config.json").exists() or
            any(args.output_dir.glob("worker_*/predictions.jsonl"))):
        raise ValueError("Output contains existing results without a parallel plan; use a new directory")
    if plan_path.exists() and json.loads(plan_path.read_text()) != config:
        raise ValueError("Parallel configuration changed; use a new output directory")
    write_json(plan_path, config)
    return samples, config


def worker_command(args, model, worker_python, plan_path, rank, adapter_url=None):
    command = [str(worker_python), "-u", "-m", "benchmark", "run", model]
    values = dict(vars(args))
    values["output_dir"] = args.output_dir / f"worker_{rank}"
    if "device" in values:
        values["device"] = "cuda:0"
    if adapter_url is not None:
        values["adapter_url"] = adapter_url
    for key, value in values.items():
        command.extend(["--" + key.replace("_", "-"), str(value)])
    command.extend(["--internal-shard-plan", str(plan_path), "--internal-rank", str(rank)])
    return command


def terminate_owned(processes, grace_seconds=10):
    """Only signal process groups created by this supervisor."""
    for process in processes:
        # Descendant services can outlive an exited launcher.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + grace_seconds
    while any(p.poll() is None for p in processes) and time.monotonic() < deadline:
        time.sleep(0.1)
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run_parallel(args, model, gpus, *, worker_python=None, poll_seconds=10,
                 services=None, sample_indices=None):
    worker_python = worker_python or sys.executable
    if not Path(worker_python).is_file():
        raise FileNotFoundError(worker_python)
    if poll_seconds <= 0:
        raise ValueError("poll-seconds must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / ".parallel.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        selected = json.loads(Path(sample_indices).read_text()) if sample_indices is not None else None
        return _supervise(args, model, gpus, worker_python, poll_seconds, services, selected)


def _supervise(args, model, gpus, worker_python, poll_seconds, services, sample_indices=None):
    if model == "joyai" and services is None:
        raise ValueError("JoyAI data parallel requires --joyai-services for independent backends")
    samples, plan = prepare_plan(args, model, gpus, worker_python, services, sample_indices)
    plan_path = args.output_dir / "parallel_plan.json"
    shard_dirs = [args.output_dir / f"worker_{rank}" for rank in range(len(gpus))]
    expected = sorted(index for shard in plan["assignments"] for index in shard)
    processes, logs = [], []
    service_manager = None
    status = {"supervisor_pid": os.getpid(), "model": model, "status": "starting",
              "expected_samples": len(expected), "workers": []}
    status_path = args.output_dir / "parallel_status.json"
    previous_handlers = {}

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[sig] = signal.signal(sig, interrupted)
        write_json(status_path, status)
        urls = None
        if services is not None:
            from benchmark.runtime.joyai_services import JoyAIServices
            service_manager = JoyAIServices(services, args.output_dir, args.cache_dir)
            urls = service_manager.start(gpus)
        for rank, gpu in enumerate(gpus):
            directory = shard_dirs[rank]
            directory.mkdir(exist_ok=True)
            log = (directory / "worker.log").open("ab")
            logs.append(log)
            environment = os.environ.copy()
            environment.update(CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID")
            command = worker_command(args, model, worker_python, plan_path, rank,
                                     urls[rank] if urls else None)
            process = subprocess.Popen(command, cwd=ROOT.parent, env=environment,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            processes.append(process)
            status["workers"].append({"rank": rank, "gpu": str(gpu), "pid": process.pid,
                                      "log": str(directory / "worker.log"), "exit_code": None})
            print(f"START model={model} rank={rank} gpu={gpu} pid={process.pid}", flush=True)
        status["status"] = "running"
        while True:
            metrics = collect(samples, shard_dirs, args.output_dir, live=True,
                              expected_indices=expected)
            for worker, process, directory in zip(status["workers"], processes, shard_dirs):
                worker["exit_code"] = process.poll()
                worker["completed_samples"] = len(read_jsonl(directory / "predictions.jsonl",
                                                             complete_only=True))
                if worker["exit_code"] not in (None, 0):
                    worker["failure_tail"] = (directory / "worker.log").read_text(
                        errors="replace")[-4000:]
            status["completed_samples"] = metrics["evaluated_samples"]
            status["missing_sample_indices"] = sorted(set(expected) - {
                row["sample_index"] for row in metrics["per_sample"]})
            predictions = read_jsonl(args.output_dir / "predictions.jsonl")
            for rank, directory in enumerate(shard_dirs):
                recorded = {row["sample_index"] for row in read_jsonl(
                    directory / "predictions.jsonl", complete_only=True)}
                if not recorded.issubset(plan["assignments"][rank]):
                    raise ValueError(f"worker_{rank} contains predictions outside its assignment")
            status["partial_sample_indices"] = [p["sample_index"] for p in predictions
                                                if p["status"] != "complete"]
            done = all(process.poll() is not None for process in processes)
            if done:
                failed = any(process.returncode != 0 for process in processes)
                status["status"] = "failed" if failed else (
                    "incomplete" if status["missing_sample_indices"] or
                    status["partial_sample_indices"] else "complete")
            write_json(status_path, status)
            print(f"PROGRESS model={model} {status['completed_samples']}/{len(expected)} "
                  f"status={status['status']}", flush=True)
            if done:
                return 0 if status["status"] == "complete" else 1
            if service_manager is not None:
                service_manager.check_alive()
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        status.update(status="interrupted")
        write_json(status_path, status)
        return 130
    except Exception as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(status_path, status)
        raise
    finally:
        terminate_owned(processes)
        if service_manager is not None:
            service_manager.close()
        for log in logs:
            log.close()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
