"""Sequential background experiment queue, configured without shell commands."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from benchmark.data.files import write_json
from benchmark.data.paths import ROOT
from benchmark.runtime.parallel import terminate_owned


def execute(args):
    config = json.loads(args.config.read_text())
    root = Path(config["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config_path = root / "queue_config.json"
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            raise ValueError("Queue configuration changed; use another output directory")
        write_json(config_path, config)
        state = {"pid": os.getpid(), "status": "running", "jobs": []}
        status_path = root / "queue_status.json"
        current = None
        previous = {}

        def interrupted(signum, frame):
            raise KeyboardInterrupt

        try:
            for sig in (signal.SIGINT, signal.SIGTERM):
                previous[sig] = signal.signal(sig, interrupted)
            write_json(status_path, state)
            for job in config["jobs"]:
                name = job["model"]
                output = root / name
                output.mkdir(exist_ok=True)
                command = [sys.executable, "-u", "-m", "benchmark", "parallel", name,
                           "--gpus", *[str(gpu) for gpu in config["gpus"]],
                           "--output-dir", str(output), "--worker-python", job["python"],
                           *job["arguments"]]
                record = {"model": name, "output_dir": str(output), "status": "running"}
                state["jobs"].append(record)
                with (output / "supervisor.log").open("ab") as log:
                    current = subprocess.Popen(command, cwd=ROOT.parent, stdout=log,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                    record["pid"] = current.pid
                    write_json(status_path, state)
                    print(f"START model={name} pid={current.pid}", flush=True)
                    code = current.wait()
                record.update(status="complete" if code == 0 else "failed", exit_code=code)
                write_json(status_path, state)
                print(f"DONE model={name} exit_code={code}", flush=True)
                current = None
            from benchmark.evaluation.comparison import execute as compare
            runs = [Path(job["output_dir"]) for job in state["jobs"]
                    if (Path(job["output_dir"]) / "metrics.json").exists() and
                    json.loads((Path(job["output_dir"]) / "metrics.json").read_text())["evaluated_samples"]]
            if runs:
                from types import SimpleNamespace
                compare(SimpleNamespace(runs=runs, output=root / "model_comparison"))
            state["status"] = "complete" if all(j["status"] == "complete" for j in state["jobs"]) else "incomplete"
            write_json(status_path, state)
        except KeyboardInterrupt:
            state["status"] = "interrupted"
            write_json(status_path, state)
            raise SystemExit(130)
        finally:
            if current is not None:
                # Allow the parallel supervisor to clean up workers and native services.
                terminate_owned([current], grace_seconds=30)
            for sig, handler in previous.items():
                signal.signal(sig, handler)
