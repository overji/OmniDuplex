"""Owned per-GPU native JoyAI main, summary and adapter services."""
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.request import urlopen

from benchmark.config.defaults import DEFAULTS
from benchmark.config.models import MODEL_SETTINGS
from benchmark.config.services import JOYAI_DEFAULTS, JOYAI_ENVIRONMENT
from benchmark.data.files import write_json


class JoyAIServices:
    def __init__(self, config, output_dir, cache_dir):
        self.config = config
        self.root = (output_dir / "services").resolve()
        self.cache_dir = cache_dir
        self.processes = []
        self.logs = []
        self.records = []

    def launch(self, command, environment, directory, name):
        log = (directory / (name + ".log")).open("ab")
        self.logs.append(log)
        if isinstance(command, Path):
            command = ["bash", str(command)]
        process = subprocess.Popen(command, env=environment, cwd=directory,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        self.processes.append(process)
        self.records.append({"name": name, "pid": process.pid,
                             "log": str(directory / (name + ".log")),
                             "gpu": environment["MAIN_GPU"]})
        write_json(self.root / "status.json", {"services": self.records})
        return process

    def check_alive(self):
        failed = [(p.pid, p.returncode) for p in self.processes if p.poll() is not None]
        if failed:
            raise RuntimeError(f"JoyAI service exited: {failed}; see {self.root}")

    def wait_ready(self, url):
        deadline = time.monotonic() + self.config.get("startup_timeout_seconds", JOYAI_DEFAULTS["startup_timeout_seconds"])
        while time.monotonic() < deadline:
            self.check_alive()
            try:
                with urlopen(url, timeout=5) as response:
                    if response.status == 200:
                        return
            except (OSError, ValueError):
                pass
            time.sleep(2)
        raise TimeoutError(f"JoyAI readiness timeout: {url}")

    def smoke(self, directory, environment, kind, api_base, model):
        script = Path(self.config["code"]) / "services/webinfer/smoke.py"
        log = (directory / (kind + "_smoke.log")).open("ab")
        self.logs.append(log)
        process = subprocess.Popen([
            self.config["python"], str(script), kind, "--api-base", api_base,
            "--model", model, "--attempts", "1", "--timeout", "120",
        ], env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        self.processes.append(process)
        try:
            code = process.wait(timeout=180)
        finally:
            if process.poll() is not None:
                self.processes.remove(process)
        if code != 0:
            raise RuntimeError(f"JoyAI {kind} real-request check failed; see {directory}")

    def start(self, gpus):
        scripts = Path(self.config["code"]) / "services/webinfer/scripts"
        urls = []
        self.root.mkdir(parents=True, exist_ok=True)
        write_json(self.root / "config.json", self.config)
        base = self.config.get("port_base", JOYAI_DEFAULTS["port_base"])
        for rank, gpu in enumerate(gpus):
            directory = self.root / f"worker_{rank}"
            directory.mkdir(exist_ok=True)
            main_port, summary_port, adapter_port = base + rank * 10, base + rank * 10 + 1, base + rank * 10 + 2
            main_api = f"http://127.0.0.1:{main_port}/v1"
            summary_api = f"http://127.0.0.1:{summary_port}/v1"
            model = DEFAULTS["joyai"]["model"]
            env = os.environ.copy()
            env.update({
                "PYTHON_BIN": self.config["python"], "VENV_ACTIVATE": "",
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "MAIN_GPU": str(gpu),
                "SUMMARY_GPU": str(gpu), "MODEL_PATH": self.config["main_model_path"],
                "SUMMARY_MODEL_PATH": self.config["summary_model_path"],
                "SUMMARIZER_MODEL": self.config["summary_model_path"],
                "MAIN_MODEL_PORT": str(main_port), "SUMMARY_PORT": str(summary_port),
                "ADAPTER_PORT": str(adapter_port), "MAIN_API_BASE": main_api,
                "SUMMARIZER_API_BASE": summary_api,
                "MAIN_BACKENDS": json.dumps([{"name": model, "model": model, "api_base": main_api}]),
                "SERVED_MODEL_NAME": model, "MAIN_MODEL": model,
                "MAIN_GPU_MEMORY_UTILIZATION": str(self.config.get("main_memory_utilization", JOYAI_DEFAULTS["main_memory_utilization"])),
                "SUMMARY_GPU_MEMORY_UTILIZATION": str(self.config.get("summary_memory_utilization", JOYAI_DEFAULTS["summary_memory_utilization"])),
                **JOYAI_ENVIRONMENT,
                "LOG_DIR": str(directory / "summary"),
                "FRAME_SAVE_DIR": str(directory / "frames"),
                "ALLOWED_LOCAL_IMAGE_ROOTS": str(self.cache_dir.resolve()),
                "CHUNK": str(MODEL_SETTINGS["joyai"]["chunk_frames"]),
                "COMPRESS_EVERY_N_CHUNKS": str(MODEL_SETTINGS["joyai"]["compression_every_n_chunks"]),
                "FRAME_SECONDS": str(MODEL_SETTINGS["joyai"]["frame_seconds"]),
            })
            env.update({str(k): str(v) for k, v in self.config.get("environment", {}).items()})
            # Check ports before launch: an existing service must never satisfy our readiness check.
            import socket
            for port in (main_port, summary_port, adapter_port):
                with socket.socket() as probe:
                    probe.bind(("127.0.0.1", port))
            self.launch(scripts / "start_model.sh", env, directory, "main")
            self.wait_ready(main_api + "/models")
            self.smoke(directory, env, "main-vlm", main_api, model)
            # The upstream summary launcher waits for its own tee process during
            # EXIT cleanup, which can conceal a failed backend. Supervise the same
            # official vLLM entrypoint directly with identical inference flags.
            summary_env = dict(env, CUDA_VISIBLE_DEVICES=str(gpu))
            self.launch([
                self.config["python"], "-m", "vllm.entrypoints.openai.api_server",
                "--model", self.config["summary_model_path"],
                "--served-model-name", self.config["summary_model_path"],
                "--port", str(summary_port), "--max-model-len", JOYAI_ENVIRONMENT["SUMMARY_MAX_MODEL_LEN"],
                "--enable-prefix-caching", "--enable-chunked-prefill",
                "--gpu-memory-utilization", env["SUMMARY_GPU_MEMORY_UTILIZATION"],
            ], summary_env, directory, "summary")
            self.wait_ready(summary_api + "/models")
            self.smoke(directory, env, "summary", summary_api, self.config["summary_model_path"])
            self.launch(scripts / "start_adapter.sh", env, directory, "adapter")
            self.wait_ready(f"http://127.0.0.1:{adapter_port}/health")
            # A real adapter request additionally checks the routed backend.
            from urllib.request import Request
            from urllib.parse import quote
            image = next(self.cache_dir.glob("*/frames/frame_*.jpg"), None)
            if image is None:
                raise FileNotFoundError("JoyAI service check requires one cached benchmark frame")
            payload = json.dumps({"model": model, "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "Describe this frame briefly."},
                    {"type": "image_url", "image_url": {
                        "url": "file://" + quote(str(image.resolve()))}},
                ]}], "frame_time_ranges": ["0.000s"]}).encode()
            with urlopen(Request(f"http://127.0.0.1:{adapter_port}/v1/chat/completions",
                                 data=payload, headers={"Content-Type": "application/json"}),
                         timeout=120) as response:
                json.loads(response.read())
            urls.append(f"http://127.0.0.1:{adapter_port}/v1/chat/completions")
        return urls

    def close(self):
        from benchmark.runtime.parallel import terminate_owned
        terminate_owned(self.processes)
        if self.root.exists():
            write_json(self.root / "status.json", {"status": "stopped", "services": self.records})
        for log in self.logs:
            log.close()
