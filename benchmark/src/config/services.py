"""JoyAI service defaults; explicit service JSON and environment overrides win."""
JOYAI_DEFAULTS = {
    "startup_timeout_seconds": 1200,
    "port_base": 17600,
    "main_memory_utilization": 0.65,
    "summary_memory_utilization": 0.25,
}
JOYAI_ENVIRONMENT = {
    "MAX_MODEL_LEN": "131072", "SUMMARY_MAX_MODEL_LEN": "65536",
    "TENSOR_PARALLEL_SIZE": "1", "DATA_PARALLEL_SIZE": "1",
    "DATA_PARALLEL_SIZE_LOCAL": "1", "SUMMARY_FOREGROUND": "1",
    "MAIN_SMOKE_ENABLE": "0", "SUMMARY_SMOKE_ENABLE": "0",
    "SAVE_SERVICE_LOGS": "0",
}
