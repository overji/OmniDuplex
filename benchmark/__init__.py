"""Benchmark package; implementation modules live under src."""
from pathlib import Path

# Preserve benchmark.cli/runtime/etc. imports and the python -m benchmark entrypoint.
__path__.append(str(Path(__file__).resolve().parent / "src"))
