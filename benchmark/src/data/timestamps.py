"""Benchmark timestamp conversion."""

def seconds(value):
    if isinstance(value, (int, float)):
        return float(value)
    parts = [float(part) for part in value.split(":")]
    if len(parts) not in (2, 3) or any(part < 0 for part in parts):
        raise ValueError(f"Invalid timestamp: {value}")
    return sum(part * 60 ** index for index, part in enumerate(reversed(parts)))
