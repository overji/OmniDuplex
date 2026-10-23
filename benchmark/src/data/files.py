"""JSON artifacts shared by running and reporting."""
import json
import os
import tempfile


def atomic_write_text(path, text):
    """Readers see either the previous artifact or the complete replacement."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent,
                                         prefix=path.name + ".", delete=False, encoding="utf-8") as output:
            temporary = output.name
            output.write(text)
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, data):
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def read_jsonl(path, *, complete_only=False):
    if not path.exists():
        return []
    data = path.read_bytes()
    if complete_only and data and not data.endswith(b"\n"):
        data = data[:data.rfind(b"\n") + 1]
    return [json.loads(line) for line in data.decode().splitlines()]


def recover_jsonl_tail(path):
    """Only a stopped worker may remove its unfinished final record."""
    if path.exists():
        data = path.read_bytes()
        if data and not data.endswith(b"\n"):
            with path.open("r+b") as output:
                output.truncate(data.rfind(b"\n") + 1)
            print(f"Discarded interrupted final record in {path}", flush=True)


def append_jsonl(path, data):
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(data, ensure_ascii=False) + "\n")
        output.flush()
