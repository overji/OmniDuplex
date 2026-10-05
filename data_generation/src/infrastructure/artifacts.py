"""Content fingerprints and atomic batch artifact writes."""
import hashlib
import json
from .paths import ROOT


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def implementation_digest():
    paths = sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "prompts").rglob("*.md"))
    return digest([(str(p.relative_to(ROOT)), file_digest(p)) for p in paths])


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


