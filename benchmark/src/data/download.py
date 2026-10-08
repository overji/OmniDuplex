"""Verify/download an exact Hugging Face snapshot, including all model assets."""
import hashlib
import json


def digest(path, lfs):
    h = hashlib.sha256() if lfs else hashlib.sha1()
    if not lfs:
        h.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def execute(args):
    import requests
    from huggingface_hub import HfApi, hf_hub_url

    model = HfApi().model_info(args.repo, revision=args.revision, files_metadata=True)
    files = []
    for entry in model.siblings:
        path = args.model_dir / entry.rfilename
        expected = entry.lfs.sha256 if entry.lfs else entry.blob_id
        valid = path.is_file() and path.stat().st_size == entry.size and digest(path, entry.lfs) == expected
        print(f"{'verified' if valid else 'downloading'} {entry.rfilename}", flush=True)
        if not valid:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + ".downloading")
            with requests.get(hf_hub_url(args.repo, entry.rfilename, revision=model.sha), stream=True, timeout=120) as response:
                response.raise_for_status()
                with temporary.open("wb") as output:
                    for block in response.iter_content(8 * 1024 * 1024):
                        output.write(block)
            if digest(temporary, entry.lfs) != expected:
                raise ValueError(f"Hash mismatch: {entry.rfilename}")
            temporary.replace(path)
        files.append({"name": entry.rfilename, "size": path.stat().st_size, "hash": expected})
    args.manifest.write_text(json.dumps({"repo": args.repo, "revision": model.sha,
        "model_dir": str(args.model_dir), "files": files}, indent=2) + "\n")
