from __future__ import annotations

"""Create a reproducibility manifest for final submission."""

import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path


def sha256_file(path: str | Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def hash_tree(root: str | Path, include_ext: tuple[str, ...] = (".py", ".txt", ".md", ".toml", ".sh", ".json", ".yml", ".yaml", "Dockerfile")) -> dict[str, str]:
    rootp = Path(root)
    out: dict[str, str] = {}
    skip = {".git", ".pytest_cache", "__pycache__", "output", "reports"}
    for p in sorted(rootp.rglob("*")):
        if not p.is_file() or any(part in skip for part in p.parts):
            continue
        if p.name == "Dockerfile" or p.suffix in include_ext:
            out[str(p.relative_to(rootp))] = sha256_file(p)
    return out


def git_commit(root: str | Path = ".") -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root), stderr=subprocess.DEVNULL, text=True, timeout=3).strip()
    except Exception:
        return None


def freeze_submission(root: str | Path = ".", model_dir: str | Path = "/models", out: str | Path = "reports/submission_manifest.json") -> dict[str, object]:
    rootp = Path(root)
    modelp = Path(model_dir)
    model_hashes: dict[str, str] = {}
    if modelp.exists():
        for p in sorted(modelp.rglob("*")):
            if p.is_file() and p.suffix.lower() in {".gguf", ".safetensors", ".bin", ".json", ".model"}:
                # Avoid hashing dozens of huge shards unless explicitly requested.
                if p.stat().st_size > 2_500_000_000 and os.getenv("HASH_HUGE_MODEL", "0") != "1":
                    model_hashes[str(p.relative_to(modelp))] = f"SKIPPED_HUGE_FILE_SIZE_{p.stat().st_size}"
                else:
                    model_hashes[str(p.relative_to(modelp))] = sha256_file(p)
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "git_commit": git_commit(rootp),
        "env_relevant": {k: os.getenv(k) for k in sorted(os.environ) if k.startswith(("LLM_", "SOLVER_", "MODEL", "STRICT", "USE_", "ENABLE_", "TIME_BUDGET", "N_CTX"))},
        "source_hashes": hash_tree(rootp),
        "model_hashes": model_hashes,
        "official_command": "docker run --rm --gpus all --network none -v $PWD/data:/data:ro -v $PWD/output:/output -v $PWD/models:/models:ro hackaithon-c:final",
    }
    outp = Path(out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".")
    p.add_argument("--models", default="/models")
    p.add_argument("--out", default="reports/submission_manifest.json")
    args = p.parse_args(argv)
    m = freeze_submission(args.root, args.models, args.out)
    print(json.dumps({"out": args.out, "files": len(m["source_hashes"]), "models": len(m["model_hashes"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
