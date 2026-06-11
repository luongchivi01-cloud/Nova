from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional


class PromptCache:
    """Small JSONL cache for local benchmarks/re-runs.

    Official runs can leave CACHE_PATH empty. When enabled, repeated prompts are
    returned without another model call, which helps public-test iteration and
    protects against crashes during long 2000-row experiments.
    """

    def __init__(self, path: str | os.PathLike[str] | None = None):
        self.path = Path(path) if path else None
        self.data: dict[str, str] = {}
        if self.path and self.path.exists():
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        k, v = obj.get("k"), obj.get("v")
                        if isinstance(k, str) and isinstance(v, str):
                            self.data[k] = v
                    except Exception:
                        continue

    @staticmethod
    def key(prompt: str, qid: str = "") -> str:
        return hashlib.sha256((qid + "\n" + prompt).encode("utf-8", "ignore")).hexdigest()

    def get(self, prompt: str, qid: str = "") -> Optional[str]:
        if not self.path:
            return None
        return self.data.get(self.key(prompt, qid))

    def set(self, prompt: str, value: str, qid: str = "") -> None:
        if not self.path:
            return
        k = self.key(prompt, qid)
        if k in self.data:
            return
        self.data[k] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"k": k, "v": value}, ensure_ascii=False) + "\n")
