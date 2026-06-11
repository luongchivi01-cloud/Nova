
from __future__ import annotations

"""Knowledge source manifest writer for reproducible offline RAG."""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

@dataclass(slots=True)
class KnowledgeSource:
    path: str
    kind: str
    files: int = 0
    docs: int = 0
    bytes: int = 0
    sha256_sample: str = ""
    notes: str = ""

@dataclass(slots=True)
class KnowledgeManifest:
    ok: bool = True
    sources: list[KnowledgeSource] = field(default_factory=list)
    total_docs: int = 0
    backends: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    def write(self, path: str | Path | None) -> None:
        if not path:
            return
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(), encoding="utf-8")


def hash_file_sample(path: Path, limit: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    try:
        with path.open("rb") as f:
            h.update(f.read(limit))
        return h.hexdigest()
    except Exception:
        return ""
