from __future__ import annotations

"""Runtime stability guard for knowledge-heavy official inference."""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

@dataclass(slots=True)
class KnowledgeRuntimeAudit:
    started_at: float = field(default_factory=time.time)
    docs: int = 0
    backends: list[str] = field(default_factory=list)
    query_count: int = 0
    empty_hits: int = 0
    max_query_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def record_query(self, ms: float, hits: int) -> None:
        self.query_count += 1
        self.max_query_ms = max(self.max_query_ms, ms)
        if hits <= 0:
            self.empty_hits += 1
        if ms > 1500:
            self.warnings.append(f"slow_knowledge_query_ms={ms:.1f}")

    def write(self, path: str | Path) -> None:
        p = Path(path)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({
                "docs": self.docs,
                "backends": self.backends,
                "query_count": self.query_count,
                "empty_hits": self.empty_hits,
                "max_query_ms": self.max_query_ms,
                "warnings": self.warnings[-50:],
                "seconds": time.time() - self.started_at,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
