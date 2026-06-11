from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from .normalization import canonical
from .schema import MCQItem


def _key(query: Any, k: int) -> str:
    text = query.text_for_retrieval() if isinstance(query, MCQItem) else str(query)
    digest = hashlib.sha1(canonical(text).encode("utf-8")).hexdigest()[:24]
    return f"{digest}:{k}"


@dataclass(slots=True)
class CachedRAG:
    rag: Any
    max_entries: int = 4096
    cache: dict[str, list[Any]] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0
    name: str = "cached_rag"

    def __post_init__(self) -> None:
        self.name = "cached_" + getattr(self.rag, "name", type(self.rag).__name__)

    @property
    def docs(self):
        return getattr(self.rag, "docs", [])

    def search(self, query: Any, k: int = 3):
        key = _key(query, k)
        if key in self.cache:
            self.hits += 1
            return list(self.cache[key])
        self.misses += 1
        res = list(self.rag.search(query, k=k))
        if len(self.cache) >= self.max_entries:
            oldest = next(iter(self.cache))
            self.cache.pop(oldest, None)
        self.cache[key] = list(res)
        return res

    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "entries": len(self.cache)}
