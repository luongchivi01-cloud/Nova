from __future__ import annotations

"""Hybrid retrieval utilities: BM25/dense results -> Reciprocal Rank Fusion."""

from dataclasses import dataclass
from typing import Iterable


@dataclass(slots=True)
class RetrievedDoc:
    doc_id: str
    text: str
    score: float
    source: str = "unknown"


def reciprocal_rank_fusion(rankings: Iterable[list[RetrievedDoc]], k: int = 60, limit: int = 5) -> list[RetrievedDoc]:
    scores: dict[str, float] = {}
    docs: dict[str, RetrievedDoc] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            docs.setdefault(doc.doc_id, doc)
            scores[doc.doc_id] = scores.get(doc.doc_id, 0.0) + 1.0 / (k + rank)
    fused = []
    for doc_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]:
        base = docs[doc_id]
        fused.append(RetrievedDoc(base.doc_id, base.text, score, base.source + "+rrf"))
    return fused


def compact_context(docs: list[RetrievedDoc], max_chars: int = 1600) -> str:
    chunks: list[str] = []
    used = 0
    for d in docs:
        text = d.text.strip().replace("\n", " ")
        if not text:
            continue
        add = f"[{d.source}:{d.doc_id}] {text}"
        if used + len(add) > max_chars:
            add = add[: max(0, max_chars - used)]
        if add:
            chunks.append(add)
            used += len(add)
        if used >= max_chars:
            break
    return "\n".join(chunks)
