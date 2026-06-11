
from __future__ import annotations

"""Evidence compression: keep high-signal snippets under context budget."""

import re
from dataclasses import dataclass

from .features import tokenize
from .normalization import truncate

@dataclass(slots=True)
class EvidenceSnippet:
    doc_id: str
    text: str
    score: float = 0.0
    source: str = "offline"


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?。！？])\s+|\n+", text)
    out = [p.strip() for p in parts if p and p.strip()]
    if len(out) <= 1 and len(text) > 600:
        return [text[i:i+500].strip() for i in range(0, len(text), 500) if text[i:i+500].strip()]
    return out


def score_text(query: str, text: str) -> float:
    q = set(tokenize(query))
    if not q:
        return 0.0
    t = tokenize(text)
    if not t:
        return 0.0
    overlap = sum(1 for x in t if x in q)
    unique = len(set(t) & q)
    return overlap / max(8, len(t)) + unique / max(4, len(q))


def compress_evidence(query: str, snippets: list[EvidenceSnippet], max_chars: int = 1800) -> str:
    candidates: list[EvidenceSnippet] = []
    for snip in snippets:
        for sent in split_sentences(snip.text):
            score = max(snip.score, score_text(query, sent))
            if score > 0:
                candidates.append(EvidenceSnippet(snip.doc_id, sent, score, snip.source))
    if not candidates:
        candidates = snippets[:]
    candidates.sort(key=lambda s: s.score, reverse=True)
    chunks: list[str] = []
    used = 0
    seen: set[str] = set()
    for c in candidates:
        text = truncate(c.text.replace("\x00", " ").strip(), 700)
        if not text or text in seen:
            continue
        label = f"[{c.source}:{c.doc_id}] "
        part = label + text
        if used + len(part) + 2 > max_chars:
            remain = max_chars - used - len(label) - 2
            if remain <= 80:
                break
            part = label + text[:remain]
        chunks.append(part)
        seen.add(text)
        used += len(part) + 2
        if used >= max_chars:
            break
    return "\n\n".join(chunks)
