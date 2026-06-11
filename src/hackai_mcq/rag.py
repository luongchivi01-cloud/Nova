from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .features import tokenize
from .third_party_registry import try_import


@dataclass(slots=True)
class RagDoc:
    doc_id: str
    text: str


class LexicalRAG:
    """Small offline RAG, no network, no heavy dependency.

    It is intentionally simple so Docker remains reproducible. You can mount
    a corpus directory and set ENABLE_RAG=1 RAG_CORPUS=/corpus.
    """

    def __init__(self, docs: list[RagDoc]):
        self.docs = docs
        self.doc_terms = [tokenize(d.text) for d in docs]
        self.df: dict[str, int] = {}
        for terms in self.doc_terms:
            for t in set(terms):
                self.df[t] = self.df.get(t, 0) + 1
        self.n = max(1, len(docs))

    @classmethod
    def from_path(cls, path: str | Path | None) -> "LexicalRAG | None":
        if not path:
            return None
        p = Path(path)
        if not p.exists():
            return None
        docs: list[RagDoc] = []
        if p.is_file():
            docs.extend(_load_file(p))
        else:
            for f in sorted(p.rglob("*")):
                if f.is_file() and f.suffix.lower() in {".txt", ".md", ".csv", ".jsonl"}:
                    docs.extend(_load_file(f))
        if not docs:
            return None
        return cls(docs)

    def search(self, query: str, k: int = 3) -> list[RagDoc]:
        q_terms = tokenize(query)
        if not q_terms:
            return []
        scores: list[tuple[float, int]] = []
        q_counts: dict[str, int] = {}
        for t in q_terms:
            q_counts[t] = q_counts.get(t, 0) + 1
        for idx, terms in enumerate(self.doc_terms):
            if not terms:
                continue
            tf: dict[str, int] = {}
            for t in terms:
                tf[t] = tf.get(t, 0) + 1
            score = 0.0
            denom = len(terms)
            for t, qf in q_counts.items():
                if t in tf:
                    idf = math.log((self.n + 1) / (1 + self.df.get(t, 0))) + 1.0
                    score += (tf[t] / denom) * idf * (1 + math.log(1 + qf))
            if score > 0:
                scores.append((score, idx))
        scores.sort(reverse=True)
        return [self.docs[i] for _, i in scores[:k]]

    def context_for(self, query: str, k: int = 3, max_chars: int = 1800) -> str:
        docs = self.search(query, k=k)
        chunks = []
        used = 0
        for d in docs:
            txt = d.text.strip().replace("\x00", " ")
            if not txt:
                continue
            remain = max_chars - used
            if remain <= 0:
                break
            part = txt[:remain]
            chunks.append(f"[{d.doc_id}] {part}")
            used += len(part)
        return "\n\n".join(chunks)


def _load_file(p: Path) -> list[RagDoc]:
    docs: list[RagDoc] = []
    try:
        if p.suffix.lower() in {".txt", ".md"}:
            text = p.read_text(encoding="utf-8", errors="ignore")
            # Split large files into blocks.
            blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
            for i, b in enumerate(blocks):
                docs.append(RagDoc(f"{p.name}:{i}", b))
        elif p.suffix.lower() == ".csv":
            with p.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    text = " | ".join(str(v) for v in row.values() if v)
                    if text.strip():
                        docs.append(RagDoc(f"{p.name}:{i}", text))
        elif p.suffix.lower() == ".jsonl":
            import json
            with p.open("r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    try:
                        obj = json.loads(line)
                        text = str(obj.get("text", obj))
                    except Exception:
                        text = line
                    if text.strip():
                        docs.append(RagDoc(f"{p.name}:{i}", text))
    except Exception:
        return []
    return docs


class BM25sRAG(LexicalRAG):
    """Optional adapter for the vendored bm25s repo.

    bm25s is a fast pure-Python BM25 implementation. We prefer it when
    installed/importable because it gives stronger retrieval than the stdlib
    fallback without relying on network calls or external services.
    """

    def __init__(self, docs: list[RagDoc]):
        super().__init__(docs)
        bm25s = try_import("bm25s", "bm25s")
        if bm25s is None:
            raise RuntimeError("bm25s is not importable. Install requirements-rag.txt or keep third_party/bm25s-main with deps.")
        self._bm25s = bm25s
        self._texts = [d.text for d in docs]
        self._tokens = bm25s.tokenize(self._texts)
        self._retriever = bm25s.BM25()
        self._retriever.index(self._tokens)

    def search(self, query: str, k: int = 3) -> list[RagDoc]:
        if not query.strip():
            return []
        q_tokens = self._bm25s.tokenize(query)
        results, scores = self._retriever.retrieve(q_tokens, k=min(k, len(self.docs)))
        out: list[RagDoc] = []
        if getattr(results, "ndim", 1) == 2:
            ids = list(results[0])
            scs = list(scores[0])
        else:
            ids = list(results)
            scs = list(scores)
        for idx, score in zip(ids, scs):
            try:
                i = int(idx)
                if i >= 0 and i < len(self.docs) and float(score) > 0:
                    out.append(self.docs[i])
            except Exception:
                continue
        return out

    @classmethod
    def from_path(cls, path: str | Path | None) -> "BM25sRAG | None":
        base = LexicalRAG.from_path(path)
        if not base:
            return None
        return cls(base.docs)

class RankBM25RAG(LexicalRAG):
    """Optional rank_bm25 adapter with LexicalRAG-compatible API."""

    def __init__(self, docs: list[RagDoc]):
        super().__init__(docs)
        try:
            from rank_bm25 import BM25Okapi  # type: ignore
        except Exception as e:
            raise RuntimeError("rank_bm25 is not installed") from e
        self._bm25 = BM25Okapi(self.doc_terms)

    def search(self, query: str, k: int = 3) -> list[RagDoc]:
        q_terms = tokenize(query)
        if not q_terms:
            return []
        scores = self._bm25.get_scores(q_terms)
        indexed = sorted(enumerate(scores), key=lambda x: float(x[1]), reverse=True)
        return [self.docs[i] for i, s in indexed[:k] if float(s) > 0]

    @classmethod
    def from_path(cls, path: str | Path | None) -> "RankBM25RAG | None":
        base = LexicalRAG.from_path(path)
        if not base:
            return None
        return cls(base.docs)


def create_rag(path: str | Path | None, mode: str = "bm25s") -> LexicalRAG | None:
    """Create offline retriever.

    Strict official mode does not silently downgrade retrieval. The default is
    bm25s because the repo is vendored and wired into the Docker image. Lexical
    fallback is available only when STRICT_NO_FALLBACK=0 for local debugging.
    """
    import os
    if not path:
        return None
    mode = (mode or "bm25s").lower()
    strict = os.getenv("STRICT_NO_FALLBACK", "1").strip().lower() in {"1", "true", "yes", "on"}
    if mode == "auto" and strict:
        mode = "bm25s"
    if mode in {"auto", "bm25s"}:
        try:
            return BM25sRAG.from_path(path)
        except Exception:
            if mode == "bm25s":
                raise
    if mode in {"auto", "rank_bm25", "bm25"}:
        try:
            return RankBM25RAG.from_path(path)
        except Exception:
            if mode in {"rank_bm25", "bm25"}:
                raise
    if strict:
        raise RuntimeError(f"RAG backend {mode!r} is not available; strict mode forbids lexical fallback")
    return LexicalRAG.from_path(path)
