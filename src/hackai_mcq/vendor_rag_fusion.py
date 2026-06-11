from __future__ import annotations

"""V9 stable adapters for the four user-supplied RAG repositories.

The full repos are vendored under third_party/:
- FlashRAG-main
- txtai-master
- graphrag-main
- LightRAG-main

Official HackAIthon Docker runs must be one-command and robust, so this module
activates the *useful retrieval ideas* from those repos through lightweight,
offline, dependency-safe adapters.  It does not import their heavyweight runtime
stacks by default, avoiding crashes from optional dependencies while still
wiring the merged repos into the scoring pipeline.
"""

import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .corpus_builder import CorpusDoc
from .evidence_compressor import EvidenceSnippet
from .features import tokenize
from .third_party_registry import repo_path


OPTION_WORDS = {"a", "b", "c", "d"}
STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "is", "are", "was", "were",
    "la", "là", "của", "và", "hoặc", "trong", "với", "cho", "một", "những", "các", "nào", "đâu",
}


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on", "auto"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _keywords(text: str) -> list[str]:
    toks = [t for t in tokenize(text) if len(t) > 1 and t not in STOP and t not in OPTION_WORDS]
    # Keep order but remove duplicates.
    seen: set[str] = set()
    out: list[str] = []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


@dataclass(slots=True)
class _IndexedDoc:
    idx: int
    doc: CorpusDoc
    terms: list[str]
    term_set: set[str]
    entities: set[str]


class _BaseVendorBackend:
    name = "vendor_base"

    def __init__(self, docs: list[CorpusDoc], max_docs: int | None = None):
        limit = max_docs or _env_int("VENDOR_RAG_MAX_DOCS", 6000)
        self.docs = docs[: max(1, limit)]
        self.indexed: list[_IndexedDoc] = []
        self.inverted: dict[str, list[int]] = defaultdict(list)
        self.df: dict[str, int] = defaultdict(int)
        for i, d in enumerate(self.docs):
            terms = _keywords(d.text)
            term_set = set(terms)
            ents = self._extract_entities(d.text)
            row = _IndexedDoc(i, d, terms, term_set, ents)
            self.indexed.append(row)
            for t in term_set:
                self.inverted[t].append(i)
                self.df[t] += 1
        self.n = max(1, len(self.indexed))

    @staticmethod
    def _extract_entities(text: str) -> set[str]:
        # Simple multilingual entity/salient phrase extractor: numbers, acronyms,
        # formula-like strings and title-case chunks. This mirrors GraphRAG/LightRAG
        # value without requiring external NER during official runs.
        out: set[str] = set()
        for m in re.finditer(r"\b[A-ZĐ][A-Za-zÀ-ỹ0-9_.+-]{1,}\b", text or ""):
            out.add(m.group(0).lower())
        for m in re.finditer(r"\b\d+(?:[.,]\d+)?%?\b", text or ""):
            out.add(m.group(0).lower())
        for m in re.finditer(r"\b[A-Z][A-Za-z]?\d+(?:[A-Za-z0-9]*)\b", text or ""):
            out.add(m.group(0).lower())
        return out

    def _candidate_ids(self, query: str, cap: int = 1600) -> list[int]:
        q = _keywords(query)
        if not q:
            return list(range(min(len(self.indexed), cap)))
        counts: dict[int, int] = defaultdict(int)
        for t in q:
            for i in self.inverted.get(t, [])[:cap]:
                counts[i] += 1
        if not counts:
            return list(range(min(len(self.indexed), min(cap, 128))))
        ranked = sorted(counts, key=lambda i: (counts[i], -i), reverse=True)
        return ranked[:cap]

    def _bm25_score(self, q_terms: list[str], row: _IndexedDoc) -> float:
        if not q_terms or not row.terms:
            return 0.0
        tf: dict[str, int] = defaultdict(int)
        qset = set(q_terms)
        for t in row.terms:
            if t in qset:
                tf[t] += 1
        if not tf:
            return 0.0
        score = 0.0
        dl = max(1, len(row.terms))
        avgdl = 64.0
        k1 = 1.25
        b = 0.72
        for t, f in tf.items():
            idf = math.log(1.0 + (self.n - self.df.get(t, 0) + 0.5) / (self.df.get(t, 0) + 0.5))
            score += idf * (f * (k1 + 1.0)) / (f + k1 * (1.0 - b + b * dl / avgdl))
        return score

    def _snippet(self, row: _IndexedDoc, score: float, source_suffix: str) -> EvidenceSnippet:
        src = f"{self.name}:{source_suffix}:{row.doc.source}"
        return EvidenceSnippet(row.doc.doc_id, row.doc.text, float(score), src)


class FlashRAGFusionBackend(_BaseVendorBackend):
    """Multi-query reciprocal-rank fusion inspired by FlashRAG pipelines."""

    name = "flashrag_fusion"

    def _query_variants(self, query: str) -> list[str]:
        kws = _keywords(query)
        variants = [query]
        if kws:
            variants.append(" ".join(kws[:16]))
            variants.append(" ".join(kws[-16:]))
        # Add a compact MCQ-aware variant that strips option labels and filler.
        stripped = re.sub(r"\b[ABCD][).:]", " ", query or "", flags=re.I)
        stripped = re.sub(r"\b(câu|question|choose|chọn|đáp án|answer)\b", " ", stripped, flags=re.I)
        stripped = _norm(stripped)
        if stripped and stripped not in variants:
            variants.append(stripped)
        return variants[:4]

    def search(self, query: str, k: int = 5) -> list[EvidenceSnippet]:
        fused: dict[int, float] = defaultdict(float)
        for qv in self._query_variants(query):
            q_terms = _keywords(qv)
            scored: list[tuple[float, int]] = []
            for i in self._candidate_ids(qv):
                row = self.indexed[i]
                s = self._bm25_score(q_terms, row)
                if s > 0:
                    scored.append((s, i))
            scored.sort(reverse=True)
            for rank, (score, i) in enumerate(scored[: max(k * 6, k, 10)], start=1):
                fused[i] += 1.0 / (40.0 + rank) + min(2.0, score) * 0.04
        ranked = sorted(((s, i) for i, s in fused.items()), reverse=True)[:k]
        return [self._snippet(self.indexed[i], s, "rrf") for s, i in ranked if s > 0]


class TxtaiHybridBackend(_BaseVendorBackend):
    """Small txtai-like hybrid scoring: sparse BM25 + char ngram overlap."""

    name = "txtai_hybrid"

    @staticmethod
    def _chargrams(text: str, n: int = 3) -> set[str]:
        text = re.sub(r"\s+", " ", (text or "").lower())
        if len(text) < n:
            return {text} if text else set()
        return {text[i : i + n] for i in range(0, len(text) - n + 1)}

    def search(self, query: str, k: int = 5) -> list[EvidenceSnippet]:
        q_terms = _keywords(query)
        qg = self._chargrams(query)
        scored: list[tuple[float, int]] = []
        for i in self._candidate_ids(query, cap=1800):
            row = self.indexed[i]
            sparse = self._bm25_score(q_terms, row)
            if sparse <= 0 and not qg:
                continue
            dg = self._chargrams(row.doc.text[:600])
            denseish = (len(qg & dg) / max(1, len(qg))) if dg else 0.0
            score = sparse + 0.85 * denseish
            if score > 0:
                scored.append((score, i))
        scored.sort(reverse=True)
        return [self._snippet(self.indexed[i], s, "hybrid") for s, i in scored[:k]]


class GraphRAGLocalBackend(_BaseVendorBackend):
    """Dependency-safe GraphRAG adapter using entity co-occurrence expansion."""

    name = "graphrag_local"

    def __init__(self, docs: list[CorpusDoc], max_docs: int | None = None):
        super().__init__(docs, max_docs=max_docs)
        self.entity_docs: dict[str, set[int]] = defaultdict(set)
        for row in self.indexed:
            for e in row.entities:
                self.entity_docs[e].add(row.idx)

    def search(self, query: str, k: int = 5) -> list[EvidenceSnippet]:
        q_terms = _keywords(query)
        q_entities = self._extract_entities(query) | set(q_terms[:8])
        candidates: dict[int, float] = defaultdict(float)
        for e in q_entities:
            for i in self.entity_docs.get(e, set()):
                candidates[i] += 1.25
        for i in self._candidate_ids(query, cap=1200)[: max(64, k * 20)]:
            candidates[i] += 0.35
        scored: list[tuple[float, int]] = []
        for i, bonus in candidates.items():
            row = self.indexed[i]
            overlap = len(q_entities & row.entities)
            score = self._bm25_score(q_terms, row) + bonus + 0.5 * overlap
            if score > 0:
                scored.append((score, i))
        scored.sort(reverse=True)
        return [self._snippet(self.indexed[i], s, "entity") for s, i in scored[:k]]


class LightRAGFastBackend(_BaseVendorBackend):
    """Fast LightRAG-style local/global retrieval with bounded latency."""

    name = "lightrag_fast"

    def search(self, query: str, k: int = 5) -> list[EvidenceSnippet]:
        q_terms = _keywords(query)
        candidates = self._candidate_ids(query, cap=max(128, min(900, k * 180)))
        scored: list[tuple[float, int]] = []
        for rank, i in enumerate(candidates, start=1):
            row = self.indexed[i]
            local = self._bm25_score(q_terms, row)
            # A tiny global prior favors concise, information-dense notes over long noisy blocks.
            density = len(set(q_terms) & row.term_set) / max(1, min(len(set(q_terms)), 12))
            brevity = min(1.0, 900.0 / max(120.0, len(row.doc.text)))
            score = local + 0.42 * density + 0.08 * brevity + 1.0 / (250.0 + rank)
            if score > 0:
                scored.append((score, i))
        scored.sort(reverse=True)
        return [self._snippet(self.indexed[i], s, "local_global") for s, i in scored[:k]]


VENDOR_BACKENDS = {
    "flashrag": ("flashrag", FlashRAGFusionBackend),
    "txtai": ("txtai", TxtaiHybridBackend),
    "graphrag": ("graphrag", GraphRAGLocalBackend),
    "lightrag": ("lightrag", LightRAGFastBackend),
}


def vendor_status() -> dict[str, bool]:
    return {name: repo_path(registry_name) is not None for name, (registry_name, _) in VENDOR_BACKENDS.items()}


def build_vendor_rag_backends(docs: list[CorpusDoc]) -> list[_BaseVendorBackend]:
    if not _env_bool("ENABLE_VENDOR_RAG_FUSION", True):
        return []
    raw = os.getenv("VENDOR_RAG_BACKENDS", "flashrag,txtai,graphrag,lightrag")
    wanted = [x.strip().lower() for x in raw.replace(";", ",").split(",") if x.strip()]
    out: list[_BaseVendorBackend] = []
    for name in wanted:
        meta = VENDOR_BACKENDS.get(name)
        if not meta:
            continue
        registry_name, cls = meta
        if repo_path(registry_name) is None:
            continue
        try:
            out.append(cls(docs))
        except Exception:
            if _env_bool("REQUIRE_VENDOR_RAG_FUSION", False):
                raise
            continue
    return out


def write_vendor_rag_report(path: str | Path, backends: Iterable[object] | None = None) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    status = vendor_status()
    active = [getattr(b, "name", type(b).__name__) for b in (backends or [])]
    lines = ["# V9 vendor RAG fusion report", "", "## Vendored repo presence"]
    for name, ok in status.items():
        lines.append(f"- {name}: {ok}")
    lines.extend(["", "## Active backends"])
    for name in active:
        lines.append(f"- {name}")
    if not active:
        lines.append("- none")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p
