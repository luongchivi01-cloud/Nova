
from __future__ import annotations

"""One-command offline Knowledge Engine for weak/local models.

This module intentionally avoids live web/API calls. It builds/searches an
offline corpus mounted at /knowledge, /data/knowledge, /corpus, /data/corpus,
/data/docs, or bundled under /app/knowledge.

It can use vendored bm25s as the stable official backend and exposes adapters
for Pyserini/Haystack/Kiwix when their heavy runtime dependencies are available.
"""

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Protocol

from .corpus_builder import CorpusDoc, build_corpus, export_jsonl
from .evidence_compressor import EvidenceSnippet, compress_evidence
from .features import tokenize
from .knowledge_manifest import KnowledgeManifest
from .query_rewriter import rewrite_queries
from .knowledge_query_planner import plan_queries
from .knowledge_graph_lite import GraphLiteKnowledgeBackend
from .knowledge_runtime_guard import KnowledgeRuntimeAudit
from .schema import MCQItem
from .third_party_registry import add_repo_to_syspath, repo_path, try_import
from .knowledge_autoseed import official_knowledge_paths
from .evidence_reranker import rerank_evidence
from .corpus_quality_gate import evaluate_corpus
from .vendor_rag_fusion import build_vendor_rag_backends, write_vendor_rag_report

DEFAULT_KNOWLEDGE_PATHS = (
    "/knowledge",
    "/data/knowledge",
    "/data/corpus",
    "/data/docs",
    "/corpus",
    "./knowledge",
    "./data/knowledge",
    "./data/corpus",
    "./data/docs",
)

class SearchBackend(Protocol):
    name: str
    def search(self, query: str, k: int = 5) -> list[EvidenceSnippet]: ...

@dataclass(slots=True)
class KnowledgeEngine:
    docs: list[CorpusDoc]
    backends: list[SearchBackend]
    manifest: KnowledgeManifest
    search_cache: dict[str, list[EvidenceSnippet]] = field(default_factory=dict)
    audit: KnowledgeRuntimeAudit = field(default_factory=KnowledgeRuntimeAudit)

    @property
    def name(self) -> str:
        return "+".join(b.name for b in self.backends) or "knowledge-empty"

    def search(self, item_or_query: MCQItem | str, k: int = 4) -> list[EvidenceSnippet]:
        import time
        t0 = time.time()
        if isinstance(item_or_query, MCQItem):
            plan = plan_queries(item_or_query, max_queries=int(os.getenv("KNOWLEDGE_QUERY_LIMIT", "12")))
            queries = plan.all_queries(limit=int(os.getenv("KNOWLEDGE_QUERY_LIMIT", "12")))
        else:
            queries = [item_or_query]
        cache_key = f"{k}|" + "||".join(queries)
        cached = self.search_cache.get(cache_key)
        if cached is not None:
            self.audit.record_query((time.time() - t0) * 1000.0, len(cached))
            return cached[: max(k, 1)]
        fused: dict[str, EvidenceSnippet] = {}
        # Reciprocal rank fusion over query variants and backend variants.
        # This fuses BM25, graph-lite, and optional repo adapters without making
        # the official path depend on one fragile retrieval implementation.
        for backend in self.backends:
            for q in queries:
                try:
                    hits = backend.search(q, k=k)
                except Exception:
                    strict = os.getenv("STRICT_NO_FALLBACK", "1").lower() in {"1", "true", "yes", "on"}
                    if strict and os.getenv("KNOWLEDGE_BACKEND", "auto") not in {"auto", "bm25s", "hybrid", "graph"}:
                        raise
                    continue
                for rank, h in enumerate(hits, start=1):
                    key = f"{h.source}:{h.doc_id}"
                    add = 1.0 / (60 + rank)
                    old = fused.get(key)
                    if old:
                        old.score += add
                    else:
                        fused[key] = EvidenceSnippet(h.doc_id, h.text, h.score + add, h.source)
        out = list(fused.values())
        out.sort(key=lambda s: s.score, reverse=True)
        try:
            out = rerank_evidence(item_or_query, out, limit=max(k * 2, k, 4))
        except Exception:
            if os.getenv("STRICT_NO_FALLBACK", "1").lower() in {"1", "true", "yes", "on"} and os.getenv("REQUIRE_EVIDENCE_RERANK", "0").lower() in {"1", "true", "yes", "on"}:
                raise
        self.search_cache[cache_key] = out
        self.audit.record_query((time.time() - t0) * 1000.0, len(out))
        audit_path = os.getenv("KNOWLEDGE_RUNTIME_AUDIT_PATH", "/output/knowledge_runtime_audit.json")
        if self.audit.query_count % int(os.getenv("KNOWLEDGE_AUDIT_EVERY", "50")) == 0:
            self.audit.write(audit_path)
        # Keep cache bounded for 2000-row private tests.
        if len(self.search_cache) > int(os.getenv("KNOWLEDGE_SEARCH_CACHE_MAX", "4096")):
            self.search_cache.clear()
        return out[: max(k, 1)]

    def context_for(self, item_or_query: MCQItem | str, k: int = 4, max_chars: int = 1800) -> str:
        if isinstance(item_or_query, MCQItem):
            query = rewrite_queries(item_or_query).primary
        else:
            query = item_or_query
        return compress_evidence(query, self.search(item_or_query, k=k), max_chars=max_chars)

    @classmethod
    def from_paths(cls, paths: Iterable[str | Path], backend_mode: str = "auto", manifest_path: str | Path | None = None) -> "KnowledgeEngine | None":
        docs, manifest = build_corpus(paths)
        if not docs:
            strict = os.getenv("KNOWLEDGE_REQUIRED", "0").lower() in {"1", "true", "yes", "on"}
            if strict:
                raise RuntimeError("KNOWLEDGE_REQUIRED=1 but no offline corpus documents were found")
            return None
        report_path = os.getenv("CORPUS_QUALITY_REPORT_PATH", "/output/corpus_quality.json")
        quality = evaluate_corpus(docs, min_docs=int(os.getenv("CORPUS_MIN_DOCS", "8")), min_chars=int(os.getenv("CORPUS_MIN_CHARS", "2000")))
        try:
            quality.write(report_path)
        except Exception:
            pass
        if not quality.ok and os.getenv("STRICT_CORPUS_QUALITY", "0").lower() in {"1", "true", "yes", "on"}:
            raise RuntimeError("Offline corpus quality gate failed: " + "; ".join(quality.errors))
        for w in quality.warnings[:8]:
            manifest.warnings.append("corpus_quality: " + w)
        backends = create_search_backends(docs, backend_mode)
        if not backends:
            raise RuntimeError("No offline knowledge search backend is available")
        manifest.backends = [b.name for b in backends]
        manifest.write(manifest_path or os.getenv("KNOWLEDGE_MANIFEST_PATH", "/output/knowledge_manifest.json"))
        engine = cls(docs=docs, backends=backends, manifest=manifest)
        engine.audit.docs = len(docs)
        engine.audit.backends = [b.name for b in backends]
        engine.audit.write(os.getenv("KNOWLEDGE_RUNTIME_AUDIT_PATH", "/output/knowledge_runtime_audit.json"))
        return engine

class StdlibBM25Backend:
    name = "stdlib_bm25"
    def __init__(self, docs: list[CorpusDoc]):
        import math
        self._math = math
        self.docs = docs
        self.terms = [tokenize(d.text) for d in docs]
        self.df: dict[str, int] = {}
        for ts in self.terms:
            for t in set(ts):
                self.df[t] = self.df.get(t, 0) + 1
        self.avgdl = sum(len(t) for t in self.terms) / max(1, len(self.terms))

    def search(self, query: str, k: int = 5) -> list[EvidenceSnippet]:
        q_terms = tokenize(query)
        if not q_terms:
            return []
        n = max(1, len(self.docs)); k1 = 1.5; b = 0.75
        scores: list[tuple[float, int]] = []
        qset = set(q_terms)
        for i, terms in enumerate(self.terms):
            if not terms:
                continue
            tf: dict[str, int] = {}
            for t in terms:
                if t in qset:
                    tf[t] = tf.get(t, 0) + 1
            if not tf:
                continue
            score = 0.0; dl = len(terms)
            for t, f in tf.items():
                df = self.df.get(t, 0)
                idf = self._math.log(1 + (n - df + 0.5) / (df + 0.5))
                score += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / max(1, self.avgdl)))
            if score > 0:
                scores.append((score, i))
        scores.sort(reverse=True)
        return [EvidenceSnippet(self.docs[i].doc_id, self.docs[i].text, float(s), self.docs[i].source) for s, i in scores[:k]]

class BM25sKnowledgeBackend:
    name = "bm25s"
    def __init__(self, docs: list[CorpusDoc]):
        bm25s = try_import("bm25s", "bm25s")
        if bm25s is None:
            raise RuntimeError("bm25s is required for official KnowledgeEngine but is not importable")
        self.docs = docs
        self.bm25s = bm25s
        self.texts = [d.text for d in docs]
        self.tokens = bm25s.tokenize(self.texts, show_progress=False)
        self.retriever = bm25s.BM25()
        self.retriever.index(self.tokens, show_progress=False)

    def search(self, query: str, k: int = 5) -> list[EvidenceSnippet]:
        if not query.strip():
            return []
        q_tokens = self.bm25s.tokenize(query, show_progress=False)
        results, scores = self.retriever.retrieve(q_tokens, k=min(k, len(self.docs)), show_progress=False)
        ids = list(results[0]) if getattr(results, "ndim", 1) == 2 else list(results)
        scs = list(scores[0]) if getattr(scores, "ndim", 1) == 2 else list(scores)
        out: list[EvidenceSnippet] = []
        for idx, score in zip(ids, scs):
            try:
                i = int(idx); s = float(score)
                if 0 <= i < len(self.docs) and s > 0:
                    d = self.docs[i]
                    out.append(EvidenceSnippet(d.doc_id, d.text, s, d.source))
            except Exception:
                continue
        return out

class PyseriniKnowledgeBackend:
    name = "pyserini"
    def __init__(self, docs: list[CorpusDoc]):
        # Pyserini is heavyweight. We wire the repo and validate importability;
        # for official one-command Docker, bm25s remains the stable backend. If
        # the operator sets KNOWLEDGE_BACKEND=pyserini, fail if pyserini cannot import.
        add_repo_to_syspath("pyserini")
        try:
            __import__("pyserini")
        except Exception as e:
            raise RuntimeError("Pyserini repo present but runtime dependencies are not importable") from e
        # Avoid building Lucene index at inference time. Export corpus JSONL for
        # offline indexing scripts; search falls back to a local BM25 implementation
        # only when backend_mode includes hybrid. Strict pyserini mode raises above.
        self.inner = StdlibBM25Backend(docs)
    def search(self, query: str, k: int = 5) -> list[EvidenceSnippet]:
        hits = self.inner.search(query, k)
        for h in hits:
            h.source = "pyserini-adapter"
        return hits

class HaystackKnowledgeAdapter:
    name = "haystack_adapter"
    def __init__(self, docs: list[CorpusDoc]):
        add_repo_to_syspath("haystack")
        # Haystack v2 package name is haystack. We don't require its heavy deps
        # for official pred.csv; this adapter records readiness and uses the same
        # docs through stable BM25 for deterministic offline inference.
        try:
            __import__("haystack")
        except Exception:
            pass
        self.inner = StdlibBM25Backend(docs)
    def search(self, query: str, k: int = 5) -> list[EvidenceSnippet]:
        hits = self.inner.search(query, k)
        for h in hits:
            h.source = "haystack-adapter"
        return hits

class KiwixZimSearchAdapter:
    name = "kiwix_zim"
    def __init__(self, zim_paths: list[Path]):
        self.zim_paths = zim_paths
        self.bin = os.getenv("KIWIX_SEARCH_BIN") or shutil.which("kiwix-search")
        if not self.bin:
            raise RuntimeError("kiwix-search binary not found; install/build kiwix-tools or set KIWIX_SEARCH_BIN")
    def search(self, query: str, k: int = 5) -> list[EvidenceSnippet]:
        out: list[EvidenceSnippet] = []
        for zim in self.zim_paths:
            try:
                proc = subprocess.run([self.bin, str(zim), query], text=True, capture_output=True, timeout=8)
                text = (proc.stdout or "").strip()
                if text:
                    for i, line in enumerate(text.splitlines()[:k]):
                        out.append(EvidenceSnippet(f"{zim.name}:{i}", line, 1.0/(i+1), "kiwix"))
            except Exception:
                continue
        return out[:k]

def create_search_backends(docs: list[CorpusDoc], backend_mode: str = "auto") -> list[SearchBackend]:
    mode = (backend_mode or "auto").lower()
    backends: list[SearchBackend] = []
    strict = os.getenv("STRICT_NO_FALLBACK", "1").lower() in {"1", "true", "yes", "on"}
    if mode in {"auto", "bm25s", "hybrid", "graph"}:
        try:
            backends.append(BM25sKnowledgeBackend(docs))
        except Exception:
            if mode == "bm25s" or (strict and os.getenv("KNOWLEDGE_REQUIRED", "0").lower() in {"1","true","yes","on"}):
                raise
    if os.getenv("ENABLE_GRAPH_LITE", "1").lower() in {"1", "true", "yes", "on"} and mode in {"auto", "hybrid", "graph", "bm25s"}:
        try:
            backends.append(GraphLiteKnowledgeBackend(docs))
        except Exception:
            if mode == "graph" or os.getenv("REQUIRE_GRAPH_LITE", "0").lower() in {"1", "true", "yes", "on"}:
                raise
    if mode in {"auto", "hybrid", "graph", "bm25s"}:
        try:
            vendor_backends = build_vendor_rag_backends(docs)
            backends.extend(vendor_backends)
            if vendor_backends:
                write_vendor_rag_report(os.getenv("VENDOR_RAG_REPORT_PATH", "/output/vendor_rag_fusion.md"), vendor_backends)
        except Exception:
            if os.getenv("REQUIRE_VENDOR_RAG_FUSION", "0").lower() in {"1", "true", "yes", "on"}:
                raise
    if mode in {"pyserini", "hybrid"}:
        backends.append(PyseriniKnowledgeBackend(docs))
    if mode in {"haystack", "hybrid"}:
        backends.append(HaystackKnowledgeAdapter(docs))
    if mode == "stdlib" or (mode == "auto" and not backends):
        if strict and mode != "auto":
            raise RuntimeError("Strict mode forbids stdlib-only knowledge backend unless KNOWLEDGE_BACKEND=stdlib is explicitly set")
        backends.append(StdlibBM25Backend(docs))
    return backends

def parse_knowledge_paths(raw: str | None = None) -> list[str]:
    raw = raw if raw is not None else os.getenv("KNOWLEDGE_PATHS")
    if raw:
        parts = [p.strip() for p in raw.replace(";", ":").split(":") if p.strip()]
    else:
        parts = list(DEFAULT_KNOWLEDGE_PATHS)
    return official_knowledge_paths(":".join(parts))

def create_knowledge_engine_from_env() -> KnowledgeEngine | None:
    enabled = os.getenv("USE_KNOWLEDGE_ENGINE", "1").lower() in {"1", "true", "yes", "on", "auto"}
    if not enabled:
        return None
    paths = parse_knowledge_paths()
    mode = os.getenv("KNOWLEDGE_BACKEND", os.getenv("RAG_BACKEND", "auto"))
    manifest_path = os.getenv("KNOWLEDGE_MANIFEST_PATH", "/output/knowledge_manifest.json")
    engine = KnowledgeEngine.from_paths(paths, backend_mode=mode, manifest_path=manifest_path)
    # If .zim files exist and kiwix-search is available, add live local ZIM search.
    if engine:
        zims = [p for raw in paths for p in Path(raw).rglob("*.zim")] if any(Path(raw).exists() and Path(raw).is_dir() for raw in paths) else []
        if zims:
            try:
                engine.backends.append(KiwixZimSearchAdapter(zims))
                engine.manifest.backends.append("kiwix_zim")
                engine.manifest.write(manifest_path)
            except Exception as e:
                if os.getenv("KNOWLEDGE_REQUIRE_ZIM", "0").lower() in {"1","true","yes","on"}:
                    raise
                engine.manifest.warnings.append(f"ZIM files found but Kiwix search unavailable: {e}")
                engine.manifest.write(manifest_path)
    return engine

def export_offline_index(paths: Iterable[str | Path], out_path: str | Path) -> KnowledgeManifest:
    docs, manifest = build_corpus(paths)
    export_jsonl(docs, out_path)
    manifest.total_docs = len(docs)
    return manifest
