from __future__ import annotations

"""Optional integrations from vendored/public OSS repos.

Important: all integrations are offline and optional. The official submission
path must still work when none of these imports succeed.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .third_party_registry import detect_third_party, try_import, vncorenlp_home


def has_package(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


@dataclass(slots=True)
class OptionalSupport:
    llama_cpp_python: bool
    transformers: bool
    flagembedding: bool
    sentence_transformers: bool
    rank_bm25: bool
    bm25s: bool
    dspy: bool
    outlines: bool
    lm_eval: bool
    vncorenlp: bool
    lm_format_enforcer: bool
    flashrag: bool
    txtai: bool
    graphrag: bool
    lightrag: bool
    third_party_summary: str


def detect_optional_support() -> OptionalSupport:
    statuses = detect_third_party()
    by_name = {s.name: s for s in statuses}
    return OptionalSupport(
        llama_cpp_python=has_package("llama_cpp"),
        transformers=has_package("transformers"),
        flagembedding=has_package("FlagEmbedding"),
        sentence_transformers=has_package("sentence_transformers"),
        rank_bm25=has_package("rank_bm25"),
        bm25s=try_import("bm25s", "bm25s") is not None,
        dspy=try_import("dspy", "dspy") is not None,
        outlines=try_import("outlines", "outlines") is not None,
        lm_eval=try_import("lm_eval", "lm_eval") is not None,
        vncorenlp=vncorenlp_home() is not None,
        lm_format_enforcer=has_package("lmformatenforcer"),
        flashrag=try_import("flashrag", "flashrag") is not None,
        txtai=try_import("txtai", "txtai") is not None,
        graphrag=try_import("graphrag", "graphrag") is not None,
        lightrag=try_import("lightrag", "lightrag") is not None,
        third_party_summary="; ".join(f"{s.name}:repo={int(s.repo_present)},import={int(s.importable)}" for s in statuses),
    )


class FlagEmbeddingReranker:
    """Thin adapter for BGE rerankers from FlagEmbedding.

    Use only when a local reranker model is mounted. It never downloads by
    default, keeping official runs reproducible/offline.
    """

    def __init__(self, model_path: str, use_fp16: bool = True):
        try:
            from FlagEmbedding import FlagReranker  # type: ignore
        except Exception as e:
            raise RuntimeError("FlagEmbedding is not installed. Install optional requirements-rag.txt.") from e
        if not Path(model_path).exists():
            raise FileNotFoundError(model_path)
        self.reranker = FlagReranker(model_path, use_fp16=use_fp16)

    def rerank(self, query: str, docs: list[str], top_k: int = 3) -> list[str]:
        if not docs:
            return []
        pairs = [[query, d] for d in docs]
        scores = self.reranker.compute_score(pairs)
        ranked = sorted(zip(scores, docs), key=lambda x: float(x[0]), reverse=True)
        return [d for _, d in ranked[:top_k]]


class SentenceTransformerEmbedder:
    """Local embedding adapter for mounted BGE-m3/SentenceTransformer models."""

    def __init__(self, model_path: str):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as e:
            raise RuntimeError("sentence-transformers is not installed. Install optional requirements-rag.txt.") from e
        if not Path(model_path).exists():
            raise FileNotFoundError(model_path)
        self.model = SentenceTransformer(model_path)

    def encode(self, texts: Iterable[str]):
        return self.model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
