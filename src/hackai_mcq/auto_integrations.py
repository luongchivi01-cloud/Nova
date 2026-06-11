from __future__ import annotations

"""One-pass automatic integration layer for official Docker runs.

Goal: the grader should only run the container. Nobody has to know which
vendored repo to enable. This module:
- exposes vendored repos on sys.path
- auto-discovers optional corpora/models
- selects safe default runtime features
- never uses network/browser automation
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .third_party_registry import add_repo_to_syspath, detect_third_party, repo_path


@dataclass(slots=True)
class AutoIntegrationState:
    enabled: bool
    python_repos_on_path: list[str]
    importable: dict[str, bool]
    discovered_corpus: str | None
    vncorenlp_ready: bool
    notes: list[str]


PY_REPOS = ("bm25s", "dspy", "outlines", "lm_eval", "pyserini", "haystack", "flashrag", "txtai", "graphrag", "lightrag")
DEFAULT_CORPUS_CANDIDATES = (
    "/knowledge",
    "/corpus",
    "/data/corpus",
    "/data/knowledge",
    "/data/docs",
    "/data/zim",
    "./knowledge",
    "./corpus",
    "./data/corpus",
    "./data/knowledge",
    "./data/docs",
    "./data/zim",
)


def _bool_env(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on", "auto"}


def _has_text_files(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_file():
        return path.suffix.lower() in {".txt", ".md", ".rst", ".html", ".htm", ".csv", ".jsonl", ".json", ".zim"}
    for p in path.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".txt", ".md", ".rst", ".html", ".htm", ".csv", ".jsonl", ".json", ".zim"}:
            return True
    return False


def discover_corpus(candidates: Iterable[str] = DEFAULT_CORPUS_CANDIDATES) -> str | None:
    # When caller supplies custom candidates (unit tests / explicit routing), do
    # not let a previous environment variable override them. In official runs
    # the default candidate tuple is used, so RAG_CORPUS remains honored.
    if candidates is DEFAULT_CORPUS_CANDIDATES:
        explicit = os.getenv("RAG_CORPUS")
        if explicit and _has_text_files(Path(explicit)):
            return explicit
    for raw in candidates:
        p = Path(raw)
        if _has_text_files(p):
            return str(p)
    return None


def prepare_auto_integrations() -> AutoIntegrationState:
    enabled = _bool_env("AUTO_INTEGRATIONS", True)
    notes: list[str] = []
    added: list[str] = []
    if enabled:
        for name in PY_REPOS:
            p = add_repo_to_syspath(name)
            if p:
                added.append(name)
        # Keep official path offline unless the user explicitly overrides.
        os.environ.setdefault("ENABLE_NETWORK", "0")
        os.environ.setdefault("SUBMISSION_STRICT", "1")
        os.environ.setdefault("RAG_BACKEND", "auto")
        os.environ.setdefault("USE_KNOWLEDGE_ENGINE", "1")
        os.environ.setdefault("KNOWLEDGE_BACKEND", os.environ.get("RAG_BACKEND", "auto"))
        os.environ.setdefault("ENABLE_VENDOR_RAG_FUSION", "0" if _bool_env("CPU_PORTABLE", False) else "1")
        os.environ.setdefault("VENDOR_RAG_BACKENDS", "flashrag,txtai,graphrag,lightrag")
        # Do not require the operator to set ENABLE_RAG when a corpus exists.
        corpus = discover_corpus()
        if corpus:
            os.environ.setdefault("RAG_CORPUS", corpus)
            os.environ.setdefault("KNOWLEDGE_PATHS", corpus)
            os.environ.setdefault("ENABLE_RAG", "1")
            notes.append(f"auto_knowledge_corpus={corpus}")
        else:
            os.environ.setdefault("ENABLE_RAG", "0")
            notes.append("auto_rag_disabled_no_corpus")
        # VnCoreNLP is useful for preprocessing but optional. Auto means try only
        # when Java+jar+models exist; fallback is pure Python Vietnamese signals.
        os.environ.setdefault("USE_VNCORENLP", "auto")
    rows = detect_third_party()
    importable = {r.name: bool(r.importable) for r in rows}
    vnc = next((r for r in rows if r.name == "vncorenlp"), None)
    return AutoIntegrationState(
        enabled=enabled,
        python_repos_on_path=added,
        importable=importable,
        discovered_corpus=os.getenv("RAG_CORPUS") or None,
        vncorenlp_ready=bool(vnc and vnc.importable),
        notes=notes,
    )


def compact_status_line(state: AutoIntegrationState) -> str:
    bits = [f"auto={int(state.enabled)}"]
    if state.python_repos_on_path:
        bits.append("path=" + "+".join(state.python_repos_on_path))
    ready = [k for k, v in state.importable.items() if v]
    if ready:
        bits.append("ready=" + "+".join(ready))
    if state.discovered_corpus:
        bits.append("corpus=" + state.discovered_corpus)
    if state.notes:
        bits.append("notes=" + ";".join(state.notes))
    return "[hackai] integrations " + " ".join(bits)
