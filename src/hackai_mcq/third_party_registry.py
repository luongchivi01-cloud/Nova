from __future__ import annotations

"""Local third-party repository registry.

The user supplied full OSS repos are vendored under ``third_party/``. This module lets official code use them safely. The Dockerfile installs the
runtime dependencies by default, while `strict_dependencies.py` fails fast if
required official pieces are missing.

- no network calls
- no browser automation
- no silent weak backend fallback in official strict mode
- imports are activated only when a feature asks for them
"""

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_LAYOUT = {
    "bm25s": ("bm25s-main", "bm25s"),
    "dspy": ("dspy-main", "dspy"),
    "outlines": ("outlines-main", "outlines"),
    "lm_eval": ("lm-evaluation-harness-main", "lm_eval"),
    "vncorenlp": ("VnCoreNLP-master", None),
    # Offline knowledge/search stack. These repos are vendored so the
    # official Docker image does not need the grader to download anything.
    "pyserini": ("pyserini-master", "pyserini"),
    "haystack": ("haystack-main", "haystack"),
    "kiwix_tools": ("kiwix-tools-main", None),
    "wikipedia_mirror": ("wikipedia-mirror-master", None),
    # V9 full RAG-agent fusion pack supplied by the user. The official runtime
    # activates their ideas through stable adapters, but never requires their
    # heavy optional dependency stacks to import during /data -> /output runs.
    "flashrag": ("FlashRAG-main", "flashrag"),
    "txtai": ("txtai-master/src/python", "txtai"),
    "graphrag": ("graphrag-main/packages/graphrag", "graphrag"),
    "lightrag": ("LightRAG-main", "lightrag"),
}


def project_root() -> Path:
    # src/hackai_mcq/third_party_registry.py -> project root
    return Path(__file__).resolve().parents[2]


def vendor_root() -> Path:
    return Path(os.getenv("HACKAI_THIRD_PARTY", str(project_root() / "third_party"))).resolve()


def repo_path(name: str) -> Path | None:
    meta = REPO_LAYOUT.get(name)
    if not meta:
        return None
    p = vendor_root() / meta[0]
    return p if p.exists() else None


def add_repo_to_syspath(name: str) -> Path | None:
    p = repo_path(name)
    if not p:
        return None
    ps = str(p)
    if ps not in sys.path:
        sys.path.insert(0, ps)
    return p


def try_import(name: str, package: str | None = None):
    """Try importing from installed packages or the vendored repo.

    Returns the module object or None. Exceptions from missing optional
    transitive dependencies are swallowed so official runs stay robust.
    """
    package = package or name
    try:
        return __import__(package)
    except Exception:
        pass
    add_repo_to_syspath(name)
    try:
        return __import__(package)
    except Exception:
        return None


@dataclass(slots=True)
class ThirdPartyStatus:
    name: str
    repo_present: bool
    importable: bool
    path: str | None
    notes: str = ""


def detect_third_party() -> list[ThirdPartyStatus]:
    rows: list[ThirdPartyStatus] = []
    for name, (_, package) in REPO_LAYOUT.items():
        p = repo_path(name)
        importable = False
        notes = ""
        if name == "vncorenlp":
            jar = (p / "VnCoreNLP-1.2.jar") if p else None
            models = (p / "models") if p else None
            importable = bool(jar and jar.exists() and models and models.exists() and shutil.which("java"))
            notes = "java+jar+models" if importable else "requires Java 1.8+ for runtime segmentation"
        else:
            if package is None:
                importable = bool(p)
                notes = "vendored source present; used by adapters/tools, no Python import required"
            else:
                # Avoid importing heavy experiment frameworks during official inference.
                # bm25s is light and may be used by the runtime. FlashRAG/txtai/
                # GraphRAG/LightRAG are merged and activated through lightweight
                # adapters in vendor_rag_fusion.py, so official runs do not need
                # to import their optional dependency stacks.
                probe_heavy = os.getenv("PROBE_HEAVY_THIRD_PARTY", "0").strip().lower() in {"1", "true", "yes", "on"}
                light_runtime = name in {"bm25s", "pyserini"}
                adapter_runtime = name in {"flashrag", "txtai", "graphrag", "lightrag"}
                if light_runtime or (probe_heavy and not adapter_runtime):
                    importable = try_import(name, package) is not None
                    notes = "python import ok" if importable else "repo present; optional deps may be missing"
                elif adapter_runtime:
                    importable = bool(p)
                    notes = "vendored; auto-activated through stable V9 adapter, heavy import skipped"
                else:
                    importable = False
                    notes = "vendored for offline experiments; not imported during official inference"
        rows.append(ThirdPartyStatus(name=name, repo_present=p is not None, importable=importable, path=str(p) if p else None, notes=notes))
    return rows


def format_inventory(rows: Iterable[ThirdPartyStatus] | None = None) -> str:
    rows = list(rows or detect_third_party())
    lines = ["name,repo_present,importable,path,notes"]
    for r in rows:
        lines.append(f"{r.name},{int(r.repo_present)},{int(r.importable)},{r.path or ''},{r.notes}")
    return "\n".join(lines)


def vncorenlp_home() -> Path | None:
    p = repo_path("vncorenlp")
    if p and (p / "VnCoreNLP-1.2.jar").exists() and (p / "models").exists():
        return p
    return None
