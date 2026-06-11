from __future__ import annotations

"""Fail-fast dependency gate for official submission runs.

This project previously had permissive fallbacks so smoke tests could always
produce a valid pred.csv. For competition runs we want the opposite: if the
real wired stack is not present, fail immediately instead of silently running a
weak heuristic path.
"""

import os
import shutil
from pathlib import Path
from typing import Iterable

from .config import RuntimeConfig
from .third_party_registry import REPO_LAYOUT, add_repo_to_syspath, repo_path, vncorenlp_home


class StrictDependencyError(RuntimeError):
    """Raised when official strict mode is not fully wired."""


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _missing(lines: Iterable[str]) -> StrictDependencyError:
    msg = ["Strict official runtime is not fully wired:"]
    msg.extend(f"- {line}" for line in lines)
    msg.append("Set STRICT_NO_FALLBACK=0 only for local debugging; do not submit that mode.")
    return StrictDependencyError("\n".join(msg))


def _require_import(module_name: str, repo_name: str | None = None) -> None:
    if repo_name:
        add_repo_to_syspath(repo_name)
    try:
        __import__(module_name)
    except Exception as e:
        raise StrictDependencyError(f"required Python module {module_name!r} is not importable") from e


def _is_hf_model_dir(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").exists()


def assert_vendored_repos_present() -> None:
    missing: list[str] = []
    for name in REPO_LAYOUT:
        if repo_path(name) is None:
            missing.append(f"vendored repo missing: third_party/{REPO_LAYOUT[name][0]}")
    if missing:
        raise _missing(missing)


def assert_vncorenlp_present() -> None:
    home = vncorenlp_home()
    missing: list[str] = []
    if not home:
        missing.append("VnCoreNLP jar/models missing under third_party/VnCoreNLP-master")
    if not shutil.which("java"):
        missing.append("Java runtime missing; VnCoreNLP cannot run")
    if missing:
        raise _missing(missing)


def assert_model_backend_ready(cfg: RuntimeConfig) -> None:
    if not cfg.require_model:
        return
    if not cfg.model_path:
        raise _missing(["MODEL_PATH is empty and no model was auto-discovered in /models"])
    model = Path(cfg.model_path)
    if not model.exists():
        raise _missing([f"MODEL_PATH does not exist: {cfg.model_path}"])
    if model.suffix.lower() == ".gguf":
        _require_import("llama_cpp")
        return
    if _is_hf_model_dir(model):
        _require_import("torch")
        _require_import("transformers")
        return
    if model.suffix.lower() in {".safetensors", ".bin"}:
        raise _missing(["MODEL_PATH points to a weight file, but HF backend needs the folder containing config.json"])
    raise _missing([f"Unrecognized model path type: {cfg.model_path}"])


def assert_strict_runtime_ready(cfg: RuntimeConfig) -> None:
    """Validate the official path before reading the full test set.

    The checks deliberately fail when dependencies are missing; no hidden
    fallback/heuristic path is allowed in strict mode.
    """
    if _env_bool("ALLOW_HEURISTIC", False):
        # Local tests may explicitly opt into the heuristic backend. Official
        # Dockerfile sets ALLOW_HEURISTIC=0.
        return
    if cfg.enable_rag:
        _require_import("bm25s", "bm25s")
    if (cfg.backend or "auto").lower() == "heuristic":
        raise _missing(["LLM_BACKEND=heuristic is forbidden in strict official mode"])
    # Check the legal local model before optional knowledge details so missing-model
    # failures are explicit and easy to diagnose.
    assert_model_backend_ready(cfg)
    if cfg.enable_rag:
        use_ke = os.getenv("USE_KNOWLEDGE_ENGINE", "1").lower() in {"1", "true", "yes", "on", "auto"}
        if use_ke:
            from .knowledge_autoseed import official_knowledge_paths
            paths = official_knowledge_paths()
            if not any(Path(x).exists() for x in paths):
                raise _missing(["ENABLE_RAG=1 but no KnowledgeEngine corpus path exists, including auto-seed"])
        else:
            if not cfg.rag_corpus:
                raise _missing(["ENABLE_RAG=1 but RAG_CORPUS is empty"])
            if not Path(cfg.rag_corpus).exists():
                raise _missing([f"RAG_CORPUS does not exist: {cfg.rag_corpus}"])
