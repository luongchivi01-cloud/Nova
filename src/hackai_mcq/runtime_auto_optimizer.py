from __future__ import annotations

"""Automatic runtime optimization for one-command official judging.

The judge should only run the container.  This module tightens runtime knobs from
input-file signals so the same image is fast on 2000 rows and still strong on
hard/small sets.  It never disables official safety or switches to heuristic.
"""

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .config import RuntimeConfig
from .dataset_signal_profiler import DatasetProfile, profile_dataset, write_dataset_profile
from .model_backends import detect_vram_gb
from .schema import MCQItem


@dataclass(slots=True)
class AutoOptimizeReport:
    enabled: bool
    dataset_fingerprint: str
    vram_gb: float | None
    changed: dict[str, Any]
    preserved_explicit: list[str]
    notes: list[str]


def _explicit(name: str) -> bool:
    # OFFICIAL_AUTORUN defaults are allowed to be refined, but local/user-supplied
    # env vars outside autorun gates must be respected. Hard locks work in both modes.
    if os.getenv(f"LOCK_{name}", "0").strip().lower() in {"1", "true", "yes", "on"}:
        return True
    autorun = os.getenv("OFFICIAL_AUTORUN", "0").strip().lower() in {"1", "true", "yes", "on", "auto"}
    return (not autorun) and (name in os.environ)


def optimize_runtime_config(cfg: RuntimeConfig, items: list[MCQItem], profile_path: str | Path | None = None) -> tuple[RuntimeConfig, AutoOptimizeReport, DatasetProfile]:
    enabled = os.getenv("AUTO_RUNTIME_OPTIMIZER", "1").strip().lower() in {"1", "true", "yes", "on", "auto"}
    profile = profile_dataset(items)
    write_dataset_profile(profile, profile_path or os.getenv("DATASET_PROFILE_PATH", "/output/dataset_profile.json"))
    if not enabled:
        return cfg, AutoOptimizeReport(False, profile.fingerprint, detect_vram_gb(), {}, [], ["AUTO_RUNTIME_OPTIMIZER disabled"]), profile

    rec = dict(profile.recommendations)
    changed: dict[str, Any] = {}
    preserved: list[str] = []
    kwargs: dict[str, Any] = {}

    def set_if_unlocked(field: str, env_name: str, value: Any) -> None:
        nonlocal kwargs
        if value is None:
            return
        if _explicit(env_name):
            preserved.append(env_name)
            return
        old = getattr(cfg, field)
        if old != value:
            kwargs[field] = value
            changed[field] = {"from": old, "to": value}

    set_if_unlocked("speed_profile", "SPEED_PROFILE", rec.get("speed_profile"))
    set_if_unlocked("rag_k", "RAG_K", int(rec.get("rag_k", cfg.rag_k)))
    set_if_unlocked("rag_max_chars", "RAG_MAX_CHARS", int(rec.get("rag_max_chars", cfg.rag_max_chars)))
    set_if_unlocked("max_pairwise_calls", "MAX_PAIRWISE_CALLS", int(rec.get("max_pairwise_calls", cfg.max_pairwise_calls)))
    set_if_unlocked("prompt_hard_limit_chars", "PROMPT_HARD_LIMIT_CHARS", int(rec.get("prompt_hard_limit_chars", cfg.prompt_hard_limit_chars)))
    set_if_unlocked("checkpoint_every", "CHECKPOINT_EVERY", int(rec.get("checkpoint_every", cfg.checkpoint_every)))
    set_if_unlocked("log_every", "LOG_EVERY", int(rec.get("log_every", cfg.log_every)))

    vram = detect_vram_gb()
    notes: list[str] = []
    if vram is not None and vram < 7:
        # Safer 6GB profile: still strong, but avoids huge prompts that cause OOM.
        if not _explicit("N_CTX") and cfg.n_ctx > 1536:
            kwargs["n_ctx"] = 1536
            changed["n_ctx"] = {"from": cfg.n_ctx, "to": 1536}
        if not _explicit("LOAD_IN_4BIT") and not cfg.load_in_4bit and not str(cfg.model_path or "").lower().endswith(".gguf"):
            kwargs["load_in_4bit"] = True
            changed["load_in_4bit"] = {"from": cfg.load_in_4bit, "to": True}
        notes.append("6GB-class VRAM detected; tightened context/4bit knobs")
    elif vram is not None and vram >= 12:
        notes.append("larger VRAM detected; keeping balanced/accuracy-capable defaults")

    new_cfg = replace(cfg, **kwargs) if kwargs else cfg
    if changed:
        notes.append("runtime knobs optimized from dataset profile")
    else:
        notes.append("runtime knobs already suitable")
    return new_cfg, AutoOptimizeReport(True, profile.fingerprint, vram, changed, preserved, notes), profile


def write_auto_optimize_report(report: AutoOptimizeReport, path: str | Path | None) -> None:
    if not path:
        return
    import json
    from dataclasses import asdict
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.tmp")
    tmp.write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)
