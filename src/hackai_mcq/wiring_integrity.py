from __future__ import annotations

"""Runtime wiring integrity guard.

The judge only runs the Docker entrypoint.  This module makes that entrypoint
self-healing for feature wiring: core accuracy/stability flags are kept enabled,
obviously unsafe fast settings are tightened, and a JSON audit report is written
for post-run inspection.  It does not use labels or external data.
"""

import json
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from .config import RuntimeConfig
from .dataset_signal_profiler import profile_dataset
from .schema import MCQItem


@dataclass(slots=True)
class WiringPatch:
    field: str
    old: Any
    new: Any
    reason: str


@dataclass(slots=True)
class WiringReport:
    ok: bool
    official_autorun: bool
    rows: int
    patches: list[WiringPatch] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


CORE_TRUE_FIELDS: tuple[tuple[str, str], ...] = (
    ("enable_rag", "offline/local knowledge and four-repo RAG must be reachable"),
    ("use_token_scoring", "token-choice scoring is the strongest low-cost accuracy path"),
    ("use_pairwise_judge", "pairwise judge repairs close/private-test choices"),
    ("use_verifier", "verifier blocks direct-prompt mistakes"),
    ("use_permutation_check", "permutation check reduces A/B/C/D position bias"),
    ("enable_time_controller", "time controller prevents runaway rows"),
    ("confidence_calibration", "confidence calibration controls early exits"),
    ("use_question_memory", "duplicate/mirrored questions should reuse stable answers"),
    ("use_risk_gate", "risk gate routes traps to deeper reasoning"),
    ("use_option_evidence_matrix", "option evidence matrix increases factual accuracy"),
    ("use_decision_arbitrator", "decision arbitration fuses noisy signals"),
    ("use_retrieval_cache", "retrieval cache stabilizes and speeds repeated queries"),
    ("use_output_watchdog", "output watchdog detects answer-collapse risks"),
    ("use_batch_inference", "batch inference keeps 2000-row private test feasible"),
)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on", "auto"}


def _locked(field: str) -> bool:
    env = "LOCK_" + field.upper()
    return _env_bool(env, False)


def _patch(kwargs: dict[str, Any], report: WiringReport, cfg: RuntimeConfig, field: str, value: Any, reason: str) -> None:
    if _locked(field):
        report.warnings.append(f"{field} locked by env; not patched ({reason})")
        return
    old = getattr(cfg, field)
    if old != value:
        kwargs[field] = value
        report.patches.append(WiringPatch(field, old, value, reason))


def apply_wiring_integrity_guard(cfg: RuntimeConfig, items: list[MCQItem]) -> tuple[RuntimeConfig, WiringReport]:
    official = _env_bool("OFFICIAL_AUTORUN", False)
    report = WiringReport(ok=True, official_autorun=official, rows=len(items))
    kwargs: dict[str, Any] = {}

    if official and not getattr(cfg, "cpu_portable", False):
        for field, reason in CORE_TRUE_FIELDS:
            if not bool(getattr(cfg, field)):
                _patch(kwargs, report, cfg, field, True, reason)
        if (cfg.backend or "").lower() == "heuristic":
            report.errors.append("official strict run cannot use heuristic backend")
        if not cfg.strict_no_fallback:
            _patch(kwargs, report, cfg, "strict_no_fallback", True, "official run must not silently fall back")
        if not cfg.require_model:
            _patch(kwargs, report, cfg, "require_model", True, "official run must use an allowed local model")

    profile = profile_dataset(items)
    high_risk_ratio = (profile.high_risk_rows / max(1, profile.rows)) if profile.rows else 0.0
    # Accuracy-preserving defaults for unknown private tests.  These changes are
    # small enough to keep speed, but they prevent premature fast exits on traps.
    if high_risk_ratio >= 0.20 and cfg.speed_profile in {"turbo", "fast"}:
        _patch(kwargs, report, cfg, "speed_profile", "balanced", f"high-risk dataset ratio={high_risk_ratio:.2%}")
    if high_risk_ratio >= 0.35 and cfg.mode == "direct":
        _patch(kwargs, report, cfg, "mode", "adaptive", f"direct mode unsafe on high-risk dataset ratio={high_risk_ratio:.2%}")
    if cfg.token_fast_exit_margin < 0.32:
        _patch(kwargs, report, cfg, "token_fast_exit_margin", 0.32, "token fast-exit margin too low for private test")
    if cfg.confidence_threshold < 0.58:
        _patch(kwargs, report, cfg, "confidence_threshold", 0.58, "confidence threshold too low for strict routing")
    if cfg.max_retries < 1 and not getattr(cfg, "cpu_portable", False):
        _patch(kwargs, report, cfg, "max_retries", 1, "one same-backend retry improves parse stability")
    if cfg.enable_rag and cfg.rag_k < 2:
        _patch(kwargs, report, cfg, "rag_k", 2, "RAG enabled but k too small for option evidence")
    if cfg.batch_size < 1:
        _patch(kwargs, report, cfg, "batch_size", 1, "invalid batch size")
    if cfg.prompt_hard_limit_chars < 2200:
        _patch(kwargs, report, cfg, "prompt_hard_limit_chars", 2200, "prompt cap too small for context + options")

    report.notes.append(
        f"dataset_profile rows={profile.rows} high_risk={profile.high_risk_rows} "
        f"ratio={high_risk_ratio:.2%} recommended={profile.recommendations.get('speed_profile', 'n/a')}"
    )
    report.ok = not report.errors
    new_cfg = replace(cfg, **kwargs) if kwargs else cfg
    return new_cfg, report


def write_wiring_report(report: WiringReport, path: str | Path | None) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.tmp")
    tmp.write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)


def compact_wiring_line(report: WiringReport) -> str:
    return f"[hackai] wiring ok={int(report.ok)} patches={len(report.patches)} warnings={len(report.warnings)} errors={len(report.errors)} rows={report.rows}"
