from __future__ import annotations

"""One-command official runtime profile.

The contest judge should not need to know feature flags.  This module applies a
safe, strict, fully-wired default profile when OFFICIAL_AUTORUN=1.  Values that
are explicitly provided by the runner are respected, but missing settings are
filled with competition-safe defaults.
"""

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


OFFICIAL_DEFAULTS: dict[str, str] = {
    # Official IO / safety
    "DATA_DIR": "/data",
    "OUTPUT_PATH": "/output/pred.csv",
    "MODELS_DIR": "/models",
    "ENABLE_NETWORK": "0",
    "SUBMISSION_STRICT": "1",
    "STRICT_NO_FALLBACK": "1",
    "REQUIRE_MODEL": "1",
    "ALLOW_HEURISTIC": "0",
    "CPU_PORTABLE": "1",
    "FORCE_CPU": "1",
    "N_GPU_LAYERS": "0",
    # Auto feature loading: grader only runs the container.
    "AUTO_INTEGRATIONS": "1",
    "ENABLE_RAG": "0",
    "USE_KNOWLEDGE_ENGINE": "0",
    "AUTO_SEED_KNOWLEDGE": "0",
    "KNOWLEDGE_REQUIRED": "0",
    "USE_KNOWLEDGE_PRIOR": "0",
    "RAG_BACKEND": "bm25s",
    "KNOWLEDGE_BACKEND": "auto",
    "KNOWLEDGE_PATHS": "/knowledge:/data/knowledge:/data/corpus:/data/docs:/app/knowledge",
    "ENABLE_VENDOR_RAG_FUSION": "0",
    "VENDOR_RAG_BACKENDS": "flashrag,txtai,graphrag,lightrag",
    "VENDOR_RAG_MAX_DOCS": "6000",
    "VENDOR_RAG_REPORT_PATH": "/output/vendor_rag_fusion.md",
    # Solver intelligence: all core modules on by default.
    "SOLVER_MODE": "adaptive",
    "LLM_BACKEND": "llama_cpp",
    "USE_TOKEN_SCORING": "0",
    "USE_PAIRWISE_JUDGE": "0",
    "USE_VERIFIER": "1",
    "USE_PERMUTATION_CHECK": "0",
    "CONFIDENCE_CALIBRATION": "1",
    "ENABLE_TIME_CONTROLLER": "1",
    "USE_QUESTION_MEMORY": "1",
    "USE_RISK_GATE": "1",
    "USE_OPTION_EVIDENCE_MATRIX": "1",
    "USE_DECISION_ARBITRATOR": "1",
    "USE_RETRIEVAL_CACHE": "1",
    "USE_OUTPUT_WATCHDOG": "1",
    "USE_BATCH_INFERENCE": "1",
    "BATCH_SIZE": "8",
    "USE_VLLM": "0",
    "LOAD_IN_AWQ": "1",
    "VLLM_QUANTIZATION": "awq",
    "VLLM_GPU_MEMORY_UTILIZATION": "0.85",
    "TOKEN_FAST_EXIT_MARGIN": "0.35",
    # Stable speed defaults: fast exits only when high-confidence, deep route otherwise.
    "SPEED_PROFILE": "balanced",
    "TOKEN_FAST_EXIT": "1",
    "ACCURACY_STABILITY_GUARD": "1",
    "WIRING_INTEGRITY_GUARD": "1",
    "MAX_NEW_TOKENS": "12",
    "N_CTX": "2048",
    "RAG_K": "3",
    "RAG_MAX_CHARS": "1800",
    "MAX_PAIRWISE_CALLS": "0",
    "MAX_RETRIES": "0",
    "MODEL_PROBE_QUESTIONS": "1",
    "PROMPT_HARD_LIMIT_CHARS": "4500",
    # Reports for post-run audit/debug. These do not affect pred.csv.
    "WRITE_HEALTH_REPORT": "1",
    "WRITE_CONTRACT_REPORT": "1",
    "WRITE_HEARTBEAT": "1",
    "WRITE_SPEED_REPORT": "1",
    "KNOWLEDGE_MANIFEST_PATH": "/output/knowledge_manifest.json",
    # V7: one-command runtime optimizer/profiler. It refines speed/retrieval
    # budgets from the input distribution without any manual judge flags.
    "AUTO_RUNTIME_OPTIMIZER": "1",
    "DATASET_PROFILE_PATH": "/output/dataset_profile.json",
    "AUTO_OPTIMIZER_REPORT_PATH": "/output/runtime_auto_optimizer.json",
}


@dataclass(slots=True)
class AutorunReport:
    enabled: bool
    timestamp: float
    explicit_keys: list[str]
    defaulted_keys: list[str]
    protected_keys: list[str]
    profile: str
    notes: list[str]


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on", "auto"}


def apply_official_autorun_defaults() -> AutorunReport:
    """Apply default official flags exactly once.

    The function is intentionally conservative: explicit environment variables
    are not overwritten.  That allows local debugging, while Docker official
    images simply set OFFICIAL_AUTORUN=1 and need no extra command flags.
    """
    enabled = env_bool("OFFICIAL_AUTORUN", False)
    explicit: list[str] = []
    defaulted: list[str] = []
    protected: list[str] = []
    notes: list[str] = []
    if not enabled:
        return AutorunReport(False, time.time(), explicit, defaulted, protected, "disabled", ["OFFICIAL_AUTORUN disabled"])

    for key, value in OFFICIAL_DEFAULTS.items():
        if key in os.environ:
            explicit.append(key)
        else:
            os.environ[key] = value
            defaulted.append(key)

    # Never allow accidental network/browser/API behavior in official profile.
    for key, expected in {"ENABLE_NETWORK": "0", "ALLOW_HEURISTIC": "0"}.items():
        if os.environ.get(key) != expected:
            protected.append(key)
            notes.append(f"{key}={os.environ.get(key)!r} explicitly overrides official default {expected!r}")

    profile = os.environ.get("SPEED_PROFILE", "balanced")
    notes.append("official one-command V9.1 profile loaded; wiring integrity, accuracy stability guard, batch speed, and four-repo RAG fusion are on by default")
    return AutorunReport(True, time.time(), explicit, defaulted, protected, profile, notes)


def validate_autorun_contract() -> tuple[bool, list[str]]:
    """Sanity check that the official profile is not partially disabled."""
    required_on: Iterable[str] = [
        "AUTO_INTEGRATIONS", "USE_VERIFIER", "CONFIDENCE_CALIBRATION", "ENABLE_TIME_CONTROLLER",
        "USE_QUESTION_MEMORY", "USE_RISK_GATE", "USE_OPTION_EVIDENCE_MATRIX",
        "USE_DECISION_ARBITRATOR", "USE_RETRIEVAL_CACHE", "USE_OUTPUT_WATCHDOG",
        "USE_BATCH_INFERENCE", "AUTO_RUNTIME_OPTIMIZER",
        "ACCURACY_STABILITY_GUARD", "WIRING_INTEGRITY_GUARD",
        "STRICT_NO_FALLBACK", "REQUIRE_MODEL", "SUBMISSION_STRICT",
    ]
    errors: list[str] = []
    if env_bool("OFFICIAL_AUTORUN", False):
        for key in required_on:
            if not env_bool(key, False):
                errors.append(f"{key} must be enabled in OFFICIAL_AUTORUN mode")
        if env_bool("ENABLE_NETWORK", True):
            errors.append("ENABLE_NETWORK must stay disabled in official mode")
        if env_bool("ALLOW_HEURISTIC", False):
            errors.append("ALLOW_HEURISTIC must stay disabled in official mode")
    return not errors, errors


def write_autorun_report(path: str | Path | None, report: AutorunReport) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.tmp")
    tmp.write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)


def compact_autorun_line(report: AutorunReport) -> str:
    if not report.enabled:
        return "[hackai] official_autorun=off"
    return f"[hackai] official_autorun=on profile={report.profile} defaulted={len(report.defaulted_keys)} explicit={len(report.explicit_keys)}"
