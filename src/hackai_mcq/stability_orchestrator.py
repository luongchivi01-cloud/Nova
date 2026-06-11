from __future__ import annotations

"""Tight wiring and stability layer for official competition runs.

This module is intentionally boring and defensive.  It makes sure the strong
solver features are connected before the first private-test row is processed,
records a machine-readable health report, and keeps official runs deterministic.
"""

import hashlib
import importlib
import json
import os
import random
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .config import RuntimeConfig
from .schema import MCQItem, VALID_ANSWERS


@dataclass(slots=True)
class FeatureStatus:
    name: str
    enabled: bool
    required: bool
    ok: bool
    detail: str = ""


@dataclass(slots=True)
class StabilityReport:
    run_id: str
    timestamp: float
    input_path: str
    output_path: str
    row_count: int
    backend: str
    mode: str
    model_path: str | None
    strict_no_fallback: bool
    require_model: bool
    features: list[FeatureStatus] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    config_digest: str = ""

    @property
    def ok(self) -> bool:
        return not self.errors and all((not f.required) or f.ok for f in self.features)


def configure_determinism(seed: int) -> None:
    """Set deterministic knobs for Python, NumPy and torch when available."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    try:
        import numpy as np  # type: ignore
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        if hasattr(torch, "cuda"):
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def _digest_config(cfg: RuntimeConfig) -> str:
    payload = {
        "backend": cfg.backend,
        "mode": cfg.mode,
        "model_path": cfg.model_path,
        "max_new_tokens": cfg.max_new_tokens,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "seed": cfg.seed,
        "enable_rag": cfg.enable_rag,
        "rag_backend": cfg.rag_backend,
        "use_token_scoring": cfg.use_token_scoring,
        "use_pairwise_judge": cfg.use_pairwise_judge,
        "use_verifier": cfg.use_verifier,
        "use_permutation_check": cfg.use_permutation_check,
        "confidence_calibration": cfg.confidence_calibration,
        "enable_time_controller": cfg.enable_time_controller,
        "strict_no_fallback": cfg.strict_no_fallback,
        "require_model": cfg.require_model,
        "use_question_memory": getattr(cfg, "use_question_memory", True),
        "use_risk_gate": getattr(cfg, "use_risk_gate", True),
        "use_option_evidence_matrix": getattr(cfg, "use_option_evidence_matrix", True),
        "use_decision_arbitrator": getattr(cfg, "use_decision_arbitrator", True),
        "use_retrieval_cache": getattr(cfg, "use_retrieval_cache", True),
        "use_output_watchdog": getattr(cfg, "use_output_watchdog", True),
        "speed_profile": getattr(cfg, "speed_profile", "balanced"),
        "token_fast_exit": getattr(cfg, "token_fast_exit", True),
        "use_batch_inference": getattr(cfg, "use_batch_inference", True),
        "batch_size": getattr(cfg, "batch_size", 1),
        "use_vllm": getattr(cfg, "use_vllm", False),
        "load_in_awq": getattr(cfg, "load_in_awq", False),
        "prompt_hard_limit_chars": getattr(cfg, "prompt_hard_limit_chars", 4500),
        "official_autorun": os.getenv("OFFICIAL_AUTORUN", "0"),
        "auto_runtime_optimizer": os.getenv("AUTO_RUNTIME_OPTIMIZER", "1"),
        "accuracy_stability_guard": os.getenv("ACCURACY_STABILITY_GUARD", "1"),
        "wiring_integrity_guard": os.getenv("WIRING_INTEGRITY_GUARD", "1"),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _module_ok(module_name: str) -> tuple[bool, str]:
    try:
        importlib.import_module(module_name)
        return True, "import ok"
    except Exception as e:
        return False, f"import failed: {type(e).__name__}: {e}"


def _feature(report: StabilityReport, name: str, enabled: bool, required: bool, module: str | None = None, ok: bool | None = None, detail: str = "") -> None:
    if not enabled:
        report.features.append(FeatureStatus(name=name, enabled=False, required=required, ok=not required, detail="disabled"))
        return
    if ok is None and module:
        ok, detail = _module_ok(module)
    elif ok is None:
        ok = True
    report.features.append(FeatureStatus(name=name, enabled=True, required=required, ok=bool(ok), detail=detail))
    if required and not ok:
        report.errors.append(f"required feature not ready: {name} ({detail})")


def _output_writable(path: Path) -> tuple[bool, str]:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(path.parent), prefix=".write_probe_", delete=False) as f:
            probe = Path(f.name)
            f.write("ok")
        probe.unlink(missing_ok=True)
        return True, "output directory writable"
    except Exception as e:
        return False, f"output directory not writable: {type(e).__name__}: {e}"


def _inspect_items(items: Iterable[MCQItem]) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    errors: list[str] = []
    seen: set[str] = set()
    count = 0
    missing_any_option = 0
    empty_questions = 0
    for item in items:
        count += 1
        if not str(item.qid).strip():
            errors.append("input contains empty qid")
        if item.qid in seen:
            errors.append(f"input contains duplicated qid: {item.qid}")
        seen.add(item.qid)
        if not item.question.strip():
            empty_questions += 1
        missing = [k for k in "ABCD" if not item.options.get(k, "").strip()]
        if missing:
            missing_any_option += 1
    if count == 0:
        errors.append("input has zero rows")
    if empty_questions:
        warnings.append(f"{empty_questions} rows have empty/weak question text")
    if missing_any_option:
        warnings.append(f"{missing_any_option} rows have at least one empty option after normalization")
    return warnings, errors


def build_stability_report(cfg: RuntimeConfig, items: list[MCQItem], integration_state: Any | None = None) -> StabilityReport:
    run_id = hashlib.sha1(f"{time.time()}:{cfg.input_path}:{cfg.output_path}:{len(items)}".encode("utf-8")).hexdigest()[:12]
    report = StabilityReport(
        run_id=run_id,
        timestamp=time.time(),
        input_path=str(cfg.input_path),
        output_path=str(cfg.output_path),
        row_count=len(items),
        backend=cfg.backend,
        mode=cfg.mode,
        model_path=cfg.model_path,
        strict_no_fallback=cfg.strict_no_fallback,
        require_model=cfg.require_model,
        config_digest=_digest_config(cfg),
    )

    item_warnings, item_errors = _inspect_items(items)
    report.warnings.extend(item_warnings)
    report.errors.extend(item_errors)

    ok, detail = _output_writable(cfg.output_path)
    _feature(report, "output_atomic_writable", True, True, ok=ok, detail=detail)

    # Core modules must always be connected.
    for name, module in [
        ("io_contract", "hackai_mcq.io_utils"),
        ("answer_parser", "hackai_mcq.answer_parser"),
        ("adaptive_solver", "hackai_mcq.solver"),
        ("token_choice_scorer", "hackai_mcq.token_choice_scorer"),
        ("weighted_ensembling", "hackai_mcq.ensembling"),
        ("confidence_calibrator", "hackai_mcq.confidence_calibrator"),
        ("time_budget_controller", "hackai_mcq.time_budget_controller"),
        ("option_permutation", "hackai_mcq.option_permutation"),
        ("multilingual_adapter", "hackai_mcq.multilingual_nlp_adapter"),
        ("advanced_error_taxonomy", "hackai_mcq.advanced_error_taxonomy"),
        ("official_contract", "hackai_mcq.official_contract"),
        ("runtime_supervisor", "hackai_mcq.runtime_supervisor"),
        ("same_model_answer_repair", "hackai_mcq.answer_repair"),
        ("knowledge_engine", "hackai_mcq.knowledge_engine"),
        ("query_rewriter", "hackai_mcq.query_rewriter"),
        ("evidence_compressor", "hackai_mcq.evidence_compressor"),
        ("corpus_builder", "hackai_mcq.corpus_builder"),
        ("question_memory", "hackai_mcq.question_memory"),
        ("risk_gate", "hackai_mcq.risk_gate"),
        ("option_evidence_matrix", "hackai_mcq.option_evidence_matrix"),
        ("knowledge_gap_analyzer", "hackai_mcq.knowledge_gap_analyzer"),
        ("runtime_invariant_guard", "hackai_mcq.runtime_invariant_guard"),
        ("resource_guard", "hackai_mcq.resource_guard"),
        ("result_ledger", "hackai_mcq.result_ledger"),
        ("decision_arbitrator", "hackai_mcq.decision_arbitrator"),
        ("retrieval_cache", "hackai_mcq.retrieval_cache"),
        ("preflight_stress_probe", "hackai_mcq.preflight_stress_probe"),
        ("model_output_watchdog", "hackai_mcq.model_output_watchdog"),
        ("speed_profile_planner", "hackai_mcq.speed_profile"),
        ("prompt_compactor", "hackai_mcq.prompt_compactor"),
        ("speed_report", "hackai_mcq.speed_report"),
        ("official_autorun_guard", "hackai_mcq.official_autorun"),
        ("dataset_signal_profiler", "hackai_mcq.dataset_signal_profiler"),
        ("runtime_auto_optimizer", "hackai_mcq.runtime_auto_optimizer"),
        ("answer_quality_gate", "hackai_mcq.answer_quality_gate"),
        ("batch_inference_path", "hackai_mcq.cli_batch_patch"),
        ("vendor_rag_fusion", "hackai_mcq.vendor_rag_fusion"),
        ("accuracy_stability_guard", "hackai_mcq.accuracy_stability_guard"),
        ("wiring_integrity_guard", "hackai_mcq.wiring_integrity"),
    ]:
        _feature(report, name, True, True, module=module)

    _feature(report, "batch_inference_enabled", getattr(cfg, "use_batch_inference", True), True, ok=bool(getattr(cfg, "batch_size", 1) >= 1), detail=f"batch_size={getattr(cfg, 'batch_size', 1)}")
    _feature(report, "rag_bm25s", cfg.enable_rag, cfg.enable_rag, module="bm25s")
    _feature(report, "offline_knowledge_stack", cfg.enable_rag, cfg.enable_rag, module="hackai_mcq.knowledge_engine")
    _feature(report, "four_repo_rag_fusion", cfg.enable_rag, False, module="hackai_mcq.vendor_rag_fusion")
    _feature(report, "no_heuristic_official", True, cfg.strict_no_fallback, ok=not (cfg.strict_no_fallback and (cfg.backend or "").lower() == "heuristic"), detail="heuristic disabled in strict path")
    if cfg.require_model:
        mp = Path(cfg.model_path) if cfg.model_path else None
        _feature(report, "model_path_present", True, True, ok=bool(mp and mp.exists()), detail=str(mp) if mp else "MODEL_PATH empty")
    if integration_state is not None:
        detail = "; ".join(getattr(integration_state, "notes", []) or []) or "auto integrations prepared"
        _feature(report, "auto_integrations", bool(getattr(integration_state, "enabled", False)), False, ok=True, detail=detail)

    return report


def write_stability_report(report: StabilityReport, path: str | Path | None) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(report)
    tmp = p.with_name(f".{p.name}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)


def compact_report_line(report: StabilityReport) -> str:
    required = [f for f in report.features if f.required]
    ok_count = sum(1 for f in required if f.ok)
    warn = len(report.warnings)
    err = len(report.errors)
    return f"[hackai] stability run={report.run_id} required={ok_count}/{len(required)} warnings={warn} errors={err} cfg={report.config_digest}"
