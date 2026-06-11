from __future__ import annotations

"""Dataset-level profiler for faster, safer official runs.

It does not inspect labels and does not use any external service.  It only looks
at public input structure (question text/options) to decide how aggressively the
runtime should spend compute.  This makes the one-command Docker entrypoint
adapt to 50-row smoke tests and 2000-row private tests without manual flags.
"""

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .features import estimate_difficulty, has_negation, option_similarity
from .multilingual_nlp_adapter import analyze_multilingual
from .risk_gate import assess_risk
from .schema import MCQItem


@dataclass(slots=True)
class DatasetProfile:
    rows: int
    fingerprint: str
    avg_question_chars: float
    avg_option_chars: float
    avg_difficulty: float
    high_risk_rows: int
    negation_rows: int
    near_duplicate_option_rows: int
    multilingual_rows: int
    duplicate_like_rows: int
    numeric_rows: int
    recommendations: dict[str, object] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def high_risk_rate(self) -> float:
        return self.high_risk_rows / max(1, self.rows)


def _canon_question(item: MCQItem) -> str:
    q = " ".join(item.question.lower().split())
    opts = sorted(" ".join(v.lower().split()) for v in item.options.values())
    return q + "||" + "||".join(opts)


def profile_dataset(items: list[MCQItem]) -> DatasetProfile:
    digest = hashlib.sha256()
    qchars = 0
    ochars = 0
    difficulty_sum = 0.0
    high_risk = neg = near_dup = multilingual = numeric = 0
    seen: set[str] = set()
    dup = 0
    for item in items:
        key = _canon_question(item)
        digest.update(key.encode("utf-8", errors="ignore"))
        if key in seen:
            dup += 1
        seen.add(key)
        qchars += len(item.question)
        ochars += sum(len(v) for v in item.options.values())
        diff = estimate_difficulty(item)
        difficulty_sum += diff
        risk = assess_risk(item)
        if risk.score >= 0.62 or risk.should_deepen:
            high_risk += 1
        if has_negation(item):
            neg += 1
        if option_similarity(item) >= 0.55:
            near_dup += 1
        lang = analyze_multilingual(item.text_for_retrieval())
        if lang.language not in {"vi", "en", "unknown"} or lang.is_mixed_language:
            multilingual += 1
        if any(ch.isdigit() for ch in item.text_for_retrieval()):
            numeric += 1
    rows = len(items)
    p = DatasetProfile(
        rows=rows,
        fingerprint=digest.hexdigest()[:16],
        avg_question_chars=round(qchars / max(1, rows), 2),
        avg_option_chars=round(ochars / max(1, rows), 2),
        avg_difficulty=round(difficulty_sum / max(1, rows), 4),
        high_risk_rows=high_risk,
        negation_rows=neg,
        near_duplicate_option_rows=near_dup,
        multilingual_rows=multilingual,
        duplicate_like_rows=dup,
        numeric_rows=numeric,
    )
    p.recommendations = recommend_runtime_knobs(p)
    if rows >= 1800 and p.high_risk_rate > 0.55:
        p.warnings.append("large dataset with many high-risk rows; runtime should avoid brute-force on every row")
    if p.avg_question_chars > 900:
        p.warnings.append("long-question dataset; prompt compaction should stay enabled")
    if p.multilingual_rows:
        p.warnings.append(f"multilingual/mixed rows detected: {p.multilingual_rows}")
    return p


def recommend_runtime_knobs(profile: DatasetProfile) -> dict[str, object]:
    """Return safe one-command defaults.

    The optimizer is conservative: it keeps strong modules on, but reduces the
    amount of expensive deep solving when the private file is large or easy.
    """
    rows = profile.rows
    risk = profile.high_risk_rate
    avg_diff = profile.avg_difficulty
    rec: dict[str, object] = {}
    if rows >= 1800:
        if risk < 0.32 and avg_diff < 0.48:
            rec["speed_profile"] = "fast"
            rec["rag_k"] = 2
            rec["max_pairwise_calls"] = 1
        elif risk > 0.62 or avg_diff > 0.62:
            rec["speed_profile"] = "balanced"
            rec["rag_k"] = 3
            rec["max_pairwise_calls"] = 2
        else:
            rec["speed_profile"] = "balanced"
            rec["rag_k"] = 2
            rec["max_pairwise_calls"] = 2
        rec["checkpoint_every"] = 250
        rec["log_every"] = 100
    elif rows <= 100:
        # Tiny public/smoke sets can afford stronger checks and help catch bad models early.
        rec["speed_profile"] = "accuracy" if risk >= 0.35 else "balanced"
        rec["rag_k"] = 3
        rec["max_pairwise_calls"] = 3
        rec["log_every"] = max(10, rows)
    else:
        rec["speed_profile"] = "balanced"
        rec["rag_k"] = 3 if risk >= 0.45 else 2
        rec["max_pairwise_calls"] = 2
        rec["checkpoint_every"] = 250
        rec["log_every"] = 50
    if profile.avg_question_chars > 900:
        rec["prompt_hard_limit_chars"] = 3600
        rec["rag_max_chars"] = 1200
    elif profile.high_risk_rate > 0.55:
        rec["prompt_hard_limit_chars"] = 4500
        rec["rag_max_chars"] = 1800
    else:
        rec["prompt_hard_limit_chars"] = 4000
        rec["rag_max_chars"] = 1400
    return rec


def write_dataset_profile(profile: DatasetProfile, path: str | Path | None) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(profile) | {"timestamp": time.time()}
    tmp = p.with_name(f".{p.name}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)
