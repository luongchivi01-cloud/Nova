from __future__ import annotations

"""Final deterministic arbitration layer for high-stability MCQ solving.

The LLM remains the official answer source, but private-test MCQ rows often
produce conflicting signals: token scoring, direct prompting, evidence prior,
permutation checks and judge prompts may disagree. This module fuses those
signals in a deterministic, inspectable way so the solver does not overreact to
one noisy prompt or one noisy evidence snippet.
"""

from dataclasses import dataclass, field
from math import exp, isfinite
from typing import Mapping

from .features import has_negation, option_similarity
from .schema import MCQItem, VALID_ANSWERS


@dataclass(slots=True)
class ArbitrationResult:
    answer: str | None
    confidence: float
    scores: dict[str, float] = field(default_factory=dict)
    margin: float = 0.0
    should_deepen: bool = False
    notes: str = ""


BASE_WEIGHTS: dict[str, float] = {
    "token_score": 1.45,
    "direct": 1.00,
    "elimination": 1.05,
    "scoring": 1.10,
    "negation_guard": 1.35,
    "multilingual_sanity": 0.95,
    "knowledge_prior": 0.70,
    "evidence_consensus": 0.65,
    "option_evidence_matrix": 0.85,
    "verifier": 1.35,
    "judge": 1.75,
    "memory": 2.00,
}


def _normalize_map(scores: Mapping[str, float] | None) -> dict[str, float]:
    raw = {k: float(v) for k, v in (scores or {}).items() if k in VALID_ANSWERS and isfinite(float(v))}
    if not raw:
        return {}
    vals = list(raw.values())
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return {k: 0.25 for k in raw}
    scaled = {k: (v - lo) / (hi - lo) for k, v in raw.items()}
    total = sum(scaled.values()) or 1.0
    return {k: v / total for k, v in scaled.items()}


def _vote_weight(source: str, item: MCQItem, risk_score: float = 0.0) -> float:
    src = source.lower()
    weight = BASE_WEIGHTS.get(src, 0.82)
    if src.startswith("pair_"):
        weight = 0.95
    if src.startswith("perm_"):
        weight = 0.80
    if has_negation(item) and src in {"negation_guard", "verifier", "judge"}:
        weight += 0.25
    if risk_score >= 0.65 and src in {"direct", "token_score"}:
        weight *= 0.82
    if risk_score >= 0.65 and src in {"verifier", "judge", "negation_guard"}:
        weight *= 1.12
    return weight


def arbitrate_decision(
    item: MCQItem,
    votes: Mapping[str, str],
    score_map: Mapping[str, float] | None = None,
    *,
    token_margin: float = 0.0,
    risk_score: float = 0.0,
    time_pressure: bool = False,
    permutation_consistent: bool | None = None,
) -> ArbitrationResult:
    fused = {k: 0.0 for k in "ABCD"}
    vote_count = 0
    for source, ans in votes.items():
        ans = (ans or "").strip().upper()[:1]
        if ans in VALID_ANSWERS:
            fused[ans] += _vote_weight(source, item, risk_score)
            vote_count += 1

    normalized = _normalize_map(score_map)
    for ans, val in normalized.items():
        fused[ans] += 1.05 * val

    trap_penalty = 0.0
    if option_similarity(item) >= 0.56:
        trap_penalty += 0.08
    if has_negation(item):
        trap_penalty += 0.07
    if risk_score >= 0.65:
        trap_penalty += 0.06
    if permutation_consistent is False:
        trap_penalty += 0.12
    elif permutation_consistent is True:
        trap_penalty -= 0.04
    if time_pressure:
        trap_penalty -= 0.03

    ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    answer = ranked[0][0] if ranked and ranked[0][1] > 0 else None
    top = ranked[0][1] if ranked else 0.0
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = max(0.0, top - second)
    base = 1.0 / (1.0 + exp(-(margin + max(0.0, token_margin) - 0.35)))
    agreement_bonus = min(0.18, 0.035 * max(0, vote_count - 1))
    confidence = max(0.02, min(0.99, base + agreement_bonus - trap_penalty))
    should_deepen = bool(answer and not time_pressure and (confidence < 0.58 or margin < 0.24 or permutation_consistent is False))
    notes = (
        f"arb_answer={answer};arb_margin={margin:.3f};arb_conf={confidence:.3f};"
        f"votes={vote_count};risk={risk_score:.2f};perm={permutation_consistent};trap_penalty={trap_penalty:.2f}"
    )
    return ArbitrationResult(answer=answer, confidence=confidence, scores=fused, margin=margin, should_deepen=should_deepen, notes=notes)
