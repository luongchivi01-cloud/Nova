from __future__ import annotations

"""Accuracy/stability guard for private-test MCQ routing.

This guard is the thin layer between fast paths and deep reasoning.  It does
not invent answers, call external services, or use ground-truth labels.  It only
checks whether the current signals are strong enough to safely return early; if
not, the normal verifier/vote/judge pipeline continues.
"""

from dataclasses import dataclass, field
from math import isfinite
from typing import Mapping

from .features import has_negation, option_similarity
from .schema import MCQItem, VALID_ANSWERS


@dataclass(slots=True)
class GuardDecision:
    allow: bool
    reason: str
    required_margin: float
    agreement: int
    risk_score: float
    difficulty: float
    vote_margin: float
    token_margin: float
    notes: list[str] = field(default_factory=list)

    @property
    def note(self) -> str:
        bits = [
            f"guard={'allow' if self.allow else 'deepen'}",
            f"reason={self.reason}",
            f"req={self.required_margin:.3f}",
            f"agree={self.agreement}",
            f"risk={self.risk_score:.3f}",
            f"diff={self.difficulty:.3f}",
            f"tm={self.token_margin:.3f}",
            f"vm={self.vote_margin:.3f}",
        ]
        bits.extend(self.notes[:4])
        return ";".join(bits)


def _clean_scores(score_map: Mapping[str, float] | None) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in (score_map or {}).items():
        key = str(k).strip().upper()[:1]
        try:
            fv = float(v)
        except Exception:
            continue
        if key in VALID_ANSWERS and isfinite(fv):
            out[key] = fv
    return out


def _agreement_for(answer: str | None, votes: Mapping[str, str] | None, score_map: Mapping[str, float] | None) -> int:
    ans = (answer or "").strip().upper()[:1]
    if ans not in VALID_ANSWERS:
        return 0
    agreement = 0
    seen_sources: set[str] = set()
    for source, vote in (votes or {}).items():
        v = (vote or "").strip().upper()[:1]
        if v == ans and source not in seen_sources:
            agreement += 1
            seen_sources.add(source)
    scores = _clean_scores(score_map)
    if scores:
        top = max(scores, key=scores.get)
        if top == ans:
            agreement += 1
    return agreement


def _best_score_margin(score_map: Mapping[str, float] | None) -> float:
    scores = _clean_scores(score_map)
    if len(scores) < 2:
        return 0.0
    vals = sorted(scores.values(), reverse=True)
    return max(0.0, vals[0] - vals[1])


def required_margin_for(
    *,
    strategy: str,
    difficulty: float,
    risk_score: float,
    vote_margin: float,
    time_pressure: bool = False,
    profile: str = "balanced",
    option_sim: float = 0.0,
    negation: bool = False,
) -> float:
    """Dynamic margin threshold for early exits.

    Higher thresholds are used for private-test traps: negation, highly similar
    options, high estimated difficulty, and risk-gate rows.  Under time pressure
    it relaxes slightly, but it never becomes a blind accept-all.
    """
    profile = (profile or "balanced").strip().lower()
    strategy = (strategy or "").strip().lower()
    base = 0.34
    if strategy in {"token_fast_exit", "batch_token_fast_exit"}:
        base = 0.36
    elif strategy in {"direct", "batch_direct"}:
        base = 0.30
    elif strategy in {"vote"}:
        base = 0.22

    if profile in {"accuracy", "max_accuracy"}:
        base += 0.08
    elif profile in {"turbo", "fast"}:
        base -= 0.04

    base += max(0.0, difficulty - 0.35) * 0.22
    base += max(0.0, risk_score - 0.35) * 0.24
    if option_sim >= 0.56:
        base += 0.06
    if negation:
        base += 0.07
    if vote_margin and vote_margin >= 0.42:
        base -= 0.03
    if time_pressure:
        base -= 0.06
    return max(0.16, min(0.72, base))


def evaluate_early_exit(
    item: MCQItem,
    *,
    answer: str | None,
    strategy: str,
    difficulty: float,
    risk_score: float = 0.0,
    token_margin: float = 0.0,
    vote_margin: float = 0.0,
    confidence: float = 0.0,
    votes: Mapping[str, str] | None = None,
    score_map: Mapping[str, float] | None = None,
    mode: str = "adaptive",
    time_pressure: bool = False,
    profile: str = "balanced",
) -> GuardDecision:
    ans = (answer or "").strip().upper()[:1]
    opt_sim = option_similarity(item)
    neg = has_negation(item)
    score_margin = _best_score_margin(score_map)
    effective_margin = max(float(token_margin or 0.0), float(vote_margin or 0.0), score_margin)
    agreement = _agreement_for(ans, votes, score_map)
    required = required_margin_for(
        strategy=strategy,
        difficulty=float(difficulty or 0.0),
        risk_score=float(risk_score or 0.0),
        vote_margin=float(vote_margin or 0.0),
        time_pressure=time_pressure,
        profile=profile,
        option_sim=opt_sim,
        negation=neg,
    )
    notes: list[str] = []
    if opt_sim >= 0.56:
        notes.append(f"option_sim={opt_sim:.3f}")
    if neg:
        notes.append("negation=1")
    if score_margin:
        notes.append(f"score_margin={score_margin:.3f}")

    if ans not in VALID_ANSWERS:
        return GuardDecision(False, "invalid_answer", required, agreement, risk_score, difficulty, vote_margin, token_margin, notes)

    # Explicit direct mode is allowed to be faster, but still blocks high-risk blind exits.
    direct_intent = (mode or "").strip().lower() == "direct"
    high_risk = risk_score >= 0.62 or difficulty >= 0.68 or opt_sim >= 0.66 or neg
    very_strong = effective_margin >= required + 0.14 and agreement >= 2
    strong_enough = effective_margin >= required and (agreement >= 2 or confidence >= 0.72 or not high_risk)

    if high_risk and not very_strong and not time_pressure:
        return GuardDecision(False, "high_risk_requires_deep_check", required, agreement, risk_score, difficulty, vote_margin, token_margin, notes)

    # Batch/direct generation has no logit margin.  Keep the V8 speed win for
    # obviously easy rows, but only when the dataset risk signals are low.
    if strategy in {"direct", "batch_direct"} and not high_risk and confidence >= 0.64 and difficulty <= 0.32 and risk_score <= 0.35:
        return GuardDecision(True, "easy_direct_row", required, agreement, risk_score, difficulty, vote_margin, token_margin, notes)

    if not strong_enough and not (direct_intent and not high_risk and effective_margin >= required * 0.75):
        return GuardDecision(False, "weak_margin_or_low_agreement", required, agreement, risk_score, difficulty, vote_margin, token_margin, notes)
    return GuardDecision(True, "signals_strong_enough", required, agreement, risk_score, difficulty, vote_margin, token_margin, notes)


def guard_note(decision: GuardDecision | None) -> str:
    return f";{decision.note}" if decision else ""
