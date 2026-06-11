from __future__ import annotations

"""Final per-row quality gate.

This gate is deliberately lightweight.  It does not invent answers and does not
call external services.  It validates the solver result, detects low-confidence
risky answers, and can request one same-model repair pass through the solver's
existing strict backend path.
"""

from dataclasses import dataclass

from .answer_parser import parse_answer
from .confidence_calibrator import calibrate_confidence
from .ensembling import VoteState
from .risk_gate import assess_risk
from .schema import MCQItem, SolverResult


@dataclass(slots=True)
class QualityDecision:
    ok: bool
    needs_rescue: bool
    reason: str


def assess_result_quality(item: MCQItem, result: SolverResult, confidence_threshold: float = 0.62) -> QualityDecision:
    ans = parse_answer(result.answer)
    if ans not in {"A", "B", "C", "D"}:
        return QualityDecision(False, True, "invalid_answer")
    risk = assess_risk(item)
    low = result.confidence < min(0.55, confidence_threshold)
    if low and risk.score >= 0.62 and result.strategy in {"direct", "token_score", "token_fast_exit"}:
        return QualityDecision(True, True, f"low_conf_high_risk:{result.confidence:.2f}/{risk.score:.2f}")
    weighted = result.scores or {}
    if weighted:
        vals = sorted([float(v) for v in weighted.values()], reverse=True)
        if len(vals) >= 2 and (vals[0] - vals[1]) < 0.04 and risk.score >= 0.55:
            return QualityDecision(True, True, "near_tie_high_risk")
    return QualityDecision(True, False, "ok")


def normalize_result(item: MCQItem, result: SolverResult) -> SolverResult:
    ans = parse_answer(result.answer)
    if ans == result.answer and ans in {"A", "B", "C", "D"}:
        return result
    if ans in {"A", "B", "C", "D"}:
        return SolverResult(result.qid, ans, result.confidence, result.strategy + ":normalized", result.votes, result.scores, (result.notes or "") + ";quality_gate=normalized")
    return result
