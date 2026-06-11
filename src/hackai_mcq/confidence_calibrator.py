from __future__ import annotations

"""Conservative confidence calibration for adaptive routing.

The goal is not to claim true probability, but to decide whether a row deserves
extra expensive passes such as permutation/pairwise/judge. It fuses vote margin,
token-score margin, difficulty, language risk and permutation consistency.
"""

from dataclasses import dataclass
from typing import Mapping

from .ensembling import VoteState
from .features import estimate_difficulty, has_negation
from .multilingual_nlp_adapter import analyze_multilingual
from .schema import MCQItem


@dataclass(slots=True)
class ConfidenceState:
    answer: str | None
    confidence: float
    risk: float
    should_escalate: bool
    notes: str


def calibrate_confidence(
    item: MCQItem,
    vote_state: VoteState,
    *,
    token_margin: float = 0.0,
    permutation_consistent: bool | None = None,
    time_pressure: bool = False,
    threshold: float = 0.62,
) -> ConfidenceState:
    difficulty = estimate_difficulty(item)
    sig = analyze_multilingual(item.text_for_retrieval())
    risk = difficulty * 0.45
    if has_negation(item):
        risk += 0.12
    if sig.is_mixed_language:
        risk += 0.06
    if sig.language not in {"vi", "en", "unknown"}:
        risk += 0.04
    if permutation_consistent is False:
        risk += 0.18
    elif permutation_consistent is True:
        risk -= 0.08
    if time_pressure:
        risk -= 0.05
    risk = max(0.0, min(0.95, risk))

    base = vote_state.confidence if vote_state.answer else 0.0
    base += min(0.18, max(0.0, token_margin) * 0.90)
    base += min(0.10, max(0.0, vote_state.margin) * 0.45)
    confidence = max(0.0, min(0.98, base - risk * 0.26))
    should_escalate = (confidence < threshold and not time_pressure) or (vote_state.margin < 0.16 and not time_pressure)
    notes = f"calibrated={confidence:.3f};risk={risk:.3f};difficulty={difficulty:.3f};perm={permutation_consistent}"
    return ConfidenceState(vote_state.answer, confidence, risk, should_escalate, notes)
