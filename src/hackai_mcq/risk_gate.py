from __future__ import annotations

"""Risk gate for adaptive MCQ solving.

The gate identifies rows where weak local models are most likely to fail:
negation, near-duplicate choices, mixed language, low evidence, numeric traps,
or high disagreement. The solver can then spend extra reasoning only on those
rows instead of brute-forcing all 2000 questions.
"""

from dataclasses import dataclass, field

from .features import has_negation, option_similarity, tokenize
from .multilingual_nlp_adapter import analyze_multilingual
from .schema import MCQItem


@dataclass(slots=True)
class RiskDecision:
    score: float
    level: str
    reasons: list[str] = field(default_factory=list)

    @property
    def should_deepen(self) -> bool:
        return self.score >= 0.55

    @property
    def should_force_evidence(self) -> bool:
        return self.score >= 0.40


def assess_risk(item: MCQItem, evidence_count: int = 0, vote_margin: float | None = None, token_margin: float | None = None) -> RiskDecision:
    score = 0.0
    reasons: list[str] = []
    q = item.question or ""
    text = item.text_for_retrieval()
    tokens = tokenize(text)
    sig = analyze_multilingual(text)

    if has_negation(item) or sig.has_negation:
        score += 0.22; reasons.append("negation_or_exception")
    sim = option_similarity(item)
    if sim >= 0.55:
        score += min(0.24, (sim - 0.45) * 0.45); reasons.append(f"near_duplicate_options:{sim:.2f}")
    if sig.is_mixed_language or sig.language not in {"vi", "en", "unknown"}:
        score += 0.14; reasons.append(f"language:{sig.language}{':mixed' if sig.is_mixed_language else ''}")
    if len(tokens) > 120:
        score += 0.10; reasons.append("long_context")
    if any(ch.isdigit() for ch in text):
        score += 0.08; reasons.append("numeric_or_date")
    if any(w in q.lower() for w in ["most", "least", "best", "đúng nhất", "phù hợp nhất", "ngoại trừ", "except", "incorrect"]):
        score += 0.10; reasons.append("comparative_or_exam_trap")
    if evidence_count == 0:
        score += 0.10; reasons.append("no_evidence")
    elif evidence_count < 2:
        score += 0.04; reasons.append("thin_evidence")
    if vote_margin is not None and vote_margin < 0.18:
        score += 0.16; reasons.append(f"low_vote_margin:{vote_margin:.2f}")
    if token_margin is not None and token_margin < 0.08:
        score += 0.12; reasons.append(f"low_token_margin:{token_margin:.2f}")
    score = max(0.0, min(1.0, score))
    level = "low" if score < 0.33 else "medium" if score < 0.62 else "high"
    return RiskDecision(score, level, reasons)
