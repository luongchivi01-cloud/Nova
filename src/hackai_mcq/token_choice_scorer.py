from __future__ import annotations

"""Robust A/B/C/D token scoring layer.

The core competition target is a four-way multiple-choice answer. Generation is
fragile because models may emit explanations or malformed text. This module
wraps backends that expose score_choices() and turns raw logits/log-probs into a
calibrated, length-aware choice distribution that can be fused with generation
votes.
"""

import math
from dataclasses import dataclass
from typing import Mapping

from .ensembling import softmax, top_margin
from .features import has_negation, lexical_overlap
from .schema import MCQItem, VALID_ANSWERS


@dataclass(slots=True)
class ChoiceScoreResult:
    answer: str | None
    margin: float
    confidence: float
    probabilities: dict[str, float]
    raw_scores: dict[str, float]
    entropy: float
    notes: str = ""


def _entropy(prob: Mapping[str, float]) -> float:
    vals = [max(1e-12, float(prob.get(k, 0.0))) for k in "ABCD"]
    h = -sum(v * math.log(v) for v in vals)
    return h / math.log(4.0)


def _option_quality_prior(item: MCQItem) -> dict[str, float]:
    """Tiny deterministic prior used only to break near ties.

    The prior is intentionally weak so it cannot dominate an LLM; it helps when
    token scores are effectively equal or when an option is empty/noisy.
    """
    q = item.question
    neg = has_negation(item)
    out: dict[str, float] = {}
    for k in "ABCD":
        opt = item.options.get(k, "") or ""
        score = 0.0
        if not opt.strip():
            score -= 0.35
        score += min(0.04, max(0, len(opt.strip()) - 4) / 900.0)
        # Weak lexical anchor. Useful for definition questions; too weak to overfit.
        score += min(0.08, lexical_overlap(q, opt) * 0.08)
        if neg and any(t in opt.lower() for t in ["not", "incorrect", "false", "sai", "không", "khong"]):
            score += 0.015
        out[k] = score
    return out


def normalize_choice_scores(raw_scores: Mapping[str, float], item: MCQItem, temperature: float = 1.0, prior_strength: float = 0.20) -> ChoiceScoreResult:
    clean = {k: float(raw_scores.get(k, -1e9)) for k in "ABCD"}
    if all(v <= -1e8 for v in clean.values()):
        return ChoiceScoreResult(None, 0.0, 0.0, {}, clean, 1.0, "all scores invalid")

    prior = _option_quality_prior(item)
    mixed = {k: clean[k] + prior_strength * prior[k] for k in "ABCD"}
    prob = softmax(mixed, temperature=max(1e-6, temperature))
    ans, margin = top_margin(prob)
    ent = _entropy(prob)
    confidence = min(0.98, max(0.0, 0.55 + margin * 1.35 - ent * 0.10)) if ans else 0.0
    return ChoiceScoreResult(ans, margin, confidence, prob, clean, ent, f"entropy={ent:.3f}")


def score_with_backend(backend, prompt: str, item: MCQItem, *, temperature: float = 1.0) -> ChoiceScoreResult:
    if not hasattr(backend, "score_choices"):
        return ChoiceScoreResult(None, 0.0, 0.0, {}, {}, 1.0, "backend has no score_choices")
    raw = backend.score_choices(prompt, item)
    raw = {str(k).strip().upper()[:1]: float(v) for k, v in raw.items() if str(k).strip().upper()[:1] in VALID_ANSWERS}
    return normalize_choice_scores(raw, item, temperature=temperature)
