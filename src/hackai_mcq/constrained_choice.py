from __future__ import annotations

"""Constrained A/B/C/D answering helpers.

This module is intentionally dependency-light. It exposes a uniform function
that first prefers token/logit scoring (best for MCQ), then optional structured
output engines if installed, and finally the existing parser fallback.

Why this matters for HackAIthon C:
- The judge expects exactly qid,answer with answer in A/B/C/D.
- Small local models often produce verbose Vietnamese explanations.
- Constraining the output to one of four labels reduces parser failure and
  saves generation tokens, which matters on 6GB VRAM.
"""

from dataclasses import dataclass
from typing import Iterable

from .answer_parser import parse_answer
from .ensembling import softmax, top_margin
from .model_backends import Backend, ChoiceScoringBackend
from .prompts import constrained_choice_prompt
from .schema import MCQItem, VALID_ANSWERS


@dataclass(slots=True)
class ConstrainedChoiceResult:
    answer: str | None
    confidence: float
    method: str
    scores: dict[str, float]
    raw: str = ""


def _normalize_allowed(allowed: Iterable[str] | None = None) -> set[str]:
    vals = {a.strip().upper() for a in (allowed or VALID_ANSWERS)}
    return vals & VALID_ANSWERS or set(VALID_ANSWERS)


def choose_with_token_scoring(
    backend: Backend,
    item: MCQItem,
    context: str = "",
    allowed: Iterable[str] | None = None,
) -> ConstrainedChoiceResult:
    allowed_set = _normalize_allowed(allowed)
    if not isinstance(backend, ChoiceScoringBackend):
        return ConstrainedChoiceResult(None, 0.0, "token_score_unavailable", {})
    prompt = constrained_choice_prompt(item, context)
    raw_scores = backend.score_choices(prompt, item)
    filtered = {k: float(v) for k, v in raw_scores.items() if k in allowed_set}
    if not filtered:
        return ConstrainedChoiceResult(None, 0.0, "token_score_empty", {})
    probs = softmax(filtered)
    answer, margin = top_margin(probs)
    return ConstrainedChoiceResult(answer, max(0.0, min(1.0, 0.50 + margin)), "token_score", probs)


def choose_with_generation(
    backend: Backend,
    item: MCQItem,
    prompt: str,
    allowed: Iterable[str] | None = None,
) -> ConstrainedChoiceResult:
    allowed_set = _normalize_allowed(allowed)
    raw = backend.generate(prompt, item)
    ans = parse_answer(raw)
    if ans in allowed_set:
        return ConstrainedChoiceResult(ans, 0.45, "generate_parse", {}, raw)
    return ConstrainedChoiceResult(None, 0.0, "generate_parse_failed", {}, raw)


def choose_constrained(
    backend: Backend,
    item: MCQItem,
    context: str = "",
    allowed: Iterable[str] | None = None,
    prompt: str | None = None,
) -> ConstrainedChoiceResult:
    """Return one answer while minimizing verbose generation.

    This function is safe to use in the official path because it does not call
    network APIs or browser automation. Optional structured-generation packages
    can be added by future adapters, but the core method remains token scoring.
    """
    score = choose_with_token_scoring(backend, item, context=context, allowed=allowed)
    if score.answer:
        return score
    return choose_with_generation(
        backend,
        item,
        prompt or constrained_choice_prompt(item, context),
        allowed=allowed,
    )
