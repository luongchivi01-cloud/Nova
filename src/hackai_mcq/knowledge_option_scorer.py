from __future__ import annotations

"""Evidence-based option prior for weak local models.

This is not a model fallback. It uses retrieved offline evidence to add a soft,
explainable prior to the normal model/vote pipeline. The final answer is still
chosen by the solver's weighted ensemble with model generations/token scores.
"""

import math
import re
from dataclasses import dataclass
from typing import Iterable, Mapping

from .evidence_compressor import EvidenceSnippet
from .features import has_negation, tokenize
from .schema import MCQItem, VALID_ANSWERS

_STOP = {
    "the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "with", "is", "are", "was", "were",
    "là", "của", "và", "hoặc", "trong", "một", "những", "các", "được", "có", "không", "nào", "gì",
    "which", "what", "who", "where", "when", "why", "how", "best", "correct", "incorrect", "except",
}

@dataclass(slots=True)
class KnowledgeOptionScore:
    answer: str | None
    confidence: float
    scores: dict[str, float]
    notes: str = ""


def _terms(text: str) -> list[str]:
    return [t for t in tokenize(text.lower()) if len(t) > 1 and t not in _STOP]


def _phrase_hits(text: str, phrase: str) -> float:
    phrase = re.sub(r"\s+", " ", phrase.lower()).strip()
    if len(phrase) < 3:
        return 0.0
    hay = text.lower()
    if phrase in hay:
        return 2.5 + min(1.5, len(phrase) / 80.0)
    # partial quoted/number phrase bonus
    nums = re.findall(r"\d+(?:[.,]\d+)?%?", phrase)
    return 0.4 * sum(1 for n in nums if n in hay)


def _normalize_scores(raw: Mapping[str, float]) -> dict[str, float]:
    vals = {k: float(raw.get(k, 0.0)) for k in "ABCD"}
    if not any(v > 0 for v in vals.values()):
        return {k: 0.25 for k in "ABCD"}
    m = max(vals.values())
    exps = {k: math.exp(max(-20.0, min(20.0, v - m))) for k, v in vals.items()}
    z = sum(exps.values()) or 1.0
    return {k: v / z for k, v in exps.items()}


def score_options_from_evidence(item: MCQItem, evidence: Iterable[EvidenceSnippet] | str | None, min_confidence: float = 0.34) -> KnowledgeOptionScore:
    if evidence is None:
        return KnowledgeOptionScore(None, 0.0, {}, "no_evidence")
    if isinstance(evidence, str):
        snippets = [EvidenceSnippet("context", evidence, 1.0, "context")]
    else:
        snippets = list(evidence)
    if not snippets:
        return KnowledgeOptionScore(None, 0.0, {}, "empty_evidence")

    question_terms = set(_terms(item.question))
    text = "\n".join(s.text for s in snippets[:8])
    evidence_terms = set(_terms(text))
    raw: dict[str, float] = {k: 0.0 for k in "ABCD"}

    for opt, value in item.options.items():
        if opt not in VALID_ANSWERS:
            continue
        opt_terms = _terms(value)
        if not opt_terms:
            continue
        overlap = sum(1.0 for t in opt_terms if t in evidence_terms)
        rareish = sum(1.0 for t in opt_terms if len(t) >= 6 and t in evidence_terms)
        phrase = _phrase_hits(text, value)
        # Avoid rewarding option words that merely repeat the question; reward evidence-specific overlap.
        question_penalty = 0.20 * sum(1.0 for t in opt_terms if t in question_terms)
        raw[opt] = phrase + overlap + 0.45 * rareish - question_penalty

    probs = _normalize_scores(raw)
    ordered = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    top, topv = ordered[0]
    second = ordered[1][1] if len(ordered) > 1 else 0.0
    margin = topv - second
    # For negation questions, evidence overlap is less decisive; keep it soft.
    neg = has_negation(item)
    confidence = min(0.82 if not neg else 0.58, 0.25 + 1.8 * margin)
    if max(raw.values()) <= 0.0 or confidence < min_confidence:
        return KnowledgeOptionScore(None, confidence, probs, f"weak_prior;max_raw={max(raw.values()):.2f};margin={margin:.3f};neg={neg}")
    return KnowledgeOptionScore(top, confidence, probs, f"knowledge_prior;raw={raw};margin={margin:.3f};neg={neg}")
