from __future__ import annotations

"""Detect knowledge gaps and produce targeted corpus needs.

This is used by offline experiments and final reports. It does not call the web;
it tells the user what corpus to add next when public/dev errors cluster in a
specific topic or language.
"""

from dataclasses import dataclass, field

from .multilingual_nlp_adapter import analyze_multilingual
from .features import has_negation, tokenize
from .schema import MCQItem


@dataclass(slots=True)
class KnowledgeGapReport:
    gap_score: float
    tags: list[str] = field(default_factory=list)
    recommended_corpus: list[str] = field(default_factory=list)


def analyze_gap(item: MCQItem, evidence_count: int = 0, evidence_coverage: float = 0.0) -> KnowledgeGapReport:
    text = item.text_for_retrieval()
    sig = analyze_multilingual(text)
    tags: list[str] = []
    recs: list[str] = []
    score = 0.0
    low_evidence = evidence_count == 0 or evidence_coverage < 0.08
    if low_evidence:
        score += 0.35; tags.append("low_evidence")
    if sig.language not in {"vi", "en", "unknown"} or sig.is_mixed_language:
        score += 0.12; tags.append("multilingual_gap"); recs.append("Add multilingual Wikipedia/grammar glossary for detected scripts")
    domains = sig.domains or []
    for d in domains:
        tags.append("domain_" + d)
        if d in {"law", "finance", "medical", "education"}:
            score += 0.10; recs.append(f"Add verified {d} corpus and definitions")
    if any(t in text.lower() for t in ["docker", "python", "api", "model", "llm", "ai"]):
        tags.append("computing_ai"); recs.append("Add computing/AI documentation corpus")
    if any(ch.isdigit() for ch in text):
        tags.append("numeric"); recs.append("Add formula/table/reference corpus for numeric facts")
    if has_negation(item):
        tags.append("exam_negation"); recs.append("Add MCQ trap/negation playbook")
    if len(tokenize(text)) > 120:
        tags.append("long_context")
    return KnowledgeGapReport(min(1.0, score), tags, sorted(set(recs)))
