from __future__ import annotations

"""Evidence reranking for MCQ-style retrieval."""

from dataclasses import dataclass
from .evidence_compressor import EvidenceSnippet, score_text
from .features import has_negation, tokenize
from .multilingual_nlp_adapter import analyze_multilingual
from .schema import MCQItem

@dataclass(slots=True)
class RerankTrace:
    language: str
    domains: list[str]
    has_negation: bool
    scored: int

def _option_terms(item: MCQItem) -> set[str]:
    terms: set[str] = set()
    for v in item.options.values():
        terms.update(tokenize(v))
    return terms

def rerank_evidence(item: MCQItem | str, snippets: list[EvidenceSnippet], limit: int = 8) -> list[EvidenceSnippet]:
    if not snippets:
        return []
    if isinstance(item, str):
        query = item
        sig = analyze_multilingual(item)
        option_terms: set[str] = set()
        question_terms = set(tokenize(item))
        neg = sig.has_negation
    else:
        query = item.text_for_retrieval()
        sig = analyze_multilingual(query)
        option_terms = _option_terms(item)
        question_terms = set(tokenize(item.question))
        neg = sig.has_negation
    scored: list[tuple[float, int, EvidenceSnippet]] = []
    for i, s in enumerate(snippets):
        text = s.text
        toks = set(tokenize(text))
        q_overlap = len(toks & question_terms) / max(1, len(question_terms))
        opt_overlap = len(toks & option_terms) / max(1, len(option_terms)) if option_terms else 0.0
        neg_bonus = 0.08 if neg and any(x in text.lower() for x in ["not", "except", "false", "không", "sai", "ngoại trừ", "不", "ない", "не"]) else 0.0
        domain_bonus = 0.0
        for d in sig.domains:
            if d.replace("_", " ") in text.lower():
                domain_bonus += 0.03
        score = float(s.score) + 1.6 * score_text(query, text) + 0.7 * q_overlap + 0.45 * opt_overlap + neg_bonus + min(domain_bonus, 0.12)
        scored.append((score, -i, EvidenceSnippet(s.doc_id, s.text, score, s.source)))
    scored.sort(reverse=True)
    return [s for _, _, s in scored[:limit]]
