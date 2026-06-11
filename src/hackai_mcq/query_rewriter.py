
from __future__ import annotations

"""Question-to-search query rewriting for offline knowledge retrieval."""

from dataclasses import dataclass
import re

from .multilingual_nlp_adapter import analyze_multilingual
from .normalization import canonical
from .schema import MCQItem

STOPWORDS = {
    "vi": {"là", "của", "và", "hoặc", "trong", "được", "các", "một", "những", "nào", "gì"},
    "en": {"the", "a", "an", "of", "and", "or", "in", "on", "to", "is", "are", "which", "what"},
}

@dataclass(slots=True)
class QueryBundle:
    primary: str
    expanded: list[str]
    language: str
    domains: list[str]
    has_negation: bool


def _terms(text: str, language: str) -> list[str]:
    c = canonical(text)
    tokens = re.findall(r"[\w\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\u0400-\u04ff\u0600-\u06ff\u0e00-\u0e7f]+", c, flags=re.UNICODE)
    stop = STOPWORDS.get(language, set())
    out: list[str] = []
    seen: set[str] = set()
    for t in tokens:
        if len(t) <= 1 or t in stop or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def rewrite_queries(item: MCQItem, max_queries: int = 5) -> QueryBundle:
    """Create compact, domain-aware queries for a multiple-choice row.

    The goal is not to call online search. These queries are used against the
    offline corpus only. We include choices because MCQ options often contain
    the entity/date/term needed for retrieval.
    """
    raw = item.text_for_retrieval()
    sig = analyze_multilingual(raw)
    lang = sig.language
    q_terms = _terms(item.question, lang)
    option_terms: list[str] = []
    for k in "ABCD":
        option_terms.extend(_terms(item.options.get(k, ""), lang)[:4])
    merged: list[str] = []
    for t in [*q_terms, *option_terms]:
        if t not in merged:
            merged.append(t)
    primary = " ".join(merged[:18]) or item.question[:180]
    expanded = [primary]
    if q_terms:
        expanded.append(" ".join(q_terms[:14]))
    # Pair question with each option to surface exact supporting evidence.
    q_head = " ".join(q_terms[:10])
    for k in "ABCD":
        ot = " ".join(_terms(item.options.get(k, ""), lang)[:8])
        if ot:
            expanded.append((q_head + " " + ot).strip())
    # Deduplicate and cap.
    dedup: list[str] = []
    for q in expanded:
        q = q.strip()
        if q and q not in dedup:
            dedup.append(q)
    return QueryBundle(primary=primary, expanded=dedup[:max_queries], language=lang, domains=sig.domains, has_negation=sig.has_negation)
