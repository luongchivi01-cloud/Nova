from __future__ import annotations

"""Robust multi-query planning for offline knowledge retrieval.

This is intentionally dependency-light. It borrows the *shape* of stronger RAG
systems: query expansion, option-aware retrieval, multilingual/domain cues, and
multi-hop relation queries. It never calls live web/API services.
"""

from dataclasses import dataclass, field
import re

from .features import has_negation, tokenize
from .multilingual_nlp_adapter import analyze_multilingual
from .schema import MCQItem

DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "math": ["formula", "definition", "theorem", "calculate", "probability", "equation", "công thức", "định lý", "xác suất"],
    "science": ["physics", "chemistry", "biology", "force", "energy", "cell", "atom", "vật lý", "hóa học", "sinh học"],
    "computing": ["algorithm", "database", "network", "AI", "machine learning", "Docker", "thuật toán", "dữ liệu"],
    "law_society": ["law", "legal", "civil", "contract", "insurance", "economics", "luật", "hợp đồng", "bảo hiểm", "kinh tế"],
    "language": ["grammar", "synonym", "meaning", "translation", "ngữ pháp", "từ đồng nghĩa", "dịch"],
    "history_geo": ["history", "capital", "country", "war", "geography", "lịch sử", "thủ đô", "quốc gia", "địa lý"],
}

@dataclass(slots=True)
class KnowledgeQueryPlan:
    primary: str
    expanded: list[str] = field(default_factory=list)
    option_queries: dict[str, str] = field(default_factory=dict)
    domain_tags: list[str] = field(default_factory=list)
    language: str = "unknown"
    risk_tags: list[str] = field(default_factory=list)

    def all_queries(self, limit: int = 12) -> list[str]:
        out: list[str] = []
        for q in [self.primary, *self.expanded, *self.option_queries.values()]:
            q = normalize_query(q)
            if q and q not in out:
                out.append(q)
            if len(out) >= limit:
                break
        return out


def normalize_query(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "").replace("\x00", " ")).strip()
    return text[:600]


def _salient_terms(text: str, max_terms: int = 16) -> list[str]:
    toks = tokenize(text)
    stop = {
        "the","and","or","of","to","in","is","are","a","an","for","with","on","as","by","be","which","what",
        "của","và","hoặc","là","trong","với","cho","được","nào","gì","hãy","câu","sau","đây",
    }
    seen: set[str] = set(); out: list[str] = []
    for t in toks:
        if len(t) < 2 or t.lower() in stop or t in seen:
            continue
        seen.add(t); out.append(t)
        if len(out) >= max_terms:
            break
    return out


def detect_domain(item: MCQItem) -> list[str]:
    text = item.text_for_retrieval().lower()
    tags: list[str] = []
    for name, kws in DOMAIN_KEYWORDS.items():
        if any(k.lower() in text for k in kws):
            tags.append(name)
    if not tags:
        # lightweight token-based fallback
        terms = set(_salient_terms(text, 24))
        if terms & {"year", "country", "capital", "war", "city", "nước", "thủ", "đô"}:
            tags.append("history_geo")
        if terms & {"ai", "model", "data", "docker", "api", "python"}:
            tags.append("computing")
    return tags or ["general"]


def plan_queries(item: MCQItem, max_queries: int = 12) -> KnowledgeQueryPlan:
    lang = analyze_multilingual(item.text_for_retrieval()).language
    domains = detect_domain(item)
    risk: list[str] = []
    if has_negation(item):
        risk.append("negation")
    if len(item.question) > 350:
        risk.append("long_question")
    if lang in {"mixed", "unknown"}:
        risk.append("language_mixed_or_unknown")

    question = normalize_query(item.question)
    terms = " ".join(_salient_terms(item.question, 18))
    primary = normalize_query(f"{question} {terms}")
    expanded: list[str] = [primary]

    for d in domains[:3]:
        hints = " ".join(DOMAIN_KEYWORDS.get(d, [])[:6])
        expanded.append(normalize_query(f"{question} {hints}"))
    if risk:
        expanded.append(normalize_query(f"{question} {' '.join(risk)} correct answer exception negation"))

    option_queries: dict[str, str] = {}
    for key in "ABCD":
        opt = normalize_query(item.options.get(key, ""))
        if opt:
            option_queries[key] = normalize_query(f"{question} {opt} {' '.join(domains)}")

    # Keep diverse order: primary, domain, option, risk.
    seen: list[str] = []
    for q in expanded:
        if q and q not in seen:
            seen.append(q)
    return KnowledgeQueryPlan(primary=primary, expanded=seen[:max_queries], option_queries=option_queries, domain_tags=domains, language=lang, risk_tags=risk)
