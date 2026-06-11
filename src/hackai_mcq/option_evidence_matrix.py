from __future__ import annotations

"""Option-aware evidence retrieval.

Normal RAG often retrieves context for the question only. For MCQ, the best
signal is frequently: which option has direct evidence support and which option
is merely lexically similar. This module retrieves evidence per option and emits
both a compact context block and soft priors for the solver.
"""

from dataclasses import dataclass, field
from typing import Any

from .evidence_compressor import EvidenceSnippet, compress_evidence, score_text
from .schema import MCQItem


@dataclass(slots=True)
class OptionEvidenceRow:
    option: str
    support: float
    evidence: list[EvidenceSnippet] = field(default_factory=list)


@dataclass(slots=True)
class OptionEvidenceMatrix:
    rows: dict[str, OptionEvidenceRow]
    best: str | None
    margin: float
    context: str

    def score_map(self) -> dict[str, float]:
        vals = {k: max(0.0, row.support) for k, row in self.rows.items()}
        total = sum(vals.values()) or 1.0
        return {k: v / total for k, v in vals.items()}


def _safe_search(rag: Any, query: str, k: int) -> list[EvidenceSnippet]:
    try:
        return list(rag.search(query, k=k))
    except TypeError:
        return list(rag.search(query)[:k])


def build_option_evidence_matrix(item: MCQItem, rag: Any, k_per_option: int = 2, max_context_chars: int = 1400) -> OptionEvidenceMatrix:
    rows: dict[str, OptionEvidenceRow] = {}
    all_evidence: list[EvidenceSnippet] = []
    question = item.question.strip()
    for opt in "ABCD":
        opt_text = item.options.get(opt, "").strip()
        if not opt_text:
            rows[opt] = OptionEvidenceRow(opt, 0.0, [])
            continue
        query = f"{question}\nCandidate {opt}: {opt_text}"
        hits = _safe_search(rag, query, k_per_option)
        all_evidence.extend(hits)
        # Support = evidence similarity to option + partial similarity to question.
        support = 0.0
        for h in hits:
            support = max(support, 0.72 * score_text(opt_text, h.text) + 0.28 * score_text(question, h.text) + 0.05 * float(h.score))
        rows[opt] = OptionEvidenceRow(opt, support, hits)
    ordered = sorted(rows.values(), key=lambda r: r.support, reverse=True)
    best = ordered[0].option if ordered and ordered[0].support > 0 else None
    margin = (ordered[0].support - ordered[1].support) if len(ordered) > 1 else (ordered[0].support if ordered else 0.0)
    # Remove duplicate snippets by source/doc_id before compression.
    seen: set[str] = set(); dedup: list[EvidenceSnippet] = []
    for e in sorted(all_evidence, key=lambda x: x.score, reverse=True):
        key = f"{e.source}:{e.doc_id}:{hash(e.text[:120])}"
        if key in seen:
            continue
        seen.add(key); dedup.append(e)
    matrix_lines = ["Option evidence matrix (soft signal, not final answer):"]
    for opt in "ABCD":
        row = rows[opt]
        matrix_lines.append(f"{opt}: support={row.support:.3f}; option={item.options.get(opt,'')[:160]}")
    evidence_context = compress_evidence(item.text_for_retrieval(), dedup[: max(4, 2 * k_per_option)], max_chars=max_context_chars)
    context = "\n".join(matrix_lines) + ("\n\nEvidence snippets:\n" + evidence_context if evidence_context else "")
    return OptionEvidenceMatrix(rows=rows, best=best, margin=margin, context=context[:max_context_chars + 700])
