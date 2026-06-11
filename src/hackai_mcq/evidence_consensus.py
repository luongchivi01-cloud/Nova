from __future__ import annotations

"""Evidence quality, diversity, and consensus helpers."""

from dataclasses import dataclass, field

from .evidence_compressor import EvidenceSnippet, score_text
from .schema import MCQItem

@dataclass(slots=True)
class EvidenceConsensusReport:
    ok: bool
    coverage: float
    source_diversity: int
    option_support: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def evaluate_evidence_consensus(item: MCQItem, snippets: list[EvidenceSnippet], min_coverage: float = 0.06) -> EvidenceConsensusReport:
    if not snippets:
        return EvidenceConsensusReport(False, 0.0, 0, {}, ["no_evidence"])
    query = item.text_for_retrieval()
    coverage = max(score_text(query, s.text) for s in snippets)
    sources = {s.source for s in snippets if s.source}
    support: dict[str, float] = {}
    for key in "ABCD":
        opt = item.options.get(key, "")
        support[key] = max((score_text(opt, s.text) + 0.35 * score_text(item.question, s.text) for s in snippets), default=0.0)
    warnings: list[str] = []
    if coverage < min_coverage:
        warnings.append("low_query_coverage")
    if len(sources) < 1:
        warnings.append("low_source_diversity")
    return EvidenceConsensusReport(coverage >= min_coverage, coverage, len(sources), support, warnings)


def consensus_score_map(item: MCQItem, snippets: list[EvidenceSnippet]) -> dict[str, float]:
    report = evaluate_evidence_consensus(item, snippets)
    if not report.option_support:
        return {}
    maxv = max(report.option_support.values()) or 1.0
    # Only provide a soft signal; the legal model remains the final judge.
    return {k: min(1.0, v / maxv) for k, v in report.option_support.items()}
