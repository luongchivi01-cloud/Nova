from __future__ import annotations

"""Tiny dependency-free graph-style retrieval over corpus chunks.

This is a lightweight alternative to heavy GraphRAG/LightRAG stacks for the
official Docker run. It builds co-occurrence neighborhoods from corpus terms and
returns snippets whose entity neighborhoods overlap the query/options.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
import math

from .corpus_builder import CorpusDoc
from .evidence_compressor import EvidenceSnippet
from .features import tokenize


def graph_most_common(counter_like, n: int):
    if hasattr(counter_like, "most_common"):
        return counter_like.most_common(n)
    return sorted(counter_like.items(), key=lambda kv: kv[1], reverse=True)[:n]


def _entities(text: str, limit: int = 32) -> list[str]:
    toks = [t.lower() for t in tokenize(text) if len(t) >= 3]
    stop = {"the","and","for","with","from","this","that","which","what","của","và","trong","được","các","những","một"}
    counts = Counter(t for t in toks if t not in stop)
    return [t for t, _ in counts.most_common(limit)]

@dataclass(slots=True)
class GraphLiteIndex:
    docs: list[CorpusDoc]
    doc_entities: list[set[str]]
    graph: dict[str, Counter]
    idf: dict[str, float]

    @classmethod
    def build(cls, docs: list[CorpusDoc], max_docs: int = 20000) -> "GraphLiteIndex":
        sample = docs[:max_docs]
        doc_entities: list[set[str]] = []
        df: Counter = Counter()
        graph: dict[str, Counter] = defaultdict(Counter)
        for d in sample:
            ents = set(_entities(d.text, 48))
            doc_entities.append(ents)
            for e in ents:
                df[e] += 1
            ent_list = list(ents)[:48]
            for i, a in enumerate(ent_list):
                for b in ent_list[i+1:i+9]:
                    graph[a][b] += 1; graph[b][a] += 1
        n = max(1, len(sample))
        idf = {e: math.log(1 + n / (1 + c)) for e, c in df.items()}
        return cls(sample, doc_entities, dict(graph), idf)

    def expand_terms(self, terms: list[str], limit: int = 24) -> list[str]:
        scores: Counter = Counter()
        for t in terms:
            scores[t] += 2.0
            for nb, w in graph_most_common(self.graph.get(t.lower(), {}), 8):
                scores[nb] += min(1.5, 0.15 * w)
        return [t for t, _ in scores.most_common(limit)]

    def search(self, query: str, k: int = 5) -> list[EvidenceSnippet]:
        q_terms = _entities(query, 24)
        if not q_terms:
            return []
        expanded = set(self.expand_terms(q_terms, 28))
        scored: list[tuple[float, int]] = []
        for i, ents in enumerate(self.doc_entities):
            if not ents:
                continue
            overlap = ents & expanded
            if not overlap:
                continue
            score = sum(self.idf.get(t, 0.5) for t in overlap) / (1 + math.log(1 + len(ents)))
            scored.append((score, i))
        scored.sort(reverse=True)
        return [EvidenceSnippet(self.docs[i].doc_id, self.docs[i].text, float(s), "graph_lite") for s, i in scored[:k]]

class GraphLiteKnowledgeBackend:
    name = "graph_lite"
    def __init__(self, docs: list[CorpusDoc]):
        self.index = GraphLiteIndex.build(docs)
    def search(self, query: str, k: int = 5) -> list[EvidenceSnippet]:
        return self.index.search(query, k)
