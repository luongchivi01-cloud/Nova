from __future__ import annotations

"""Pluggable map for future knowledge repositories.

The official Docker path stays stable even when a new repo is dropped into
third_party/. Each recipe tells the system whether it is safe for official
inference, useful only for offline indexing/experiments, or too heavy.
"""

from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True, slots=True)
class KnowledgeRepoRecipe:
    name: str
    expected_dirs: tuple[str, ...]
    official_use: str
    merge_action: str
    risk: str

RECIPES: tuple[KnowledgeRepoRecipe, ...] = (
    KnowledgeRepoRecipe("LightRAG", ("LightRAG", "lightrag"), "offline-index/graph inspiration", "add adapter to graph_lite or pre-index outside official run", "medium-heavy"),
    KnowledgeRepoRecipe("GraphRAG", ("graphrag", "GraphRAG"), "offline graph indexing only", "export corpus to graph artifacts, do not build graph during timed run", "heavy"),
    KnowledgeRepoRecipe("FlashRAG", ("FlashRAG", "flashrag"), "benchmark/research harness", "use algorithms for experiments, keep official path slim", "heavy"),
    KnowledgeRepoRecipe("txtai", ("txtai",), "semantic search optional", "use as optional dense retrieval if dependencies are installed", "medium"),
    KnowledgeRepoRecipe("WikiExtractor", ("wikiextractor", "WikiExtractor"), "corpus preparation", "convert Wikipedia XML to text/jsonl before Docker final", "low"),
)

def inventory_merge_candidates(third_party: str | Path = "third_party") -> list[dict[str, str]]:
    root = Path(third_party)
    out: list[dict[str, str]] = []
    for r in RECIPES:
        found = "no"
        for d in r.expected_dirs:
            if (root / d).exists() or any(p.name.lower().startswith(d.lower()) for p in root.glob("*")):
                found = "yes"; break
        out.append({"name": r.name, "found": found, "official_use": r.official_use, "merge_action": r.merge_action, "risk": r.risk})
    return out
