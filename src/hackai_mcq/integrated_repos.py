from __future__ import annotations

"""Adapters around the merged OSS repos.

These classes do not replace the competition model. They provide system-level
intelligence: evaluation exports, retrieval, output constraints, prompt-program
optimization, and Vietnamese NLP. Every adapter is optional and offline.
"""

from dataclasses import dataclass
from pathlib import Path

from .third_party_registry import detect_third_party, format_inventory, try_import, vncorenlp_home


@dataclass(slots=True)
class EnhancementReport:
    bm25s_ready: bool
    dspy_ready: bool
    outlines_ready: bool
    lm_eval_ready: bool
    vncorenlp_ready: bool
    flashrag_ready: bool
    txtai_ready: bool
    graphrag_ready: bool
    lightrag_ready: bool
    summary: str


def enhancement_report() -> EnhancementReport:
    return EnhancementReport(
        bm25s_ready=try_import("bm25s", "bm25s") is not None,
        dspy_ready=try_import("dspy", "dspy") is not None,
        outlines_ready=try_import("outlines", "outlines") is not None,
        lm_eval_ready=try_import("lm_eval", "lm_eval") is not None,
        vncorenlp_ready=vncorenlp_home() is not None,
        flashrag_ready=try_import("flashrag", "flashrag") is not None,
        txtai_ready=try_import("txtai", "txtai") is not None,
        graphrag_ready=try_import("graphrag", "graphrag") is not None,
        lightrag_ready=try_import("lightrag", "lightrag") is not None,
        summary=format_inventory(detect_third_party()),
    )


def write_enhancement_report(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    r = enhancement_report()
    p.write_text(
        "# Integrated repo readiness\n\n"
        f"- bm25s: {r.bm25s_ready}\n"
        f"- DSPy: {r.dspy_ready}\n"
        f"- Outlines: {r.outlines_ready}\n"
        f"- lm-evaluation-harness: {r.lm_eval_ready}\n"
        f"- VnCoreNLP assets: {r.vncorenlp_ready}\n"
        f"- FlashRAG repo/import: {r.flashrag_ready}\n"
        f"- txtai repo/import: {r.txtai_ready}\n"
        f"- GraphRAG repo/import: {r.graphrag_ready}\n"
        f"- LightRAG repo/import: {r.lightrag_ready}\n\n"
        "```csv\n" + r.summary + "\n```\n",
        encoding="utf-8",
    )
    return p


if __name__ == "__main__":
    print(enhancement_report().summary)
