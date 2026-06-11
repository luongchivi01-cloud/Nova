from __future__ import annotations

"""Quality checks for offline corpus before official inference."""

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean
from .corpus_builder import CorpusDoc

@dataclass(slots=True)
class CorpusQualityReport:
    ok: bool
    total_docs: int
    total_chars: int
    avg_chars: float
    duplicate_ratio: float
    warnings: list[str]
    errors: list[str]
    timestamp: float = 0.0
    def write(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        data["timestamp"] = self.timestamp or time.time()
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def evaluate_corpus(docs: list[CorpusDoc], min_docs: int = 8, min_chars: int = 2000) -> CorpusQualityReport:
    warnings: list[str] = []
    errors: list[str] = []
    texts = [d.text.strip() for d in docs if d.text and d.text.strip()]
    total = sum(len(t) for t in texts)
    avg = mean([len(t) for t in texts]) if texts else 0.0
    unique = len(set(texts))
    dup_ratio = 1.0 - unique / max(1, len(texts))
    if len(texts) < min_docs:
        errors.append(f"too_few_docs: {len(texts)} < {min_docs}")
    if total < min_chars:
        errors.append(f"too_few_chars: {total} < {min_chars}")
    if dup_ratio > 0.35:
        warnings.append(f"high_duplicate_ratio: {dup_ratio:.2f}")
    if avg > 3000:
        warnings.append(f"large_avg_doc_chars: {avg:.0f}; retrieval may be less precise")
    if any("api_key" in t.lower() or "password" in t.lower() for t in texts[:200]):
        warnings.append("possible_secret_terms_detected_in_corpus_sample")
    return CorpusQualityReport(ok=not errors, total_docs=len(texts), total_chars=total, avg_chars=avg, duplicate_ratio=dup_ratio, warnings=warnings, errors=errors, timestamp=time.time())
