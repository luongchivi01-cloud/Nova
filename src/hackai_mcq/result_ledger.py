from __future__ import annotations

"""Durable per-row ledger for debugging long private-test runs.

The official judge only consumes pred.csv, but a JSONL ledger makes crashes and
slow rows diagnosable without changing the required output format.
"""

import json
import os
from pathlib import Path

from .schema import MCQItem, SolverResult


class ResultLedger:
    def __init__(self, path: str | Path | None, *, enabled: bool = True, fsync_every: int = 25) -> None:
        self.path = Path(path) if path else None
        self.enabled = enabled and self.path is not None
        self.fsync_every = max(1, int(fsync_every))
        self._count = 0
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("", encoding="utf-8")

    def append(self, index: int, item: MCQItem, result: SolverResult, elapsed: float) -> None:
        if not self.enabled or self.path is None:
            return
        payload = {
            "index": index,
            "qid": item.qid,
            "answer": result.answer,
            "confidence": result.confidence,
            "strategy": result.strategy,
            "elapsed_seconds": round(elapsed, 4),
            "notes": (result.notes or "")[:1200],
        }
        self._count += 1
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            f.flush()
            if self._count % self.fsync_every == 0:
                os.fsync(f.fileno())
