from __future__ import annotations

"""Lightweight runtime supervisor for long private-test runs.

It records progress, detects row-level slowdowns, keeps qid/order invariants, and
writes an atomic heartbeat file.  It never changes the answer; it only protects
against silent hangs and hard-to-debug judge failures.
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from .schema import MCQItem, SolverResult, VALID_ANSWERS


@dataclass(slots=True)
class RuntimeSupervisorState:
    started_at: float
    total_rows: int
    processed_rows: int = 0
    errors: int = 0
    slow_rows: int = 0
    last_qid: str = ""
    last_strategy: str = ""
    last_confidence: float = 0.0
    avg_seconds_per_row: float = 0.0
    warnings: list[str] = field(default_factory=list)


class RuntimeSupervisor:
    def __init__(self, items: list[MCQItem], heartbeat_path: str | Path | None = None, slow_row_seconds: float = 20.0, heartbeat_every: int = 50):
        self.items = items
        self.expected_qids = [str(x.qid) for x in items]
        self.heartbeat_path = Path(heartbeat_path) if heartbeat_path else None
        self.slow_row_seconds = slow_row_seconds
        self.heartbeat_every = max(1, int(heartbeat_every or 50))
        self.state = RuntimeSupervisorState(started_at=time.time(), total_rows=len(items))
        self._seen: set[str] = set()
        self._row_start = time.time()

    def begin_row(self, item: MCQItem) -> None:
        self._row_start = time.time()
        if str(item.qid) in self._seen:
            self.state.warnings.append(f"duplicate qid during run: {item.qid}")
        self._seen.add(str(item.qid))

    def end_row(self, result: SolverResult) -> None:
        elapsed = time.time() - self._row_start
        self.state.processed_rows += 1
        if elapsed > self.slow_row_seconds:
            self.state.slow_rows += 1
            self.state.warnings.append(f"slow row qid={result.qid} seconds={elapsed:.2f}")
        if result.answer not in VALID_ANSWERS:
            self.state.errors += 1
            self.state.warnings.append(f"invalid answer during run qid={result.qid} answer={result.answer!r}")
        self.state.last_qid = str(result.qid)
        self.state.last_strategy = result.strategy
        self.state.last_confidence = float(result.confidence or 0.0)
        total_elapsed = max(1e-9, time.time() - self.state.started_at)
        self.state.avg_seconds_per_row = total_elapsed / max(1, self.state.processed_rows)
        if self.state.processed_rows % self.heartbeat_every == 0 or self.state.processed_rows == self.state.total_rows:
            self.write_heartbeat()

    def write_heartbeat(self) -> None:
        if not self.heartbeat_path:
            return
        p = self.heartbeat_path
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f".{p.name}.tmp")
        tmp.write_text(json.dumps(asdict(self.state), ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, p)

    def final_checks(self, results: Iterable[SolverResult]) -> list[str]:
        warnings = list(self.state.warnings)
        result_list = list(results)
        qids = [str(r.qid) for r in result_list]
        if qids != self.expected_qids:
            warnings.append("final qid order mismatch")
        bad = [r.qid for r in result_list if r.answer not in VALID_ANSWERS]
        if bad:
            warnings.append(f"final invalid answers: {bad[:5]}")
        if len(result_list) != len(self.items):
            warnings.append(f"final row count mismatch expected={len(self.items)} got={len(result_list)}")
        return warnings
