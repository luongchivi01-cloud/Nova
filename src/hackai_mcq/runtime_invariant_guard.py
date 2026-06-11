from __future__ import annotations

"""Strict runtime invariants for official HackAIthon runs.

This guard catches subtle production failures that normal unit tests miss:
wrong qid ordering, invalid answers, confidence NaN/out-of-range, repeated qids,
extreme answer collapse, and too many low-confidence rows.  It is intentionally
model-agnostic: it does not decide answers, it verifies that the whole pipeline
is stable enough for the judge to consume.
"""

import json
import math
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from .schema import MCQItem, SolverResult, VALID_ANSWERS


@dataclass(slots=True)
class InvariantIssue:
    kind: str
    qid: str
    detail: str
    required: bool = True


@dataclass(slots=True)
class RuntimeInvariantReport:
    ok: bool
    timestamp: float
    rows_expected: int
    rows_seen: int = 0
    answer_distribution: dict[str, int] = field(default_factory=dict)
    strategy_distribution: dict[str, int] = field(default_factory=dict)
    low_confidence_rows: int = 0
    duplicate_results: int = 0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class RuntimeInvariantGuard:
    def __init__(
        self,
        items: list[MCQItem],
        *,
        strict: bool = True,
        low_confidence_threshold: float = 0.38,
        collapse_ratio: float = 0.92,
        report_path: str | Path | None = None,
    ) -> None:
        self.items = items
        self.strict = strict
        self.low_confidence_threshold = low_confidence_threshold
        self.collapse_ratio = collapse_ratio
        self.report_path = Path(report_path) if report_path else None
        self.expected_qids = [str(x.qid) for x in items]
        self.seen_qids: set[str] = set()
        self.issues: list[InvariantIssue] = []
        self.answer_counts: Counter[str] = Counter()
        self.strategy_counts: Counter[str] = Counter()
        self.low_confidence_rows = 0
        self.rows_seen = 0

    def _add(self, kind: str, qid: str, detail: str, *, required: bool = True) -> None:
        self.issues.append(InvariantIssue(kind=kind, qid=str(qid), detail=detail, required=required))

    def check_result(self, item: MCQItem, result: SolverResult, index: int) -> None:
        self.rows_seen += 1
        expected_qid = str(item.qid)
        actual_qid = str(result.qid)
        if actual_qid != expected_qid:
            self._add("qid_mismatch", expected_qid, f"result qid={actual_qid!r} at row={index}")
        if actual_qid in self.seen_qids:
            self._add("duplicate_result_qid", actual_qid, f"duplicate result qid at row={index}")
        self.seen_qids.add(actual_qid)
        answer = (result.answer or "").strip().upper()
        if answer not in VALID_ANSWERS:
            self._add("invalid_answer", expected_qid, f"answer={result.answer!r}")
        else:
            self.answer_counts[answer] += 1
        strategy = (result.strategy or "unknown").strip() or "unknown"
        self.strategy_counts[strategy] += 1
        try:
            conf = float(result.confidence)
            if not math.isfinite(conf):
                self._add("bad_confidence", expected_qid, f"confidence not finite: {result.confidence!r}")
            elif conf < 0.0 or conf > 1.25:
                self._add("bad_confidence", expected_qid, f"confidence out of expected range: {conf}", required=False)
            elif conf < self.low_confidence_threshold:
                self.low_confidence_rows += 1
        except Exception:
            self._add("bad_confidence", expected_qid, f"confidence unparsable: {result.confidence!r}")
        if self.strict:
            required_errors = [x for x in self.issues if x.required]
            if required_errors:
                last = required_errors[-1]
                raise RuntimeError(f"runtime invariant failed: {last.kind} qid={last.qid} {last.detail}")

    def final_report(self) -> RuntimeInvariantReport:
        errors = [f"{i.kind}:{i.qid}:{i.detail}" for i in self.issues if i.required]
        warnings = [f"{i.kind}:{i.qid}:{i.detail}" for i in self.issues if not i.required]
        if self.rows_seen != len(self.items):
            errors.append(f"row_count_seen_mismatch expected={len(self.items)} got={self.rows_seen}")
        missing = [qid for qid in self.expected_qids if qid not in self.seen_qids]
        if missing:
            errors.append(f"missing_result_qids count={len(missing)} sample={missing[:5]}")
        if self.rows_seen >= 80 and self.answer_counts:
            top_ans, top_count = self.answer_counts.most_common(1)[0]
            ratio = top_count / max(1, self.rows_seen)
            if ratio >= self.collapse_ratio:
                warnings.append(f"answer_distribution_collapse top={top_ans} ratio={ratio:.3f}; investigate model/parser bias")
        if self.rows_seen >= 50:
            low_ratio = self.low_confidence_rows / max(1, self.rows_seen)
            if low_ratio >= 0.65:
                warnings.append(f"low_confidence_rate={low_ratio:.3f}; consider stronger model/corpus/max_accuracy mode")
        return RuntimeInvariantReport(
            ok=not errors,
            timestamp=time.time(),
            rows_expected=len(self.items),
            rows_seen=self.rows_seen,
            answer_distribution=dict(self.answer_counts),
            strategy_distribution=dict(self.strategy_counts),
            low_confidence_rows=self.low_confidence_rows,
            duplicate_results=sum(1 for i in self.issues if i.kind == "duplicate_result_qid"),
            warnings=warnings,
            errors=errors,
        )

    def write_report(self) -> RuntimeInvariantReport:
        report = self.final_report()
        if self.report_path:
            self.report_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.report_path.with_name(f".{self.report_path.name}.tmp")
            tmp.write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, self.report_path)
        return report
