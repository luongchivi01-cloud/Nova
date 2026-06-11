from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from time import perf_counter
from typing import Callable, Iterable

from .schema import MCQItem, SolverResult


@dataclass(slots=True)
class PreflightProbeReport:
    ok: bool
    sampled_rows: int
    seconds: float
    median_row_seconds: float
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def run_preflight_probe(items: Iterable[MCQItem], solve_one: Callable[[MCQItem, int], SolverResult], *, max_rows: int = 3, slow_row_seconds: float = 20.0) -> PreflightProbeReport:
    sample = list(items)[: max(0, max_rows)]
    errors: list[str] = []
    warnings: list[str] = []
    durations: list[float] = []
    start = perf_counter()
    for i, item in enumerate(sample, start=1):
        row_start = perf_counter()
        try:
            res = solve_one(item, i)
            if res.answer not in {"A", "B", "C", "D"}:
                errors.append(f"preflight qid={item.qid} invalid answer={res.answer!r}")
        except Exception as e:
            errors.append(f"preflight qid={item.qid} failed: {type(e).__name__}: {e}")
        dt = perf_counter() - row_start
        durations.append(dt)
        if dt > slow_row_seconds:
            warnings.append(f"preflight qid={item.qid} slow row {dt:.2f}s > {slow_row_seconds:.2f}s")
    seconds = perf_counter() - start
    med = median(durations) if durations else 0.0
    return PreflightProbeReport(ok=not errors, sampled_rows=len(sample), seconds=seconds, median_row_seconds=med, errors=errors, warnings=warnings)
