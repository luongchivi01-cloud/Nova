from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .schema import SolverResult


@dataclass(slots=True)
class OutputWatchdog:
    window: int = 120
    collapse_threshold: float = 0.82
    answers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def observe(self, result: SolverResult, row_index: int) -> None:
        ans = (result.answer or "").strip().upper()[:1]
        if ans not in {"A", "B", "C", "D"}:
            self.warnings.append(f"row={row_index} invalid observed answer={result.answer!r}")
            return
        self.answers.append(ans)
        if len(self.answers) > self.window:
            self.answers = self.answers[-self.window:]
        if len(self.answers) >= min(40, self.window):
            counts = Counter(self.answers)
            most_ans, most_count = counts.most_common(1)[0]
            ratio = most_count / len(self.answers)
            if ratio >= self.collapse_threshold:
                msg = f"answer distribution collapse risk: {most_ans}={ratio:.2%} over last {len(self.answers)} rows"
                if not self.warnings or self.warnings[-1] != msg:
                    self.warnings.append(msg)

    def summary(self) -> dict[str, object]:
        counts = Counter(self.answers)
        return {"window": self.window, "counts": dict(counts), "warnings": list(self.warnings[-20:])}
