from __future__ import annotations

"""Runtime controller that spends extra inference only when useful.

Round 2 includes inference time, so max-accuracy logic must not blindly run every
expensive subsolver on all 2000 rows. This controller estimates remaining time
and downgrades/upgrade strategy row-by-row.
"""

from dataclasses import dataclass


@dataclass(slots=True)
class TimeBudgetDecision:
    mode: str
    time_pressure: bool
    projected_total: float
    seconds_per_row: float
    notes: str


@dataclass
class TimeBudgetController:
    total_rows: int = 0
    budget_seconds: float = 0.0
    min_rows_before_pressure: int = 25

    def decide(self, configured: str, base_mode: str, elapsed: float, done: int, difficulty: float) -> TimeBudgetDecision:
        configured = (configured or "adaptive").lower()
        if not self.budget_seconds or not self.total_rows or done <= 0:
            return TimeBudgetDecision(base_mode, False, 0.0, 0.0, "no budget")
        sec_per = elapsed / max(1, done)
        remaining = max(0, self.total_rows - done)
        projected = elapsed + sec_per * remaining
        pressure = done >= self.min_rows_before_pressure and projected > self.budget_seconds * 0.90
        severe = done >= self.min_rows_before_pressure and projected > self.budget_seconds * 1.02
        mode = base_mode
        if severe:
            mode = "direct" if difficulty < 0.75 else "vote"
        elif pressure and base_mode == "judge" and difficulty < 0.82:
            mode = "vote"
        elif pressure and base_mode == "max_accuracy":
            mode = "judge" if difficulty >= 0.70 else "vote"
        return TimeBudgetDecision(mode, pressure or severe, projected, sec_per, f"projected={projected:.1f}/{self.budget_seconds:.1f};sec_per={sec_per:.3f}")
