from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class SpeedReport:
    timestamp: float = field(default_factory=time.time)
    profile: str = "balanced"
    rows: int = 0
    seconds: float = 0.0
    direct_rows: int = 0
    deep_rows: int = 0
    fast_token_exits: int = 0
    memory_reuse_rows: int = 0
    retrieval_cache_hits: int = 0
    retrieval_cache_misses: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def rows_per_second(self) -> float:
        return round(self.rows / self.seconds, 4) if self.seconds > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["rows_per_second"] = self.rows_per_second
        return d


def write_speed_report(path: str | Path | None, report: SpeedReport) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.tmp")
    tmp.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)
