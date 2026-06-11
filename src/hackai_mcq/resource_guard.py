from __future__ import annotations

"""Lightweight resource and progress guard for stable judge execution."""

import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(slots=True)
class ResourceSnapshot:
    timestamp: float
    rows_done: int
    elapsed_seconds: float
    output_free_mb: float
    max_rss_mb: float | None = None


@dataclass(slots=True)
class ResourceReport:
    ok: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    snapshots: list[ResourceSnapshot] = field(default_factory=list)


class ResourceGuard:
    def __init__(self, output_dir: str | Path, *, report_path: str | Path | None = None, min_free_mb: float = 64.0, sample_every: int = 100) -> None:
        self.output_dir = Path(output_dir)
        self.report_path = Path(report_path) if report_path else None
        self.min_free_mb = min_free_mb
        self.sample_every = max(1, int(sample_every))
        self.start = time.time()
        self.report = ResourceReport(ok=True)

    def _max_rss_mb(self) -> float | None:
        try:
            import resource  # type: ignore
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # Linux reports KB, macOS reports bytes. Docker judge is Linux, but keep safe.
            return float(rss) / 1024.0 if rss > 10_000 else float(rss) / (1024.0 * 1024.0)
        except Exception:
            return None

    def snapshot(self, rows_done: int, *, force: bool = False) -> None:
        if not force and rows_done % self.sample_every != 0:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(self.output_dir)
        free_mb = usage.free / (1024 * 1024)
        snap = ResourceSnapshot(
            timestamp=time.time(),
            rows_done=rows_done,
            elapsed_seconds=time.time() - self.start,
            output_free_mb=free_mb,
            max_rss_mb=self._max_rss_mb(),
        )
        self.report.snapshots.append(snap)
        if free_mb < self.min_free_mb:
            self.report.errors.append(f"low_output_disk_space_mb={free_mb:.1f}")
            self.report.ok = False

    def write_report(self) -> ResourceReport:
        if self.report_path:
            self.report_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.report_path.with_name(f".{self.report_path.name}.tmp")
            tmp.write_text(json.dumps(asdict(self.report), ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, self.report_path)
        return self.report
