from __future__ import annotations

"""Track public/dev benchmark submissions to avoid overfitting."""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean


@dataclass(slots=True)
class ScoreRecord:
    timestamp: float
    version: str
    dataset: str
    accuracy: float
    seconds: float
    notes: str = ""


class AntiOverfitTracker:
    def __init__(self, path: str | Path = "reports/score_history.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, version: str, dataset: str, accuracy: float, seconds: float, notes: str = "") -> None:
        rec = ScoreRecord(time.time(), version, dataset, float(accuracy), float(seconds), notes)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

    def records(self) -> list[ScoreRecord]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            out.append(ScoreRecord(**d))
        return out

    def warning_report(self) -> dict[str, object]:
        recs = self.records()
        by_version: dict[str, list[ScoreRecord]] = {}
        for r in recs:
            by_version.setdefault(r.version, []).append(r)
        warnings = []
        for version, rows in by_version.items():
            public = [r.accuracy for r in rows if "public" in r.dataset.lower()]
            holdout = [r.accuracy for r in rows if "holdout" in r.dataset.lower() or "adversarial" in r.dataset.lower()]
            if public and holdout and max(public) - mean(holdout) > 0.08:
                warnings.append({"version": version, "issue": "public/dev gap > 8%; possible overfit", "public_best": max(public), "holdout_mean": mean(holdout)})
        return {"records": len(recs), "warnings": warnings}


def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--path", default="reports/score_history.jsonl")
    p.add_argument("--add", action="store_true")
    p.add_argument("--version", default="dev")
    p.add_argument("--dataset", default="public")
    p.add_argument("--accuracy", type=float, default=0.0)
    p.add_argument("--seconds", type=float, default=0.0)
    p.add_argument("--notes", default="")
    args = p.parse_args(argv)
    tr = AntiOverfitTracker(args.path)
    if args.add:
        tr.append(args.version, args.dataset, args.accuracy, args.seconds, args.notes)
    print(json.dumps(tr.warning_report(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
