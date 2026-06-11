from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from .io_utils import _get_first, _norm_header
from .schema import QID_COLUMNS, VALID_ANSWERS


@dataclass(slots=True)
class AnswerPriorCalibrator:
    """Tiny local calibrator for legal dev/public sets that contain labels.

    Official private files should not contain labels. This module is only used
    when the user explicitly mounts a calibration file. It never reads private
    answers from the test input.
    """

    priors: dict[str, float] = field(default_factory=lambda: {k: 0.25 for k in "ABCD"})
    strength: float = 0.06

    @classmethod
    def from_path(cls, path: str | Path | None, strength: float = 0.06) -> "AnswerPriorCalibrator | None":
        if not path:
            return None
        p = Path(path)
        if not p.exists():
            return None
        counts = {k: 1.0 for k in "ABCD"}  # smoothing
        try:
            if p.suffix.lower() == ".json":
                obj = json.loads(p.read_text(encoding="utf-8"))
                raw = obj.get("priors", obj) if isinstance(obj, dict) else {}
                for k in "ABCD":
                    if k in raw:
                        counts[k] += float(raw[k])
            else:
                with p.open("r", encoding="utf-8-sig", newline="") as f:
                    reader = csv.DictReader(f)
                    if not reader.fieldnames:
                        return None
                    label_cols = [c for c in reader.fieldnames if _norm_header(c) in {"label", "answer", "correct", "correct_answer", "dap_an_dung", "đap_an_đung", "đáp_án_đúng"}]
                    if not label_cols:
                        return None
                    col = label_cols[0]
                    for row in reader:
                        ans = str(row.get(col, "")).strip().upper()[:1]
                        if ans in VALID_ANSWERS:
                            counts[ans] += 1.0
        except Exception:
            return None
        s = sum(counts.values()) or 1.0
        return cls({k: counts[k] / s for k in "ABCD"}, strength=strength)

    def apply(self, scores: Mapping[str, float]) -> dict[str, float]:
        if not scores:
            return dict(self.priors)
        out = {k: max(0.0, float(scores.get(k, 0.0))) for k in "ABCD"}
        s = sum(out.values()) or 1.0
        out = {k: out[k] / s for k in "ABCD"}
        for k in "ABCD":
            out[k] = (1.0 - self.strength) * out[k] + self.strength * self.priors.get(k, 0.25)
        z = sum(out.values()) or 1.0
        return {k: out[k] / z for k in "ABCD"}
