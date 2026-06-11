from __future__ import annotations

import csv
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Iterable, Mapping, Any

from .schema import MCQItem, OPTION_COLUMNS, QID_COLUMNS, QUESTION_COLUMNS, SolverResult, VALID_ANSWERS


def _norm_header(x: str) -> str:
    return str(x).strip().lower().replace(" ", "_")


def _get_first(row: Mapping[str, Any], candidates: Iterable[str], default: str = "") -> str:
    lower = {_norm_header(k): v for k, v in row.items()}
    for c in candidates:
        if c in row and row[c] is not None:
            return str(row[c]).strip()
        val = lower.get(_norm_header(c))
        if val is not None:
            return str(val).strip()
    return default


def _split_combined_options(combined: str) -> dict[str, str]:
    opts: dict[str, str] = {}
    if not combined:
        return opts
    # Supports "A. ... B. ...", "A) ...", "A: ...", each on same line or newlines.
    matches = list(re.finditer(r"(?im)(?:^|\n|\r|\s)([ABCD])\s*[\.\):：\-]\s*", combined))
    for i, m in enumerate(matches):
        label = m.group(1).upper()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(combined)
        text = combined[start:end].strip(" \n\r\t;|")
        if text:
            opts[label] = text
    return opts


def _normalize_options(row: Mapping[str, Any]) -> dict[str, str]:
    opts: dict[str, str] = {}
    for label, names in OPTION_COLUMNS.items():
        opts[label] = _get_first(row, names, "")
    combined = _get_first(row, ("options", "choices", "answers", "dap_an", "đáp_án", "lua_chon", "lựa_chọn"), "")
    if combined and not all(opts.values()):
        parsed = _split_combined_options(combined)
        for k, v in parsed.items():
            if not opts.get(k):
                opts[k] = v
    for k in "ABCD":
        opts.setdefault(k, "")
    return opts


def read_items(path: str | Path) -> list[MCQItem]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input CSV not found: {p}")
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(8192)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(f, dialect=dialect)
        if not reader.fieldnames:
            raise ValueError(f"Input CSV has no header: {p}")
        items: list[MCQItem] = []
        for idx, row in enumerate(reader, start=1):
            qid = _get_first(row, QID_COLUMNS, str(idx)) or str(idx)
            question = _get_first(row, QUESTION_COLUMNS, "")
            opts = _normalize_options(row)
            if not question:
                option_names = {_norm_header(n) for names in OPTION_COLUMNS.values() for n in names}
                option_names.update(_norm_header(c) for c in QID_COLUMNS)
                question = " ".join(str(v).strip() for k, v in row.items() if _norm_header(k) not in option_names and v)
            items.append(MCQItem(qid=str(qid), question=question.strip(), options=opts, raw=dict(row)))
    return items


def write_predictions(results: Iterable[SolverResult], path: str | Path) -> None:
    """Atomically write the official pred.csv file.

    The judge should never see a half-written file if the process is killed while
    writing.  We write to a temp file in the same directory, fsync it, then
    replace the target path atomically.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{p.name}.", suffix=".tmp", dir=str(p.parent), text=True)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["qid", "answer"])
            writer.writeheader()
            for r in results:
                ans = (r.answer or "A").strip().upper()[:1]
                if ans not in VALID_ANSWERS:
                    ans = "A"
                writer.writerow({"qid": r.qid, "answer": ans})
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def validate_predictions(path: str | Path, expected_count: int | None = None) -> tuple[bool, list[str]]:
    errors: list[str] = []
    p = Path(path)
    if not p.exists():
        return False, [f"Missing output file: {p}"]
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["qid", "answer"]:
            errors.append(f"Invalid columns: {reader.fieldnames}, expected ['qid', 'answer']")
        count = 0
        seen: set[str] = set()
        for i, row in enumerate(reader, start=2):
            count += 1
            qid = (row.get("qid") or "").strip()
            ans = (row.get("answer") or "").strip()
            if not qid:
                errors.append(f"line {i}: missing qid")
            if qid in seen:
                errors.append(f"line {i}: duplicated qid {qid}")
            seen.add(qid)
            if ans not in VALID_ANSWERS:
                errors.append(f"line {i}: invalid answer {ans!r}")
        if expected_count is not None and count != expected_count:
            errors.append(f"Expected {expected_count} rows, got {count}")
    return not errors, errors


def write_trace(results: Iterable[SolverResult], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps({
                "qid": r.qid,
                "answer": r.answer,
                "confidence": r.confidence,
                "strategy": r.strategy,
                "votes": r.votes,
                "scores": r.scores,
                "notes": r.notes,
            }, ensure_ascii=False) + "\n")


def read_labels_if_available(path: str | Path) -> dict[str, str]:
    """For local benchmark only. Official private file should not contain labels."""
    p = Path(path)
    labels: dict[str, str] = {}
    if not p.exists():
        return labels
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return labels
        possible = [c for c in reader.fieldnames if _norm_header(c) in {"label", "answer", "correct", "correct_answer", "dap_an_dung", "đáp_án_đúng"}]
        if not possible:
            return labels
        label_col = possible[0]
        for idx, row in enumerate(reader, start=1):
            qid = _get_first(row, QID_COLUMNS, str(idx)) or str(idx)
            ans = str(row.get(label_col, "")).strip().upper()[:1]
            if ans in VALID_ANSWERS:
                labels[str(qid)] = ans
    return labels
