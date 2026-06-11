from __future__ import annotations

"""Official-run contracts for Round 1/2 robustness.

This layer is deliberately strict but not model-specific: it validates that the
competition runtime is self-contained, the mounted model can actually answer in
A/B/C/D format, and every row/output invariant is checked before the judge sees
pred.csv.
"""

import csv
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Protocol

from .answer_parser import parse_answer
from .prompts import constrained_choice_prompt, direct_prompt
from .schema import MCQItem, SolverResult, VALID_ANSWERS


class ProbeBackend(Protocol):
    name: str
    def generate(self, prompt: str, item: MCQItem) -> str: ...


@dataclass(slots=True)
class ContractCheck:
    name: str
    ok: bool
    required: bool = True
    detail: str = ""


@dataclass(slots=True)
class ContractReport:
    ok: bool
    timestamp: float
    checks: list[ContractCheck] = field(default_factory=list)
    runtime: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def add(self, name: str, ok: bool, required: bool = True, detail: str = "") -> None:
        self.checks.append(ContractCheck(name=name, ok=bool(ok), required=required, detail=detail))
        if required and not ok:
            self.errors.append(f"{name}: {detail}")
        elif not ok:
            self.warnings.append(f"{name}: {detail}")
        self.ok = not self.errors


def _sha256_file(path: Path, max_bytes: int = 32 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        remaining = max_bytes
        while remaining > 0:
            chunk = f.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            h.update(chunk)
            remaining -= len(chunk)
    return h.hexdigest()


def runtime_fingerprint(model_path: str | None = None) -> dict[str, str]:
    fp = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
        "cwd": os.getcwd(),
        "pid": str(os.getpid()),
    }
    if model_path:
        p = Path(model_path)
        fp["model_path"] = str(p)
        if p.exists() and p.is_file():
            fp["model_sha256_prefix"] = _sha256_file(p, max_bytes=64 * 1024 * 1024)[:16]
            fp["model_size_bytes"] = str(p.stat().st_size)
        elif p.exists() and p.is_dir():
            configs = sorted(p.rglob("config.json"))
            fp["model_dir_files"] = str(sum(1 for _ in p.rglob("*")))
            if configs:
                fp["model_config_sha256_prefix"] = _sha256_file(configs[0])[:16]
    return fp


def validate_input_contract(items: Iterable[MCQItem]) -> list[ContractCheck]:
    checks: list[ContractCheck] = []
    seen: set[str] = set()
    count = 0
    bad_qid = 0
    missing_options = 0
    weak_questions = 0
    very_long = 0
    for item in items:
        count += 1
        qid = str(item.qid).strip()
        if not qid or qid in seen:
            bad_qid += 1
        seen.add(qid)
        if not item.question.strip():
            weak_questions += 1
        if any(not item.options.get(k, "").strip() for k in "ABCD"):
            missing_options += 1
        if len(item.text_for_retrieval()) > 12000:
            very_long += 1
    checks.append(ContractCheck("input_non_empty", count > 0, True, f"rows={count}"))
    checks.append(ContractCheck("input_unique_qids", bad_qid == 0, True, f"bad_or_duplicate={bad_qid}"))
    checks.append(ContractCheck("input_options_present", missing_options == 0, False, f"rows_with_missing_options={missing_options}"))
    checks.append(ContractCheck("input_question_present", weak_questions == 0, False, f"weak_questions={weak_questions}"))
    checks.append(ContractCheck("input_context_reasonable", very_long == 0, False, f"very_long_rows={very_long}"))
    return checks


def validate_output_contract(results: Iterable[SolverResult], items: list[MCQItem]) -> list[ContractCheck]:
    checks: list[ContractCheck] = []
    result_list = list(results)
    expected_qids = [str(x.qid) for x in items]
    actual_qids = [str(x.qid) for x in result_list]
    answers_ok = all((r.answer or "").strip().upper() in VALID_ANSWERS for r in result_list)
    checks.append(ContractCheck("output_row_count_matches", len(result_list) == len(items), True, f"expected={len(items)} got={len(result_list)}"))
    checks.append(ContractCheck("output_qid_order_preserved", actual_qids == expected_qids, True, "qid order exact" if actual_qids == expected_qids else "qid order mismatch"))
    checks.append(ContractCheck("output_answers_valid", answers_ok, True, "all answers A/B/C/D" if answers_ok else "invalid answer exists"))
    return checks


def validate_pred_file_contract(path: str | Path, items: list[MCQItem]) -> list[ContractCheck]:
    p = Path(path)
    checks: list[ContractCheck] = [ContractCheck("pred_file_exists", p.exists(), True, str(p))]
    if not p.exists():
        return checks
    try:
        with p.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        cols_ok = reader.fieldnames == ["qid", "answer"]
        qids_ok = [str(r.get("qid", "")) for r in rows] == [str(x.qid) for x in items]
        answers_ok = all(str(r.get("answer", "")).strip().upper() in VALID_ANSWERS for r in rows)
        checks.append(ContractCheck("pred_columns_exact", cols_ok, True, f"columns={reader.fieldnames}"))
        checks.append(ContractCheck("pred_qids_exact", qids_ok, True, "qid order exact" if qids_ok else "qid mismatch"))
        checks.append(ContractCheck("pred_answers_valid", answers_ok, True, "answers valid" if answers_ok else "invalid answers"))
    except Exception as e:
        checks.append(ContractCheck("pred_file_readable", False, True, f"{type(e).__name__}: {e}"))
    return checks


def official_model_probe(backend: ProbeBackend, max_questions: int = 3) -> ContractReport:
    report = ContractReport(ok=True, timestamp=time.time())
    samples = [
        MCQItem("probe_en", "Which option is the letter B?", {"A": "Letter A", "B": "Letter B", "C": "Letter C", "D": "Letter D"}),
        MCQItem("probe_vi", "Đáp án nào là số 4?", {"A": "1", "B": "2", "C": "4", "D": "8"}),
        MCQItem("probe_neg", "Which option is NOT a fruit?", {"A": "Apple", "B": "Banana", "C": "Car", "D": "Orange"}),
    ][:max(1, max_questions)]
    parsed = 0
    details: list[str] = []
    for item in samples:
        try:
            raw = backend.generate(direct_prompt(item), item)
            ans = parse_answer(raw)
            if not ans:
                raw = backend.generate(constrained_choice_prompt(item), item)
                ans = parse_answer(raw)
            if ans in VALID_ANSWERS:
                parsed += 1
                details.append(f"{item.qid}:{ans}")
            else:
                details.append(f"{item.qid}:invalid({str(raw)[:40]})")
        except Exception as e:
            details.append(f"{item.qid}:error({type(e).__name__})")
    # The probe checks backend health and parsability, not semantic perfection.
    report.add("model_generates_parseable_abcd", parsed == len(samples), True, ";".join(details))
    return report


def write_contract_report(report: ContractReport, path: str | Path | None) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.tmp")
    tmp.write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)
