from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from .features import estimate_difficulty, has_negation, has_numeric_reasoning, option_similarity, tokenize
from .io_utils import read_items, read_labels_if_available, validate_predictions
from .multilingual_nlp_adapter import analyze_multilingual
from .schema import MCQItem, VALID_ANSWERS


def load_predictions(path: str | Path) -> dict[str, str]:
    preds: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ans = str(row.get("answer", "")).strip().upper()[:1]
            if ans in VALID_ANSWERS:
                preds[str(row.get("qid", "")).strip()] = ans
    return preds


def load_trace(path: str | Path | None) -> dict[str, dict]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    out: dict[str, dict] = {}
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                qid = str(obj.get("qid", ""))
                if qid:
                    out[qid] = obj
            except Exception:
                continue
    return out


def classify_item(item: MCQItem) -> list[str]:
    cats: list[str] = []
    q_tokens = tokenize(item.question)
    multi = analyze_multilingual(item.text_for_retrieval())
    cats.append(f"lang_{multi.language}")
    if multi.is_mixed_language:
        cats.append("mixed_language")
    if multi.primary_script not in {"latin", "unknown"}:
        cats.append(f"script_{multi.primary_script}")
    if has_negation(item):
        cats.append("negation_except_not")
    if has_numeric_reasoning(item):
        cats.append("numeric_or_calculation")
    if estimate_difficulty(item) >= 0.64:
        cats.append("high_difficulty")
    elif estimate_difficulty(item) >= 0.36:
        cats.append("medium_difficulty")
    else:
        cats.append("low_difficulty")
    if option_similarity(item) >= 0.45:
        cats.append("near_duplicate_options")
    if len(q_tokens) >= 70:
        cats.append("long_question")
    if any(len(tokenize(item.options.get(k, ""))) >= 22 for k in "ABCD"):
        cats.append("long_options")
    if not all(item.options.get(k, "").strip() for k in "ABCD"):
        cats.append("missing_option")
    for name in multi.domains:
        cats.append(f"domain_{name}")
    if multi.hard_markers:
        cats.append("hard_marker")
    return cats or ["uncategorized"]


def analyze(input_csv: str | Path, pred_csv: str | Path, trace_path: str | Path | None = None) -> dict:
    ok, errors = validate_predictions(pred_csv)
    if not ok:
        raise ValueError("Invalid predictions: " + "; ".join(errors[:10]))
    items = read_items(input_csv)
    labels = read_labels_if_available(input_csv)
    if not labels:
        raise ValueError("Input file has no label/correct_answer column. Error analysis needs a labeled dev/public set.")
    preds = load_predictions(pred_csv)
    trace = load_trace(trace_path)

    total = correct = 0
    wrong_rows: list[dict] = []
    category_total: Counter[str] = Counter()
    category_wrong: Counter[str] = Counter()
    strategy_total: Counter[str] = Counter()
    strategy_wrong: Counter[str] = Counter()
    confusion: dict[str, Counter[str]] = defaultdict(Counter)

    for item in items:
        if item.qid not in labels:
            continue
        gold = labels[item.qid]
        pred = preds.get(item.qid, "")
        if gold not in VALID_ANSWERS:
            continue
        total += 1
        cats = classify_item(item)
        for c in cats:
            category_total[c] += 1
        strategy = str(trace.get(item.qid, {}).get("strategy", "unknown"))
        strategy_total[strategy] += 1
        confusion[gold][pred or "?"] += 1
        if pred == gold:
            correct += 1
        else:
            for c in cats:
                category_wrong[c] += 1
            strategy_wrong[strategy] += 1
            wrong_rows.append({
                "qid": item.qid,
                "gold": gold,
                "pred": pred,
                "categories": ";".join(cats),
                "strategy": strategy,
                "difficulty": round(estimate_difficulty(item), 3),
                "question": item.question[:280],
                "A": item.options.get("A", "")[:160],
                "B": item.options.get("B", "")[:160],
                "C": item.options.get("C", "")[:160],
                "D": item.options.get("D", "")[:160],
            })

    cat_report = []
    for c, n in category_total.most_common():
        w = category_wrong[c]
        cat_report.append({"category": c, "total": n, "wrong": w, "error_rate": round(w / n, 4) if n else 0})
    strat_report = []
    for s, n in strategy_total.most_common():
        w = strategy_wrong[s]
        strat_report.append({"strategy": s, "total": n, "wrong": w, "error_rate": round(w / n, 4) if n else 0})

    return {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 6) if total else 0.0,
        "category_report": cat_report,
        "strategy_report": strat_report,
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "wrong_rows": wrong_rows,
    }


def write_reports(report: dict, out_dir: str | Path) -> None:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    (p / "error_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (p / "category_errors.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "total", "wrong", "error_rate"])
        writer.writeheader(); writer.writerows(report["category_report"])
    with (p / "strategy_errors.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["strategy", "total", "wrong", "error_rate"])
        writer.writeheader(); writer.writerows(report["strategy_report"])
    with (p / "wrong_examples.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["qid", "gold", "pred", "categories", "strategy", "difficulty", "question", "A", "B", "C", "D"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(report["wrong_rows"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Analyze wrong answers by error category")
    ap.add_argument("--input", required=True, help="Labeled CSV containing label/answer/correct_answer")
    ap.add_argument("--pred", required=True, help="pred.csv to analyze")
    ap.add_argument("--trace", default=None, help="Optional trace JSONL generated by TRACE_PATH")
    ap.add_argument("--out-dir", default="reports/error_analysis")
    args = ap.parse_args(argv)
    report = analyze(args.input, args.pred, args.trace)
    write_reports(report, args.out_dir)
    print(f"accuracy={report['accuracy']:.4f} correct={report['correct']} total={report['total']}")
    print(f"wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
