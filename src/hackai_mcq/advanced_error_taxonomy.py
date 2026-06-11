from __future__ import annotations

"""Detailed error taxonomy for local public/dev benchmarking."""

import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from .features import estimate_difficulty, has_negation, has_numeric_reasoning, option_similarity
from .io_utils import read_items, read_labels_if_available
from .multilingual_nlp_adapter import analyze_multilingual
from .schema import MCQItem


@dataclass(slots=True)
class Taxonomy:
    qid: str
    tags: list[str]
    language: str
    difficulty: float


def classify_item(item: MCQItem) -> Taxonomy:
    tags: list[str] = []
    sig = analyze_multilingual(item.text_for_retrieval())
    dif = estimate_difficulty(item)
    if has_negation(item):
        tags.append("negation_exception")
    if has_numeric_reasoning(item):
        tags.append("numeric_reasoning")
    if len(item.question) > 280 or sig.token_count > 90:
        tags.append("long_context")
    if option_similarity(item) > 0.45:
        tags.append("near_duplicate_options")
    if sig.is_mixed_language:
        tags.append("mixed_language")
    if sig.language not in {"vi", "en", "unknown"}:
        tags.append(f"lang_{sig.language}")
    for d in sig.domains:
        tags.append(f"domain_{d}")
    if not tags:
        tags.append("general")
    return Taxonomy(item.qid, sorted(set(tags)), sig.language, dif)


def load_predictions(path: str | Path) -> dict[str, str]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return {str(r.get("qid", "")).strip(): str(r.get("answer", "")).strip().upper()[:1] for r in csv.DictReader(f)}


def analyze_errors(input_csv: str | Path, pred_csv: str | Path, out_json: str | Path | None = None) -> dict[str, object]:
    items = read_items(input_csv)
    labels = read_labels_if_available(input_csv)
    preds = load_predictions(pred_csv)
    total = len(labels)
    correct = 0
    tag_total: Counter[str] = Counter()
    tag_wrong: Counter[str] = Counter()
    examples: dict[str, list[dict[str, str]]] = defaultdict(list)
    for item in items:
        tax = classify_item(item)
        gold = labels.get(item.qid)
        pred = preds.get(item.qid)
        if not gold:
            continue
        ok = pred == gold
        correct += int(ok)
        for tag in tax.tags:
            tag_total[tag] += 1
            if not ok:
                tag_wrong[tag] += 1
                if len(examples[tag]) < 5:
                    examples[tag].append({"qid": item.qid, "pred": pred or "", "gold": gold, "question": item.question[:220]})
    report = {
        "total_labeled": total,
        "correct": correct,
        "accuracy": correct / total if total else None,
        "tags": {
            tag: {
                "total": tag_total[tag],
                "wrong": tag_wrong[tag],
                "error_rate": tag_wrong[tag] / tag_total[tag] if tag_total[tag] else 0.0,
                "examples": examples.get(tag, []),
            }
            for tag in sorted(tag_total)
        },
    }
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("input_csv")
    p.add_argument("pred_csv")
    p.add_argument("--out", default="reports/advanced_error_taxonomy.json")
    args = p.parse_args(argv)
    rep = analyze_errors(args.input_csv, args.pred_csv, args.out)
    print(json.dumps(rep, ensure_ascii=False, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
