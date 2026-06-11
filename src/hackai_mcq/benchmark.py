from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .io_utils import read_labels_if_available, validate_predictions


def main() -> int:
    ap = argparse.ArgumentParser(description="Local benchmark helper if labels exist")
    ap.add_argument("--input", required=True)
    ap.add_argument("--pred", required=True)
    args = ap.parse_args()

    ok, errors = validate_predictions(args.pred)
    if not ok:
        print("\n".join(errors))
        return 2

    labels = read_labels_if_available(args.input)
    if not labels:
        print("No labels found in input; only format validation is available.")
        return 0

    correct = 0
    total = 0
    with Path(args.pred).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = str(row["qid"])
            if qid in labels:
                total += 1
                correct += int(row["answer"] == labels[qid])
    if total == 0:
        print("No overlapping labels/qids.")
        return 1
    print(f"accuracy={correct/total:.4f} correct={correct} total={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
