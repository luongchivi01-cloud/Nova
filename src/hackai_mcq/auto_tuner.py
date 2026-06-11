from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_experiments(path: str | Path) -> list[dict]:
    p = Path(path)
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def choose_best(rows: list[dict], min_rps: float = 0.0, speed_weight: float = 0.03) -> dict:
    candidates = []
    for r in rows:
        try:
            acc = None if r.get("accuracy") in {"", "NA", None} else float(r["accuracy"])
            rps = float(r.get("rows_per_second") or 0)
        except Exception:
            continue
        if rps < min_rps:
            continue
        # Accuracy dominates; speed breaks ties. If labels do not exist, choose speed among valid modes.
        score = (acc if acc is not None else 0.0) + min(rps, 50.0) * speed_weight / 50.0
        candidates.append((score, acc if acc is not None else -1, rps, r))
    if not candidates:
        raise ValueError("No valid experiment rows to tune from")
    candidates.sort(reverse=True, key=lambda x: (x[0], x[1], x[2]))
    return candidates[0][3]


def build_env_recommendation(best: dict) -> dict:
    mode = best.get("mode", "adaptive")
    rec = {
        "SOLVER_MODE": mode,
        "USE_TOKEN_SCORING": "1",
        "MAX_NEW_TOKENS": "8" if mode in {"direct", "adaptive"} else "12",
        "MAX_RETRIES": "1",
        "CONFIDENCE_THRESHOLD": "0.62",
        "UNCERTAIN_MARGIN": "0.08",
    }
    if mode == "max_accuracy":
        rec.update({"USE_VERIFIER": "1", "USE_PAIRWISE_JUDGE": "1", "MAX_PAIRWISE_CALLS": "3", "MAX_NEW_TOKENS": "12"})
    elif mode == "direct":
        rec.update({"USE_VERIFIER": "0", "USE_PAIRWISE_JUDGE": "0", "MAX_PAIRWISE_CALLS": "0"})
    else:
        rec.update({"USE_VERIFIER": "1", "USE_PAIRWISE_JUDGE": "1", "MAX_PAIRWISE_CALLS": "2"})
    return rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Choose best solver mode from experiment_results.csv")
    ap.add_argument("--experiments", required=True)
    ap.add_argument("--out", default="reports/tuned_config.json")
    ap.add_argument("--min-rps", type=float, default=0.0)
    args = ap.parse_args(argv)
    rows = load_experiments(args.experiments)
    best = choose_best(rows, min_rps=args.min_rps)
    payload = {"best_experiment": best, "recommended_env": build_env_recommendation(best)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
