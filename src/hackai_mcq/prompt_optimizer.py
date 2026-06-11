from __future__ import annotations

"""Lightweight prompt/program optimizer for MCQ solvers.

Inspired by DSPy-style optimization, but kept dependency-free for Docker safety.
It tests multiple solver modes/prompt families on a labeled dev CSV and writes a
small JSON config that the runtime can load.
"""

import csv
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


@dataclass(slots=True)
class OptimizationCandidate:
    name: str
    env: dict[str, str]


@dataclass(slots=True)
class OptimizationResult:
    name: str
    accuracy: float
    seconds: float
    correct: int
    total: int
    env: dict[str, str]


DEFAULT_CANDIDATES = [
    OptimizationCandidate("direct_low_tokens", {"SOLVER_MODE": "direct", "MAX_NEW_TOKENS": "4", "USE_VERIFIER": "0", "USE_PAIRWISE_JUDGE": "0"}),
    OptimizationCandidate("adaptive_balanced", {"SOLVER_MODE": "adaptive", "MAX_NEW_TOKENS": "8", "USE_VERIFIER": "1", "USE_PAIRWISE_JUDGE": "1"}),
    OptimizationCandidate("vote_no_pairwise", {"SOLVER_MODE": "vote", "MAX_NEW_TOKENS": "8", "USE_VERIFIER": "0", "USE_PAIRWISE_JUDGE": "0"}),
    OptimizationCandidate("max_accuracy_guarded", {"SOLVER_MODE": "max_accuracy", "MAX_NEW_TOKENS": "12", "USE_VERIFIER": "1", "USE_PAIRWISE_JUDGE": "1"}),
]


def read_labels(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    labels: dict[str, str] = {}
    for i, row in enumerate(rows, 1):
        qid = str(row.get("qid") or row.get("id") or i)
        ans = str(row.get("label") or row.get("answer") or row.get("correct") or "").strip().upper()
        if ans in {"A", "B", "C", "D"}:
            labels[qid] = ans
    return labels


def score_predictions(pred_path: Path, labels: dict[str, str]) -> tuple[float, int, int]:
    with pred_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    total = 0
    correct = 0
    for row in rows:
        qid = str(row.get("qid", ""))
        gold = labels.get(qid)
        pred = str(row.get("answer", "")).strip().upper()
        if gold:
            total += 1
            correct += int(pred == gold)
    return (correct / total if total else 0.0), correct, total


def run_candidate(input_csv: Path, candidate: OptimizationCandidate, python_exe: str = sys.executable) -> OptimizationResult:
    import os, time

    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update(candidate.env)
        env["INPUT_CSV"] = str(input_csv)
        env["OUTPUT_DIR"] = str(out_dir)
        start = time.time()
        subprocess.run([python_exe, "-m", "hackai_mcq.cli"], check=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        seconds = time.time() - start
        labels = read_labels(input_csv)
        acc, correct, total = score_predictions(out_dir / "pred.csv", labels)
        return OptimizationResult(candidate.name, acc, seconds, correct, total, candidate.env)


def choose_best(results: Iterable[OptimizationResult], accuracy_weight: float = 0.92) -> OptimizationResult:
    results = list(results)
    if not results:
        raise ValueError("No optimization results")
    max_seconds = max(r.seconds for r in results) or 1.0
    def utility(r: OptimizationResult) -> float:
        speed_score = 1.0 - min(1.0, r.seconds / max_seconds)
        return accuracy_weight * r.accuracy + (1.0 - accuracy_weight) * speed_score
    return max(results, key=utility)


def optimize(input_csv: Path, output_json: Path) -> OptimizationResult:
    results = [run_candidate(input_csv, c) for c in DEFAULT_CANDIDATES]
    best = choose_best(results)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {"best": asdict(best), "results": [asdict(r) for r in results]}
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return best


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/prompt_optimizer.json"))
    args = parser.parse_args()
    best = optimize(args.input_csv, args.output)
    print(json.dumps(asdict(best), ensure_ascii=False, indent=2))
