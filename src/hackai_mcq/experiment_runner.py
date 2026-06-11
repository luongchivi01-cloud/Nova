from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Iterable

from .benchmark import main as _benchmark_main  # noqa: F401 - keeps module discoverable
from .calibration import AnswerPriorCalibrator
from .cache import PromptCache
from .config import RuntimeConfig, from_env
from .io_utils import read_items, read_labels_if_available, validate_predictions, write_predictions, write_trace
from .model_backends import create_backend
from .rag import create_rag
from .schema import RunStats
from .solver import AdaptiveSolver


def _accuracy(pred_path: Path, input_path: Path) -> tuple[float | None, int, int]:
    labels = read_labels_if_available(input_path)
    if not labels:
        return None, 0, 0
    ok, errors = validate_predictions(pred_path)
    if not ok:
        raise ValueError("Invalid pred: " + "; ".join(errors[:6]))
    correct = total = 0
    with pred_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = str(row.get("qid", ""))
            if qid in labels:
                total += 1
                correct += int(str(row.get("answer", "")).strip().upper()[:1] == labels[qid])
    return (correct / total if total else None), correct, total


def run_once(base: RuntimeConfig, mode: str, out_dir: Path, name: str, limit: int = 0) -> dict:
    cfg = replace(base, mode=mode, output_path=out_dir / f"{name}.pred.csv", trace_path=out_dir / f"{name}.trace.jsonl")
    backend = create_backend(
        cfg.backend,
        cfg.model_path,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        n_ctx=cfg.n_ctx,
        n_gpu_layers=cfg.n_gpu_layers,
        force_cpu=cfg.force_cpu,
        load_in_4bit=cfg.load_in_4bit,
        load_in_8bit=cfg.load_in_8bit,
        torch_dtype=cfg.torch_dtype,
    )
    items = read_items(cfg.input_path)
    if limit:
        items = items[:limit]
    solver = AdaptiveSolver(
        backend=backend,
        config=cfg,
        rag=create_rag(cfg.rag_corpus, mode="auto") if cfg.enable_rag else None,
        calibrator=AnswerPriorCalibrator.from_path(cfg.calibration_path, cfg.calibration_strength),
        cache=PromptCache(str(out_dir / f"{name}.cache.jsonl")),
        total_rows=len(items),
    )
    results = []
    t0 = time.time()
    for i, item in enumerate(items, start=1):
        results.append(solver.solve(item, i))
    seconds = time.time() - t0
    write_predictions(results, cfg.output_path)
    write_trace(results, cfg.trace_path)
    ok, errors = validate_predictions(cfg.output_path, expected_count=len(items))
    if not ok:
        raise RuntimeError("Output validation failed: " + "; ".join(errors[:8]))
    acc, correct, total = _accuracy(cfg.output_path, cfg.input_path)
    return {
        "name": name,
        "mode": mode,
        "backend": backend.name,
        "rows": len(items),
        "seconds": round(seconds, 3),
        "rows_per_second": round(len(items) / seconds, 3) if seconds else 0,
        "accuracy": round(acc, 6) if acc is not None else "NA",
        "correct": correct,
        "labeled_total": total,
        "pred_path": str(cfg.output_path),
        "trace_path": str(cfg.trace_path),
    }


def default_experiments() -> list[tuple[str, str]]:
    return [
        ("direct", "direct"),
        ("vote", "vote"),
        ("adaptive", "adaptive"),
        ("judge", "judge"),
        ("max_accuracy", "max_accuracy"),
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run multiple solver modes and compare accuracy/time")
    ap.add_argument("--input", required=True, help="CSV to test; labels optional but recommended")
    ap.add_argument("--out-dir", default="reports/experiments")
    ap.add_argument("--backend", default=None, choices=["auto", "heuristic", "llama_cpp", "transformers"])
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--modes", default="direct,vote,adaptive,judge,max_accuracy")
    ap.add_argument("--limit", type=int, default=0, help="Use first N rows for quick iteration")
    ap.add_argument("--enable-rag", action="store_true")
    ap.add_argument("--rag-corpus", default=None)
    args = ap.parse_args(argv)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    base = from_env()
    base = replace(
        base,
        input_path=Path(args.input),
        output_path=out_dir / "pred.csv",
        backend=args.backend or base.backend,
        model_path=args.model_path or base.model_path,
        enable_rag=args.enable_rag or base.enable_rag,
        rag_corpus=args.rag_corpus or base.rag_corpus,
        submission_strict=False,
    )
    requested = [m.strip() for m in args.modes.split(",") if m.strip()]
    rows = []
    for mode in requested:
        name = mode
        print(f"[experiment] {name}...")
        rows.append(run_once(base, mode, out_dir, name, limit=args.limit))
    with (out_dir / "experiment_results.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["name", "mode", "backend", "rows", "seconds", "rows_per_second", "accuracy", "correct", "labeled_total", "pred_path", "trace_path"]
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    (out_dir / "experiment_results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    ranked = sorted(rows, key=lambda r: ((-1 if r["accuracy"] == "NA" else float(r["accuracy"])), float(r["rows_per_second"])), reverse=True)
    print(json.dumps(ranked, ensure_ascii=False, indent=2))
    print(f"wrote {out_dir / 'experiment_results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
