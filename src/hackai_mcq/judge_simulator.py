from __future__ import annotations

"""BTC-like local judge simulator.

It checks the exact official contract: read-only /data input, output/pred.csv,
qid,answer columns, row count, valid A/B/C/D, no network expectation, and optional
accuracy when labels are present.
"""

import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .io_utils import read_items, read_labels_if_available, validate_predictions
from .advanced_error_taxonomy import load_predictions


def score_predictions(input_csv: str | Path, pred_csv: str | Path) -> dict[str, object]:
    items = read_items(input_csv)
    ok, errors = validate_predictions(pred_csv, expected_count=len(items))
    report: dict[str, object] = {"format_ok": ok, "errors": errors, "rows": len(items)}
    labels = read_labels_if_available(input_csv)
    if labels:
        preds = load_predictions(pred_csv)
        correct = sum(1 for qid, gold in labels.items() if preds.get(qid) == gold)
        report.update({"labeled": len(labels), "correct": correct, "accuracy": correct / len(labels) if labels else None})
    return report


def run_local_entrypoint(input_csv: str | Path, out_dir: str | Path, *, env: dict[str, str] | None = None, timeout: int = 3600) -> dict[str, object]:
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    pred = outp / "pred.csv"
    cmd = [sys.executable, "-m", "hackai_mcq.cli", "--input", str(input_csv), "--output", str(pred)]
    e = None
    if env:
        import os
        e = os.environ.copy(); e.update(env)
    start = time.time()
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, env=e)
    seconds = time.time() - start
    rep = score_predictions(input_csv, pred) if pred.exists() else {"format_ok": False, "errors": ["pred.csv not created"]}
    rep.update({"returncode": proc.returncode, "seconds": seconds, "stdout_tail": proc.stdout[-2000:], "stderr_tail": proc.stderr[-2000:]})
    return rep


def simulate_submission(input_csv: str | Path, out_json: str | Path = "reports/judge_simulator_report.json", *, allow_heuristic: bool = True) -> dict[str, object]:
    with tempfile.TemporaryDirectory() as td:
        data_dir = Path(td) / "data"
        out_dir = Path(td) / "output"
        data_dir.mkdir(); out_dir.mkdir()
        dest = data_dir / "private_test.csv"
        shutil.copy2(input_csv, dest)
        env = {"DATA_DIR": str(data_dir), "OUTPUT_PATH": str(out_dir / "pred.csv")}
        if allow_heuristic:
            env.update({"STRICT_NO_FALLBACK": "0", "REQUIRE_MODEL": "0", "ALLOW_HEURISTIC": "1", "LLM_BACKEND": "heuristic"})
        rep = run_local_entrypoint(dest, out_dir, env=env)
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    return rep


def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("input_csv")
    p.add_argument("--out", default="reports/judge_simulator_report.json")
    p.add_argument("--strict", action="store_true", help="Do not force heuristic; use current env/model settings")
    args = p.parse_args(argv)
    if args.strict:
        rep = run_local_entrypoint(args.input_csv, Path(args.out).parent)
        Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        rep = simulate_submission(args.input_csv, args.out, allow_heuristic=True)
    print(json.dumps(rep, ensure_ascii=False, indent=2)[:4000])
    return 0 if rep.get("format_ok") and rep.get("returncode") == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
