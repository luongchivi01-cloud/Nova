from __future__ import annotations

"""Export HackAIthon C CSV into evaluation-harness friendly JSONL.

The official submission path remains /data -> /output/pred.csv. This module is
for research: it lets you reuse evaluation frameworks without changing the
competition runner.
"""

import csv
import json
from pathlib import Path


def csv_to_mcq_jsonl(input_csv: Path, output_jsonl: Path) -> int:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with input_csv.open("r", encoding="utf-8-sig", newline="") as f, output_jsonl.open("w", encoding="utf-8", newline="") as out:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, 1):
            qid = str(row.get("qid") or row.get("id") or i)
            question = str(row.get("question") or row.get("cau_hoi") or row.get("câu_hỏi") or "")
            choices = [str(row.get(k) or row.get(k.lower()) or row.get(f"option_{k.lower()}") or "") for k in "ABCD"]
            label = str(row.get("label") or row.get("answer") or row.get("correct") or "").strip().upper()
            rec = {"id": qid, "question": question, "choices": choices}
            if label in {"A", "B", "C", "D"}:
                rec["answer"] = "ABCD".index(label)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_lm_eval_yaml(task_name: str, dataset_path: Path, output_yaml: Path) -> None:
    """Write a minimal YAML sketch for lm-evaluation-harness custom task.

    Users may need to adapt paths to their local harness version. We keep it as
    a generated artifact, not a hard dependency.
    """
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    output_yaml.write_text(
        f"""task: {task_name}\ndataset_path: json\ndataset_kwargs:\n  data_files:\n    validation: {dataset_path.as_posix()}\ntraining_split: null\nvalidation_split: validation\noutput_type: multiple_choice\ndoc_to_text: \"{{{{question}}}}\\nA. {{{{choices[0]}}}}\\nB. {{{{choices[1]}}}}\\nC. {{{{choices[2]}}}}\\nD. {{{{choices[3]}}}}\\nAnswer:\"\ndoc_to_choice: [\"A\", \"B\", \"C\", \"D\"]\ndoc_to_target: answer\nmetric_list:\n  - metric: acc\n    aggregation: mean\n    higher_is_better: true\n""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("input_csv", type=Path)
    p.add_argument("--jsonl", type=Path, default=Path("reports/hackai_mcq.jsonl"))
    p.add_argument("--yaml", type=Path, default=Path("reports/hackai_mcq_lm_eval.yaml"))
    p.add_argument("--task-name", default="hackai_mcq")
    args = p.parse_args()
    n = csv_to_mcq_jsonl(args.input_csv, args.jsonl)
    write_lm_eval_yaml(args.task_name, args.jsonl, args.yaml)
    print(f"exported {n} rows to {args.jsonl} and {args.yaml}")
