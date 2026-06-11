from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from llama_cpp import Llama, LlamaGrammar


LABELS = "ABCD"
QUESTION_MARKERS = (
    r"(?i)\bcâu\s*hỏi\s*:",
    r"(?i)\bquestion\s*:",
)
NEGATION_RE = re.compile(
    r"(?i)\b(không|sai|ngoại trừ|không đúng|không phải|not|except|incorrect|false|least)\b"
)
WORD_RE = re.compile(r"[0-9A-Za-zÀ-ỹ]+", re.UNICODE)


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").replace("\x00", " ")).strip()


def canonical(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", normalize_space(text).lower())
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def tokens(text: str) -> set[str]:
    return {canonical(x) for x in WORD_RE.findall(text) if len(x) > 1}


def split_question_and_context(raw: str) -> tuple[str, str]:
    best: re.Match[str] | None = None
    for pattern in QUESTION_MARKERS:
        matches = list(re.finditer(pattern, raw))
        if matches and (best is None or matches[-1].start() > best.start()):
            best = matches[-1]
    if best is None:
        return normalize_space(raw), ""
    question = normalize_space(raw[best.end():])
    context = raw[:best.start()].strip()
    return question or normalize_space(raw), context


def relevant_context(context: str, question: str, choices: list[str], max_chars: int = 2400) -> str:
    if not context:
        return ""
    context = context.replace("\r", "\n")
    chunks = [normalize_space(x) for x in re.split(r"\n\s*\n|\n|(?<=[.!?])\s+", context)]
    chunks = [x for x in chunks if len(x) >= 30]
    if not chunks:
        return normalize_space(context)[-max_chars:]

    query = tokens(question + " " + " ".join(choices))
    scored: list[tuple[float, int, str]] = []
    for index, chunk in enumerate(chunks):
        overlap = len(tokens(chunk) & query)
        title_bonus = 0.5 if index < 2 else 0.0
        scored.append((overlap + title_bonus, index, chunk))

    selected: list[tuple[int, str]] = []
    used = 0
    for _, index, chunk in sorted(scored, key=lambda x: (-x[0], x[1])):
        if used >= max_chars:
            break
        piece = chunk[: max_chars - used]
        if piece:
            selected.append((index, piece))
            used += len(piece) + 1
    return "\n".join(text for _, text in sorted(selected))[:max_chars]


def option_similarity(choices: list[str]) -> float:
    best = 0.0
    for i, left in enumerate(choices):
        lt = tokens(left)
        for right in choices[i + 1:]:
            rt = tokens(right)
            if lt and rt:
                best = max(best, len(lt & rt) / max(1, min(len(lt), len(rt))))
    return best


def is_high_risk(question: str, choices: list[str]) -> bool:
    return bool(NEGATION_RE.search(question) or option_similarity(choices) >= 0.60)


def make_prompt(question: str, choices: list[str], context: str = "", candidate: str = "") -> str:
    lines = [
        "Solve this Vietnamese multiple-choice question.",
        "Reply with exactly one allowed answer letter and no explanation.",
    ]
    if context:
        lines.extend(("Relevant passage:", context))
    lines.extend(("Question:", question))
    lines.append("Choices:")
    for index, choice in enumerate(choices):
        lines.append(f"{LABELS[index]}. {choice}")
    if candidate:
        lines.append(f"Independently verify proposed answer {candidate}; change it if needed.")
    lines.append("Answer:")
    return "\n".join(lines)


def grammar_for(labels: str) -> LlamaGrammar:
    alternatives = " | ".join(f'"{label}"' for label in labels)
    return LlamaGrammar.from_string(f"root ::= {alternatives}")


def ask(llm: Llama, prompt: str, labels: str) -> str:
    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1,
        temperature=0.0,
        top_p=1.0,
        grammar=grammar_for(labels),
    )
    answer = str(response["choices"][0]["message"]["content"]).strip().upper()[:1]
    if answer not in labels:
        raise RuntimeError(f"Model returned invalid answer {answer!r}; allowed={labels}")
    return answer


def load_dataset(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list) or not data:
        raise ValueError("Public test JSON must be a non-empty list")
    qids: list[str] = []
    for index, row in enumerate(data, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Row {index} is not an object")
        qid = str(row.get("qid", "")).strip()
        question = str(row.get("question", "")).strip()
        choices = row.get("choices")
        if not qid or not question or not isinstance(choices, list) or len(choices) < 2:
            raise ValueError(f"Invalid row {index}: qid/question/choices")
        qids.append(qid)
    if len(qids) != len(set(qids)):
        raise ValueError("Dataset contains duplicate qids")
    return data


def load_checkpoint(path: Path, expected_qids: list[str]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if [str(row.get("qid")) for row in rows] != expected_qids[: len(rows)]:
        raise ValueError("Checkpoint qid order does not match public test")
    if any(str(row.get("answer")) not in LABELS for row in rows):
        raise ValueError("Checkpoint contains invalid answer")
    return rows


def append_checkpoint(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_submission(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["qid", "answer"])
        writer.writeheader()
        writer.writerows({"qid": row["qid"], "answer": row["answer"]} for row in records)
    os.replace(tmp, path)


def validate_submission(dataset: list[dict[str, Any]], submission: Path) -> dict[str, Any]:
    expected = [str(row["qid"]) for row in dataset]
    raw_lines = submission.read_text(encoding="utf-8").splitlines()
    with submission.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = reader.fieldnames
    actual = [str(row.get("qid", "")) for row in rows]
    answers = [str(row.get("answer", "")) for row in rows]
    checks = {
        "header_exact": fields == ["qid", "answer"],
        "row_count_exact": len(rows) == len(dataset) == 463,
        "qid_order_exact": actual == expected,
        "no_missing_qids": not (set(expected) - set(actual)),
        "no_extra_qids": not (set(actual) - set(expected)),
        "no_duplicate_qids": len(actual) == len(set(actual)),
        "answers_abcd_only": all(re.fullmatch(r"[ABCD]", answer) for answer in answers),
        "no_blank_lines": all(line.strip() for line in raw_lines),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "rows": len(rows),
        "sha256": hashlib.sha256(submission.read_bytes()).hexdigest(),
        "answer_distribution": dict(sorted(Counter(answers).items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("/models/qwen3.5-9b-q4_k_m.gguf"))
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=int(os.getenv("LLAMA_N_THREADS", "12")))
    args = parser.parse_args()

    started = time.time()
    dataset = load_dataset(args.input)
    expected_qids = [str(row["qid"]) for row in dataset]
    records = load_checkpoint(args.checkpoint, expected_qids)
    print(f"[round1] rows={len(dataset)} resume={len(records)}", flush=True)

    llm = Llama(
        model_path=str(args.model),
        n_ctx=args.n_ctx,
        n_gpu_layers=0,
        n_batch=512,
        n_ubatch=256,
        n_threads=args.threads,
        n_threads_batch=args.threads,
        verbose=False,
        seed=42,
    )

    for index, row in enumerate(dataset[len(records):], start=len(records) + 1):
        row_started = time.time()
        all_choices = [normalize_space(str(choice)) for choice in row["choices"]]
        choices = all_choices[:4]
        labels = LABELS[: len(choices)]
        question, raw_context = split_question_and_context(str(row["question"]))
        context = relevant_context(raw_context, question, choices)
        high_risk = is_high_risk(question, choices)
        direct = ask(llm, make_prompt(question, choices, context), labels)
        answer = direct
        calls = 1
        if high_risk:
            answer = ask(llm, make_prompt(question, choices, context, direct), labels)
            calls = 2
        record = {
            "qid": str(row["qid"]),
            "answer": answer,
            "direct": direct,
            "high_risk": high_risk,
            "model_calls": calls,
            "choice_count_source": len(all_choices),
            "seconds": round(time.time() - row_started, 3),
        }
        append_checkpoint(args.checkpoint, record)
        records.append(record)
        elapsed = time.time() - started
        print(
            f"[round1] {index}/{len(dataset)} qid={record['qid']} answer={answer} "
            f"calls={calls} seconds={record['seconds']:.1f} elapsed={elapsed / 60:.1f}m",
            flush=True,
        )

    write_submission(records, args.output)
    report = validate_submission(dataset, args.output)
    report.update(
        {
            "input": str(args.input),
            "output": str(args.output),
            "checkpoint": str(args.checkpoint),
            "total_seconds": round(time.time() - started, 3),
            "verifier_calls": sum(1 for row in records if row["model_calls"] == 2),
            "total_model_calls": sum(int(row["model_calls"]) for row in records),
            "source_choice_count_distribution": dict(
                sorted(Counter(len(row["choices"]) for row in dataset).items())
            ),
        }
    )
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
