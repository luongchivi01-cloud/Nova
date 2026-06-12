from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


LABELS = "ABCD"
RUNNER_VERSION = "2.3"
FINAL_RE = re.compile(r"(?im)^\s*Final\s*:\s*([A-D])\s*$")
OUTPUT_GRAMMARS: dict[str, Any] = {}
NEGATION_TERMS = (
    "khong", "sai", "ngoai tru", "khong dung", "khong phai",
    "not", "except", "incorrect", "false", "least",
)
CONTRADICTION_TERMS = (
    "khong", "khong phai", "sai", "trai lai", "tuy nhien", "ngoai tru",
    "not", "never", "false", "incorrect", "however", "except",
)
STOPWORDS = {
    "va", "la", "cua", "cho", "trong", "mot", "nhung", "cac", "duoc", "voi",
    "theo", "tu", "tai", "khi", "nao", "gi", "co", "khong", "hoi", "sau", "day",
    "the", "a", "an", "of", "to", "in", "is", "are", "which", "what", "for",
}


@dataclass(slots=True)
class Profile:
    name: str
    easy_tokens: int
    hard_tokens: int
    verifier_tokens: int
    verifier_threshold: float


BALANCED = Profile("balanced", 48, 96, 64, 0.68)
FAST = Profile("fast", 32, 64, 48, 0.78)


@dataclass(slots=True)
class DeterministicResult:
    answer: str
    formula: str
    inputs: dict[str, Any]
    value: str
    matched_option: str


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").replace("\x00", " ")).strip()


def canonical(text: str) -> str:
    text = unicodedata.normalize("NFD", str(text or "").lower())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return normalize_space(text.replace("đ", "d"))


def words(text: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9]+", canonical(text))
        if len(token) > 1 and token not in STOPWORDS
    }


def numbers(text: str) -> list[float]:
    out: list[float] = []
    for raw in re.findall(r"(?<![A-Za-z])[-+]?\d+(?:[.,]\d+)?", str(text)):
        try:
            out.append(float(raw.replace(",", ".")))
        except ValueError:
            pass
    return out


def extract_final(text: str, labels: str = LABELS) -> str | None:
    matches = FINAL_RE.findall(str(text or ""))
    if not matches:
        return None
    answer = matches[-1].upper()
    return answer if answer in labels else None


def split_question_and_context(raw: str) -> tuple[str, str]:
    matches = list(re.finditer(r"(?im)^\s*(?:câu\s*hỏi|question)\s*:\s*", raw))
    if not matches:
        return normalize_space(raw), ""
    marker = matches[-1]
    question = normalize_space(raw[marker.end():])
    context = raw[:marker.start()].strip()
    return question or normalize_space(raw), context


def split_paragraphs(context: str) -> list[str]:
    paragraphs = [
        normalize_space(part)
        for part in re.split(r"\n\s*\n|^\s*(?=[IVXLCDM]+\.\s|[0-9]+(?:\.[0-9]+)*\.\s)", context, flags=re.M)
        if normalize_space(part)
    ]
    expanded: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= 900:
            expanded.append(paragraph)
            continue
        sentences = [
            normalize_space(part)
            for part in re.split(r"(?<=[.!?])\s+", paragraph)
            if normalize_space(part)
        ]
        current = ""
        for sentence in sentences:
            if current and len(current) + len(sentence) + 1 > 750:
                expanded.append(current)
                current = ""
            current = normalize_space(current + " " + sentence)
        if current:
            expanded.append(current)
    return [part for part in expanded if len(part) >= 25]


def select_relevant_context(
    context: str, question: str, choices: list[str], max_chars: int = 3600
) -> tuple[str, dict[str, Any]]:
    if not context:
        return "", {"paragraphs": 0, "selected": 0, "key_terms": []}
    paragraphs = split_paragraphs(context)
    query_words = words(question + " " + " ".join(choices))
    query_numbers = set(numbers(question + " " + " ".join(choices)))
    query_entities = {
        token for token in re.findall(r"\b[A-ZÀ-Ỹ][\wÀ-ỹ-]{2,}\b", question + " " + " ".join(choices))
    }
    scored: list[tuple[float, int]] = []
    for index, paragraph in enumerate(paragraphs):
        pwords = words(paragraph)
        overlap = len(query_words & pwords)
        entity_overlap = sum(1 for entity in query_entities if entity in paragraph)
        numeric_overlap = len(query_numbers & set(numbers(paragraph)))
        score = overlap + 2.0 * entity_overlap + 1.5 * numeric_overlap
        if index < 2:
            score += 0.25
        scored.append((score, index))

    selected_indexes: set[int] = set()
    ranked = sorted(scored, key=lambda item: (-item[0], item[1]))
    for score, index in ranked:
        if score <= 0 and selected_indexes:
            break
        selected_indexes.add(index)
        if index > 0:
            selected_indexes.add(index - 1)
        if index + 1 < len(paragraphs):
            selected_indexes.add(index + 1)
        candidate = "\n\n".join(paragraphs[i] for i in sorted(selected_indexes))
        if len(candidate) >= max_chars:
            break
    if not selected_indexes:
        selected_indexes.add(max(0, len(paragraphs) - 1))

    chunks: list[str] = []
    used = 0
    for index in sorted(selected_indexes):
        paragraph = paragraphs[index]
        remaining = max_chars - used
        if remaining <= 0:
            break
        chunks.append(paragraph[:remaining])
        used += min(len(paragraph), remaining) + 2
    return "\n\n".join(chunks)[:max_chars], {
        "paragraphs": len(paragraphs),
        "selected": len(chunks),
        "key_terms": sorted(query_words)[:24],
        "entities": sorted(query_entities)[:12],
        "numbers": sorted(query_numbers)[:12],
    }


def option_similarity(choices: list[str]) -> float:
    best = 0.0
    for index, left in enumerate(choices):
        left_words = words(left)
        for right in choices[index + 1:]:
            right_words = words(right)
            if left_words and right_words:
                best = max(best, len(left_words & right_words) / min(len(left_words), len(right_words)))
    return best


def option_evidence_matrix(question: str, choices: list[str], evidence: str) -> dict[str, dict[str, Any]]:
    qwords = words(question)
    evidence_sentences = [
        normalize_space(part)
        for part in re.split(r"\n+|(?<=[.!?])\s+", evidence)
        if normalize_space(part)
    ]
    matrix: dict[str, dict[str, Any]] = {}
    for index, choice in enumerate(choices):
        label = LABELS[index]
        cwords = words(choice)
        entities = set(re.findall(r"\b[A-ZÀ-Ỹ][\wÀ-ỹ-]{2,}\b", choice))
        cnums = set(numbers(choice))
        candidates: list[tuple[float, str, bool]] = []
        for sentence in evidence_sentences:
            swords = words(sentence)
            overlap = len(cwords & swords)
            entity_overlap = sum(1 for entity in entities if entity in sentence)
            numeric_overlap = len(cnums & set(numbers(sentence)))
            contradiction = any(term in canonical(sentence) for term in CONTRADICTION_TERMS)
            score = overlap + 1.5 * entity_overlap + 1.5 * numeric_overlap + 0.2 * len(qwords & swords)
            if score > 0:
                candidates.append((score, sentence, contradiction))
        candidates.sort(key=lambda item: item[0], reverse=True)
        support = [text for _, text, contradiction in candidates if not contradiction][:2]
        contradictions = [text for _, text, contradiction in candidates if contradiction][:2]
        matrix[label] = {
            "support_score": round(sum(item[0] for item in candidates[:2]), 3),
            "support": support,
            "contradictions": contradictions,
            "entity_overlap": sum(1 for entity in entities if entity in evidence),
            "numeric_consistency": bool(cnums and cnums & set(numbers(evidence))) if evidence else False,
            "temporal_consistency": bool(
                set(re.findall(r"\b(?:1[0-9]{3}|20[0-9]{2})\b", choice))
                & set(re.findall(r"\b(?:1[0-9]{3}|20[0-9]{2})\b", evidence))
            ),
            "negation_consistency": not (
                any(term in canonical(question) for term in NEGATION_TERMS)
                and not any(term in canonical(choice) for term in NEGATION_TERMS)
            ),
        }
    return matrix


def compact_matrix(matrix: dict[str, dict[str, Any]], max_chars: int = 1600) -> str:
    lines = ["Option evidence matrix (signals only; independently solve the question):"]
    for label in LABELS:
        row = matrix.get(label, {})
        support = " | ".join(row.get("support", [])[:1]) or "none"
        contradiction = " | ".join(row.get("contradictions", [])[:1]) or "none"
        lines.append(
            f"{label}: support={support}; contradiction={contradiction}; "
            f"entity_overlap={row.get('entity_overlap', 0)}; "
            f"numeric_consistency={row.get('numeric_consistency', False)}; "
            f"temporal_consistency={row.get('temporal_consistency', False)}"
        )
    return "\n".join(lines)[:max_chars]


def match_numeric_option(choices: list[str], value: float, tolerance: float | None = None) -> str | None:
    tolerance = tolerance if tolerance is not None else max(1e-4, abs(value) * 0.015)
    matched: list[str] = []
    for index, choice in enumerate(choices):
        values = numbers(choice)
        if any(math.isclose(candidate, value, rel_tol=0.015, abs_tol=tolerance) for candidate in values):
            matched.append(LABELS[index])
    return matched[0] if len(matched) == 1 else None


def _result(answer: str | None, formula: str, inputs: dict[str, Any], value: Any) -> DeterministicResult | None:
    if not answer:
        return None
    return DeterministicResult(answer, formula, inputs, str(value), answer)


def solve_midpoint_elasticity(question: str, choices: list[str]) -> DeterministicResult | None:
    text = canonical(question)
    if "co gian" not in text or "gia" not in text:
        return None
    pairs = re.findall(
        r"(?:gia|price)\D{0,20}(\d+(?:[.,]\d+)?)\D{0,80}(?:luong cau|quantity|q)\D{0,20}(\d+(?:[.,]\d+)?)",
        text,
    )
    if len(pairs) < 2:
        values = numbers(question)
        if len(values) != 4:
            return None
        p1, q1, p2, q2 = values
    else:
        p1, q1 = map(lambda x: float(x.replace(",", ".")), pairs[0])
        p2, q2 = map(lambda x: float(x.replace(",", ".")), pairs[1])
    if p1 == p2 or q1 + q2 == 0 or p1 + p2 == 0:
        return None
    value = abs(((q2 - q1) / ((q1 + q2) / 2)) / ((p2 - p1) / ((p1 + p2) / 2)))
    return _result(match_numeric_option(choices, value), "midpoint_price_elasticity", locals_inputs(p1=p1, q1=q1, p2=p2, q2=q2), value)


def solve_cylinder_rate(question: str, choices: list[str]) -> DeterministicResult | None:
    text = canonical(question)
    if not ("hinh tru" in text and ("toc do" in text or "rate" in text) and ("ban kinh" in text or "radius" in text)):
        return None
    rate_match = re.search(r"(?:toc do(?: khong doi)? la|rate(?: of)?|d[vV]/d[tT]\s*=)\D{0,20}(\d+(?:[.,]\d+)?)", text)
    radius_match = re.search(r"(?:ban kinh(?: cua be)? la|radius(?: is)?|r\s*=)\D{0,20}(\d+(?:[.,]\d+)?)", text)
    if not rate_match or not radius_match:
        return None
    rate = float(rate_match.group(1).replace(",", "."))
    radius = float(radius_match.group(1).replace(",", "."))
    if radius <= 0:
        return None
    value = rate / (math.pi * radius * radius)
    return _result(match_numeric_option(choices, value, tolerance=0.05), "cylinder_dh_dt=dV_dt/(pi*r^2)", locals_inputs(rate=rate, radius=radius), value)


def solve_exponential_decay(question: str, choices: list[str]) -> DeterministicResult | None:
    text = canonical(question).replace(" ", "")
    derivative = "db/dt=-kb" in text or "dbdt=-kb" in text or "frac{db}{dt}=-kb" in text
    if not (derivative and ("b_0" in text or "b0" in text)):
        return None
    matches = []
    for index, choice in enumerate(choices):
        compact = canonical(choice).replace(" ", "").replace("\\", "")
        if ("e^{-kt}" in compact or "e^(-kt)" in compact or "exp(-kt)" in compact) and ("b_0" in compact or "b0" in compact):
            matches.append(LABELS[index])
    return _result(matches[0] if len(matches) == 1 else None, "B(t)=B0*exp(-k*t)", {}, "symbolic")


def solve_resistor_equivalent(question: str, choices: list[str]) -> DeterministicResult | None:
    text = canonical(question)
    if not ("dien tro" in text and ("cat thanh hai" in text or "cut into two" in text)):
        return None
    if not ("song song" in text or "parallel" in text):
        return None
    target = 4.0
    matched: list[str] = []
    for index, choice in enumerate(choices):
        compact = canonical(choice).replace(" ", "")
        if re.search(r"(?:i'?=)?4i\b", compact):
            matched.append(LABELS[index])
    return _result(matched[0] if len(matched) == 1 else None, "two_half_resistors_parallel=>I'=4I", {}, target)


def solve_expected_distinct(question: str, choices: list[str]) -> DeterministicResult | None:
    text = canonical(question).replace(" ", "")
    if not (("bienngaunhien" in text or "randomvariable" in text) and ("giatrikhacnhau" in text or "distinct" in text)):
        return None
    exponent_match = re.search(r"n=([^$,.?]+)", text)
    exponent = exponent_match.group(1).strip("{}()") if exponent_match else "n"
    desired = canonical(f"k (1 - (1 - 1/k)^{exponent})").replace(" ", "")
    matches = []
    for index, choice in enumerate(choices):
        compact = canonical(choice).replace(" ", "").replace("\\left", "").replace("\\right", "")
        compact = compact.replace("\\frac{1}{k}", "1/k")
        if "k" in compact and "1-1/k" in compact and (
            f"^{exponent}" in compact or f"^{{{exponent}}}" in compact
        ):
            matches.append(LABELS[index])
    return _result(matches[0] if len(matches) == 1 else None, "E[distinct]=k*(1-(1-1/k)^n)", {}, desired)


def solve_hess_sum(question: str, choices: list[str]) -> DeterministicResult | None:
    text = canonical(question)
    if not ("hess" in text and ("delta h_1" in text or "h_1" in text) and ("delta h_2" in text or "h_2" in text)):
        return None
    first = re.search(r"(?:delta\s*)?h_1\s*=\s*([-+]?\d+(?:[.,]\d+)?)", text)
    second = re.search(r"(?:delta\s*)?h_2\s*=\s*([-+]?\d+(?:[.,]\d+)?)", text)
    if not first or not second:
        return None
    h1 = float(first.group(1).replace(",", "."))
    h2 = float(second.group(1).replace(",", "."))
    value = h1 + h2
    return _result(match_numeric_option(choices, value), "Hess_direct_sum=deltaH1+deltaH2", {"delta_h1": h1, "delta_h2": h2}, value)


def solve_gdp_deflator(question: str, choices: list[str]) -> DeterministicResult | None:
    text = canonical(question)
    if "gdp" not in text or "giam phat" not in text:
        return None
    nominal = re.search(r"(?:gdp danh nghia|nominal gdp)\D{0,20}(\d+(?:[.,]\d+)?)", text)
    real = re.search(r"(?:gdp thuc|real gdp)\D{0,20}(\d+(?:[.,]\d+)?)", text)
    if not nominal or not real:
        return None
    nominal_value = float(nominal.group(1).replace(",", "."))
    real_value = float(real.group(1).replace(",", "."))
    if real_value == 0:
        return None
    value = 100.0 * nominal_value / real_value
    return _result(match_numeric_option(choices, value), "GDP_deflator=nominal_GDP/real_GDP*100", locals_inputs(nominal=nominal_value, real=real_value), value)


def solve_simple_mean(question: str, choices: list[str]) -> DeterministicResult | None:
    text = canonical(question)
    match = re.search(r"(?:trung binh|mean|average)(?: cua| of)?\s*[:=]?\s*((?:[-+]?\d+(?:[.,]\d+)?(?:\s*[,;]\s*|\s+)){2,}[-+]?\d+(?:[.,]\d+)?)", text)
    if not match:
        return None
    values = numbers(match.group(1))
    if len(values) < 2:
        return None
    value = sum(values) / len(values)
    return _result(match_numeric_option(choices, value), "arithmetic_mean=sum(values)/n", {"values": values}, value)


def locals_inputs(**kwargs: Any) -> dict[str, Any]:
    return kwargs


DETERMINISTIC_SOLVERS = (
    solve_midpoint_elasticity,
    solve_cylinder_rate,
    solve_exponential_decay,
    solve_resistor_equivalent,
    solve_expected_distinct,
    solve_hess_sum,
    solve_gdp_deflator,
    solve_simple_mean,
)


def deterministic_solve(question: str, choices: list[str]) -> DeterministicResult | None:
    for solver in DETERMINISTIC_SOLVERS:
        try:
            result = solver(question, choices)
        except (ArithmeticError, ValueError, OverflowError):
            result = None
        if result:
            return result
    return None


def classify(question: str, choices: list[str], context: str) -> list[str]:
    text = canonical(question + " " + " ".join(choices))
    categories: list[str] = []
    if context:
        categories.append("long_context_reading")
    if any(term in text for term in NEGATION_TERMS):
        categories.append("negation_or_exception")
    if any(term in text for term in ("so sanh", "comparison", "khac nhau", "tot nhat", "best")):
        categories.append("comparison")
    if numbers(text) or any(term in text for term in ("tinh", "cong thuc", "xac suat", "toc do", "gdp", "co gian")):
        categories.append("quantitative")
    if any(term in text for term in ("kinh te", "gdp", "tai chinh", "cau", "cung", "chi phi")):
        categories.append("economics_finance")
    if any(term in text for term in ("phap luat", "luat", "nghi dinh", "hien phap", "hanh chinh")):
        categories.append("law_administration")
    if any(term in text for term in ("vat ly", "hoa hoc", "sinh hoc", "nang luong", "phan ung", "dien tro")):
        categories.append("scientific_reasoning")
    if any(term in text for term in ("an toan", "safety", "policy", "chinh sach")):
        categories.append("safety_policy")
    if not categories:
        categories.append("factual_or_general")
    if len(question) > 1200 or "quantitative" in categories and len(numbers(text)) >= 4:
        categories.append("multi_step")
    return sorted(set(categories))


def risk_score(
    question: str,
    choices: list[str],
    context: str,
    context_meta: dict[str, Any],
    matrix: dict[str, dict[str, Any]],
    deterministic: DeterministicResult | None,
) -> tuple[float, list[str]]:
    text = canonical(question)
    score = 0.0
    reasons: list[str] = []
    if any(term in text for term in NEGATION_TERMS):
        score += 0.32
        reasons.append("negation_or_exception")
    similarity = option_similarity(choices)
    if similarity >= 0.60:
        score += 0.30
        reasons.append(f"near_duplicate_options:{similarity:.2f}")
    elif similarity >= 0.42:
        score += 0.16
        reasons.append(f"similar_options:{similarity:.2f}")
    if context:
        score += 0.22
        reasons.append("long_context")
        if context_meta.get("selected", 0) <= 1:
            score += 0.10
            reasons.append("thin_context")
    numeric = bool(numbers(question)) and any(
        term in text for term in ("tinh", "bao nhieu", "toc do", "xac suat", "gdp", "co gian", "phuong trinh")
    )
    if numeric and deterministic is None:
        score += 0.24
        reasons.append("unresolved_quantitative")
    if len(question) > 900:
        score += 0.12
        reasons.append("multi_step_or_long_question")
    if any(row.get("contradictions") for row in matrix.values()):
        score += 0.08
        reasons.append("contradictory_evidence")
    if not context and not numeric and option_similarity(choices) < 0.42:
        score -= 0.08
    return max(0.0, min(1.0, score)), reasons


def make_prompt(
    question: str,
    choices: list[str],
    evidence: str,
    matrix_text: str,
    verifier: bool = False,
) -> str:
    role = (
        "Independently solve and verify this Vietnamese multiple-choice question."
        if verifier else
        "Solve this Vietnamese multiple-choice question."
    )
    lines = [
        role,
        "Reason briefly and independently. Use at most two short sentences.",
        "Do not guess from option wording. Check negation, numbers, and dates.",
        "End with exactly one line in the form Final: A, Final: B, Final: C, or Final: D.",
    ]
    if evidence:
        lines.extend(("Relevant passage:", evidence))
    if matrix_text:
        lines.extend(("Evidence signals:", matrix_text))
    lines.extend(("Question:", question, "Choices:"))
    for index, choice in enumerate(choices):
        lines.append(f"{LABELS[index]}. {choice}")
    if verifier:
        lines.extend(("Required verifier output:", "Final: <A/B/C/D>"))
    else:
        lines.extend(("Required output:", "Evidence: <short evidence or calculation>", "Final: <A/B/C/D>"))
    return "\n".join(lines)


def ask(llm: Any, prompt: str, max_tokens: int, labels: str, final_only: bool = False) -> tuple[str | None, str]:
    grammar_key = f"{labels}:final_only={final_only}"
    if grammar_key not in OUTPUT_GRAMMARS:
        from llama_cpp import LlamaGrammar
        alternatives = " | ".join(f'"{label}"' for label in labels)
        grammar = (
            f'root ::= "Final: " ({alternatives})'
            if final_only else
            'root ::= "Evidence: " [a-zA-Z0-9 .,;:+*/()=_%\'-]{1,120} '
            f'"\\nFinal: " ({alternatives})'
        )
        OUTPUT_GRAMMARS[grammar_key] = LlamaGrammar.from_string(grammar)
    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        grammar=OUTPUT_GRAMMARS[grammar_key],
    )
    message = response["choices"][0]["message"]
    content = str(message.get("content") or "")
    reasoning = str(message.get("reasoning_content") or "")
    raw = (reasoning + "\n" + content).strip()
    return extract_final(raw, labels), raw


def config_hash(profile: Profile, args: argparse.Namespace) -> str:
    payload = {
        "version": RUNNER_VERSION,
        "profile": asdict(profile),
        "model": Path(args.model).name,
        "n_ctx": args.n_ctx,
        "threads": args.threads,
        "max_hours": args.max_hours,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def load_dataset(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list) or not data:
        raise ValueError("Input JSON must be a non-empty list")
    qids: list[str] = []
    for index, row in enumerate(data, start=1):
        choices = row.get("choices") if isinstance(row, dict) else None
        qid = str(row.get("qid", "")).strip() if isinstance(row, dict) else ""
        if not qid or not str(row.get("question", "")).strip() or not isinstance(choices, list) or len(choices) < 2:
            raise ValueError(f"Invalid row {index}")
        qids.append(qid)
    if len(qids) != len(set(qids)):
        raise ValueError("Duplicate qids")
    return data


def load_checkpoint(path: Path, qids: list[str], expected_hash: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if [str(row.get("qid")) for row in rows] != qids[:len(rows)]:
        raise ValueError("Checkpoint qid order mismatch")
    if any(row.get("config_hash") != expected_hash for row in rows):
        raise ValueError("Checkpoint configuration hash mismatch")
    if any(row.get("answer") not in LABELS or int(row.get("model_calls", 0)) > 2 for row in rows):
        raise ValueError("Checkpoint contains invalid answer or call count")
    return rows


def append_checkpoint(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_submission(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["qid", "answer"])
        writer.writeheader()
        writer.writerows({"qid": row["qid"], "answer": row["answer"]} for row in records)
    os.replace(temporary, path)


def validate_submission(dataset: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    expected = [str(row["qid"]) for row in dataset]
    lines = path.read_text(encoding="utf-8").splitlines()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = reader.fieldnames
    actual = [str(row.get("qid", "")) for row in rows]
    answers = [str(row.get("answer", "")) for row in rows]
    checks = {
        "header_exact": fields == ["qid", "answer"],
        "row_count_exact": len(rows) == len(dataset),
        "qid_order_exact": actual == expected,
        "no_missing_qids": not (set(expected) - set(actual)),
        "no_extra_qids": not (set(actual) - set(expected)),
        "no_duplicate_qids": len(actual) == len(set(actual)),
        "answers_abcd_only": all(re.fullmatch(r"[ABCD]", answer) for answer in answers),
        "no_blank_lines": all(line.strip() for line in lines),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "rows": len(rows),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "answer_distribution": dict(sorted(Counter(answers).items())),
    }


def projected_total_seconds(records: list[dict[str, Any]], total_rows: int) -> float:
    measured = sum(float(row["seconds"]) for row in records)
    return measured / max(1, len(records)) * total_rows


def build_reports(records: list[dict[str, Any]], started: float, profile: Profile) -> tuple[dict[str, Any], dict[str, Any]]:
    route_counts = Counter(row["route"] for row in records)
    measured_seconds = sum(float(row["seconds"]) for row in records)
    category_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in records:
        for category in row["categories"]:
            category_counts[category][row["route"]] += 1
    verified = [row for row in records if row["model_calls"] == 2]
    risk_verified = [row for row in verified if not row.get("primary_invalid")]
    invalid_recovery = [row for row in verified if row.get("primary_invalid")]
    runtime = {
        "profile": asdict(profile),
        "rows": len(records),
        "total_seconds": round(measured_seconds, 3),
        "process_seconds_this_session": round(time.time() - started, 3),
        "mean_seconds_per_row": round(measured_seconds / max(1, len(records)), 3),
        "total_model_calls": sum(row["model_calls"] for row in records),
        "deterministic_rows": route_counts["deterministic"],
        "verified_rows": len(verified),
        "verifier_rate": round(len(verified) / max(1, len(records)), 4),
        "verifier_due_to_invalid_primary": len(invalid_recovery),
        "verifier_due_to_risk": len(risk_verified),
        "verifier_agreement_rate": round(
            sum(row.get("primary") == row.get("verifier") for row in risk_verified) / max(1, len(risk_verified)), 4
        ),
        "invalid_primary_outputs": sum(bool(row.get("primary_invalid")) for row in records),
        "invalid_verifier_outputs": sum(bool(row.get("verifier_invalid")) for row in records),
        "max_model_calls_per_row": max((row["model_calls"] for row in records), default=0),
        "long_context_rows": sum("long_context_reading" in row["categories"] for row in records),
        "long_context_selected_paragraphs": sum(row["context_meta"].get("selected", 0) for row in records),
    }
    summary = {
        "routes": dict(sorted(route_counts.items())),
        "categories": {
            category: dict(sorted(routes.items()))
            for category, routes in sorted(category_counts.items())
        },
        "risk_bands": dict(sorted(Counter(
            "high" if row["risk_score"] >= profile.verifier_threshold else
            "medium" if row["risk_score"] >= 0.40 else "low"
            for row in records
        ).items())),
        "deterministic_formulas": dict(sorted(Counter(
            row.get("deterministic", {}).get("formula", "")
            for row in records if row.get("deterministic")
        ).items())),
    }
    return runtime, summary


def profile_from_name(name: str) -> Profile:
    return FAST if name == "fast" else BALANCED


def main() -> None:
    parser = argparse.ArgumentParser(description="Round 1 V2 reasoning-first JSON runner")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--runtime-report", type=Path, required=True)
    parser.add_argument("--category-report", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("/models/qwen3.5-9b-q4_k_m.gguf"))
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=int(os.getenv("LLAMA_N_THREADS", "12")))
    parser.add_argument("--profile", choices=["balanced", "fast"], default="balanced")
    parser.add_argument("--max-hours", type=float, default=8.0)
    parser.add_argument("--pilot-rows", type=int, default=30)
    parser.add_argument("--stop-after", type=int, default=0, help="Stop after N rows for a pilot; 0 runs all")
    args = parser.parse_args()

    from llama_cpp import Llama

    profile = profile_from_name(args.profile)
    cfg_hash = config_hash(profile, args)
    started = time.time()
    dataset = load_dataset(args.input)
    qids = [str(row["qid"]) for row in dataset]
    records = load_checkpoint(args.checkpoint, qids, cfg_hash)
    print(f"[v2] profile={profile.name} rows={len(dataset)} resume={len(records)} cfg={cfg_hash}", flush=True)

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

    limit = min(len(dataset), args.stop_after) if args.stop_after else len(dataset)
    for index, row in enumerate(dataset[len(records):limit], start=len(records) + 1):
        row_started = time.time()
        all_choices = [normalize_space(str(choice)) for choice in row["choices"]]
        choices = all_choices[:4]
        labels = LABELS[:len(choices)]
        question, raw_context = split_question_and_context(str(row["question"]))
        evidence, context_meta = select_relevant_context(raw_context, question, choices)
        matrix = option_evidence_matrix(question, choices, evidence)
        matrix_text = compact_matrix(matrix)
        deterministic = deterministic_solve(question, choices)
        risk, risk_reasons = risk_score(question, choices, evidence, context_meta, matrix, deterministic)
        categories = classify(question, choices, evidence)
        model_calls = 0
        primary = verifier = None
        primary_raw = verifier_raw = ""
        primary_invalid = verifier_invalid = False
        token_budget = 0

        if deterministic:
            answer = deterministic.answer
            route = "deterministic"
        else:
            hard = risk >= 0.40
            token_budget = profile.hard_tokens if hard else profile.easy_tokens
            primary, primary_raw = ask(
                llm, make_prompt(question, choices, evidence, matrix_text), token_budget, labels
            )
            model_calls = 1
            primary_invalid = primary is None
            needs_verifier = primary_invalid or risk >= profile.verifier_threshold
            if needs_verifier:
                verifier, verifier_raw = ask(
                    llm,
                    make_prompt(question, choices, evidence, matrix_text, verifier=True),
                    profile.verifier_tokens,
                    labels,
                    final_only=True,
                )
                model_calls = 2
                verifier_invalid = verifier is None
            answer = verifier or primary
            if not answer:
                print(f"[v2] invalid primary raw tail={primary_raw[-800:]!r}", flush=True)
                print(f"[v2] invalid verifier raw tail={verifier_raw[-800:]!r}", flush=True)
                raise RuntimeError(f"No valid Final answer for qid={row['qid']}; strict mode forbids fallback")
            route = "verified" if model_calls == 2 else ("hard_primary" if hard else "easy_primary")

        record = {
            "qid": str(row["qid"]),
            "answer": answer,
            "route": route,
            "categories": categories,
            "risk_score": round(risk, 4),
            "risk_reasons": risk_reasons,
            "context_meta": context_meta,
            "matrix": matrix,
            "deterministic": asdict(deterministic) if deterministic else None,
            "primary": primary,
            "verifier": verifier,
            "primary_invalid": primary_invalid,
            "verifier_invalid": verifier_invalid,
            "primary_raw": primary_raw[-1200:],
            "verifier_raw": verifier_raw[-1200:],
            "model_calls": model_calls,
            "token_budget": token_budget,
            "choice_count_source": len(all_choices),
            "seconds": round(time.time() - row_started, 3),
            "config_hash": cfg_hash,
        }
        append_checkpoint(args.checkpoint, record)
        records.append(record)
        elapsed = time.time() - started
        eta = projected_total_seconds(records, len(dataset))
        print(
            f"[v2] {index}/{len(dataset)} answer={answer} route={route} risk={risk:.2f} "
            f"calls={model_calls} seconds={record['seconds']:.1f} eta_hours={eta / 3600:.2f}",
            flush=True,
        )
        if len(records) == args.pilot_rows and eta > args.max_hours * 3600:
            runtime, summary = build_reports(records, started, profile)
            runtime.update({"status": "ABORTED_ETA", "projected_total_seconds": round(eta, 3)})
            write_json(args.runtime_report, runtime)
            write_json(args.category_report, summary)
            raise SystemExit(3)

    if len(records) < len(dataset):
        runtime, summary = build_reports(records, started, profile)
        runtime.update({"status": "PILOT_COMPLETE", "projected_total_seconds": round(projected_total_seconds(records, len(dataset)), 3)})
        write_json(args.runtime_report, runtime)
        write_json(args.category_report, summary)
        print(json.dumps(runtime, ensure_ascii=False, indent=2), flush=True)
        return

    write_submission(records, args.output)
    validation = validate_submission(dataset, args.output)
    runtime, summary = build_reports(records, started, profile)
    runtime["status"] = "PASS" if runtime["total_seconds"] <= args.max_hours * 3600 else "FAIL_RUNTIME"
    write_json(args.validation_report, validation)
    write_json(args.runtime_report, runtime)
    write_json(args.category_report, summary)
    print(json.dumps({"validation": validation, "runtime": runtime}, ensure_ascii=False, indent=2), flush=True)
    if validation["status"] != "PASS" or runtime["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
