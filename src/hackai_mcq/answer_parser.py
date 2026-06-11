from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from typing import Iterable, Mapping

from .schema import VALID_ANSWERS


JSON_PATTERNS = [
    re.compile(r"\{[^{}]*['\"]answer['\"]\s*:\s*['\"]([ABCD])['\"][^{}]*\}", re.I),
    re.compile(r"\{[^{}]*['\"]choice['\"]\s*:\s*['\"]([ABCD])['\"][^{}]*\}", re.I),
]
ANSWER_PATTERNS = [
    re.compile(r"^\s*([ABCD])\s*$", re.I),
    re.compile(r"(?:đáp\s*án|dap\s*an|answer|choice|chọn|chon|final|kết\s*quả|ket\s*qua)\s*(?:là|la|:|=|-)?\s*[\(\[]?\s*([ABCD])\s*[\)\]]?", re.I),
    re.compile(r"\b([ABCD])\s*(?:là|la)\s*(?:đúng|dung|correct)\b", re.I),
    re.compile(r"(?:option|phương\s*án|phuong\s*an)\s*([ABCD])\b", re.I),
    re.compile(r"\b([ABCD])[\.\)]\s", re.I),
]


def parse_answer(text: object) -> str | None:
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None

    # Try JSON first.
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            val = str(obj.get("answer", obj.get("choice", ""))).strip().upper()[:1]
            if val in VALID_ANSWERS:
                return val
    except Exception:
        pass

    for pat in JSON_PATTERNS + ANSWER_PATTERNS:
        m = pat.search(s)
        if m:
            ans = m.group(1).upper()
            if ans in VALID_ANSWERS:
                return ans

    # Last resort: if exactly one of A/B/C/D appears as standalone token.
    tokens = re.findall(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])", s.upper())
    uniq = sorted(set(tokens))
    if len(uniq) == 1:
        return uniq[0]
    return None


def deterministic_fallback(qid: str, question: str = "") -> str:
    key = f"{qid}|{question}".encode("utf-8", "ignore")
    h = hashlib.sha256(key).digest()[0]
    return "ABCD"[h % 4]


def majority_vote(values: Iterable[str | None], scores: Mapping[str, float] | None = None) -> tuple[str | None, float]:
    vals = [v.strip().upper() for v in values if v and v.strip().upper() in VALID_ANSWERS]
    if not vals:
        if scores:
            return max(scores, key=lambda k: scores[k]), 0.45
        return None, 0.0
    c = Counter(vals)
    top = c.most_common()
    if len(top) == 1:
        return top[0][0], 0.50 + 0.50 * (top[0][1] / max(1, len(vals)))
    if top[0][1] > top[1][1]:
        return top[0][0], 0.50 + 0.45 * (top[0][1] / max(1, len(vals)))
    if scores:
        tied = [k for k, v in c.items() if v == top[0][1]]
        return max(tied, key=lambda k: scores.get(k, 0.0)), 0.52
    return top[0][0], 0.50
