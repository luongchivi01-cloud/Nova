from __future__ import annotations

import math
import re
from .normalization import canonical
from .multilingual_nlp_adapter import analyze_multilingual, multilingual_tokenize
from .schema import MCQItem

NEGATION_TERMS = (
    "không", "sai", "ngoại trừ", "không đúng", "không phải", "except", "not", "incorrect", "least", "trừ",
    "不", "不是", "没有", "錯誤", "错误", "除外", "ない", "誤り", "제외", "아니다", "не", "кроме", "لا", "ليس", "ไม่",
)
HARD_TERMS = (
    "đúng nhất", "phù hợp nhất", "best", "most likely", "suy luận", "logic", "tính", "calculate", "infer", "nguyên nhân", "hệ quả",
    "最", "推断", "推論", "原因", "結果", "가장", "추론", "наиболее", "вероят", "الأكثر", "สรุป",
)
DOMAIN_TERMS = (
    "pháp luật", "kinh tế", "y tế", "ngân hàng", "lịch sử", "địa lý", "toán", "vật lý", "hóa học", "sinh học", "ai", "công nghệ",
    "law", "finance", "bank", "medical", "education", "physics", "chemistry", "biology", "technology",
    "法律", "金融", "医疗", "教育", "数学", "物理", "化学", "법", "금융", "здоров", "قانون",
)
NUMERIC_TERMS = (
    "bao nhiêu", "tính", "%", "kg", "km", "m/s", "vnd", "đồng", "usd",
    "calculate", "how many", "how much", "percent", "percentage", "ratio", "sum", "average",
    "多少", "計算", "计算", "몇", "계산", "сколько", "احسب", "كم", "คำนวณ",
)


def tokenize(text: str) -> list[str]:
    return multilingual_tokenize(text)


def lexical_overlap(a: str, b: str) -> float:
    ta, tb = set(tokenize(a)), set(tokenize(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, min(len(ta), len(tb)))


def option_similarity(item: MCQItem) -> float:
    options = [item.options.get(k, "") for k in "ABCD"]
    overlaps = []
    for i in range(4):
        for j in range(i + 1, 4):
            overlaps.append(lexical_overlap(options[i], options[j]))
    return max(overlaps) if overlaps else 0.0


def has_negation(item: MCQItem) -> bool:
    q = canonical(item.question)
    if any(canonical(term) in q for term in NEGATION_TERMS):
        return True
    try:
        return analyze_multilingual(item.question).has_negation
    except Exception:
        return False


def has_numeric_reasoning(item: MCQItem) -> bool:
    q = canonical(item.question)
    if any(canonical(t) in q for t in NUMERIC_TERMS):
        return True
    return bool(re.search(r"\d+(?:[\.,]\d+)?", item.text_for_retrieval()))


def estimate_difficulty(item: MCQItem) -> float:
    q = canonical(item.question)
    options = [item.options.get(k, "") for k in "ABCD"]
    q_len = len(tokenize(q))
    opt_lens = [len(tokenize(o)) for o in options]
    dif = 0.0
    if q_len > 35:
        dif += 0.12
    if q_len > 65:
        dif += 0.14
    if q_len > 100:
        dif += 0.08
    if has_negation(item):
        dif += 0.23
    if any(canonical(term) in q for term in HARD_TERMS):
        dif += 0.16
    if any(canonical(term) in q for term in DOMAIN_TERMS):
        dif += 0.08
    try:
        multi = analyze_multilingual(item.text_for_retrieval())
        if multi.domains:
            dif += 0.06
        if multi.has_number:
            dif += 0.04
        if multi.hard_markers:
            dif += 0.07
        if multi.is_mixed_language:
            dif += 0.05
        # CJK/Thai/etc. often have fewer spaces; do not underestimate long no-space rows.
        if multi.token_count > 120 or multi.cjk_char_count > 90:
            dif += 0.06
        if multi.language not in {"vi", "en", "unknown"}:
            dif += 0.04
    except Exception:
        pass
    if has_numeric_reasoning(item):
        dif += 0.08
    if max(opt_lens or [0]) > 22:
        dif += 0.10
    sim = option_similarity(item)
    if sim > 0.55:
        dif += 0.20
    elif sim > 0.35:
        dif += 0.10
    missing = sum(1 for o in options if not o.strip())
    if missing:
        dif += 0.20
    return min(1.0, dif)


def choose_solver_mode(difficulty: float, configured: str, elapsed: float = 0.0, time_budget: float = 0.0, done: int = 0, total: int = 0) -> str:
    configured = (configured or "adaptive").lower()
    if configured in {"direct", "vote", "judge", "max_accuracy"}:
        return configured
    # Time pressure: fall back to fast mode for remaining rows.
    if time_budget and total and done:
        avg = elapsed / max(1, done)
        remaining = max(0, total - done)
        projected_total = elapsed + avg * remaining
        if projected_total > time_budget * 0.96:
            return "direct"
        if projected_total > time_budget * 0.88 and difficulty < 0.80:
            return "vote" if difficulty > 0.45 else "direct"
    if difficulty < 0.24:
        return "direct"
    if difficulty < 0.64:
        return "vote"
    return "judge"
